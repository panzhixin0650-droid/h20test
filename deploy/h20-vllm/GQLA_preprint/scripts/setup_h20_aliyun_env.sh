#!/usr/bin/env bash
# Recreate the H20 benchmark environment from an Aliyun-only wheelhouse.
#
# Every locked Python distribution is resolved from the Aliyun PyPI mirror,
# downloaded by aria2 with its mirror-published SHA256, and then installed with
# --no-index.  The converted model and the local GQLA/HPC sources are inputs;
# they are never downloaded or modified by this script.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
MODEL_NAME=dsv3p1_g8_sim_hess_no_mean_subtract

infer_gqla_root() {
    local probe
    probe=$REPO_DIR
    while [[ "$probe" != / ]]; do
        if [[ -f "$probe/outputs/convert/$MODEL_NAME/config.json" ]]; then
            printf '%s\n' "$probe"
            return 0
        fi
        probe=$(dirname "$probe")
    done
    if [[ -f "/mnt/public03/task/236362/GQLA/outputs/convert/$MODEL_NAME/config.json" ]]; then
        printf '%s\n' /mnt/public03/task/236362/GQLA
    else
        printf '%s\n' /mnt/tidalfs-alwl01/task/236362/GQLA
    fi
}

die() {
    echo "error: $*" >&2
    exit 2
}

require_bool() {
    local name=$1 value=$2
    case "$value" in
        0|1) ;;
        *) die "$name must be 0 or 1; got $value" ;;
    esac
}

GQLA_ROOT=${GQLA_ROOT:-$(infer_gqla_root)}
ENV_ROOT=${ENV_ROOT:-$GQLA_ROOT/envs}
MODEL_DIR=${MODEL_DIR:-$GQLA_ROOT/outputs/convert/$MODEL_NAME}
HPC_OPS_DIR=${HPC_OPS_DIR:-$GQLA_ROOT/code/hpc-ops}
VENV_DIR=${VENV_DIR:-$ENV_ROOT/h20-aliyun-py312}
WHEEL_DIR=${WHEEL_DIR:-$ENV_ROOT/wheels/h20-aliyun-py312}
UV_CACHE_DIR=${UV_CACHE_DIR:-$VENV_DIR/.uv-cache}
LOCK_FILE=${LOCK_FILE:-$REPO_DIR/requirements-h20-aliyun-py312.lock}
BOOTSTRAP_SCRIPT=${BOOTSTRAP_SCRIPT:-$SCRIPT_DIR/bootstrap_dsv3p1_g8_h20_env.sh}
SETUP_LOG=${SETUP_LOG:-$ENV_ROOT/logs/setup-h20-aliyun.log}
ALIYUN_INDEX=${ALIYUN_INDEX:-https://mirrors.aliyun.com/pypi/simple/}
BASE_PYTHON=${BASE_PYTHON:-$ENV_ROOT/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12}
UV_BIN=${UV_BIN:-$REPO_DIR/tools/uv}
CUDA_COMPAT_ARCHIVE=${CUDA_COMPAT_ARCHIVE:-$ENV_ROOT/downloads/cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb}
CUDA_COMPAT_SHA256=${CUDA_COMPAT_SHA256:-14a3d14373f882297f368d6282fc7fba85e46682f34166291d61df1913a59c8f}

ARIA2_CONNECTIONS=${ARIA2_CONNECTIONS:-8}
ARIA2_CONCURRENT_FILES=${ARIA2_CONCURRENT_FILES:-8}
RESOLVE_WORKERS=${RESOLVE_WORKERS:-16}
FORCE_RECREATE=${FORCE_RECREATE:-1}
FORCE_HPC_REBUILD=${FORCE_HPC_REBUILD:-1}
RESOLVE_ONLY=${RESOLVE_ONLY:-0}

require_bool FORCE_RECREATE "$FORCE_RECREATE"
require_bool FORCE_HPC_REBUILD "$FORCE_HPC_REBUILD"
require_bool RESOLVE_ONLY "$RESOLVE_ONLY"

for command_name in aria2c bash sha256sum sort awk sed tar; do
    command -v "$command_name" >/dev/null 2>&1 \
        || die "required host command is missing: $command_name"
done
[[ "$(uname -m)" == x86_64 ]] || die "this lock supports x86_64 only"
[[ -x "$BASE_PYTHON" ]] || die "bundled Python is missing: $BASE_PYTHON"
[[ -x "$UV_BIN" ]] || die "bundled uv is missing: $UV_BIN"
[[ -f "$LOCK_FILE" ]] || die "Aliyun lock file is missing: $LOCK_FILE"
[[ -f "$BOOTSTRAP_SCRIPT" ]] || die "bootstrap script is missing: $BOOTSTRAP_SCRIPT"
[[ -f "$MODEL_DIR/config.json" ]] || die "converted model is missing: $MODEL_DIR"
[[ -f "$HPC_OPS_DIR/setup.py" ]] || die "HPC-Ops source is missing: $HPC_OPS_DIR"

mkdir -p "$ENV_ROOT/logs" "$WHEEL_DIR"
exec > >(tee -a "$SETUP_LOG") 2>&1
echo "[aliyun-setup] log=$SETUP_LOG"
echo "[aliyun-setup] sole package source=$ALIYUN_INDEX"
echo "[aliyun-setup] model is read-only input=$MODEL_DIR"

if command -v flock >/dev/null 2>&1; then
    exec 9>"$ENV_ROOT/.setup-h20-aliyun.lock"
    echo "[aliyun-setup] waiting for environment lock"
    flock 9
fi

metadata_file=$WHEEL_DIR/aliyun-wheel-metadata.tsv
aria2_input=$WHEEL_DIR/aria2-input.txt
offline_lock=$WHEEL_DIR/requirements.lock

echo "[aliyun-setup] resolving every locked wheel from Aliyun"
"$BASE_PYTHON" - \
    "$LOCK_FILE" "$ALIYUN_INDEX" "$RESOLVE_WORKERS" >"$metadata_file.tmp" <<'PY'
import concurrent.futures
import html.parser
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

lock_path = Path(sys.argv[1])
index = sys.argv[2].rstrip("/") + "/"
workers = int(sys.argv[3])


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


specs = []
for raw in lock_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if "==" not in line or ";" in line or "://" in line or line.startswith(("-", "/")):
        raise SystemExit(f"unsupported lock entry: {raw}")
    name, version = line.split("==", 1)
    specs.append((canonical(name.strip()), version.strip()))


class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None
        self.items = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")

    def handle_data(self, data):
        if self.href and data.strip():
            self.items.append((self.href, data.strip()))

    def handle_endtag(self, tag):
        if tag == "a":
            self.href = None


def wheel_tags(filename: str):
    if not filename.lower().endswith(".whl"):
        return None
    try:
        _, py_tag, abi_tag, platform_tag = filename[:-4].rsplit("-", 3)
    except ValueError:
        return None
    return py_tag.lower(), abi_tag.lower(), platform_tag.lower()


def compatible(filename: str) -> bool:
    tags = wheel_tags(filename)
    if tags is None:
        return False
    py_tag, abi_tag, platform_tag = tags
    if platform_tag != "any":
        if "x86_64" not in platform_tag:
            return False
        # musllinux also contains the substring "linux", but those wheels
        # target musl and cannot be installed in this glibc container.
        platform_tags = platform_tag.split(".")
        if any(tag.startswith("musllinux") for tag in platform_tags):
            return False
        if not any(
            tag.startswith("manylinux") or tag == "linux_x86_64"
            for tag in platform_tags
        ):
            return False
    python_ok = False
    py_tags = py_tag.split(".")
    abi_tags = abi_tag.split(".")
    for tag in py_tags:
        if tag in {"py3", "py2.py3", "cp312"} or tag.startswith("py3"):
            python_ok = True
        if tag.startswith("cp") and tag[2:].isdigit() and "abi3" in abi_tags:
            python_ok = python_ok or int(tag[2:]) <= 312
    return python_ok


def candidate_rank(filename: str):
    py_tag, abi_tag, platform_tag = wheel_tags(filename)
    if "cp312" in py_tag.split("."):
        python_rank = 3
    elif "abi3" in abi_tag.split("."):
        python_rank = 2
    else:
        python_rank = 1
    platform_rank = 2 if platform_tag != "any" else 1
    return python_rank, platform_rank


def fetch_page(url: str) -> str:
    last = None
    for attempt in range(6):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "gqla-h20-aliyun-resolver/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read().decode("utf-8", "replace")
        except Exception as exc:
            last = exc
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def resolve(spec):
    name, version = spec
    page_url = urllib.parse.urljoin(index, urllib.parse.quote(name) + "/")
    parser = Links()
    parser.feed(fetch_page(page_url))
    version_tokens = {
        f"-{version.lower()}-",
        f"-{version.lower().replace('-', '_')}-",
    }
    candidates = []
    for href, text in parser.items:
        filename = urllib.parse.unquote(text)
        normalized_filename = filename.lower().replace("_", "-")
        if not any(token.replace("_", "-") in normalized_filename for token in version_tokens):
            continue
        if not compatible(filename):
            continue
        absolute = urllib.parse.urljoin(page_url, href)
        parsed = urllib.parse.urlsplit(absolute)
        digest = urllib.parse.parse_qs(parsed.fragment).get("sha256", [""])[0]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            continue
        download_url = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
        )
        if parsed.netloc != "mirrors.aliyun.com" or not parsed.path.startswith("/pypi/"):
            raise RuntimeError(f"non-Aliyun wheel URL for {name}: {download_url}")
        candidates.append((candidate_rank(filename), filename, download_url, digest))
    if not candidates:
        raise RuntimeError(f"no CPython 3.12 x86_64 wheel for {name}=={version}")
    _, filename, url, digest = max(candidates)
    return name, version, url, filename, digest


errors = []
resolved = []
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    future_map = {executor.submit(resolve, spec): spec for spec in specs}
    for future in concurrent.futures.as_completed(future_map):
        try:
            resolved.append(future.result())
        except Exception as exc:
            errors.append(f"{future_map[future]}: {exc}")
if errors:
    raise SystemExit("Aliyun resolution failed:\n" + "\n".join(sorted(errors)))
if len(resolved) != len(specs):
    raise SystemExit(f"resolved {len(resolved)} of {len(specs)} locked packages")
for row in sorted(resolved):
    print(*row, sep="\t")
PY
mv "$metadata_file.tmp" "$metadata_file"

locked_count=$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$LOCK_FILE")
resolved_count=$(awk 'NF {count++} END {print count+0}' "$metadata_file")
[[ "$resolved_count" == "$locked_count" ]] \
    || die "resolved $resolved_count of $locked_count locked wheels"
if awk -F '\t' '$3 !~ /^https:\/\/mirrors\.aliyun\.com\/pypi\// {exit 1}' \
    "$metadata_file"; then
    echo "[aliyun-setup] source audit passed for $resolved_count wheel URLs"
else
    die "metadata contains a non-Aliyun URL"
fi

if [[ "$RESOLVE_ONLY" == 1 ]]; then
    echo "H20_ALIYUN_RESOLVE_OK wheels=$resolved_count metadata=$metadata_file"
    exit 0
fi

[[ -f "$CUDA_COMPAT_ARCHIVE" ]] \
    || die "local CUDA compatibility package is missing: $CUDA_COMPAT_ARCHIVE"
actual_compat_sha=$(sha256sum "$CUDA_COMPAT_ARCHIVE" | awk '{print $1}')
[[ "$actual_compat_sha" == "$CUDA_COMPAT_SHA256" ]] \
    || die "verified local CUDA compatibility package is missing: $CUDA_COMPAT_ARCHIVE"

: >"$aria2_input"
bad_suffix=$(date -u +%Y%m%dT%H%M%SZ)

wheel_archive_is_valid() {
    "$BASE_PYTHON" - "$1" <<'PY'
import sys
import zipfile

try:
    with zipfile.ZipFile(sys.argv[1]) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("corrupt member")
except Exception:
    raise SystemExit(1)
PY
}

while IFS=$'\t' read -r name version url filename digest; do
    wheel=$WHEEL_DIR/$filename
    valid=0
    if [[ -s "$wheel" ]]; then
        actual=$(sha256sum "$wheel" | awk '{print $1}')
        if [[ "$actual" == "$digest" ]] && wheel_archive_is_valid "$wheel"; then
            valid=1
        else
            mv "$wheel" "$wheel.bad.$bad_suffix"
            [[ ! -e "$wheel.aria2" ]] \
                || mv "$wheel.aria2" "$wheel.aria2.bad.$bad_suffix"
        fi
    fi
    if [[ "$valid" == 0 ]]; then
        {
            printf '%s\n' "$url"
            printf '  dir=%s\n' "$WHEEL_DIR"
            printf '  out=%s\n' "$filename"
            printf '  checksum=sha-256=%s\n' "$digest"
        } >>"$aria2_input"
    fi
done <"$metadata_file"

if [[ -s "$aria2_input" ]]; then
    echo "[aliyun-setup] downloading missing wheels with aria2"
    aria2c \
        --input-file="$aria2_input" \
        --continue=true \
        --allow-overwrite=true \
        --auto-file-renaming=false \
        --file-allocation=none \
        --max-concurrent-downloads="$ARIA2_CONCURRENT_FILES" \
        --max-connection-per-server="$ARIA2_CONNECTIONS" \
        --split="$ARIA2_CONNECTIONS" \
        --min-split-size=4M \
        --connect-timeout=20 \
        --timeout=120 \
        --retry-wait=2 \
        --max-tries=20
else
    echo "[aliyun-setup] all locked Aliyun wheels are already verified"
fi

echo "[aliyun-setup] validating the complete offline wheelhouse"
while IFS=$'\t' read -r name version url filename digest; do
    wheel=$WHEEL_DIR/$filename
    [[ -s "$wheel" ]] || die "missing wheel after aria2: $filename"
    actual=$(sha256sum "$wheel" | awk '{print $1}')
    [[ "$actual" == "$digest" ]] || die "SHA256 mismatch after aria2: $filename"
    wheel_archive_is_valid "$wheel" || die "invalid wheel archive: $filename"
done <"$metadata_file"
cp "$LOCK_FILE" "$offline_lock"

if [[ "$FORCE_RECREATE" == 1 && -e "$VENV_DIR" ]]; then
    backup=$VENV_DIR.previous.$(date -u +%Y%m%dT%H%M%SZ)
    mv "$VENV_DIR" "$backup"
    echo "[aliyun-setup] moved previous environment to $backup"
fi
mkdir -p "$VENV_DIR"
export UV_CACHE_DIR
export UV_PYTHON_DOWNLOADS=never
export UV_DEFAULT_INDEX=$ALIYUN_INDEX
export UV_INDEX_URL=$ALIYUN_INDEX
export PIP_INDEX_URL=$ALIYUN_INDEX
export UV_INDEX_STRATEGY=first-index
unset UV_EXTRA_INDEX_URL PIP_EXTRA_INDEX_URL UV_FIND_LINKS PIP_FIND_LINKS
unset UV_TORCH_BACKEND

echo "[aliyun-setup] creating a fresh Python environment"
"$UV_BIN" venv --no-project --no-config --clear \
    --python "$BASE_PYTHON" "$VENV_DIR"
PYTHON=$VENV_DIR/bin/python

echo "[aliyun-setup] installing with --no-index from the verified wheelhouse"
"$UV_BIN" pip install --no-config --python "$PYTHON" \
    --no-index --find-links "$WHEEL_DIR" \
    --requirements "$offline_lock" --strict --compile-bytecode

export GQLA_ROOT ENV_ROOT MODEL_DIR HPC_OPS_DIR VENV_DIR UV_CACHE_DIR UV_BIN
export FORCE_CORE_REINSTALL=0
export FORCE_HPC_REBUILD
export NEEDS_HPC=1
export CUDA_COMPAT_MODE=auto
export REQUIREMENTS_FILE=$offline_lock

echo "[aliyun-setup] configuring CUDA compatibility and rebuilding patched HPC-Ops"
bash "$BOOTSTRAP_SCRIPT"

runtime_env=$VENV_DIR/h20-runtime.env
[[ -f "$runtime_env" ]] || die "runtime environment was not written: $runtime_env"
lock_sha=$(sha256sum "$LOCK_FILE" | awk '{print $1}')
metadata_sha=$(sha256sum "$metadata_file" | awk '{print $1}')
wheel_bytes=$(du -sb "$WHEEL_DIR" | awk '{print $1}')
cat >"$VENV_DIR/h20-aliyun-manifest.env" <<EOF
setup_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
index_url=$ALIYUN_INDEX
install_mode=aria2-then-uv-no-index
locked_wheels=$locked_count
lock_sha256=$lock_sha
metadata_sha256=$metadata_sha
wheelhouse=$WHEEL_DIR
wheelhouse_bytes=$wheel_bytes
venv_dir=$VENV_DIR
runtime_env=$runtime_env
EOF

echo "H20_ALIYUN_SETUP_OK wheels=$locked_count venv=$VENV_DIR"
echo "H20_ALIYUN_RUNTIME_ENV=$runtime_env"
