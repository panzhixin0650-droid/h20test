#!/usr/bin/env python3
"""Convert a pip --report JSON file into an aria2 wheel download manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse


def collect_downloads(report: dict) -> list[tuple[str, str, str | None]]:
    downloads: dict[str, tuple[str, str | None]] = {}
    for item in report.get("install", []):
        download_info = item.get("download_info") or {}
        url = download_info.get("url", "")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue

        filename = unquote(os.path.basename(parsed.path))
        if not filename.endswith(".whl"):
            raise ValueError(f"resolved dependency is not a wheel: {url}")

        archive_info = download_info.get("archive_info") or {}
        hashes = archive_info.get("hashes") or {}
        sha256 = hashes.get("sha256")
        if not sha256:
            legacy_hash = archive_info.get("hash", "")
            if legacy_hash.startswith("sha256="):
                sha256 = legacy_hash.removeprefix("sha256=")

        clean_url = parsed._replace(fragment="").geturl()
        existing = downloads.get(filename)
        if existing and existing[0] != clean_url:
            raise ValueError(f"conflicting URLs resolve to the same filename: {filename}")
        downloads[filename] = (clean_url, sha256)

    return [
        (url, filename, sha256)
        for filename, (url, sha256) in sorted(downloads.items())
    ]


def write_manifest(
    downloads: list[tuple[str, str, str | None]],
    destination: Path,
    output_path: Path,
) -> None:
    destination = destination.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as manifest:
        for url, filename, sha256 in downloads:
            manifest.write(f"{url}\n")
            manifest.write(f"  dir={destination}\n")
            manifest.write(f"  out={filename}\n")
            if sha256:
                manifest.write(f"  checksum=sha-256={sha256}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.report.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    downloads = collect_downloads(report)
    if not downloads:
        raise SystemExit("pip report contains no remote wheel downloads")

    args.destination.mkdir(parents=True, exist_ok=True)
    write_manifest(downloads, args.destination, args.output)
    total_hashed = sum(sha256 is not None for _, _, sha256 in downloads)
    print(
        f"ARIA2_WHEEL_MANIFEST_OK wheels={len(downloads)} "
        f"sha256={total_hashed} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
