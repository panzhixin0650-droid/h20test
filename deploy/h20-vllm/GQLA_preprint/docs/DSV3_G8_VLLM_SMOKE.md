# DeepSeek-V3.1 G=8 vLLM dummy smoke

This smoke validates the end-to-end vLLM plumbing without opening the 642 GiB
DeepSeek-V3.1 checkpoint. It creates a one-layer, weights-free config and forces
vLLM's `load_format=dummy` loader. The fixture retains the production attention
geometry and ranks:

| field | value |
|---|---:|
| query heads (`H`) | 128 |
| KV heads (`G`) | 8 |
| query heads per KV head | 16 |
| Q/K head width | 192 = 128 NoPE + 64 RoPE |
| V head width | 128 |
| Q LoRA rank | 1536 |
| KV latent rank | 512 |
| configured context length | 163840 |
| YaRN | factor 40, original context 4096, `mscale_all_dim=1.0` |
| normalized RoPE type | `deepseek_yarn` |
| attention softmax scale | 0.1352337788608801 |
| true-GQA cache | 2560 elements/token = 8 × (192 + 128) |
| MLA-absorb cache | 576 elements/token = 512 + 64 |

Only `hidden_size`, vocabulary, MLP width, and decoder-layer count are reduced.
The production 163840-position YaRN config is retained exactly; the runner's
`max_model_len=64` bounds smoke-time KV-cache allocation without changing RoPE
semantics. The single layer is dense. No tokenizer, safetensors, or quantization
config is generated.

## Generate and inspect the fixture

```bash
python scripts/make_dsv3p1_g8_tiny.py --output-dir /tmp/dsv3p1-g8-tiny
python -m unittest tests.test_dsv3p1_g8_tiny_fixture
```

The vLLM runner refuses any directory without `gqla_smoke_fixture.json`, any
fixture whose config checksum changed, any unexpected weight file, and any
config that no longer has exactly H=128/G=8/QK=192/V=128. This prevents an
accidental full-checkpoint load. It also rejects any drift from production YaRN
or its expected attention scale, `0.1352337788608801`. The runner always sets
`skip_tokenizer_init=True` and `load_format="dummy"`. Both paths explicitly use
`block_size=64`: this is an HPC-Ops GQA ABI requirement and keeping the control
identical avoids changing cache-page geometry between cases. Worker startup is
pinned to `VLLM_WORKER_MULTIPROC_METHOD=spawn` so platform discovery cannot
initialize CUDA in a parent that vLLM subsequently forks.

## Run prefill plus mixed prefill/decode

The two architecture names are intentionally different:

- `DeepseekV3GQLAHPCForCausalLM`: true GQA cache and HPC-Ops attention.
- `DeepseekV3GQLAForCausalLM`: existing MQA/MLA-absorb control.

Run TP=1 for both paths:

```bash
bash scripts/run_dsv3p1_g8_vllm_smoke.sh
```

The default `VERIFY_HPC_MIXED_SPLIT=1` run submits a 16-token prompt together
with a 48-token prompt under a 32-token chunked-prefill budget. The short
request therefore enters decode while the long request is still prefilling.
This is the regression case that previously let decode tokens inside a mixed
batch fall through to DiffKV.

Run TP=1 and TP=2:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TPS=1,2 \
SMOKE_PATHS=gqa-hpc,mqa-absorb \
bash scripts/run_dsv3p1_g8_vllm_smoke.sh
```

Useful overrides:

```bash
PYTHON=/absolute/path/to/python \
TINY_MODEL_DIR=/persistent/tiny/fixture \
LOG_ROOT=/persistent/smoke/logs \
GQA_ARCHITECTURE=DeepseekV3GQLAHPCForCausalLM \
MQA_ARCHITECTURE=DeepseekV3GQLAForCausalLM \
bash scripts/run_dsv3p1_g8_vllm_smoke.sh
```

For TP=2, install this repository editable in the selected environment. The
`vllm.general_plugins` entry point must register both architectures inside each
worker, not only in the driver process.

## Strict and trace proof

For `gqa-hpc`, the Python runner arms this contract before importing vLLM:

```text
GQLA_HPC_STRICT=1
GQLA_HPC_TRACE=1
```

Strict mode means an unavailable or unsupported HPC kernel must fail rather
than fall back. It also enables the all-decode contract: pure prefill remains
on DiffKV, a uniform decode batch uses HPC-Ops directly, and a mixed batch is
split so its decode prefix uses HPC-Ops while only its prefill suffix uses
DiffKV. Trace mode must emit `GQLA_HPC_TRACE_HIT` from an actual kernel
invocation. Merely resolving `DeepseekV3GQLAHPCForCausalLM`, constructing the
model, or selecting an attention backend is not proof that HPC-Ops ran.

The shell driver captures all worker output and enforces:

1. the vLLM request completes the requested prefill and decode token counts;
2. the GQA log contains `GQLA_HPC_ALL_DECODE_STRICT_ENABLED`;
3. the mixed regression run contains `GQLA_HPC_TRACE_MIXED_SPLIT`;
4. the GQA log contains `GQLA_HPC_TRACE_HIT`;
5. the same GQA trace reports `softmax_scale=0.135233...`, proving production
   YaRN scaling rather than the unscaled `0.0721688...` value;
6. the GQA log does not contain `GQLA_HPC_TRACE_FALLBACK`;
7. the MQA/MLA control log contains none of the HPC routing markers.

If the backend uses different stable markers, override only the verification
patterns:

```bash
HPC_HIT_PATTERN='your-hit-regex' \
HPC_FALLBACK_PATTERN='your-fallback-regex' \
HPC_SCALE_PATTERN='your-softmax-scale-regex' \
HPC_ALL_DECODE_PATTERN='your-all-decode-contract-regex' \
HPC_MIXED_SPLIT_PATTERN='your-mixed-split-regex' \
bash scripts/run_dsv3p1_g8_vllm_smoke.sh
```

`VERIFY_HPC_TRACE=0` is available for bring-up diagnostics, but such a run is
not evidence that the HPC kernel was exercised. Likewise,
`VERIFY_HPC_MIXED_SPLIT=0` disables the constructed mixed-batch regression and
cannot prove the all-decode behavior for mixed scheduling.

## Scope

This is a routing, shape, cache, TP, prefill, and decode smoke. Dummy random
weights cannot establish numerical equivalence with the converted checkpoint,
model quality, or production throughput. The MQA control also bypasses the
real-checkpoint `kv_b_proj` expansion because dummy loading creates parameters
directly. Those checks belong in a later, explicitly scheduled full-model run.
