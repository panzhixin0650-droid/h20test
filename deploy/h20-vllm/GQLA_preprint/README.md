# Minimal GQLA vLLM serving package

This is a deployment snapshot for serving the converted DeepSeek-V3.1 G=8
checkpoint through vLLM 0.22.1. It contains only the model registration,
MLA-absorb and true-GQA model paths, the HPC-Ops attention backend, and H20
environment/benchmark launchers.

Use the parent `deploy_h20_vllm.sh` script to materialize this directory and
patch the pinned HPC-Ops source tree. Do not run this directory in isolation
with an unpatched HPC-Ops checkout: the model requires the runtime
`softmax_scale` API for DeepSeek YaRN.
