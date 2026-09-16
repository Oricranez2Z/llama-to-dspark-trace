# Experimental vLLM integration

This directory contains portable source patches and the worker used by the
unified AR/EAGLE3/DFlash/DFlare/DSpark benchmark. It is a learning artifact,
not a claim of upstream support.

## Unified V2 patch contract

| Item | Value |
|---|---|
| vLLM base | `ff6173997d54c5027971df8ecd1280f046a832b3` |
| patched revision | `7160b69e68eca230d26e6e72dbc72c3886741799` |
| patch directory | `patches/unified/` |
| target | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |
| verified path | V2 GPU runner, CUDA, TP=1, greedy, eager, FP16 |

Apply the four commits to a clean branch or worktree:

```bash
git checkout ff6173997d54c5027971df8ecd1280f046a832b3
git am <LLM_SERVING_LAB>/integrations/vllm/patches/unified/*.patch
git rev-parse HEAD
```

Patch SHA-256 values:

```text
55381557540d89b24f9c587425870d16db396792e7671a7c7c0393312939aa35  0001
fe981864301eb8bf3d3bcbdb0f6cf1a35d08e617eb975da7f5129acb0cb13545  0002
81869fca8f08b34ef81915808fe87e1a65abd6cc8241db4dcd3dd1e667232707  0003
86aa2a9d71e756aeef945a60f5c4ce947da5aa5aefeceaa2041202dba97c9f26  0004
```

The series registers Qwen3 DFlare, loads its released checkpoints, accepts the
native DSpark mask-token config, and gives DFlare a V2 DFlash-family execution
path. DFlare retains one fused target context per draft layer and inserts each
layer into its own KV-cache slot mapping.

The worker `unified_spec_worker.py` deliberately runs only one method per
process. It fixes dtype, runner, target, tokenizer, prompt rendering, sampling,
batch size, eager mode, prefix-cache setting, and trace collection. The outer
runner validates hashes and performs AR equality checks.

## Validation performed

```text
vLLM changed-file Ruff checks: passed
vLLM DFlare model tests: 3 passed
main repository tests: 30 passed
five-method V2 smoke: passed; all tokens matched AR
formal RTX 8000 run: all methods matched AR through EOS
vLLM result commit: clean
```

The RTX 8000 comparison is provisional because another process used the GPU.
That limitation is stored in `summary.json`, not left as prose only.

## Scope

Supported and measured: Qwen3-8B, CUDA, TP=1, eager, FP16, greedy offline
batches. Not established: CUDA graphs, online request arrivals, quantization,
pipeline/tensor parallelism above one, multi-node execution, or other model
families.

The older single-file patch in `patches/` is retained for the historical
Qwen3-4B V1 smoke. New comparison work should use the unified patch series.

The original DFlare implementation was adapted from vLLM pull request #49023
with attribution retained in Git history. That proposal was not merged; this
repository therefore describes DFlare support as experimental.
