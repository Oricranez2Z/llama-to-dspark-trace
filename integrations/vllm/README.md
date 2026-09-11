# Experimental vLLM DFlare patch

This directory carries a source patch, not a fork or a claim of upstream
support. It adds a small V1 Qwen3 DFlare path by reusing vLLM's existing
DFlash proposer, rejection sampler, scheduler lookahead, and lossless target
verification.

## Compatibility contract

| Item | Value |
|---|---|
| vLLM base | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| lab patch commit | `055065dee7c657392129f851e4e8949fa2b65152` |
| patch SHA-256 | `31a94a535dd500c3b48543047f3834b4a5ebd61e925a99fcae4aaa03288dd032` |
| target | `Qwen/Qwen3-4B@1cfa9a7` |
| draft | `AngelSlim/Qwen3-4b-dflare@71dcbb0` |
| verified path | V1, CUDA, TP=1, greedy, eager execution |

The patch was exported with `git format-patch` and independently applied to a
clean worktree at the base commit.

## Apply

From a clean vLLM checkout:

```bash
git checkout 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
git am <LLM_SERVING_LAB>/integrations/vllm/patches/0001-Add-experimental-Qwen3-DFlare-V1-support.patch
git rev-parse HEAD
```

The resulting commit hash may differ because Git records committer metadata.
Confirm the patch itself with:

```bash
sha256sum <LLM_SERVING_LAB>/integrations/vllm/patches/*.patch
```

Do not apply this patch over unrelated local changes. Use a branch or a Git
worktree so the source revision and result provenance stay auditable.

## What changes

- Registers `QwenDFlareDraftModel` and loads its checkpoint mapping.
- Computes a distinct learned target-hidden-state fusion for every draft layer.
- Adds separate target-context K/V projections required by the checkpoint.
- Collects nine configured target auxiliary hidden states.
- Routes DFlare through the existing DFlash block proposer and verifier.
- Reserves the same extra scheduler lookahead slot as DFlash.

This is deliberately V1 scope. It does not implement production serving
benchmarks, distributed execution, CUDA graphs, quantization, or sampling-mode
coverage.

## Tests run

```text
ruff check (all changed vLLM files): passed
tests/models/test_qwen3_dflare.py: 2 passed
clean git-am application: passed
RTX 8000 FP16 end-to-end smoke: passed
AR versus DFlare token IDs: 3/3 requests matched
```

An existing upstream DFlash lookahead test attempted to resolve an online 8B
model and stopped at restricted network access before exercising patch logic.
It is therefore recorded as unexecuted, not as a pass or regression.

## Scope warning

The earlier upstream DFlare PR
[#49023](https://github.com/vllm-project/vllm/pull/49023) was closed without
merge after maintainers reported difficulty reproducing its ablations. This
lab patch is adapted and attributed to that work, then adjusted to the pinned
local V1 revision. It is a learning and validation artifact, not evidence that
vLLM officially supports DFlare or that DFlare universally outperforms DFlash.

See [stage 7](../../reports/dflare/stage-07-vllm-patch.md) for the measured
smoke and the RTX 4090 validation gate.
