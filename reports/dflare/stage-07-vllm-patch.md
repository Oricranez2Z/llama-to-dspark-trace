# Stage 7 — experimental vLLM DFlare patch

Status: complete for the declared V1 smoke scope on 2026-09-11.

## Deliverable

The standalone patch applies to vLLM
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` and produces the lab commit
`055065dee7c657392129f851e4e8949fa2b65152`. It changes 9 files with 483
insertions and 13 deletions. Its SHA-256 is
`31a94a535dd500c3b48543047f3834b4a5ebd61e925a99fcae4aaa03288dd032`.

Implementation boundaries:

```text
Qwen3 target hidden states (9 layers)
        ↓
per-draft-layer learned fusion [tokens, draft_layers, hidden]
        ↓
layer-specific target K/V projection + noncausal block draft
        ↓
existing DFlashProposer
        ↓
existing vLLM greedy rejection sampler / lossless verifier
        ↓
scheduler commit and normal output path
```

The patch also adds config detection, checkpoint layer-index translation,
model registration, auxiliary hidden-state collection, and DFlash-equivalent
scheduler lookahead.

## Verification

| Check | Result |
|---|---|
| Ruff on all changed vLLM files | passed |
| Focused unit tests | 2 passed |
| Apply formatted patch to clean base worktree | passed |
| Load target + draft on RTX 8000 | passed, 7.49 GiB + 1.38 GiB checkpoint data |
| Recognize auxiliary layers | `(2, 6, 10, 14, 18, 22, 26, 30, 34)` |
| End-to-end greedy generation | passed |
| AR/DFlare output IDs | 3/3 requests, 24/24 output tokens matched |

The final smoke manifest records a clean vLLM patch commit. It used a Quadro
RTX 8000, FP16 fallback, target FlashInfer attention, DFlare FlexAttention,
TP=1, eager mode, and one warmup request. The measured three-request batch was
2.509 s (9.56 output token/s).

This throughput must not be compared with the AR control's 0.258 s. Six
shape-specific Triton kernels compiled during the DFlare warmup/run sequence,
and neither run is a repeated steady-state benchmark. The meaningful assertion
here is end-to-end execution and token equality.

Artifacts:

- [Patch and application guide](../../integrations/vllm/README.md)
- [DFlare manifest](../../results/measured/vllm_dflare_patch_rtx8000_fp16_smoke/manifest.json)
- [DFlare scheduler trace](../../results/measured/vllm_dflare_patch_rtx8000_fp16_smoke/trace.jsonl)
- [AR control manifest](../../results/measured/vllm_ar_rtx8000_fp16_control/manifest.json)

## RTX 4090 acceptance gate

Before quoting any speedup on Ada:

1. Build vLLM and FlashAttention for compute capability 8.9 in a fresh environment.
2. Keep the exact target/draft revisions and greedy prompts fixed.
3. Run BF16 and, separately, FP16 AR controls on the same machine.
4. Assert speculative and same-dtype AR token equality for every request.
5. Clear/record compilation cache state, warm all measured shapes, then repeat trials.
6. Report target backend, draft backend, clocks, power limit, driver, memory, and batch/load.
7. Only claim performance after p50/p95 TTFT and TPOT plus throughput sweeps stabilize.

The upstream proposal [vLLM #49023](https://github.com/vllm-project/vllm/pull/49023)
was closed after a maintainer reported unreproduced ablations. This patch credits
that proposal and remains intentionally labeled experimental.
