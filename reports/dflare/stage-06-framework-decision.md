# Stage 6 — vLLM versus SGLang decision

Status: vLLM selected for this repository on 2026-09-11.

## Decision

Use a pinned vLLM V1 experimental patch for the portfolio implementation.
Keep SGLang as the explicit next branch only if the objective changes to a
paper-style online-serving study.

| Criterion | vLLM | SGLang | Decision effect |
|---|---|---|---|
| Existing local work | Trace adapter, environment, DFlash path already validated | Source available but not integrated in this lab | vLLM |
| Small DFlare delta | Existing DFlash proposer/verifier/scheduler can be reused | Current public path documents DFlash, not DFlare | vLLM |
| End-to-end evidence now | Patch completed an RTX 8000 smoke | Would require another port before evidence | vLLM |
| Paper-style server evaluation | Possible, but this patch is eager/offline V1 scope | Strong DFlash serving CLI and serving-oriented benchmark path | SGLang if scope changes |
| Upstream status | DFlare PR #49023 closed without merge | Public docs list DFlash but not DFlare | neither is official DFlare support |

## Why this is the honest choice

The goal of this GitHub repository is to demonstrate source understanding:
model fusion, hidden-state transport, block proposal, scheduler lookahead, and
lossless verification. The vLLM patch exposes all five boundaries with a small,
reviewable diff and a runnable result.

For a paper submission, the requirements change. The project would need an
online endpoint, concurrency sweeps, TTFT/TPOT percentiles, request-rate control,
prefix-cache policy, CUDA graph coverage, and repeated trials across datasets.
At that point SGLang's documented DFlash server path is an attractive baseline,
but DFlare itself would still need a custom implementation and verification.

## Revisit conditions

Switch the implementation track to SGLang only when all are true:

1. The deliverable is an online-serving evaluation rather than a learning patch.
2. RTX 4090 results show stable DFlare acceptance after BF16/FA2 validation.
3. A DFlash SGLang baseline is reproduced with the same target, prompts, and load.
4. Time is budgeted for scheduler, CUDA graph, sampling, and concurrency coverage.

Primary references: [vLLM PR #49023](https://github.com/vllm-project/vllm/pull/49023),
[SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding),
and [AngelSlim](https://github.com/Tencent/AngelSlim).
