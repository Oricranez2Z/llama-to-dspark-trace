# DFlare implementation journal

This directory is the auditable, stage-by-stage record for the DFlare extension.
Each stage separates implemented facts, measured observations, and unsupported
performance claims.

| Stage | Deliverable | Status |
|---|---|---|
| [1](stage-01-educational-model.md) | Independent fusion and block-proposal model | complete |
| [2](stage-02-verifier-and-trace.md) | Lossless verifier and normalized trace schema | complete |
| [3](stage-03-angelslim-runner.md) | Isolated AngelSlim external runner | complete |
| [4](stage-04-rtx8000-smoke.md) | Pinned Qwen3-4B/DFlare RTX 8000 smoke | complete |
| [5](stage-05-ar-dflash-dflare.md) | AR/DFlash/DFlare controlled comparison | complete |
| [6](stage-06-framework-decision.md) | Framework scorecard and decision | vLLM selected |
| [7](stage-07-vllm-patch.md) | Portable vLLM V1 patch and GPU validation | complete for smoke scope |

The next hardware gate is RTX 4090 BF16/FlashAttention 2. It is intentionally
documented as a future reproduction matrix rather than represented by simulated
or extrapolated numbers.
