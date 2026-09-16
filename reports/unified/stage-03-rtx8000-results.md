# Stage 3 — RTX 8000 local comparison

## Controlled run

- GPU: Quadro RTX 8000, compute capability 7.5
- driver: 560.35.03
- Python/PyTorch/CUDA: 3.12.13 / 2.11.0+cu129 / 12.9
- target: Qwen3-8B FP16
- batch: 8 prompts × 64 generated tokens
- repetitions: 2 warmups + 3 measured
- runtime: clean vLLM `7160b69e6`, V2 runner, eager, TP=1

## Observed medians

| Method | K | tok/s | vs AR | committed/step | to EOS | strict 64-token |
|---|---:|---:|---:|---:|---|---|
| AR | 0 | 122.74 | 1.000× | 1.016 | yes | yes |
| EAGLE3 | 7 | 133.35 | 1.086× | 1.730 | yes | no |
| DFlash | 15 | 68.66 | 0.559× | 1.016 | yes | yes |
| DFlare | 15 | 161.06 | 1.312× | 2.133 | yes | yes |
| DSpark | 7 | 204.76 | 1.668× | 2.560 | yes | no |

EAGLE3 and DSpark differ from AR only in forced continuation after AR's first
`<|im_end|>` token. All production-visible prefixes match AR.

## Validity limitation

This is a real local GPU measurement but a **provisional shared-GPU result**,
not a headline benchmark. Before every method the preflight observed two
external compute processes, about 25,666 MiB allocated, and 57–100% GPU
utilization. The runner therefore sets `benchmark_claim=false` automatically.

The table is useful for validating the common execution path and exposing
method behavior: DFlash committed almost no additional tokens on this workload,
whereas DFlare and DSpark reduced target decode steps. The absolute rates and
speedup ordering must be rerun on an exclusive RTX 4090 before use in a paper,
README badge, or resume bullet.

Artifacts:

- `results/measured/unified_spec_qwen3_8b_rtx8000_fp16/summary.json`
- one `result.json` and `trace.jsonl` per method;
- `results/figures/unified_spec_qwen3_8b_rtx8000_fp16.svg`.

Status: implementation complete; exclusive 4090 performance gate remains.
