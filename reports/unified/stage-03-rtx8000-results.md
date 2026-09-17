# Stage 3 — RTX 8000 matched-K=7 local comparison

## Controlled run

- GPU: Quadro RTX 8000, compute capability 7.5
- driver: 560.35.03
- Python/PyTorch/CUDA: 3.12.13 / 2.11.0+cu129 / 12.9
- target: Qwen3-8B FP16
- batch: 8 prompts × 64 generated tokens
- repetitions: 2 warmups + 3 measured
- runtime: clean vLLM `7160b69e6`, V2 runner, eager, TP=1
- proposal width: runtime `K=7` for every speculative method

## Corrected DFlash integration

The earlier DFlash result used the distinct z-lab block-16 mask-layout
checkpoint, cast its BF16 weights to FP16 on the Turing GPU, accepted zero
draft tokens, and reported 68.66 tok/s. The direct performance cause was the
zero acceptance; that checkpoint/precision/backend result neither isolates a
single root cause nor satisfies the matched-`K=7` contract, so it is retained
only as a failed diagnostic and is not treated as representative DFlash
performance.

The corrected run uses `deepseek-ai/dflash_qwen3_8b_block7@9e44dbbb6c`. Its
config is a `Qwen3DSparkModel` with `markov_rank=0`: DeepSpec packages DFlash
in the common anchor-sampling model container. The benchmark therefore invokes
the DSpark runtime layout while retaining the algorithm label `dflash`. Routing
this DeepSpec checkpoint through the z-lab mask-layout runtime would be a
semantic mismatch. A preflight contract now checks architecture, Markov-head
state, checkpoint block size, and runtime width before launching the GPU job.

## Observed medians

| Method | Runtime K | Checkpoint block | tok/s | vs AR | committed/step | draft acceptance | to EOS | strict 64-token |
|---|---:|---:|---:|---:|---:|---:|---|---|
| AR | 0 | — | 236.13 | 1.000× | 1.000 | — | yes | yes |
| EAGLE3 | 7 | 7 | 225.18 | 0.954× | 2.558 | 24.48% | yes | yes |
| DFlash | 7 | 7 | 393.58 | 1.667× | 3.429 | 38.43% | yes | no |
| DFlare | 7 | 16 (truncated) | 384.87 | 1.630× | 3.600 | 42.10% | yes | yes |
| DSpark | 7 | 7 | 432.35 | 1.831× | 3.847 | 45.97% | yes | no |

All methods are lossless through the first EOS token and stable across the
three measured repetitions. DFlash and DSpark differ from AR only in forced
continuation after AR's first `<|im_end|>` token. Exact scheduler counters show
that corrected DFlash accepted 382 of 994 proposed draft tokens, rather than
zero.

DFlare uses runtime `K=7`, but its available checkpoint was trained for block
16. Its row is an equal-runtime-width comparison, not a fully matched training
ablation. The other three speculative checkpoints are native block 7.

## Validity

This rerun passed the automatic isolation gate. Every pre-method snapshot
observed 0% utilization, no external model process, and only the 26 MiB CUDA
MPS daemon. The runner therefore records `measurement_validity.status=valid`
and `benchmark_claim=true`.

The result establishes that the corrected DFlash path is functional and that
all four speculative methods now share the same target, workload, verifier
width, runtime revision, precision, and trace schema. These figures apply to
the fixed offline eager batch on this RTX 8000. RTX 4090, online-serving,
CUDA-graph, and concurrency claims require their own measurements.

Artifacts:

- `results/measured/unified_spec_qwen3_8b_rtx8000_fp16/summary.json`
- one `result.json` and `trace.jsonl` per method;
- `results/figures/unified_spec_qwen3_8b_rtx8000_fp16.svg`.

Status: corrected matched-K=7 RTX 8000 measurement complete; RTX 4090
validation remains a separate hardware/backend experiment.
