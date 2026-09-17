# Experiment methodology

## CPU demonstrations

The checked-in demonstrations validate mechanisms rather than hardware speed:

```bash
make demo
make demo-spec
make visualize
```

The scheduler experiment varies prompt/output lengths and records every engine
step. The speculative experiment sweeps fixed draft lengths and concurrency for
the confidence policy.

## Primary unified comparison

Use `docs/11_unified_speculative_benchmark.md` for the common AR, EAGLE3,
DFlash, DFlare, and DSpark protocol. Unlike the historical smokes below, it
pins the same target, workload, precision, runner, process isolation, timing,
trace, and correctness rules for every method.

The runner records GPU state before every method. It briefly resamples high
utilization when external allocation is small so the preceding worker's NVML
tail is not mistaken for contention. Persistent utilization above 5% or
external compute allocation above 512 MiB makes the result provisional and
sets `benchmark_claim=false`.

## Historical GPU smoke tests

`results/measured/` contains a traced vLLM offline batch and a one-sample
DeepSpec DSpark evaluation. Use them to verify environment capture, source
instrumentation, and visualization. Do not use them to claim a speedup: the
workloads are deliberately tiny and no baseline/speculative comparison was run.

Reproduction commands are in `docs/10_gpu_reproduction.md`.

## GPU experiment matrix

For a broader serving study after the fixed comparison passes, use at least:

| Dimension | Suggested values |
|---|---|
| Input length | 128, 512, 2048 |
| Output length | 32, 128, 512 |
| Concurrency | 1, 4, 8, 16, 32 |
| Method | AR, EAGLE3, DFlash, DFlare, DSpark |
| Draft length | 2, 4, 6, 8 |

Record:

- Request throughput.
- Output tokens per second.
- TTFT and TPOT, including percentiles.
- Mean accepted length and per-position acceptance.
- Draft and target-verification GPU time.
- Input slots per target forward.
- KV block occupancy and peak memory.

## Required metadata

Each result directory should include a manifest such as:

```json
{
  "git_commit": "...",
  "dirty": false,
  "gpu": "...",
  "driver": "...",
  "cuda": "...",
  "target_model": "...",
  "draft_model": "...",
  "sampling": {"temperature": 0},
  "workload": {"input_length": 512, "output_length": 128},
  "mode": "measured_gpu_serving"
}
```

Keep algorithmic simulation results separate from measured serving results.
