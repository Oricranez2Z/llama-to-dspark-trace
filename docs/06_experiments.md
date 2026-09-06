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

## Checked-in GPU smoke tests

`results/measured/` contains a traced vLLM offline batch and a one-sample
DeepSpec DSpark evaluation. Use them to verify environment capture, source
instrumentation, and visualization. Do not use them to claim a speedup: the
workloads are deliberately tiny and no baseline/speculative comparison was run.

Reproduction commands are in `docs/10_gpu_reproduction.md`.

## GPU experiment matrix

For real vLLM/DSpark measurements, use at least:

| Dimension | Suggested values |
|---|---|
| Input length | 128, 512, 2048 |
| Output length | 32, 128, 512 |
| Concurrency | 1, 4, 8, 16, 32 |
| Method | AR, fixed speculative, DSpark |
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
