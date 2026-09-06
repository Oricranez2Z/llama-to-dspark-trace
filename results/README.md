# Results

This directory separates deterministic CPU demonstrations from measured GPU
smoke tests.

```text
sample_traces/scheduler_trace.jsonl
    Request admission, scheduling, block allocation, execution, and output.

sample_traces/speculative_summary.json
    Fixed-length and confidence-scheduled algorithmic simulation results.

figures/scheduler_timeline.svg
    Prefill/decode work assigned to each request by engine step.

figures/kv_blocks.svg
    Physical KV-block ownership across engine steps.

figures/speculative_acceptance.svg
    Mean accepted draft length for the demonstration configurations.

measured/vllm_gpu_smoke/
    vLLM environment manifest, generated outputs, and scheduler/KV trace.

measured/dspark_gpu_smoke/
    DeepSpec result metadata and proposal/verification trace.

figures/vllm_scheduler_trace.svg
figures/dspark_verification_rounds.svg
    Visual summaries generated from the measured traces.
```

Regenerate them with:

```bash
make samples
```

These artifacts are not GPU benchmarks and must not be presented as measured
vLLM or DSpark speedups. The GPU directories are measured smoke tests, but their
tiny workloads are not statistically meaningful performance comparisons.
