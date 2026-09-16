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

measured/dflare_rtx8000_fp16_smoke/
    Official DFlare checkpoint load and short AngelSlim lossless smoke.

measured/ar_dflash_dflare_rtx8000_fp16_mixed4/
    Four-prompt FP16/SDPA controlled comparison with full request records.

measured/dflare_comparison_rtx8000_fp16.json
    Compact derived metrics and AngelSlim/vLLM exact-token checks.

measured/vllm_ar_rtx8000_fp16_control/
measured/vllm_dflare_patch_rtx8000_fp16_smoke/
    Same-prompt AR control and experimental patched-vLLM DFlare smoke.

measured/unified_spec_qwen3_8b_rtx8000_fp16/
    Common vLLM V2 AR/EAGLE3/DFlash/DFlare/DSpark results. The summary records
    both EOS-bounded and strict fixed-length equality plus pre-method GPU
    contention snapshots.

figures/vllm_scheduler_trace.svg
figures/dspark_verification_rounds.svg
figures/dflare_rtx8000_fp16_comparison.svg
figures/unified_spec_qwen3_8b_rtx8000_fp16.svg
    Visual summaries generated from the measured traces.
```

Regenerate them with:

```bash
make samples
make compare-dflare
make plot-unified
```

The unified directory is a controlled measured comparison, but its checked-in
RTX 8000 run detected a competing GPU workload and therefore records
`benchmark_claim=false`. Earlier GPU directories are integration smokes and
must not be presented as comparative speedups.
