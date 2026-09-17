# Limitations

- Mini Llama uses NumPy and random weights; it demonstrates execution semantics,
  not language quality or production performance.
- Mini Serving models KV allocation and scheduling but does not execute a
  flattened multi-request transformer batch.
- The block manager does not yet implement prefix caching, copy-on-write,
  swapping, or production preemption policies.
- The speculative verifier models one batched target verification as one
  conceptual call, while internally evaluating a deterministic target function.
- The confidence scheduler uses a configurable illustrative load profile, not
  DeepSeek's production throughput profile.
- The vLLM adapter monkey-patches selected Python methods and is pinned to a
  studied source revision. Internal APIs may change.
- The unified runner is an offline fixed batch, not an online arrival process;
  it does not report TTFT/TPOT percentiles or scheduler behavior under load.
- The checked-in RTX 8000 comparison passed its isolation gate, but covers one
  fixed offline eager batch rather than online arrivals, CUDA graphs, or load
  sweeps.
- The primary checked-in comparison uses runtime `K=7` for every speculative
  method. EAGLE3, DFlash, and DSpark use native block-7 checkpoints, while
  DFlare truncates a released block-16 checkpoint to `K=7`; this is equal
  verifier width, not a fully matched-training-block ablation.
- CUDA graph, quantized, tensor-parallel, and multi-node paths are not covered.
- The RTX 8000 uses compute capability 7.5 and Triton attention fallback;
  kernels and conclusions must be rebuilt and remeasured on the RTX 4090.

These boundaries are deliberate. Claims in the README and resume should not
extend beyond the implemented and measured behavior.
