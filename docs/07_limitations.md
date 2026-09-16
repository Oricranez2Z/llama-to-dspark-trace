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
- The checked-in RTX 8000 comparison ran beside another GPU process. Its
  machine-readable validity gate marks all speedups provisional.
- Proposal lengths are checkpoint-native (`K=7` or `K=15`), so the comparison
  evaluates released method/checkpoint pairs rather than equal-K algorithms.
- CUDA graph, quantized, tensor-parallel, and multi-node paths are not covered.
- The RTX 8000 uses compute capability 7.5 and Triton attention fallback;
  kernels and conclusions must be rebuilt and remeasured on the RTX 4090.

These boundaries are deliberate. Claims in the README and resume should not
extend beyond the implemented and measured behavior.
