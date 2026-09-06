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
- The checked-in results are CPU mechanism demonstrations, not GPU speedups.

These boundaries are deliberate. Claims in the README and resume should not
extend beyond the implemented and measured behavior.
