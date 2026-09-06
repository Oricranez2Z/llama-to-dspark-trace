# Speculative decoding correctness

## Greedy verification

Given draft tokens `[d0, d1, d2]`, the target model computes the token it would
select at each corresponding prefix.

```text
draft:  [A, B, C, D]
target: [A, B, X]
emit:   [A, B, X]
```

Only the matching prefix is accepted. At the first mismatch, the target token
is emitted and the remaining draft suffix is discarded.

If every draft token is accepted, the verifier emits one target bonus token:

```text
draft:  [A, B, C]
target: [A, B, C, X]
emit:   [A, B, C, X]
```

## Correctness invariant

For deterministic greedy decoding:

```text
speculative_decode(target, proposer, prompt)
== autoregressive_decode(target, prompt)
```

The proposer can affect performance, but it cannot affect output tokens. The
tests inject deterministic draft errors to validate this property.

## Performance interpretation

`target_forward_calls` in the CPU simulation counts conceptual verification
rounds. It is not a GPU kernel measurement. Real performance also depends on:

- Draft-model latency.
- Verification batch shape.
- KV-cache writes and rollback behavior.
- Concurrent request pressure.
- CUDA graph tiers.
- CPU/GPU synchronization.
