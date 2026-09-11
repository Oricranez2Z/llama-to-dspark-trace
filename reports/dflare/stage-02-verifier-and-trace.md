# Stage 2 — Lossless verification and trace integration

Status: implemented; GPU-independent.

## Event sequence

Each educational DFlare round emits:

```text
dflare_block_proposal
→ speculative_round
```

The proposal event records target/fused tensor shapes, selected target layers,
per-draft-layer fusion weights, and proposed token IDs. The generic
`speculative_round` records target verification, accepted/rejected tokens,
emitted tokens, and bonus-token use.

This separation makes proposal architecture observable without coupling the
existing lossless verifier to DFlare-specific state.

## Correctness boundary

The generated sequence is always defined by the target model. Draft quality
changes only the number of accepted tokens and target verification rounds; it
must never change final greedy output.
