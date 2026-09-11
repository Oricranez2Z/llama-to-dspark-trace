# Stage 1 — Educational DFlare model

Status: implemented; GPU-independent.

## Scope

The NumPy teaching path preserves three distinguishing DFlare contracts:

1. Selected target hidden states use shape `[sequence, target_layer, hidden]`.
2. Softmax fusion weights use shape `[draft_layer, target_layer]`.
3. One draft call predicts every position in a token block in parallel.

`DFlashProposer` is the shared-fusion control. `DFlareProposer` uses a different
target-layer mixture for every draft layer. Both satisfy the existing
`DraftProposer` protocol and are consumed without changes by
`SpeculativeDecoder` and `GreedyVerifier`.

## Deliberate simplifications

The teaching network uses deterministic random NumPy matrices instead of a
trained Qwen Transformer. It does not model RoPE, GQA, target/draft KV caches,
checkpoint quality, or GPU latency. Those belong to the external AngelSlim and
framework-integration stages.

## Evidence

- Fusion shape, softmax, and RMS-normalization tests.
- Shared-versus-layerwise fusion test.
- Block-size clamping and proposal trace test.
- End-to-end token equality with autoregressive target decoding for both
  DFlash and DFlare proposals, including rejection paths.

Upstream reference: AngelSlim commit
`ee8ddb2b43e20800bcfdda1e9ac34ea2aab5de5d`.
