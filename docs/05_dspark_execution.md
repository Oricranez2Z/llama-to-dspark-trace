# DSpark execution path

## Model-side path

```text
target auxiliary hidden states
→ projected context feature
→ parallel draft backbone
→ block hidden states
→ base draft logits
→ lightweight sequential Markov head
→ draft token block
→ confidence values
```

The parallel backbone avoids one full draft-model forward per draft position.
The sequential head restores inexpensive intra-block dependence that a purely
parallel draft lacks.

## Runtime-side path

```text
draft proposal
→ choose verification prefix
→ reserve lookahead KV slots
→ target verifies multiple positions
→ rejection sampler
→ accepted/rejected counts
→ commit request and KV state
```

## Source map for the studied revisions

| Concept | DeepSpec | vLLM |
|---|---|---|
| Proposal loop | `build_dspark_proposal` | `DSparkSpeculator.propose` |
| Parallel backbone | `forward_dspark_draft_block` | `_run_model` |
| Sequential head | `sample_draft_tokens` | `_sample_sequential` |
| Verification | `verify_draft_tokens` | target forward + rejection sampler |
| Accepted length | `VerificationResult` | `num_sampled` / `num_rejected` |
| Cache update | evaluator context | model runner + scheduler |

## Confidence scheduling

This project computes cumulative prefix survival:

```text
P(prefix survives through i) = product(confidence[0:i+1])
```

The educational scheduler increases the required survival threshold and lowers
the verification-token budget as concurrency rises.

This is a policy demonstration, not a reproduction of a private production
throughput profile. A real scheduler should use measured engine-specific cost
curves and validate throughput, TTFT, and TPOT under dynamic arrivals.
