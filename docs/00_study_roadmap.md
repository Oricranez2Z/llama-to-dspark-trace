# Source-reading roadmap: Llama → vLLM → DSpark

The goal is to follow one token from model math to a production-style serving
state machine, then understand how speculative decoding changes that loop.
Use the local source checkouts listed in `third_party/README.md`; do not begin by
reading every directory.

## Working method

For each stage, repeat the same five actions:

1. Write the smallest call graph before reading implementation details.
2. Run one request and stop at the boundary currently under study.
3. Record tensor shapes, request state, cache state, and ownership changes.
4. Reimplement the mechanism in this repository and test one invariant.
5. Compare the trace with the upstream implementation and write the difference.

Keep a source-reading record with four columns:

| Entry | Question |
|---|---|
| Entry function | Where does control enter this subsystem? |
| State | Which fields change before and after the call? |
| Data shape | What are the batch, sequence, head, and cache dimensions? |
| Invariant | What must remain true if the implementation is correct? |

## Stage 1 — single-token generation

Start with llama2.c to establish the irreducible autoregressive loop:

```text
token → embedding → transformer layers → logits → sampler → next token
```

Trace prompt prefill once, then trace one-token decode. Explain why prefill can
process many positions together while decode appends one KV entry per layer.
Use `mini_llm` to validate cached versus uncached logits and generated tokens.

Deliverable: `docs/01_token_generation.md`, Mini Llama tests, and a shape table.

## Stage 2 — serving one request

Enter vLLM through `LLM.generate`, follow request construction, scheduling,
KV-slot allocation, the GPU model runner, output processing, and detokenization.
Ignore networking and multi-GPU execution until this path is clear.

Run `examples/trace_vllm.py` or the measured GPU runner. Explain every event in
one scheduler step and identify the prefill-to-decode transition.

Deliverable: `docs/03_vllm_source_map.md` and a measured scheduler trace.

## Stage 3 — serving many requests

Study continuous batching as a state machine rather than as model code:

```text
waiting → admitted → prefill/decode scheduled → executed → committed → finished
```

Vary token budget, prompt lengths, output lengths, block size, and arrival time
in `mini_serving`. Check that work never exceeds budget and all KV blocks are
released after completion. Then compare the local events with vLLM's scheduler.

Deliverable: scheduler/KV visualizations and invariant tests.

## Stage 4 — speculative decoding correctness

Implement the target greedy sequence first. Add a draft proposal, target
verification, accepted prefix, replacement token, and bonus token. Treat
`speculative output == target greedy output` as the primary invariant.

Only after correctness is stable should you measure acceptance length and
forward-pass reduction.

Deliverable: `speculative` tests and the fixed-length experiment.

## Stage 5 — DSpark execution and policy

Read DeepSpec in this order: evaluator entry, proposal construction, parallel
draft backbone, Markov head, target verification, metrics. Trace a single
sample before attempting serving integration.

Separate three questions:

- Does the draft/verify algorithm preserve target output semantics?
- How many proposed tokens survive at each position?
- Does an adaptive proposal length improve end-to-end serving metrics?

The checked-in one-sample GPU trace answers only the second at smoke-test scale.
A resume-grade performance claim requires the matrix in `docs/06_experiments.md`.

## Suggested twelve-week cadence

| Weeks | Focus | Evidence |
|---|---|---|
| 1–2 | llama2.c and Mini Llama | shape notes, cache-equivalence tests |
| 3–4 | vLLM single-request path | call graph, instrumented trace |
| 5–6 | scheduler and paged KV cache | timeline, block map, invariants |
| 7–8 | API/streaming and concurrency | request lifecycle notes |
| 9–10 | speculative decoding | lossless verifier and sweep |
| 11 | DeepSpec/DSpark | proposal/verification trace |
| 12 | benchmark and presentation | manifests, figures, conclusions |

Do not count a section as complete until another reader can reproduce its
artifact from a command and connect the artifact back to named source boundaries.
