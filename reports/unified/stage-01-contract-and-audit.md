# Stage 1 — comparison contract and repository audit

## Problem

The original repository had useful components, but AR, DFlash, DFlare, and
DSpark measurements came from different target sizes, framework revisions,
workloads, precisions, and tracing boundaries. Those artifacts remain useful as
integration smokes, but they cannot support a comparative performance claim.

## Decision

Keep llama2.c/Mini Llama as the independent model-execution learning stage.
Move every later speculative method into one vLLM offline-batch protocol:

- one Qwen3-8B target revision;
- one immutable eight-prompt workload;
- one V2 model runner and trace adapter;
- one FP16/eager/TP=1 configuration;
- isolated method processes and deterministic execution order;
- AR token IDs as the correctness oracle;
- proposal length visible as a method property, never hidden as a control.

## Deliverables

- typed configuration and aggregation contract in
  `src/llm_serving_lab/benchmark/unified.py`;
- fixed config and workload under `experiments/`;
- unit tests for fingerprints, method contracts, timing aggregation, strict
  token equality, and EOS-bounded equality.

Status: complete.
