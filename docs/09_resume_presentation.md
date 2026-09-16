# Resume presentation

Suggested project title:

> **LLM Serving Lab — Inference and Serving Runtime Study**

Use only statements supported by code and measured results. Example bullets:

- Implemented a NumPy Llama-style decoder with RoPE, GQA, SwiGLU, KV-cache,
  prefill/decode separation, and cached/full-forward equivalence tests.
- Built a deterministic continuous-batching simulator with token-budget
  scheduling, chunked prefill, paged KV allocation, and request-level JSONL
  traces.
- Mapped and instrumented vLLM V1 scheduler, KV-cache, model-runner, and output
  boundaries against a pinned source revision.
- Implemented lossless greedy speculative decoding and analyzed fixed versus
  confidence-scheduled verification under changing concurrency.
- Reproduced DSpark draft/verification behavior with public code and clearly
  separated algorithmic simulation from measured serving performance.
- Unified AR, EAGLE3, DFlash, DFlare, and DSpark behind one pinned vLLM V2
  benchmark contract with workload hashing, process isolation, normalized
  traces, and EOS-bounded lossless verification.

The checked-in GPU smoke tests support an integration bullet, for example:

- Instrumented vLLM V1 and DeepSpec DSpark on a Quadro RTX 8000, capturing
  scheduler/KV events and seven proposal/verification rounds with portable
  environment manifests.

The shared RTX 8000 run supports an integration/reproducibility bullet, not a
performance bullet: its automatic contention gate failed. Add the measured
speedups only after the unchanged protocol passes on an exclusive RTX 4090.
