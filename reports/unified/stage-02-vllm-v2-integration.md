# Stage 2 — one vLLM V2 execution path

## Runtime

All five methods execute against vLLM commit
`7160b69e68eca230d26e6e72dbc72c3886741799`, based on upstream
`ff6173997d54c5027971df8ecd1280f046a832b3`.

The four-commit patch series:

1. registers the experimental Qwen3 DFlare model and checkpoint mapping;
2. fixes CUDA checkpoint loading and provides an SM75 minimal build option;
3. accepts the native DSpark top-level mask-token contract;
4. routes DFlare through the V2 DFlash-family speculator while preserving its
   three-dimensional per-draft-layer context and per-layer KV slot mapping.

The last point is essential: flattening DFlare's fused target features into the
two-dimensional DFlash staging buffer loses the layer dimension and is not the
same algorithm.

## Validation

- vLLM changed-file Ruff checks: passed;
- DFlare model unit tests: 3 passed;
- main repository tests: 30 passed;
- AR/EAGLE3/DFlash/DFlare/DSpark 8-token V2 smokes: passed;
- every smoke matched AR token-for-token;
- formal workers record a clean vLLM commit and V2 runner.

Portable patches are under `integrations/vllm/patches/unified/`.

Status: complete for eager, CUDA, TP=1, greedy offline batches. CUDA graphs,
online serving, quantization, and distributed execution are outside this patch.
