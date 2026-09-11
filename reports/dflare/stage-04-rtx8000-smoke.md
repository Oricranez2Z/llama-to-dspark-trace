# Stage 4 — Qwen3-4B RTX 8000 smoke

Status: complete on 2026-09-11.

## Goal and outcome

The official Qwen3-4B target and DFlare draft were pinned, downloaded, and
validated before inference. The isolated AngelSlim runner then completed a
single-GPU FP16/SDPA smoke on physical GPU 1. DFlare generated exactly the same
token IDs as greedy autoregressive decoding for all prompts.

| Artifact | Immutable revision | Local validation |
|---|---|---|
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | 3 safetensors shards, 398 keys, 8,044,982,000 bytes |
| `AngelSlim/Qwen3-4b-dflare` | `71dcbb0645c9357c938ab4e30a677c5a63bff900` | 1 safetensors file, 94 keys, 1,486,452,278 bytes |
| AngelSlim source | `ee8ddb2b43e20800bcfdda1e9ac34ea2aab5de5d` | isolated draft and benchmark imports passed |

The model card defines a 16-token block and target layers
`[1, 5, 9, 13, 17, 21, 25, 29, 33]`; the runner reads these fields from the
checkpoint rather than overriding them.

## Measured smoke

Environment: Quadro RTX 8000, compute capability 7.5, driver 560.35.03,
PyTorch 2.6.0+cu124, Transformers 4.57.6, FP16 and SDPA.

| Method | Output tokens | Mean TPOT | Mean committed/round | Exact vs AR |
|---|---:|---:|---:|---:|
| AR | 24 | 51.997 ms | 1.00 | reference |
| DFlare | 24 | 14.209 ms | 8.00 | yes |

These three short requests validate loading, proposal, verification, and output
commit. The apparent 3.66× ratio is not a benchmark claim: the sample is tiny,
sequential, and lacks warmup repetitions, percentiles, and concurrency sweeps.
The source artifact is
[`result.json`](../../results/measured/dflare_rtx8000_fp16_smoke/result.json).

## RTX 8000 versus RTX 4090 gate

| Concern | RTX 8000 run | RTX 4090 plan | Consequence |
|---|---|---|---|
| Compute capability | 7.5 | 8.9 | Compile a fresh kernel cache; do not copy binaries |
| Dtype | FP16 fallback | BF16 native | Re-run exact-output checks because ties/numerics may differ |
| Attention | SDPA/FlexAttention fallback | FlashAttention 2 candidate | Re-measure TTFT/TPOT and acceptance independently |
| Memory | 48 GiB | 24 GiB | Lower memory utilization/concurrency before changing model semantics |
| Thermals/clocks | workstation Turing | consumer Ada | Record power mode, clocks, driver, and warmup policy |

Hardware changes may alter numerical tie-breaking and therefore the particular
greedy token sequence, but lossless verification must still make speculative
output equal to the AR output produced under the same hardware/software
configuration. Equality is checked within each machine, never against the
RTX 8000 token file.
