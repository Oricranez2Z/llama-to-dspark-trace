# Unified speculative-decoding benchmark

This is the comparison path for all post-llama2.c speculative-decoding work in
the repository. AR, EAGLE3, DFlash, DFlare, and DSpark run through one pinned
vLLM checkout instead of separate framework-specific harnesses.

## Comparison contract

The following dimensions are identical for every method:

| Dimension | Contract |
|---|---|
| Target | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |
| Runtime | patched vLLM `7160b69e68eca230d26e6e72dbc72c3886741799` |
| Model runner | V2 GPU model runner |
| Precision | target and draft loaded as FP16 |
| Workload | eight immutable chat prompts; SHA-256 recorded |
| Sampling | greedy, seed `20260916`, `ignore_eos=True` |
| Batch | eight requests, 64 output tokens each |
| Execution | TP=1, eager, prefix cache disabled |
| Measurement | two warmups, three timed repetitions, median reported |
| Isolation | one fresh process per method; deterministic shuffled order |
| Correctness | token equality to AR through first EOS and at full length |
| Trace | the same normalized vLLM scheduler trace schema |

Proposal length is the one intentional method-specific variable because the
released checkpoints have different native block sizes: EAGLE3 and DSpark use
`K=7`; DFlash and DFlare use `K=15`. It is always shown next to performance.

The source-of-truth inputs are:

- `experiments/configs/unified_spec_qwen3_8b_rtx8000.json`
- `experiments/workloads/unified_qwen3_8b_v1.jsonl`
- `experiments/run_unified_spec_benchmark.py`
- `integrations/vllm/unified_spec_worker.py`

## Why two correctness checks exist

Fixed-length performance runs set `ignore_eos=True`, so every request produces
exactly 64 tokens. `exact_match_ar_until_stop` compares the production-visible
sequence through AR's first EOS token. `exact_match_ar_full_length` also checks
the artificial continuation after EOS.

The RTX 8000 run is lossless through EOS for all four speculative methods.
EAGLE3 and DSpark choose a different continuation only after AR has emitted
`<|im_end|>` on one request. The repository records that strict difference
instead of hiding it.

## Run locally

Apply the patch series described in `integrations/vllm/README.md`, build vLLM
for the destination GPU, and prepare local target/draft snapshots. Then run:

```bash
cd <LLM_SERVING_LAB>
PYTHONPATH=$PWD/src <VLLM_PYTHON> experiments/run_unified_spec_benchmark.py \
  --python <VLLM_PYTHON> \
  --vllm-root <PATCHED_VLLM> \
  --config experiments/configs/unified_spec_qwen3_8b_rtx8000.json \
  --workload experiments/workloads/unified_qwen3_8b_v1.jsonl \
  --target-path <QWEN3_8B_SNAPSHOT> \
  --draft eagle3=<EAGLE3_SNAPSHOT> \
  --draft dflash=<DFLASH_SNAPSHOT> \
  --draft dflare=<DFLARE_SNAPSHOT> \
  --draft dspark=<DSPARK_SNAPSHOT> \
  --gpu 0 \
  --output-dir results/generated/unified_spec_qwen3_8b
```

The runner forces offline loading and `VLLM_USE_V2_MODEL_RUNNER=1`. It rejects
missing methods, mismatched workload fingerprints, and worker results from a
different benchmark contract.

## RTX 8000 versus RTX 4090

Do not copy compiled vLLM/Triton/FlashAttention artifacts between the cards.
The RTX 8000 is compute capability 7.5 and used the Triton attention fallback;
the RTX 4090 is compute capability 8.9 and should use a fresh environment and
fresh kernel cache. Use
`experiments/configs/unified_spec_qwen3_8b_rtx4090.json`; it changes only the
run name and memory reservation (`0.88` for a headless, exclusive 24 GiB card).
Lower that reservation if the card drives a display. Keep the workload and all
other conditions unchanged, and write to a new result directory.

Before accepting a run, require:

1. `measurement_validity.status == "valid"`;
2. every worker reports the same clean vLLM commit and `model_runner == "v2"`;
3. `all_speculative_outputs_match_ar_until_stop == true`;
4. all repetitions are stable;
5. no framework silently changes dtype or attention backend.

Only an exclusive, contention-free 4090 run should be used for a headline
performance claim. The checked-in RTX 8000 result is explicitly provisional
because another process occupied the GPU throughout the measurement.
