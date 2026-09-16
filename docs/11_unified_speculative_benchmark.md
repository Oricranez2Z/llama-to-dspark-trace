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
| Precision | configured per suite and recorded (`float16` or `bfloat16`) |
| Workload | eight immutable chat prompts; SHA-256 recorded |
| Sampling | greedy, seed `20260916`, `ignore_eos=True` |
| Batch | eight requests, 64 output tokens each |
| Execution | TP=1, eager, prefix cache disabled |
| Measurement | two warmups, three timed repetitions, median reported |
| Isolation | one fresh process per method; deterministic shuffled order |
| Correctness | token equality to AR through first EOS and at full length |
| Trace | the same normalized vLLM scheduler trace schema |

The repository now keeps two distinct comparison contracts. The native-system
4090 precision diagnostic retains each released checkpoint's native proposal
length. The matched-width comparison sets every method to `K=7`: EAGLE3 and
DSpark remain native, DFlash uses DeepSpec's native block-7 checkpoint, and the
released DFlare block-16 checkpoint is truncated to `K=7` at runtime. The last
case controls execution width but is not a matched-training-block ablation.

The source-of-truth matched-width inputs are:

- `experiments/configs/unified_spec_qwen3_8b_rtx8000.json`
- `experiments/configs/unified_spec_qwen3_8b_rtx4090_matched_k7_bf16.json`
- `experiments/configs/unified_spec_qwen3_8b_rtx4090_bf16.json` (native lane)
- `experiments/workloads/unified_qwen3_8b_v1.jsonl`
- `experiments/run_unified_spec_benchmark.py`
- `experiments/run_rtx4090_validation.py`
- `integrations/vllm/unified_spec_worker.py`

## Why two correctness checks exist

Fixed-length performance runs set `ignore_eos=True`, so every request produces
exactly 64 tokens. `exact_match_ar_until_stop` compares the production-visible
sequence through AR's first EOS token. `exact_match_ar_full_length` also checks
the artificial continuation after EOS.

The RTX 8000 run is lossless through EOS for all four speculative methods.
DFlash and DSpark choose a different continuation only after AR has emitted
`<|im_end|>` on one request. The repository records that strict difference
instead of hiding it.

## Run locally

Apply the patch series described in `integrations/vllm/README.md`, build vLLM
for the destination GPU, and prepare local target/draft snapshots. The released
DeepSpec EAGLE3 config uses its training-code architecture name; create a small
vLLM-compatible view without copying or modifying the weights:

```bash
python experiments/prepare_deepspec_checkpoint.py \
  --source <DEEPSPEC_EAGLE3_SNAPSHOT> \
  --output <EAGLE3_VLLM_VIEW>
```

Then run:

```bash
cd <LLM_SERVING_LAB>
PYTHONPATH=$PWD/src <VLLM_PYTHON> experiments/run_unified_spec_benchmark.py \
  --python <VLLM_PYTHON> \
  --vllm-root <PATCHED_VLLM> \
  --config experiments/configs/unified_spec_qwen3_8b_rtx8000.json \
  --workload experiments/workloads/unified_qwen3_8b_v1.jsonl \
  --target-path <QWEN3_8B_SNAPSHOT> \
  --draft eagle3=<EAGLE3_VLLM_VIEW> \
  --draft dflash=<DEEPSPEC_DFLASH_BLOCK7_SNAPSHOT> \
  --draft dflare=<DFLARE_SNAPSHOT> \
  --draft dspark=<DSPARK_SNAPSHOT> \
  --gpu 0 \
  --output-dir results/generated/unified_spec_qwen3_8b
```

The DeepSpec DFlash checkpoint uses the common `Qwen3DSparkModel` container
with `markov_rank=0`; the benchmark therefore routes it through the anchor-
sampling DSpark runtime while continuing to report the algorithm as DFlash.
Routing it through the z-lab mask-layout DFlash runtime is a semantic mismatch.

The runner forces offline loading and `VLLM_USE_V2_MODEL_RUNNER=1`. It rejects
missing methods, mismatched workload fingerprints, and worker results from a
different benchmark contract.

## RTX 8000 versus RTX 4090

Do not copy compiled vLLM/Triton/FlashAttention artifacts between the cards.
The RTX 8000 is compute capability 7.5 and used the Triton attention fallback;
the RTX 4090 is compute capability 8.9 and should use a fresh environment and
fresh kernel cache. Use
`experiments/configs/unified_spec_qwen3_8b_rtx4090_bf16.json` for the native
lane and `experiments/configs/unified_spec_qwen3_8b_rtx4090_matched_k7_bf16.json`
for the K=7 lane. Both reserve `0.88` for a headless, exclusive 24 GiB card.
Lower that reservation if the card drives a display, and write to a new result
directory.

Before accepting a run, require:

1. `measurement_validity.status == "valid"`;
2. every worker reports the same clean vLLM commit and `model_runner == "v2"`;
3. `all_speculative_outputs_match_ar_until_stop == true`;
4. all repetitions are stable;
5. no framework silently changes dtype or attention backend.

Only an exclusive, contention-free 4090 run should be used for a headline
performance claim. The checked-in RTX 8000 result is explicitly provisional
because another process occupied the GPU throughout the measurement.

## One-command RTX 4090 validation

`run_rtx4090_validation.py` packages the recommended follow-up into one run:

1. verifies that the selected device is an RTX 4090 and that PyTorch exposes
   BF16 support;
2. snapshots GPU and checkpoint metadata;
3. runs AR/EAGLE3/DFlash/DFlare/DSpark in BF16 with their native checkpoint
   proposal lengths;
4. runs a same-workload FP16 AR/DFlash control;
5. runs a BF16 `K=7` runtime-width comparison using the DeepSpec DFlash block-7
   checkpoint;
6. records exact scheduler-reported acceptance counts, per-position acceptance
   rates, lossless correctness, timing, memory, and contention;
7. writes a machine-readable summary, Markdown report, and SVG figures.

Prepare local snapshots first. The runner is offline by design and will not
silently download a different model revision. On the 4090 host, run:

```bash
cd <LLM_SERVING_LAB>

PYTHONPATH=$PWD/src <VLLM_PYTHON> experiments/run_rtx4090_validation.py \
  --python <VLLM_PYTHON> \
  --vllm-root <PATCHED_VLLM> \
  --target-path <QWEN3_8B_SNAPSHOT> \
  --draft eagle3=<EAGLE3_VLLM_VIEW> \
  --draft dflash=<DFLASH_B16_SNAPSHOT> \
  --draft dflare=<DFLARE_B16_SNAPSHOT> \
  --draft dspark=<DSPARK_BLOCK7_CONVERTED_SNAPSHOT> \
  --matched-dflash-path <DEEPSPEC_DFLASH_BLOCK7_SNAPSHOT> \
  --gpu 0 \
  --require-exclusive \
  --output-dir results/generated/rtx4090_validation
```

The default FP16 control contains only AR and DFlash because its purpose is to
isolate the precision hypothesis without doubling the full benchmark cost. Add
`--full-fp16` to run all five methods in both precisions. If a long run is
interrupted after a suite has written its `summary.json`, rerun the identical
command with `--resume`. Use `--skip-matched-k7` only when the DeepSpec DFlash
checkpoint is unavailable and the goal is limited to the precision diagnosis.

The primary hand-off files are:

- `results/generated/rtx4090_validation/REPORT.md`;
- `results/generated/rtx4090_validation/validation_summary.json`;
- `results/generated/rtx4090_validation/native_bf16/summary.json`;
- `results/generated/rtx4090_validation/dflash_fp16/summary.json`;
- `results/generated/rtx4090_validation/matched_k7_bf16/summary.json`;
- per-method `result.json` and `trace.jsonl` below each suite.

Interpret `dflash_dtype_ab.verdict` as follows:

| Verdict | Meaning |
|---|---|
| `fp16_collapse_reproduced_bf16_recovers` | Strong evidence that precision caused the RTX 8000 failure |
| `bf16_also_collapses_investigate_runtime_or_checkpoint` | Precision alone is insufficient; inspect integration/checkpoint alignment |
| `fp16_collapse_not_reproduced_on_rtx4090` | The failure depends on more than dtype, such as GPU/backend behavior |
| `partial_bf16_recovery` | BF16 helps, but the separation is not decisive |

The output intentionally contains two comparison lanes. The native lane uses
DFlash/DFlare `K=15` and DSpark/EAGLE3 `K=7`. The matched-runtime-width lane
uses `K=7` for every method, but DFlare still comes from a block-16 checkpoint
and is marked `runtime_truncated`; it is therefore not a fully matched-training
ablation.
