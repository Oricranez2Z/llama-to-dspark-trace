# GPU reproduction guide

GPU dependencies are intentionally external to this small CPU-installable
package. Run each experiment with the environment belonging to its framework.
Model checkpoints are not redistributed.

## vLLM trace

Set the placeholders to your local checkout, model snapshot, and this lab:

```bash
cd <VLLM_HARNESS>
CUDA_VISIBLE_DEVICES=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_V2_MODEL_RUNNER=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=<LLM_SERVING_LAB>/src \
./.venv/bin/python \
  <LLM_SERVING_LAB>/experiments/run_vllm_gpu_experiment.py \
  --model <LOCAL_QWEN3_8B_SNAPSHOT> \
  --model-label Qwen/Qwen3-8B@b968826 \
  --output-dir <LLM_SERVING_LAB>/results/generated/vllm_gpu_smoke
```

The script initializes vLLM, warms up once, traces a three-request offline
batch, and writes `manifest.json`, `outputs.json`, and `trace.jsonl`.

## DeepSpec DSpark trace

The wrapper launches the evaluator already present in the local harness:

```bash
cd <LLM_SERVING_LAB>
uv run python experiments/run_dspark_gpu_experiment.py \
  --harness-root <VLLM_HARNESS> \
  --target-path <LOCAL_QWEN3_8B_SNAPSHOT> \
  --draft-path <LOCAL_DSPARK_CHECKPOINT> \
  --gpu 1 \
  --output-dir results/generated/dspark_gpu_smoke
```

The harness environment must contain DeepSpec's dependencies. The wrapper uses
offline mode and replaces local absolute paths with portable labels before the
result is published.

## AngelSlim DFlare and DFlash

The wrapper keeps AngelSlim in its own external environment and forces offline
checkpoint loading. On an RTX 8000 use FP16/SDPA:

```bash
cd <LLM_SERVING_LAB>
uv run python experiments/run_dflare_gpu_experiment.py \
  --python <ANGELSLIM_ENV>/bin/python \
  --angelslim-root <ANGELSLIM_CHECKOUT> \
  --target-path <QWEN3_4B_SNAPSHOT> \
  --dflare-path <DFLARE_SNAPSHOT> \
  --dflash-path <DFLASH_SNAPSHOT> \
  --dtype float16 \
  --attention sdpa \
  --max-new-tokens 32 \
  --gpu 1 \
  --output-dir results/generated/ar_dflash_dflare_rtx8000_fp16
```

On an RTX 4090, create a separate result directory and use the native path:

```bash
uv run python experiments/run_dflare_gpu_experiment.py \
  --python <ANGELSLIM_4090_ENV>/bin/python \
  --angelslim-root <ANGELSLIM_CHECKOUT> \
  --target-path <QWEN3_4B_SNAPSHOT> \
  --dflare-path <DFLARE_SNAPSHOT> \
  --dflash-path <DFLASH_SNAPSHOT> \
  --dtype bfloat16 \
  --attention flash_attention_2 \
  --max-new-tokens 32 \
  --gpu 0 \
  --output-dir results/generated/ar_dflash_dflare_rtx4090_bf16_fa2
```

The 4090 environment must contain a FlashAttention build compatible with its
PyTorch/CUDA stack. Never reuse RTX 8000 compiled kernels.

## Patched vLLM DFlare

Apply the [pinned patch](../integrations/vllm/README.md), build vLLM in that
worktree, and run the common experiment script:

```bash
cd <PATCHED_VLLM>
CUDA_VISIBLE_DEVICES=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_V2_MODEL_RUNNER=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=<PATCHED_VLLM>:<LLM_SERVING_LAB>/src \
<VLLM_ENV>/bin/python \
  <LLM_SERVING_LAB>/experiments/run_vllm_gpu_experiment.py \
  --model <QWEN3_4B_SNAPSHOT> \
  --model-label Qwen/Qwen3-4B@1cfa9a7 \
  --speculative-model <DFLARE_SNAPSHOT> \
  --speculative-model-label AngelSlim/Qwen3-4b-dflare@71dcbb0 \
  --speculative-method dflare \
  --num-speculative-tokens 15 \
  --max-tokens 128 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.7 \
  --output-dir <LLM_SERVING_LAB>/results/generated/vllm_dflare_rtx4090_bf16
```

Run the same command without the four speculative arguments into an AR control
directory. Compare output token IDs within the 4090 pair before examining any
timing. For performance work, warm all batch/sequence shapes, disable eager
mode only after correctness, and use an online serving harness.

## Visualize measured traces

```bash
make visualize-measured
```

## What the checked-in run proves

The checked-in artifacts prove that the tracing boundaries and public
reproduction commands execute on a Quadro RTX 8000. They are intentionally
small smoke tests:

| Run | Workload | Observation |
|---|---|---|
| vLLM | 3 requests, 26 input, 48 output tokens | 0.465 s batch, 103.3 output tok/s |
| DSpark | 1 GSM8K sample, 7 × 7-token proposals | mean accepted length 5.57 |
| AngelSlim DFlare | 4 prompts, 128 output tokens, FP16/SDPA | exact AR match; mean committed length 4.40 |
| Patched vLLM DFlare | 3 prompts, 24 output tokens, eager FP16 | clean patch commit; exact AR match |

The original vLLM trace source was dirty and eager execution was forced. The
patched DFlare manifest records a clean patch commit. The samples are too small
for statistics. No row is a production performance claim; read the exact
revisions and limitations from each result manifest.
