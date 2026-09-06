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

The vLLM source was dirty and eager execution was forced. The DSpark sample is
too small for statistics. Neither row is a production performance claim; read
the exact revisions and limitations from each result manifest.
