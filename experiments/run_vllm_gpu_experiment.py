#!/usr/bin/env python3
"""Run a small, measured vLLM GPU batch and record execution boundaries.

Run this file with the Python environment belonging to the local vLLM checkout.
The lab itself deliberately does not declare vLLM or PyTorch as dependencies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from llm_serving_lab.tracing.vllm_adapter import VLLMTraceAdapter

DEFAULT_PROMPTS = [
    "Explain KV cache in one sentence.",
    "Why does continuous batching improve GPU utilization?",
    "Describe prefill and decode in two short sentences.",
]


def _git_state(source_file: str) -> dict[str, Any]:
    source = Path(source_file).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        return {"commit": None, "dirty": None}

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    versions = sorted(set(result.stdout.split()))
    return ",".join(versions) or None


def _output_row(output: Any) -> dict[str, Any]:
    candidates = []
    for candidate in output.outputs:
        candidates.append(
            {
                "index": int(candidate.index),
                "text": candidate.text,
                "token_ids": list(candidate.token_ids),
                "finish_reason": candidate.finish_reason,
            }
        )
    return {
        "request_id": str(output.request_id),
        "prompt": output.prompt,
        "prompt_token_ids": list(output.prompt_token_ids),
        "outputs": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-label",
        default=None,
        help="Portable model name written to the manifest; defaults to --model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "trace.jsonl"
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=args.seed)

    init_started = time.perf_counter()
    engine = LLM(
        model=args.model,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - init_started

    if not args.skip_warmup:
        engine.generate(["Warm up."], SamplingParams(temperature=0, max_tokens=2))
        torch.cuda.synchronize()

    with VLLMTraceAdapter(trace_path):
        torch.cuda.synchronize()
        measured_started = time.perf_counter()
        outputs = engine.generate(DEFAULT_PROMPTS, sampling)
        torch.cuda.synchronize()
        measured_seconds = time.perf_counter() - measured_started

    rows = [_output_row(output) for output in outputs]
    input_tokens = sum(len(row["prompt_token_ids"]) for row in rows)
    output_tokens = sum(
        len(candidate["token_ids"]) for row in rows for candidate in row["outputs"]
    )
    device = torch.cuda.get_device_properties(0)
    manifest = {
        "schema_version": 1,
        "mode": "measured_gpu_offline_batch_smoke",
        "benchmark_claim": False,
        "model": args.model_label or args.model,
        "sampling": {
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "workload": {
            "request_count": len(rows),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "timing": {
            "engine_init_seconds": init_seconds,
            "measured_batch_seconds": measured_seconds,
            "output_tokens_per_second": output_tokens / measured_seconds,
        },
        "environment": {
            "gpu": device.name,
            "gpu_memory_bytes": device.total_memory,
            "driver": _driver_version(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "vllm_source": _git_state(vllm.__file__),
        },
        "artifacts": {"trace": "trace.jsonl", "outputs": "outputs.json"},
        "limitations": [
            "Small offline batch; this is not an online serving benchmark.",
            "No TTFT, TPOT percentile, or concurrency sweep is reported.",
            "enforce_eager=True favors source tracing over peak performance.",
        ],
    }
    (args.output_dir / "outputs.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
