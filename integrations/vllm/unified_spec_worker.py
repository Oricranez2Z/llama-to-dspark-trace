#!/usr/bin/env python3
"""Run one method in an isolated process for the unified vLLM benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from llm_serving_lab.benchmark.unified import (
    load_benchmark_spec,
    load_workload,
    summarize_timings,
)
from llm_serving_lab.tracing.vllm_adapter import VLLMTraceAdapter


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
    return ",".join(sorted(set(result.stdout.split()))) or None


def _render_workload(tokenizer: Any, rows: list[dict[str, Any]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]


def _output_rows(ids: list[str], outputs: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for request_id, output in zip(ids, outputs, strict=True):
        candidate = output.outputs[0]
        rows.append(
            {
                "id": request_id,
                "prompt_tokens": len(output.prompt_token_ids),
                "token_ids": list(candidate.token_ids),
                "output_tokens": len(candidate.token_ids),
                "text": candidate.text,
            }
        )
    return rows


def _trace_summary(
    path: Path, output_tokens: int, request_count: int
) -> dict[str, Any]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    scheduled = [
        event
        for event in events
        if event["kind"] == "vllm_scheduler_step"
        and event["fields"].get("scheduled_tokens")
    ]
    prefill_steps = 1 if scheduled else 0
    decode_steps = max(0, len(scheduled) - prefill_steps)
    committed = output_tokens / (decode_steps * request_count) if decode_steps else 0.0
    return {
        "nonempty_scheduler_steps": len(scheduled),
        "assumed_prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "mean_committed_tokens_per_decode_step": committed,
        "definition": (
            "output tokens / (post-prefill scheduler steps * request count); "
            "all requests use ignore_eos and equal output length"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--draft-path")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    spec = load_benchmark_spec(args.config)
    method = spec.method(args.method)
    workload, workload_sha256 = load_workload(args.workload)
    if len(workload) != spec.batch_size:
        raise ValueError("workload size must equal configured batch_size")
    if method.name != "ar" and args.draft_path is None:
        raise ValueError(f"{method.name} requires --draft-path")

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_path, local_files_only=True, trust_remote_code=False
    )
    prompts = _render_workload(tokenizer, workload)
    sampling = SamplingParams(
        temperature=0,
        max_tokens=spec.max_tokens,
        ignore_eos=True,
        seed=spec.seed,
    )
    speculative_config = None
    if method.name != "ar":
        speculative_config = {
            "method": method.speculative_method,
            "model": args.draft_path,
            "num_speculative_tokens": method.num_speculative_tokens,
        }

    init_started = time.perf_counter()
    engine = LLM(
        model=args.target_path,
        dtype="float16",
        tensor_parallel_size=1,
        enforce_eager=spec.enforce_eager,
        max_model_len=spec.max_model_len,
        max_num_seqs=spec.batch_size,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        enable_prefix_caching=False,
        disable_log_stats=True,
        seed=spec.seed,
        speculative_config=speculative_config,
    )
    torch.cuda.synchronize()
    engine_init_seconds = time.perf_counter() - init_started

    for _ in range(spec.warmup_rounds):
        engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    timings = []
    canonical_outputs: list[dict[str, Any]] | None = None
    stable = True
    request_ids = [str(row["id"]) for row in workload]
    for _ in range(spec.repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        rows = _output_rows(request_ids, outputs)
        if canonical_outputs is None:
            canonical_outputs = rows
        else:
            stable = stable and all(
                left["token_ids"] == right["token_ids"]
                for left, right in zip(canonical_outputs, rows, strict=True)
            )

    assert canonical_outputs is not None
    output_tokens = sum(row["output_tokens"] for row in canonical_outputs)
    input_tokens = sum(row["prompt_tokens"] for row in canonical_outputs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "trace.jsonl"
    with VLLMTraceAdapter(trace_path):
        traced_outputs = engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
    traced_rows = _output_rows(request_ids, traced_outputs)
    trace_matches = all(
        left["token_ids"] == right["token_ids"]
        for left, right in zip(canonical_outputs, traced_rows, strict=True)
    )

    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "benchmark_fingerprint": spec.fingerprint(workload_sha256),
        "workload_sha256": workload_sha256,
        "method": {
            "name": method.name,
            "label": method.label,
            "speculative_method": method.speculative_method,
            "num_speculative_tokens": method.num_speculative_tokens,
            "draft_revision": method.draft_revision,
        },
        "target": {
            "label": spec.target_label,
            "revision": spec.target_revision,
            "eos_token_ids": (
                [int(tokenizer.eos_token_id)]
                if tokenizer.eos_token_id is not None
                else []
            ),
        },
        "workload": {
            "label": spec.workload_label,
            "request_count": len(workload),
            "input_tokens": input_tokens,
            "output_tokens_per_repetition": output_tokens,
        },
        "timing": {
            "engine_init_seconds": engine_init_seconds,
            "warmup_rounds": spec.warmup_rounds,
            "repetitions": spec.repetitions,
            **summarize_timings(timings, output_tokens),
        },
        "correctness": {
            "stable_across_repetitions": stable,
            "traced_run_matches_timed_run": trace_matches,
        },
        "scheduler": _trace_summary(trace_path, output_tokens, len(workload)),
        "environment": {
            "gpu": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_bytes": properties.total_memory,
            "driver": _driver_version(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "vllm_source": _git_state(vllm.__file__),
            "model_runner": (
                "v2" if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1" else "v1"
            ),
            "requested_dtype": "float16",
            "effective_target_dtype": str(engine.llm_engine.model_config.dtype),
            "enforce_eager": spec.enforce_eager,
            "engine_seed": spec.seed,
            "attention_backend_request": "auto",
        },
        "memory": {
            "max_allocated_bytes_after_warmup": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes_after_warmup": torch.cuda.max_memory_reserved(),
        },
        "outputs": canonical_outputs,
        "artifacts": {"trace": "trace.jsonl"},
    }
    output_path = args.output_dir / "result.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()
