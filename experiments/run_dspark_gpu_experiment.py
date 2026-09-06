#!/usr/bin/env python3
"""Launch the local DeepSpec evaluator and publish portable result metadata."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _git_state(repository: Path) -> dict[str, Any]:
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


def _framework_environment(python: Path, environment: dict[str, str]) -> dict[str, Any]:
    probe = (
        "import json,sys,torch,transformers;"
        "print(json.dumps({"
        "'python':sys.version.split()[0],"
        "'torch':torch.__version__,"
        "'torch_cuda':torch.version.cuda,"
        "'transformers':transformers.__version__,"
        "'gpu':torch.cuda.get_device_name(0)"
        "}))"
    )
    result = subprocess.run(
        [str(python), "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--draft-path", type=Path, required=True)
    parser.add_argument("--target-label", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-label", default="Qwen3-8B-DSpark-5L-block7")
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    harness = args.harness_root.resolve()
    python = harness / ".venv" / "bin" / "python"
    evaluator = harness / "tools" / "eval_deepspec_subset.py"
    dataset_root = harness / "vendor" / "DeepSpec" / "eval_datasets"
    for required in (
        python,
        evaluator,
        dataset_root,
        args.target_path,
        args.draft_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_result = output_dir / "result.json"
    trace_prefix = output_dir / "trace"
    command = [
        str(python),
        str(evaluator),
        "--target-name-or-path",
        str(args.target_path.resolve()),
        "--draft-name-or-path",
        str(args.draft_path.resolve()),
        "--tasks",
        args.task,
        "--max-samples",
        str(args.max_samples),
        "--dataset-root",
        str(dataset_root),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        "0",
        "--confidence-threshold",
        "0",
        "--seed",
        "980406",
        "--trace-output",
        str(trace_prefix),
        "--output-json",
        str(raw_result),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    subprocess.run(command, check=True, cwd=harness, env=environment)

    result = json.loads(raw_result.read_text(encoding="utf-8"))
    result["mode"] = "measured_gpu_offline_smoke"
    result["benchmark_claim"] = False
    result["config"].update(
        {
            "target_name_or_path": args.target_label,
            "draft_name_or_path": args.draft_label,
            "dataset_root": "<DeepSpec>/eval_datasets",
            "trace_output": "trace",
            "output_json": "result.json",
        }
    )
    result["trace_prefix"] = "trace"
    result["rank0_trace"] = "trace.rank0.jsonl"
    result["environment"] = _framework_environment(python, environment) | {
        "gpu_index": args.gpu,
        "driver": _driver_version(),
        "deepspec_source": _git_state(harness / "vendor" / "DeepSpec"),
    }
    result["limitations"] = [
        "One-sample correctness and trace smoke test; not a speedup benchmark.",
        (
            "Evaluation time includes only this tiny workload and is not "
            "statistically stable."
        ),
        "The trace reports draft proposal time, not end-to-end per-round latency.",
    ]
    raw_result.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(raw_result)


if __name__ == "__main__":
    main()
