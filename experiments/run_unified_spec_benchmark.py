#!/usr/bin/env python3
"""Run AR/EAGLE3/DFlash/DFlare/DSpark under one pinned vLLM contract."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

from llm_serving_lab.benchmark import (
    aggregate_results,
    load_benchmark_spec,
    load_workload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "integrations" / "vllm" / "unified_spec_worker.py"


def _draft_mapping(values: list[str]) -> dict[str, Path]:
    mapping = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("--draft must use METHOD=PATH")
        mapping[name] = Path(raw_path).resolve()
    return mapping


def _gpu_snapshot(index: str) -> dict[str, object]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.split(", ") for row in query.stdout.splitlines() if row.strip()]
    matches = [row for row in rows if row[0] == index]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one physical GPU index {index}")
    _, gpu_uuid, name, memory_used, memory_free, utilization = matches[0]

    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    process_memory = []
    for row in processes.stdout.splitlines():
        fields = row.split(", ")
        if len(fields) == 2 and fields[0] == gpu_uuid:
            process_memory.append(int(fields[1]))
    external_compute_memory_mib = sum(process_memory)
    utilization_percent = int(utilization)
    return {
        "physical_index": int(index),
        "uuid": gpu_uuid,
        "name": name,
        "memory_used_mib": int(memory_used),
        "memory_free_mib": int(memory_free),
        "utilization_percent": utilization_percent,
        "external_compute_process_count": len(process_memory),
        "external_compute_memory_mib": external_compute_memory_mib,
        "contention_detected": (
            utilization_percent > 5 or external_compute_memory_mib > 512
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--draft", action="append", default=[])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = load_benchmark_spec(args.config)
    _, workload_sha256 = load_workload(args.workload)
    drafts = _draft_mapping(args.draft)
    required_drafts = {method.name for method in spec.methods if method.name != "ar"}
    if set(drafts) != required_drafts:
        raise ValueError(
            f"draft mapping must contain exactly {sorted(required_drafts)}"
        )

    paths = [
        args.python,
        args.vllm_root,
        args.config,
        args.workload,
        args.target_path,
        WORKER,
        *drafts.values(),
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    python_executable = args.python.absolute()
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    python_paths = [str(args.vllm_root.resolve()), str(PROJECT_ROOT / "src")]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": str(spec.seed),
            "PYTHONPATH": os.pathsep.join(python_paths),
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            # All methods use the same native GPU model runner.  Keeping this
            # explicit prevents an architecture-dependent V1/V2 fallback from
            # becoming a hidden benchmark variable.
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        }
    )

    method_order = [method.name for method in spec.methods]
    random.Random(spec.seed).shuffle(method_order)
    results = []
    gpu_snapshots = {}
    for method_name in method_order:
        gpu_snapshots[method_name] = _gpu_snapshot(args.gpu)
        method_output = output_dir / "methods" / method_name
        command = [
            str(python_executable),
            str(WORKER),
            "--config",
            str(args.config.resolve()),
            "--workload",
            str(args.workload.resolve()),
            "--method",
            method_name,
            "--target-path",
            str(args.target_path.resolve()),
            "--output-dir",
            str(method_output),
        ]
        if method_name != "ar":
            command.extend(["--draft-path", str(drafts[method_name])])
        print(f"running {method_name}", flush=True)
        subprocess.run(
            command,
            check=True,
            cwd=args.vllm_root.resolve(),
            env=environment,
        )
        results.append(
            json.loads((method_output / "result.json").read_text(encoding="utf-8"))
        )

    summary = aggregate_results(spec, workload_sha256, results)
    summary["execution_order"] = method_order
    summary["physical_gpu_index"] = args.gpu
    summary["pre_method_gpu_snapshots"] = gpu_snapshots
    contention_detected = any(
        snapshot["contention_detected"] for snapshot in gpu_snapshots.values()
    )
    summary["measurement_validity"] = {
        "status": "provisional_shared_gpu" if contention_detected else "valid",
        "contention_detected": contention_detected,
        "benchmark_claim": not contention_detected,
        "definition": (
            "contention when pre-method GPU utilization exceeds 5% or external "
            "compute allocations exceed 512 MiB"
        ),
    }
    summary["benchmark_claim"] = not contention_detected
    summary["artifacts"] = {
        method.name: f"methods/{method.name}/result.json" for method in spec.methods
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary_path)


if __name__ == "__main__":
    main()
