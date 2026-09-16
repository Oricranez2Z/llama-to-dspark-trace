#!/usr/bin/env python3
"""Run the RTX 4090 BF16 native comparison and DFlash FP16/BF16 A/B.

The script intentionally delegates every measured suite to
``run_unified_spec_benchmark.py`` so workload rendering, lossless checks,
process isolation, timing, and trace collection remain identical.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_RUNNER = PROJECT_ROOT / "experiments" / "run_unified_spec_benchmark.py"
PLOTTER = PROJECT_ROOT / "visualization" / "plot_unified_spec_comparison.py"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "configs"
    / "unified_spec_qwen3_8b_rtx4090_bf16.json"
)
DEFAULT_MATCHED_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "configs"
    / "unified_spec_qwen3_8b_rtx4090_matched_k7_bf16.json"
)
DEFAULT_WORKLOAD = (
    PROJECT_ROOT / "experiments" / "workloads" / "unified_qwen3_8b_v1.jsonl"
)
SPEC_METHODS = ("eagle3", "dflash", "dflare", "dspark")


def _draft_mapping(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or name not in SPEC_METHODS:
            raise ValueError(
                "--draft must be one of eagle3=PATH, dflash=PATH, "
                "dflare=PATH, dspark=PATH"
            )
        if name in mapping:
            raise ValueError(f"duplicate --draft entry: {name}")
        mapping[name] = Path(raw_path).expanduser().resolve()
    if set(mapping) != set(SPEC_METHODS):
        raise ValueError(f"--draft must contain exactly {list(SPEC_METHODS)}")
    return mapping


def _gpu_probe(python: Path, physical_index: str) -> dict[str, Any]:
    smi = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_index}",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in smi.stdout.strip().split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"unexpected nvidia-smi output: {smi.stdout!r}")
    index, uuid, name, total, used, utilization, driver = fields

    code = (
        "import json,torch; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'name':p.name,'capability':[p.major,p.minor],"
        "'total_memory':p.total_memory,'bf16_supported':torch.cuda.is_bf16_supported(),"
        "'torch':torch.__version__,'cuda':torch.version.cuda}))"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = physical_index
    torch_probe = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    torch_data = json.loads(torch_probe.stdout.strip().splitlines()[-1])
    return {
        "physical_index": int(index),
        "uuid": uuid,
        "name": name,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "utilization_percent": int(utilization),
        "driver": driver,
        **torch_data,
    }


def _checkpoint_contract(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    nested = config.get("dflash_config") or {}
    return {
        "path": str(path),
        "architectures": config.get("architectures"),
        "checkpoint_dtype": config.get("dtype") or config.get("torch_dtype"),
        "block_size": config.get("block_size", nested.get("block_size")),
        "mask_token_id": config.get("mask_token_id", nested.get("mask_token_id")),
        "target_layer_ids": config.get(
            "target_layer_ids", nested.get("target_layer_ids")
        ),
    }


def _write_suite_configs(
    base_path: Path, destination: Path, full_fp16: bool
) -> tuple[Path, Path]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("dtype") != "bfloat16":
        raise ValueError("the RTX 4090 base config must use dtype=bfloat16")
    names = {row["name"] for row in base["methods"]}
    if names != {"ar", *SPEC_METHODS}:
        raise ValueError("the BF16 base config must contain AR and all four methods")

    destination.mkdir(parents=True, exist_ok=True)
    bf16_path = destination / "native_bf16.json"
    bf16_path.write_text(
        json.dumps(base, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fp16 = dict(base)
    fp16["name"] = (
        "qwen3_8b_unified_spec_rtx4090_fp16_eager"
        if full_fp16
        else "qwen3_8b_dflash_dtype_control_rtx4090_fp16_eager"
    )
    fp16["dtype"] = "float16"
    if not full_fp16:
        fp16["methods"] = [
            row for row in base["methods"] if row["name"] in {"ar", "dflash"}
        ]
    fp16_path = destination / (
        "full_fp16.json" if full_fp16 else "dflash_fp16.json"
    )
    fp16_path.write_text(
        json.dumps(fp16, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return bf16_path, fp16_path


def _run_suite(
    *,
    python: Path,
    vllm_root: Path,
    config: Path,
    workload: Path,
    target: Path,
    drafts: dict[str, Path],
    gpu: str,
    output: Path,
    resume: bool,
) -> dict[str, Any]:
    summary_path = output / "summary.json"
    if resume and summary_path.is_file():
        print(f"resume: using {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    config_data = json.loads(config.read_text(encoding="utf-8"))
    required = [
        row["name"] for row in config_data["methods"] if row["name"] != "ar"
    ]
    command = [
        str(python),
        str(UNIFIED_RUNNER),
        "--python",
        str(python),
        "--vllm-root",
        str(vllm_root),
        "--config",
        str(config),
        "--workload",
        str(workload),
        "--target-path",
        str(target),
        "--gpu",
        gpu,
        "--output-dir",
        str(output),
    ]
    for method in required:
        command.extend(["--draft", f"{method}={drafts[method]}"])
    environment = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, environment.get("PYTHONPATH")) if part
    )
    print(f"starting suite {output.name}: {config_data['dtype']}", flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _method(summary: dict[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in summary["methods"] if row["method"] == name)


def _dflash_observation(summary: dict[str, Any]) -> dict[str, Any]:
    row = _method(summary, "dflash")
    acceptance = row.get("speculative_acceptance") or {}
    return {
        "dtype": summary["controlled_conditions"].get(
            "dtype", summary.get("environment", {}).get("requested_dtype")
        ),
        "median_output_tokens_per_second": row["median_output_tokens_per_second"],
        "speedup_vs_ar": row["speedup_vs_ar"],
        "mean_committed_tokens_per_decode_step": row[
            "mean_committed_tokens_per_decode_step"
        ],
        "num_drafts": acceptance.get("num_drafts"),
        "num_accepted_tokens": acceptance.get("num_accepted_tokens"),
        "draft_token_acceptance_rate": acceptance.get(
            "draft_token_acceptance_rate"
        ),
        "mean_acceptance_length_including_bonus": acceptance.get(
            "mean_acceptance_length_including_bonus"
        ),
        "per_position_acceptance_rate": acceptance.get(
            "per_position_acceptance_rate"
        ),
        "exact_match_ar_until_stop": row["exact_match_ar_until_stop"],
    }


def _precision_verdict(bf16: dict[str, Any], fp16: dict[str, Any]) -> str:
    bf16_rate = bf16.get("draft_token_acceptance_rate")
    fp16_rate = fp16.get("draft_token_acceptance_rate")
    if bf16_rate is None or fp16_rate is None:
        return "inconclusive_missing_acceptance_trace"
    if bf16_rate >= 0.05 and fp16_rate <= 0.001:
        return "fp16_collapse_reproduced_bf16_recovers"
    if bf16_rate <= 0.001:
        return "bf16_also_collapses_investigate_runtime_or_checkpoint"
    if fp16_rate > 0.001:
        return "fp16_collapse_not_reproduced_on_rtx4090"
    return "partial_bf16_recovery"


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _write_report(path: Path, result: dict[str, Any]) -> None:
    native = result["suites"]["native_bf16"]["methods"]
    ab = result["dflash_dtype_ab"]
    lines = [
        "# RTX 4090 speculative-decoding validation",
        "",
        f"- GPU: `{result['gpu_probe']['name']}`; compute capability "
        f"`{'.'.join(map(str, result['gpu_probe']['capability']))}`",
        f"- BF16 supported: `{result['gpu_probe']['bf16_supported']}`",
        f"- DFlash dtype verdict: `{ab['verdict']}`",
        "",
        "## Native-checkpoint BF16 comparison",
        "",
        "| Method | K | tok/s | vs AR | committed/step | draft acceptance | "
        "lossless to EOS |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in native:
        acceptance = row.get("speculative_acceptance") or {}
        rate = acceptance.get("draft_token_acceptance_rate")
        rate_text = "n/a" if rate is None else f"{100 * rate:.2f}%"
        lines.append(
            f"| {row['method']} | {row['num_speculative_tokens']} | "
            f"{row['median_output_tokens_per_second']:.2f} | "
            f"{row['speedup_vs_ar']:.3f}× | "
            f"{row['mean_committed_tokens_per_decode_step']:.3f} | "
            f"{rate_text} | {row['exact_match_ar_until_stop']} |"
        )
    matched = result["suites"].get("matched_k7_bf16")
    if matched is not None:
        lines.extend(
            [
                "",
                "## Matched runtime-width BF16 comparison",
                "",
                "| Method | runtime | checkpoint block | mode | K | tok/s | "
                "vs AR | draft acceptance |",
                "|---|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in matched["methods"]:
            acceptance = row.get("speculative_acceptance") or {}
            rate = acceptance.get("draft_token_acceptance_rate")
            rate_text = "n/a" if rate is None else f"{100 * rate:.2f}%"
            lines.append(
                f"| {row['method']} | {row.get('runtime_method')} | "
                f"{row.get('checkpoint_block_size') or 'n/a'} | "
                f"{row.get('block_mode')} | {row['num_speculative_tokens']} | "
                f"{row['median_output_tokens_per_second']:.2f} | "
                f"{row['speedup_vs_ar']:.3f}× | {rate_text} |"
            )
    lines.extend(
        [
            "",
            "## DFlash precision A/B",
            "",
            "| dtype | tok/s | vs same-dtype AR | accepted draft tokens | "
            "draft acceptance | mean acceptance length |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("bfloat16", "float16"):
        row = ab[key]
        rate = row.get("draft_token_acceptance_rate")
        rate_text = "n/a" if rate is None else f"{100 * rate:.2f}%"
        lines.append(
            f"| {key} | {_fmt(row.get('median_output_tokens_per_second'), 2)} | "
            f"{_fmt(row.get('speedup_vs_ar'))}× | "
            f"{row.get('num_accepted_tokens', 'n/a')} | {rate_text} | "
            f"{_fmt(row.get('mean_acceptance_length_including_bonus'))} |"
        )
    lines.extend(["", "Per-position DFlash draft acceptance:", ""])
    for key in ("bfloat16", "float16"):
        rates = ab[key].get("per_position_acceptance_rate")
        rendered = (
            "n/a"
            if rates is None
            else ", ".join(f"{100 * rate:.1f}%" for rate in rates)
        )
        lines.append(f"- `{key}`: {rendered}")
    lines.extend(
        [
            "",
            "## Gamma contract",
            "",
            "The native lane uses DFlash/DFlare `K=15` with a 16-position "
            "anchor-plus-mask block and DSpark/EAGLE3 `K=7`.",
            "The matched-runtime-width lane uses `K=7` for every method. Its "
            "DFlare checkpoint was trained at block 16 and is explicitly marked "
            "runtime-truncated, so this is not a fully matched-training ablation.",
            "",
            "## Validity gates",
            "",
        ]
    )
    for name, passed in result["validity_gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--draft", action="append", default=[])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--matched-config", type=Path, default=DEFAULT_MATCHED_CONFIG
    )
    parser.add_argument(
        "--matched-dflash-path",
        type=Path,
        help="local deepseek-ai/dflash_qwen3_8b_block7 snapshot",
    )
    parser.add_argument(
        "--skip-matched-k7",
        action="store_true",
        help="skip the K=7 runtime-width suite and run only precision validation",
    )
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--full-fp16",
        action="store_true",
        help="run all methods in FP16; default runs only the AR/DFlash control",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-non-4090", action="store_true")
    parser.add_argument("--require-exclusive", action="store_true")
    args = parser.parse_args()

    python = args.python.expanduser().resolve()
    vllm_root = args.vllm_root.expanduser().resolve()
    target = args.target_path.expanduser().resolve()
    config = args.config.expanduser().resolve()
    matched_config = args.matched_config.expanduser().resolve()
    workload = args.workload.expanduser().resolve()
    drafts = _draft_mapping(args.draft)
    output = args.output_dir.expanduser().resolve()

    matched_dflash = (
        args.matched_dflash_path.expanduser().resolve()
        if args.matched_dflash_path is not None
        else None
    )
    if not args.skip_matched_k7 and matched_dflash is None:
        raise ValueError(
            "--matched-dflash-path is required for the complete K=7 suite; "
            "use --skip-matched-k7 to run only the native precision validation"
        )
    required_paths = [python, vllm_root, target, config, workload, *drafts.values()]
    if not args.skip_matched_k7:
        assert matched_dflash is not None
        required_paths.extend([matched_config, matched_dflash])
    for item in required_paths:
        if not item.exists():
            raise FileNotFoundError(item)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    probe = _gpu_probe(python, args.gpu)
    if "4090" not in probe["name"] and not args.allow_non_4090:
        raise RuntimeError(
            f"expected an RTX 4090, found {probe['name']}; use --allow-non-4090 "
            "only for a deliberately separate hardware result"
        )
    if not probe["bf16_supported"]:
        raise RuntimeError("selected GPU/PyTorch build does not support BF16")
    if args.require_exclusive and (
        probe["memory_used_mib"] > 512 or probe["utilization_percent"] > 5
    ):
        raise RuntimeError(f"GPU is not exclusive/idle: {probe}")

    bf16_config, fp16_config = _write_suite_configs(
        config, output / "configs", args.full_fp16
    )
    matched_run_config: Path | None = None
    if not args.skip_matched_k7:
        matched_run_config = output / "configs" / "matched_k7_bf16.json"
        matched_run_config.write_text(
            matched_config.read_text(encoding="utf-8"), encoding="utf-8"
        )
    contracts = {name: _checkpoint_contract(path) for name, path in drafts.items()}
    contracts["target"] = _checkpoint_contract(target)
    expected_blocks = {"dflash": 16, "dflare": 16, "dspark": 7}
    for method, expected in expected_blocks.items():
        actual = contracts[method]["block_size"]
        if actual != expected:
            raise ValueError(
                f"{method} checkpoint must have block_size={expected}; got {actual}"
            )
    if not args.skip_matched_k7:
        assert matched_dflash is not None
        contracts["matched_dflash"] = _checkpoint_contract(matched_dflash)
        actual = contracts["matched_dflash"]["block_size"]
        if actual != 7:
            raise ValueError(
                f"matched DFlash checkpoint must have block_size=7; got {actual}"
            )
    (output / "preflight.json").write_text(
        json.dumps(
            {"gpu": probe, "checkpoints": contracts}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )

    bf16_summary = _run_suite(
        python=python,
        vllm_root=vllm_root,
        config=bf16_config,
        workload=workload,
        target=target,
        drafts=drafts,
        gpu=args.gpu,
        output=output / "native_bf16",
        resume=args.resume,
    )
    fp16_summary = _run_suite(
        python=python,
        vllm_root=vllm_root,
        config=fp16_config,
        workload=workload,
        target=target,
        drafts=drafts,
        gpu=args.gpu,
        output=output / ("full_fp16" if args.full_fp16 else "dflash_fp16"),
        resume=args.resume,
    )
    matched_summary: dict[str, Any] | None = None
    if not args.skip_matched_k7:
        assert matched_dflash is not None
        assert matched_run_config is not None
        matched_drafts = dict(drafts)
        matched_drafts["dflash"] = matched_dflash
        matched_summary = _run_suite(
            python=python,
            vllm_root=vllm_root,
            config=matched_run_config,
            workload=workload,
            target=target,
            drafts=matched_drafts,
            gpu=args.gpu,
            output=output / "matched_k7_bf16",
            resume=args.resume,
        )

    figure = output / "native_bf16.svg"
    subprocess.run(
        [
            str(python),
            str(PLOTTER),
            str(output / "native_bf16" / "summary.json"),
            str(figure),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    if matched_summary is not None:
        subprocess.run(
            [
                str(python),
                str(PLOTTER),
                str(output / "matched_k7_bf16" / "summary.json"),
                str(output / "matched_k7_bf16.svg"),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )

    bf16_dflash = _dflash_observation(bf16_summary)
    fp16_dflash = _dflash_observation(fp16_summary)
    completed_summaries = [bf16_summary, fp16_summary]
    if matched_summary is not None:
        completed_summaries.append(matched_summary)
    all_stable = all(
        row["stable_across_repetitions"]
        for summary in completed_summaries
        for row in summary["methods"]
    )
    result = {
        "schema_version": 1,
        "gpu_probe": probe,
        "checkpoint_contracts": contracts,
        "suites": {
            "native_bf16": bf16_summary,
            "fp16_control": fp16_summary,
            **(
                {"matched_k7_bf16": matched_summary}
                if matched_summary is not None
                else {}
            ),
        },
        "dflash_dtype_ab": {
            "bfloat16": bf16_dflash,
            "float16": fp16_dflash,
            "verdict": _precision_verdict(bf16_dflash, fp16_dflash),
        },
        "gamma_contract": {
            "native_checkpoint_lane": {
                "dflash": {
                    "checkpoint_block_size": 16,
                    "K": 15,
                    "verify_width": 16,
                },
                "dflare": {
                    "checkpoint_block_size": 16,
                    "K": 15,
                    "verify_width": 16,
                },
                "dspark": {
                    "checkpoint_block_size": 7,
                    "K": 7,
                    "verify_width": 8,
                },
            },
            "matched_runtime_width_lane": (
                {
                    "K": 7,
                    "dflash_checkpoint_block_size": 7,
                    "dspark_checkpoint_block_size": 7,
                    "dflare_checkpoint_block_size": 16,
                    "dflare_block_mode": "runtime_truncated",
                    "fully_matched_training_block": False,
                }
                if matched_summary is not None
                else None
            ),
        },
        "validity_gates": {
            "rtx4090": "4090" in probe["name"],
            "bf16_supported": bool(probe["bf16_supported"]),
            "native_bf16_no_contention": bool(bf16_summary.get("benchmark_claim")),
            "fp16_control_no_contention": bool(fp16_summary.get("benchmark_claim")),
            "native_bf16_lossless_to_eos": bool(
                bf16_summary["all_speculative_outputs_match_ar_until_stop"]
            ),
            "fp16_control_lossless_to_eos": bool(
                fp16_summary["all_speculative_outputs_match_ar_until_stop"]
            ),
            "all_repetitions_stable": all_stable,
            **(
                {
                    "matched_k7_lossless_to_eos": bool(
                        matched_summary[
                            "all_speculative_outputs_match_ar_until_stop"
                        ]
                    ),
                    "matched_k7_no_contention": bool(
                        matched_summary.get("benchmark_claim")
                    ),
                }
                if matched_summary is not None
                else {}
            ),
        },
        "artifacts": {
            "native_bf16": "native_bf16/summary.json",
            "fp16_control": (
                "full_fp16/summary.json"
                if args.full_fp16
                else "dflash_fp16/summary.json"
            ),
            "figure": figure.name,
            **(
                {
                    "matched_k7": "matched_k7_bf16/summary.json",
                    "matched_k7_figure": "matched_k7_bf16.svg",
                }
                if matched_summary is not None
                else {}
            ),
            "report": "REPORT.md",
        },
    }
    result_path = output / "validation_summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(output / "REPORT.md", result)
    print(result_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
