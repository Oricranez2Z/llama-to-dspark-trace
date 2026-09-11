#!/usr/bin/env python3
"""Create a compact AR/DFlash/DFlare comparison from measured artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_vllm_outputs(ar_path: Path, dflare_path: Path) -> dict[str, Any]:
    ar_rows = _load(ar_path)
    dflare_rows = _load(dflare_path)
    if len(ar_rows) != len(dflare_rows):
        return {
            "request_count_matches": False,
            "all_token_ids_match": False,
            "per_request": [],
        }

    matches = []
    for index, (ar_row, dflare_row) in enumerate(
        zip(ar_rows, dflare_rows, strict=True)
    ):
        ar_ids = ar_row["outputs"][0]["token_ids"]
        dflare_ids = dflare_row["outputs"][0]["token_ids"]
        matches.append(
            {
                "prompt_index": index,
                "token_ids_match": ar_ids == dflare_ids,
                "ar_output_tokens": len(ar_ids),
                "dflare_output_tokens": len(dflare_ids),
            }
        )
    return {
        "request_count_matches": True,
        "all_token_ids_match": all(row["token_ids_match"] for row in matches),
        "per_request": matches,
    }


def build_comparison(
    angelslim_path: Path,
    vllm_ar_path: Path | None = None,
    vllm_dflare_path: Path | None = None,
) -> dict[str, Any]:
    source = _load(angelslim_path)
    methods = []
    for row in source["summaries"]:
        methods.append(
            {
                "method": row["method"],
                "mean_time_per_output_token_seconds": row[
                    "mean_time_per_output_token_seconds"
                ],
                "mean_committed_tokens_per_round": row[
                    "mean_committed_tokens_per_round"
                ],
                "round_count": row["round_count"],
                "speedup_vs_ar": row["speedup_vs_ar"],
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "source": str(angelslim_path),
        "benchmark_claim": False,
        "environment": source["environment"],
        "config": source["config"],
        "methods": methods,
        "angelslim_lossless_check": source["all_outputs_match_ar"],
        "limitations": [
            "Four prompts and one decode length are a smoke test, not a benchmark.",
            "The methods run sequentially and do not include a concurrency sweep.",
            "RTX 8000 FP16/SDPA results must not be extrapolated to RTX 4090 "
            "BF16/FlashAttention.",
        ],
    }
    if (vllm_ar_path is None) != (vllm_dflare_path is None):
        raise ValueError("both vLLM output paths must be provided together")
    if vllm_ar_path is not None and vllm_dflare_path is not None:
        result["vllm_lossless_check"] = compare_vllm_outputs(
            vllm_ar_path, vllm_dflare_path
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angelslim-result", type=Path, required=True)
    parser.add_argument("--vllm-ar-outputs", type=Path)
    parser.add_argument("--vllm-dflare-outputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = build_comparison(
        args.angelslim_result,
        args.vllm_ar_outputs,
        args.vllm_dflare_outputs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
