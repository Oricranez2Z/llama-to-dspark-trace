from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_serving_lab.benchmark.unified import (
    BenchmarkSpec,
    aggregate_results,
    load_workload,
    summarize_timings,
)


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec.from_dict(
        {
            "schema_version": 1,
            "name": "test",
            "target_label": "target",
            "target_revision": "revision",
            "workload_label": "workload",
            "batch_size": 1,
            "max_tokens": 4,
            "max_model_len": 32,
            "warmup_rounds": 1,
            "repetitions": 2,
            "seed": 7,
            "gpu_memory_utilization": 0.5,
            "enforce_eager": True,
            "methods": [
                {
                    "name": "ar",
                    "label": "AR",
                    "speculative_method": None,
                    "num_speculative_tokens": 0,
                },
                {
                    "name": "dflare",
                    "label": "DFlare",
                    "speculative_method": "dflare",
                    "num_speculative_tokens": 15,
                },
            ],
        }
    )


def _result(
    spec: BenchmarkSpec, workload_hash: str, method: str, token_ids: list[int]
) -> dict:
    return {
        "benchmark_fingerprint": spec.fingerprint(workload_hash),
        "workload_sha256": workload_hash,
        "method": {"name": method},
        "target": {"eos_token_ids": [2]},
        "timing": {
            "median_batch_seconds": 2.0 if method == "ar" else 1.0,
            "median_output_tokens_per_second": 2.0 if method == "ar" else 4.0,
            "batch_seconds": [2.0, 2.0] if method == "ar" else [1.0, 1.0],
        },
        "correctness": {"stable_across_repetitions": True},
        "scheduler": {"mean_committed_tokens_per_decode_step": 1.0},
        "environment": {"gpu": "test"},
        "outputs": [{"id": "one", "token_ids": token_ids}],
    }


def test_workload_hash_covers_exact_bytes(tmp_path: Path) -> None:
    workload = tmp_path / "workload.jsonl"
    workload.write_text(
        json.dumps({"id": "one", "messages": [{"role": "user", "content": "x"}]})
        + "\n",
        encoding="utf-8",
    )

    rows, digest = load_workload(workload)

    assert rows[0]["id"] == "one"
    assert len(digest) == 64


def test_aggregate_checks_exact_tokens_and_speedup() -> None:
    spec = _spec()
    results = [
        _result(spec, "abc", "ar", [1, 2]),
        _result(spec, "abc", "dflare", [1, 2]),
    ]

    summary = aggregate_results(spec, "abc", results)

    assert summary["all_speculative_outputs_match_ar"] is True
    assert summary["methods"][1]["speedup_vs_ar"] == 2.0


def test_aggregate_exposes_lossless_mismatch() -> None:
    spec = _spec()
    results = [
        _result(spec, "abc", "ar", [1, 2]),
        _result(spec, "abc", "dflare", [1, 3]),
    ]

    summary = aggregate_results(spec, "abc", results)

    assert summary["all_speculative_outputs_match_ar"] is False
    assert summary["methods"][1]["exact_match_ar"] is False


def test_aggregate_distinguishes_post_eos_padding_from_semantic_mismatch() -> None:
    spec = _spec()
    results = [
        _result(spec, "abc", "ar", [1, 2, 8]),
        _result(spec, "abc", "dflare", [1, 2, 9]),
    ]

    summary = aggregate_results(spec, "abc", results)

    assert summary["all_speculative_outputs_match_ar_full_length"] is False
    assert summary["all_speculative_outputs_match_ar_until_stop"] is True
    assert summary["methods"][1]["exact_match_ar_until_stop"] is True


def test_invalid_ar_contract_is_rejected() -> None:
    value = {
        "name": "ar",
        "label": "AR",
        "speculative_method": "eagle3",
        "num_speculative_tokens": 1,
    }
    with pytest.raises(ValueError, match="AR cannot"):
        BenchmarkSpec.from_dict(
            {
                "schema_version": 1,
                "name": "bad",
                "target_label": "target",
                "target_revision": "revision",
                "workload_label": "workload",
                "batch_size": 1,
                "max_tokens": 1,
                "max_model_len": 2,
                "warmup_rounds": 1,
                "repetitions": 1,
                "seed": 1,
                "gpu_memory_utilization": 0.5,
                "enforce_eager": True,
                "methods": [value],
            }
        )


def test_timing_summary_uses_median() -> None:
    summary = summarize_timings([1.0, 3.0, 2.0], output_tokens=12)

    assert summary["median_batch_seconds"] == 2.0
    assert summary["median_output_tokens_per_second"] == 6.0
