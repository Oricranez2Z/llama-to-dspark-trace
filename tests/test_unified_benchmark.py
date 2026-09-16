from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.prepare_deepspec_checkpoint import (
    adapt_eagle3_config,
    prepare_checkpoint,
)
from experiments.run_unified_spec_benchmark import _validate_draft_contract
from integrations.vllm.unified_spec_worker import _trace_summary
from llm_serving_lab.benchmark.unified import (
    BenchmarkSpec,
    MethodSpec,
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


def test_deepspec_dflash_can_use_anchor_sampling_runtime() -> None:
    value = {
        "name": "dflash",
        "label": "DeepSpec DFlash block7",
        "speculative_method": "dspark",
        "num_speculative_tokens": 7,
        "checkpoint_block_size": 7,
    }

    method = BenchmarkSpec.from_dict(
        {
            "schema_version": 1,
            "name": "matched-k7",
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
            "methods": [
                {
                    "name": "ar",
                    "label": "AR",
                    "speculative_method": None,
                    "num_speculative_tokens": 0,
                },
                value,
            ],
        }
    ).method("dflash")

    assert method.speculative_method == "dspark"
    assert method.checkpoint_block_size == 7


def test_deepspec_dflash_runtime_contract_checks_layout(tmp_path: Path) -> None:
    method = MethodSpec.from_dict(
        {
            "name": "dflash",
            "label": "DeepSpec DFlash block7",
            "speculative_method": "dspark",
            "num_speculative_tokens": 7,
            "checkpoint_block_size": 7,
        }
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3DSparkModel"],
                "block_size": 7,
                "markov_rank": 0,
            }
        ),
        encoding="utf-8",
    )

    _validate_draft_contract(method, tmp_path)

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3DSparkModel"],
                "block_size": 7,
                "markov_rank": 256,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="markov_rank=0"):
        _validate_draft_contract(method, tmp_path)


def test_prepare_deepspec_eagle3_checkpoint_view(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    config = {
        "architectures": ["Qwen3Eagle3Model"],
        "target_layer_ids": [1, 9, 17, 25, 33],
        "ttt_length": 7,
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")

    prepare_checkpoint(source, output)
    adapted = json.loads((output / "config.json").read_text(encoding="utf-8"))

    assert adapted["architectures"] == ["Eagle3Qwen3ForCausalLM"]
    assert adapted["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]
    assert adapted["num_aux_layers"] == 5
    assert (output / "model.safetensors").is_symlink()
    assert (output / "model.safetensors").read_bytes() == b"weights"


def test_deepspec_eagle3_adapter_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="unsupported EAGLE3 architecture"):
        adapt_eagle3_config(
            {"architectures": ["Unknown"], "target_layer_ids": [1]}
        )


def test_timing_summary_uses_median() -> None:
    summary = summarize_timings([1.0, 3.0, 2.0], output_tokens=12)

    assert summary["median_batch_seconds"] == 2.0
    assert summary["median_output_tokens_per_second"] == 6.0


def test_benchmark_dtype_defaults_and_validates() -> None:
    assert _spec().dtype == "float16"
    value = {
        "schema_version": 1,
        "name": "bad-dtype",
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
        "dtype": "float32",
        "methods": [
            {
                "name": "ar",
                "label": "AR",
                "speculative_method": None,
                "num_speculative_tokens": 0,
            }
        ],
    }
    with pytest.raises(ValueError, match="dtype"):
        BenchmarkSpec.from_dict(value)


def test_trace_summary_reports_exact_speculative_acceptance(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    events = [
        {
            "kind": "vllm_scheduler_step",
            "fields": {"scheduled_tokens": {"r1": 8}},
        },
        {
            "kind": "vllm_scheduler_step",
            "fields": {"scheduled_tokens": {"r1": 8}},
        },
        {
            "kind": "vllm_spec_decode_acceptance",
            "fields": {"num_draft_tokens": 7, "num_accepted_tokens": 3},
        },
        {
            "kind": "vllm_spec_decode_acceptance",
            "fields": {"num_draft_tokens": 7, "num_accepted_tokens": 1},
        },
    ]
    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    summary = _trace_summary(trace, output_tokens=4, request_count=1)
    acceptance = summary["speculative_acceptance"]

    assert acceptance["num_drafts"] == 2
    assert acceptance["num_accepted_tokens"] == 4
    assert acceptance["mean_acceptance_length_including_bonus"] == 3.0
    assert acceptance["per_position_acceptance_rate"][:4] == [1.0, 0.5, 0.5, 0.0]
    assert summary["decode_request_steps"] == 1
    assert summary["mean_committed_tokens_per_decode_step"] == 3.0
