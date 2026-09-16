"""Configuration and result contracts for the unified speculative benchmark."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

METHODS = {"ar", "eagle3", "dflash", "dflare", "dspark"}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    speculative_method: str | None
    num_speculative_tokens: int
    draft_revision: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MethodSpec:
        method = cls(
            name=str(value["name"]),
            label=str(value["label"]),
            speculative_method=value.get("speculative_method"),
            num_speculative_tokens=int(value.get("num_speculative_tokens", 0)),
            draft_revision=value.get("draft_revision"),
        )
        if method.name not in METHODS:
            raise ValueError(f"unsupported method: {method.name}")
        if method.name == "ar":
            if method.speculative_method is not None or method.num_speculative_tokens:
                raise ValueError("AR cannot have a speculative method or draft length")
        elif (
            method.speculative_method != method.name
            or method.num_speculative_tokens <= 0
        ):
            raise ValueError(f"invalid speculative contract for {method.name}")
        return method


@dataclass(frozen=True)
class BenchmarkSpec:
    schema_version: int
    name: str
    target_label: str
    target_revision: str
    workload_label: str
    batch_size: int
    max_tokens: int
    max_model_len: int
    warmup_rounds: int
    repetitions: int
    seed: int
    gpu_memory_utilization: float
    enforce_eager: bool
    methods: tuple[MethodSpec, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BenchmarkSpec:
        methods = tuple(MethodSpec.from_dict(row) for row in value["methods"])
        names = [method.name for method in methods]
        if len(names) != len(set(names)):
            raise ValueError("method names must be unique")
        if "ar" not in names:
            raise ValueError("the unified benchmark requires an AR control")
        spec = cls(
            schema_version=int(value["schema_version"]),
            name=str(value["name"]),
            target_label=str(value["target_label"]),
            target_revision=str(value["target_revision"]),
            workload_label=str(value["workload_label"]),
            batch_size=int(value["batch_size"]),
            max_tokens=int(value["max_tokens"]),
            max_model_len=int(value["max_model_len"]),
            warmup_rounds=int(value["warmup_rounds"]),
            repetitions=int(value["repetitions"]),
            seed=int(value["seed"]),
            gpu_memory_utilization=float(value["gpu_memory_utilization"]),
            enforce_eager=bool(value["enforce_eager"]),
            methods=methods,
        )
        if spec.schema_version != 1:
            raise ValueError("only benchmark schema version 1 is supported")
        if (
            min(
                spec.batch_size,
                spec.max_tokens,
                spec.max_model_len,
                spec.warmup_rounds,
                spec.repetitions,
            )
            <= 0
        ):
            raise ValueError("benchmark dimensions must be positive")
        if not 0 < spec.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        return spec

    def method(self, name: str) -> MethodSpec:
        return next(method for method in self.methods if method.name == name)

    def fingerprint(self, workload_sha256: str) -> str:
        payload = asdict(self)
        payload["workload_sha256"] = workload_sha256
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_benchmark_spec(path: Path) -> BenchmarkSpec:
    return BenchmarkSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_workload(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not rows:
        raise ValueError("workload must not be empty")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("workload request ids must be unique")
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"request {row['id']} must contain messages")
    return rows, hashlib.sha256(raw).hexdigest()


def _token_map(result: dict[str, Any]) -> dict[str, list[int]]:
    return {row["id"]: row["token_ids"] for row in result["outputs"]}


def _matches_until_first_stop(
    reference: dict[str, list[int]],
    candidate: dict[str, list[int]],
    stop_token_ids: set[int],
) -> bool:
    if set(reference) != set(candidate):
        return False
    for request_id, reference_tokens in reference.items():
        boundary = len(reference_tokens)
        for index, token_id in enumerate(reference_tokens):
            if token_id in stop_token_ids:
                boundary = index + 1
                break
        if candidate[request_id][:boundary] != reference_tokens[:boundary]:
            return False
    return True


def aggregate_results(
    spec: BenchmarkSpec,
    workload_sha256: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_fingerprint = spec.fingerprint(workload_sha256)
    by_name = {result["method"]["name"]: result for result in results}
    expected_names = {method.name for method in spec.methods}
    if set(by_name) != expected_names:
        raise ValueError("worker results do not match configured methods")
    for result in results:
        if result["benchmark_fingerprint"] != expected_fingerprint:
            raise ValueError("worker benchmark fingerprint mismatch")
        if result["workload_sha256"] != workload_sha256:
            raise ValueError("worker workload hash mismatch")

    reference = _token_map(by_name["ar"])
    stop_token_ids = {
        int(token_id) for token_id in by_name["ar"]["target"].get("eos_token_ids", [])
    }
    if not stop_token_ids:
        raise ValueError("AR worker result must report at least one EOS token id")
    ar_latency = by_name["ar"]["timing"]["median_batch_seconds"]
    rows = []
    for method in spec.methods:
        result = by_name[method.name]
        token_map = _token_map(result)
        full_length_exact = token_map == reference
        until_stop_exact = _matches_until_first_stop(
            reference, token_map, stop_token_ids
        )
        latency = result["timing"]["median_batch_seconds"]
        rows.append(
            {
                "method": method.name,
                "label": method.label,
                "num_speculative_tokens": method.num_speculative_tokens,
                "median_batch_seconds": latency,
                "median_output_tokens_per_second": result["timing"][
                    "median_output_tokens_per_second"
                ],
                "speedup_vs_ar": ar_latency / latency,
                "exact_match_ar": full_length_exact,
                "exact_match_ar_full_length": full_length_exact,
                "exact_match_ar_until_stop": until_stop_exact,
                "stable_across_repetitions": result["correctness"][
                    "stable_across_repetitions"
                ],
                "mean_committed_tokens_per_decode_step": result["scheduler"][
                    "mean_committed_tokens_per_decode_step"
                ],
                "batch_seconds": result["timing"]["batch_seconds"],
            }
        )

    return {
        "schema_version": 1,
        "mode": "unified_vllm_offline_batch",
        "benchmark_claim": True,
        "benchmark_fingerprint": expected_fingerprint,
        "workload_sha256": workload_sha256,
        "target": {
            "label": spec.target_label,
            "revision": spec.target_revision,
        },
        "controlled_conditions": {
            "workload": spec.workload_label,
            "batch_size": spec.batch_size,
            "max_tokens": spec.max_tokens,
            "max_model_len": spec.max_model_len,
            "warmup_rounds": spec.warmup_rounds,
            "repetitions": spec.repetitions,
            "temperature": 0,
            "ignore_eos": True,
            "seed": spec.seed,
            "enforce_eager": spec.enforce_eager,
            "prefix_caching": False,
        },
        "methods": rows,
        "all_speculative_outputs_match_ar": all(
            row["exact_match_ar_full_length"] for row in rows if row["method"] != "ar"
        ),
        "all_speculative_outputs_match_ar_full_length": all(
            row["exact_match_ar_full_length"] for row in rows if row["method"] != "ar"
        ),
        "all_speculative_outputs_match_ar_until_stop": all(
            row["exact_match_ar_until_stop"] for row in rows if row["method"] != "ar"
        ),
        "environment": by_name["ar"]["environment"],
        "notes": [
            "All methods use one vLLM source revision and one target checkpoint.",
            "Proposal lengths remain checkpoint-native and are reported per method.",
            "Lossless correctness is gated through the first EOS token; strict "
            "fixed-length equality is also reported because ignore_eos=True.",
            "Results describe this fixed offline batch on this GPU, not online "
            "serving.",
        ],
    }


def summarize_timings(seconds: list[float], output_tokens: int) -> dict[str, Any]:
    throughputs = [output_tokens / value for value in seconds]
    return {
        "batch_seconds": seconds,
        "mean_batch_seconds": statistics.fmean(seconds),
        "median_batch_seconds": statistics.median(seconds),
        "min_batch_seconds": min(seconds),
        "max_batch_seconds": max(seconds),
        "median_output_tokens_per_second": statistics.median(throughputs),
    }
