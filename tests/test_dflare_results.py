from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "experiments" / "compare_dflare_results.py"
SPEC = importlib.util.spec_from_file_location("compare_dflare_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_comparison = MODULE.build_comparison
compare_vllm_outputs = MODULE.compare_vllm_outputs


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_compare_vllm_outputs_uses_token_ids(tmp_path: Path) -> None:
    rows = [{"outputs": [{"token_ids": [1, 2], "text": "ignored"}]}]
    ar = _write(tmp_path / "ar.json", rows)
    dflare = _write(tmp_path / "dflare.json", rows)

    result = compare_vllm_outputs(ar, dflare)

    assert result["all_token_ids_match"] is True
    assert result["per_request"][0]["ar_output_tokens"] == 2


def test_build_comparison_requires_both_vllm_paths(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "result.json",
        {
            "summaries": [],
            "environment": {},
            "config": {},
            "all_outputs_match_ar": {},
        },
    )

    with pytest.raises(ValueError, match="both vLLM"):
        build_comparison(source, tmp_path / "ar.json")
