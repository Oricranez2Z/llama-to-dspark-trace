#!/usr/bin/env python3
"""Execute AngelSlim DFlash/DFlare models without installing its full package.

This process runs inside a CUDA-enabled external environment. It imports the
pinned model modules and benchmark loop from an AngelSlim source checkout while
avoiding unrelated package-level training and deployment dependencies.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_PROMPTS = [
    "Explain why KV caching accelerates autoregressive decoding in one sentence.",
    "What is the difference between prefill and decode?",
    "Give two benefits of continuous batching.",
    "How many positive whole-number divisors does 196 have?",
]


def _register_namespace(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def _prepare_angelslim_namespaces(root: Path) -> None:
    packages = [
        ("angelslim", root / "angelslim"),
        ("angelslim.compressor", root / "angelslim" / "compressor"),
        (
            "angelslim.compressor.speculative",
            root / "angelslim" / "compressor" / "speculative",
        ),
        (
            "angelslim.compressor.speculative.train",
            root / "angelslim" / "compressor" / "speculative" / "train",
        ),
        (
            "angelslim.compressor.speculative.train.models",
            root / "angelslim" / "compressor" / "speculative" / "train" / "models",
        ),
        (
            "angelslim.compressor.speculative.train.models.draft",
            root
            / "angelslim"
            / "compressor"
            / "speculative"
            / "train"
            / "models"
            / "draft",
        ),
    ]
    for name, path in packages:
        if not path.is_dir():
            raise FileNotFoundError(path)
        _register_namespace(name, path)


def _load_benchmark_module(root: Path):
    path = root / "tools" / "dflash_benchmark.py"
    spec = importlib.util.spec_from_file_location("angelslim_dflare_benchmark", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_draft_contract(root: Path, architecture: str):
    _prepare_angelslim_namespaces(root)
    module_name = (
        f"angelslim.compressor.speculative.train.models.draft.qwen_{architecture}"
    )
    module = importlib.import_module(module_name)
    class_name = (
        "QwenDFlareDraftModel" if architecture == "dflare" else "QwenDFlashDraftModel"
    )
    return (
        getattr(module, class_name),
        module.sample,
        module.extract_context_feature,
    )


def _dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _request_row(result: Any, tokenizer: Any, prompt_index: int) -> dict[str, Any]:
    generated = result.output_ids[0, result.num_input_tokens :].tolist()
    return {
        "prompt_index": prompt_index,
        "input_tokens": int(result.num_input_tokens),
        "output_tokens": int(result.num_output_tokens),
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "ttft_seconds": float(result.time_to_first_token),
        "time_per_output_token_seconds": float(result.time_per_output_token),
        "committed_tokens_per_round": [int(x) for x in result.acceptance_lengths],
    }


def _summarize(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    committed = [value for row in rows for value in row["committed_tokens_per_round"]]
    return {
        "method": method,
        "request_count": len(rows),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "mean_ttft_seconds": mean(row["ttft_seconds"] for row in rows),
        "mean_time_per_output_token_seconds": mean(
            row["time_per_output_token_seconds"] for row in rows
        ),
        "mean_committed_tokens_per_round": mean(committed) if committed else 1.0,
        "round_count": len(committed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angelslim-root", type=Path, required=True)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--dflare-path", type=Path, required=True)
    parser.add_argument("--dflash-path", type=Path)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--dflare-label", required=True)
    parser.add_argument("--dflash-label")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--attention",
        choices=["sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    requested_dtype = _dtype(args.dtype)
    if requested_dtype == torch.bfloat16 and properties.major < 8:
        raise RuntimeError("bfloat16 requires compute capability 8.0 or newer")
    if args.attention == "flash_attention_2":
        try:
            importlib.import_module("flash_attn")
        except ImportError as error:
            raise RuntimeError(
                "flash_attention_2 was requested but flash-attn is not installed"
            ) from error

    root = args.angelslim_root.resolve()
    benchmark = _load_benchmark_module(root)
    tokenizer = AutoTokenizer.from_pretrained(args.target_path, local_files_only=True)
    target_load_started = time.perf_counter()
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target_path,
            attn_implementation=args.attention,
            dtype=requested_dtype,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    torch.cuda.synchronize()
    target_load_seconds = time.perf_counter() - target_load_started

    tokenized_prompts = []
    for prompt in DEFAULT_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenized_prompts.append(
            tokenizer.encode(rendered, return_tensors="pt").to(device)
        )

    method_results: dict[str, list[dict[str, Any]]] = {}
    model_specs = [("dflare", args.dflare_path, args.dflare_label)]
    if args.dflash_path is not None:
        model_specs.append(("dflash", args.dflash_path, args.dflash_label))

    ar_rows: list[dict[str, Any]] | None = None
    draft_metadata = []
    for architecture, draft_path, label in model_specs:
        if label is None:
            raise ValueError(f"a label is required for {architecture}")
        DraftModel, sample_fn, feature_fn = _load_draft_contract(root, architecture)
        load_started = time.perf_counter()
        draft = (
            DraftModel.from_pretrained(
                draft_path,
                attn_implementation=args.attention,
                dtype=requested_dtype,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        torch.cuda.synchronize()
        draft_metadata.append(
            {
                "method": architecture,
                "label": label,
                "load_seconds": time.perf_counter() - load_started,
                "block_size": int(draft.block_size),
                "target_layer_ids": list(draft.target_layer_ids),
            }
        )

        if ar_rows is None:
            ar_rows = []
            for index, input_ids in enumerate(tokenized_prompts):
                result = benchmark.dflash_generate(
                    model=draft,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=1,
                    stop_token_ids=[tokenizer.eos_token_id],
                    sample_fn=sample_fn,
                    extract_context_feature_fn=feature_fn,
                    temperature=0.0,
                )
                ar_rows.append(_request_row(result, tokenizer, index))
            method_results["ar"] = ar_rows

        rows = []
        for index, input_ids in enumerate(tokenized_prompts):
            result = benchmark.dflash_generate(
                model=draft,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=int(draft.block_size),
                stop_token_ids=[tokenizer.eos_token_id],
                sample_fn=sample_fn,
                extract_context_feature_fn=feature_fn,
                temperature=0.0,
            )
            rows.append(_request_row(result, tokenizer, index))
        method_results[architecture] = rows
        del draft
        gc.collect()
        torch.cuda.empty_cache()

    assert ar_rows is not None
    ar_tokens = [row["generated_token_ids"] for row in ar_rows]
    comparisons = {
        method: [
            row["generated_token_ids"] == ar_tokens[row["prompt_index"]] for row in rows
        ]
        for method, rows in method_results.items()
        if method != "ar"
    }
    summaries = [_summarize(method, rows) for method, rows in method_results.items()]
    ar_tpot = summaries[0]["mean_time_per_output_token_seconds"]
    for summary in summaries:
        summary["speedup_vs_ar"] = (
            ar_tpot / summary["mean_time_per_output_token_seconds"]
        )

    payload = {
        "schema_version": 1,
        "mode": "measured_gpu_reference_smoke",
        "benchmark_claim": False,
        "models": {
            "target": args.target_label,
            "drafts": draft_metadata,
        },
        "config": {
            "dtype": args.dtype,
            "attention": args.attention,
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "prompt_count": len(DEFAULT_PROMPTS),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_bytes": properties.total_memory,
        },
        "timing": {"target_load_seconds": target_load_seconds},
        "summaries": summaries,
        "all_outputs_match_ar": {
            method: all(matches) for method, matches in comparisons.items()
        },
        "requests": method_results,
        "limitations": [
            "Small fixed prompt set; not a statistically meaningful benchmark.",
            "Eager Python control flow favors transparency over peak serving speed.",
        ]
        + (
            ["SDPA favors portability over the peak speed of FlashAttention."]
            if args.attention == "sdpa"
            else []
        )
        + (
            [
                "Pre-Ampere FP16 results are not comparable to BF16 "
                "FlashAttention results."
            ]
            if properties.major < 8
            else []
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
