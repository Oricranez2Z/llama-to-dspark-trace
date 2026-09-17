#!/usr/bin/env python3
"""Create a vLLM-compatible view of a DeepSpec checkpoint without copying weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def adapt_eagle3_config(config: dict[str, Any]) -> dict[str, Any]:
    """Map the released DeepSpec Qwen3 EAGLE3 config to vLLM field names."""
    architectures = config.get("architectures") or []
    if architectures not in (
        ["Qwen3Eagle3Model"],
        ["Eagle3Qwen3ForCausalLM"],
    ):
        raise ValueError(f"unsupported EAGLE3 architecture: {architectures}")
    target_layers = config.get("target_layer_ids")
    if not isinstance(target_layers, list) or not target_layers:
        raise ValueError("EAGLE3 config must contain target_layer_ids")

    adapted = dict(config)
    adapted["architectures"] = ["Eagle3Qwen3ForCausalLM"]
    adapted["eagle_aux_hidden_state_layer_ids"] = [
        int(layer) + 1 for layer in target_layers
    ]
    adapted["num_aux_layers"] = len(target_layers)
    return adapted


def prepare_checkpoint(source: Path, destination: Path) -> None:
    source = source.resolve()
    if source == destination.resolve():
        raise ValueError("source and destination must differ")
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination is not empty: {destination}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    adapted = adapt_eagle3_config(config)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.name == "config.json":
            continue
        (destination / entry.name).symlink_to(
            entry.resolve(), target_is_directory=entry.is_dir()
        )
    (destination / "config.json").write_text(
        json.dumps(adapted, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_checkpoint(args.source, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
