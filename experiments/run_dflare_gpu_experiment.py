#!/usr/bin/env python3
"""Launch the pinned AngelSlim DFlare smoke test in an external GPU env."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE = PROJECT_ROOT / "integrations" / "angelslim" / "dflare_smoke.py"


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
    return ",".join(sorted(set(result.stdout.split()))) or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--angelslim-root", type=Path, required=True)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--dflare-path", type=Path, required=True)
    parser.add_argument("--dflash-path", type=Path)
    parser.add_argument("--target-label", default="Qwen/Qwen3-4B")
    parser.add_argument("--dflare-label", default="AngelSlim/Qwen3-4b-dflare")
    parser.add_argument("--dflash-label", default="z-lab/Qwen3-4B-DFlash")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--attention",
        choices=["sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    required = [
        args.python,
        args.angelslim_root / "tools" / "dflash_benchmark.py",
        args.target_path,
        args.dflare_path,
    ]
    if args.dflash_path is not None:
        required.append(args.dflash_path)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "result.json"
    command = [
        str(args.python.resolve()),
        str(ENGINE),
        "--angelslim-root",
        str(args.angelslim_root.resolve()),
        "--target-path",
        str(args.target_path.resolve()),
        "--dflare-path",
        str(args.dflare_path.resolve()),
        "--target-label",
        args.target_label,
        "--dflare-label",
        args.dflare_label,
        "--dtype",
        args.dtype,
        "--attention",
        args.attention,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output",
        str(output),
    ]
    if args.dflash_path is not None:
        command.extend(
            [
                "--dflash-path",
                str(args.dflash_path.resolve()),
                "--dflash-label",
                args.dflash_label,
            ]
        )
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)

    result = json.loads(output.read_text(encoding="utf-8"))
    result["environment"]["physical_gpu_index"] = args.gpu
    result["environment"]["driver"] = _driver_version()
    result["provenance"] = {
        "angelslim": _git_state(args.angelslim_root.resolve()),
        "runner": str(ENGINE.relative_to(PROJECT_ROOT)),
    }
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
