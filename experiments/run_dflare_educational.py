#!/usr/bin/env python3
"""Compare autoregressive, shared-fusion DFlash, and DFlare mechanisms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_serving_lab.dflare import (
    DeterministicTargetFeatures,
    DFlareConfig,
    DFlareProposer,
    DFlashProposer,
)
from llm_serving_lab.speculative import DeterministicTarget, SpeculativeDecoder
from llm_serving_lab.speculative.decoder import autoregressive_decode
from llm_serving_lab.tracing import TraceRecorder


def run_variant(name, proposer, target, prompt, output_tokens, recorder):
    result = SpeculativeDecoder(
        target,
        proposer,
        draft_length=proposer.config.block_size,
        recorder=recorder,
    ).decode(prompt, max_new_tokens=output_tokens)
    return {
        "method": name,
        "matches_autoregressive": result.token_ids
        == autoregressive_decode(target, prompt, max_new_tokens=output_tokens),
        "verification_rounds": result.target_forward_calls,
        "proposed_tokens": result.proposed_tokens,
        "accepted_tokens": result.accepted_tokens,
        "mean_accepted_length": result.mean_accepted_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    config = DFlareConfig(vocab_size=32, hidden_size=16, block_size=6)
    features = DeterministicTargetFeatures(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        total_target_layers=max(config.target_layer_ids) + 1,
        seed=9,
    )
    target = DeterministicTarget(vocab_size=config.vocab_size)
    prompt = [1, 5, 9]
    output_tokens = 24
    rows = []
    for name, proposer_type in (
        ("dflash_shared_fusion", DFlashProposer),
        ("dflare_layerwise_fusion", DFlareProposer),
    ):
        recorder = TraceRecorder()
        proposer = proposer_type(config, features, recorder=recorder)
        rows.append(
            run_variant(name, proposer, target, prompt, output_tokens, recorder)
        )
        if name == "dflare_layerwise_fusion":
            recorder.write(args.trace)

    payload = {
        "mode": "educational_algorithmic_simulation",
        "benchmark_claim": False,
        "baseline_target_steps": output_tokens,
        "config": {
            "block_size": config.block_size,
            "target_layers": list(config.target_layer_ids),
            "draft_layers": config.num_draft_layers,
        },
        "results": rows,
        "disclaimer": (
            "Random teaching weights do not model checkpoint quality or GPU speed. "
            "Only the execution contract and lossless verification are evaluated."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
