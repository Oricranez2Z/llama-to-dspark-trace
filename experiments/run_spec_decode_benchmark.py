#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from llm_serving_lab.speculative import (
    ConfidenceScheduler,
    DeterministicTarget,
    NoisyProposer,
    SpeculativeDecoder,
)
from llm_serving_lab.speculative.decoder import autoregressive_decode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    target = DeterministicTarget()
    prompt = config["prompt_token_ids"]
    max_new = config["max_new_tokens"]
    baseline = autoregressive_decode(target, prompt, max_new_tokens=max_new)
    results: list[dict[str, object]] = []

    for draft_length in config["draft_lengths"]:
        proposer = NoisyProposer(
            target,
            error_every=config["error_every"],
        )
        decoded = SpeculativeDecoder(
            target,
            proposer,
            draft_length=draft_length,
        ).decode(prompt, max_new_tokens=max_new)
        results.append(
            {
                "method": "fixed",
                "draft_length": draft_length,
                "matches_autoregressive": decoded.token_ids == baseline,
                "target_forward_calls": decoded.target_forward_calls,
                "proposed_tokens": decoded.proposed_tokens,
                "accepted_tokens": decoded.accepted_tokens,
                "mean_accepted_length": decoded.mean_accepted_length,
            }
        )

    for concurrency in config["concurrency_levels"]:
        proposer = NoisyProposer(target, error_every=config["error_every"])
        decoded = SpeculativeDecoder(
            target,
            proposer,
            draft_length=max(config["draft_lengths"]),
            confidence_scheduler=ConfidenceScheduler(),
        ).decode(
            prompt,
            max_new_tokens=max_new,
            confidences=config["confidences"],
            concurrency=concurrency,
        )
        results.append(
            {
                "method": "confidence",
                "concurrency": concurrency,
                "matches_autoregressive": decoded.token_ids == baseline,
                "target_forward_calls": decoded.target_forward_calls,
                "proposed_tokens": decoded.proposed_tokens,
                "accepted_tokens": decoded.accepted_tokens,
                "mean_accepted_length": decoded.mean_accepted_length,
            }
        )

    payload = {
        "disclaimer": (
            "Algorithmic simulation only; target_forward_calls are conceptual "
            "verification rounds, not measured GPU kernel launches."
        ),
        "baseline_target_steps": max_new,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
