import argparse
import json

from .mini_llm import MiniLlama, generate
from .mini_serving import (
    BlockManager,
    DeterministicModelRunner,
    Engine,
    Request,
    Scheduler,
    SchedulerConfig,
)
from .speculative import DeterministicTarget, NoisyProposer, SpeculativeDecoder
from .speculative.decoder import autoregressive_decode


def _model_demo() -> None:
    model = MiniLlama()
    result = generate(model, [1, 7, 11], max_new_tokens=4)
    print(
        json.dumps(
            {
                "tokens": result.token_ids,
                "steps": [step.__dict__ for step in result.steps],
            },
            indent=2,
        )
    )


def _serving_demo() -> None:
    blocks = BlockManager(block_size=4, num_blocks=32)
    scheduler = Scheduler(
        blocks,
        SchedulerConfig(max_num_batched_tokens=8, max_num_seqs=3),
    )
    engine = Engine(scheduler, DeterministicModelRunner())
    engine.add_request(Request("short", [1, 2, 3], 3))
    engine.add_request(Request("long", list(range(1, 14)), 2))
    engine.run()
    payload = {
        request.request_id: request.output_token_ids for request in scheduler.finished
    }
    print(json.dumps(payload, indent=2))


def _speculative_demo() -> None:
    target = DeterministicTarget()
    proposer = NoisyProposer(target, error_every=4)
    result = SpeculativeDecoder(target, proposer, draft_length=5).decode(
        [1, 4, 9],
        max_new_tokens=12,
    )
    baseline = autoregressive_decode(target, [1, 4, 9], max_new_tokens=12)
    print(
        json.dumps(
            {
                "matches_autoregressive": result.token_ids == baseline,
                "generated": result.generated_token_ids,
                "rounds": len(result.rounds),
                "mean_accepted_length": result.mean_accepted_length,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "demo",
        choices=("model", "serving", "speculative"),
        nargs="?",
        default="serving",
    )
    args = parser.parse_args()
    {"model": _model_demo, "serving": _serving_demo, "speculative": _speculative_demo}[
        args.demo
    ]()


if __name__ == "__main__":
    main()
