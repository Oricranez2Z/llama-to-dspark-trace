#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from llm_serving_lab.mini_serving import (
    BlockManager,
    DeterministicModelRunner,
    Engine,
    Request,
    Scheduler,
    SchedulerConfig,
)
from llm_serving_lab.tracing import TraceRecorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    recorder = TraceRecorder(args.output)
    block_manager = BlockManager(
        block_size=config["block_size"],
        num_blocks=config["num_blocks"],
    )
    scheduler = Scheduler(
        block_manager,
        SchedulerConfig(
            max_num_batched_tokens=config["max_num_batched_tokens"],
            max_num_seqs=config["max_num_seqs"],
            chunked_prefill=config["chunked_prefill"],
        ),
        recorder,
    )
    engine = Engine(
        scheduler,
        DeterministicModelRunner(vocab_size=config["vocab_size"]),
        recorder,
    )
    for item in config["requests"]:
        engine.add_request(Request(**item))
    outputs = engine.run()
    path = recorder.write()
    print(f"trace={path}")
    print(f"engine_steps={len(outputs)}")
    for request in scheduler.finished:
        print(
            f"{request.request_id}: prompt={request.prompt_length}, "
            f"output={request.output_token_ids}"
        )


if __name__ == "__main__":
    main()
