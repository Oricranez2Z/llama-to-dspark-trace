"""Example: attach the lab's trace adapter to a local vLLM checkout.

Run this from an environment in which the pinned vLLM source tree is installed.
The model name is intentionally supplied by the user; no checkpoint is bundled.
"""

import argparse
from pathlib import Path

from llm_serving_lab.tracing.vllm_adapter import VLLMTraceAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("vllm_trace.jsonl"))
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    with VLLMTraceAdapter(args.output):
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            enforce_eager=True,
            max_model_len=256,
        )
        llm.generate(
            ["Explain KV cache in one sentence."],
            SamplingParams(temperature=0, max_tokens=4),
        )
    print(args.output)


if __name__ == "__main__":
    main()
