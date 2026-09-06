import json

from llm_serving_lab.mini_llm import MiniLlama, generate


def main() -> None:
    model = MiniLlama()
    cached = generate(model, [1, 7, 11, 3], max_new_tokens=5, use_cache=True)
    uncached = generate(model, [1, 7, 11, 3], max_new_tokens=5, use_cache=False)
    print(
        json.dumps(
            {
                "cached_tokens": cached.generated_token_ids,
                "uncached_tokens": uncached.generated_token_ids,
                "match": cached.token_ids == uncached.token_ids,
                "trace": [step.__dict__ for step in cached.steps],
                "cache_bytes": cached.cache.memory_bytes() if cached.cache else 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
