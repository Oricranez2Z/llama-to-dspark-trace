import numpy as np

from llm_serving_lab.mini_llm import KVCache, MiniLlama, MiniLlamaConfig, generate


def small_model() -> MiniLlama:
    return MiniLlama(
        MiniLlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )


def test_cached_logits_match_full_forward() -> None:
    model = small_model()
    tokens = np.asarray([[1, 7, 3, 12, 9]], dtype=np.int64)
    full = model.forward(tokens, use_cache=False).logits
    cache = KVCache()
    incremental = []
    for position in range(tokens.shape[1]):
        output = model.forward(
            tokens[:, position : position + 1],
            cache=cache,
            use_cache=True,
        )
        incremental.append(output.logits)
    cached = np.concatenate(incremental, axis=1)
    np.testing.assert_allclose(cached, full, rtol=2e-5, atol=2e-5)


def test_cached_and_uncached_generation_match() -> None:
    model = small_model()
    cached = generate(model, [1, 5, 9], max_new_tokens=6, use_cache=True)
    uncached = generate(model, [1, 5, 9], max_new_tokens=6, use_cache=False)
    assert cached.token_ids == uncached.token_ids
    assert cached.steps[0].phase == "prefill"
    assert all(step.input_shape == (1, 1) for step in cached.steps[1:])
    assert cached.cache is not None
    assert cached.cache.sequence_length == len(cached.token_ids) - 1


def test_gqa_reduces_cache_heads() -> None:
    model = small_model()
    output = model.forward(np.asarray([[1, 2, 3]]), use_cache=True)
    assert output.cache is not None
    assert output.cache.keys[0].shape == (1, 2, 3, 4)
