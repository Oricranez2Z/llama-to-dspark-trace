# Token generation and KV cache

## Execution path

The Mini Llama implementation separates the first model call from subsequent
decode calls:

```text
prefill: prompt[N] → logits[N, vocab] → sample token N
decode:  token[1]  + cached K/V[N] → logits[1, vocab] → sample token N+1
```

The first call processes every prompt token. Each later call processes only the
newest sampled token while attending to cached keys and values.

## Tensor shapes

For batch `B`, sequence length `S`, query heads `H`, KV heads `Hkv`, and head
dimension `D`:

```text
hidden states: [B, S, H*D]
queries:       [B, H,   S, D]
keys/values:   [B, Hkv, S, D]
scores:        [B, H, query_length, total_kv_length]
```

GQA repeats each KV head across a group of query heads for attention, but the
cache stores only `Hkv` heads.

## Cache invariant

After generating `M` tokens from an `N`-token prompt, the cache contains
`N + M - 1` positions. The final emitted token has not yet been processed by a
forward pass.

The test `test_cached_logits_match_full_forward` verifies that incremental
cached execution matches a full causal forward pass at every position.

## Relevant code

- `src/llm_serving_lab/mini_llm/model.py`
- `src/llm_serving_lab/mini_llm/cache.py`
- `src/llm_serving_lab/mini_llm/generate.py`
- `tests/test_mini_llm.py`
