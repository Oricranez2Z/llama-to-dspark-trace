from dataclasses import dataclass

import numpy as np

from .cache import KVCache
from .model import MiniLlama
from .sampling import sample_token


@dataclass(frozen=True)
class GenerationStep:
    step: int
    phase: str
    input_shape: tuple[int, ...]
    cache_length_before: int
    cache_length_after: int
    sampled_token: int


@dataclass(frozen=True)
class GenerationResult:
    token_ids: list[int]
    generated_token_ids: list[int]
    steps: list[GenerationStep]
    cache: KVCache | None


def generate(
    model: MiniLlama,
    prompt_token_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    use_cache: bool = True,
    seed: int = 0,
) -> GenerationResult:
    """Generate tokens and retain an execution-level trace for every step."""

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids cannot be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if max_new_tokens == 0:
        return GenerationResult(prompt_token_ids.copy(), [], [], None)

    rng = np.random.default_rng(seed)
    all_tokens = prompt_token_ids.copy()
    generated: list[int] = []
    steps: list[GenerationStep] = []
    cache: KVCache | None = KVCache() if use_cache else None

    for step in range(max_new_tokens):
        phase = "prefill" if step == 0 else "decode"
        if use_cache:
            model_input = all_tokens if step == 0 else [all_tokens[-1]]
            before = cache.sequence_length if cache is not None else 0
        else:
            model_input = all_tokens
            before = 0
        input_array = np.asarray([model_input], dtype=np.int64)
        output = model.forward(input_array, cache=cache, use_cache=use_cache)
        cache = output.cache
        next_token = sample_token(
            output.logits[0, -1],
            temperature=temperature,
            top_p=top_p,
            rng=rng,
        )
        all_tokens.append(next_token)
        generated.append(next_token)
        after = cache.sequence_length if cache is not None else 0
        steps.append(
            GenerationStep(
                step=step,
                phase=phase,
                input_shape=tuple(input_array.shape),
                cache_length_before=before,
                cache_length_after=after,
                sampled_token=next_token,
            )
        )

    return GenerationResult(all_tokens, generated, steps, cache)
