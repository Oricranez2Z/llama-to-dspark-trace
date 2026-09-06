import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def sample_token(
    logits: np.ndarray,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token from a one-dimensional logit vector."""

    if logits.ndim != 1:
        raise ValueError("sample_token expects a one-dimensional logit vector")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return int(np.argmax(logits))

    generator = rng or np.random.default_rng()
    probabilities = softmax(logits.astype(np.float64) / temperature)
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probabilities = probabilities[sorted_indices]
    cutoff = int(np.searchsorted(np.cumsum(sorted_probabilities), top_p)) + 1
    candidate_indices = sorted_indices[:cutoff]
    candidate_probabilities = sorted_probabilities[:cutoff]
    candidate_probabilities /= candidate_probabilities.sum()
    return int(generator.choice(candidate_indices, p=candidate_probabilities))
