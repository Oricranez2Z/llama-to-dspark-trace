from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class TargetFeatureProvider(Protocol):
    def hidden_states(
        self,
        prefix: list[int],
        target_layer_ids: tuple[int, ...],
    ) -> FloatArray:
        """Return ``[sequence, target_layer, hidden]`` target features."""


class DeterministicTargetFeatures:
    """Synthetic causal hidden states used by the CPU teaching model."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        total_target_layers: int,
        seed: int,
    ):
        if min(vocab_size, hidden_size, total_target_layers) <= 0:
            raise ValueError("feature-provider dimensions must be positive")
        rng = np.random.default_rng(seed)
        scale = hidden_size**-0.5
        self.token_embeddings = rng.normal(0.0, scale, size=(vocab_size, hidden_size))
        self.layer_kernels = rng.normal(
            0.0,
            scale,
            size=(total_target_layers, hidden_size, hidden_size),
        )
        self.layer_biases = rng.normal(
            0.0, scale, size=(total_target_layers, hidden_size)
        )
        self.total_target_layers = total_target_layers

    def hidden_states(
        self,
        prefix: list[int],
        target_layer_ids: tuple[int, ...],
    ) -> FloatArray:
        if not prefix:
            raise ValueError("prefix cannot be empty")
        if any(token < 0 or token >= len(self.token_embeddings) for token in prefix):
            raise ValueError("prefix contains an out-of-range token ID")
        if any(
            layer < 0 or layer >= self.total_target_layers for layer in target_layer_ids
        ):
            raise ValueError("target_layer_ids contains an out-of-range layer")

        embedded = self.token_embeddings[np.asarray(prefix)]
        positions = np.arange(1, len(prefix) + 1, dtype=np.float64)[:, None]
        causal_summary = np.cumsum(embedded, axis=0) / positions
        states = [
            np.tanh(
                causal_summary @ self.layer_kernels[layer_id]
                + self.layer_biases[layer_id]
            )
            for layer_id in target_layer_ids
        ]
        return np.stack(states, axis=1)
