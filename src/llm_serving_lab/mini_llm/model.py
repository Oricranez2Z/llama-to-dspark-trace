from dataclasses import dataclass

import numpy as np

from .cache import KVCache
from .config import MiniLlamaConfig


@dataclass(frozen=True)
class ModelOutput:
    logits: np.ndarray
    cache: KVCache | None


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(variance + eps)) * weight


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _apply_rope(
    tensor: np.ndarray,
    positions: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Apply rotary embeddings to ``[batch, heads, seq, head_dim]``."""

    head_dim = tensor.shape[-1]
    inverse_frequency = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    angles = np.outer(positions.astype(np.float32), inverse_frequency)
    cos = np.cos(angles)[None, None, :, :]
    sin = np.sin(angles)[None, None, :, :]
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    rotated = np.empty_like(tensor)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


@dataclass
class _LayerWeights:
    attention_norm: np.ndarray
    ffn_norm: np.ndarray
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray


class MiniLlama:
    """A deliberately small, readable Llama-style decoder implemented in NumPy."""

    def __init__(self, config: MiniLlamaConfig | None = None):
        self.config = config or MiniLlamaConfig()
        rng = np.random.default_rng(self.config.seed)
        scale = 1.0 / np.sqrt(self.config.hidden_size)
        self.embedding = rng.normal(
            0.0,
            scale,
            (self.config.vocab_size, self.config.hidden_size),
        ).astype(np.float32)
        self.layers = [
            self._make_layer(rng, scale) for _ in range(self.config.num_hidden_layers)
        ]
        self.final_norm = np.ones(self.config.hidden_size, dtype=np.float32)

    def _make_layer(
        self,
        rng: np.random.Generator,
        scale: float,
    ) -> _LayerWeights:
        cfg = self.config

        def weight(rows: int, columns: int) -> np.ndarray:
            return rng.normal(0.0, scale, (rows, columns)).astype(np.float32)

        return _LayerWeights(
            attention_norm=np.ones(cfg.hidden_size, dtype=np.float32),
            ffn_norm=np.ones(cfg.hidden_size, dtype=np.float32),
            wq=weight(cfg.hidden_size, cfg.hidden_size),
            wk=weight(cfg.hidden_size, cfg.kv_hidden_size),
            wv=weight(cfg.hidden_size, cfg.kv_hidden_size),
            wo=weight(cfg.hidden_size, cfg.hidden_size),
            w1=weight(cfg.hidden_size, cfg.intermediate_size),
            w2=weight(cfg.intermediate_size, cfg.hidden_size),
            w3=weight(cfg.hidden_size, cfg.intermediate_size),
        )

    def _attention(
        self,
        hidden_states: np.ndarray,
        layer: _LayerWeights,
        layer_idx: int,
        cache: KVCache | None,
    ) -> np.ndarray:
        cfg = self.config
        batch_size, query_length, _ = hidden_states.shape
        past_length = 0 if cache is None else cache.past_length(layer_idx)
        positions = np.arange(
            past_length,
            past_length + query_length,
            dtype=np.int64,
        )

        queries = hidden_states @ layer.wq
        keys = hidden_states @ layer.wk
        values = hidden_states @ layer.wv
        queries = queries.reshape(
            batch_size,
            query_length,
            cfg.num_attention_heads,
            cfg.head_dim,
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size,
            query_length,
            cfg.num_key_value_heads,
            cfg.head_dim,
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size,
            query_length,
            cfg.num_key_value_heads,
            cfg.head_dim,
        ).transpose(0, 2, 1, 3)

        queries = _apply_rope(queries, positions, cfg.rope_theta)
        keys = _apply_rope(keys, positions, cfg.rope_theta)
        if cache is not None:
            keys, values = cache.update(layer_idx, keys, values)

        repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        repeated_keys = np.repeat(keys, repeats, axis=1)
        repeated_values = np.repeat(values, repeats, axis=1)
        scores = queries @ repeated_keys.transpose(0, 1, 3, 2)
        scores = scores / np.sqrt(cfg.head_dim)

        key_positions = np.arange(repeated_keys.shape[2])[None, :]
        query_positions = positions[:, None]
        causal_mask = key_positions > query_positions
        scores = np.where(causal_mask[None, None, :, :], -1e30, scores)
        probabilities = _softmax(scores)
        context = probabilities @ repeated_values
        context = context.transpose(0, 2, 1, 3).reshape(
            batch_size,
            query_length,
            cfg.hidden_size,
        )
        return context @ layer.wo

    def forward(
        self,
        input_ids: np.ndarray,
        *,
        cache: KVCache | None = None,
        use_cache: bool = True,
    ) -> ModelOutput:
        """Run a causal forward pass over ``input_ids`` with shape ``[batch, seq]``."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] == 0:
            raise ValueError("input sequence cannot be empty")
        if np.any(input_ids < 0) or np.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contain a token outside the vocabulary")

        active_cache = cache
        if use_cache and active_cache is None:
            active_cache = KVCache()
        if not use_cache:
            active_cache = None
        if active_cache is not None:
            new_length = active_cache.sequence_length + input_ids.shape[1]
            if new_length > self.config.max_position_embeddings:
                raise ValueError("input exceeds max_position_embeddings")

        hidden_states = self.embedding[input_ids]
        for layer_idx, layer in enumerate(self.layers):
            residual = hidden_states
            normalized = _rms_norm(
                hidden_states,
                layer.attention_norm,
                self.config.rms_norm_eps,
            )
            hidden_states = residual + self._attention(
                normalized,
                layer,
                layer_idx,
                active_cache,
            )
            residual = hidden_states
            normalized = _rms_norm(
                hidden_states,
                layer.ffn_norm,
                self.config.rms_norm_eps,
            )
            hidden_states = (
                residual
                + (_silu(normalized @ layer.w1) * (normalized @ layer.w3)) @ layer.w2
            )

        hidden_states = _rms_norm(
            hidden_states,
            self.final_norm,
            self.config.rms_norm_eps,
        )
        logits = hidden_states @ self.embedding.T
        return ModelOutput(logits=logits, cache=active_cache)
