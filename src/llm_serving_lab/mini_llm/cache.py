from dataclasses import dataclass, field

import numpy as np


@dataclass
class KVCache:
    """Per-layer key/value tensors with shape ``[batch, heads, seq, dim]``."""

    keys: dict[int, np.ndarray] = field(default_factory=dict)
    values: dict[int, np.ndarray] = field(default_factory=dict)

    def past_length(self, layer_idx: int = 0) -> int:
        key = self.keys.get(layer_idx)
        return 0 if key is None else int(key.shape[2])

    @property
    def sequence_length(self) -> int:
        return self.past_length(0)

    def update(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if key.shape != value.shape:
            raise ValueError("key and value tensors must have the same shape")
        if key.ndim != 4:
            raise ValueError("cache tensors must have shape [batch, heads, seq, dim]")

        old_key = self.keys.get(layer_idx)
        old_value = self.values.get(layer_idx)
        if old_key is not None:
            if old_key.shape[:2] != key.shape[:2] or old_key.shape[3] != key.shape[3]:
                raise ValueError(
                    "new cache tensors are incompatible with cached tensors"
                )
            key = np.concatenate((old_key, key), axis=2)
            value = np.concatenate((old_value, value), axis=2)

        self.keys[layer_idx] = key
        self.values[layer_idx] = value
        return key, value

    def clone(self) -> "KVCache":
        return KVCache(
            keys={idx: tensor.copy() for idx, tensor in self.keys.items()},
            values={idx: tensor.copy() for idx, tensor in self.values.items()},
        )

    def memory_bytes(self) -> int:
        return sum(t.nbytes for t in (*self.keys.values(), *self.values.values()))
