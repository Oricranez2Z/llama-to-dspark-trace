from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def softmax(values: FloatArray, *, axis: int = -1) -> FloatArray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def rms_norm(values: FloatArray, *, eps: float = 1e-6) -> FloatArray:
    scale = np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True) + eps)
    return values / scale


class LayerwiseFusion:
    """Fuse T target layers independently for each of D draft layers.

    Input follows the unbatched form of the reference DFlare contract:
    ``[sequence, target_layer, hidden]``. Output is
    ``[sequence, draft_layer, hidden]``.
    """

    def __init__(self, logits: FloatArray, *, eps: float = 1e-6):
        if logits.ndim != 2:
            raise ValueError(
                "fusion logits must have shape [draft_layer, target_layer]"
            )
        if 0 in logits.shape:
            raise ValueError("fusion logits cannot have an empty dimension")
        self.logits = np.asarray(logits, dtype=np.float64).copy()
        self.eps = eps

    @property
    def weights(self) -> FloatArray:
        return softmax(self.logits, axis=1)

    def fuse(self, target_hidden: FloatArray) -> FloatArray:
        values = np.asarray(target_hidden, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError(
                "target_hidden must have shape [sequence, target_layer, hidden]"
            )
        if values.shape[1] != self.logits.shape[1]:
            raise ValueError(
                "target_hidden target-layer dimension does not match fusion logits"
            )
        fused = np.einsum("sth,dt->sdh", values, self.weights)
        return rms_norm(fused, eps=self.eps)


class SharedFusion(LayerwiseFusion):
    """DFlash-like control where every draft layer receives one fusion."""

    def __init__(
        self,
        logits: FloatArray,
        *,
        num_draft_layers: int,
        eps: float = 1e-6,
    ):
        shared = np.asarray(logits, dtype=np.float64)
        if shared.ndim != 1:
            raise ValueError("shared fusion logits must have shape [target_layer]")
        if num_draft_layers <= 0:
            raise ValueError("num_draft_layers must be positive")
        super().__init__(np.repeat(shared[None, :], num_draft_layers, axis=0), eps=eps)
