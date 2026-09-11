from dataclasses import dataclass


@dataclass(frozen=True)
class DFlareConfig:
    """Small dimensions for demonstrating DFlare without model weights."""

    vocab_size: int = 64
    hidden_size: int = 24
    num_target_layers: int = 5
    num_draft_layers: int = 3
    block_size: int = 6
    target_layer_ids: tuple[int, ...] = (0, 1, 2, 3, 4)
    rms_norm_eps: float = 1e-6
    seed: int = 17

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_target_layers": self.num_target_layers,
            "num_draft_layers": self.num_draft_layers,
            "block_size": self.block_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.vocab_size < 3:
            raise ValueError("vocab_size must leave room for reserved token IDs")
        if len(self.target_layer_ids) != self.num_target_layers:
            raise ValueError(
                "target_layer_ids must contain exactly num_target_layers entries"
            )
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError("target_layer_ids must be unique")
        if any(layer_id < 0 for layer_id in self.target_layer_ids):
            raise ValueError("target_layer_ids cannot contain negative values")
