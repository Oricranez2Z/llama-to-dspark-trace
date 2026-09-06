from dataclasses import dataclass


@dataclass(frozen=True)
class MiniLlamaConfig:
    """Configuration for the educational Llama-style decoder."""

    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 256
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    seed: int = 7

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_hidden_size(self) -> int:
        return self.num_key_value_heads * self.head_dim
