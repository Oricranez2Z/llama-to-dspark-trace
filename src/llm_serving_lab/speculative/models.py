from dataclasses import dataclass

from .interfaces import TargetModel


@dataclass(frozen=True)
class DeterministicTarget:
    """A deterministic causal target used to demonstrate lossless verification."""

    vocab_size: int = 128

    def next_token(self, prefix: list[int]) -> int:
        if not prefix:
            raise ValueError("prefix cannot be empty")
        state = 23
        for position, token_id in enumerate(prefix):
            state = (state * 65_537 + token_id * 31 + position * 7) % 2**32
        return 2 + state % (self.vocab_size - 2)


class PerfectProposer:
    """A draft model that exactly matches the target model."""

    def __init__(self, target: TargetModel):
        self.target = target

    def propose(self, prefix: list[int], max_tokens: int) -> list[int]:
        working = prefix.copy()
        output: list[int] = []
        for _ in range(max_tokens):
            token = self.target.next_token(working)
            output.append(token)
            working.append(token)
        return output


class NoisyProposer:
    """Match most target tokens and inject deterministic errors for testing."""

    def __init__(
        self,
        target: TargetModel,
        *,
        error_every: int = 3,
        vocab_size: int = 128,
    ):
        if error_every <= 0:
            raise ValueError("error_every must be positive")
        self.target = target
        self.error_every = error_every
        self.vocab_size = vocab_size
        self._proposal_count = 0

    def propose(self, prefix: list[int], max_tokens: int) -> list[int]:
        working = prefix.copy()
        output: list[int] = []
        for _ in range(max_tokens):
            token = self.target.next_token(working)
            self._proposal_count += 1
            if self._proposal_count % self.error_every == 0:
                token = (token + 1) % self.vocab_size
                if token < 2:
                    token += 2
            output.append(token)
            working.append(token)
        return output
