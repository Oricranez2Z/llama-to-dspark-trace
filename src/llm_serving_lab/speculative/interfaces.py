from typing import Protocol


class TargetModel(Protocol):
    def next_token(self, prefix: list[int]) -> int:
        """Return the target model's greedy next token."""


class DraftProposer(Protocol):
    def propose(self, prefix: list[int], max_tokens: int) -> list[int]:
        """Return up to ``max_tokens`` speculative tokens."""
