from dataclasses import dataclass

from .interfaces import TargetModel


@dataclass(frozen=True)
class GreedyVerification:
    draft_tokens: list[int]
    target_tokens: list[int]
    accepted_count: int
    emitted_tokens: list[int]
    rejected_count: int
    used_bonus_token: bool


class GreedyVerifier:
    """Losslessly verify a draft block against greedy target decoding."""

    def __init__(self, target: TargetModel):
        self.target = target

    def verify(self, prefix: list[int], draft_tokens: list[int]) -> GreedyVerification:
        if not draft_tokens:
            token = self.target.next_token(prefix)
            return GreedyVerification([], [token], 0, [token], 0, False)

        working = prefix.copy()
        target_tokens: list[int] = []
        emitted: list[int] = []
        accepted = 0
        used_bonus = False

        for draft_token in draft_tokens:
            target_token = self.target.next_token(working)
            target_tokens.append(target_token)
            if target_token != draft_token:
                emitted.append(target_token)
                break
            emitted.append(draft_token)
            working.append(draft_token)
            accepted += 1
        else:
            bonus = self.target.next_token(working)
            target_tokens.append(bonus)
            emitted.append(bonus)
            used_bonus = True

        return GreedyVerification(
            draft_tokens=draft_tokens.copy(),
            target_tokens=target_tokens,
            accepted_count=accepted,
            emitted_tokens=emitted,
            rejected_count=len(draft_tokens) - accepted,
            used_bonus_token=used_bonus,
        )
