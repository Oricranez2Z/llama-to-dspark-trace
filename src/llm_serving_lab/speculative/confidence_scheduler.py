from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class LoadProfile:
    """A simple hardware-profile proxy for load-aware verification."""

    low_load_max_tokens: int = 8
    high_load_max_tokens: int = 3
    high_load_concurrency: int = 16
    base_survival_threshold: float = 0.20
    high_load_survival_threshold: float = 0.55

    def __post_init__(self) -> None:
        if self.low_load_max_tokens <= 0 or self.high_load_max_tokens <= 0:
            raise ValueError("verification budgets must be positive")
        if self.high_load_concurrency <= 1:
            raise ValueError("high_load_concurrency must be greater than one")


@dataclass(frozen=True)
class ConfidenceSchedule:
    verification_length: int
    prefix_survival: list[float]
    survival_threshold: float
    token_budget: int
    concurrency: int


class ConfidenceScheduler:
    """Choose a verification prefix from confidence and current serving load.

    This is an educational scheduler inspired by DSpark. It is not claimed to
    reproduce a production engine's hardware throughput profile.
    """

    def __init__(self, profile: LoadProfile | None = None):
        self.profile = profile or LoadProfile()

    def _load_fraction(self, concurrency: int) -> float:
        if concurrency <= 1:
            return 0.0
        return min(
            1.0,
            (concurrency - 1) / (self.profile.high_load_concurrency - 1),
        )

    def schedule(
        self,
        confidences: list[float],
        *,
        concurrency: int,
    ) -> ConfidenceSchedule:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if any(value < 0 or value > 1 for value in confidences):
            raise ValueError("confidence values must be in [0, 1]")
        fraction = self._load_fraction(concurrency)
        budget = round(
            self.profile.low_load_max_tokens
            + fraction
            * (self.profile.high_load_max_tokens - self.profile.low_load_max_tokens)
        )
        threshold = self.profile.base_survival_threshold + fraction * (
            self.profile.high_load_survival_threshold
            - self.profile.base_survival_threshold
        )
        survival = [prod(confidences[: index + 1]) for index in range(len(confidences))]
        verification_length = 0
        for index, probability in enumerate(survival[:budget], start=1):
            if probability < threshold:
                break
            verification_length = index
        return ConfidenceSchedule(
            verification_length=verification_length,
            prefix_survival=survival,
            survival_threshold=threshold,
            token_budget=budget,
            concurrency=concurrency,
        )
