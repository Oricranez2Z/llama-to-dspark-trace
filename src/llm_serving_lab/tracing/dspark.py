from dataclasses import dataclass
from typing import Any

from .recorder import TraceRecorder


@dataclass(frozen=True)
class DSparkRound:
    step: int
    request_id: str
    draft_tokens: list[int]
    confidences: list[float]
    verification_length: int
    accepted_count: int
    rejected_count: int


class DSparkTraceRecorder:
    """Normalize DSpark round data from reference or serving implementations."""

    def __init__(self, recorder: TraceRecorder | None = None):
        self.recorder = recorder or TraceRecorder()

    def record(self, round_data: DSparkRound, **extra: Any) -> None:
        self.recorder.emit(
            "dspark_round",
            step=round_data.step,
            request_id=round_data.request_id,
            draft_tokens=round_data.draft_tokens,
            confidences=round_data.confidences,
            verification_length=round_data.verification_length,
            accepted_count=round_data.accepted_count,
            rejected_count=round_data.rejected_count,
            **extra,
        )
