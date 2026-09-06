from dataclasses import dataclass

from .scheduler import ScheduledRequest, SchedulerOutput


@dataclass(frozen=True)
class RunnerResult:
    request_id: str
    sampled_token_id: int | None
    processed_tokens: int


class DeterministicModelRunner:
    """A cheap deterministic stand-in for a causal model.

    The runner preserves per-request context state, so chunked and unchunked
    prefill produce identical tokens. It makes scheduler tests deterministic
    without requiring a model checkpoint or GPU.
    """

    def __init__(self, *, vocab_size: int = 128):
        if vocab_size <= 2:
            raise ValueError("vocab_size must be greater than two")
        self.vocab_size = vocab_size
        self._states: dict[str, int] = {}

    def _process(self, item: ScheduledRequest) -> RunnerResult:
        request_id = item.request.request_id
        state = self._states.get(request_id, 17)
        absolute_start = item.request.num_computed_tokens
        for offset, token_id in enumerate(item.token_ids):
            position = absolute_start + offset
            state = (state * 1_000_003 + token_id * 97 + position * 13) % 2**32
        self._states[request_id] = state
        sampled = None
        if item.reaches_sequence_end:
            sampled = 2 + state % (self.vocab_size - 2)
        return RunnerResult(request_id, sampled, item.num_scheduled_tokens)

    def execute(self, scheduler_output: SchedulerOutput) -> list[RunnerResult]:
        return [self._process(item) for item in scheduler_output.scheduled]

    def forget(self, request_id: str) -> None:
        self._states.pop(request_id, None)
