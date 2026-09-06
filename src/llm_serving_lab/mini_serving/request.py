from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    eos_token_id: int | None = None
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    block_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids cannot be empty")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def phase(self) -> str:
        return "prefill" if self.num_computed_tokens < self.prompt_length else "decode"

    @property
    def num_uncomputed_tokens(self) -> int:
        return len(self.all_token_ids) - self.num_computed_tokens

    @property
    def can_sample_after_compute(self) -> bool:
        return self.num_computed_tokens >= len(self.all_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    def append_output(self, token_id: int) -> None:
        if self.is_finished:
            raise RuntimeError("cannot append a token to a finished request")
        self.output_token_ids.append(int(token_id))

    def should_finish(self, token_id: int) -> bool:
        reached_limit = len(self.output_token_ids) >= self.max_new_tokens
        reached_eos = self.eos_token_id is not None and token_id == self.eos_token_id
        return reached_limit or reached_eos
