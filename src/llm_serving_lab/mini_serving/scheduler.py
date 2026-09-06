from collections import deque
from dataclasses import dataclass

from llm_serving_lab.tracing import TraceRecorder

from .block_manager import BlockManager
from .request import Request, RequestStatus


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_batched_tokens: int = 32
    max_num_seqs: int = 8
    chunked_prefill: bool = True

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens <= 0 or self.max_num_seqs <= 0:
            raise ValueError("scheduler limits must be positive")


@dataclass(frozen=True)
class ScheduledRequest:
    request: Request
    phase: str
    token_ids: list[int]
    num_scheduled_tokens: int
    new_block_ids: list[int]
    slot_mapping: list[int]
    reaches_sequence_end: bool


@dataclass(frozen=True)
class SchedulerOutput:
    step: int
    scheduled: list[ScheduledRequest]
    token_budget_used: int


class Scheduler:
    """FCFS continuous-batching scheduler with chunked prefill."""

    def __init__(
        self,
        block_manager: BlockManager,
        config: SchedulerConfig | None = None,
        recorder: TraceRecorder | None = None,
    ):
        self.block_manager = block_manager
        self.config = config or SchedulerConfig()
        self.recorder = recorder or TraceRecorder()
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.current_step = 0

    def add_request(self, request: Request) -> None:
        if any(item.request_id == request.request_id for item in self.all_requests):
            raise ValueError(f"duplicate request id: {request.request_id}")
        if request.max_new_tokens == 0:
            request.status = RequestStatus.FINISHED
            self.finished.append(request)
        else:
            self.waiting.append(request)
        self.recorder.emit(
            "request_admitted",
            step=self.current_step,
            request_id=request.request_id,
            prompt_length=request.prompt_length,
            max_new_tokens=request.max_new_tokens,
            status=request.status.value,
        )

    @property
    def all_requests(self) -> list[Request]:
        return [*self.waiting, *self.running, *self.finished]

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def _admit_waiting(self) -> None:
        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            self.running.append(request)
            self.recorder.emit(
                "request_started",
                step=self.current_step,
                request_id=request.request_id,
            )

    def schedule(self) -> SchedulerOutput:
        self._admit_waiting()
        remaining_budget = self.config.max_num_batched_tokens
        scheduled: list[ScheduledRequest] = []

        for request in list(self.running):
            if remaining_budget == 0:
                break
            available = request.num_uncomputed_tokens
            if available <= 0:
                continue
            phase = request.phase
            if phase == "decode":
                num_tokens = 1
            elif self.config.chunked_prefill:
                num_tokens = min(available, remaining_budget)
            else:
                num_tokens = available
                if num_tokens > remaining_budget:
                    continue
            num_tokens = min(num_tokens, remaining_budget)
            new_blocks = self.block_manager.allocate(request, num_tokens)
            if new_blocks is None:
                self.recorder.emit(
                    "request_blocked",
                    step=self.current_step,
                    request_id=request.request_id,
                    reason="insufficient_kv_blocks",
                    free_blocks=self.block_manager.num_free_blocks,
                )
                continue

            start = request.num_computed_tokens
            token_ids = request.all_token_ids[start : start + num_tokens]
            slots = self.block_manager.slots_for_range(request, start, num_tokens)
            reaches_end = start + num_tokens == len(request.all_token_ids)
            item = ScheduledRequest(
                request=request,
                phase=phase,
                token_ids=token_ids,
                num_scheduled_tokens=num_tokens,
                new_block_ids=new_blocks,
                slot_mapping=slots,
                reaches_sequence_end=reaches_end,
            )
            scheduled.append(item)
            remaining_budget -= num_tokens
            self.recorder.emit(
                "request_scheduled",
                step=self.current_step,
                request_id=request.request_id,
                phase=phase,
                token_ids=token_ids,
                num_computed_tokens=request.num_computed_tokens,
                num_scheduled_tokens=num_tokens,
                block_ids=request.block_ids.copy(),
                new_block_ids=new_blocks,
                slot_mapping=slots,
                free_blocks=self.block_manager.num_free_blocks,
            )

        output = SchedulerOutput(
            step=self.current_step,
            scheduled=scheduled,
            token_budget_used=self.config.max_num_batched_tokens - remaining_budget,
        )
        self.recorder.emit(
            "scheduler_step",
            step=self.current_step,
            scheduled_request_ids=[item.request.request_id for item in scheduled],
            token_budget_used=output.token_budget_used,
            waiting=len(self.waiting),
            running=len(self.running),
            free_blocks=self.block_manager.num_free_blocks,
        )
        return output

    def finish(self, request: Request) -> list[int]:
        if request in self.running:
            self.running.remove(request)
        request.status = RequestStatus.FINISHED
        released = self.block_manager.free(request)
        self.finished.append(request)
        self.recorder.emit(
            "request_finished",
            step=self.current_step,
            request_id=request.request_id,
            output_token_ids=request.output_token_ids.copy(),
            released_block_ids=released,
            free_blocks=self.block_manager.num_free_blocks,
        )
        return released
