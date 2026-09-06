from dataclasses import dataclass

from llm_serving_lab.tracing import TraceRecorder

from .model_runner import DeterministicModelRunner
from .request import Request
from .scheduler import Scheduler


@dataclass(frozen=True)
class EngineStepOutput:
    step: int
    emitted_tokens: dict[str, int]
    token_budget_used: int


class Engine:
    """Coordinate scheduling, deterministic model execution, and state commit."""

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: DeterministicModelRunner,
        recorder: TraceRecorder | None = None,
    ):
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.recorder = recorder or scheduler.recorder

    def add_request(self, request: Request) -> None:
        self.scheduler.add_request(request)

    def step(self) -> EngineStepOutput:
        schedule = self.scheduler.schedule()
        results = self.model_runner.execute(schedule)
        by_request = {result.request_id: result for result in results}
        emitted: dict[str, int] = {}

        for item in schedule.scheduled:
            request = item.request
            result = by_request[request.request_id]
            request.num_computed_tokens += result.processed_tokens
            self.recorder.emit(
                "model_executed",
                step=schedule.step,
                request_id=request.request_id,
                phase=item.phase,
                processed_tokens=result.processed_tokens,
                num_computed_tokens=request.num_computed_tokens,
                sampled_token_id=result.sampled_token_id,
            )
            if result.sampled_token_id is None:
                continue
            request.append_output(result.sampled_token_id)
            emitted[request.request_id] = result.sampled_token_id
            self.recorder.emit(
                "token_emitted",
                step=schedule.step,
                request_id=request.request_id,
                token_id=result.sampled_token_id,
                output_length=len(request.output_token_ids),
            )
            if request.should_finish(result.sampled_token_id):
                self.scheduler.finish(request)
                self.model_runner.forget(request.request_id)

        self.scheduler.block_manager.validate(self.scheduler.all_requests)
        output = EngineStepOutput(
            step=schedule.step,
            emitted_tokens=emitted,
            token_budget_used=schedule.token_budget_used,
        )
        self.scheduler.current_step += 1
        return output

    def run(self, *, max_steps: int = 10_000) -> list[EngineStepOutput]:
        outputs: list[EngineStepOutput] = []
        while self.scheduler.has_unfinished_requests:
            if len(outputs) >= max_steps:
                raise RuntimeError("engine exceeded max_steps without completing")
            output = self.step()
            outputs.append(output)
            if output.token_budget_used == 0:
                raise RuntimeError("scheduler made no progress")
        return outputs
