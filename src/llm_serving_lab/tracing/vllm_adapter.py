"""Opt-in, non-invasive tracing hooks for a pinned vLLM source checkout.

The adapter wraps selected methods at runtime and intentionally avoids importing
vLLM until ``install()`` is called. It is a learning aid, not a production
observability package.
"""

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from .recorder import TraceRecorder


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - defensive adapter code
            pass
    return repr(value)


def _request_id(request: Any) -> str | None:
    value = getattr(request, "request_id", None)
    return None if value is None else str(value)


class VLLMTraceAdapter:
    """Monkey-patch a few stable vLLM execution boundaries and emit JSONL."""

    def __init__(self, output_path: str | Path):
        self.recorder = TraceRecorder(output_path)
        self._originals: list[tuple[type[Any], str, Callable[..., Any]]] = []
        self._step = 0

    def _patch(
        self,
        owner: type[Any],
        method_name: str,
        wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]],
    ) -> None:
        original = getattr(owner, method_name)
        self._originals.append((owner, method_name, original))
        setattr(owner, method_name, wrapper_factory(original))

    def install(self) -> "VLLMTraceAdapter":
        try:
            from vllm.v1.core.kv_cache_manager import KVCacheManager
            from vllm.v1.core.sched.scheduler import Scheduler
            from vllm.v1.engine.output_processor import OutputProcessor
        except ImportError as error:  # pragma: no cover - optional integration
            raise RuntimeError(
                "vLLM is not importable; install the pinned checkout first"
            ) from error

        adapter = self

        def wrap_schedule(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(instance, *args, **kwargs)
                token_counts = getattr(result, "num_scheduled_tokens", {})
                adapter.recorder.emit(
                    "vllm_scheduler_step",
                    step=adapter._step,
                    scheduled_tokens=_plain(token_counts),
                    waiting=len(getattr(instance, "waiting", [])),
                    running=len(getattr(instance, "running", [])),
                    finished_req_ids=_plain(getattr(result, "finished_req_ids", None)),
                )
                adapter._step += 1
                return result

            return wrapped

        def wrap_allocate(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def wrapped(
                instance: Any,
                request: Any,
                num_new_tokens: int,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                result = original(
                    instance,
                    request,
                    num_new_tokens,
                    *args,
                    **kwargs,
                )
                block_ids = None
                if result is not None and hasattr(result, "get_block_ids"):
                    block_ids = result.get_block_ids(allow_none=True)
                adapter.recorder.emit(
                    "vllm_kv_allocate",
                    step=adapter._step,
                    request_id=_request_id(request),
                    num_new_tokens=int(num_new_tokens),
                    num_computed_tokens=int(
                        getattr(request, "num_computed_tokens", -1)
                    ),
                    block_ids=_plain(block_ids),
                    allocation_succeeded=result is not None,
                )
                return result

            return wrapped

        def wrap_outputs(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def wrapped(instance: Any, outputs: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(instance, outputs, *args, **kwargs)
                adapter.recorder.emit(
                    "vllm_outputs_processed",
                    step=max(0, adapter._step - 1),
                    output_count=len(outputs) if hasattr(outputs, "__len__") else None,
                    request_output_count=(
                        len(result.request_outputs)
                        if hasattr(result, "request_outputs")
                        else None
                    ),
                )
                return result

            return wrapped

        def wrap_spec_stats(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def wrapped(
                instance: Any,
                spec_decoding_stats: Any,
                num_draft_tokens: int,
                num_accepted_tokens: int,
                num_invalid_spec_tokens: dict[str, int] | None,
                request_id: str,
            ) -> Any:
                invalid = (
                    num_invalid_spec_tokens.get(request_id, 0)
                    if num_invalid_spec_tokens
                    else 0
                )
                adapter.recorder.emit(
                    "vllm_spec_decode_acceptance",
                    step=max(0, adapter._step - 1),
                    request_id=str(request_id),
                    num_draft_tokens=max(0, int(num_draft_tokens) - int(invalid)),
                    num_accepted_tokens=int(num_accepted_tokens),
                    num_invalid_spec_tokens=int(invalid),
                )
                return original(
                    instance,
                    spec_decoding_stats,
                    num_draft_tokens,
                    num_accepted_tokens,
                    num_invalid_spec_tokens,
                    request_id,
                )

            return wrapped

        self._patch(Scheduler, "schedule", wrap_schedule)
        self._patch(KVCacheManager, "allocate_slots", wrap_allocate)
        self._patch(OutputProcessor, "process_outputs", wrap_outputs)
        self._patch(Scheduler, "make_spec_decoding_stats", wrap_spec_stats)
        return self

    def uninstall(self) -> None:
        while self._originals:
            owner, method_name, original = self._originals.pop()
            setattr(owner, method_name, original)

    def close(self) -> Path:
        return self.recorder.write()

    def __enter__(self) -> "VLLMTraceAdapter":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.uninstall()
        self.close()
