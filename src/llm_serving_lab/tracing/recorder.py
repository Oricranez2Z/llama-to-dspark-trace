import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schema import TraceEvent


class TraceRecorder:
    """Collect trace events in memory and optionally stream them to JSONL."""

    def __init__(self, output_path: str | Path | None = None):
        self.output_path = Path(output_path) if output_path is not None else None
        self.events: list[TraceEvent] = []

    def emit(
        self,
        kind: str,
        *,
        step: int,
        request_id: str | None = None,
        **fields: Any,
    ) -> TraceEvent:
        event = TraceEvent(
            kind=kind,
            step=step,
            request_id=request_id,
            fields=fields,
        )
        self.events.append(event)
        return event

    def extend(self, events: Iterable[TraceEvent]) -> None:
        self.events.extend(events)

    def write(self, output_path: str | Path | None = None) -> Path:
        path = Path(output_path) if output_path is not None else self.output_path
        if path is None:
            raise ValueError("an output path is required")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        temporary.replace(path)
        return path

    @staticmethod
    def read(path: str | Path) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(TraceEvent.from_dict(json.loads(line)))
        return events
