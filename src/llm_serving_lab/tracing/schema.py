from dataclasses import asdict, dataclass, field
from time import time_ns
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    """A stable, JSON-serializable event shared by all lab components."""

    kind: str
    step: int
    request_id: str | None = None
    timestamp_ns: int = field(default_factory=time_ns)
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TraceEvent":
        return cls(
            kind=str(value["kind"]),
            step=int(value["step"]),
            request_id=value.get("request_id"),
            timestamp_ns=int(value.get("timestamp_ns", 0)),
            fields=dict(value.get("fields", {})),
        )
