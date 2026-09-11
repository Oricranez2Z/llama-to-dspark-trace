from __future__ import annotations

from dataclasses import dataclass

from .recorder import TraceRecorder


@dataclass(frozen=True)
class DFlareProposalRound:
    step: int
    prefix_length: int
    variant: str
    target_layer_ids: list[int]
    fusion_weights: list[list[float]]
    draft_tokens: list[int]
    target_hidden_shape: list[int]
    fused_hidden_shape: list[int]


class DFlareTraceRecorder:
    """Emit DFlash/DFlare proposal events using the shared trace schema."""

    def __init__(self, recorder: TraceRecorder | None = None):
        self.recorder = recorder or TraceRecorder()

    def record_proposal(self, round_data: DFlareProposalRound) -> None:
        self.recorder.emit(
            "dflare_block_proposal",
            step=round_data.step,
            prefix_length=round_data.prefix_length,
            variant=round_data.variant,
            target_layer_ids=round_data.target_layer_ids,
            fusion_weights=round_data.fusion_weights,
            draft_tokens=round_data.draft_tokens,
            target_hidden_shape=round_data.target_hidden_shape,
            fused_hidden_shape=round_data.fused_hidden_shape,
        )
