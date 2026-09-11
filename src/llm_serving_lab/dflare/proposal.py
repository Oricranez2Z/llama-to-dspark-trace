from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from llm_serving_lab.tracing import TraceRecorder
from llm_serving_lab.tracing.dflare import DFlareProposalRound, DFlareTraceRecorder

from .config import DFlareConfig
from .features import TargetFeatureProvider
from .fusion import LayerwiseFusion, SharedFusion

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BlockProposal:
    token_ids: list[int]
    logits: FloatArray
    fusion_weights: FloatArray
    variant: str


class _BlockParallelProposer:
    """Small block-parallel draft network shared by DFlash and DFlare."""

    def __init__(
        self,
        config: DFlareConfig,
        feature_provider: TargetFeatureProvider,
        *,
        variant: str,
        recorder: TraceRecorder | None = None,
    ):
        if variant not in {"dflash", "dflare"}:
            raise ValueError("variant must be dflash or dflare")
        self.config = config
        self.feature_provider = feature_provider
        self.variant = variant
        self.trace = DFlareTraceRecorder(recorder or TraceRecorder())
        self._round_index = 0

        rng = np.random.default_rng(config.seed)
        scale = config.hidden_size**-0.5
        raw_fusion = rng.normal(
            0.0,
            0.8,
            size=(config.num_draft_layers, config.num_target_layers),
        )
        if variant == "dflare":
            self.fusion = LayerwiseFusion(raw_fusion, eps=config.rms_norm_eps)
        else:
            self.fusion = SharedFusion(
                raw_fusion.mean(axis=0),
                num_draft_layers=config.num_draft_layers,
                eps=config.rms_norm_eps,
            )

        self.mask_embedding = rng.normal(0.0, scale, size=config.hidden_size)
        self.position_embeddings = rng.normal(
            0.0, scale, size=(config.block_size, config.hidden_size)
        )
        self.noise_kernels = rng.normal(
            0.0,
            scale,
            size=(config.num_draft_layers, config.hidden_size, config.hidden_size),
        )
        self.context_kernels = rng.normal(
            0.0,
            scale,
            size=(config.num_draft_layers, config.hidden_size, config.hidden_size),
        )
        self.biases = rng.normal(
            0.0, scale, size=(config.num_draft_layers, config.hidden_size)
        )
        self.lm_head = rng.normal(
            0.0, scale, size=(config.vocab_size, config.hidden_size)
        )
        self.last_proposal: BlockProposal | None = None

    @property
    def recorder(self) -> TraceRecorder:
        return self.trace.recorder

    def propose_block(self, prefix: list[int], max_tokens: int) -> BlockProposal:
        if not prefix:
            raise ValueError("prefix cannot be empty")
        if max_tokens < 0:
            raise ValueError("max_tokens cannot be negative")
        length = min(max_tokens, self.config.block_size)
        if length == 0:
            proposal = BlockProposal(
                token_ids=[],
                logits=np.empty((0, self.config.vocab_size)),
                fusion_weights=self.fusion.weights.copy(),
                variant=self.variant,
            )
            self.last_proposal = proposal
            return proposal

        target_hidden = self.feature_provider.hidden_states(
            prefix, self.config.target_layer_ids
        )
        layer_context = self.fusion.fuse(target_hidden)[-1]
        hidden = np.repeat(self.mask_embedding[None, :], length, axis=0)
        hidden += self.position_embeddings[:length]
        for layer_index in range(self.config.num_draft_layers):
            hidden = np.tanh(
                hidden @ self.noise_kernels[layer_index]
                + layer_context[layer_index] @ self.context_kernels[layer_index]
                + self.biases[layer_index]
            )
        logits = hidden @ self.lm_head.T
        tokens = np.argmax(logits, axis=-1).astype(int).tolist()
        proposal = BlockProposal(
            token_ids=tokens,
            logits=logits,
            fusion_weights=self.fusion.weights.copy(),
            variant=self.variant,
        )
        self.last_proposal = proposal
        self.trace.record_proposal(
            DFlareProposalRound(
                step=self._round_index,
                prefix_length=len(prefix),
                variant=self.variant,
                target_layer_ids=list(self.config.target_layer_ids),
                fusion_weights=proposal.fusion_weights.tolist(),
                draft_tokens=tokens,
                target_hidden_shape=list(target_hidden.shape),
                fused_hidden_shape=[
                    target_hidden.shape[0],
                    self.config.num_draft_layers,
                    self.config.hidden_size,
                ],
            )
        )
        self._round_index += 1
        return proposal

    def propose(self, prefix: list[int], max_tokens: int) -> list[int]:
        return self.propose_block(prefix, max_tokens).token_ids


class DFlareProposer(_BlockParallelProposer):
    def __init__(
        self,
        config: DFlareConfig,
        feature_provider: TargetFeatureProvider,
        *,
        recorder: TraceRecorder | None = None,
    ):
        super().__init__(
            config,
            feature_provider,
            variant="dflare",
            recorder=recorder,
        )


class DFlashProposer(_BlockParallelProposer):
    """Control model with one target fusion shared by all draft layers."""

    def __init__(
        self,
        config: DFlareConfig,
        feature_provider: TargetFeatureProvider,
        *,
        recorder: TraceRecorder | None = None,
    ):
        super().__init__(
            config,
            feature_provider,
            variant="dflash",
            recorder=recorder,
        )
