import numpy as np
import pytest

from llm_serving_lab.dflare import (
    DeterministicTargetFeatures,
    DFlareConfig,
    DFlareProposer,
    DFlashProposer,
    LayerwiseFusion,
)
from llm_serving_lab.speculative import DeterministicTarget, SpeculativeDecoder
from llm_serving_lab.speculative.decoder import autoregressive_decode
from llm_serving_lab.tracing import TraceRecorder


def _components():
    config = DFlareConfig(
        vocab_size=32,
        hidden_size=12,
        num_target_layers=3,
        num_draft_layers=2,
        block_size=4,
        target_layer_ids=(0, 2, 4),
        seed=11,
    )
    features = DeterministicTargetFeatures(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        total_target_layers=5,
        seed=5,
    )
    return config, features


def test_layerwise_fusion_contract_and_normalization() -> None:
    logits = np.array([[3.0, 0.0, -1.0], [-1.0, 0.0, 3.0]])
    target = np.arange(4 * 3 * 6, dtype=np.float64).reshape(4, 3, 6)
    fusion = LayerwiseFusion(logits)
    output = fusion.fuse(target)
    assert output.shape == (4, 2, 6)
    np.testing.assert_allclose(fusion.weights.sum(axis=1), np.ones(2))
    assert not np.allclose(output[:, 0], output[:, 1])
    np.testing.assert_allclose(
        np.mean(np.square(output), axis=-1), np.ones((4, 2)), rtol=1e-4
    )


def test_fusion_rejects_incompatible_target_layer_dimension() -> None:
    fusion = LayerwiseFusion(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="target-layer dimension"):
        fusion.fuse(np.zeros((4, 2, 8)))


def test_dflare_proposes_one_parallel_block_and_records_shapes() -> None:
    config, features = _components()
    recorder = TraceRecorder()
    proposer = DFlareProposer(config, features, recorder=recorder)
    proposal = proposer.propose_block([1, 4, 7], max_tokens=10)
    assert len(proposal.token_ids) == config.block_size
    assert proposal.logits.shape == (config.block_size, config.vocab_size)
    assert proposal.fusion_weights.shape == (
        config.num_draft_layers,
        config.num_target_layers,
    )
    assert recorder.events[0].kind == "dflare_block_proposal"
    assert recorder.events[0].fields["target_hidden_shape"] == [3, 3, 12]
    assert recorder.events[0].fields["fused_hidden_shape"] == [3, 2, 12]


def test_dflash_uses_shared_fusion_but_dflare_is_layer_specific() -> None:
    config, features = _components()
    dflash = DFlashProposer(config, features)
    dflare = DFlareProposer(config, features)
    np.testing.assert_allclose(dflash.fusion.weights[0], dflash.fusion.weights[1])
    assert not np.allclose(dflare.fusion.weights[0], dflare.fusion.weights[1])


@pytest.mark.parametrize("proposer_type", [DFlashProposer, DFlareProposer])
def test_block_parallel_draft_remains_lossless_after_target_verification(
    proposer_type,
) -> None:
    config, features = _components()
    target = DeterministicTarget(vocab_size=config.vocab_size)
    recorder = TraceRecorder()
    proposer = proposer_type(config, features, recorder=recorder)
    result = SpeculativeDecoder(
        target,
        proposer,
        draft_length=config.block_size,
        recorder=recorder,
    ).decode([1, 3, 7], max_new_tokens=12)
    assert result.token_ids == autoregressive_decode(
        target, [1, 3, 7], max_new_tokens=12
    )
    assert any(event.kind == "dflare_block_proposal" for event in recorder.events)
    assert any(event.kind == "speculative_round" for event in recorder.events)
