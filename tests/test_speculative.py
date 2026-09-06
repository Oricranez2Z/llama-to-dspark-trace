from llm_serving_lab.speculative import (
    ConfidenceScheduler,
    DeterministicTarget,
    LoadProfile,
    NoisyProposer,
    PerfectProposer,
    SpeculativeDecoder,
)
from llm_serving_lab.speculative.decoder import autoregressive_decode


def test_speculative_greedy_matches_target_with_rejections() -> None:
    target = DeterministicTarget(vocab_size=64)
    prompt = [1, 3, 7]
    expected = autoregressive_decode(target, prompt, max_new_tokens=25)
    proposer = NoisyProposer(target, error_every=4, vocab_size=64)
    result = SpeculativeDecoder(target, proposer, draft_length=6).decode(
        prompt,
        max_new_tokens=25,
    )
    assert result.token_ids == expected
    assert result.accepted_tokens < result.proposed_tokens
    assert any(round_data.accepted_count < 6 for round_data in result.rounds)


def test_perfect_proposer_uses_bonus_tokens() -> None:
    target = DeterministicTarget()
    result = SpeculativeDecoder(
        target,
        PerfectProposer(target),
        draft_length=4,
    ).decode([1, 2], max_new_tokens=11)
    expected = autoregressive_decode(target, [1, 2], max_new_tokens=11)
    assert result.token_ids == expected
    assert any(round_data.used_bonus_token for round_data in result.rounds)
    assert result.target_forward_calls < 11


def test_high_load_uses_no_more_verification_capacity_than_low_load() -> None:
    scheduler = ConfidenceScheduler(
        LoadProfile(
            low_load_max_tokens=8,
            high_load_max_tokens=3,
            high_load_concurrency=16,
        )
    )
    confidences = [0.98, 0.95, 0.91, 0.85, 0.78, 0.69, 0.55, 0.42]
    low = scheduler.schedule(confidences, concurrency=1)
    high = scheduler.schedule(confidences, concurrency=16)
    assert high.token_budget <= low.token_budget
    assert high.verification_length <= low.verification_length
    assert high.survival_threshold >= low.survival_threshold


def test_zero_verification_length_falls_back_to_target_token() -> None:
    target = DeterministicTarget()
    scheduler = ConfidenceScheduler(
        LoadProfile(base_survival_threshold=0.9, high_load_survival_threshold=0.9)
    )
    result = SpeculativeDecoder(
        target,
        PerfectProposer(target),
        draft_length=4,
        confidence_scheduler=scheduler,
    ).decode(
        [1, 2],
        max_new_tokens=5,
        confidences=[0.1, 0.1, 0.1, 0.1],
    )
    assert result.token_ids == autoregressive_decode(target, [1, 2], max_new_tokens=5)
