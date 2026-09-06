from dataclasses import dataclass

from llm_serving_lab.tracing import TraceRecorder

from .confidence_scheduler import ConfidenceScheduler
from .interfaces import DraftProposer, TargetModel
from .verifier import GreedyVerification, GreedyVerifier


@dataclass(frozen=True)
class DecodeRound:
    round_index: int
    prefix_length: int
    proposed_tokens: list[int]
    verified_tokens: list[int]
    accepted_count: int
    emitted_tokens: list[int]
    used_bonus_token: bool


@dataclass(frozen=True)
class DecodeResult:
    token_ids: list[int]
    generated_token_ids: list[int]
    rounds: list[DecodeRound]
    target_forward_calls: int
    proposed_tokens: int
    accepted_tokens: int

    @property
    def mean_accepted_length(self) -> float:
        return self.accepted_tokens / len(self.rounds) if self.rounds else 0.0


class SpeculativeDecoder:
    def __init__(
        self,
        target: TargetModel,
        proposer: DraftProposer,
        *,
        draft_length: int = 4,
        confidence_scheduler: ConfidenceScheduler | None = None,
        recorder: TraceRecorder | None = None,
    ):
        if draft_length <= 0:
            raise ValueError("draft_length must be positive")
        self.target = target
        self.proposer = proposer
        self.draft_length = draft_length
        self.confidence_scheduler = confidence_scheduler
        self.verifier = GreedyVerifier(target)
        self.recorder = recorder or TraceRecorder()

    def decode(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        confidences: list[float] | None = None,
        concurrency: int = 1,
    ) -> DecodeResult:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids cannot be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        all_tokens = prompt_token_ids.copy()
        generated: list[int] = []
        rounds: list[DecodeRound] = []
        total_proposed = 0
        total_accepted = 0

        while len(generated) < max_new_tokens:
            proposal_length = min(
                self.draft_length,
                max_new_tokens - len(generated),
            )
            confidence_data = None
            if self.confidence_scheduler is not None and confidences is not None:
                confidence_data = self.confidence_scheduler.schedule(
                    confidences[:proposal_length],
                    concurrency=concurrency,
                )
                proposal_length = confidence_data.verification_length

            proposal = self.proposer.propose(all_tokens, proposal_length)
            total_proposed += len(proposal)
            verification: GreedyVerification = self.verifier.verify(
                all_tokens, proposal
            )
            remaining = max_new_tokens - len(generated)
            emitted = verification.emitted_tokens[:remaining]
            all_tokens.extend(emitted)
            generated.extend(emitted)
            accepted_for_output = min(verification.accepted_count, len(emitted))
            total_accepted += accepted_for_output
            round_data = DecodeRound(
                round_index=len(rounds),
                prefix_length=len(all_tokens) - len(emitted),
                proposed_tokens=proposal,
                verified_tokens=verification.target_tokens,
                accepted_count=accepted_for_output,
                emitted_tokens=emitted,
                used_bonus_token=(
                    verification.used_bonus_token
                    and len(emitted) == len(verification.emitted_tokens)
                ),
            )
            rounds.append(round_data)
            self.recorder.emit(
                "speculative_round",
                step=round_data.round_index,
                prefix_length=round_data.prefix_length,
                proposed_tokens=proposal,
                verified_tokens=verification.target_tokens,
                accepted_count=accepted_for_output,
                rejected_count=verification.rejected_count,
                emitted_tokens=emitted,
                used_bonus_token=round_data.used_bonus_token,
                confidence_schedule=(
                    None
                    if confidence_data is None
                    else {
                        "verification_length": confidence_data.verification_length,
                        "prefix_survival": confidence_data.prefix_survival,
                        "survival_threshold": confidence_data.survival_threshold,
                        "token_budget": confidence_data.token_budget,
                        "concurrency": confidence_data.concurrency,
                    }
                ),
            )

        return DecodeResult(
            token_ids=all_tokens,
            generated_token_ids=generated,
            rounds=rounds,
            target_forward_calls=len(rounds),
            proposed_tokens=total_proposed,
            accepted_tokens=total_accepted,
        )


def autoregressive_decode(
    target: TargetModel,
    prompt_token_ids: list[int],
    *,
    max_new_tokens: int,
) -> list[int]:
    tokens = prompt_token_ids.copy()
    for _ in range(max_new_tokens):
        tokens.append(target.next_token(tokens))
    return tokens
