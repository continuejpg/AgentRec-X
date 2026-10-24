"""Phase 3.2: the terminal state distinguishes "cannot verify" from "execution failed".

``no_verified_candidate`` is not a new success path and it does not relax the completion
guard: the guard still refuses, the run still fails closed, and nothing unverified is
presented.  What changes is the controller-owned reason, so a caller can tell this outcome
apart from an integrity/verification failure (``completion_refused``) or a capability
failure (``execution_failed``).
"""

from __future__ import annotations

from types import SimpleNamespace

from recommendation.control.completion import CompletionGuard
from recommendation.control.schemas import RunStatus, TerminationReason, VerificationResult
from recommendation.demo.agent_chat_adapter import NO_VERIFIED_CANDIDATE, agent_outcome
from recommendation.demo.observation import BLOCKING_TERMINATIONS
from recommendation.evaluation.agent.runner import _terminal_outcome
from recommendation.evaluation.agent.schemas import TerminalOutcome


def _refusal(code: str) -> VerificationResult:
    return VerificationResult(verified=False, code=code, detail="detail", checks=())


def test_no_verified_candidate_gets_its_own_reason() -> None:
    guard = CompletionGuard()
    reason = guard.termination_reason(_refusal("no_verified_compliant_candidate"))
    assert reason is TerminationReason.NO_VERIFIED_CANDIDATE
    assert reason.value == "no_verified_candidate"


def test_other_refusals_keep_the_accepted_reason() -> None:
    guard = CompletionGuard()
    for code in ("candidates_not_grounded", "last_verification_refused", "last_execution_failed"):
        assert guard.termination_reason(_refusal(code)) is TerminationReason.COMPLETION_REFUSED


def test_the_new_reason_still_fails_closed() -> None:
    guard = CompletionGuard()
    refusal = _refusal("no_verified_compliant_candidate")
    assert guard.terminal_status(refusal) is RunStatus.FAILED


def test_the_new_reason_is_distinct_from_execution_failure() -> None:
    """No success path is introduced: the refusal remains distinct from execution failure."""
    assert TerminationReason.NO_VERIFIED_CANDIDATE is not TerminationReason.COMPLETED
    assert TerminationReason.NO_VERIFIED_CANDIDATE is not TerminationReason.EXECUTION_FAILED


def test_a_chat_turn_with_the_new_reason_is_still_refused() -> None:
    response = SimpleNamespace(
        recommendations=(),
        terminal=SimpleNamespace(succeeded=False, termination_reason=NO_VERIFIED_CANDIDATE),
    )
    assert agent_outcome(response, proposed_route="recommend") == "refused"


def test_a_replan_cannot_route_around_the_new_reason() -> None:
    assert "no_verified_candidate" in BLOCKING_TERMINATIONS


def test_the_evaluation_vocabulary_keeps_the_refusal_outcome() -> None:
    result = SimpleNamespace(
        status=RunStatus.FAILED,
        control=SimpleNamespace(termination_reason=TerminationReason.NO_VERIFIED_CANDIDATE),
    )
    assert _terminal_outcome(result) is TerminalOutcome.COMPLETION_REFUSED
