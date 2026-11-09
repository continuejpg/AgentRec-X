"""M7 — candidate-grounded interaction policy and experimental ranking refresh.

This module is the **smallest** policy layer on top of the accepted candidate-aware
preference construction (:mod:`recommendation.demo.preference_construction`).  It adds
nothing about products, scores or user facts; it only *chooses among four interaction
actions* and, in an explicitly gated experiment, asks the M6.1 ranking foundation to
refresh the order of the candidates it was given.

Actions
-------
* ``ASK``      -- ask for the preference dimension whose answer changes the presented
                  top-K (reuses the accepted evidence-backed prompt);
* ``COMPARE``  -- show the trusted per-candidate values of that dimension, then ask the
                  same bounded preference question;
* ``EXPLAIN``  -- show a grounded explanation of the attribute (values copied from the
                  accepted product-research tool), then ask the same bounded question;
* ``FINALIZE`` -- no more interaction value: end with the current list.

Rules are deterministic and inspectable: uncertainty/known-state blocking comes from the
accepted prompt builder, candidate difference/coverage/impact are its measured fields,
and the remaining interaction budget is an explicit integer.  The module never invents a
catalogue attribute and never changes a candidate set.

The experimental ranking bridge at the bottom is **off by default** and is only called by
:class:`recommendation.demo.agent_service.DemoAgentService` when the independent
``AGENTRECX_M7_GROUNDED_ACTIONS`` switch is on.  It converts the already-filtered,
already-presented candidates into the M6.1 public ``CandidateSnapshot`` contract and uses
the M6.1 A2 policy (explicit preference first, grounded ordinal inferred tie-break) to
refresh their order.  A failure of that experimental step is a diagnostic, never a
user-visible failure: the accepted ordering path remains the fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from recommendation.control.product_research import ProductResearchTool
from recommendation.state.schemas import (
    PreferenceDirection,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
)
from recommendation.state.state_projector import (
    RankerFeature,
    RankerStateProjection,
)

from .preference_construction import (
    PREFERENCE_ATTRIBUTES,
    PreferenceOption,
    PreferencePrompt,
    PreferenceState,
    TaskPreference,
    choose_preference_action,
)

__all__ = [
    "ENV_M7_ACTIONS",
    "ENV_M7_BUDGET",
    "GroundedAction",
    "GroundedDecision",
    "GroundedRankingRefresh",
    "choose_grounded_action",
    "grounded_actions_enabled",
    "grounded_budget",
    "is_uncertain_message",
    "refresh_candidate_order",
]

#: Independent experiment switch.  Unset/off = the accepted Serving/Agent behaviour.
ENV_M7_ACTIONS = "AGENTRECX_M7_GROUNDED_ACTIONS"
#: Maximum bounded interactions per shopping task in experiment mode.
ENV_M7_BUDGET = "AGENTRECX_M7_BUDGET"
DEFAULT_BUDGET = 2

_COMPARE_TERMS = re.compile(
    r"\b(compare|comparison|contrast|difference|differences|differ|versus|vs\.?|"
    r"trade[- ]?off)\b",
    re.IGNORECASE,
)
_EXPLAIN_TERMS = re.compile(
    r"\b(explain|explanation|meaning|what does|what is|why|understand|mean)\b",
    re.IGNORECASE,
)
_EXPLAIN_ATTRIBUTES = re.compile(
    r"\b(price|weight|color|colour|material|brand|fabric|cost|heaviness)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_TERMS = re.compile(
    r"\b("
    r"not\s+sure|unsure|don'?t\s+know|do\s+not\s+know|no\s+idea|"
    r"help\s+me\s+(?:choose|decide|pick)|which\s+(?:type|kind|one|option)|"
    r"what(?:'s|\s+is)\s+the\s+difference|difference\s+between"
    r")\b",
    re.IGNORECASE,
)


def _is_uncertain(message: str) -> bool:
    """Detect an explicit 'help me choose / I don't know' shopping statement."""
    return bool(_UNCERTAINTY_TERMS.search(str(message or "")))


def is_uncertain_message(message: str) -> bool:
    """Public uncertainty predicate used before recording M5 preferences."""
    return _is_uncertain(message)


class GroundedAction(str, Enum):
    """The four M7 interaction actions."""

    ASK = "ASK"
    COMPARE = "COMPARE"
    EXPLAIN = "EXPLAIN"
    FINALIZE = "FINALIZE"


@dataclass(frozen=True)
class GroundedDecision:
    """One bounded interaction decision.  ``prompt`` is the accepted prompt object."""

    action: GroundedAction
    prompt: PreferencePrompt | None
    reason: str
    comparison_rows: tuple[dict[str, Any], ...] = ()
    explanation: str | None = None
    #: True when the user's own message asked for this comparison/explanation.  Used by
    #: the M5 recorder so a proactive agent display never becomes a fake user request.
    user_requested: bool = False
    #: True when the action was triggered by an explicit uncertainty/ “help me choose” cue.
    uncertain: bool = False
    #: True when the deterministic policy chose the action without an explicit request.
    proactive: bool = False
    #: Deterministic diagnostics copied from the accepted prompt builder.
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundedRankingRefresh:
    """Result of the experimental M6.1 refresh, or ``None`` when it did not run."""

    order: tuple[str, ...] = ()
    changed: bool = False
    ran: bool = False
    error: str | None = None


def _env_flag(value: str | None = None) -> bool:
    raw = (
        value
        if value is not None
        else __import__("os").environ.get(ENV_M7_ACTIONS, "")
    )
    return str(raw).strip().lower() in {"1", "true", "on", "yes", "enabled", "shadow", "apply"}


def grounded_actions_enabled(value: str | None = None) -> bool:
    """True only when the independent M7 experiment switch is explicitly on.

    The default (unset, empty, ``off``/``0``/``false``) keeps the accepted behaviour.
    """
    return _env_flag(value)


def grounded_budget(value: str | None = None) -> int:
    """Bounded interaction budget from the experiment environment (default 2)."""
    import os

    raw = value if value is not None else os.environ.get(ENV_M7_BUDGET, DEFAULT_BUDGET)
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_BUDGET
    return max(1, min(parsed, 5))


def _asks_compare(message: str) -> bool:
    return bool(_COMPARE_TERMS.search(str(message or "")))


def _asks_explain(message: str) -> bool:
    text = str(message or "")
    return bool(_EXPLAIN_TERMS.search(text) and _EXPLAIN_ATTRIBUTES.search(text))


def _grounded_values(
    prompt: PreferencePrompt, records: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Copy the accepted comparison values; return ``({}, False)`` if unavailable."""
    values = dict(prompt.values)
    if len(values) < 2:
        return {}, False
    # The prompt is built by the accepted comparison tool, so every value is trusted.
    # Re-read only to confirm the fact is still a concrete value, never to guess one.
    for identity in prompt.products:
        try:
            inspection = ProductResearchTool(records).inspect_product(
                identity, attributes=(prompt.attribute,)
            )
        except Exception:  # noqa: BLE001 - absent evidence must not fabricate
            return {}, False
        fact = inspection.facts.get(prompt.attribute)
        if fact is None or fact.status != "value":
            return {}, False
    return values, True


def _compare_rows(prompt: PreferencePrompt, values: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """A compact, wire-safe view of the real product differences for one attribute."""
    rows = tuple(
        {
            "attribute": prompt.attribute,
            "comparison_kind": prompt.comparison_kind,
            "identity": identity,
            "value": values[identity],
        }
        for identity in prompt.products
        if identity in values
    )
    return rows


_ATTRIBUTE_INTERPRETATIONS: dict[str, str] = {
    "price": (
        "price is the recorded amount the catalogue lists; a lower amount costs less, "
        "but the catalogue does not establish that lower price is better for your use"
    ),
    "weight": (
        "weight is the shipped weight recorded by the catalogue; a lower weight may be "
        "easier to carry, but that trade-off depends on how you use the item and is not "
        "a catalogue quality ranking"
    ),
    "color": (
        "color is the recorded colour label; the catalogue does not rank colours as "
        "better or worse"
    ),
    "colour": (
        "colour is the recorded colour label; the catalogue does not rank colours as "
        "better or worse"
    ),
    "material": (
        "material is the recorded fabric/material text; the catalogue does not rank "
        "materials by quality"
    ),
    "brand": (
        "brand is the recorded store/brand label; the catalogue does not establish a "
        "quality ranking between brands"
    ),
}


def _explanation_text(
    prompt: PreferencePrompt,
    records: Mapping[str, Any],
    values: Mapping[str, Any],
) -> str | None:
    """Explain an attribute and its recorded values without inventing evidence.

    The text has three explicitly separate parts: (1) the catalogue values that are
    actually recorded, (2) what the attribute label generally means (interpretation, not
    a product claim), and (3) what is UNKNOWN.  It never asserts that the user understood
    it and never states which value is better.
    """
    known: list[tuple[str, Any]] = []
    missing: list[str] = []
    for identity in prompt.products:
        try:
            inspection = ProductResearchTool(records).inspect_product(
                identity, attributes=(prompt.attribute,)
            )
        except Exception:  # noqa: BLE001 - an absent fact is not a reason to invent one
            missing.append(identity)
            continue
        fact = inspection.facts.get(prompt.attribute)
        if fact is None or fact.status != "value":
            missing.append(identity)
            continue
        known.append((identity, fact.value))
    if not known:
        return None
    recorded = "; ".join(f"{identity}: {value}" for identity, value in known)
    parts = [
        f"The trusted catalogue records {prompt.attribute} for {len(known)} of "
        f"{len(prompt.products)} shown candidates: {recorded}."
    ]
    interpretation = _ATTRIBUTE_INTERPRETATIONS.get(prompt.attribute)
    if interpretation:
        parts.append(f"Attribute meaning: {interpretation}.")
    if prompt.comparison_kind == "numeric":
        numeric: list[float] = []
        for _, value in known:
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                continue
        if len(numeric) >= 2:
            parts.append(
                f"The recorded numeric range is {min(numeric):g} to {max(numeric):g}; "
                "this is a difference, not a judgement about which end you prefer."
            )
    if missing:
        parts.append(
            "UNKNOWN: no trusted "
            f"{prompt.attribute} value is recorded for "
            + ", ".join(missing)
            + "; those candidates are not compared on this attribute."
        )
    parts.append(
        "This explanation describes recorded catalogue values only; it does not assert "
        "that you understand the attribute or that you now prefer one of its values."
    )
    return " ".join(parts)


def choose_grounded_action(
    candidate_ids: Sequence[str],
    records: Mapping[str, Any],
    *,
    intent: Any = None,
    current_preferences: Sequence[TaskPreference] = (),
    blocked_attributes: Sequence[str] = (),
    message: str = "",
    top_k: int = 5,
    round_no: int = 1,
    state: PreferenceState | None = None,
    budget_remaining: int = DEFAULT_BUDGET,
) -> GroundedDecision:
    """Choose ASK / COMPARE / EXPLAIN / FINALIZE deterministically.

    ``FINALIZE`` wins whenever the interaction budget is exhausted, fewer than two
    grounded candidates remain, or the accepted prompt builder finds no dimension whose
    answer can change the presented top-K.  Otherwise the user's explicit comparison /
    explanation request selects COMPARE / EXPLAIN when trusted evidence exists; the
    default is ASK.  No action adds, removes or rescored a candidate.
    """
    if int(budget_remaining) <= 0:
        return GroundedDecision(GroundedAction.FINALIZE, None, "interaction_budget_exhausted")
    identities = tuple(str(item) for item in candidate_ids if str(item))
    if len(identities) < 2:
        return GroundedDecision(GroundedAction.FINALIZE, None, "fewer_than_two_candidates")
    accepted = choose_preference_action(
        identities,
        records,
        intent=intent,
        current_preferences=current_preferences,
        blocked_attributes=blocked_attributes,
        message=message,
        top_k=top_k,
        round_no=round_no,
        state=state,
    )
    prompt = accepted.dimension
    if accepted.action == "SKIP" or prompt is None:
        return GroundedDecision(
            GroundedAction.FINALIZE,
            None,
            "no_grounded_dimension_changes_top_k",
            diagnostics={"accepted_reason": accepted.reason},
        )
    values, values_ok = _grounded_values(prompt, records)
    explanation = (
        _explanation_text(prompt, records, values) if values_ok else None
    )
    distinct_values = len({str(value) for value in values.values()}) if values_ok else 0
    technical = prompt.attribute in {"weight", "material", "brand"}
    contrast_high = (
        values_ok
        and prompt.coverage >= 0.999
        and prompt.difference >= 0.4
        and distinct_values >= 3
        and prompt.impact >= 0.2
    )
    diagnostics = {
        "attribute": prompt.attribute,
        "coverage": prompt.coverage,
        "difference": prompt.difference,
        "impact": prompt.impact,
        "utility": prompt.utility,
        "known_state": prompt.status,
        "accepted_interaction": prompt.interaction,
        "values_grounded": values_ok,
        "distinct_values": distinct_values,
        "technical_attribute": technical,
        "contrast_high": contrast_high,
    }
    uncertain = _is_uncertain(message)
    diagnostics["uncertainty"] = uncertain
    requested_explain = _asks_explain(message)
    requested_compare = _asks_compare(message)
    # 0. The user explicitly says they do not know how to choose.  Prefer showing
    #    grounded example values, then asking; an explanation is used only for a
    #    technical attribute whose values are actually available.  This is an
    #    interaction choice, never a preference confirmation.
    if uncertain and values_ok:
        if technical and explanation is not None:
            return GroundedDecision(
                GroundedAction.EXPLAIN,
                prompt,
                "uncertain_user_gets_grounded_explanation",
                comparison_rows=_compare_rows(prompt, values),
                explanation=explanation,
                uncertain=True,
                diagnostics=diagnostics,
            )
        return GroundedDecision(
            GroundedAction.COMPARE,
            prompt,
            "uncertain_user_gets_grounded_examples",
            comparison_rows=_compare_rows(prompt, values),
            uncertain=True,
            diagnostics=diagnostics,
        )
    # 1. An explicit user request is honoured only when the evidence exists.  The other
    #    groundings are diagnostic only; no missing value is filled with a guess.
    if requested_explain and values_ok and explanation is not None:
        return GroundedDecision(
            GroundedAction.EXPLAIN,
            prompt,
            "user_requested_explanation_with_grounded_values",
            comparison_rows=_compare_rows(prompt, values),
            explanation=explanation,
            user_requested=True,
            diagnostics=diagnostics,
        )
    if requested_compare and values_ok:
        return GroundedDecision(
            GroundedAction.COMPARE,
            prompt,
            "user_requested_comparison_with_grounded_values",
            comparison_rows=_compare_rows(prompt, values),
            user_requested=True,
            diagnostics=diagnostics,
        )
    # 2. Proactive EXPLAIN: a technical attribute whose recorded values are available and
    #    whose answer can still change the presented list.  This does not claim the user
    #    asked for it, and the explanation is grounded in catalogue facts only.
    if (
        values_ok
        and explanation is not None
        and technical
        and prompt.coverage >= 0.5
        and prompt.impact >= 0.15
        and round_no <= 2
    ):
        return GroundedDecision(
            GroundedAction.EXPLAIN,
            prompt,
            "proactive_explanation_of_technical_attribute",
            comparison_rows=_compare_rows(prompt, values),
            explanation=explanation,
            proactive=True,
            diagnostics=diagnostics,
        )
    # 3. Proactive COMPARE: a material numeric contrast across the current candidates is
    #    shown before asking, because the user is otherwise choosing between values they
    #    have not seen together.
    if contrast_high and prompt.attribute in {"price", "weight"}:
        return GroundedDecision(
            GroundedAction.COMPARE,
            prompt,
            "proactive_comparison_of_material_numeric_contrast",
            comparison_rows=_compare_rows(prompt, values),
            proactive=True,
            diagnostics=diagnostics,
        )
    # 4. ASK is the conservative default: show the accepted evidence-backed contrast
    #    question without claiming a comparison/explanation was requested.
    return GroundedDecision(
        GroundedAction.ASK,
        prompt,
        "grounded_dimension_changes_top_k",
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# Experimental M6.1 refresh (off by default; same candidate set; no fabrication)
# --------------------------------------------------------------------------- #

_KIND_BY_ATTRIBUTE = {
    "color": PreferenceKind.COLOR,
    "colour": PreferenceKind.COLOR,
    "material": PreferenceKind.MATERIAL,
    "brand": PreferenceKind.BRAND,
}


def _task_feature(preference: TaskPreference) -> RankerFeature | None:
    """Translate one accepted task preference into an M5 ranker feature.

    A categorical value is an explicit soft preference (M10A match evidence).  A
    ``lower``/``higher`` answer is a directional preference: it is represented as a
    task-scoped hypothesis so it can only act through M6.1's ordinal inferred tie-break,
    never as a hard filter or a fabricated exact value.
    """
    attribute = str(preference.attribute or "").strip().lower()
    if not attribute:
        return None
    preference_id = f"m7:task:{attribute}:{preference.action}"
    if preference.action in {"lower", "higher"}:
        direction = (
            PreferenceDirection.MINIMIZE
            if preference.action == "lower"
            else PreferenceDirection.MAXIMIZE
        )
        # Numeric answers are EXPLICIT user preferences.  They must stay explicit all the
        # way into M6.1; M6.1 grounds the direction against trusted, same-category numeric
        # catalogue values.  Never downgrade this to an inferred soft hypothesis.
        return RankerFeature(
            preference_id=preference_id,
            kind=PreferenceKind.FEATURE,
            value=attribute,
            polarity=PreferencePolarity.PREFER,
            direction=direction,
            scope=PreferenceScope.TASK,
            lifecycle=PreferenceLifecycle.EXPLICIT,
            strength_rank=2,
            is_soft=False,
        )
    if preference.action == "value" and preference.value is not None:
        value = str(preference.value).strip()
        if not value:
            return None
        return RankerFeature(
            preference_id=preference_id + ":" + value.casefold(),
            kind=_KIND_BY_ATTRIBUTE.get(attribute, PreferenceKind.FEATURE),
            value=value,
            polarity=PreferencePolarity.PREFER,
            direction=None,
            scope=PreferenceScope.TASK,
            lifecycle=PreferenceLifecycle.EXPLICIT,
            strength_rank=2,
            is_soft=False,
        )
    return None


def _experimental_projection(
    base: RankerStateProjection, preferences: Sequence[TaskPreference]
) -> RankerStateProjection:
    features = list(base.features)
    filtered: list[RankerFeature] = []
    for preference in preferences:
        feature = _task_feature(preference)
        if feature is not None:
            filtered.append(feature)
    seen: set[tuple[Any, str, Any]] = set()
    result: list[RankerFeature] = []
    # M5 already records explicit statements in the projection.  An explicit task
    # preference is appended only when the same (kind, value, direction) is not already
    # present, so the event-derived feature remains authoritative and no duplicate
    # evidence is created.  No soft/inferred conversion happens here.
    for feature in (*features, *filtered):
        key = (
            feature.kind,
            feature.value.casefold(),
            feature.direction,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(feature)
    return base.model_copy(update={"features": tuple(result)})


def _candidate_snapshot(
    *,
    user_key: str,
    task_id: str,
    turn_key: str,
    recommendations: Sequence[Any],
    records: Mapping[str, Any],
) -> Any:
    """Build the M6.1 public candidate snapshot from already-presented candidates."""
    from recommendation.catalog.schemas import ProductMetadata
    from recommendation.ranking.schemas import CandidateRecord, CandidateSnapshot

    candidates = []
    for position, item in enumerate(recommendations, start=1):
        identity = str(item.parent_asin)
        raw = records.get(identity)
        metadata = raw if isinstance(raw, ProductMetadata) else None
        candidates.append(
            CandidateRecord(
                candidate_id=identity,
                item_id=int(item.item_id),
                original_rank=position,
                score=float(item.score),
                score_kind=str(item.score_kind or "presented_order"),
                metadata_status="found" if metadata is not None else "missing",
                metadata=metadata,
                rrf_score=None,
            )
        )
    return CandidateSnapshot(
        snapshot_id=f"m7:{task_id}:{turn_key}"[:200],
        user_key=user_key,
        task_id=task_id,
        turn_key=turn_key,
        candidates=tuple(candidates),
        source="m7_presented_candidates",
    )


def refresh_candidate_order(
    *,
    recommendations: Sequence[Any],
    records: Mapping[str, Any],
    base_projection: RankerStateProjection | None,
    preferences: Sequence[TaskPreference],
    user_key: str,
    task_id: str,
    turn_key: str,
) -> GroundedRankingRefresh:
    """Refresh order through the M6.1 public interface, preserving the candidate set.

    Returns ``ran`` False (with no user-visible effect) when there is nothing to do or
    the experimental path raises.  The caller always keeps the accepted order as the
    fallback, so a diagnostics-only failure cannot break a shopping turn.
    """
    if base_projection is None or not preferences or len(recommendations) < 2:
        return GroundedRankingRefresh()
    try:
        from recommendation.ranking.features import BeliefAwareFeatureBuilder
        from recommendation.ranking.policies import rank_all_arms
        from recommendation.ranking.schemas import RankingArm

        snapshot = _candidate_snapshot(
            user_key=user_key,
            task_id=task_id,
            turn_key=turn_key,
            recommendations=recommendations,
            records=records,
        )
        projection = _experimental_projection(base_projection, preferences)
        feature_set = BeliefAwareFeatureBuilder().build(
            snapshot=snapshot, ranker_projection=projection
        )
        arms = rank_all_arms(feature_set, snapshot)
        a2 = next((arm for arm in arms if arm.arm is RankingArm.A2_BELIEF_TIEBREAK), None)
        if a2 is None:
            return GroundedRankingRefresh(error="a2_arm_missing")
        order = tuple(a2.candidate_ids)
        before = tuple(str(item.parent_asin) for item in recommendations)
        if set(order) != set(before) or len(order) != len(before):
            return GroundedRankingRefresh(error="candidate_set_changed")
        return GroundedRankingRefresh(
            order=order,
            changed=order != before,
            ran=True,
        )
    except Exception as exc:  # noqa: BLE001 - experiment failure must not break serving
        return GroundedRankingRefresh(error=type(exc).__name__)
