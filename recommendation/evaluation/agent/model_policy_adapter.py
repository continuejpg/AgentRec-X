"""Model-driven policy support for the agent evaluation plane (Phase 1).

This module is the evaluator-side bridge between a **model-driven policy** and the existing
trajectory evaluation.  It adds no production behaviour: a model policy is injected exactly like
any other policy, so no ablation switch enters the runtime.

Two pieces:

:func:`build_model_policy_factory`
    Returns a factory that gives each case a fresh
    :class:`~recommendation.control.model_policy.LLMAgentPolicy` over an
    :class:`ObservationReactiveModel`.  The runner calls it per case, so a suite run under the
    model variant uses one scripted model per case and stays deterministic.

:class:`ObservationReactiveModel`
    A **model double that reads the observation it is given**.  This is what makes the model
    variant a genuine test of observation-conditioned control rather than a replay of a fixed
    script: the double decides from ``last_observation`` in the context payload, exactly as a
    real model would, and it is fed through the same
    :class:`~recommendation.control.model_client.StructuredModelClient` seam.

    It is a double, not an intelligence: it encodes the behaviour a competent policy should show
    on these cases (retrieve when nothing is grounded, re-read facts before answering, ask when
    the request is underspecified, finish once candidates are grounded).  Its purpose is to prove
    the *runtime* is model-driven and observation-dependent while remaining reproducible offline.
"""

from __future__ import annotations

from typing import Any, Callable

from recommendation.control.model_client import ModelRequest, ModelResponse
from recommendation.control.model_policy import LLMAgentPolicy

from .cases import EvaluationCase
from .runner import ABLATION_MODEL_POLICY

__all__ = [
    "ObservationReactiveModel",
    "build_live_model_policy_factory",
    "build_model_policy_factory",
]


class ObservationReactiveModel:
    """A deterministic model double that chooses from the observation it is given.

    The decision rule mirrors the policy contract:

    * nothing grounded yet and a catalogue search is offered -> search the catalogue;
    * the previous attempt failed or came back empty -> switch source rather than repeat;
    * candidates grounded but the facts not read yet -> read the grounded facts;
    * facts read and a comparison is offered and useful -> compare;
    * candidates grounded and the request answered -> finish.

    It never invents a product identity: the identities it may target come from
    ``candidates.candidate_refs`` in the request payload, and it chooses among them the way a
    real model would.  It selects the target itself - the binding policy no longer picks for it
    - and it never emits a catalogue fact or an unsupported action.  Tests assert both the legal
    selection and the refusals.
    """

    #: One synthetic query term suffices for the suite catalogue; a real model would derive terms
    #: from the user request, which is why this is a double and not a policy.
    DEFAULT_TERMS: tuple[str, ...] = ("redwidget",)

    def __init__(
        self,
        *,
        terms: tuple[str, ...] | None = None,
    ) -> None:
        self._terms = terms or self.DEFAULT_TERMS
        #: Case-derived expectations injected by the binding policy.  These describe the *task*
        #: (does it need a catalogue search? does it need evidence?), never an identity: the
        #: candidate identities this double may target come from the payload, exactly as they
        #: would for a real model.
        self._hints: dict[str, Any] = {}
        self.requests: list[ModelRequest] = []
        #: The candidate references the last request offered, so a test can assert the double
        #: chose among the ones it was actually given.
        self.offered_refs: tuple[str, ...] = ()
        #: The references from the most recent payload, for target ordering.
        self.context_refs: tuple[str, ...] = ()
        self._last_payload: dict[str, Any] = {}

    @property
    def call_count(self) -> int:
        """How many times this model was asked for an answer."""
        return len(self.requests)

    # -- the seam ---------------------------------------------------------- #

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Read the offered actions and the observation, and return one action object as JSON."""
        import json

        self.requests.append(request)
        offered = {entry["action"] for entry in request.action_schema}
        payload = dict(request.context_payload)
        # The legal targets come from the payload, exactly as they would for a real model.  The
        # double is *told the task shape* through hints; it is never told an identity out of band.
        self.offered_refs = tuple(
            str(entry.get("parent_asin", ""))
            for entry in payload.get("candidates", {}).get("candidate_refs", [])
            if entry.get("parent_asin")
        )
        self.context_refs = self.offered_refs
        payload.update(self._hints)
        self._last_payload = payload
        action = self._decide(payload, offered)
        return ModelResponse(
            text=json.dumps(action, sort_keys=True),
            model_id="observation-reactive-double",
        )

    # -- decision rule ----------------------------------------------------- #

    def _decide(self, payload: dict[str, Any], offered: set[str]) -> dict[str, Any]:
        """Choose one action object from the bounded context payload and the offered menu."""
        self._last_offered = offered
        wants_reasoning = bool(payload.get("evidence_needed", False))
        observation = payload.get("last_observation")
        grounded = bool(payload.get("candidates", {}).get("grounded"))
        count = int(payload.get("candidates", {}).get("count", 0) or 0)
        awaiting = bool(payload.get("awaiting_user"))

        # An underspecified request should be asked about rather than guessed at.
        if self._underspecified(payload) and "ask_clarification" in offered and not awaiting:
            return {
                "action": "ask_clarification",
                "arguments": {
                    "question": "What will you use it for, so I can narrow the options?",
                    "blocks": "candidate_set",
                },
                "rationale": "the request does not say what the product is for",
            }

        # Nothing usable yet: prefer retrieval, and switch source after a failure or an empty
        # result rather than repeating the action that just failed.
        if not grounded:
            failed = isinstance(observation, dict) and (
                observation.get("status") == "empty"
                or observation.get("verification_status") == "refused"
                or observation.get("code") is not None
            )
            if failed and "search_catalog" in offered:
                return {
                    "action": "search_catalog",
                    "arguments": {"terms": list(self._terms), "limit": 4},
                    "rationale": "the previous attempt produced nothing usable",
                }
            if "search_catalog" in offered and self._wants_catalog(payload):
                return {
                    "action": "search_catalog",
                    "arguments": {"terms": list(self._terms), "limit": 4},
                    "rationale": "the user named a specific product need, so search the catalogue",
                }
            if "recommend_from_history" in offered and payload.get("has_trusted_history"):
                return {
                    "action": "recommend_from_history",
                    "k": 4,
                    "rationale": "try the trusted history recommender first",
                }
            if "search_catalog" in offered:
                return {
                    "action": "search_catalog",
                    "arguments": {"terms": list(self._terms), "limit": 4},
                    "rationale": "no history route is available, so search the catalogue",
                }
            return {"action": "finish", "rationale": "nothing else is available"}

        # Candidates are grounded.  Whether to gather evidence first is a decision: a simple
        # request should be answered efficiently, and only a request that needs evidence should
        # spend steps on it.  A double cannot judge that from the request, so the evaluator
        # supplies the case's own expectation through the payload.
        if not wants_reasoning:
            return {
                "action": "finish",
                "rationale": "candidates are grounded and the request is simple",
            }
        kind = observation.get("kind") if isinstance(observation, dict) else None
        # Target selection is the model's own decision, made from the bounded reference list it
        # was offered.  A candidate whose evidence is unsettled is the informative one to read,
        # so it is preferred; the choice is still expressed as a normal reasoning argument.
        names = self._target_order()
        if kind not in ("details", "comparison") and "get_details" in offered and names:
            return {
                "action": "get_details",
                "arguments": {"parent_asins": [names[0]]},
                "rationale": f"read the grounded facts for {names[0]}",
            }
        if kind == "details" and "compare" in offered and len(names) >= 2:
            return {
                "action": "compare",
                "arguments": {"parent_asins": names[:2], "attributes": ["weight_text"]},
                "rationale": f"compare {names[0]} with {names[1]}",
            }
        return {"action": "finish", "rationale": "the candidates answer the request"}

    # -- helpers ----------------------------------------------------------- #

    def _target_order(self) -> list[str]:
        """The offered references, most-informative-first.

        Orders the *legal* targets - it never adds one - so the double behaves like a competent
        model: read a candidate whose evidence is unsettled before one already settled.  The
        identity set is exactly what the payload offered, which is what makes the resulting
        action a legal reference rather than a manufactured one.
        """
        refs = self.context_refs or self.offered_refs
        payload = self._last_payload or {}
        by_identity = {
            str(entry.get("parent_asin", "")): str(entry.get("evidence", "unconstrained"))
            for entry in payload.get("candidates", {}).get("candidate_refs", [])
        }
        return sorted(
            refs,
            key=lambda identity: (by_identity.get(identity) != "attention", refs.index(identity)),
        )

    @staticmethod
    def _underspecified(payload: dict[str, Any]) -> bool:
        """True when the request names no product, so a question is warranted.

        Deliberately crude: a double is allowed a crude heuristic.  A real model would judge
        this, and the point of the variant is the runtime, not the judgement.
        """
        request = str(payload.get("user_request", "")).casefold()
        markers = ("something", "anything", "gift", "surprise me")
        return any(marker in request for marker in markers)

    @staticmethod
    def _wants_catalog(payload: dict[str, Any]) -> bool:
        """True when the request states a specific need the history recommender cannot serve.

        The evaluator supplies the case's expectation through the payload, because a double
        cannot judge intent from the request text - and pretending it could would make the
        variant a fiction rather than a test of the runtime.
        """
        return bool(payload.get("search_first", False))


def build_model_policy_factory(
    *,
    terms: tuple[str, ...] | None = None,
    max_attempts: int = 2,
) -> Callable[[EvaluationCase], Any]:
    """Return a per-case factory that builds an ``LLMAgentPolicy`` over a model double.

    The runner calls the returned factory once per case, so each case gets a fresh model and a
    fresh policy - the same isolation a real model would need, and the reason a suite run is
    reproducible.
    """

    def factory(case: EvaluationCase) -> Any:
        model = ObservationReactiveModel(terms=terms)
        hints = {
            # A case that requires a catalog search has stated a need the history route cannot
            # serve; a fast-path case has not, and must try history first.
            "search_first": "search_catalog" in case.required_actions,
            "evidence_needed": case.needs_reasoning,
        }
        return _IdentityBindingPolicy(model, max_attempts=max_attempts, hints=hints)

    return factory


class _IdentityBindingPolicy(LLMAgentPolicy):
    """An ``LLMAgentPolicy`` that binds grounded identities for a facts question.

    A model proposes *which action* to take.  ``GET_DETAILS`` and ``COMPARE`` additionally name
    products, and those identities come from the run's verified candidate set - the same
    projection, ``PolicyContext.grounded_parent_asins``, a policy already receives.  This small
    subclass reads them from the context and hands them to the model double, so the prompt
    payload itself stays free of product identities (one test asserts that) while the reasoning
    actions remain reachable.

    Nothing here widens authority: the identities are the run's own grounded candidates, and the
    validator and capability still refuse any identity that is not grounded.
    """

    def __init__(self, model: Any, *, max_attempts: int = 2, hints: dict[str, Any] | None = None) -> None:
        super().__init__(model, max_attempts=max_attempts)
        #: Case-derived expectations for the double.  A real model would infer these from the
        #: request; the double is told, and that is stated rather than hidden.
        self._hints = dict(hints or {})

    def choose(self, context: Any) -> Any:
        """Bind the case hints and delegate.

        It deliberately does **not** bind identities.  The double used to be handed the run's
        grounded set out of band, which meant a model could only ever emit "all of them" and
        trusted code effectively chose the target.  Legal targets now travel through the same
        payload a real model would read, so the model's choice is a real choice - and the
        executor refuses any identity that is not in the run's allowlist whatever the source.
        """
        model = self._model
        if isinstance(model, ObservationReactiveModel):
            model._hints = dict(self._hints)  # noqa: SLF001
        return super().choose(context)


# --------------------------------------------------------------------------- #
# Phase 3: the same policy seam, driven by a real provider
# --------------------------------------------------------------------------- #


def build_live_model_policy_factory(
    *,
    client: Any,
    max_attempts: int = 2,
    prompt_version: int | None = None,
    system_prompt: str | None = None,
) -> Callable[[EvaluationCase], Any]:
    """Return a per-case factory that builds an ``LLMAgentPolicy`` over a **real** client.

    Nothing here is provider-specific and nothing here relaxes a boundary: the factory builds the
    same :class:`~recommendation.control.model_policy.LLMAgentPolicy` the scripted variant builds,
    over the client the caller supplies.  What differs between the ``scripted_model`` and
    ``live_model`` variants is the client - never the policy, the runner, the validator or the
    trust boundary.  That is what makes the ablation a comparison of *policies* rather than of
    runtimes.

    ``system_prompt`` overrides the active contract, which is what an archived recording needs:
    a request fingerprint includes the prompt, so replaying a v1 trace while sending v2's text
    would be a different experiment.  Defaulting to the active prompt is correct for a current
    recording and wrong for an archived one, so the archived case must say so.

    ``client`` is typically a
    :class:`~recommendation.control.model_recorder.RecordingModelClient`, so the same factory
    serves a live run, a recording run and an offline replay: the client decides which of those
    is happening, and the policy cannot tell the difference.
    """
    from recommendation.control.model_policy import LLMAgentPolicy

    def factory(case: EvaluationCase) -> Any:
        return LLMAgentPolicy(
            client,
            max_attempts=max_attempts,
            name="live_agent_policy",
            system_prompt=system_prompt,
        )

    return factory
