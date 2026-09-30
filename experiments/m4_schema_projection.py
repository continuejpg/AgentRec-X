"""M4 provider-visible action-space projection.

Preregistration §7.3 restricts the adaptive action space **shown to the provider** to exactly
``SELECT_SOURCE`` and ``FINISH``.  This module implements that restriction as an injected model
client, so **no frozen file is modified**.

Why a client wrapper and not a loop change
------------------------------------------
Verified in the repository:

* ``model_policy.py:515`` computes ``action_schema = build_action_schema(available)`` **once**, puts
  it in the :class:`~recommendation.control.model_client.ModelRequest` at ``:524``, and parses the
  answer against the same ``available`` at ``:547``;
* ``ModelRequest`` is ``frozen=True`` (``model_client.py:76``) but a **replacement instance may be
  constructed**;
* ``LLMAgentPolicy`` accepts an **injected model** (``model_policy.py:381-388``) and calls it at
  exactly one site: ``self._model.complete(request)``.

Semantic placement: this is a **policy-level projection**, not a change to the control-plane menu.
The controller still offers whatever the frozen rule offers; the policy narrows the choice set it is
willing to disclose to the provider.

This projection alone is **not sufficient** -- the parser still knows the original available set --
which is why the post-response guard in :mod:`experiments.m4_policies` exists as well
(preregistration §8).
"""

from __future__ import annotations

from typing import Any, Iterable

from recommendation.control.arguments import CandidateSource
from recommendation.control.model_client import ModelRequest

__all__ = [
    "M4_PROVIDER_VISIBLE_ACTIONS",
    "ADAPTIVE_PROVIDER_ACTIONS",
    "SchemaProjectionClient",
    "project_action_schema",
]

#: The only actions M4's adaptive arm may show its provider (preregistration §7.3).
ADAPTIVE_PROVIDER_ACTIONS: tuple[str, ...] = ("select_source", "finish")

#: The four frozen sources, as the values the provider must be able to choose between. This is the
#: amendment-1 repair: the frozen universe was always these four, but ``build_action_schema``
#: emits only name/type/required, so the provider was never told the legal names and consistently
#: answered ``"default"``, which is not a member. Making the already-valid values visible changes
#: nothing about what may be selected -- only whether the provider can know it.
M4_SOURCE_VALUES: tuple[str, ...] = tuple(
    source.value for source in (
        CandidateSource.HISTORY,
        CandidateSource.CATALOG_SEARCH,
        CandidateSource.SIMILAR_ITEM,
        CandidateSource.TWO_TOWER,
    )
)

#: Defensive alias kept for tests and provenance records.
M4_PROVIDER_VISIBLE_ACTIONS = ADAPTIVE_PROVIDER_ACTIONS


def annotate_source_values(
    schema: Iterable[dict[str, Any]],
    values: tuple[str, ...] = M4_SOURCE_VALUES,
) -> tuple[dict[str, Any], ...]:
    """Add the closed value set to ``select_source``'s ``source`` argument.

    The trusted argument model already restricts ``source`` to :class:`CandidateSource` members;
    this only makes that existing restriction *visible* in the provider-facing contract. No new
    value is introduced and none is removed.

    Applied to a **copy** of each entry, so the caller's schema object is untouched.
    """
    allowed = list(values)
    annotated: list[dict[str, Any]] = []
    for entry in schema:
        copied = dict(entry)
        if str(copied.get("action")) == "select_source":
            arguments = []
            for argument in copied.get("arguments", ()) or ():
                item = dict(argument)
                if str(item.get("name")) == "source":
                    item["type"] = "string"
                    item["enum"] = allowed
                    item["allowed_values"] = allowed
                arguments.append(item)
            copied["arguments"] = arguments
        annotated.append(copied)
    return tuple(annotated)


def project_action_schema(
    schema: Iterable[dict[str, Any]],
    allowed: tuple[str, ...] = ADAPTIVE_PROVIDER_ACTIONS,
) -> tuple[dict[str, Any], ...]:
    """Filter an action schema down to ``allowed``, preserving the original order.

    The controller decides the order it offers actions in; this preserves it rather than imposing a
    new one, so the projected schema is a strict subsequence of what the system offered.
    """
    permitted = frozenset(allowed)
    filtered = tuple(
        dict(entry) for entry in schema if str(entry.get("action")) in permitted
    )
    # Amendment 1: make the already-valid source values visible to the provider. M4-scoped: this
    # runs inside the injected client, so no shared control-plane behaviour changes.
    return annotate_source_values(filtered)


class SchemaProjectionClient:
    """A :class:`~recommendation.control.model_client.StructuredModelClient` that narrows the
    provider-visible action schema before forwarding the call.

    Parameters
    ----------
    inner:
        The real client (normally the provider adapter). It is never asked to do anything else.
    allowed:
        The action names the provider may see. Defaults to the frozen M4 pair.
    on_projection:
        Optional callback invoked once per call with ``(original_names, projected_names)``, so a
        test or a provenance recorder can assert what the provider actually saw.
    """

    def __init__(
        self,
        inner: Any,
        *,
        allowed: tuple[str, ...] = ADAPTIVE_PROVIDER_ACTIONS,
        on_projection: Any = None,
    ) -> None:
        if not callable(getattr(inner, "complete", None)):
            raise TypeError("inner must provide a callable complete(request) method")
        self._inner = inner
        self._allowed = tuple(allowed)
        self._on_projection = on_projection
        #: The most recent projection, for assertions and provenance.
        self.last_projected: tuple[str, ...] = ()
        self.last_original: tuple[str, ...] = ()

    @property
    def allowed(self) -> tuple[str, ...]:
        """The action names this client discloses."""
        return self._allowed

    def complete(self, request: ModelRequest) -> Any:
        """Forward ``request`` with its action schema narrowed to :attr:`allowed`."""
        original = tuple(str(entry.get("action")) for entry in request.action_schema)
        projected = project_action_schema(request.action_schema, self._allowed)
        names = tuple(str(entry.get("action")) for entry in projected)
        self.last_original = original
        self.last_projected = names
        if self._on_projection is not None:
            self._on_projection(original, names)
        # ModelRequest is frozen, so a replacement instance is the only way to change it. Every
        # other field is carried through unchanged: this narrows disclosure, it does not rewrite
        # the policy contract or the context payload.
        narrowed = ModelRequest(
            system_prompt=request.system_prompt,
            context_payload=request.context_payload,
            action_schema=projected,
            correction=request.correction,
        )
        return self._inner.complete(narrowed)
