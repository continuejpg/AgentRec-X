"""The trusted evidence executor (Phase 4).

Maps a validated ``ACQUIRE_EVIDENCE`` action onto a trusted
:class:`~recommendation.control.evidence.EvidenceSource` and shapes the result into a
policy-visible observation.

The authority boundary this module exists to hold
-------------------------------------------------
The model chooses **which held candidate needs which fact**.  It cannot choose what the fact is,
cannot name an attribute no configured source models, cannot name a product the run does not hold,
and cannot produce a verdict.  Concretely:

* the request is one identity plus one attribute name, validated against the sources' own
  declared surface - never free text;
* the identity must be a grounded candidate of the current run, checked **before** any source is
  consulted, exactly as the reasoning executor checks its targets;
* what comes back is an :class:`~recommendation.control.evidence.EvidenceItem`, which has no
  verdict field.  Only
  :class:`~recommendation.control.grounded_reasoning.GroundedReasoner` turns evidence into
  ``SATISFIED`` / ``VIOLATED`` / ``UNKNOWN``, through the same Phase-2 kernel as before.

Acquiring evidence also cannot widen the candidate set: this executor has no ledger, no plane and
no catalogue access, so it cannot add a candidate even if asked.
"""

from __future__ import annotations

from typing import Any, Sequence

from .evidence import (
    EvidenceDocument,
    EvidenceItem,
    EvidenceStatus,
    EvidenceStore,
)
from .schemas import (
    AcquireEvidenceArguments,
    ActionKind,
    EvidenceObservation,
    PolicyActionError,
    ValidatedAction,
)

__all__ = ["EvidenceExecutor", "EVIDENCE_EXECUTOR_ACTIONS"]

#: The actions this executor is authorised to run.  Derived from the contract, so a new
#: evidence action cannot be added without this executor's authority set moving with it.
EVIDENCE_EXECUTOR_ACTIONS: frozenset[ActionKind] = frozenset({ActionKind.ACQUIRE_EVIDENCE})


class EvidenceExecutor:
    """Execute one evidence acquisition and return a policy-visible observation.

    Parameters
    ----------
    sources:
        The trusted evidence sources, in the order they should be consulted.  Required: without a
        source there is no evidence to acquire, and this executor will not answer from anywhere
        else.
    store:
        The overlay the acquired facts enter.  Required, because evidence that is read but not
        recorded could not be re-evaluated by the constraint kernel - the whole point of the
        capability is that the fact becomes part of trusted state.
    """

    def __init__(self, sources: Sequence[Any], store: EvidenceStore) -> None:
        if not sources:
            raise PolicyActionError(
                "an evidence executor needs at least one trusted evidence source"
            )
        if not isinstance(store, EvidenceStore):
            raise PolicyActionError("an evidence executor needs an EvidenceStore")
        self._sources = tuple(sources)
        self._store = store
        self._executed = 0

    # -- metadata ---------------------------------------------------------- #

    @property
    def store(self) -> EvidenceStore:
        """The overlay acquired facts enter."""
        return self._store

    @property
    def sources(self) -> tuple[Any, ...]:
        """The trusted sources, in consultation order."""
        return self._sources

    @property
    def executed_count(self) -> int:
        """How many acquisitions have run (diagnostics only)."""
        return self._executed

    @property
    def source_names(self) -> tuple[str, ...]:
        """Stable names of the configured sources, for provenance and diagnostics."""
        return tuple(str(source.name) for source in self._sources)

    def supported_attributes(self) -> tuple[str, ...]:
        """The attributes a policy may legally ask for.

        Read from the sources rather than from the vocabulary, so the menu never advertises a
        question no configured source can answer.  A source that does not declare a surface is
        treated as modelling nothing, which is the fail-closed reading.
        """
        declared: list[str] = []
        for source in self._sources:
            for attribute in getattr(source, "attributes", ()) or ():
                if attribute not in declared:
                    declared.append(str(attribute))
        return tuple(declared)

    def handles(self, action: ActionKind) -> bool:
        """True when this executor is authorised to run ``action``."""
        return action in EVIDENCE_EXECUTOR_ACTIONS

    def available_actions(self) -> tuple[ActionKind, ...]:
        """The evidence actions this deployment offers."""
        return tuple(sorted(EVIDENCE_EXECUTOR_ACTIONS, key=lambda a: a.value))

    # -- execution --------------------------------------------------------- #

    def execute(
        self, action: ValidatedAction, *, authorized_candidates: Sequence[str] | None = None
    ) -> EvidenceObservation:
        """Acquire one fact about one held candidate.

        Parameters
        ----------
        action:
            The validated ``ACQUIRE_EVIDENCE`` action.
        authorized_candidates:
            The run's live grounded allowlist.  Supplied by the loop on every step; ``None`` means
            the caller asserts there is no run to scope to, which only a direct executor test
            should do.

        Raises
        ------
        PolicyActionError
            The action is not an evidence action; the identity is not a grounded candidate of this
            run; or the attribute is not one a configured source models.  Each is a refusal with a
            stable code, never a repair.
        """
        if action.action not in EVIDENCE_EXECUTOR_ACTIONS:
            raise PolicyActionError(
                f"'{action.action.value}' is not an evidence action; "
                f"allowed: {', '.join(a.value for a in self.available_actions())}"
            )
        arguments = action.arguments
        if not isinstance(arguments, AcquireEvidenceArguments):  # pragma: no cover - validator
            raise PolicyActionError(
                "acquire_evidence requires AcquireEvidenceArguments",
                code="invalid_arguments",
            )
        identity = arguments.parent_asin
        attribute = arguments.attribute

        # Authority before evidence, in the same order the reasoning executor uses: the run
        # boundary is checked before any source is consulted, so an unauthorized identity never
        # causes a read.
        self._require_run_membership(identity, authorized_candidates)
        # The attribute is checked against the sources' declared surface rather than accepted as
        # free text, so a policy cannot probe for something no source models.
        supported = self.supported_attributes()
        if supported and attribute not in supported:
            raise PolicyActionError(
                f"no configured evidence source models {attribute!r}; "
                f"available: {', '.join(supported)}",
                code="unsupported_evidence_attribute",
            )
        # ``attribute`` is the *question the policy is asking* and is validated above; the source
        # answers with everything it holds for the product, because one consultation is one tool
        # call and splitting it per attribute would charge the budget for a bookkeeping detail.
        # The observation reports which attributes arrived, so the policy still learns what its
        # question returned.

        document = self._consult(identity)
        newly_acquired = self._store.absorb(document)
        self._executed += 1
        return self._observation(
            action=action,
            document=document,
            requested_attribute=attribute,
            newly_acquired=newly_acquired,
        )

    def _require_run_membership(
        self, identity: str, authorized_candidates: Sequence[str] | None
    ) -> None:
        """Refuse an identity that is not a grounded candidate of this run."""
        if authorized_candidates is None:
            return
        if identity not in frozenset(authorized_candidates):
            raise PolicyActionError(
                "the requested identity is not a grounded candidate of this run",
                code="candidate_not_in_run",
            )

    def _consult(self, identity: str) -> EvidenceDocument:
        """Ask each trusted source in order, stopping at the first that states a fact.

        Stopping at the first ``FOUND`` is deliberate: the sources are a precedence order, not a
        vote.  Their statements are still all retained, so corroboration from a later source remains
        available if a deployment consults them all - this executor simply does not need to.
        """
        best: EvidenceDocument | None = None
        for source in self._sources:
            document = source.read(identity)
            if document.status is EvidenceStatus.FOUND:
                return document
            # Keep the most informative miss: "cannot answer" is a stronger statement than
            # "consulted and empty", and the policy should be told the stronger one.
            if best is None or document.status is EvidenceStatus.UNSUPPORTED:
                best = document
        if best is None:  # pragma: no cover - guarded by the constructor
            raise PolicyActionError("no evidence source was available", code="evidence_unavailable")
        return best

    def _observation(
        self,
        *,
        action: ValidatedAction,
        document: EvidenceDocument,
        requested_attribute: str,
        newly_acquired: tuple[str, ...],
    ) -> EvidenceObservation:
        """Shape the acquisition into a bounded, payload-light observation."""
        found = document.status is EvidenceStatus.FOUND
        return EvidenceObservation(
            action_id=action.action_id,
            step_index=action.step_index,
            action=action.action,
            verification_status="verified",
            status=document.status.value,
            parent_asin=document.parent_asin,
            requested_attribute=requested_attribute,
            source=document.source,
            found=found,
            attribute_count=len(document.items),
            # Attribute names, not values: the fact belongs to the constraint report, where it is
            # attached to the verdict it produced.  A trajectory says evidence arrived and what it
            # covered.
            acquired_attributes=tuple(item.attribute for item in document.items),
            newly_acquired_attributes=newly_acquired,
            provenance_note=document.note,
            evidence_state=self._store.as_dict(),
        )
