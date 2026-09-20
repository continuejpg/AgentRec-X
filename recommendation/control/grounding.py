"""Grounding: is a candidate identity one the trusted side actually knows?

This is the single most load-bearing check in the multi-source candidate plane.  Stage 2
lets a policy choose *where* candidates come from, and lets more than one tool produce
them; the corresponding risk is that a candidate identity is invented - by a buggy source, a
drifting index, or an adapter that fabricated a plausible-looking id.

The rule this module enforces:

> A product may be shown, ranked, compared or reasoned about **only** if its identity is
> present in the trusted item mapping, and - when a catalogue is configured - in the
> catalogue as well.

Three design points worth stating explicitly.

**Grounding is not scored.**  This module answers a yes/no question about identity.  It
never ranks, never scores and never selects: a grounded candidate is *eligible*, not
*preferred*.

**It fails closed.**  An identity that cannot be confirmed is reported as ungrounded.  The
ledger retains it as an audit record so a misbehaving source is visible in evaluation,
but it is never presented as a candidate.

**It reads no user data.**  Grounding looks at two things only: the trusted identity map and
the catalogue.  It has no access to behavioural history, memory or the request text, so it
cannot be steered by untrusted input.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .candidate_ledger import CandidateLedger

__all__ = [
    "GROUNDING_VERSION",
    "CatalogLike",
    "GroundingReport",
    "GroundingVerifier",
    "IdentityMapLike",
    "MappingOnlyCatalog",
    "UngroundedCandidate",
]

#: Version of the grounding contract.
GROUNDING_VERSION = 1


@runtime_checkable
class IdentityMapLike(Protocol):
    """The trusted identity mapping: canonical product identity <-> model item id.

    Satisfied by the accepted
    :class:`~recommendation.inference.sasrec.SASRecInferenceEngine` without modification.
    Declared structurally so the control plane does not depend on the inference package,
    and so tests can supply a small stand-in.
    """

    def has_parent_asin(self, parent_asin: str) -> bool:
        """True when the mapping knows this canonical product identity."""
        ...

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        """Return the model item id for a known identity (raises when unknown)."""
        ...

    def item_id_to_parent_asin(self, item_id: int) -> str:
        """Return the canonical identity for a known model item id."""
        ...


@runtime_checkable
class CatalogLike(Protocol):
    """The trusted catalogue: does a normalized metadata record exist for this identity?"""

    def __contains__(self, parent_asin: object) -> bool:
        """True when the catalogue holds a record for this identity."""
        ...


class UngroundedCandidate:
    """One identity a source returned that the trusted side does not know.

    Recorded rather than discarded.  A source that returns unknown identities is a *finding*
    - an evaluation signal about the source - and dropping it silently would hide the very
    behaviour the trust boundary exists to catch.
    """

    __slots__ = ("parent_asin", "source", "reason")

    def __init__(self, parent_asin: str, source: str, reason: str) -> None:
        self.parent_asin = parent_asin
        self.source = source
        self.reason = reason

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable view."""
        return {"parent_asin": self.parent_asin, "source": self.source, "reason": self.reason}

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"UngroundedCandidate({self.parent_asin!r}, {self.source!r}, {self.reason!r})"


class GroundingReport:
    """The outcome of grounding one source's candidate list.

    ``grounded`` is the set a capability may record in the ledger; ``ungrounded`` is the
    audit trail.  ``item_ids`` is the identity -> item id map the trusted mapping supplied,
    so downstream stages never have to re-derive it (and can never invent one).
    """

    __slots__ = ("grounded", "ungrounded", "item_ids")

    def __init__(
        self,
        *,
        grounded: frozenset[str],
        ungrounded: tuple[UngroundedCandidate, ...],
        item_ids: dict[str, int],
    ) -> None:
        self.grounded = grounded
        self.ungrounded = ungrounded
        self.item_ids = item_ids

    @property
    def grounded_count(self) -> int:
        """How many identities were confirmed."""
        return len(self.grounded)

    @property
    def ungrounded_count(self) -> int:
        """How many identities could not be confirmed."""
        return len(self.ungrounded)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (no identities beyond the audit list)."""
        return {
            "grounded": len(self.grounded),
            "ungrounded": len(self.ungrounded),
            "ungrounded_detail": [entry.as_dict() for entry in self.ungrounded],
        }


class GroundingVerifier:
    """Confirm candidate identities against the trusted mapping and catalogue.

    Parameters
    ----------
    identity_map:
        Any :class:`IdentityMapLike`.  Normally the accepted inference engine.
    catalog:
        Optional :class:`CatalogLike`.  When supplied, an identity must be in **both** the
        mapping and the catalogue.  When omitted, mapping membership is the whole check and
        that fact is reported, so a caller cannot mistake a weaker check for a stronger one.
    """

    def __init__(
        self,
        identity_map: IdentityMapLike,
        catalog: CatalogLike | None = None,
    ) -> None:
        if not isinstance(identity_map, IdentityMapLike):
            raise TypeError(
                "identity_map must provide has_parent_asin / parent_asin_to_item_id / "
                "item_id_to_parent_asin"
            )
        self._identity_map = identity_map
        self._catalog = catalog

    # -- metadata ---------------------------------------------------------- #

    @property
    def identity_map(self) -> IdentityMapLike:
        """The trusted identity mapping in use."""
        return self._identity_map

    @property
    def catalog(self) -> CatalogLike | None:
        """The trusted catalogue, when one was configured."""
        return self._catalog

    @property
    def checks_catalog(self) -> bool:
        """True when grounding also confirms catalogue presence."""
        return self._catalog is not None

    # -- grounding --------------------------------------------------------- #

    def ground(
        self,
        parent_asins: list[str],
    ) -> GroundingReport:
        """Classify ``parent_asins`` into grounded and ungrounded.

        Order is preserved only in the sense that the report is a set plus an audit list;
        the *source's* order is preserved by the caller, which passes its candidates to the
        ledger verbatim.
        """
        grounded: set[str] = set()
        item_ids: dict[str, int] = {}
        ungrounded: list[UngroundedCandidate] = []

        for raw in parent_asins:
            identity = raw.strip() if isinstance(raw, str) else ""
            if not identity:
                ungrounded.append(UngroundedCandidate(str(raw), "unknown", "blank_identity"))
                continue
            if not self._identity_map.has_parent_asin(identity):
                ungrounded.append(
                    UngroundedCandidate(identity, "identity_map", "not_in_item_mapping")
                )
                continue
            if self._catalog is not None and identity not in self._catalog:
                ungrounded.append(
                    UngroundedCandidate(identity, "catalog", "not_in_catalog")
                )
                continue
            try:
                item_ids[identity] = self._identity_map.parent_asin_to_item_id(identity)
            except Exception:  # noqa: BLE001 - a mapping that knows but cannot resolve
                ungrounded.append(
                    UngroundedCandidate(identity, "identity_map", "item_id_unresolvable")
                )
                continue
            grounded.add(identity)

        return GroundingReport(
            grounded=frozenset(grounded),
            ungrounded=tuple(ungrounded),
            item_ids=item_ids,
        )

    def is_grounded(self, parent_asin: str) -> bool:
        """Convenience single-identity check."""
        return self.ground([parent_asin]).grounded_count == 1

    # -- ledger integration ------------------------------------------------ #

    def record(
        self,
        ledger: CandidateLedger,
        *,
        source: Any,
        candidates: list[tuple[str, int, float | None]],
        score_kind: str = "unspecified",
        source_query: str | None = None,
        step_index: int = 0,
    ) -> GroundingReport:
        """Ground a source's candidates and record them in the ledger.

        This is the *only* supported way for a candidate-producing tool to reach the
        ledger: grounding and recording happen together, so there is no code path that
        records a candidate whose identity was not verified first.
        """
        report = self.ground([identity for identity, _, _ in candidates])
        ledger.record_from_source(
            source=source,
            candidates=candidates,
            score_kind=score_kind,
            item_ids=report.item_ids,
            source_query=source_query,
            step_index=step_index,
            grounded=report.grounded,
        )
        if report.ungrounded:
            ledger.record_ungrounded_identities(
                source=source,
                identities=[entry.parent_asin for entry in report.ungrounded],
                step_index=step_index,
            )
        return report


class MappingOnlyCatalog:
    """A :class:`CatalogLike` that treats the identity map as the catalogue.

    Used when a deployment has an item mapping but no metadata index.  It is an explicit,
    named adapter rather than an implicit fallback, so ``checks_catalog`` never overstates
    what was verified.
    """

    def __init__(self, identity_map: IdentityMapLike) -> None:
        self._identity_map = identity_map

    def __contains__(self, parent_asin: object) -> bool:
        """True when the mapping knows the identity."""
        return isinstance(parent_asin, str) and self._identity_map.has_parent_asin(parent_asin)
