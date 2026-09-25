"""The multi-source candidate plane (AgentRec-X 2.0-alpha Stage 2).

Stage 1 had one candidate source: the accepted history recommender, reached through
:class:`~recommendation.control.capability.RecommendFromHistoryCapability`.  Stage 2 adds
more, and this module is the boundary that keeps that safe.

The responsibility split is the whole point::

    AgentPolicy                     chooses WHICH source, and nothing else
        |
        v
    CandidatePlane (this module)    executes that source, grounds, records
        |-- history        -> accepted RecommendationTool -> SASRec
        |-- catalog_search -> CatalogSearchSource over the trusted catalogue
        `-- similar_item   -> registered tool, or unavailable
        |
        v
    GroundingVerifier               confirms every identity against trusted data
        |
        v
    CandidateLedger                 deduplicates identity, retains ALL provenance

A policy names a source; it cannot name a product, cannot supply a query for a source that
does not take one, and cannot reach the ledger.  Every identity a source returns is
grounded *before* it is recorded, and an identity that fails grounding is retained as an
audit record rather than silently dropped.

Why sources are executed one at a time
-------------------------------------
The prompt for this stage is explicit that retrieval must be *adaptive*, not a fixed
fan-out.  Structurally, that means one action consults one source.  A policy that wants two
sources proposes two actions, sees the first observation, and may change its mind - which
is the property the whole control plane exists to provide.  There is deliberately no
"call everything in parallel" method here: adding one would make the fixed fan-out the
easiest thing to do, and the adaptive path the awkward one.

The fast path survives: a single ``RECOMMEND_FROM_HISTORY`` action is one source, one
tool call, exactly as in Stage 1.

Unavailable sources
-------------------
A source the deployment does not configure is **absent**, not faked.  The plane reports it
as unavailable and the controller never offers it to the policy.  ``SIMILAR_ITEM`` in
particular has a clean seam here and is unavailable unless an item-item relationship index
is explicitly registered, because this repository's SASRec checkpoint is a sequential
item-ID model and does not itself provide item-item similarity.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from recommendation.tools import (
    MissingUserHistory,
    RecommendationContext,
    RecommendationTool,
    RecommendationToolRequest,
)

from .arguments import CandidateSource
from .candidate_ledger import CandidateLedger
from .capability import TrustedHistoryReader
from .catalog_search import CATALOG_SEARCH_SCORE_KIND, CatalogSearchSource
from .grounding import GroundingVerifier
from .schemas import (
    ActionKind,
    CandidateSetObservation,
    PolicyActionError,
    RecommendationDomainResult,
    ValidatedAction,
)

__all__ = [
    "CANDIDATE_PLANE_VERSION",
    "CandidatePlane",
    "CandidateSourceTool",
    "CandidateSourceUnavailable",
    "SourceRun",
]

#: Version of the candidate-plane contract.
CANDIDATE_PLANE_VERSION = 1


class CandidateSourceUnavailable(Exception):
    """A requested candidate source is not configured in this deployment.

    Raised rather than substituted: the plane will not silently answer a request for one
    source with another's results, because that would make the result's provenance a lie.
    """


@runtime_checkable
class CandidateSourceTool(Protocol):
    """A trusted tool that can produce candidate identities.

    Implementations are the history recommender, catalogue search, and any registered
    similar-item index.  A tool's job is to return candidates and their own ordering; it
    does **not** ground identities (the plane does that with the trusted verifier) and it
    does not touch the ledger.
    """

    @property
    def source(self) -> CandidateSource:
        """Which :class:`CandidateSource` this tool implements."""
        ...

    @property
    def score_kind(self) -> str:
        """What this tool's scores mean, e.g. ``bm25`` or ``sasrec_logit``."""
        ...

    def propose(self, *, arguments: Any, limit: int) -> list[tuple[str, int, float]]:
        """Return ``(parent_asin, rank, score)`` triples in this tool's own order."""
        ...


class SourceRun:
    """What one source execution produced, before and after grounding.

    Carried internally and summarised into observations.  The *domain result* keeps the
    grounded identities because the trusted renderer needs them; the *observation* keeps
    only counts, which is why the policy never sees a product identity.
    """

    __slots__ = ("source", "grounded", "ungrounded", "candidates", "query", "matched")

    def __init__(
        self,
        *,
        source: CandidateSource,
        grounded: frozenset[str],
        ungrounded: int,
        candidates: list[tuple[str, int, float | None]],
        query: str | None = None,
        matched: bool = True,
    ) -> None:
        self.source = source
        self.grounded = grounded
        self.ungrounded = ungrounded
        self.candidates = candidates
        self.query = query
        self.matched = matched

    @property
    def grounded_count(self) -> int:
        """How many identities this source contributed that were confirmed."""
        return len(self.grounded)


class _HistorySourceTool:
    """The accepted history recommender, adapted to the source-tool seam.

    This adapter is the migration boundary: it calls the **accepted**
    :class:`RecommendationTool` exactly as Stage 1's capability did - same request type,
    same trusted-context argument, same engine - so the recommender itself is untouched and
    Stage 1 behaviour is reproduced exactly.
    """

    def __init__(self, tool: RecommendationTool) -> None:
        self._tool = tool

    @property
    def source(self) -> CandidateSource:
        """The history source."""
        return CandidateSource.HISTORY

    @property
    def score_kind(self) -> str:
        """SASRec logits: ordering-only, never comparable across sources."""
        return "sasrec_logit"

    def propose(
        self,
        *,
        arguments: Any,
        limit: int,
        history: tuple[str, ...] = (),
    ) -> list[tuple[str, int, float]]:
        """Call the accepted Tool with trusted history and return its ranking."""
        if not history:
            raise MissingUserHistory(
                "the run has no trusted user history; supply it from application state"
            )
        request = RecommendationToolRequest(k=max(1, min(limit, 100)))
        context = RecommendationContext(user_history=history)
        result = self._tool.run(request=request, context=context)
        return [
            (item.parent_asin, item.rank, item.score) for item in result.recommendations
        ]


class _CatalogSearchSourceTool:
    """The full-catalogue lexical search, adapted to the source-tool seam."""

    def __init__(self, search: CatalogSearchSource) -> None:
        self._search = search

    @property
    def source(self) -> CandidateSource:
        """The catalogue-search source."""
        return CandidateSource.CATALOG_SEARCH

    @property
    def score_kind(self) -> str:
        """BM25: a lexical score over catalogue text, not a model score."""
        return CATALOG_SEARCH_SCORE_KIND

    def propose(
        self,
        *,
        arguments: Any,
        limit: int,
    ) -> list[tuple[str, int, float]]:
        """Run the query the arguments carry.  Blank queries yield zero candidates."""
        terms = tuple(getattr(arguments, "terms", ()) or ())
        if not terms:
            return []
        return self._search.candidates(" ".join(terms), limit=limit)


class CandidatePlane:
    """Execute one trusted candidate source per action, ground it, and record it.

    Parameters
    ----------
    ledger:
        The run's :class:`~recommendation.control.candidate_ledger.CandidateLedger`.
    grounding:
        The :class:`~recommendation.control.grounding.GroundingVerifier` that confirms
        identities.  Required: the plane will not record an ungrounded candidate.
    history_tool:
        The accepted Recommendation Tool, wrapped as the history source.
    catalog_search:
        Optional full-catalogue lexical search.
    similar_item_tool:
        Optional item-item similarity tool.  Absent unless explicitly registered, because
        this repository has no item-item index without a separate build step.
    tiger_tool:
        Optional TIGER-FP32 generative-retrieval tool.  Absent unless explicitly registered;
        Step 2.7 qualifies it as a source but does not wire it into the canonical runtime.
    """

    def __init__(
        self,
        *,
        ledger: CandidateLedger,
        grounding: GroundingVerifier,
        history_tool: RecommendationTool | None = None,
        catalog_search: CatalogSearchSource | None = None,
        similar_item_tool: CandidateSourceTool | None = None,
        two_tower_tool: CandidateSourceTool | None = None,
        tiger_tool: CandidateSourceTool | None = None,
    ) -> None:
        if (
            history_tool is None
            and catalog_search is None
            and similar_item_tool is None
            and two_tower_tool is None
            and tiger_tool is None
        ):
            raise PolicyActionError(
                "a candidate plane needs at least one trusted candidate source"
            )
        self._ledger = ledger
        self._grounding = grounding
        self._tools: dict[CandidateSource, Any] = {}
        if history_tool is not None:
            self._tools[CandidateSource.HISTORY] = _HistorySourceTool(history_tool)
        if catalog_search is not None:
            self._tools[CandidateSource.CATALOG_SEARCH] = _CatalogSearchSourceTool(
                catalog_search
            )
        if similar_item_tool is not None:
            if similar_item_tool.source is not CandidateSource.SIMILAR_ITEM:
                raise PolicyActionError(
                    "similar_item_tool must declare source=CandidateSource.SIMILAR_ITEM"
                )
            self._tools[CandidateSource.SIMILAR_ITEM] = similar_item_tool
        if two_tower_tool is not None:
            if two_tower_tool.source is not CandidateSource.TWO_TOWER:
                raise PolicyActionError(
                    "two_tower_tool must declare source=CandidateSource.TWO_TOWER"
                )
            self._tools[CandidateSource.TWO_TOWER] = two_tower_tool
        if tiger_tool is not None:
            if tiger_tool.source is not CandidateSource.TIGER:
                raise PolicyActionError(
                    "tiger_tool must declare source=CandidateSource.TIGER"
                )
            self._tools[CandidateSource.TIGER] = tiger_tool

    # -- metadata ---------------------------------------------------------- #

    @property
    def ledger(self) -> CandidateLedger:
        """The run's candidate ledger."""
        return self._ledger

    @property
    def grounding(self) -> GroundingVerifier:
        """The grounding verifier in use."""
        return self._grounding

    @property
    def version(self) -> int:
        """Candidate-plane contract version."""
        return CANDIDATE_PLANE_VERSION

    def available_sources(self) -> tuple[CandidateSource, ...]:
        """The trusted sources this deployment actually configures.

        This is what the controller offers the policy.  A source that is not here is
        *absent* - it is never offered, and a proposal naming it is refused.
        """
        return tuple(sorted(self._tools, key=lambda source: source.value))

    def has_source(self, source: CandidateSource) -> bool:
        """True when this deployment configures ``source``."""
        return source in self._tools

    def status(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the configured sources."""
        return {
            "version": self._version_tuple(),
            "sources": [
                {
                    "source": source.value,
                    "score_kind": self._tools[source].score_kind,
                }
                for source in self.available_sources()
            ],
            "grounding_checks_catalog": self._grounding.checks_catalog,
        }

    @staticmethod
    def _version_tuple() -> int:
        """Return the plane version (helper kept for a stable ``status`` shape)."""
        return CANDIDATE_PLANE_VERSION

    # -- execution --------------------------------------------------------- #

    def execute(
        self,
        action: ValidatedAction,
        *,
        read_trusted_history: TrustedHistoryReader,
        step_index: int = 0,
    ) -> RecommendationDomainResult:
        """Execute one candidate-source action and return a **raw, unverified** result.

        The result is not an observation and is never handed to the policy: it carries
        grounded identities for the trusted renderer.  Verification and minimisation happen
        in the verifier/adapter, exactly as in Stage 1.

        Raises
        ------
        CandidateSourceUnavailable
            The action names a source this deployment does not configure.
        PolicyActionError
            The action is not a candidate-producing action.
        """
        if action.action is ActionKind.RECOMMEND_FROM_HISTORY:
            source = CandidateSource.HISTORY
            limit = action.k
            arguments = None
        elif action.action is ActionKind.SELECT_SOURCE:
            arguments = action.arguments
            source = arguments.source
            limit = arguments.limit
        elif action.action is ActionKind.SEARCH_CATALOG:
            arguments = action.arguments
            source = CandidateSource.CATALOG_SEARCH
            limit = arguments.limit
        elif action.action is ActionKind.FIND_SIMILAR:
            arguments = action.arguments
            source = CandidateSource.SIMILAR_ITEM
            limit = arguments.limit
        else:
            raise PolicyActionError(
                f"'{action.action.value}' is not a candidate-producing action"
            )

        if source not in self._tools:
            raise CandidateSourceUnavailable(
                f"candidate source '{source.value}' is not configured in this deployment; "
                f"available: {', '.join(s.value for s in self.available_sources())}"
            )
        tool = self._tools[source]

        # -- execute the source -------------------------------------------- #
        try:
            if source in (CandidateSource.HISTORY, CandidateSource.TWO_TOWER,
                          CandidateSource.TIGER):
                # Identity-keyed sources are given the run's trusted history, which is the only
                # channel through which behaviour reaches them.  A source never supplies its
                # own history, so it cannot answer for a user it was not given.
                history = tuple(read_trusted_history())
                candidates = tool.propose(arguments=arguments, limit=limit, history=history)
            else:
                candidates = tool.propose(arguments=arguments, limit=limit)
        except CandidateSourceUnavailable:
            raise
        except Exception:
            # Source failures are not swallowed here: the accepted Tool's domain errors
            # (for example MissingUserHistory) must reach the caller unchanged so the loop
            # can classify them as recoverable or not.  Only this plane's own bookkeeping
            # happens below.
            raise

        # -- ground every identity before recording anything ---------------- #
        report = self._grounding.record(
            self._ledger,
            source=source,
            candidates=candidates,
            score_kind=tool.score_kind,
            source_query=self._query_for(source, arguments, tool),
            step_index=step_index,
        )

        grounded_candidates = [
            (parent_asin, rank, score)
            for parent_asin, rank, score in candidates
            if parent_asin in report.grounded
        ]
        candidate_set_ref = (
            f"{source.value}:{action.action_id}:{len(grounded_candidates)}"
        )
        grounded_identities = tuple(
            parent_asin for parent_asin, _, _ in grounded_candidates
        )
        return RecommendationDomainResult(
            action_id=action.action_id,
            status="empty" if not grounded_candidates else "ok",
            tool_result=None,
            requested_k=limit,
            returned_k=len(grounded_candidates),
            candidate_set_ref=candidate_set_ref,
            grounded_parent_asins=grounded_identities,
            item_ids=dict(report.item_ids),
            source=source.value,
            ungrounded_count=report.ungrounded_count,
            provenance=tuple(
                {
                    "parent_asin": parent_asin,
                    "source": source.value,
                    "rank": rank,
                    "score": score,
                    "score_kind": tool.score_kind,
                }
                for parent_asin, rank, score in grounded_candidates
            ),
        )

    # -- observation ------------------------------------------------------- #

    def observe(
        self,
        *,
        action: ValidatedAction,
        source: CandidateSource,
        requested_k: int,
        returned_k: int,
        ungrounded_count: int,
        step_index: int,
        candidate_set_ref: str,
    ) -> CandidateSetObservation:
        """Build the minimised observation for one candidate-source execution.

        Counts only.  This is the channel through which a policy learns that the source it
        chose found nothing (and should therefore try another), or that a source returned
        identities the trusted side rejects.
        """
        grounded_total = len(self._ledger)
        return CandidateSetObservation(
            action_id=action.action_id,
            step_index=step_index,
            action=action.action,
            verification_status="verified" if returned_k > 0 or not ungrounded_count else "refused",
            source=source.value,
            status="ok" if returned_k > 0 else "empty",
            requested_k=requested_k,
            returned_k=returned_k,
            has_candidates=returned_k > 0,
            candidate_set_ref=candidate_set_ref,
            sources_used=len(self._ledger.sources_present()),
            multi_source_count=len(self._ledger.multi_source_parent_asins()),
            ungrounded_count=ungrounded_count,
            verification_note=(
                f"{grounded_total} grounded candidate(s) in the ledger"
            ),
        )

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _query_for(source: CandidateSource, arguments: Any, tool: Any = None) -> str | None:
        """Return the (untrusted) query text a source was given, for provenance only.

        A source with no query text can still have provenance worth recording.  TIGER carries a
        retrieval *mode* and the artifact hashes that produced a candidate, and section 5 of the
        Step-2.7 contract requires any downstream consumer to be able to inspect the mode.  The
        shared provenance schema has no dedicated field for that, so a tool may expose a
        ``provenance_token()`` and it is recorded here.  No existing source defines one, so their
        recorded provenance is byte-identical to before.
        """
        if arguments is not None:
            terms = getattr(arguments, "terms", None)
            if terms:
                return " ".join(str(term) for term in terms)[:200]
        token_source = tool if tool is not None else None
        if token_source is not None and callable(getattr(token_source, "provenance_token", None)):
            try:
                token = token_source.provenance_token()
            except Exception:  # noqa: BLE001 - provenance must never fail a source call
                return None
            return str(token)[:200] if token else None
        return None
