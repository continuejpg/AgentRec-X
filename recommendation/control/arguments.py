"""Typed, per-action arguments for the control plane (AgentRec-X 2.0-alpha Stage 2+).

Stage 1 had a single model-facing argument, ``k``, carried directly on
:class:`~recommendation.control.schemas.ActionProposal`.  As the action space grows, one
flat field per action does not scale: it invites a proposal to carry an argument that
belongs to a *different* action, and it makes "which arguments does this action accept" a
matter of prose rather than of type.

This module replaces that with one frozen argument model per action, validated before the
proposal is even constructed.  What the design buys:

* **A proposal cannot carry a foreign argument.**  ``SELECT_SOURCE`` takes a
  ``candidate_source``; a proposal for ``FINISH`` has no field in which to put one, and
  ``extra="forbid"`` means an attempt is a hard validation error rather than a silent
  merge.
* **Arguments stay untrusted.**  These models constrain *shape*.  Whether a value is
  *authorised* - is this source registered, is this ``parent_asin`` real, is this product
  id allowed to be compared - is decided at the trusted boundary by
  :class:`~recommendation.control.validation.ActionValidator` and by the capability that
  executes the action.  Shape validation is not authorisation.
* **No free-form channel.**  There is deliberately no ``query: str``, no ``sql``, no
  ``item_ids`` and no ``parent_asins`` field anywhere in this module.  A policy composes a
  *structured* request (terms, sources, limits) and never a product identity.

The single exception worth naming is
:class:`CompareArguments`.  Comparing products requires naming them, so it accepts
``parent_asins``.  That is not a candidate-generation channel: the capability only compares
products it can ground in the trusted catalogue, and the comparison result carries no
candidate provenance into the ledger.  It cannot add a product to the candidate set.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from recommendation.tools.schemas import MAX_K, MIN_K

__all__ = [
    "ASK_QUESTION_MAX_LENGTH",
    "MAX_COMPARE_ITEMS",
    "MAX_SEARCH_TERMS",
    "AskClarificationArguments",
    "BundleArguments",
    "CandidateSource",
    "CheckCompatibilityArguments",
    "CompareArguments",
    "EmptyArguments",
    "GetDetailsArguments",
    "SearchCatalogArguments",
    "SelectSourceArguments",
    "TradeOffArguments",
]

#: Upper bound on a clarification question, so a question cannot become an essay prompt.
ASK_QUESTION_MAX_LENGTH = 300

#: Upper bound on a search query's term count.  A query is untrusted text used only to
#: select catalogue records; bounding it keeps one action from scanning the whole catalogue.
MAX_SEARCH_TERMS = 16

#: Upper bound on how many products one comparison may name.
MAX_COMPARE_ITEMS = 6


class CandidateSource(str, Enum):
    """A **trusted** candidate source, by name.

    This enum is the vocabulary in which a policy selects where candidates may come from.
    Membership is a statement that the repository has a tool for it - it is not a promise
    that the tool is configured in the running process.  Whether a source is actually
    available is computed by the controller and offered to the policy as
    ``available_actions``/``available_sources``; a proposal for an unregistered source is
    refused at the trusted boundary.
    """

    #: The accepted SASRec / history recommender, reached through the Recommendation Tool.
    HISTORY = "history"
    #: Full-catalogue lexical search over normalized catalogue text (title, brand,
    #: category, features, description).
    CATALOG_SEARCH = "catalog_search"
    #: Similar-item retrieval from a precomputed item-item relationship index.
    SIMILAR_ITEM = "similar_item"


#: Sources that produce candidate identities.  A source that is not listed here may still
#: be a trusted *evidence* source, but it can never add a candidate to the ledger.
CANDIDATE_PRODUCING_SOURCES: tuple[CandidateSource, ...] = (
    CandidateSource.HISTORY,
    CandidateSource.CATALOG_SEARCH,
    CandidateSource.SIMILAR_ITEM,
)


class EmptyArguments(BaseModel):
    """Arguments for an action that takes none.

    Used by ``FINISH`` and ``RECOMMEND_FROM_HISTORY`` (whose only parameter, ``k``, is a
    control-plane budget rather than a domain argument and stays on the proposal).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class SelectSourceArguments(BaseModel):
    """Arguments for ``SELECT_SOURCE``: which trusted candidate source to consult.

    The policy names a source and a result budget.  It does not name products, and it
    cannot supply a query - source-specific query text belongs to the action for that
    source, so that the authority to widen the scope is granted per action rather than
    smuggled through a generic selector.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: CandidateSource
    limit: int = Field(
        default=10,
        ge=MIN_K,
        le=MAX_K,
        description="How many candidates to ask this source for.",
    )


class SearchCatalogArguments(BaseModel):
    """Arguments for ``SEARCH_CATALOG``: a structured lexical query.

    ``terms`` is a list of short tokens/phrases, not a free-form query string.  The
    trusted search tool is what assembles and scores the query, so a policy can express
    *what* it is looking for without gaining any control over how the catalogue is
    matched.  Terms are stripped, whitespace-collapsed and bounded; a blank term is a
    validation error rather than being silently dropped.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    terms: tuple[str, ...] = Field(
        ...,
        min_length=1,
        max_length=MAX_SEARCH_TERMS,
        description="Short lexical search terms, at least one.",
    )
    limit: int = Field(default=10, ge=MIN_K, le=MAX_K)

    @classmethod
    def from_text(cls, text: str, *, limit: int = 10) -> SearchCatalogArguments:
        """Build arguments from a whitespace-separated phrase.

        A convenience for deterministic policies and tests.  It performs no semantic
        interpretation: the text is split, not understood.
        """
        terms = tuple(part for part in text.split() if part.strip())
        if not terms:
            raise ValueError("a catalog search needs at least one non-blank term")
        return cls(terms=terms[:MAX_SEARCH_TERMS], limit=limit)

    def query_text(self) -> str:
        """Return the terms joined as the search tool's query string."""
        return " ".join(self.terms)


class GetDetailsArguments(BaseModel):
    """Arguments for ``GET_DETAILS``: which already-grounded candidates to expand.

    ``parent_asins`` must name products that are already in the candidate ledger.  This
    action exists to fetch *more evidence about candidates the run already has*, not to
    introduce new ones; the capability refuses an unknown identity rather than looking it
    up in the catalogue.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asins: tuple[str, ...] = Field(..., min_length=1, max_length=MAX_COMPARE_ITEMS)


class CompareArguments(BaseModel):
    """Arguments for ``COMPARE``: which grounded candidates to compare.

    Naming products is necessary to compare them, so this is the one action that accepts
    ``parent_asins``.  It is still not a candidate-generation channel: comparison reads
    facts about products the run already holds, and its result carries no candidate
    provenance.  The capability grounds every identity against the trusted catalogue and
    reports ``UNKNOWN`` for any fact the catalogue does not contain.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asins: tuple[str, ...] = Field(..., min_length=2, max_length=MAX_COMPARE_ITEMS)
    #: Which catalogue attributes to compare.  Empty means "the comparison defaults".
    attributes: tuple[str, ...] = Field(default=(), max_length=8)


class TradeOffArguments(BaseModel):
    """Arguments for ``TRADE_OFF``: the priority the trade-off should reason about.

    A trade-off is a comparison under a stated priority ("lighter matters more than
    cheaper").  The priority is a short label the policy proposes; the capability maps it
    onto grounded attributes and reports ``UNKNOWN`` when the catalogue cannot support it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asins: tuple[str, ...] = Field(..., min_length=2, max_length=MAX_COMPARE_ITEMS)
    priority: str = Field(..., min_length=1, max_length=80)


class CheckCompatibilityArguments(BaseModel):
    """Arguments for ``CHECK_COMPATIBILITY``: which products to test against each other.

    Compatibility needs a *requirement* as well as products, and the requirement must come
    from a trusted source: an explicit user statement, a stored preference, or a
    catalogue fact.  The policy therefore proposes only the products and the requirement
    text; the capability resolves that requirement into deterministic checks and returns
    ``UNKNOWN`` for any requirement it cannot ground.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asins: tuple[str, ...] = Field(..., min_length=1, max_length=MAX_COMPARE_ITEMS)
    requirement: str = Field(..., min_length=1, max_length=160)


class BundleArguments(BaseModel):
    """Arguments for ``BUNDLE``: which grounded candidates to consider as a set.

    A bundle is a set, not a list of individually good items, so the capability checks
    individual feasibility, cross-item compatibility and aggregate constraints.  No
    transactional concept (cart, order, purchase) exists anywhere in this type or its
    result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asins: tuple[str, ...] = Field(..., min_length=2, max_length=MAX_COMPARE_ITEMS)
    #: What the bundle is for, as a short label.  Used to select which aggregate
    #: constraints are worth checking; never treated as a product fact.
    purpose: str | None = Field(default=None, max_length=120)


class AskClarificationArguments(BaseModel):
    """Arguments for ``ASK_CLARIFICATION``: the question and why it is being asked.

    Clarification is expensive - it ends the turn and waits for a human - so the proposal
    must state which decision the answer would change.  ``blocks`` is the policy's
    *claim* about what is blocked; the trusted side records it for the trajectory and for
    evaluation, and does not treat it as a fact about the catalogue.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(..., min_length=1, max_length=ASK_QUESTION_MAX_LENGTH)
    #: The decision the answer would change, e.g. ``"candidate_set"`` or ``"ranking"``.
    blocks: str = Field(default="candidate_set", max_length=40)
    #: Optional short list of the options the user is being offered, when the question is
    #: a choice rather than open-ended.  Presentation only; never a product identity.
    options: tuple[str, ...] = Field(default=(), max_length=6)
