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
* **No free-form channel.**  There is deliberately no ``query: str`` and no ``sql``.  A
  policy composes *structured* requests (terms, sources, limits, requirements), never a raw
  expression, and no argument model has a field a tool would execute verbatim.

Several models do accept ``parent_asins`` - :class:`GetDetailsArguments`,
:class:`CompareArguments`, :class:`TradeOffArguments`,
:class:`CheckCompatibilityArguments` and :class:`BundleArguments` - because inspecting and
comparing products requires naming them.  The action kind decides who validates it:

* a **candidate-producing** action has no identity field at all, which is what keeps
  identity introduction in the hands of trusted candidate sources;
* a **reasoning** action may name identities, and every one of them must be a member of the
  run's own grounded candidate set.  That membership check is enforced by
  :class:`~recommendation.control.reasoning_executor.ReasoningExecutor` *before* the
  catalogue is consulted, so naming a product the run does not hold fails closed.  Catalogue
  existence is necessary for grounding and is never sufficient for authority.

Reasoning results carry no candidate provenance into the ledger, so no argument model here
is a candidate-generation channel.
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
    #: Two-Tower / DSSM dual-encoder retrieval (post-Phase-5 model expansion).  A distinct
    #: member rather than a reuse of ``HISTORY`` because the two produce different score kinds
    #: (a learned cosine versus a SASRec logit) and the ledger must be able to attribute a
    #: candidate to the model that actually proposed it.
    TWO_TOWER = "two_tower"


#: Sources that produce candidate identities.  A source that is not listed here may still
#: be a trusted *evidence* source, but it can never add a candidate to the ledger.
CANDIDATE_PRODUCING_SOURCES: tuple[CandidateSource, ...] = (
    CandidateSource.HISTORY,
    CandidateSource.CATALOG_SEARCH,
    CandidateSource.SIMILAR_ITEM,
    # Post-Phase-5 model expansion.  A source becomes "trusted" by being listed here *and* by
    # having a registered tool that returns identities the grounding verifier confirms; the
    # ledger refuses an unlisted source so a typo or a spoofed name cannot write provenance.
    CandidateSource.TWO_TOWER,
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

    ``parent_asins`` must name products that are already grounded candidates of the current
    run - not merely products that exist in the catalogue.  This action exists to fetch *more
    evidence about candidates the run already has*, not to introduce or browse new ones, and
    the trusted executor enforces that literally: a non-member identity is refused before any
    catalogue lookup happens, so this cannot become a catalogue browser.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asins: tuple[str, ...] = Field(..., min_length=1, max_length=MAX_COMPARE_ITEMS)


class CompareArguments(BaseModel):
    """Arguments for ``COMPARE``: which grounded candidates to compare.

    Naming products is necessary to compare them.  It is still not a candidate-generation
    channel: comparison reads facts about candidates the run already holds, and its result
    carries no candidate provenance.

    **Every** named identity is checked against the run's grounded allowlist, and the check is
    all-or-nothing: a comparison naming one candidate the run holds and one it does not is
    refused whole, rather than answered partially.  A partial answer would leak which of the
    named identities the run happens to hold, and would answer a question about a product the
    run has no authority over.  Within the run, a comparison also reports ``UNKNOWN`` for any
    attribute the catalogue does not contain.
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


class FindSimilarArguments(BaseModel):
    """Arguments for ``FIND_SIMILAR``: which already-grounded candidate to find neighbours of.

    ``FIND_SIMILAR`` used to reuse :class:`SelectSourceArguments`, which carries only a source and
    a limit - so the action could not say *similar to what*, and an implementation would have had
    to pick a seed itself.  Phase 4 gives the model the choice explicitly, because choosing the
    target is exactly the authority Phase 2.1 established the model should have.

    The seed is validated against the run's own grounded candidate allowlist **before** any
    retrieval happens, so a model can select a seed but cannot introduce one.  It names a single
    product: similarity is defined from one item, and a multi-seed list would be a different
    (and unimplemented) query.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed_parent_asin: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="A candidate the run already holds, to find similar products for.",
    )
    limit: int = Field(default=10, ge=MIN_K, le=MAX_K)


class AcquireEvidenceArguments(BaseModel):
    """Arguments for ``ACQUIRE_EVIDENCE``: which held candidate needs which fact.

    The model names a **product the run already holds** and an attribute the trusted evidence
    source models.  It cannot name a value: the observation carries what the source stated, never
    what the model expected.  Producing an :class:`EvidenceItem` is trusted code's job, and
    ``EvidenceItem`` has no verdict field for a policy to influence.

    ``attribute`` is validated against the trusted source's own declared surface rather than
    accepted as free text, so a policy cannot probe for attributes no source models.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: str = Field(..., min_length=1, max_length=120)
    attribute: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description=(
            "Which fact to acquire, e.g. 'material' or 'feature'. Must be one the configured "
            "evidence source models."
        ),
    )


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
