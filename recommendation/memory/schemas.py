"""Memory schemas (Milestone 9).

Two memory domains, deliberately kept apart
-------------------------------------------
**Trusted interaction memory** is the chronological behavioural history that SASRec
consumes: a sequence of Amazon ``parent_asin`` values owned by the application.  It is
*not* modelled here.  It stays in the graph/application input state, reaches
:class:`~recommendation.tools.RecommendationTool` through the accepted
:class:`~recommendation.tools.schemas.RecommendationContext`, and no type in this
module has a field for it.  There is therefore no way to write an interaction event
through the memory API, and no way for conversational text to become one.

**Preference memory** is conversational state derived *only* from explicit
user-authored statements ("I prefer black", "I don't want red", "my budget is under
$100").  It is user-scoped, typed, attributable to a source turn, mutable, supersedable
and removable.

The two domains are represented by different types in different packages and are never
collapsed into one free-form profile.

Extraction output is untrusted
------------------------------
:class:`PreferenceCandidate` is what an extractor *proposes*.  It is not what gets
stored: every candidate is validated against strict schemas (``extra="forbid"``) and
converted into a :class:`PreferenceMemoryEntry` by the memory service.  A candidate has
no field for an interaction event, an item id, a SASRec score, a candidate list, a file
path or a memory id, so a hostile or buggy extractor has nowhere to put those things.
"""

from __future__ import annotations

import re
import time
from enum import Enum
from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "PREFERENCE_POLARITIES",
    "SECRET_REDACTION",
    "PreferenceCandidate",
    "PreferenceExtraction",
    "PreferenceExtractionError",
    "PreferenceExtractor",
    "PreferenceKind",
    "PreferenceMemoryEntry",
    "PreferenceMemorySnapshot",
    "PreferenceMode",
    "PreferencePolarity",
    "PreferenceRemoval",
    "PreferenceStatus",
    "contains_secret_like_text",
]

#: Version of the preference-memory schema.  Recorded in the SQLite database's
#: ``meta`` table and in every artifact, so a future migration is explicit.
MEMORY_SCHEMA_VERSION = 1

#: Text stored in place of a value that looks like a credential.
SECRET_REDACTION = "[redacted]"

#: Accepted polarity values.
PREFERENCE_POLARITIES: tuple[str, ...] = ("prefer", "avoid")

#: A non-empty, whitespace-normalised identifier or value.
NonEmptyText = Annotated[str, Field(min_length=1)]

#: Maximum length accepted for a preference value.  Keeps conversational noise and
#: pasted documents out of the store without truncating a genuine preference.
MAX_VALUE_LENGTH = 120

#: Maximum length accepted for a provenance span.
MAX_SOURCE_TEXT_LENGTH = 500


class PreferenceKind(str, Enum):
    """The deliberately narrow preference ontology.

    This is not a user-profile schema.  Each kind exists because a product-shopping
    constraint needs it.  An explicit statement that does not fit any of them is stored
    as :attr:`FREE_FORM_CONSTRAINT` rather than being forced into a wrong kind.
    """

    CATEGORY = "category"
    FEATURE = "feature"
    BRAND = "brand"
    PRICE_MAX = "price_max"
    PRICE_MIN = "price_min"
    COLOR = "color"
    MATERIAL = "material"
    FREE_FORM_CONSTRAINT = "free_form_constraint"


class PreferencePolarity(str, Enum):
    """Whether the user wants or rejects something."""

    PREFER = "prefer"
    AVOID = "avoid"


class PreferenceMode(str, Enum):
    """How an explicit statement relates to what is already stored.

    The distinction exists because *stating another preference* is not the same as
    *replacing one*.  "I don't want red" followed by "I don't want blue" adds a second
    independent avoidance; only an explicit correction ("...instead", "actually...")
    retracts the earlier one.

    * :attr:`ADD` -- add an independent constraint; nothing already stored is retracted
      (the default, and the conservative choice).
    * :attr:`REPLACE` -- the statement corrects an earlier one, so conflicting entries
      of the same kind and polarity are superseded.  ``replaces`` may name the exact
      value being corrected.
    * :attr:`REMOVE` -- the statement retracts constraints rather than adding one; the
      candidate is converted into a removal directive and stores no new value.
    """

    ADD = "add"
    REPLACE = "replace"
    REMOVE = "remove"


class PreferenceStatus(str, Enum):
    """Lifecycle status of a stored entry.

    ``SUPERSEDED`` and ``REMOVED`` records are retained so provenance and audit
    history survive; they are simply not active.
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REMOVED = "removed"


#: Shortest secret-looking token worth redacting, to avoid matching ordinary words.
_MIN_SECRET_LENGTH = 12

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # A single long run mixing letters and digits: API keys, tokens, hashes.
    # Requiring both a letter and a digit avoids flagging hyphenated phrases such as
    # "nothing-like-this", which are ordinary preference values.
    re.compile(
        rf"\b(?=[A-Za-z0-9_\-]{{{_MIN_SECRET_LENGTH},}}\b)"
        r"(?=[^\s]*[A-Za-z])(?=[^\s]*\d)[A-Za-z0-9_\-]+\b"
    ),
    # A long run of upper-case letters and digits (AWS-style access keys).
    re.compile(rf"\b(?=[A-Z0-9]{{{_MIN_SECRET_LENGTH},}}\b)[A-Z0-9]+\b"),
    # Card-like digit groups, separated by spaces or dashes.
    re.compile(r"\b\d{4}(?:[ -]\d{4}){3}\b"),
    # Well-known credential prefixes.
    re.compile(r"\b(?:sk|pk|rk|ghp|gho|xox[baprs])[-_][A-Za-z0-9]{8,}\b", re.IGNORECASE),
    # key=value / key: value assignment of a credential-ish name.
    re.compile(
        r"\b(?:password|passwd|secret|api[_-]?key|token|bearer|authorization)\b\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
)


def contains_secret_like_text(text: str) -> bool:
    """True when ``text`` looks like a credential rather than a shopping preference.

    M9 is not a conversation archive: preference extraction must not persist passwords,
    API keys, card numbers or tokens.  The check is conservative and deliberately
    simple -- it errs toward refusing to store a value.
    """
    if not isinstance(text, str) or not text:
        return False
    return any(pattern.search(text) is not None for pattern in _SECRET_PATTERNS)


def _clean_value(value: object) -> str:
    """Normalise a preference value: strip, collapse whitespace, reject blanks."""
    if not isinstance(value, str):
        raise ValueError("preference value must be a string")
    collapsed = " ".join(value.split())
    if not collapsed:
        raise ValueError("preference value must not be blank")
    if len(collapsed) > MAX_VALUE_LENGTH:
        raise ValueError(
            f"preference value must be at most {MAX_VALUE_LENGTH} characters"
        )
    if contains_secret_like_text(collapsed):
        raise ValueError(
            "preference value looks like a credential and must not be stored"
        )
    return collapsed


def _clean_source_text(value: object) -> str:
    """Normalise a provenance span."""
    if not isinstance(value, str):
        raise ValueError("source_text must be a string")
    collapsed = " ".join(value.split())
    if not collapsed:
        raise ValueError("source_text must not be blank")
    return collapsed[:MAX_SOURCE_TEXT_LENGTH]


class PreferenceCandidate(BaseModel):
    """One preference proposed by an extractor, before validation and storage.

    This is **untrusted** input.  It carries no memory id, no logical sequence, no
    status and no interaction-event field, so an extractor cannot address an existing
    record, forge ordering, or emit a behavioural event.  ``extra="forbid"`` means an
    attempt to add such a field is a hard validation error.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PreferenceKind
    value: NonEmptyText
    polarity: PreferencePolarity = PreferencePolarity.PREFER
    #: Exact span of the user's text that supports this preference.  Required, so an
    #: unsupported paraphrase can never be stored without its provenance.
    source_text: NonEmptyText
    #: Optional label indicating which extractor produced the candidate.
    extractor: NonEmptyText = "unspecified"
    #: How this statement relates to existing memory.  Defaults to ``ADD`` so an
    #: extractor must *state* correction intent to retract anything.
    mode: PreferenceMode = PreferenceMode.ADD
    #: Optional value this statement explicitly corrects.  Only meaningful for
    #: ``REPLACE``; restricts the supersession to entries holding that value.
    replaces: str | None = None

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: object) -> str:
        return _clean_value(value)

    @field_validator("source_text")
    @classmethod
    def _validate_source_text(cls, value: object) -> str:
        return _clean_source_text(value)

    @field_validator("replaces")
    @classmethod
    def _validate_replaces(cls, value: object) -> str | None:
        if value is None:
            return None
        return _clean_value(value)

    def model_post_init(self, __context: Any) -> None:
        """Keep ``replaces`` meaningful: it only applies to a replacement."""
        if self.replaces is not None and self.mode is not PreferenceMode.REPLACE:
            raise ValueError("'replaces' is only valid when mode is 'replace'")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class PreferenceRemoval(BaseModel):
    """An explicit instruction to stop applying a preference.

    A removal is distinct from an avoidance: *"I don't want red"* adds an active
    negative constraint, while *"I don't care about color anymore"* retracts the
    colour constraint entirely.  Removing produces no new active entry -- it tombstones
    matching entries so their provenance survives in the audit trail.

    Exactly one of ``kind`` / ``value`` must be supplied.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PreferenceKind | None = None
    value: str | None = None
    source_text: NonEmptyText
    extractor: NonEmptyText = "unspecified"
    #: Fixed to ``REMOVE`` so a retraction is self-describing in the extraction payload.
    mode: PreferenceMode = PreferenceMode.REMOVE

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: object) -> str | None:
        if value is None:
            return None
        return _clean_value(value)

    @field_validator("source_text")
    @classmethod
    def _validate_source_text(cls, value: object) -> str:
        return _clean_source_text(value)

    def model_post_init(self, __context: Any) -> None:
        """Require exactly one selector so a removal is never ambiguous."""
        if (self.kind is None) == (self.value is None):
            raise ValueError(
                "a removal must specify exactly one of 'kind' or 'value'"
            )
        if self.mode is not PreferenceMode.REMOVE:
            raise ValueError("a removal must have mode 'remove'")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class PreferenceExtraction(BaseModel):
    """Everything one extractor observed in a single user turn.

    Separating additions from removals keeps the "no new active constraint" case
    explicit: a retraction is not stored as a junk preference, it is stored as a
    tombstone on the entries it actually retracts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    preferences: tuple[PreferenceCandidate, ...] = ()
    removals: tuple[PreferenceRemoval, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when the turn produced neither an addition nor a removal."""
        return not self.preferences and not self.removals

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class PreferenceExtractionError(Exception):
    """An extractor produced output that violates the extraction contract."""


@runtime_checkable
class PreferenceExtractor(Protocol):
    """The narrow extraction seam.

    Implementations must be deterministic and offline.  Milestone 9 ships a
    conservative rule-based extractor and uses scripted fakes in tests; a future LLM
    adapter can implement this same method without changing the store or service
    contracts.

    ``extract`` is called **only** with user-authored text.  It is never given the
    trusted interaction history, recommendation candidates, catalogue metadata,
    retrieval evidence or model output, so preference memory cannot be derived from
    them.
    """

    def extract(self, user_message: str) -> PreferenceExtraction:
        """Return the explicit preferences and retractions stated in the turn."""
        ...


class PreferenceMemoryEntry(BaseModel):
    """One stored preference, with full provenance and lifecycle state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: NonEmptyText = Field(..., description="Stable unique id for this entry.")
    user_key: NonEmptyText = Field(..., description="Owning user/session key.")
    kind: PreferenceKind
    value: NonEmptyText
    polarity: PreferencePolarity = PreferencePolarity.PREFER
    source_text: NonEmptyText = Field(
        ..., description="Exact span of the user's message that supports this entry."
    )
    source_turn_id: NonEmptyText = Field(
        ..., description="Identifier of the user turn this entry came from."
    )
    extractor: NonEmptyText = "unspecified"
    status: PreferenceStatus = PreferenceStatus.ACTIVE
    logical_seq: int = Field(
        ..., ge=1, description="Monotonic per-user sequence; the stable sort key."
    )
    created_at: float = Field(..., description="Unix timestamp of creation.")
    supersedes: str | None = Field(
        default=None, description="memory_id of the entry this one replaced, if any."
    )
    superseded_by: str | None = Field(
        default=None, description="memory_id of the entry that replaced this one, if any."
    )

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: object) -> str:
        return _clean_value(value)

    @field_validator("source_text")
    @classmethod
    def _validate_source_text(cls, value: object) -> str:
        return _clean_source_text(value)

    @property
    def is_active(self) -> bool:
        """True when this entry is an active constraint."""
        return self.status is PreferenceStatus.ACTIVE

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class PreferenceMemorySnapshot(BaseModel):
    """An immutable, deterministically ordered view of a user's memory.

    Produced by a read; contains only the entries requested (active only, or the full
    audit trail).  A snapshot is what the Agent consumes, so a later write cannot
    change a decision already taken during the same turn.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_key: NonEmptyText
    entries: tuple[PreferenceMemoryEntry, ...] = ()
    read_at: float = Field(default_factory=time.time)

    @property
    def active_entries(self) -> tuple[PreferenceMemoryEntry, ...]:
        """Only the active entries, in the snapshot's deterministic order."""
        return tuple(entry for entry in self.entries if entry.is_active)

    @property
    def active_count(self) -> int:
        """Number of active entries."""
        return sum(1 for entry in self.entries if entry.is_active)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()
