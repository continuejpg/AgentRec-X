"""Product-level feedback: identity binding, validation and the audited write (V2 E-03C).

What this module is
-------------------
A user can react to a product the session **actually presented**: "I like the second one",
"don't show me that black pair again", "forget what I said about it".  Turning that sentence into a
stored fact is an identity decision, so it happens here, in the trusted layer, and never in the
model:

1. **Resolve the reference** against the session's own record of its last recommendation turn - the
   cards that were really presented, with their real ``item_id``, ``parent_asin`` and final position.
   An ordinal ("the first one"), a bare identity ("B0..."), a title fragment that matches exactly one
   presented card, or an unambiguous "that one" all resolve; anything else is reported as ambiguous
   and **writes nothing**.
2. **Classify the attitude** with deterministic wording rules: like, dislike, an explicit request not
   to be shown it again, a correction, or a retraction.  Only what the user said is recorded - a
   dislike of one product is never widened into a brand, colour or category preference.
3. **Write through M9** (``PreferenceMemoryService.process_turn``) as a ``product`` entry whose value
   is the canonical ``parent_asin``, so it is validated, deduplicated, supersedeable and auditable
   exactly like any other preference, and read back as ACTIVE next turn.

What it deliberately does not do
--------------------------------
* It does not recall a product.  Feedback can reorder or sink a candidate the run already retrieved;
  it can never retrieve one (that would be recollection, not ranking).
* It does not treat a demotion as a hard exclusion.  This deployment orders by M10A evidence, so a
  disliked product sinks - and is genuinely absent from the answer whenever the fused pool is larger
  than ``k`` - but it is not deleted from the pool, and the audit says `demoted`, not `excluded`.
* It does not let free text name a product: a ``parent_asin`` or ``item_id`` mentioned by the user is
  only accepted when it is one of the cards this session presented.
* It does not touch another session: the record it resolves against belongs to this session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from recommendation.memory.schemas import (
    PreferenceCandidate,
    PreferenceKind,
    PreferenceMode,
    PreferencePolarity,
)

__all__ = [
    "FEEDBACK_ACTIONS",
    "feedback_marks",
    "product_feedback_view",
    "PresentedCard",
    "ProductFeedbackResolution",
    "classify_feedback",
    "feedback_candidate",
    "resolve_reference",
]

#: The closed action vocabulary.  ``demote``/``promote`` are ranking effects; ``retract`` and
#: ``correct`` change what is stored; ``none`` means "this was not product feedback".
FEEDBACK_ACTIONS = ("like", "dislike", "exclude", "correct", "retract", "none")

#: Wording that retracts or corrects an earlier statement about a product.
_RETRACT = re.compile(
    r"\b(forget|ignore|never\s*mind|nevermind|disregard|undo|take\s*back|remove\s+my|"
    r"delete\s+my|cancel\s+my)\b",
    re.IGNORECASE,
)
_CORRECT = re.compile(
    r"\b(actually|instead|i\s+changed\s+my\s+mind|on\s+second\s+thought|correction)\b",
    re.IGNORECASE,
)
_DISLIKE = re.compile(
    r"\b(did\s*n[o']?t\s+like|do\s*n[o']?t\s+like|does\s*n[o']?t\s+like|dislike|hate|"
    r"not\s+a\s+fan|no\s+thanks|nope)\b",
    re.IGNORECASE,
)
_EXCLUDE = re.compile(
    r"\b(don'?t|do\s+not|never|stop)\b[^.]{0,40}?"
    r"\b(recommend|show|suggest|offer|include)\b[^.]{0,20}?\b(again|anymore|any\s+more)\b"
    r"|\b(exclude|remove)\b[^.]{0,30}?\b(from\s+(the\s+)?(list|results|recommendations))\b",
    re.IGNORECASE,
)
_LIKE = re.compile(
    r"\b(i\s+like|i\s+love|i\s+prefer|i\s+enjoy|i\s+want|liked|love\s+it|"
    r"that'?s\s+the\s+one|keep\s+it|more\s+like\s+(this|that))\b",
    re.IGNORECASE,
)

#: Ordinal words, in the order a user counts what they were shown.
_ORDINALS: dict[str, int] = {
    "first": 1, "1st": 1, "one": 1,
    "second": 2, "2nd": 2, "two": 2,
    "third": 3, "3rd": 3, "three": 3,
    "fourth": 4, "4th": 4, "four": 4,
    "fifth": 5, "5th": 5, "five": 5,
}
_ORDINAL_RE = re.compile(
    r"\b(?:the\s+|number\s+|#)?(" + "|".join(sorted(_ORDINALS, key=len, reverse=True)) + r")\b"
    r"(?:\s+(?:one|item|product|result|option|card))?",
    re.IGNORECASE,
)
#: Number words that are *not* in the counting vocabulary above ("the seventh one"): they make a
#: reference unresolvable rather than silently meaning "the first".
_ORDINAL_NOISE = re.compile(
    r"(?:\d+(?:st|nd|rd|th)|seventh|eighth|ninth|tenth|eleventh|twelfth)\s*$", re.IGNORECASE
)
_LAST_RE = re.compile(r"\b(the\s+)?last\s+(one|item|product|result|option|card)\b", re.IGNORECASE)
#: Wording that points at *what the session presented* ("the ones you showed me", "that
#: recommendation").  It is what separates product feedback from an ordinary attribute statement:
#: "I prefer blue" is memory, while "I don't like the black shoes you showed me" is feedback about
#: a card - and only the second may have its generic reading dropped.
_PRESENTATION_RE = re.compile(
    r"\b(you\s+(showed|recommended|suggested|gave|listed|sent)|"
    r"(showed|recommended|suggested|listed)\s+me|"
    r"the\s+(one|ones|item|items|product|products|result|results|option|options)\b|"
    r"that\s+(recommendation|suggestion|result|item|product)|"
    r"from\s+(the\s+)?(list|results|recommendations))\b",
    re.IGNORECASE,
)
_DEICTIC_RE = re.compile(
    r"\b(this|that|it|the\s+one|these|those)\b", re.IGNORECASE
)
#: A token shaped like a catalogue identity (Amazon ``parent_asin``s are upper-case alphanumeric).
#: Any such token counts as a product *reference*, so it is validated against the session's own
#: presented cards - and refused when it is not one of them - rather than read as ordinary text.
_IDENTITY_RE = re.compile(r"\b[A-Z0-9]{8,14}\b")
_ITEM_ID_RE = re.compile(r"\bitem[_\s-]?(\d{1,9})\b", re.IGNORECASE)


@dataclass(frozen=True)
class PresentedCard:
    """One product this session really presented, as the user saw it.

    Only what identity binding needs: the canonical identity, the integer id the cards showed, the
    final (post-ranking) position, and the run that produced it.  No score, no metadata, no ledger.
    """

    parent_asin: str
    item_id: int
    position: int
    title: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_asin": self.parent_asin,
            "item_id": self.item_id,
            "position": self.position,
            "title": self.title,
        }


@dataclass(frozen=True)
class ProductFeedbackResolution:
    """The outcome of reading one message as product feedback.

    ``status`` is the honest verdict: ``none`` (not product feedback at all), ``resolved`` (bound to
    a real presented card), ``ambiguous`` (several cards fit the reference), ``unknown_product`` (an
    identity was named that this session never presented), or ``no_presentation`` (this session has
    not presented anything, or the user referred to a product without naming one).
    """

    status: str
    action: str = "none"
    reference: str = ""
    span: str = ""
    card: PresentedCard | None = None
    detail: str = ""
    #: The attitude the wording actually expressed, independent of the *mode* ("correct" is a
    #: replacement of an earlier statement, not an attitude of its own: "actually I like it" is a
    #: positive correction and must never be stored as an avoidance).
    polarity: PreferencePolarity | None = None

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"


def _sentence_span(message: str, match: re.Match[str]) -> str:
    """The sentence containing a match, so suppression is scoped to what the user meant."""
    start = message.rfind(".", 0, match.start())
    end = message.find(".", match.end())
    return message[start + 1 : len(message) if end == -1 else end].strip()


def _find_wording(message: str, pattern: re.Pattern[str]) -> re.Match[str] | None:
    return pattern.search(message)


def attitude_of(message: str) -> PreferencePolarity | None:
    """The attitude a message expresses, independent of how it relates to earlier statements."""
    if _find_wording(message, _DISLIKE) is not None:
        return PreferencePolarity.AVOID
    if _find_wording(message, _EXCLUDE) is not None:
        return PreferencePolarity.AVOID
    if _find_wording(message, _LIKE) is not None:
        return PreferencePolarity.PREFER
    return None


def classify_feedback(message: str) -> tuple[str, str]:
    """``(action, span)`` for one message, from its wording alone.

    Retraction and correction win over attitude, and an explicit "do not show me it again" wins over
    a plain dislike, because the three mean different things and must not be conflated.
    """
    retract = _find_wording(message, _RETRACT)
    if retract is not None:
        return "retract", _sentence_span(message, retract)
    exclude = _find_wording(message, _EXCLUDE)
    if exclude is not None:
        return "exclude", _sentence_span(message, exclude)
    dislike = _find_wording(message, _DISLIKE)
    correct = _find_wording(message, _CORRECT)
    if dislike is not None and (correct is None or dislike.start() > correct.start()):
        return "dislike", _sentence_span(message, dislike)
    if correct is not None:
        # "Actually ..." alone carries no new attitude; whether it corrects depends on the wording
        # that follows it, which is classified above.  Unqualified, it is a correction request.
        return "correct", _sentence_span(message, correct)
    like = _find_wording(message, _LIKE)
    if like is not None:
        return "like", _sentence_span(message, like)
    return "none", ""


#: Reference kinds, in the order they are tried: an explicit count, an explicit identity, then a
#: deictic that only resolves when the presentation was unambiguous.
def _reference_tokens(message: str) -> tuple[str, str, Any]:
    """``(text, kind, value)`` for the reference a message carries, for resolution and audit."""
    ordinal = _ORDINAL_RE.search(message)
    if ordinal is not None:
        word = (ordinal.group(1) or "").lower()
        before = message[max(0, ordinal.start() - 20) : ordinal.start()]
        if _ORDINAL_NOISE.search(before):
            # "the seventh one" is not "the first one": refuse rather than guess.
            return ordinal.group(0).strip(), "ordinal", None
        return ordinal.group(0).strip(), "ordinal", _ORDINALS.get(word)
    last = _LAST_RE.search(message)
    if last is not None:
        return last.group(0).strip(), "last", None
    identity = _IDENTITY_RE.search(message)
    if identity is not None:
        return identity.group(0).strip(), "parent_asin", identity.group(0).strip()
    item_id = _ITEM_ID_RE.search(message)
    if item_id is not None:
        return item_id.group(0).strip(), "item_id", int(item_id.group(1))
    deictic = _DEICTIC_RE.search(message)
    if deictic is not None:
        return deictic.group(0).strip(), "deictic", None
    return "", "", None


def resolve_reference(
    message: str, presented: Sequence[PresentedCard]
) -> ProductFeedbackResolution:
    """Bind one message to a card this session really presented.

    Refuses rather than guesses: a reference that fits several cards, an identity this session never
    showed, or feedback wording with no resolvable product all produce a non-resolved status and
    therefore no write.  The card list is the caller's - normally the session's own last
    recommendation turn - so a message can never reach another session's cards.
    """
    action, span = classify_feedback(message)
    if action == "none":
        return ProductFeedbackResolution(status="none", action="none")
    reference, kind, value = _reference_tokens(message)
    refers_to_a_product = (
        bool(reference)
        or bool(_PRESENTATION_RE.search(message))
        or any(
            card.title and card.title.casefold() in message.casefold() for card in presented
        )
    )
    if not refers_to_a_product:
        # An attitude about an attribute ("I prefer blue", "I love waterproof boots") is ordinary
        # preference memory.  Only a statement that points at a product is product feedback, and
        # only that statement's generic reading may be dropped.
        return ProductFeedbackResolution(status="none", action="none")
    if not presented:
        return ProductFeedbackResolution(
            status="no_presentation",
            action=action,
            reference=reference,
            span=span,
            detail="this session has not presented any product yet",
        )

    ordered = sorted(presented, key=lambda card: card.position)
    if kind == "ordinal":
        position = value if isinstance(value, int) else None
        if position is None or position > len(ordered):
            return ProductFeedbackResolution(
                status="ambiguous",
                action=action,
                reference=reference,
                span=span,
                detail=f"no card at position {position}",
            )
        return ProductFeedbackResolution(
            status="resolved", action=action, reference=reference, span=span,
            card=ordered[position - 1],
            polarity=(
                None if action == "retract" else attitude_of(message) or PreferencePolarity.PREFER
            ),
        )
    if kind == "last":
        return ProductFeedbackResolution(
            status="resolved", action=action, reference=reference, span=span, card=ordered[-1],
            polarity=(
                None if action == "retract" else attitude_of(message) or PreferencePolarity.PREFER
            ),
        )
    if kind == "parent_asin":
        matches = [card for card in ordered if card.parent_asin == reference]
    elif kind == "item_id":
        matches = [card for card in ordered if card.item_id == value]
    else:
        # A deictic ("this one") is only unambiguous when exactly one card was presented; a title
        # fragment is accepted only when it fits exactly one card.
        titled = [
            card
            for card in ordered
            if card.title and card.title.casefold() in message.casefold()
        ]
        if titled:
            matches = titled
        elif len(ordered) == 1:
            matches = ordered
        else:
            return ProductFeedbackResolution(
                status="ambiguous",
                action=action,
                reference=reference or "this one",
                span=span,
                detail="several products were presented and none was named",
            )
    if not matches:
        return ProductFeedbackResolution(
            status="unknown_product",
            action=action,
            reference=reference,
            span=span,
            detail="the named product is not one this session presented",
        )
    if len(matches) > 1:
        return ProductFeedbackResolution(
            status="ambiguous",
            action=action,
            reference=reference,
            span=span,
            detail="the reference fits several presented products",
        )
    return ProductFeedbackResolution(
        status="resolved",
        action=action,
        reference=reference,
        span=span,
        card=matches[0],
        polarity=(
            None if action == "retract" else attitude_of(message) or PreferencePolarity.PREFER
        ),
    )


def feedback_candidate(
    resolution: ProductFeedbackResolution, *, extractor: str = "product-feedback"
) -> PreferenceCandidate:
    """The M9 statement one resolved feedback produces.

    ``like`` -> a ``product`` preference; ``dislike``/``exclude`` -> the same kind with ``avoid``
    polarity (this deployment orders by evidence, so an exclusion request is honoured as a demotion
    and reported as one); ``retract`` -> a retraction of that product's value; ``correct`` -> a
    replacement of the product's previous entry, so the new attitude is the active one.
    """
    card = resolution.card
    if card is None:  # pragma: no cover - callers check ``resolved`` first
        raise ValueError("a feedback candidate needs a resolved card")
    span = resolution.span or f"feedback about {card.parent_asin}"
    if resolution.action == "retract":
        return PreferenceCandidate(
            kind=PreferenceKind.PRODUCT,
            value=card.parent_asin,
            mode=PreferenceMode.REMOVE,
            source_text=span,
            extractor=extractor,
        )
    polarity = resolution.polarity or PreferencePolarity.PREFER
    mode = PreferenceMode.REPLACE if resolution.action == "correct" else PreferenceMode.ADD
    return PreferenceCandidate(
        kind=PreferenceKind.PRODUCT,
        value=card.parent_asin,
        polarity=polarity,
        mode=mode,
        replaces=card.parent_asin if mode is PreferenceMode.REPLACE else None,
        source_text=span,
        extractor=extractor,
    )


def _active_product_entries(snapshot: Any) -> list[tuple[str, str]]:
    """``(parent_asin, polarity)`` for every ACTIVE ``product`` entry of a snapshot."""
    rows: list[tuple[str, str]] = []
    for entry in getattr(snapshot, "entries", ()) or ():
        if not getattr(entry, "is_active", False):
            continue
        if getattr(getattr(entry, "kind", None), "value", None) != PreferenceKind.PRODUCT.value:
            continue
        rows.append(
            (str(entry.value), str(getattr(entry.polarity, "value", entry.polarity)))
        )
    return rows


def product_feedback_view(
    resolution: ProductFeedbackResolution,
    *,
    before: Any = None,
    after: Any = None,
    written: bool = False,
    entry_status: str | None = None,
) -> dict[str, Any]:
    """The response's account of one feedback statement, stage by stage.

    The six things a reader must be able to tell apart are all here: what the user asked
    (``action``/``span``), what was parsed as the reference (``reference``), the verified identity
    (``parent_asin``/``item_id``/``final_position``), whether a record was written (``written``,
    ``memory_id``, ``entry_status``), what is ACTIVE now (``active_product_feedback``), and what the
    *next* turn will therefore do with it (``effect``).
    """
    card = resolution.card
    effect = {
        "like": "promote",
        "dislike": "demote",
        "exclude": "demote",   # this deployment orders by evidence: reported as a demotion
        "correct": "replace",
        "retract": "retract",
    }.get(resolution.action)
    return {
        "status": resolution.status,
        "action": resolution.action,
        "reference": resolution.reference,
        "span": resolution.span,
        "detail": resolution.detail,
        "parent_asin": None if card is None else card.parent_asin,
        "item_id": None if card is None else card.item_id,
        "final_position": None if card is None else card.position,
        "written": bool(written),
        "entry_status": entry_status,
        "effect": None if not written else effect,
        "active_product_feedback": [
            {"parent_asin": asin, "polarity": polarity}
            for asin, polarity in _active_product_entries(after)
        ],
        "active_before": [
            {"parent_asin": asin, "polarity": polarity}
            for asin, polarity in _active_product_entries(before)
        ],
    }


def feedback_marks(snapshot: Any) -> dict[str, str]:
    """``{parent_asin: "liked"|"disliked"}`` for the products a card may be marked with.

    A card marker is about the user's attitude, never about the product: it is kept apart from the
    catalogue knowledge evidence so no reader can mistake "you disliked this" for a fact about the
    product itself.
    """
    marks: dict[str, str] = {}
    for asin, polarity in _active_product_entries(snapshot):
        marks[asin] = "liked" if polarity == PreferencePolarity.PREFER.value else "disliked"
    return marks
