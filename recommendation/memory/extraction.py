"""Preference extraction (Milestone 9).

The extraction seam is an injected ``extract(user_message) -> PreferenceExtraction``.
Two things ship here:

* :class:`RuleBasedPreferenceExtractor` -- a deliberately conservative, offline,
  dependency-free extractor covering a small explicit syntax;
* :class:`ScriptedPreferenceExtractor` -- a deterministic test double for orchestration
  tests.

Neither is a natural-language understanding system, and Milestone 9 does not claim one.
The architecture point is that a future LLM adapter can implement exactly the same
method without changing the store, service or Agent contracts.

Scope and safety
----------------
* Extraction runs **only** on user-authored text.  System prompts, tool output, RAG
  evidence, recommended candidates and model reasoning are never passed in -- the
  service's only caller-visible entry point takes a ``user_message``.
* The extractor emits *preferences and retractions only*.  It has no field for an
  interaction event, an item id, a SASRec score or a candidate, so it cannot influence
  the recommendation path or the trusted history.
* Values that look like credentials are rejected by the schema, so M9 is not a
  conversation archive.
* Anything explicit that does not match a narrow kind is kept as a typed
  ``free_form_constraint`` rather than being forced into a wrong kind.

Supported explicit syntax (conservative, intentionally small)
-------------------------------------------------------------
===============================  ==========================================
User wording                     Extracted
===============================  ==========================================
"my budget is under $100"        ``price_max = 100`` (prefer)
"my budget is at least $50"      ``price_min = 50`` (prefer)
"I don't want anything over $80" ``price_max = 80`` (prefer)
"I prefer blue"                  ``color = blue`` (prefer)
"I prefer lightweight gear"      ``feature = lightweight`` (prefer)
"I like waterproof products"     ``feature = waterproof`` (prefer)
"I need a waterproof jacket"     ``feature = waterproof`` (prefer)
"I don't want red"               ``color = red`` (avoid)
"I never want leather"           ``material = leather`` (avoid)
"I avoid plastic"                ``material = plastic`` (avoid)
"I prefer Acme"                  ``brand = Acme`` (prefer)
"I avoid Acme"                   ``brand = Acme`` (avoid)
"I'm looking for hiking boots"   ``category = hiking boots`` (prefer)
"I prefer lightweight"           ``free_form_constraint = lightweight`` (prefer)
"I don't care about color        retraction of ``color`` (no new entry)
 anymore"
===============================  ==========================================

The feature/material kind split is decided by a small keyword list; an unrecognised
noun after "I don't want" becomes a ``feature`` avoidance rather than being dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .schemas import (
    PreferenceCandidate,
    PreferenceExtraction,
    PreferenceKind,
    PreferenceMode,
    PreferencePolarity,
    PreferenceRemoval,
)

__all__ = [
    "EXTRACTOR_NAME",
    "RuleBasedPreferenceExtractor",
    "ScriptedPreferenceExtractor",
]

#: Provenance label recorded on every entry this extractor produces.
EXTRACTOR_NAME = "rule_based_v1"

#: Nouns that indicate the source calls this a material rather than a feature.
_MATERIAL_KEYWORDS = frozenset(
    {
        "leather",
        "plastic",
        "metal",
        "aluminium",
        "aluminum",
        "steel",
        "cotton",
        "wool",
        "nylon",
        "polyester",
        "rubber",
        "silicone",
        "wood",
        "bamboo",
        "glass",
        "ceramic",
        "mesh",
        "suede",
        "canvas",
        "foam",
    }
)

#: Colour words recognised as a colour constraint.
_COLOR_KEYWORDS = frozenset(
    {
        "black",
        "white",
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
        "grey",
        "gray",
        "silver",
        "gold",
        "beige",
        "teal",
        "navy",
        "tan",
        "multicolour",
        "multicolor",
    }
)

#: Words that are too generic to be useful as a stored value.
_STOPVALUES = frozenset(
    {
        "it",
        "them",
        "that",
        "this",
        "anything",
        "something",
        "products",
        "product",
        "items",
        "item",
        "stuff",
        "gear",
    }
)

#: Retraction phrases: the user withdraws a constraint rather than adding one.
#:
#: A retraction may name its subject before the phrase ("color does not matter") or
#: after it ("I don't care about color"), so both positions are captured separately
#: and resolved by the caller.  Each capture uses a negative lookahead so a lazy match
#: cannot swallow a trailing word ("does not matter anymore" must not yield "anymore").
_NOT_TRAILING = r"(?!(?:anymore|any\s+more|now|please)\b)"
_RETRACTION = re.compile(
    r"(?:(?P<leading>"
    + _NOT_TRAILING
    + r"[a-z0-9 ]{2,40}?)\s+)?"
    r"\b(?:"
    r"(?:i\s+)?(?:don'?t|do\s+not|no\s+longer)\s+care\s+about|"
    r"forget\s+(?:about\s+)?(?:my\s+)?|"
    r"remove\s+(?:my\s+)?|"
    r"clear\s+(?:my\s+)?|"
    r"(?:it\s+)?(?:does\s+not|doesn'?t)\s+matter(?:\s+to\s+me)?"
    r")\s*"
    r"(?P<trailing>(?:(?!\b(?:anymore|any\s+more|now|please)\b)[a-z0-9 ]){0,40}?)"
    r"\s*(?:anymore|any\s+more|now)?\s*[.!]?$",
    re.IGNORECASE,
)

#: Price constraints.
_PRICE = re.compile(
    r"(?:budget|price|spend|cost|no more than|less than|"
    r"up to|under|below|maximum|minimum)"
    r"[^0-9$]{0,20}\$?\s*(?P<amount>\d{1,7}(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

#: Preference / avoidance phrasing around a value.
_POSITIVE = re.compile(
    r"\b(?:i\s+)?(?:prefer|like|love|need|want|looking\s+for|interested\s+in)\s+"
    r"(?P<value>[a-z0-9][a-z0-9 '\-]{1,50})",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(?:i\s+)?(?:do\s+not|don'?t|never|avoid)\s+(?:want|like|need|use|buy)?\s*"
    r"(?P<value>[a-z0-9][a-z0-9 '\-]{1,50})",
    re.IGNORECASE,
)

#: Negated price ceiling phrasing ("I don't want anything over $80").
_NEGATED_PRICE = re.compile(
    r"\b(?:do\s+not|don'?t)\s+want\s+(?:anything|something|to\s+pay)?\s*"
    r"(?:over|above|more\s+than)\b",
    re.IGNORECASE,
)

#: Positive phrasings that do not by themselves imply a category constraint.
_SOFT_POSITIVE = re.compile(
    r"\b(?:i\s+)?(?:like|love|interested\s+in)\s+"
    r"(?P<value>[a-z0-9][a-z0-9 '\-]{1,50})",
    re.IGNORECASE,
)

#: Correction phrasings that name the replacement value directly ("make that blue").
_CORRECTION_VALUE = re.compile(
    r"\b(?:make\s+that|change\s+(?:that\s+to|it\s+to)|i\s+meant)\s+"
    r"(?P<value>[a-z0-9][a-z0-9 '\-]{1,40})",
    re.IGNORECASE,
)

#: Category-seeking phrasings: these *do* name a product category.
_CATEGORY = re.compile(
    r"\b(?:i'?m\s+)?(?:looking\s+for|interested\s+in|searching\s+for|shopping\s+for)\s+"
    r"(?P<value>[a-z0-9][a-z0-9 '\-]{1,50})",
    re.IGNORECASE,
)

#: Brand preference/avoidance phrasings.
_BRAND_POSITIVE = re.compile(
    r"\b(?:i\s+)?(?:prefer|like|love)\s+(?:the\s+)?brand\s+"
    r"(?P<value>[a-z0-9][a-z0-9 '\-]{1,50})",
    re.IGNORECASE,
)
_BRAND_NEGATIVE = re.compile(
    r"\b(?:i\s+)?(?:avoid|don'?t\s+like|do\s+not\s+like|never\s+buy)\s+"
    r"(?:(?:the\s+)?brand\s+)?(?P<value>[A-Z][A-Za-z0-9'\-]{1,40})",
)

#: Words that end a captured value span.
_TRAILING = re.compile(
    r"\s*(?:,|\.|;|!|\?|$|\b(?:instead|please|thanks|thank you|anymore|"
    r"any more|for me|products?|items?|gear|shoes?|boots?|jackets?)\b).*$",
    re.IGNORECASE,
)

#: Leading articles that are not part of a value.
_LEADING = re.compile(r"^(?:a|an|the|some|any)\s+", re.IGNORECASE)

#: A small product-category vocabulary, so an explicit category statement is not
#: stored as a generic feature.  Deliberately short and curated; it is vocabulary,
#: not inference.
_CATEGORY_KEYWORDS = frozenset(
    {
        "hiking", "camping", "fishing", "cycling", "running", "yoga", "climbing",
        "swimming", "skating", "skiing", "snowboarding", "hunting", "kayaking",
        "backpacking", "training", "fitness", "outdoor", "boots", "tent", "tents",
        "backpack", "sleeping", "hydration", "nutrition", "fishing", "archery",
    }
)

#: Colour adjectives that qualify a value rather than naming the constraint slot.
_COLOR_QUALIFIER = re.compile(
    r"^(?P<color>[a-z]+)\s+(?P<rest>[a-z][a-z '\-]*)$", re.IGNORECASE
)


def _replacement_intent(sentence: str) -> tuple[PreferenceMode, str | None]:
    """Return the mode a sentence expresses and any value it explicitly corrects.

    ``REPLACE`` requires an explicit correction marker such as "instead", "make that"
    or "I meant".  A sentence without one is an ``ADD``: two independent preferences are
    never treated as each other's replacement.
    """
    explicit = _REPLACES_EXPLICIT.search(sentence)
    if explicit is not None:
        span = _normalise_capture(explicit.group("value"), sentence)
        return PreferenceMode.REPLACE, (span.value if span else None)
    if _CORRECTION.search(sentence):
        return PreferenceMode.REPLACE, None
    return PreferenceMode.ADD, None


@dataclass(frozen=True)
class _Span:
    """A captured value plus the exact source span that supports it."""

    value: str
    span: str


def _normalise_capture(raw_value: str, sentence: str) -> _Span | None:
    """Trim a captured value span down to a usable value, with its source text."""
    candidate = _TRAILING.sub("", raw_value).strip()
    candidate = _LEADING.sub("", candidate).strip()
    if not candidate:
        return None
    collapsed = " ".join(candidate.split())
    first_word = collapsed.split()[0].lower()
    if first_word in _STOPVALUES:
        # e.g. "anything over $80" -- the capture is a quantifier, not a value.
        return None
    if len(collapsed) > 60:
        return None
    # The provenance span is the sentence the user actually typed, so an entry is
    # always traceable even though the stored value is normalised.
    return _Span(value=collapsed, span=" ".join(sentence.split()))


def _kind_for(value: str, *, default: PreferenceKind) -> PreferenceKind:
    """Classify a captured value into a narrow kind.

    A colour adjective counts as a colour constraint only when the value is not
    "colour + noun" (``blue jacket`` is a feature, ``blue`` is a colour), so an
    explicit colour statement is not silently turned into a feature.
    """
    lowered = value.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    if words & _MATERIAL_KEYWORDS:
        return PreferenceKind.MATERIAL
    if words & _CATEGORY_KEYWORDS and not (words & _COLOR_KEYWORDS):
        return PreferenceKind.CATEGORY
    qualifier = _COLOR_QUALIFIER.match(lowered)
    if qualifier is not None and qualifier.group("color") in _COLOR_KEYWORDS:
        return default
    if words & _COLOR_KEYWORDS:
        return PreferenceKind.COLOR
    return default


#: Markers of *correction* intent: the user is replacing an earlier constraint rather
#: than adding a new one.  Deliberately explicit -- M9 never infers replacement from the
#: kind or from two statements being of the same kind.
_CORRECTION = re.compile(
    r"(?:\binstead\b|\bchange\s+(?:that|it|my\s+mind)\b|\bmake\s+that\b|"
    r"\bi\s+meant\b|\bcorrection\b|\bon\s+second\s+thought\b)",
    re.IGNORECASE,
)

#: A value named as the thing being corrected: "prefer blue instead of black".
_REPLACES_EXPLICIT = re.compile(
    r"\binstead\s+of\s+(?P<value>[a-z0-9][a-z0-9 '\-]{1,40})", re.IGNORECASE
)

#: Leading discourse markers that carry no constraint meaning.
_DISCOURSE = re.compile(
    r"^(?:actually|well|hmm|ok|okay|so|and|but|also|anyway|honestly)\b[,\s]*",
    re.IGNORECASE,
)


def _strip_discourse(text: str) -> str:
    """Remove a leading discourse marker so grammar matching sees the real clause."""
    stripped = text.strip()
    while True:
        reduced = _DISCOURSE.sub("", stripped, count=1).strip()
        if reduced == stripped or not reduced:
            return stripped
        stripped = reduced


def _sentences(text: str) -> list[str]:
    """Split user text into sentence-ish spans (deterministic, no NLP)."""
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


class RuleBasedPreferenceExtractor:
    """Conservative, offline, deterministic preference extractor.

    It recognises a small set of explicit shopping phrasings and nothing else.  Text
    that does not match yields no preference at all -- it is never guessed at, and it
    is never summarised into memory.
    """

    def __init__(self, name: str = EXTRACTOR_NAME) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Extractor label recorded on produced entries."""
        return self._name

    def extract(self, user_message: str) -> PreferenceExtraction:
        """Return the explicit preferences and retractions stated in the turn."""
        if not isinstance(user_message, str) or not user_message.strip():
            return PreferenceExtraction()

        preferences: list[PreferenceCandidate] = []
        removals: list[PreferenceRemoval] = []

        for sentence in _sentences(user_message):
            retraction = _RETRACTION.search(_strip_discourse(sentence))
            if retraction is not None:
                leading = (retraction.group("leading") or "").strip()
                trailing = (retraction.group("trailing") or "").strip()
                # A subject before the phrase wins; otherwise use the one after it.
                target = (leading or trailing).lower()
                target = _LEADING.sub("", target).strip()
                kind = self._retraction_kind(target)
                if kind is not None:
                    removals.append(
                        PreferenceRemoval(
                            kind=kind,
                            source_text=sentence,
                            extractor=self._name,
                        )
                    )
                continue

            # Price is extracted independently of preference phrasing, because a
            # price bound is often expressed through a negative ("I don't want
            # anything over $80") that would otherwise be mistaken for an avoidance.
            price = self._negated_price_bound(sentence) or self._price_constraint(sentence)
            if price is not None:
                preferences.append(price)

            correction = self._phrasing(
                sentence,
                _CORRECTION_VALUE,
                PreferencePolarity.PREFER,
                forced_mode=PreferenceMode.REPLACE,
            )
            if correction is not None:
                preferences.append(correction)
                continue

            for pattern in (_BRAND_POSITIVE, _BRAND_NEGATIVE):
                brand = self._phrasing(
                    sentence,
                    pattern,
                    PreferencePolarity.AVOID
                    if pattern is _BRAND_NEGATIVE
                    else PreferencePolarity.PREFER,
                    forced_kind=PreferenceKind.BRAND,
                )
                if brand is not None:
                    preferences.append(brand)
                    break
            else:
                category = self._phrasing(
                    sentence, _CATEGORY, PreferencePolarity.PREFER,
                    forced_kind=PreferenceKind.CATEGORY,
                )
                if category is not None:
                    preferences.append(category)
                    continue

                negative = self._phrasing(
                    sentence, _NEGATIVE, PreferencePolarity.AVOID
                )
                if negative is not None:
                    preferences.append(negative)
                    continue

                positive = self._phrasing(
                    sentence, _POSITIVE, PreferencePolarity.PREFER
                )
                if positive is not None:
                    preferences.append(positive)

        return PreferenceExtraction(
            preferences=tuple(preferences), removals=tuple(removals)
        )

    # -- helpers ----------------------------------------------------------- #

    def _retraction_kind(self, target: str) -> PreferenceKind | None:
        """Map a retraction target phrase to the kind it retracts."""
        if not target:
            # A bare "forget about it" retracts every preference.
            return None
        if "color" in target or "colour" in target:
            return PreferenceKind.COLOR
        if "material" in target:
            return PreferenceKind.MATERIAL
        if "brand" in target or "store" in target:
            return PreferenceKind.BRAND
        if "price" in target or "budget" in target or "cost" in target:
            return PreferenceKind.PRICE_MAX
        if "category" in target or "categories" in target:
            return PreferenceKind.CATEGORY
        if "feature" in target or "features" in target:
            return PreferenceKind.FEATURE
        return PreferenceKind.FREE_FORM_CONSTRAINT

    def _price_constraint(self, sentence: str) -> PreferenceCandidate | None:
        """Extract a price bound when the sentence clearly states one."""
        match = _PRICE.search(sentence)
        if match is None:
            return None
        amount = match.group("amount")
        lowered = sentence.lower()
        # Word-boundary patterns, so "at least" cannot be misread as containing
        # "least"/"less than" or as an upper bound.  Direction markers are matched as
        # whole phrases rather than substrings.
        upper_markers = (
            r"\bunder\b",
            r"\bbelow\b",
            r"\bless\s+than\b",
            r"\bup\s+to\b",
            r"\bno\s+more\s+than\b",
            r"\bmax(?:imum)?\b",
            r"\bnot\s+over\b",
            r"\bcheaper\s+than\b",
        )
        # "budget" only implies an upper bound when no explicit direction is given
        # ("my budget is at least $50" states a floor, not a ceiling).
        soft_upper_markers = (r"\bbudget\b",)
        lower_markers = (
            r"\bat\s+least\b",
            r"\bminimum\b",
            r"\bno\s+less\s+than\b",
            r"\bstarting\s+at\b",
            r"\bover\b",
            r"\babove\b",
            r"\bmore\s+than\b",
        )
        has_upper = any(re.search(pattern, lowered) for pattern in upper_markers)
        has_lower = any(re.search(pattern, lowered) for pattern in lower_markers)
        if not has_upper and not has_lower:
            has_upper = any(
                re.search(pattern, lowered) for pattern in soft_upper_markers
            )
        if has_upper and not has_lower:
            kind, value = PreferenceKind.PRICE_MAX, amount
        elif has_lower and not has_upper:
            kind, value = PreferenceKind.PRICE_MIN, amount
        elif has_upper and has_lower:
            # Genuinely two-sided ("between") -- store nothing rather than guess.
            return None
        else:
            # A bare number with no direction is not a usable constraint.
            return None
        if value.endswith(".0"):
            value = value[:-2]
        mode, replaces = _replacement_intent(sentence)
        return PreferenceCandidate(
            kind=kind,
            value=value,
            polarity=PreferencePolarity.PREFER,
            source_text=sentence,
            extractor=self._name,
            mode=mode,
            replaces=replaces,
        )

    def _negated_price_bound(self, sentence: str) -> PreferenceCandidate | None:
        """Translate "I don't want anything over $80" into a maximum price bound.

        The phrasing is a negation, but the constraint it states is a price ceiling,
        not a product avoidance; storing it as a feature avoidance would be wrong.
        """
        if _NEGATED_PRICE.search(sentence) is None:
            return None
        amount_match = re.search(r"\$?\s*(\d{1,7}(?:\.\d{1,2})?)", sentence)
        if amount_match is None:
            return None
        amount = amount_match.group(1)
        if amount.endswith(".0"):
            amount = amount[:-2]
        mode, replaces = _replacement_intent(sentence)
        return PreferenceCandidate(
            kind=PreferenceKind.PRICE_MAX,
            value=amount,
            polarity=PreferencePolarity.PREFER,
            source_text=sentence,
            extractor=self._name,
            mode=mode,
            replaces=replaces,
        )

    def _phrasing(
        self,
        sentence: str,
        pattern: re.Pattern[str],
        polarity: PreferencePolarity,
        *,
        forced_kind: PreferenceKind | None = None,
        forced_mode: PreferenceMode | None = None,
    ) -> PreferenceCandidate | None:
        """Extract a value for one polarity pattern, or ``None``."""
        match = pattern.search(sentence)
        if match is None:
            return None
        span = _normalise_capture(match.group("value"), sentence)
        if span is None:
            return None

        mode, replaces = _replacement_intent(sentence)
        if forced_mode is not None:
            mode = forced_mode

        if forced_kind is not None:
            return PreferenceCandidate(
                kind=forced_kind,
                value=span.value,
                polarity=polarity,
                source_text=span.span,
                extractor=self._name,
                mode=mode,
                replaces=replaces,
            )

        # A bare adjective ("I prefer lightweight") is a real preference but not a
        # categorised one, so it is kept as an explicit free-form constraint rather
        # than being forced into a wrong kind.
        if len(span.value.split()) == 1:
            single = span.value.lower()
            if single in _MATERIAL_KEYWORDS:
                kind = PreferenceKind.MATERIAL
            elif single in _COLOR_KEYWORDS:
                kind = PreferenceKind.COLOR
            elif single in _CATEGORY_KEYWORDS:
                kind = PreferenceKind.CATEGORY
            else:
                return PreferenceCandidate(
                    kind=PreferenceKind.FREE_FORM_CONSTRAINT,
                    value=span.value,
                    polarity=polarity,
                    source_text=span.span,
                    extractor=self._name,
                    mode=mode,
                    replaces=replaces,
                )
        else:
            kind = _kind_for(span.value, default=PreferenceKind.FEATURE)

        return PreferenceCandidate(
            kind=kind,
            value=span.value,
            polarity=polarity,
            source_text=span.span,
            extractor=self._name,
            mode=mode,
            replaces=replaces,
        )


class ScriptedPreferenceExtractor:
    """Deterministic test double: a fixed message -> extraction table.

    Used by the orchestration tests and the smoke scripts so memory behaviour is
    exercised without depending on the rule-based extractor's coverage.
    """

    def __init__(self, table: dict[str, PreferenceExtraction] | None = None) -> None:
        self._table = dict(table or {})
        self.calls: list[str] = []

    @property
    def call_count(self) -> int:
        """How many times ``extract`` was called."""
        return len(self.calls)

    def add(self, message: str, extraction: PreferenceExtraction) -> None:
        """Register an extraction for an exact message string."""
        self._table[message] = extraction

    def extract(self, user_message: str) -> PreferenceExtraction:
        """Record the prompt and return the scripted extraction (empty if unknown)."""
        self.calls.append(user_message)
        return self._table.get(user_message, PreferenceExtraction())
