"""Task-scoped shopping requirements: budget, product category and use case.

Why this module exists
----------------------
The clarification continuation must carry the *task* the user is shopping for across an MCP
process restart, including a budget that the user may revise.  M9 is a user-scoped permanent
preference store and is not a task ledger; the P17 catalogue also has no price in its C4 text.
So the task requirements are extracted deterministically from the original query and the real
clarification reply, merged (the newest statement wins), persisted in the continuation, and
handed to ``recommend_products``.

Everything here is pure and offline:

* extraction recognises a small, explicit Chinese/English shopping vocabulary and nothing
  else; an unmatched sentence yields no requirement rather than a guess;
* a budget keeps its currency and the exact source text; a currency is never inferred from an
  unrelated symbol and a cross-currency comparison is refused;
* price verification treats the catalogue's ``historical_metadata`` price as a verifiable
  historical value **only** when the currency matches, and reports UNKNOWN otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

__all__ = [
    "PRICE_CURRENCY_MISMATCH",
    "PRICE_NOT_APPLICABLE",
    "PRICE_OVER",
    "PRICE_UNKNOWN",
    "PRICE_WITHIN",
    "PriceBudget",
    "PriceEvaluation",
    "TASK_REQUIREMENTS_SCHEMA",
    "TaskRequirements",
    "augment_query",
    "evaluate_price_budget",
    "extract_task_requirements",
    "strip_budget_expressions",
]

TASK_REQUIREMENTS_SCHEMA = "agentrecx.task_requirements.v1"

PRICE_WITHIN = "within"
PRICE_OVER = "over"
PRICE_UNKNOWN = "unknown"
PRICE_CURRENCY_MISMATCH = "currency_mismatch"
PRICE_NOT_APPLICABLE = "not_applicable"

_CNY_UNITS = ("元", "块", "人民币", "rmb", "cny")
_USD_UNITS = ("$", "usd", "dollar", "dollars", "美金", "美元")

#: Chinese budget/price phrases.  ``元`` (or another currency unit) is required so a bare
#: number (a model number, a rating) is never mistaken for a budget.
_ZH_PRICE = re.compile(
    r"(?:(?:预算|价格|价位|上限|最高|最多|不超过|封顶|控制在)\s*"
    r"(?:是|为|改成|改为|调整到|调到|改到|到|:|：)?\s*[¥￥]?\s*)?"
    r"(?P<amount>\d{1,7}(?:\.\d{1,2})?)\s*(?P<unit>元|块|人民币|rmb|RMB|cny|CNY)"
)
_ZH_LOWER = re.compile(r"至少|不低于|起价|以上|不少于")
_ZH_UPPER = re.compile(r"预算|低于|不超过|最多|最高|封顶|以内|以下|左右|上限")

#: English budget phrases (the existing M9 vocabulary, reused here for task requirements).
_EN_PRICE = re.compile(
    r"(?:budget|price|spend|cost|no\s+more\s+than|less\s+than|up\s+to|under|below|"
    r"maximum|max|at\s+least|minimum|min)\s*[^0-9$]{0,20}\$?\s*"
    r"(?P<amount>\d{1,7}(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_EN_LOWER = re.compile(r"\bat\s+least\b|\bminimum\b|\bno\s+less\s+than\b|\bstarting\s+at\b",
                       re.IGNORECASE)
_EN_UPPER = re.compile(r"\bbudget\b|\bunder\b|\bbelow\b|\bless\s+than\b|\bup\s+to\b|"
                       r"\bno\s+more\s+than\b|\bmax(?:imum)?\b", re.IGNORECASE)

#: Budget expressions are removed from the **retrieval** text.  A budget is applied as a
#: structured hard constraint; leaving an old amount in the lexical query makes BM25 match
#: model numbers ("5000"/"8000") and, after a revision, keeps the expired amount alive.
_ZH_BUDGET_SPAN = re.compile(
    r"(?:(?:预算|价格|价位|上限|最高|最多|不超过|封顶|控制在)\s*"
    r"(?:是|为|改成|改为|调整到|调到|改到|到|:|：)?\s*)?[¥￥]?\s*"
    r"\d{1,7}(?:\.\d{1,2})?\s*(?:元|块|人民币|rmb|RMB|cny|CNY)\s*的?"
)
_EN_BUDGET_SPAN = re.compile(
    r"\b(?:budget|price|spend|cost|under|below|less\s+than|up\s+to|no\s+more\s+than|"
    r"maximum|max|at\s+least|minimum|min)\b[^.!?;]{0,25}?\$?\s*\d{1,7}(?:\.\d{1,2})?",
    re.IGNORECASE,
)

_ZH_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("笔记本电脑", "laptop"), ("笔记本", "laptop"), ("耳机", "headphones"),
    ("键盘", "keyboard"), ("鼠标", "mouse"), ("显示器", "monitor"),
    ("手机", "phone"), ("平板", "tablet"), ("音箱", "speaker"), ("音响", "speaker"),
)
_EN_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("laptop", "laptop"), ("notebook", "laptop"), ("headphones", "headphones"),
    ("headset", "headphones"), ("earbuds", "earbuds"), ("earphones", "earbuds"),
    ("keyboard", "keyboard"), ("mouse", "mouse"), ("monitor", "monitor"),
    ("phone", "phone"), ("tablet", "tablet"), ("speaker", "speaker"),
)
_ZH_USE_CASES: tuple[tuple[str, str], ...] = (
    ("电竞", "gaming"), ("游戏", "gaming"), ("办公", "office"), ("学习", "study"),
    ("编程", "programming"), ("设计", "design"), ("剪辑", "editing"),
    ("影音", "media"), ("音乐", "music"), ("旅行", "travel"), ("出差", "travel"),
    ("运动", "sports"), ("健身", "fitness"), ("降噪", "noise_cancelling"),
)
_EN_USE_CASES: tuple[tuple[str, str], ...] = (
    ("gaming", "gaming"), ("office", "office"), ("study", "study"),
    ("programming", "programming"), ("design", "design"), ("editing", "editing"),
    ("travel", "travel"), ("fitness", "fitness"),
)


def _currency_for(unit: str | None, text: str) -> str | None:
    unit_text = str(unit or "").strip().lower()
    if unit_text and any(alias in unit_text for alias in _CNY_UNITS):
        return "CNY"
    if unit_text and any(alias in unit_text for alias in ("$", "usd", "dollar", "美金", "美元")):
        return "USD"
    lowered = str(text or "").lower()
    if "人民币" in lowered or re.search(r"\b(rmb|cny)\b", lowered):
        return "CNY"
    if "美元" in lowered or "美金" in lowered or re.search(r"\b(usd|dollars?)\b", lowered):
        return "USD"
    if "$" in lowered:
        return "USD"
    return None


@dataclass(frozen=True)
class PriceBudget:
    """One task-scoped price bound with its currency and the user's exact source text."""

    max_amount: float | None = None
    min_amount: float | None = None
    currency: str | None = None
    source_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_amount": self.max_amount,
            "min_amount": self.min_amount,
            "currency": self.currency,
            "source_text": self.source_text,
        }

    @property
    def label(self) -> str:
        bound = f"<= {self.max_amount:g}" if self.max_amount is not None else (
            f">= {self.min_amount:g}" if self.min_amount is not None else "unbounded"
        )
        unit = self.currency or "currency unknown"
        return f"{bound} {unit}"


@dataclass(frozen=True)
class TaskRequirements:
    """The task-scoped requirements used for the current purchase."""

    budget: PriceBudget | None = None
    category: str | None = None
    use_cases: tuple[str, ...] = ()
    source_text: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_REQUIREMENTS_SCHEMA,
            "budget": None if self.budget is None else self.budget.as_dict(),
            "category": self.category,
            "use_cases": list(self.use_cases),
            "source_text": list(self.source_text),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "TaskRequirements":
        if not isinstance(payload, Mapping):
            return cls()
        if payload.get("schema") not in (None, TASK_REQUIREMENTS_SCHEMA):
            return cls()
        budget_payload = payload.get("budget")
        budget = None
        if isinstance(budget_payload, Mapping):
            max_amount = budget_payload.get("max_amount")
            min_amount = budget_payload.get("min_amount")
            budget = PriceBudget(
                max_amount=float(max_amount) if isinstance(max_amount, (int, float)) else None,
                min_amount=float(min_amount) if isinstance(min_amount, (int, float)) else None,
                currency=(
                    str(budget_payload.get("currency"))
                    if budget_payload.get("currency") in {"CNY", "USD"} else None
                ),
                source_text=str(budget_payload.get("source_text") or ""),
            )
        category = payload.get("category")
        use_cases = payload.get("use_cases")
        sources = payload.get("source_text")
        return cls(
            budget=budget,
            category=str(category) if isinstance(category, str) and category else None,
            use_cases=tuple(
                str(item) for item in (use_cases or ()) if isinstance(item, str) and item
            ),
            source_text=tuple(
                str(item) for item in (sources or ()) if isinstance(item, str) and item
            ),
        )

    def merged(self, update: "TaskRequirements") -> "TaskRequirements":
        """Merge a later user statement: a stated budget/category replaces the old one."""
        use_cases = list(self.use_cases)
        for item in update.use_cases:
            if item not in use_cases:
                use_cases.append(item)
        sources = list(self.source_text)
        for item in update.source_text:
            if item not in sources:
                sources.append(item)
        return TaskRequirements(
            budget=update.budget if update.budget is not None else self.budget,
            category=update.category if update.category is not None else self.category,
            use_cases=tuple(use_cases),
            source_text=tuple(sources),
        )


def _first_present(text: str, table: tuple[tuple[str, str], ...]) -> list[str]:
    found: list[str] = []
    lowered = text.lower()
    for needle, value in table:
        if value in found:
            continue
        if needle.isascii():
            # Word-bounded for English so "designed" is not "design" and "mousepad" is not
            # "mouse".  Chinese terms are matched as substrings (no word delimiters).
            if re.search(rf"\b{re.escape(needle)}s?\b", lowered):
                found.append(value)
        elif needle in lowered:
            found.append(value)
    return found


def extract_task_requirements(text: str) -> TaskRequirements:
    """Extract explicit task requirements from one user-authored sentence or query."""
    value = str(text or "").strip()
    if not value:
        return TaskRequirements()
    budget = None
    zh = _ZH_PRICE.search(value)
    if zh is not None:
        amount = float(zh.group("amount"))
        lower = _ZH_LOWER.search(value) is not None and _ZH_UPPER.search(value) is None
        budget = PriceBudget(
            min_amount=amount if lower else None,
            max_amount=None if lower else amount,
            currency=_currency_for(zh.group("unit"), value),
            source_text=value,
        )
    else:
        en = _EN_PRICE.search(value)
        if en is not None:
            amount = float(en.group("amount"))
            lower = _EN_LOWER.search(value) is not None and _EN_UPPER.search(value) is None
            budget = PriceBudget(
                min_amount=amount if lower else None,
                max_amount=None if lower else amount,
                currency=_currency_for(None, value),
                source_text=value,
            )
    categories = _first_present(value, _ZH_CATEGORIES) + _first_present(value, _EN_CATEGORIES)
    category = categories[0] if categories else None
    use_cases = _first_present(value, _ZH_USE_CASES) + _first_present(value, _EN_USE_CASES)
    seen: list[str] = []
    for item in use_cases:
        if item not in seen:
            seen.append(item)
    return TaskRequirements(
        budget=budget,
        category=category,
        use_cases=tuple(seen),
        source_text=(value,),
    )


def strip_budget_expressions(text: str) -> str:
    """Remove budget amounts from retrieval text while leaving the original untouched.

    The caller must keep the original user text for history; only the BM25 query is stripped.
    """
    value = str(text or "")
    if not value:
        return ""
    stripped = _ZH_BUDGET_SPAN.sub(" ", value)
    stripped = _EN_BUDGET_SPAN.sub(" ", stripped)
    return " ".join(stripped.split())


def augment_query(query: str, requirements: TaskRequirements) -> str:
    """Build the lexical retrieval query from the current (non-expired) requirements.

    Budget amounts are deliberately **not** appended, and any budget expression already in the
    user text is stripped: on a lexical index a number matches model numbers ("8000") and an
    expired amount must never keep influencing retrieval.  The original query is preserved
    elsewhere for history and audit.
    """
    parts = [strip_budget_expressions(query)]
    lowered = " ".join(parts).lower()
    if requirements.category and requirements.category not in lowered:
        parts.append(requirements.category)
    for use_case in requirements.use_cases:
        if use_case not in lowered:
            parts.append(use_case)
    return " ".join(part for part in parts if part)


def evaluate_price_budget(
    price_fact: Mapping[str, Any] | None,
    budget: PriceBudget | None,
) -> PriceEvaluation:
    """Compare one catalogue price fact with the task budget, currency-aware.

    ``historical_metadata`` is the catalogue's sourced crawl-time price; it is usable for a
    historical budget check when the currency matches.  A missing/UNKNOWN price, a missing
    currency on either side, or two different currencies is never treated as compliance.
    """
    fact = price_fact if isinstance(price_fact, Mapping) else {}
    extra: dict[str, Any] = {}
    original_usd = fact.get("original_usd_price")
    if isinstance(original_usd, (int, float)):
        extra["original_usd_price"] = float(original_usd)
    for key in ("fx_rate", "converted_cny_price", "price_note", "verification"):
        value = fact.get(key)
        if isinstance(value, str) and value:
            extra[key] = value
    if budget is None or (budget.max_amount is None and budget.min_amount is None):
        return PriceEvaluation(PRICE_NOT_APPLICABLE, "no budget constraint for this task")
    status = str(fact.get("status") or "")
    value = fact.get("value")
    currency = fact.get("currency")
    if status not in {"present", "historical_metadata"} or not isinstance(value, (int, float)):
        return PriceEvaluation(
            PRICE_UNKNOWN,
            "the catalogue provides no price for this candidate",
            source=fact.get("source"),
            price_kind=fact.get("price_kind"),
            **extra,
        )
    if not budget.currency or not currency:
        return PriceEvaluation(
            PRICE_UNKNOWN,
            "the budget or catalogue price has no explicit currency",
            amount=float(value), currency=currency,
            source=fact.get("source"), price_kind=fact.get("price_kind"),
            **extra,
        )
    if str(budget.currency) != str(currency):
        return PriceEvaluation(
            PRICE_CURRENCY_MISMATCH,
            f"budget is in {budget.currency} but the catalogue price is in {currency}",
            amount=float(value), currency=str(currency),
            source=fact.get("source"), price_kind=fact.get("price_kind"),
            **extra,
        )
    amount = float(value)
    if budget.max_amount is not None and amount > float(budget.max_amount):
        return PriceEvaluation(
            PRICE_OVER,
            f"{amount:g} {currency} exceeds the {budget.max_amount:g} {budget.currency} budget",
            amount=amount, currency=str(currency),
            source=fact.get("source"), price_kind=fact.get("price_kind"),
            **extra,
        )
    if budget.min_amount is not None and amount < float(budget.min_amount):
        return PriceEvaluation(
            PRICE_OVER,
            f"{amount:g} {currency} is below the {budget.min_amount:g} {budget.currency} floor",
            amount=amount, currency=str(currency),
            source=fact.get("source"), price_kind=fact.get("price_kind"),
            **extra,
        )
    return PriceEvaluation(
        PRICE_WITHIN,
        f"{amount:g} {currency} is within the task budget",
        amount=amount, currency=str(currency),
        source=fact.get("source"), price_kind=fact.get("price_kind"),
        **extra,
    )


@dataclass(frozen=True)
class PriceEvaluation:
    """The outcome of comparing one price fact with the task budget."""

    status: str
    reason: str = ""
    amount: float | None = None
    currency: str | None = None
    source: str | None = None
    price_kind: str | None = None
    original_usd_price: float | None = None
    fx_rate: str | None = None
    converted_cny_price: str | None = None
    price_note: str | None = None
    verification: str | None = None

    @property
    def verified(self) -> bool:
        return self.status == PRICE_WITHIN

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "verified": self.verified,
            "reason": self.reason,
            "amount": self.amount,
            "currency": self.currency,
            "source": self.source,
            "price_kind": self.price_kind,
            "verification": self.verification,
        }
        if self.original_usd_price is not None:
            payload["original_usd_price"] = self.original_usd_price
        if self.fx_rate is not None:
            payload["fx_rate"] = self.fx_rate
        if self.converted_cny_price is not None:
            payload["converted_cny_price"] = self.converted_cny_price
        if self.price_note is not None:
            payload["price_note"] = self.price_note
        return payload
