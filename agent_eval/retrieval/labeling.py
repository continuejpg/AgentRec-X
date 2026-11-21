"""Machine-assisted pooled relevance labels for the M3-C experiment.

These labels are produced by the Harness from product titles (and, for the review artifact,
catalogue descriptions).  They are **machine-assisted and pending human review**, not an
independent human annotation.  BM25 scores and ranks are never read by this module.
"""

from __future__ import annotations

import re
from typing import Any

from agent_eval.retrieval.scope import SCOPE_RULES, is_accessory_title
from recommendation.preference.product_role import infer_product_target

__all__ = [
    "LABEL_POLICY",
    "annotate_pool",
    "label_title",
]

LABEL_POLICY = (
    "binary relevance: title names the requested product type, does not name an accessory, "
    "and satisfies the query's explicit use-case/feature terms when present; machine-assisted "
    "by the Harness from titles; pending human review; BM25 scores were not used"
)

_INTENT_PATTERNS = {
    "gaming": re.compile(r"\b(gaming|game)\b", re.IGNORECASE),
    "office": re.compile(r"\b(office|business|professional|work)\b", re.IGNORECASE),
    "study": re.compile(
        r"\b(study|student|learning|kids|education|school|classroom)\b", re.IGNORECASE
    ),
    "noise_cancelling": re.compile(
        r"\b(noise[ -]?cancel\w*|noise[ -]?isolat\w*|anc)\b", re.IGNORECASE
    ),
    "wireless": re.compile(r"\b(wireless|bluetooth|2\.4g|2\.4 ghz)\b", re.IGNORECASE),
    "mechanical": re.compile(r"\bmechanical\b", re.IGNORECASE),
}


def label_title(
    target: str,
    title: str,
    *,
    use_cases: list[str] | None = None,
    features: list[str] | None = None,
) -> tuple[int, str]:
    """Return ``(grade, reason)`` from the title, target and explicit query intent."""
    rule = SCOPE_RULES.get(target)
    if rule is None:
        return 0, f"unsupported target category {target!r}"
    text = str(title or "")
    if is_accessory_title(target, text):
        return 0, "title names an accessory/part"
    title_type = infer_product_target(text).product_type
    if title_type != target:
        # A few catalogue titles use a plural/compound noun that the legacy type regex does
        # not recognise.  When no competing head type is present, fall back to the explicit
        # target term.
        if title_type is None and rule["title"].search(text):
            pass
        else:
            return 0, f"title head product type is {title_type!r}, not {target!r}"
    for intent in list(use_cases or []) + list(features or []):
        pattern = _INTENT_PATTERNS.get(intent)
        if pattern is not None and not pattern.search(text):
            return 0, f"title does not satisfy query intent {intent!r}"
    return 1, f"title matches target category and query intents"


def annotate_pool(
    *,
    query_id: str,
    target: str,
    union_ids: list[str],
    catalogue: Any,
    use_cases: list[str] | None = None,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Build the reviewable pooled annotation for one query.

    Every id in the baseline/candidate top-5 union is judged; no unjudged id is treated as
    non-relevant in this artifact.  The experiment excludes a query from P@5 if its actual
    top-5 contains an id outside this pool.
    """
    seen: set[str] = set()
    rows: dict[str, dict[str, Any]] = {}
    for item_id in union_ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        product = catalogue.get(item_id)
        title = str(getattr(product, "title", "") or "")
        grade, reason = label_title(
            target, title, use_cases=use_cases, features=features,
        )
        rows[item_id] = {
            "grade": grade,
            "title": title,
            "reason": reason,
            "product_type": getattr(product, "product_type", None),
        }
    counts = {"relevant": sum(1 for row in rows.values() if row["grade"] == 1)}
    return {
        "query_id": query_id,
        "target_category": target,
        "required_use_cases": list(use_cases or []),
        "required_features": list(features or []),
        "label_policy": LABEL_POLICY,
        "review_status": "pending_human_review",
        "union_ids": list(rows),
        "judged_ids": {item_id: row["grade"] for item_id, row in rows.items()},
        "rows": rows,
        "relevant_union_count": counts["relevant"],
        "pool_size": len(rows),
    }
