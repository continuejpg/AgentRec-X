"""P7-D regression: normal follow-up keeps the effective product type in query terms."""

from __future__ import annotations

from recommendation.demo.agent_service import _with_effective_target_type
from recommendation.demo.product_role import ProductTarget, TargetRole


def test_follow_up_without_type_keeps_effective_type() -> None:
    target = ProductTarget("cookware", TargetRole.MAIN)
    terms = ("Those", "look", "fine", "if", "you", "have", "eligible", "options;")
    actual = _with_effective_target_type(terms, target)
    assert actual[0] == "cookware"
    assert actual[1:] == terms

    # A query that already names the type is left unchanged.
    unchanged = ("cookware", *terms)
    assert _with_effective_target_type(unchanged, target) == unchanged
