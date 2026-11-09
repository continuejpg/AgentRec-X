from __future__ import annotations

from types import SimpleNamespace

from recommendation.demo.agent_service import DemoAgentService
from recommendation.demo.soft_preference_ranking import _feature_to_soft
from recommendation.state import shopping_adapter


def test_accessory_request_without_type_preserves_parent_type() -> None:
    service = SimpleNamespace(
        _role_current_targets={"task": SimpleNamespace(type="water_filter", role="main")},
        _role_main_targets={"task": SimpleNamespace(type="water_filter", role="main")},
    )
    changed = DemoAgentService._update_role_target(service, "task", "Show me a replacement mouthpiece instead.")
    assert changed is True
    target = service._role_current_targets["task"]
    assert target.type == "water_filter"
    assert str(getattr(target.role, "value", target.role)) == "accessory"


def test_budget_pronoun_removal_maps_to_price_retraction() -> None:
    removal = SimpleNamespace(kind=None, value="it", source_text="clear it.")
    extraction = SimpleNamespace(removals=(removal,))
    actions = DemoAgentService._retractions_from_extraction(
        extraction, shopping_adapter, message="The budget no longer matters; clear it."
    )
    assert len(actions) == 1
    assert str(getattr(actions[0].kind, "value", actions[0].kind)) == "price_max"


def test_lighter_option_is_a_supported_m5_weight_preference() -> None:
    feature = SimpleNamespace(
        is_soft=False,
        lifecycle="explicit",
        kind="feature",
        direction="",
        value="lighter option",
        preference_id="pref-1",
        scope="task",
    )
    preference = _feature_to_soft(feature)
    assert preference is not None
    assert preference.attribute == "weight"
    assert preference.action == "lower"
