"""Gate C-1/C-2: the two single-source serving profiles are config-only and isolate their source."""
from __future__ import annotations

from recommendation.inference.fusion import FUSION_PROFILES, FUSION_SOURCES, FusionInferenceConfig


def test_default_profile_is_untouched() -> None:
    assert FUSION_PROFILES["serving_v1"] == FUSION_SOURCES == ("sequential", "two_tower", "catalog_lexical")
    default = FusionInferenceConfig()
    assert getattr(default, "profile", None) in (None, "serving_v1")
    assert tuple(default.sources) == FUSION_SOURCES


def test_single_source_profiles_select_exactly_one_source() -> None:
    sequential = FusionInferenceConfig.for_profile("serving_v1_sequential_only")
    two_tower = FusionInferenceConfig.for_profile("serving_v1_two_tower_only")
    assert tuple(sequential.sources) == ("sequential",)
    assert tuple(two_tower.sources) == ("two_tower",)
    for config in (sequential, two_tower):
        assert "tiger" not in tuple(config.sources)      # never TIGER
        assert "catalog_lexical" not in tuple(config.sources) or tuple(config.sources) == ("catalog_lexical",)


def test_the_new_profiles_do_not_leak_into_the_v2_tiger_profile() -> None:
    assert "tiger" not in FUSION_PROFILES["serving_v1"]
    assert "tiger" not in FUSION_PROFILES["serving_v1_sequential_only"]
    assert "tiger" not in FUSION_PROFILES["serving_v1_two_tower_only"]
    assert FUSION_PROFILES["serving_v2_tiger"][-1] == "tiger"


def test_unknown_profile_is_refused() -> None:
    import pytest

    from recommendation.inference.fusion import FusionError

    with pytest.raises(FusionError):
        FusionInferenceConfig.for_profile("serving_v1_lexical_only_bogus")
