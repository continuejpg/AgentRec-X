from recommendation.state.shopping_adapter import merge_constraint_pairs


def test_authorized_survives_current_kind_without_trusted_pair() -> None:
    assert merge_constraint_pairs((), (("price_max", "50"),), current_turn_kinds=("price_max",)) == (("price_max", "50"),)


def test_current_trusted_pair_still_wins_same_kind() -> None:
    assert merge_constraint_pairs((("price_max", "30"),), (("price_max", "50"),), current_turn_kinds=("price_max",)) == (("price_max", "30"),)


def test_authorized_added_when_no_current_turn_suppression() -> None:
    assert merge_constraint_pairs((), (("price_max", "50"),)) == (("price_max", "50"),)
