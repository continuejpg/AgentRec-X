"""Phase 2C: feature-contract parity check for one materialised LightGBM group.

Recomputes the 15 features for the first stored group (``query_id 0``, ``user_int_id 1001``)
with the same engine calls the independent-evaluation driver uses, and compares them
value-by-value with ``runs/lr_v1_r2/rlr01_groups.jsonl``.  A non-zero difference means the
online feature builder and the frozen training rows disagree, and the learned-ranker
comparison must stop.

    .venv/bin/python -m experiments.r_lr05_feature_parity
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # runnable as a file as well as with -m
    sys.path.insert(0, str(REPO))

from recommendation.api.app import DEFAULT_CATALOG  # noqa: E402
from recommendation.catalog import MetadataIndex  # noqa: E402
from recommendation.control.candidate_ledger import reciprocal_rank_fusion  # noqa: E402
from recommendation.evaluation.split import split_sequence  # noqa: E402
from recommendation.inference.learned_ranker import build_feature_rows  # noqa: E402
from recommendation.inference.lexical import CatalogueLexicalIndex, query_text_for  # noqa: E402
from recommendation.inference.sasrec import InferenceConfig, SASRecInferenceEngine  # noqa: E402
from recommendation.inference.twotower import (  # noqa: E402
    TwoTowerInferenceConfig,
    TwoTowerInferenceEngine,
)
from experiments.e04b2_gate_a import stream_sequences  # noqa: E402

SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SASREC = REPO / "runs" / "sasrec_canonical_2026" / "best.pt"
TWO_TOWER = REPO / "runs" / "twotower_public_2026_full"
GROUPS = REPO / "runs" / "lr_v1_r2" / "rlr01_groups.jsonl"
FEATURES = [
    "seq_mask", "seq_rank", "seq_inv_rank", "seq_score",
    "tt_mask", "tt_rank", "tt_inv_rank", "tt_score",
    "lex_mask", "lex_rank", "lex_inv_rank", "lex_score",
    "rrf_score", "rrf_rank", "source_count",
]
HEAD = 100
TARGET_USER = 1001


def main() -> int:
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    num_items = int(mappings["num_items"])
    id_to_asin = {int(value): str(key) for key, value in mappings["item2id"].items()}
    asin_to_id = {asin: item_id for item_id, asin in id_to_asin.items()}

    sequential = SASRecInferenceEngine(
        InferenceConfig(checkpoint_path=SASREC, mappings_path=MAPPINGS, device="cpu")
    )
    two_tower = TwoTowerInferenceEngine(
        TwoTowerInferenceConfig(
            checkpoint_path=TWO_TOWER / "checkpoint.pt",
            categories_path=TWO_TOWER / "metadata_categories.json",
            mappings_path=MAPPINGS,
            device="cpu",
        )
    )
    records = MetadataIndex.load(DEFAULT_CATALOG).records
    lexical = CatalogueLexicalIndex.build(records)
    catalogue = {
        str(getattr(value, "parent_asin", None) or key): value
        for key, value in records.items()
    }

    case = None
    for element in stream_sequences(SEQUENCES):
        if int(element.get("user_int_id", -1)) != TARGET_USER:
            continue
        case = split_sequence(
            str(element.get("user_id", "")),
            [int(value) for value in element["item_ids"]],
            num_items,
            TARGET_USER,
        )
        break
    if case is None:
        raise SystemExit(f"case {TARGET_USER} not found")

    history = list(case.test_history)
    hist_asins = [id_to_asin[item] for item in history if item in id_to_asin]
    hist_set = set(history)
    per_source: dict[str, dict[str, tuple[int, float]]] = {"seq": {}, "tt": {}, "lex": {}}
    for item in sequential.recommend(hist_asins, k=HEAD).recommendations:
        per_source["seq"][item.parent_asin] = (int(item.rank), float(item.score))
    for item in two_tower.recommend(history, k=HEAD).recommendations:
        per_source["tt"][item.parent_asin] = (int(item.rank), float(item.score))
    usable = [
        catalogue[asin] for asin in hist_asins[-5:]
        if asin in catalogue and getattr(catalogue[asin], "title", None)
    ]
    query = query_text_for(usable) if usable else ""
    for index, (asin, score) in enumerate(lexical.scored(query, limit=HEAD)):
        per_source["lex"][asin] = (index + 1, float(score))

    rankings = [
        [asin for asin, _ in sorted(per_source[name].items(), key=lambda kv: kv[1][0])]
        for name in ("seq", "tt", "lex")
    ]
    fused = dict(reciprocal_rank_fusion([ranking for ranking in rankings if ranking], k=60))
    fused_order = [asin for asin, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
    pool: list[str] = []
    for name in ("seq", "tt", "lex"):
        for asin in sorted(per_source[name], key=lambda key: per_source[name][key][0]):
            if asin in asin_to_id and asin_to_id[asin] not in hist_set and asin not in pool:
                pool.append(asin)
    rows = build_feature_rows(
        [asin_to_id[asin] for asin in pool],
        pool,
        {name: {asin: pair[0] for asin, pair in per_source[name].items()} for name in per_source},
        {name: {asin: pair[1] for asin, pair in per_source[name].items()} for name in per_source},
        fused,
        fused_order,
    )

    stored: dict[str, dict[str, float]] = {}
    with GROUPS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["query_id"]) != 0:
                break
            stored[str(row["parent_asin"])] = row

    print("pool", len(pool), "stored rows", len(stored), "same asins", set(pool) == set(stored))
    max_diff = {name: 0.0 for name in FEATURES}
    for index, asin in enumerate(pool):
        row = stored.get(asin)
        if row is None:
            print("missing stored row", asin)
            continue
        for position, name in enumerate(FEATURES):
            max_diff[name] = max(max_diff[name], abs(float(rows[index][position]) - float(row[name])))
    for name, diff in max_diff.items():
        print(f"  {name:14s} {diff:.9f}")
    worst = max(max_diff.values())
    print("overall max diff:", worst)
    return 0 if worst <= 5e-7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
