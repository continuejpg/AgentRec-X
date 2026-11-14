"""Build category-matched sampled candidate sets for a harder ranking protocol.

The random 1+99 candidate protocol is dominated by global item popularity.  This
script keeps the same real positive target and the same observed-only policy but
samples unobserved negatives from the same leaf category first, then the same main
category, then random catalog items.  These candidates are still sampled unobserved
items, not confirmed negatives.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/root/autodl-tmp/neural_memory_v1")
TRAIN_K = 32
EVAL_K = 99


def _seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def _categories(item_features: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    leaf_idx = [index for index, name in enumerate(names) if name.startswith("meta_leaf_cat_")]
    main_idx = [index for index, name in enumerate(names) if name.startswith("meta_main_cat_")]
    leaf_values = item_features[:, leaf_idx]
    main_values = item_features[:, main_idx]
    leaf = np.where(leaf_values.sum(axis=1) > 0, leaf_values.argmax(axis=1), -1)
    main = np.where(main_values.sum(axis=1) > 0, main_values.argmax(axis=1), -1)
    return leaf, main


def _sample_candidates(
    *,
    positive: int,
    excluded: set[int],
    leaf: np.ndarray,
    main: np.ndarray,
    leaf_members: dict[int, list[int]],
    main_members: dict[int, list[int]],
    n_items: int,
    k: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    selected_set: set[int] = set()

    def add_from(pool: list[int]) -> None:
        if not pool:
            return
        # random probe; pools are large enough in practice, fallback walks the pool
        attempts = 0
        while len(selected) < k and attempts < max(10000, len(pool) * 4):
            value = int(pool[int(rng.integers(0, len(pool)))])
            attempts += 1
            if value == positive or value in excluded or value in selected_set:
                continue
            selected.append(value)
            selected_set.add(value)
        if len(selected) < k:
            for value in pool:
                if value == positive or value in excluded or value in selected_set:
                    continue
                selected.append(int(value))
                selected_set.add(int(value))
                if len(selected) >= k:
                    break

    leaf_pool = leaf_members.get(int(leaf[positive]), []) if leaf[positive] >= 0 else []
    main_pool = main_members.get(int(main[positive]), []) if main[positive] >= 0 else []
    add_from(leaf_pool)
    add_from(main_pool)
    while len(selected) < k:
        value = int(rng.integers(0, n_items))
        if value == positive or value in excluded or value in selected_set:
            continue
        selected.append(value)
        selected_set.add(value)
    candidates = np.asarray([positive] + selected, dtype=np.int32)
    positive_position = int(rng.integers(0, k + 1))
    candidates[[0, positive_position]] = candidates[[positive_position, 0]]
    return candidates, positive_position


def main() -> int:
    item_features = np.load(DATA / "item_features.npy")
    names = json.loads((DATA / "item_feature_names.json").read_text())
    leaf, main = _categories(item_features, names)
    n_items = item_features.shape[0] - 1
    leaf_members: dict[int, list[int]] = defaultdict(list)
    main_members: dict[int, list[int]] = defaultdict(list)
    for index in range(n_items):
        if leaf[index] >= 0:
            leaf_members[int(leaf[index])].append(index)
        if main[index] >= 0:
            main_members[int(main[index])].append(index)
    stats = {}
    for split, k in (("train", TRAIN_K), ("val", EVAL_K), ("test", EVAL_K)):
        frame = pd.read_parquet(DATA / f"{split}.parquet")
        negatives = np.zeros((len(frame), k), dtype=np.int32)
        candidates_out = np.zeros((len(frame), k + 1), dtype=np.int32)
        positive_positions = np.zeros(len(frame), dtype=np.int32)
        for position, row in enumerate(frame.itertuples(index=False)):
            positive = int(row.target_item_index)
            excluded = set(int(value) for value in row.catalog_history_index)
            candidates, positive_position = _sample_candidates(
                positive=positive,
                excluded=excluded,
                leaf=leaf,
                main=main,
                leaf_members=leaf_members,
                main_members=main_members,
                n_items=n_items,
                k=k,
                seed=_seed(row.user_id, row.target_timestamp, split, "hard"),
            )
            if split == "train":
                negatives[position] = [value for value in candidates if value != positive]
            else:
                candidates_out[position] = candidates
                positive_positions[position] = positive_position
        if split == "train":
            np.save(DATA / "train_negatives_hard.npy", negatives)
        else:
            np.save(DATA / f"{split}_candidates_hard.npy", candidates_out)
            np.save(DATA / f"{split}_positive_position_hard.npy", positive_positions)
        stats[split] = {"rows": len(frame), "candidate_rows": int(len(frame) * (k + 1))}
    stats["protocol"] = {
        "negative_sampling": "same leaf category, then same main category, then random catalog",
        "train_negatives": TRAIN_K,
        "eval_candidates": EVAL_K + 1,
        "positive": "real observed next catalog item",
        "unobserved_policy": "sampled unobserved catalog candidates; not confirmed negatives",
    }
    (DATA / "hard_candidates_manifest.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
