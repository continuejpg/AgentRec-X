"""Phase 2B.4: source complementarity from the frozen M4 evidence (20,000 users).

Why this exists
---------------
The 2B.4 brief asks which candidate source is most likely to add recall.  Re-running the
Agent on 40 cases cannot answer that at scale.  The M4 preregistered run already stored
per-source, depth-100 heads for the frozen 20,000-user cohort, under the frozen task
(``test_history`` = train history + validation target):

    history         SASRec logits
    catalog_search  BM25 over the frozen ``deterministic_terms(h)`` query
    two_tower       Two-Tower cosine
    similar_item    item-item similarity

This driver only *reads* those frozen heads and the frozen test targets and reports, per
depth, how often each source's head contains the target and how much each source contributes
that the others do not.  It runs no model and changes no artifact.

What the numbers are **not**
----------------------------
``history`` here is **SASRec alone**, not the Agent's ``history`` source (Serving V1 fuses
SASRec, Two-Tower and an internal lexical source).  The complementarity numbers are therefore
about *raw sources*; the Agent-level comparison is
:mod:`experiments.p2b4_three_source_comparison`.

    .venv/bin/python -m experiments.p2b4_source_diagnosis --out runs/p2b4_source_diagnosis

Outputs: summary.json + SHA256SUMS.json (refuses to overwrite an existing directory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from recommendation import config as project_config

REPO = project_config.REPO_ROOT
SEQUENCES = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_sequences.json"
HEADS = REPO / "runs" / "m4_evidence" / "heads.npz"
DEFAULT_OUT = REPO / "runs" / "p2b4_source_diagnosis"

SOURCES: tuple[str, ...] = ("history", "catalog_search", "two_tower", "similar_item")
DEPTHS: tuple[int, ...] = (50, 100)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 - a missing git must not fail the analysis
        return None
    return completed.stdout.strip() or None


def _hits(
    heads: np.ndarray, targets: np.ndarray, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean hit vector and 1-based rank of the target inside ``heads[:, :depth]``."""
    found = np.zeros(len(targets), dtype=bool)
    ranks = np.full(len(targets), -1, dtype=np.int64)
    window = heads[:, :depth]
    for index in range(len(targets)):
        positions = np.nonzero(window[index] == targets[index])[0]
        if positions.size:
            found[index] = True
            ranks[index] = int(positions[0]) + 1
    return found, ranks


def _distinct_ratio(heads: np.ndarray, depth: int) -> float:
    """Fraction of a head's slots that hold distinct identities (1.0 = no duplicates)."""
    values = []
    for row in heads[:, :depth]:
        values.append(len(set(row.tolist())) / len(row))
    return round(sum(values) / len(values), 3) if values else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")

    document = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    target_by_user = {
        int(sequence["user_int_id"]): str(sequence["parent_asins"][-1])
        for sequence in document["sequences"]
        if sequence.get("user_int_id") is not None
    }

    data = np.load(HEADS, allow_pickle=True)
    users = [int(value) for value in data["user_int_ids"].tolist()]
    heads = {name: data[f"{name}_identities"].reshape(len(users), -1) for name in SOURCES}
    targets = np.array([target_by_user[user] for user in users], dtype=object)

    results: dict[str, Any] = {}
    for depth in DEPTHS:
        hits: dict[str, np.ndarray] = {}
        ranks: dict[str, np.ndarray] = {}
        for name in SOURCES:
            hits[name], ranks[name] = _hits(heads[name], targets, depth)
        union = np.zeros(len(users), dtype=bool)
        for name in SOURCES:
            union |= hits[name]

        per_source = {}
        for name in SOURCES:
            others = np.zeros(len(users), dtype=bool)
            for other in SOURCES:
                if other != name:
                    others |= hits[other]
            hit_ranks = ranks[name][ranks[name] > 0]
            per_source[name] = {
                "hits": int(hits[name].sum()),
                "hit_rate": round(float(hits[name].mean()), 4),
                "unique_hits": int((hits[name] & ~others).sum()),
                "mean_rank_when_hit": round(float(hit_ranks.mean()), 2)
                if hit_ranks.size
                else None,
                "distinct_identity_ratio": _distinct_ratio(heads[name], depth),
            }

        pairs = {}
        for left, right in (
            ("history", "catalog_search"),
            ("history", "two_tower"),
            ("two_tower", "catalog_search"),
            ("history", "similar_item"),
        ):
            both = int((hits[left] & hits[right]).sum())
            left_only = int((hits[left] & ~hits[right]).sum())
            right_only = int((hits[right] & ~hits[left]).sum())
            neither = int((~hits[left] & ~hits[right]).sum())
            jaccard = []
            for index in range(len(users)):
                a = set(heads[left][index, :depth].tolist())
                b = set(heads[right][index, :depth].tolist())
                union_size = len(a | b)
                jaccard.append(len(a & b) / union_size if union_size else 1.0)
            pairs[f"{left}_vs_{right}"] = {
                "both": both,
                "left_only": left_only,
                "right_only": right_only,
                "neither": neither,
                "mean_head_jaccard": round(sum(jaccard) / len(jaccard), 3),
            }

        base = hits["history"] | hits["two_tower"]
        results[str(depth)] = {
            "users": len(users),
            "per_source": per_source,
            "union": {
                "hits": int(union.sum()),
                "hit_rate": round(float(union.mean()), 4),
            },
            "pairs": pairs,
            "catalog_unique_beyond_history_or_two_tower": int(
                (hits["catalog_search"] & ~base).sum()
            ),
            "similar_unique_beyond_history_two_tower_catalog": int(
                (hits["similar_item"] & ~(base | hits["catalog_search"])).sum()
            ),
        }

    summary = {
        "task": "next_item_prediction",
        "cohort": {
            "users": len(users),
            "source": "runs/m4_evidence/heads.npz",
            "semantics": "M4 frozen cohort, test_history = train_history + validation_target",
            "head_depth": 100,
        },
        "sources": list(SOURCES),
        "caveat": (
            "history is SASRec only, not the Agent's Serving V1 source; these are raw-source "
            "numbers, not Agent-composition numbers"
        ),
        "depths": results,
        "inputs": {
            "heads_sha256": _sha256(HEADS),
            "sequences_sha256": _sha256(SEQUENCES),
        },
        "python": sys.version.split()[0],
        "git_head": _git_head(),
        "driver_sha256": _sha256(Path(__file__)),
    }

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {}
    for name in ("summary.json",):
        payload = (args.out / name).read_bytes()
        manifest[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    (args.out / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
