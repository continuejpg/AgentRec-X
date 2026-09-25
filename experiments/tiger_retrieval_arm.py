"""Step 2.6 AgentRec-X-side arm: ``tiger_fp32``.

This is the AgentRec-X half of the Step-2.6 boundary.  It lives **outside** ``backends/`` and
``recommendation/backends/``, which is the only place canonical identity may be attached.

What it does
------------
* consumes the sparse retrieval artifact written by the backend producer
  (``tiger_public.retrieve_cli``) and **refuses** any artifact from another producer, so the
  Stage-4 stub scorer can never feed a certified TIGER metric;
* materialises the ``[batch, num_items + 1]`` score matrix the shared evaluator wants, in
  batches, with every non-frontier item on a documented floor score;
* re-attaches ``parent_asin`` here, and only here, and runs the AgentRec-X
  :class:`~recommendation.control.grounding.GroundingVerifier` over the resolved identities;
* keeps ``tiger_fp32`` unmistakably distinct from ``recommendation.semantic_id`` (frozen
  GenRec-v0), which it neither imports nor overwrites.

What it does not do
-------------------
It never masks, never ranks for presentation and never computes a metric.  Ranking, tie-breaking
and Recall/NDCG/HR all belong to ``recommendation.evaluation.batched``.  It forwards each case's
``test_history`` exactly as every accepted arm does, because the evaluator masks precisely the
history it is handed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

#: The only producer this arm will consume from.
EXPECTED_PRODUCER = "tiger_public.retrieve_cli"
#: The arm name that must stay distinct from GenRec-v0 in any results table.
ARM_NAME = "tiger_fp32"

DEFAULT_MAPPINGS = Path("data/processed/Sports_and_Outdoors_mappings.json")
DEFAULT_PRODUCTS = Path("data/processed/Sports_and_Outdoors_products.jsonl")


class ArmError(ValueError):
    """Raised when the arm is handed something it must not silently accept."""


class IdentityMapAdapter:
    """``IdentityMapLike`` over the frozen preprocessing mappings (item id <-> parent_asin)."""

    def __init__(self, mappings_path: str | Path = DEFAULT_MAPPINGS) -> None:
        payload = json.loads(Path(mappings_path).read_text(encoding="utf-8"))
        self.path = Path(mappings_path)
        self.num_items = int(payload["num_items"])
        self._item2id: dict[str, int] = payload["item2id"]
        self._id2item: list[str | None] = payload["id2item"]

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._item2id

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        if parent_asin not in self._item2id:
            raise ArmError(f"unknown parent_asin {parent_asin!r}")
        return int(self._item2id[parent_asin])

    def item_id_to_parent_asin(self, item_id: int) -> str:
        if not 1 <= int(item_id) <= self.num_items:
            raise ArmError(f"item_id {item_id} is outside the catalogue 1..{self.num_items}")
        value = self._id2item[int(item_id)]
        if value is None:
            raise ArmError(f"item_id {item_id} has no parent_asin in the mapping")
        return value


class CatalogueAdapter:
    """``CatalogLike`` over the accepted products artifact (membership by parent_asin)."""

    def __init__(self, products_path: str | Path = DEFAULT_PRODUCTS) -> None:
        self.path = Path(products_path)
        self._idents: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                identity = json.loads(line).get("parent_asin")
                if identity:
                    self._idents.add(str(identity))

    def __contains__(self, parent_asin: object) -> bool:
        return str(parent_asin) in self._idents

    def __len__(self) -> int:
        return len(self._idents)


# --------------------------------------------------------------------------- #
# Artifact loading (fail closed)
# --------------------------------------------------------------------------- #


def load_retrieval_artifact(
    candidates_path: str | Path,
    summary_path: str | Path,
    *,
    require_certified: bool = True,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Load a producer artifact, refusing anything that is not a certified TIGER retrieval."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if summary.get("producer") != EXPECTED_PRODUCER:
        raise ArmError(
            f"refusing a retrieval artifact from producer {summary.get('producer')!r}; the arm "
            f"consumes {EXPECTED_PRODUCER!r} only, so the Stage-4 stub can never feed a "
            "certified TIGER metric"
        )
    if require_certified:
        if summary.get("status") != "CERTIFIED":
            raise ArmError(
                f"a canonical comparison requires CERTIFIED retrieval, got "
                f"status={summary.get('status')!r} (mode_requested="
                f"{summary.get('mode_requested')!r}); APPROXIMATE output is NON-CANONICAL"
            )
        if not summary.get("comparison_eligible"):
            raise ArmError(
                "the artifact reports comparison_eligible=false; a budget-exhausted run is never "
                "a canonical TIGER result"
            )
    by_case: dict[int, dict[str, Any]] = {}
    with Path(candidates_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_case[int(record["case_id"])] = record
    return summary, by_case


def arm_tiger_fp32(
    *,
    candidates_path: str | Path,
    summary_path: str | Path,
    num_items: int,
    cohort: str = "test",
    require_certified: bool = True,
    device: str = "cpu",
):
    """The evaluator-facing batch factory.

    Returns a callable ``(cases, batch_size) -> Iterator[(histories, targets, scores)]``, the
    retrieval-arm form :func:`experiments.benchmark_public.evaluate_arm` detects by signature.
    ``scores`` is ``[batch, num_items + 1]`` with index 0 reserved for PAD; the evaluator owns
    masking, tie-breaking and ranking.
    """
    summary, by_case = load_retrieval_artifact(
        candidates_path, summary_path, require_certified=require_certified
    )
    tail_score = float(summary["tail_score"])
    artifact_items = int(summary["num_items"])
    if artifact_items != int(num_items):
        raise ArmError(
            f"the artifact was produced against {artifact_items} items but the evaluator has "
            f"{num_items}; refusing to compare across catalogue sizes"
        )

    def batches(cases: Sequence[Any], batch_size: int) -> Iterator[tuple[list, list, torch.Tensor]]:
        if batch_size < 1:
            raise ArmError(f"batch_size must be >= 1, got {batch_size}")
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            width = len(chunk)
            scores = torch.full((width, num_items + 1), tail_score, dtype=torch.float32)
            scores[:, 0] = 0.0  # PAD, excluded positionally by the evaluator
            histories: list[Sequence[int]] = []
            targets: list[int] = []
            for row, case in enumerate(chunk):
                case_id = start + row
                record = by_case.get(case_id)
                if record is None:
                    raise ArmError(f"the artifact has no record for case_id {case_id}")
                if record.get("status") != "CERTIFIED" and require_certified:
                    raise ArmError(
                        f"case {case_id} returned status {record.get('status')!r}, not CERTIFIED"
                    )
                for candidate in record["candidates"]:
                    item_id = int(candidate["item_id"])
                    if not 1 <= item_id <= num_items:
                        raise ArmError(f"case {case_id}: item_id {item_id} is out of catalogue")
                    scores[row, item_id] = float(candidate["score"])
                # The evaluator masks exactly the history it is handed, so the history must be
                # the items strictly BEFORE the target being ranked - i.e. `test_history`
                # (= train_history + validation_target) for the test cohort, which is what every
                # accepted arm forwards.  Forwarding `train_history` here would leave the
                # validation target unmasked and eligible, silently changing the metric.
                if cohort == "test":
                    histories.append(tuple(case.test_history))
                    targets.append(int(case.test_target))
                elif cohort == "validation":
                    histories.append(tuple(case.train_history))
                    targets.append(int(case.validation_target))
                else:
                    raise ArmError(
                        f"cohort must be 'test' or 'validation', got {cohort!r}; refusing to guess "
                        "which history the evaluator should mask"
                    )
            yield histories, targets, scores

    return batches


# --------------------------------------------------------------------------- #
# Identity re-attachment and grounding (AgentRec-X side; G14)
# --------------------------------------------------------------------------- #


def resolve_candidate_identities(
    candidates_path: str | Path,
    summary_path: str | Path,
    *,
    mappings_path: str | Path = DEFAULT_MAPPINGS,
    products_path: str | Path = DEFAULT_PRODUCTS,
    verify_grounding: bool = True,
) -> dict[str, Any]:
    """Resolve every returned item to ``parent_asin`` and ground the result.

    Retains, per candidate: ``parent_asin``, ``semantic_id``, score, rank, ``retrieval_mode``,
    the checkpoint hash, the SID artifact hash and the resolver provenance - the retention list
    the Step-2.6 task requires, kept on the side of the boundary that is allowed to hold
    canonical identity.
    """
    summary, by_case = load_retrieval_artifact(candidates_path, summary_path)
    identity = IdentityMapAdapter(mappings_path)
    catalogue = CatalogueAdapter(products_path) if verify_grounding else None
    hashes = summary.get("hashes") or {}

    ledger: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    out_of_range = 0
    for case_id in sorted(by_case):
        record = by_case[case_id]
        for candidate in record["candidates"]:
            item_id = int(candidate["item_id"])
            if not 1 <= item_id <= identity.num_items:
                out_of_range += 1
                unresolved.append({"case_id": case_id, "item_id": item_id, "reason": "range"})
                continue
            try:
                parent_asin = identity.item_id_to_parent_asin(item_id)
            except ArmError as error:
                unresolved.append({"case_id": case_id, "item_id": item_id, "reason": str(error)})
                continue
            ledger.append({
                "case_id": case_id,
                "rank": int(candidate["rank"]),
                "item_id": item_id,
                "parent_asin": parent_asin,
                "semantic_id": candidate["semantic_id"],
                "score": float(candidate["score"]),
                "retrieval_mode": summary["status"],
                "model_checkpoint_sha256": hashes.get("checkpoint_tiger_pt"),
                "sid_semantic_ids_sha256": hashes.get("sid_semantic_ids"),
                "sid_layout_sha256": hashes.get("sid_layout"),
                "generator_layout_sha256": hashes.get("generator_layout"),
                "resolver": {
                    "candidate_item_for_path": "tiger_public.retrieve.SidResolver",
                    "resolution": "token_path -> integer item_id (trie leaf)",
                    "identity_reattachment": "recommendation layer via the frozen mappings",
                    "collision_policy": "smallest_item_id_wins",
                },
            })

    report: dict[str, Any] = {
        "candidates": len(ledger),
        "unresolved_candidates": len(unresolved),
        "unresolved_examples": unresolved[:10],
        "out_of_range": out_of_range,
        "identity_map": str(mappings_path),
        "identity_map_parent_asins": len(identity._item2id),  # noqa: SLF001 - provenance only
        "distinct_items": len({row["item_id"] for row in ledger}),
        "distinct_parent_asins": len({row["parent_asin"] for row in ledger}),
        "artifact_status": summary["status"],
        "comparison_eligible": summary["comparison_eligible"],
        "checkpoint_sha256": hashes.get("checkpoint_tiger_pt"),
    }
    if verify_grounding:
        from recommendation.control.grounding import GroundingVerifier

        verifier = GroundingVerifier(identity, catalogue)
        report["checks_catalog"] = verifier.checks_catalog
        report["catalog_size"] = len(catalogue) if catalogue is not None else None
        grounded: set[str] = set()
        ungrounded: list[str] = []
        seen: set[str] = set()
        for row in ledger:
            if row["parent_asin"] in seen:
                continue
            seen.add(row["parent_asin"])
            result = verifier.ground([row["parent_asin"]])
            if result.grounded:
                grounded |= set(result.grounded)
            for item in result.ungrounded or ():
                ungrounded.append(getattr(item, "parent_asin", row["parent_asin"]))
        report["grounded_distinct"] = len(grounded)
        report["ungrounded_distinct"] = len(ungrounded)
        report["ungrounded_examples"] = ungrounded[:10]
        report["grounding_all_accepted"] = bool(
            len(ungrounded) == 0 and len(grounded) == len(seen)
        )
    return {"summary": summary, "identity_report": report, "ledger": ledger}
