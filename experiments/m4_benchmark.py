"""M4 benchmark runner (D7): the preregistered fixed-vs-adaptive source-selection comparison.

Implements ``docs/M4_PREREGISTRATION.md`` and nothing else. Every scientific parameter is read
from, or checked against, that document; nothing here may introduce a metric, an arm, a source, a
tuning step or a post-hoc choice.

Arms (preregistration §7)
-------------------------
* **fixed** — deterministic traversal of all four frozen sources at depth 100, then ``FINISH``.
* **adaptive** — the LLM chooses which unqueried source to consult next and when to ``FINISH``,
  under the same four sources, the same depth, the same budget and the same provider contract.

Both arms' final ranking is the ``CandidateLedger`` rank-fused order (§4), the evaluator is the
shared frozen implementation (§12.2), and the statistics are the inherited M2/M3 pair (§15).

Fail-closed design
------------------
The runner refuses to produce a result unless every frozen precondition holds. It does not repair,
default or warn-and-continue: a mismatch is a hard failure, because a silently different cohort,
head table or hash would make the numbers meaningless while still looking plausible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from recommendation.control.arguments import CandidateSource  # noqa: E402
from recommendation.evaluation.batched import evaluate_batched  # noqa: E402
from recommendation.evaluation.metrics import DEFAULT_K_VALUES  # noqa: E402

from experiments.m4_harness import (  # noqa: E402
    M4_DEFAULT_LIMITS,
    M4HarnessFactory,
    M4HeadTable,
    M4IdentityMap,
    build_identity_map,
)
from experiments.m4_policies import (  # noqa: E402
    FIXED_SOURCE_ORDER,
    FixedTraversalPolicy,
    GuardedAdaptivePolicy,
    ProtocolDeviation,
)
from experiments.m4_query_rules import deterministic_terms  # noqa: E402
from experiments.m4_sharding import merge_user_outputs  # noqa: E402

__all__ = [
    "M4Arms",
    "M4CohortRef",
    "load_frozen_heads",
    "load_frozen_cohort",
    "run_arm",
    "paired_statistics",
    "assert_preregistration_hash",
]

# -- frozen constants (preregistration §2, §3, §12, §15) -------------------- #

PREREGISTRATION = REPO / "docs/M4_PREREGISTRATION.md"
#: Recorded in the local freeze record; a change here means the protocol moved.
PREREGISTRATION_SHA256 = "7a18ea7258a3a175227a270ee38177c58bde66eadd51af8ccb02c91f83b981af"

ASSEMBLED_HEADS = REPO / "runs/m4_evidence/heads.npz"
HEADS_PROVENANCE = REPO / "runs/m4_evidence/heads_provenance.json"
M3_PAIRED_INPUTS = REPO / "runs/m3_execution/paired_inputs.npz"
SEQUENCES = REPO / "data/processed/Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO / "data/processed/Sports_and_Outdoors_mappings.json"

#: Frozen cohort size and seed (§12.2).
COHORT_SIZE = 20_000
COHORT_SEED = 20260201
#: Frozen depth (§3) and budget (§6).
DEPTH = 100
MAX_TOOL_CALLS = 4
MAX_STEPS = 6
#: Frozen K values (§12.2).
K_VALUES = tuple(DEFAULT_K_VALUES)
#: Bootstrap protocol inherited from M2/M3 (§15).
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260201

#: The frozen four-source universe (§2).
SOURCE_UNIVERSE: tuple[CandidateSource, ...] = (
    CandidateSource.CATALOG_SEARCH,
    CandidateSource.HISTORY,
    CandidateSource.SIMILAR_ITEM,
    CandidateSource.TWO_TOWER,
)


class M4ContractViolation(RuntimeError):
    """A frozen precondition failed. The runner stops; it never repairs or defaults."""


# -- hashing ---------------------------------------------------------------- #


def sha256_file(path: pathlib.Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_preregistration_hash(*, expected: str = PREREGISTRATION_SHA256) -> str:
    """Refuse to run unless the preregistration is byte-identical to the frozen one."""
    if not PREREGISTRATION.is_file():
        raise M4ContractViolation(f"missing preregistration {PREREGISTRATION}")
    got = sha256_file(PREREGISTRATION)
    if got != expected:
        raise M4ContractViolation(
            f"preregistration sha256 {got} != frozen {expected}; the protocol moved"
        )
    return got


# -- cohort ----------------------------------------------------------------- #


@dataclass(frozen=True)
class M4CohortRef:
    """The frozen cohort: cases in frozen order plus its identity hash."""

    cases: tuple[Any, ...]
    user_int_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    identity_sha256: str

    def __len__(self) -> int:
        return len(self.cases)


def cohort_identity_sha256(user_int_ids: Sequence[int], target_ids: Sequence[int]) -> str:
    """Hash the cohort's identity **and** order.

    Both arrays are included in frozen order, so a reordering or a substitution changes the digest.
    """
    combined = ";".join(
        f"{int(uid)}:{int(tid)}" for uid, tid in zip(user_int_ids, target_ids)
    )
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def load_frozen_cohort(*, limit: int | None = None) -> M4CohortRef:
    """Load the frozen 20,000-user cohort and verify it against the frozen M3 comparator artifact.

    Uses the repository's own frozen selection function so the cohort is not re-implemented. Refuses
    to continue on a size, identity or order mismatch.
    """
    from recommendation.evaluation.split import load_cohort_from_artifacts

    from experiments.benchmark_public import cohort_from_cases

    all_cases, report = load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
    selection = cohort_from_cases(all_cases, size=COHORT_SIZE, seed=COHORT_SEED)
    cases = list(selection["cases"])
    if len(cases) != COHORT_SIZE:
        raise M4ContractViolation(
            f"cohort selection produced {len(cases)} users, expected {COHORT_SIZE}"
        )

    user_int_ids = tuple(int(case.user_int_id) for case in cases)
    target_ids = tuple(int(case.test_target) for case in cases)
    if len(set(user_int_ids)) != len(user_int_ids):
        raise M4ContractViolation("cohort contains duplicate user ids")

    # Order/identity gate against the frozen M3 artifact.
    if not M3_PAIRED_INPUTS.is_file():
        raise M4ContractViolation(f"missing frozen cohort reference {M3_PAIRED_INPUTS}")
    frozen_targets = np.load(M3_PAIRED_INPUTS)["target_ids"]
    mine = np.asarray(target_ids, dtype=frozen_targets.dtype)
    if mine.shape != frozen_targets.shape:
        raise M4ContractViolation(
            f"cohort size {mine.shape[0]} != frozen {frozen_targets.shape[0]}"
        )
    if not np.array_equal(frozen_targets, mine):
        first = int(np.flatnonzero(frozen_targets != mine)[0])
        raise M4ContractViolation(
            "cohort identity/order mismatch vs frozen M3 artifact at index "
            f"{first}: frozen={int(frozen_targets[first])} selected={int(mine[first])}"
        )
    # The catalogue must still be the frozen one (§12.2).
    if int(report.catalog_size) <= 0:
        raise M4ContractViolation("catalogue size is not positive")

    digest = cohort_identity_sha256(user_int_ids, target_ids)
    if limit is not None:
        cases = cases[:limit]
        user_int_ids = user_int_ids[:limit]
        target_ids = target_ids[:limit]
    return M4CohortRef(
        cases=tuple(cases),
        user_int_ids=user_int_ids,
        target_ids=target_ids,
        identity_sha256=digest,
    )


# -- heads ------------------------------------------------------------------ #


def load_frozen_heads(
    cohort: M4CohortRef,
    *,
    path: pathlib.Path = ASSEMBLED_HEADS,
    verify_provenance: bool = True,
) -> M4HeadTable:
    """Load the assembled head table and verify it covers exactly the frozen cohort, in order.

    Fail-closed on: missing artifact, hash mismatch against its provenance sidecar, wrong user set,
    wrong user order, a source with no heads at all, or a head longer than the frozen depth.
    """
    if not path.is_file():
        raise M4ContractViolation(f"missing assembled heads {path}")
    if verify_provenance and not HEADS_PROVENANCE.is_file():
        raise M4ContractViolation(f"missing heads provenance {HEADS_PROVENANCE}")
    if verify_provenance:
        recorded = json.loads(HEADS_PROVENANCE.read_text(encoding="utf-8"))
        expected = recorded.get("sha256")
        got = sha256_file(path)
        if expected and got != expected:
            raise M4ContractViolation(
                f"assembled heads sha256 {got} != recorded {expected}"
            )

    z = np.load(path, allow_pickle=True)
    user_ids = np.asarray(z["user_int_ids"], dtype=np.int64)
    if user_ids.shape[0] != len(cohort):
        raise M4ContractViolation(
            f"head artifact holds {user_ids.shape[0]} users, cohort has {len(cohort)}"
        )
    if len(set(user_ids.tolist())) != len(user_ids):
        raise M4ContractViolation("head artifact contains duplicate users")
    if not np.array_equal(user_ids, np.asarray(cohort.user_int_ids, dtype=np.int64)):
        raise M4ContractViolation(
            "head artifact user order does not match the frozen cohort order"
        )

    heads: dict[CandidateSource, dict[int, tuple[tuple[str, int, float], ...]]] = {}
    queries: dict[CandidateSource, dict[int, str]] = {}
    score_kinds: dict[CandidateSource, str] = {}
    for source in SOURCE_UNIVERSE:
        key = source.value
        if f"{key}_lengths" not in z.files:
            raise M4ContractViolation(f"head artifact has no arrays for source {key!r}")
        lengths = np.asarray(z[f"{key}_lengths"], dtype=np.int64)
        identities = z[f"{key}_identities"]
        ranks = np.asarray(z[f"{key}_ranks"], dtype=np.int64)
        scores = np.asarray(z[f"{key}_scores"], dtype=np.float64)
        if len(lengths) != len(user_ids):
            raise M4ContractViolation(
                f"source {key!r}: {len(lengths)} length entries for {len(user_ids)} users"
            )
        if not (int(lengths.sum()) == len(ranks) == len(scores) == len(identities)):
            raise M4ContractViolation(
                f"source {key!r}: declared lengths {int(lengths.sum())} but "
                f"{len(ranks)}/{len(scores)}/{len(identities)} rank/score/identity rows"
            )
        if int(lengths.max()) > DEPTH:
            raise M4ContractViolation(
                f"source {key!r}: a head exceeds the frozen depth {DEPTH}"
            )

        per_user: dict[int, tuple[tuple[str, int, float], ...]] = {}
        cursor = 0
        for position, uid in enumerate(user_ids.tolist()):
            count = int(lengths[position])
            rows = tuple(
                (str(identities[cursor + i]), int(ranks[cursor + i]), float(scores[cursor + i]))
                for i in range(count)
            )
            cursor += count
            per_user[int(uid)] = rows
        heads[source] = per_user

        queries_key = f"{key}_queries"
        if queries_key in z.files:
            queries[source] = {
                int(uid): str(value)
                for uid, value in zip(user_ids.tolist(), z[queries_key].tolist())
            }
        else:
            queries[source] = {}
        score_kinds[source] = str(
            (json.loads(HEADS_PROVENANCE.read_text(encoding="utf-8")).get("score_kinds", {}) or {})
            .get(key, "unspecified")
            if HEADS_PROVENANCE.is_file()
            else "unspecified"
        )

    # Source universe gate (§2).
    if set(heads) != set(SOURCE_UNIVERSE):
        raise M4ContractViolation(
            f"head sources {sorted(s.value for s in heads)} != frozen universe "
            f"{sorted(s.value for s in SOURCE_UNIVERSE)}"
        )
    return M4HeadTable(
        heads=heads,
        score_kinds=score_kinds,
        queries=queries,
        indexed_records=int(z["catalog_size"]) if "catalog_size" in z.files else 0,
        search_fields=(
            "title",
            "store",
            "main_category",
            "categories",
            "features",
            "description",
        ),
    )


# -- arms ------------------------------------------------------------------- #


@dataclass
class M4Arms:
    """The two preregistered arms and the per-user records they produced."""

    rankings: dict[str, list[tuple[int, ...]]] = field(default_factory=dict)
    behaviours: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    cost: dict[str, dict[str, Any]] = field(default_factory=dict)


def _ranking_from_ledger(ledger: Any, item2id: Mapping[str, int]) -> tuple[int, ...]:
    """The run's own final ranking: the ledger's frozen rank-fused order, mapped and deduplicated.

    Identical semantics to the M3 arm: the ledger leads, nothing is re-sorted or re-scored, and an
    identity with no catalogue id is dropped rather than guessed at.
    """
    ranked: list[int] = []
    seen: set[int] = set()
    for identity in ledger.ranked_parent_asins():
        item_id = item2id.get(identity)
        if item_id is None or item_id <= 0 or item_id in seen:
            continue
        seen.add(item_id)
        ranked.append(int(item_id))
    return tuple(ranked)


def _behaviour(result: Any, ledger: Any, ranking: Sequence[int], tools: Mapping[Any, Any]) -> dict[str, Any]:
    """Per-user behaviour record, read from the run. Every field is read, none is inferred."""
    control = result.control
    reason = control.termination_reason
    reason_value = None if reason is None else str(getattr(reason, "value", reason))
    sources_present = sorted(str(s.value) for s in ledger.sources_present())
    queried = sorted(
        str(source.value) for source, tool in tools.items() if getattr(tool, "consumed", False)
    )
    actions: list[str] = []
    for step in getattr(result.trajectory, "steps", ()) or ():
        proposal = getattr(step, "action_proposal", None)
        if isinstance(proposal, dict):
            action = proposal.get("action")
            actions.append(str(getattr(action, "value", action)))
    return {
        "status": str(getattr(control.status, "value", control.status)),
        "termination_reason": reason_value,
        "succeeded": bool(result.succeeded),
        "steps": int(control.step_count),
        "tool_calls": int(control.tool_call_count),
        "retries": int(control.retry_count),
        "action_sequence": actions,
        "queried_sources": queried,
        "sources_present": sources_present,
        "source_count": len(sources_present),
        "candidates": len(ledger),
        "grounded_candidates": len(ledger.grounded_entries),
        "ranking_size": len(ranking),
        "empty_ranking": len(ranking) == 0,
    }


def run_arm(
    *,
    name: str,
    cohort: M4CohortRef,
    table: M4HeadTable,
    identity_map: M4IdentityMap,
    policy_factory: Callable[[], Any],
    user_message: str,
    progress_every: int = 2000,
) -> tuple[list[tuple[int, ...]], list[dict[str, Any]], dict[str, Any]]:
    """Run one arm over the whole frozen cohort, in frozen order.

    Fails closed if any user produces no behaviour record, or if the number of results does not
    equal the cohort size, so a silently dropped user can never reach the statistics.
    """
    item2id = identity_map.item2id
    factory = M4HarnessFactory(
        identity_map=identity_map, table=table, policy_factory=policy_factory
    )
    rankings: list[tuple[int, ...]] = []
    behaviours: list[dict[str, Any]] = []
    deviations: list[dict[str, Any]] = []
    started = time.time()

    for position, case in enumerate(cohort.cases):
        harness = factory(case)
        try:
            result = harness.controller.run(user_message, harness.trusted_history)
        except ProtocolDeviation as exc:
            # The frozen deviation rule (§9, §10): the user stays in the primary analysis and is
            # evaluated on whatever the ledger holds at that moment.
            deviations.append(
                {"position": position, "user_int_id": int(case.user_int_id), "reason": exc.reason}
            )
            result = None
        if result is None:
            ranking = _ranking_from_ledger(harness.ledger, item2id)
            behaviours.append(
                {
                    "status": "failed",
                    "termination_reason": "protocol_deviation",
                    "succeeded": False,
                    "steps": 0,
                    "tool_calls": 0,
                    "retries": 0,
                    "action_sequence": [],
                    "queried_sources": [],
                    "sources_present": sorted(
                        str(s.value) for s in harness.ledger.sources_present()
                    ),
                    "source_count": len(harness.ledger.sources_present()),
                    "candidates": len(harness.ledger),
                    "grounded_candidates": len(harness.ledger.grounded_entries),
                    "ranking_size": len(ranking),
                    "empty_ranking": len(ranking) == 0,
                }
            )
            rankings.append(ranking)
        else:
            ranking = _ranking_from_ledger(harness.ledger, item2id)
            rankings.append(ranking)
            behaviours.append(_behaviour(result, harness.ledger, ranking, harness.tools))
        if progress_every and position and position % progress_every == 0:
            print(f"  [{name}] {position}/{len(cohort)} users", flush=True)

    elapsed = time.time() - started
    if len(rankings) != len(cohort) or len(behaviours) != len(cohort):
        raise M4ContractViolation(
            f"arm {name!r} produced {len(rankings)} rankings / {len(behaviours)} behaviour "
            f"records for {len(cohort)} users"
        )
    if len(set(cohort.user_int_ids)) != len(cohort.user_int_ids):
        raise M4ContractViolation(f"arm {name!r}: cohort has duplicate users")

    queried_counts: dict[str, int] = {}
    for record in behaviours:
        for source in record["queried_sources"]:
            queried_counts[source] = queried_counts.get(source, 0) + 1
    summary = {
        "arm": name,
        "users": len(rankings),
        "wall_seconds": elapsed,
        "users_per_second": len(rankings) / elapsed if elapsed else 0.0,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
        "failures": sum(1 for b in behaviours if not b["succeeded"]),
        "empty_rankings": sum(1 for b in behaviours if b["empty_ranking"]),
        "total_tool_calls": sum(int(b["tool_calls"]) for b in behaviours),
        "total_steps": sum(int(b["steps"]) for b in behaviours),
        "total_retries": sum(int(b["retries"]) for b in behaviours),
        "queried_source_counts": queried_counts,
        "protocol_deviations": deviations,
        "protocol_deviation_count": len(deviations),
    }
    return rankings, behaviours, summary


# -- statistics (preregistration §14, §15) ---------------------------------- #


def _hit_at(ranking: Sequence[int], target: int, k: int) -> bool:
    """Whether ``target`` is inside the arm's own top-``k``."""
    return target in ranking[:k]


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value, no continuity correction (M2/M3 protocol)."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _ndcg_at(ranking: Sequence[int], target: int, k: int) -> float:
    """Single-positive NDCG@k: ``1/log2(rank+1)`` when the target is inside ``k``."""
    for position, item in enumerate(ranking[:k], start=1):
        if item == target:
            return 1.0 / np.log2(position + 1)
    return 0.0


def paired_statistics(
    *,
    arm_a: Sequence[Sequence[int]],
    arm_b: Sequence[Sequence[int]],
    targets: Sequence[int],
    k_values: Sequence[int] = K_VALUES,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """The preregistered paired comparison of two arms over the frozen cohort.

    ``arm_a`` is the fixed arm and ``arm_b`` the adaptive arm, so every delta is
    ``adaptive - fixed`` and a negative delta means the adaptive arm is worse.

    Reports, per K: hit counts, gained/lost, ``ΔRecall``, ``ΔNDCG``, their paired user-level
    percentile bootstrap 95% CIs (users resampled with replacement), and the exact McNemar p-value
    for the hit indicator. Also applies the classification rule declared in §15.
    """
    n = len(targets)
    if not (len(arm_a) == len(arm_b) == n):
        raise M4ContractViolation("paired statistics need equal-length arms")
    rng = np.random.default_rng(seed)
    per_k: dict[str, Any] = {}
    for k in k_values:
        hit_a = np.asarray([_hit_at(r, t, k) for r, t in zip(arm_a, targets)], dtype=bool)
        hit_b = np.asarray([_hit_at(r, t, k) for r, t in zip(arm_b, targets)], dtype=bool)
        ndcg_a = np.asarray([_ndcg_at(r, t, k) for r, t in zip(arm_a, targets)], dtype=np.float64)
        ndcg_b = np.asarray([_ndcg_at(r, t, k) for r, t in zip(arm_b, targets)], dtype=np.float64)

        gained = int(np.sum(hit_b & ~hit_a))
        lost = int(np.sum(hit_a & ~hit_b))
        recall_delta = float(hit_b.mean() - hit_a.mean())
        ndcg_delta = float(ndcg_b.mean() - ndcg_a.mean())

        # Paired user-level percentile bootstrap: resample users, recompute the paired mean delta.
        idx = rng.integers(0, n, size=(resamples, n))
        recall_boot = hit_b[idx].mean(axis=1) - hit_a[idx].mean(axis=1)
        ndcg_boot = ndcg_b[idx].mean(axis=1) - ndcg_a[idx].mean(axis=1)
        recall_ci = (
            float(np.percentile(recall_boot, 2.5)),
            float(np.percentile(recall_boot, 97.5)),
        )
        ndcg_ci = (
            float(np.percentile(ndcg_boot, 2.5)),
            float(np.percentile(ndcg_boot, 97.5)),
        )
        per_k[str(k)] = {
            "k": int(k),
            "fixed_hits": int(hit_a.sum()),
            "adaptive_hits": int(hit_b.sum()),
            "gained": gained,
            "lost": lost,
            "recall_delta": recall_delta,
            "recall_ci95": list(recall_ci),
            "ndcg_delta": ndcg_delta,
            "ndcg_ci95": list(ndcg_ci),
            "mcnemar_p": exact_mcnemar(lost, gained),
            "recall_ci_excludes_zero": bool(recall_ci[0] > 0 or recall_ci[1] < 0),
            "ndcg_ci_excludes_zero": bool(ndcg_ci[0] > 0 or ndcg_ci[1] < 0),
        }

    # Classification rule declared before results (§15).
    every_k_significant = all(
        per_k[str(k)]["mcnemar_p"] < 0.05
        and per_k[str(k)]["recall_ci_excludes_zero"]
        and per_k[str(k)]["ndcg_ci_excludes_zero"]
        for k in k_values
    )
    directions = {np.sign(per_k[str(k)]["ndcg_delta"]) for k in k_values if per_k[str(k)]["ndcg_delta"]}
    if every_k_significant:
        verdict = "SUPPORTED"
    elif len(directions) == 1:
        verdict = "DIRECTIONAL ONLY"
    else:
        verdict = "NO DETECTABLE DIFFERENCE"
    return {
        "n_users": n,
        "primary_endpoint": "NDCG@10",
        "delta_convention": "adaptive - fixed",
        "bootstrap": {"resamples": resamples, "seed": seed, "unit": "user"},
        "per_k": per_k,
        "verdict": verdict,
        "verdict_rule": (
            "SUPPORTED requires every K to have McNemar p<0.05 and both CIs excluding 0; "
            "DIRECTIONAL ONLY when the NDCG delta sign is consistent but not all conditions hold; "
            "otherwise NO DETECTABLE DIFFERENCE"
        ),
    }


# -- evaluation (shared frozen evaluator) ----------------------------------- #


def evaluate_arm_rankings(
    rankings: Sequence[Sequence[int]],
    *,
    cohort: M4CohortRef,
    num_items: int,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Score one arm with the **shared frozen evaluator**, ranking-convention preserved.

    The arm supplies a full ordering; the evaluator owns PAD exclusion, seen-item masking,
    tie-breaking, ranking and metrics. Strictly decreasing scores encode the arm's own order, which
    is the convention ``benchmark_public`` already uses for ranking-based arms.

    A ranking that is empty or contains an out-of-catalogue or PAD id is a malformed output and is
    refused rather than silently scored.
    """
    import torch

    if len(rankings) != len(cohort):
        raise M4ContractViolation(
            f"arm produced {len(rankings)} rankings for a cohort of {len(cohort)}"
        )
    for position, ranking in enumerate(rankings):
        if len(set(ranking)) != len(ranking):
            raise M4ContractViolation(f"ranking {position} contains duplicate item ids")
        for item in ranking:
            if not (1 <= int(item) <= num_items):
                raise M4ContractViolation(
                    f"ranking {position} contains item {item} outside 1..{num_items} (PAD or OOR)"
                )

    def batches() -> Iterable[tuple[Sequence[Sequence[int]], Sequence[int], Any]]:
        for start in range(0, len(rankings), batch_size):
            stop = min(start + batch_size, len(rankings))
            scores = torch.full((stop - start, num_items + 1), -1.0, dtype=torch.float64)
            for row, position in enumerate(range(start, stop)):
                ranking = rankings[position]
                for rank, item in enumerate(ranking):
                    scores[row, int(item)] = float(len(ranking) - rank)
            histories = [() for _ in range(start, stop)]
            targets = [int(cohort.target_ids[p]) for p in range(start, stop)]
            yield histories, targets, scores

    result = evaluate_batched(
        num_items=num_items,
        score_batches=batches(),
        k_values=K_VALUES,
        cohort="test",
        protocol="temporal_leave_two_out",
        device="cpu",
    )
    report = result.report
    metrics = {
        name: {str(k): float(v) for k, v in dict(values).items()}
        for name, values in report.metrics.items()
    }
    return {
        "users": int(report.num_cases),
        "metrics": metrics,
        "catalog_size": int(report.catalog_size),
        "mean_target_rank": (
            None if report.mean_target_rank is None else float(report.mean_target_rank)
        ),
        "mean_num_candidates": (
            None if report.mean_num_candidates is None else float(report.mean_num_candidates)
        ),
        "num_batches": int(result.num_batches),
        "seconds": float(result.seconds),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="M4 fixed vs adaptive source selection")
    parser.add_argument("--limit", type=int, default=None, help="preflight only; not the cohort")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument(
        "--fixed-only",
        action="store_true",
        help="run the deterministic arm only (no provider); for wiring checks",
    )
    args = parser.parse_args(argv)

    prereg = assert_preregistration_hash()
    cohort = load_frozen_cohort(limit=args.limit)
    print(
        f"cohort={len(cohort)} identity_sha256={cohort.identity_sha256[:16]}… "
        f"prereg={prereg[:16]}…",
        flush=True,
    )
    identity_map = build_identity_map(MAPPINGS)
    table = load_frozen_heads(cohort)
    print(f"heads loaded for {len(SOURCE_UNIVERSE)} sources", flush=True)

    user_message = "recommend something useful for me"
    fixed_rankings, fixed_behaviour, fixed_summary = run_arm(
        name="fixed",
        cohort=cohort,
        table=table,
        identity_map=identity_map,
        policy_factory=FixedTraversalPolicy,
        user_message=user_message,
    )
    print(f"fixed arm done: {fixed_summary['wall_seconds']:.1f}s", flush=True)
    out: dict[str, Any] = {
        "preregistration_sha256": prereg,
        "cohort_identity_sha256": cohort.identity_sha256,
        "cohort_users": len(cohort),
        "fixed": fixed_summary,
    }
    if not args.fixed_only:
        raise SystemExit(
            "the adaptive arm needs a provider configuration; run with --fixed-only for wiring "
            "checks, or supply the frozen provider configuration before the formal execution"
        )
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
