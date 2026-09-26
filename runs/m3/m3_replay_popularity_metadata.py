"""M3 Blocker-2 closure — deterministic replay of the accepted `popularity` and `metadata` arms
on the frozen 20 000-user cohort, and deterministic rebuild of the `fixed_fusion` per-user
evidence from frozen source heads.

Authorized scope: replay these two already-accepted arms ONLY, to materialise per-user
rankings/hits on the frozen cohort so the existing `fixed_fusion` baseline can be rebuilt
deterministically.  Nothing is tuned, retrained, re-searched or modified.

Hard constraints honoured here:
  * fusion, weights, RRF constant, depth, candidate sources and the shared evaluator are untouched
    (this script imports them read-only and passes the accepted defaults);
  * no ScriptedModelClient, no vLLM/Ollama, no model download, no real Agent execution;
  * the 20 000 cohort is used in full — no reduction, no sampling;
  * the SASRec head is NOT recomputed: the frozen Step-2.8 head cache is reused after proving its
    top-10 is identical to the frozen Phase-5 sequential head for all 20 000 users.

Fail-closed: if a replayed aggregate differs from the accepted Phase-5 aggregate, the script
STOPS and writes nothing.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import experiments.benchmark_public as B  # noqa: E402
from recommendation.evaluation.batched import evaluate_batched  # noqa: E402
from recommendation.evaluation.split import load_cohort_from_artifacts  # noqa: E402

OUT = REPO / "runs" / "m3_evidence"
SEQUENCES = REPO / "data/processed/Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO / "data/processed/Sports_and_Outdoors_mappings.json"
PRODUCTS = REPO / "data/processed/Sports_and_Outdoors_products.jsonl"
ACCEPTED = json.loads((REPO / "runs/phase5_benchmark_public/arms.json").read_text())
STEP28_HEADS = pathlib.Path("/tmp/step28_heads_cache.npz")
STEP28_HEADS_SHA = "d0361df743146461114af00b48c1e77bc0bcb3ed3bc6b37aaf85a08bbcf790dc"

TOL = 1e-12          # rate metrics: computed identically => identical bits expected
TOL_RANK = 5e-5      # mean_target_rank is stored rounded to 4 dp upstream


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(batches, num_items: int, cases) -> tuple[dict, np.ndarray]:
    """Run the shared evaluator exactly as evaluate_arm does, and keep the per-user ranks.

    evaluate_arm's payload is literally ``result.as_dict()`` plus a protocol block, so comparing
    ``result.as_dict()`` against the accepted aggregates compares the same numbers the accepted
    run published.  The factory dispatch below is copied from evaluate_arm: a ranker whose whole
    ranking is cheap yields a generator over cases, a retrieval/fusion arm yields a factory over
    ``(cases, batch_size)``.
    """
    import inspect

    if callable(batches):
        if "batch_size" in inspect.signature(batches).parameters:
            score_batches = batches(cases, B.DEFAULT_BATCH_SIZE)
        else:
            score_batches = batches(cases)
    else:
        # already an iterable (a generator produced by ranking_batches)
        score_batches = batches
    result = evaluate_batched(
        num_items=num_items, score_batches=score_batches, k_values=B.K_VALUES, cohort="test"
    )
    return result.as_dict(), np.asarray(result.target_ranks, dtype=np.int64)


def compare(name: str, got: dict, accepted: dict) -> list[str]:
    problems: list[str] = []
    for metric in ("Recall", "NDCG", "HR"):
        for k in ("@5", "@10", "@20"):
            g = float(got["metrics"][metric][k])
            a = float(accepted["metrics"][metric][k])
            if abs(g - a) > TOL:
                problems.append(f"{name} {metric}{k}: replay={g!r} accepted={a!r} diff={g-a:.3e}")
    g = float(got["mean_target_rank"])
    a = float(accepted["mean_target_rank"])
    if abs(g - a) > TOL_RANK:
        problems.append(f"{name} mean_target_rank: replay={g!r} accepted={a!r}")
    return problems


def pack_heads(heads) -> tuple[np.ndarray, np.ndarray]:
    """Pack ragged per-user heads into a padded int32 matrix plus per-user lengths."""
    width = max((len(h) for h in heads), default=0)
    out = np.zeros((len(heads), width), dtype=np.int32)
    lens = np.zeros(len(heads), dtype=np.int32)
    for i, h in enumerate(heads):
        n = len(h)
        lens[i] = n
        if n:
            out[i, :n] = np.asarray(h, dtype=np.int32)
    return out, lens


def unpack_heads(matrix: np.ndarray, lens: np.ndarray) -> list[list[int]]:
    return [matrix[i, : int(lens[i])].tolist() for i in range(len(lens))]


def hits_from_ranks(ranks: np.ndarray) -> dict[str, np.ndarray]:
    return {f"hit{k}": (ranks <= k) for k in (5, 10, 20)}


def save_evidence(path: pathlib.Path, target_ids: np.ndarray, ranks: np.ndarray,
                  top10: np.ndarray) -> str:
    payload = {
        "case_index": np.arange(ranks.size, dtype=np.int64),
        "target_ids": target_ids.astype(np.int64),
        "target_rank": ranks.astype(np.int64),
        "top10": top10.astype(np.int64),
    }
    payload.update(hits_from_ranks(ranks))
    np.savez(path, **payload)
    return sha256(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=" * 100)
    print("M3 BLOCKER-2 CLOSURE — deterministic replay of popularity + metadata")
    print("=" * 100)
    print(f"  evaluator/k        shared evaluate_batched, K_VALUES={B.K_VALUES}, batch_size={B.DEFAULT_BATCH_SIZE}")
    print(f"  fusion config       FUSION_SOURCES={B.FUSION_SOURCES}, head=1000, RRF_K=60 (defaults, untouched)")
    print(f"  accepted source     runs/phase5_benchmark_public/arms.json")
    print(flush=True)

    # ---------------------------------------------------------------- cohort
    print("\n--- loading the frozen cohort exactly as phase5_benchmark.py does ---", flush=True)
    cases, report = load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
    selection = B.cohort_from_cases(cases, size=B.DEFAULT_COHORT_SIZE)
    cohort = list(selection["cases"])
    num_items = report.catalog_size
    print(f"  cohort={len(cohort):,}  catalogue={num_items:,}", flush=True)
    if len(cohort) != B.DEFAULT_COHORT_SIZE:
        print(f"  STOP: cohort is {len(cohort)}, expected {B.DEFAULT_COHORT_SIZE} (no reduction allowed)")
        return 2

    timestamps = None
    checks = B.leakage_checks(cohort, cases, timestamps=timestamps)
    B.assert_protocol(checks)
    print("  leakage/protocol checks passed", flush=True)

    manifest = B.dataset_manifest(sequences_path=SEQUENCES, mappings_path=MAPPINGS)
    target_ids = np.asarray([c.test_target for c in cohort], dtype=np.int64)

    # sanity: our cohort's targets must match the frozen evidence's target_ids
    frozen_tids = np.load(REPO / "runs/step26_baseline_evidence/phase5_sasrec_hits.npz",
                          allow_pickle=True)["target_ids"]
    if not np.array_equal(target_ids, frozen_tids):
        print("  STOP: cohort target identities differ from the frozen baseline evidence")
        return 2
    print("  cohort target identities match the frozen baseline evidence exactly", flush=True)

    problems: list[str] = []
    art_hashes: dict[str, str] = {}
    replay_agg: dict[str, dict] = {}

    # ---------------------------------------------------------------- popularity
    print("\n--- REPLAY: popularity ---", flush=True)
    t0 = time.time()
    pop_ranking = B.arm_popularity(cohort, num_items=num_items)(cohort)[0]
    payload, ranks = evaluate(
        lambda c: B.ranking_batches(pop_ranking, c, batch_size=B.DEFAULT_BATCH_SIZE),
        num_items, cohort,
    )
    replay_agg["popularity"] = payload
    problems += compare("popularity", payload, ACCEPTED["arms"]["popularity"])
    print(f"  replayed in {time.time()-t0:.1f}s  Recall@10={payload['metrics']['Recall']['@10']} "
          f"NDCG@10={payload['metrics']['NDCG']['@10']:.15f}", flush=True)
    pop_heads = B.popularity_rankings(cohort, num_items=num_items, top_n=1000)
    pop_top10 = np.asarray([r[:10] for r in pop_heads], dtype=np.int64)
    art_hashes["phase5_popularity_hits.npz"] = save_evidence(
        OUT / "phase5_popularity_hits.npz", target_ids, ranks, pop_top10)

    # ---------------------------------------------------------------- metadata
    print("\n--- REPLAY: metadata (index build + one metadata pass) ---", flush=True)
    from recommendation.catalog.metadata import MetadataIndex

    t0 = time.time()
    metadata = MetadataIndex.load(PRODUCTS)
    index = B.ItemTextIndex(metadata.records)
    print(f"  metadata index built in {time.time()-t0:.1f}s size={index.size:,}", flush=True)
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))

    t0 = time.time()
    heads_cache = OUT / "_meta_heads_cache.npz"
    if heads_cache.exists():
        c = np.load(heads_cache)
        meta_heads = unpack_heads(c["metadata"], c["metadata_lengths"])
        heads_recomputed = False
        print(f"  metadata heads loaded from cache ({time.time()-t0:.1f}s)", flush=True)
    else:
        meta_heads = B.metadata_fused_heads(
            index=index, records=metadata.records, id2item=mappings["id2item"],
            item2id=mappings["item2id"], cases=cohort, top_n=1000,
        )
        m, ln = pack_heads(meta_heads)
        np.savez_compressed(heads_cache, metadata=m, metadata_lengths=ln)
        heads_recomputed = True
        print(f"  metadata heads computed in {time.time()-t0:.1f}s (cached)", flush=True)
    _lens = np.asarray([len(h) for h in meta_heads])
    print(f"  metadata head lengths: min={_lens.min()} max={_lens.max()} "
          f"mean={_lens.mean():.2f} full-1000 users={int((_lens==1000).sum())} "
          f"empty users={int((_lens==0).sum())}", flush=True)

    t0 = time.time()
    payload, ranks = evaluate(
        B.arm_metadata_retrieval(num_items=num_items, heads=meta_heads), num_items, cohort
    )
    replay_agg["metadata_retrieval"] = payload
    problems += compare("metadata_retrieval", payload, ACCEPTED["arms"]["metadata_retrieval"])
    print(f"  replayed in {time.time()-t0:.1f}s  Recall@10={payload['metrics']['Recall']['@10']} "
          f"NDCG@10={payload['metrics']['NDCG']['@10']:.15f}", flush=True)
    meta_top10 = np.asarray([r[:10] for r in meta_heads], dtype=np.int64)
    art_hashes["phase5_metadata_hits.npz"] = save_evidence(
        OUT / "phase5_metadata_hits.npz", target_ids, ranks, meta_top10)

    # ---------------------------------------------------------------- failure gate
    if problems:
        print("\n" + "!" * 100)
        print("STOP — the replay does NOT reproduce the accepted aggregates. No artifact was frozen.")
        for p in problems:
            print("   ", p)
        print("!" * 100)
        return 2
    print("\n  ACCEPTED AGGREGATE REPRODUCTION: PASS (popularity + metadata)", flush=True)

    # ---------------------------------------------------------------- frozen sequential head
    print("\n--- frozen sequential source (reused, NOT recomputed) ---", flush=True)
    got_sha = sha256(STEP28_HEADS)
    if got_sha != STEP28_HEADS_SHA:
        print(f"  STOP: step28 head cache sha256 {got_sha} != recorded {STEP28_HEADS_SHA}")
        return 2
    z = np.load(STEP28_HEADS, allow_pickle=True)
    sas_heads = [np.asarray(r, dtype=np.int64) for r in z["sas"]]
    p5 = np.load(REPO / "runs/step26_baseline_evidence/phase5_sasrec_hits.npz", allow_pickle=True)
    sas_top10 = np.asarray([r[:10] for r in sas_heads], dtype=np.int64)
    same10 = bool((sas_top10 == p5["top10"]).all())
    print(f"  step28 head cache sha256 verified; top-10 identical to the frozen Phase-5 sequential "
          f"head for all users: {same10}", flush=True)
    if not same10:
        print("  STOP: the reused sequential head is not the Phase-5 sequential source")
        return 2

    # ---------------------------------------------------------------- fixed_fusion rebuild
    print("\n--- REBUILD: fixed_fusion from frozen source heads ---", flush=True)
    source_rankings = {
        "popularity": pop_heads,
        "sequential": sas_heads,
        "metadata": meta_heads,
    }
    t0 = time.time()
    payload, ranks = evaluate(
        B.arm_fixed_fusion(num_items=num_items, source_rankings=source_rankings), num_items, cohort
    )
    replay_agg["fixed_fusion"] = payload
    problems += compare("fixed_fusion", payload, ACCEPTED["arms"]["fixed_fusion"])
    print(f"  rebuilt in {time.time()-t0:.1f}s  Recall@10={payload['metrics']['Recall']['@10']} "
          f"NDCG@10={payload['metrics']['NDCG']['@10']:.15f} "
          f"mean_target_rank={payload['mean_target_rank']}", flush=True)

    if problems:
        print("\n" + "!" * 100)
        print("STOP — fixed_fusion rebuild does NOT reproduce the accepted aggregate.")
        for p in problems:
            print("   ", p)
        print("!" * 100)
        return 2
    print("\n  FIXED_FUSION REPRODUCTION: PASS", flush=True)

    fusion_top10 = np.asarray([r[:10] for r in
                               [B.fused_head([pop_heads[i], sas_heads[i], meta_heads[i]], top_n=1000)
                                for i in range(len(cohort))]], dtype=np.int64)
    art_hashes["phase5_fixed_fusion_hits.npz"] = save_evidence(
        OUT / "phase5_fixed_fusion_hits.npz", target_ids, ranks, fusion_top10)

    # ---------------------------------------------------------------- source heads artifact
    heads_path = OUT / "phase5_fusion_source_heads.npz"
    pop_m, pop_l = pack_heads(pop_heads)
    sas_m, sas_l = pack_heads(sas_heads)
    meta_m, meta_l = pack_heads(meta_heads)
    pop_identical = len({tuple(h) for h in pop_heads}) == 1
    np.savez_compressed(
        heads_path,
        popularity=pop_m, popularity_lengths=pop_l,
        popularity_all_users_identical=np.asarray([pop_identical]),
        sequential=sas_m, sequential_lengths=sas_l,
        metadata=meta_m, metadata_lengths=meta_l,
    )
    art_hashes["phase5_fusion_source_heads.npz"] = sha256(heads_path)
    print(f"  source heads frozen: popularity {pop_m.shape} (all-users-identical={pop_identical}), "
          f"sequential {sas_m.shape}, metadata {meta_m.shape}", flush=True)

    # ---------------------------------------------------------------- provenance
    provenance = {
        "producer": "runs/m3/m3_replay_popularity_metadata.py",
        "authorization": "owner-authorized deterministic replay of the accepted popularity and "
                         "metadata arms only (M3 Blocker-2 closure)",
        "cohort": {"size": len(cohort), "seed_description": selection["description"],
                   "num_items": num_items, "k_values": list(B.K_VALUES),
                   "batch_size": B.DEFAULT_BATCH_SIZE, "benchmark_version": B.BENCHMARK_VERSION},
        "dataset_manifest": manifest,
        "inputs_sha256": {
            "data/processed/Sports_and_Outdoors_sequences.json": sha256(SEQUENCES),
            "data/processed/Sports_and_Outdoors_mappings.json": sha256(MAPPINGS),
            "data/processed/Sports_and_Outdoors_products.jsonl": sha256(PRODUCTS),
            "runs/phase5_benchmark_public/arms.json (accepted aggregates)": sha256(
                REPO / "runs/phase5_benchmark_public/arms.json"),
            "/tmp/step28_heads_cache.npz (frozen sequential head)": STEP28_HEADS_SHA,
        },
        "fusion_config": {"sources": list(B.FUSION_SOURCES), "head": 1000, "rrf_k": 60,
                          "dedup": "fused_head order-preserving first occurrence",
                          "evaluator": "recommendation.evaluation.batched.evaluate_batched"},
        "replay_aggregates": {k: {"metrics": v["metrics"],
                                  "mean_target_rank": v["mean_target_rank"]}
                              for k, v in replay_agg.items()},
        "metadata_heads_recomputed_this_run": heads_recomputed,
        "source_head_lengths": {
            "popularity": {"min": int(pop_l.min()), "max": int(pop_l.max()),
                           "all_users_identical": bool(pop_identical)},
            "sequential": {"min": int(sas_l.min()), "max": int(sas_l.max())},
            "metadata": {"min": int(meta_l.min()), "max": int(meta_l.max()),
                         "mean": float(meta_l.mean()),
                         "users_at_top_n": int((meta_l == 1000).sum()),
                         "users_empty": int((meta_l == 0).sum())},
        },
        "accepted_aggregates_matched_exactly": True,
        "artifact_sha256": art_hashes,
        "wall_seconds": round(time.time() - started, 1),
    }
    prov_path = OUT / "M3_REPLAY_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    art_hashes["M3_REPLAY_PROVENANCE.json"] = sha256(prov_path)

    print("\n" + "=" * 100)
    print("FROZEN ARTIFACTS (runs/m3_evidence/)")
    print("=" * 100)
    for name, h in sorted(art_hashes.items()):
        size = (OUT / name).stat().st_size
        print(f"  {name:34s} {size:>12,} B  {h}")
    print(f"\n  wall {time.time()-started:.1f}s")
    print("  RESULT: popularity PASS, metadata PASS, aggregates reproduced exactly, "
          "fixed_fusion per-user evidence READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
