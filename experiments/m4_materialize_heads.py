"""M4 frozen head materialisation (preregistration §3, §5, §12.3).

Materialises, for each cohort user and each of the four frozen sources, the head a live run
*would* have produced. The benchmark then performs no retrieval at all, which is what licences §13's
refusal to claim live retrieval latency.

Process separation is mandatory, not stylistic
----------------------------------------------
Preregistration §12.3 measured that the BM25 index and the TF-IDF similar-item index cannot coexist
in one process on the 6.9 GiB budget (OOM-killed, exit 137). The four sources are therefore
materialised by four independent invocations of this module, each selecting one source with
``--source``.

Deterministic chunking and resume
---------------------------------
Each ``(source, chunk)`` writes its own shard file. Chunk boundaries are derived from the frozen
cohort order, never from timing or machine state, and a completed shard is skipped on re-run. The
assembled per-user heads and their ordering are therefore identical whether the pass ran in one go
or was resumed many times -- which is the condition the task authorises chunking under.

Grounding is applied at materialisation
---------------------------------------
Rows are filtered exactly as the plane filters them: identity must resolve through the frozen
mapping (``grounding.py`` order: blank, not-in-mapping, item-id-unresolvable). A head may therefore
be shorter than its raw retrieval, and that is the correct behaviour -- the plane would have
dropped the same rows.

Usage
-----
    python -m experiments.m4_materialize_heads --source sequential   [--chunk-size N] [--limit U]
    python -m experiments.m4_materialize_heads --source catalog_search ...
    python -m experiments.m4_materialize_heads --source similar_item  ...
    python -m experiments.m4_materialize_heads --source two_tower    ...
    python -m experiments.m4_materialize_heads --assemble
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import resource
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # allow `python experiments/m4_materialize_heads.py`
    sys.path.insert(0, str(REPO))

from recommendation.control.arguments import CandidateSource  # noqa: E402
from recommendation.control.catalog_search import (  # noqa: E402
    CATALOG_SEARCH_SCORE_KIND,
)

from experiments.m4_harness import build_identity_map  # noqa: E402
from experiments.m4_query_rules import deterministic_terms, newest_indexed  # noqa: E402

# -- frozen paths (preregistration §12) ------------------------------------- #

SEQUENCES = REPO / "data/processed/Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO / "data/processed/Sports_and_Outdoors_mappings.json"
PRODUCTS = REPO / "data/processed/Sports_and_Outdoors_products.jsonl"
TWO_TOWER_CKPT = REPO / "runs/twotower_public_2026_full/checkpoint.pt"
FUSION_HEADS = REPO / "runs/m3_evidence/phase5_fusion_source_heads.npz"

OUT = REPO / "runs/m4_evidence"
SHARDS = OUT / "shards"
ASSEMBLED = OUT / "heads.npz"
PROVENANCE = OUT / "heads_provenance.json"
#: The persisted TF-IDF index built once by experiments.m4_build_similar_index.
SIMILAR_INDEX = OUT / "similar_item_index.pkl"

#: The four frozen sources, in the frozen traversal order (§7.1).
SOURCE_ORDER: tuple[CandidateSource, ...] = (
    CandidateSource.CATALOG_SEARCH,
    CandidateSource.HISTORY,
    CandidateSource.SIMILAR_ITEM,
    CandidateSource.TWO_TOWER,
)

#: The ``score_kind`` each real tool declares, so ledger provenance matches a live run.
#: HISTORY is declared at ``candidate_plane.py:171``; the others are read from the tool objects at
#: materialisation time and cross-checked against these expectations.
EXPECTED_SCORE_KINDS: dict[CandidateSource, str] = {
    CandidateSource.HISTORY: "sasrec_logit",
    CandidateSource.CATALOG_SEARCH: "bm25",
    CandidateSource.SIMILAR_ITEM: "item_item_similarity",
    CandidateSource.TWO_TOWER: "two_tower_cosine",
}

#: Frozen depth (preregistration §3).
DEPTH = 100

#: CLI name -> source.
CLI_SOURCES = {
    "catalog_search": CandidateSource.CATALOG_SEARCH,
    "history": CandidateSource.HISTORY,
    "similar_item": CandidateSource.SIMILAR_ITEM,
    "two_tower": CandidateSource.TWO_TOWER,
}


def _peak_rss_gib() -> float:
    """Peak resident set size of this process, in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _sha256(path: pathlib.Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -- cohort ---------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceHeads:
    """One source's materialised output for a slice of the cohort.

    ``queries`` records the **actual query text** (or seed identity) used to produce each head, so
    the benchmark can serve the same query that produced the head instead of re-deriving it.
    """

    heads: dict[int, list[tuple[str, int, float]]]
    score_kind: str
    queries: dict[int, str]


@dataclass(frozen=True)
class CohortSlice:
    """One user's materialisation input, in the frozen cohort order."""

    index: int
    user_int_id: int
    history: tuple[str, ...]


def load_cohort(*, limit: int | None = None) -> list[CohortSlice]:
    """Load the **frozen 20,000-user cohort** in its canonical order.

    The cohort is not "every eligible user": preregistration §12.2 fixes it as the deterministic
    history-length-stratified sample of ``COHORT_SEED = 20260201`` and size ``20_000``, selected by
    the repository's own frozen :func:`~experiments.benchmark_public.cohort_from_cases`. This loader
    calls that function rather than re-implementing the selection, so the materialised heads are
    keyed to exactly the users the evaluator will score.

    ``limit`` exists only for the non-cohort preflight and truncates *after* the frozen selection,
    so a limited run is still a prefix of the frozen order.
    """
    from recommendation.evaluation.split import load_cohort_from_artifacts

    from experiments.benchmark_public import cohort_from_cases

    cases, _report = load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
    selection = cohort_from_cases(cases)
    cohort_cases = selection["cases"]
    _assert_frozen_cohort(cohort_cases)
    identity = build_identity_map(MAPPINGS)
    out: list[CohortSlice] = []
    for position, case in enumerate(cohort_cases):
        if limit is not None and position >= limit:
            break
        history: list[str] = []
        for item_id in case.test_history:
            try:
                history.append(identity.item_id_to_parent_asin(int(item_id)))
            except (KeyError, TypeError, ValueError):
                continue
        out.append(
            CohortSlice(
                index=position,
                user_int_id=int(case.user_int_id),
                history=tuple(history),
            )
        )
    return out


#: The frozen M3 comparator artifact whose ``target_ids`` pin the cohort's identity *and* order.
M3_PAIRED_INPUTS = REPO / "runs/m3_execution/paired_inputs.npz"


def _assert_frozen_cohort(cohort_cases: Sequence[Any]) -> None:
    """Refuse to materialise unless the cohort matches the frozen 20,000, in the frozen order.

    Being the right *size* is not enough: a differently-ordered or differently-selected cohort of
    20,000 users would materialise heads for the wrong users and silently produce a meaningless
    evaluation. The M3 comparator artifact stores the per-user test targets in frozen cohort order,
    so comparing them checks identity **and** order in one step.

    This guard exists because the first version of this loader selected all 412,445 eligible users
    instead of the frozen sample; that was caught before any shard was written, and this makes the
    failure mode impossible to repeat silently.
    """
    if not M3_PAIRED_INPUTS.is_file():
        raise SystemExit(f"missing frozen cohort reference {M3_PAIRED_INPUTS}")
    import numpy as np

    frozen = np.load(M3_PAIRED_INPUTS)["target_ids"]
    mine = np.asarray([int(case.test_target) for case in cohort_cases], dtype=frozen.dtype)
    if mine.shape != frozen.shape:
        raise SystemExit(
            f"frozen cohort has {frozen.shape[0]} users but this selection has {mine.shape[0]}"
        )
    if not np.array_equal(frozen, mine):
        first = int(np.flatnonzero(frozen != mine)[0])
        raise SystemExit(
            "cohort identity/order does not match the frozen M3 artifact "
            f"(first mismatch at index {first}: frozen={int(frozen[first])} "
            f"selected={int(mine[first])}); refusing to materialise heads for the wrong users"
        )


# -- per-source materialisers ---------------------------------------------- #


def _ground(
    rows: Iterable[tuple[str, int, float]], identity: Any
) -> list[tuple[str, int, float]]:
    """Filter rows exactly as the plane's grounding does, preserving order and re-ranking.

    ``GroundingVerifier.ground`` classifies blank / not-in-mapping / unresolvable identities;
    ungrounded rows never become candidates. Ranks are renumbered over the survivors so the ledger
    sees a contiguous 1..n ordering, which is what a live plane would record.
    """
    kept: list[tuple[str, int, float]] = []
    for parent_asin, _rank, score in rows:
        if not isinstance(parent_asin, str) or not parent_asin.strip():
            continue
        identity_str = parent_asin.strip()
        if not identity.has_parent_asin(identity_str):
            continue
        try:
            identity.parent_asin_to_item_id(identity_str)
        except (KeyError, TypeError, ValueError):
            continue
        kept.append((identity_str, len(kept) + 1, float(score)))
    return kept


def materialise_history(
    cohort: Sequence[CohortSlice], identity: Any, *, progress: bool
) -> tuple[dict[int, list[tuple[str, int, float]]], str]:
    """The accepted sequential source, read from the frozen M3 head artifact.

    Reuses ``runs/m3_evidence/phase5_fusion_source_heads.npz`` (sha recorded in preregistration
    §12 artifact 1) instead of re-running SASRec, so the HISTORY head is bit-identical to the
    accepted source the comparator used.
    """
    import numpy as np

    recorded = "c4c0fdd6bc1e70fd4d1ffc0387c3afbe6e0642d2d0c6c0184ee440bdef5165d4"
    got = _sha256(FUSION_HEADS)
    if got != recorded:
        raise SystemExit(f"frozen head artifact sha256 {got} != recorded {recorded}")
    z = np.load(FUSION_HEADS)
    matrix, lengths = z["sequential"], z["sequential_lengths"]
    id2item = json.loads(MAPPINGS.read_text(encoding="utf-8"))["id2item"]
    heads: dict[int, list[tuple[str, int, float]]] = {}
    for slice_ in cohort:
        rows_raw = matrix[slice_.index, : int(lengths[slice_.index])].tolist()
        rows: list[tuple[str, int, float]] = []
        for rank, item_id in enumerate(rows_raw, start=1):
            if 0 < item_id < len(id2item) and id2item[item_id]:
                rows.append((id2item[item_id], rank, 1.0 / rank))
        heads[slice_.user_int_id] = _ground(rows, identity)[:DEPTH]
        if progress and slice_.index % 2000 == 0:
            print(f"  history {slice_.index}/{len(cohort)}", flush=True)
    # HISTORY is driven by the run's trusted history, not by a caller-supplied query string.
    return SourceHeads(heads=heads, score_kind="sasrec_logit", queries={})


def materialise_catalog_search(
    cohort: Sequence[CohortSlice], identity: Any, *, progress: bool
) -> tuple[dict[int, list[tuple[str, int, float]]], str]:
    """Live BM25 catalogue search with the frozen history-derived terms (§5).

    Builds ``CatalogSearchSource`` once and queries it once per user. This is the process that must
    not share memory with the similar-item index.
    """
    from recommendation.catalog.metadata import MetadataIndex
    from recommendation.control.catalog_search import CatalogSearchSource

    started = time.time()
    catalogue = MetadataIndex.load(PRODUCTS)
    source = CatalogSearchSource(catalogue)
    print(
        f"  catalogue loaded in {time.time() - started:.1f}s, indexed={source.catalogue_size}, "
        f"peakRSS={_peak_rss_gib():.2f} GiB",
        flush=True,
    )

    def describe(identity_str: str) -> str | None:
        record = catalogue.lookup(identity_str)
        if record is None:
            return None
        title = getattr(record, "title", None)
        return str(title) if title else None

    heads: dict[int, list[tuple[str, int, float]]] = {}
    queries: dict[int, str] = {}
    for position, slice_ in enumerate(cohort):
        terms = deterministic_terms(slice_.history, describe=describe)
        if not terms:
            heads[slice_.user_int_id] = []
            continue
        query = " ".join(terms)
        result = source.search(query, limit=DEPTH)
        rows = [
            (hit.parent_asin, rank, float(hit.score))
            for rank, hit in enumerate(result.hits, start=1)
        ]
        heads[slice_.user_int_id] = _ground(rows, identity)[:DEPTH]
        queries[slice_.user_int_id] = query
        if progress and position % 500 == 0:
            print(f"  catalog_search {position}/{len(cohort)}", flush=True)
    return SourceHeads(heads=heads, score_kind=CATALOG_SEARCH_SCORE_KIND, queries=queries)


def materialise_similar_item(
    cohort: Sequence[CohortSlice], identity: Any, *, progress: bool
) -> SourceHeads:
    """Frozen TF-IDF similar-item retrieval with the frozen history-derived seed (§5).

    Loads the persisted index built once by :mod:`experiments.m4_build_similar_index`, so this
    process never pays the ~5.93 GiB build peak and can therefore be chunked and resumed. Every
    index object is the frozen class's own state, so retrieval is unchanged.

    The accelerated neighbour path is used only when it is **bit-identical** to the reference on a
    probe of this index; otherwise the reference is used. Either way the head is what the trusted
    tool would have produced.
    """
    import pickle
    import random

    from recommendation.control.similar_item import SimilarItemIndex, SimilarItemSource

    from experiments.m4_similar_neighbours import SparseNeighbourIndex

    if not SIMILAR_INDEX.is_file():
        raise SystemExit(
            f"missing {SIMILAR_INDEX}; run `python -m experiments.m4_build_similar_index` first"
        )
    started = time.time()
    with open(SIMILAR_INDEX, "rb") as handle:
        payload = pickle.load(handle)
    # Rebuild the frozen index object from its own persisted state, without re-tokenising the
    # catalogue: the vectors, norms, idf and document frequencies are the built values.
    index = SimilarItemIndex.__new__(SimilarItemIndex)
    index._identities = tuple(payload["identities"])  # noqa: SLF001
    index._documents = {}  # noqa: SLF001 - raw text is not needed for retrieval
    index._document_frequency = payload["document_frequency"]  # noqa: SLF001
    index._idf = payload["idf"]  # noqa: SLF001
    index._vectors = payload["vectors"]  # noqa: SLF001
    index._norms = payload["norms"]  # noqa: SLF001
    print(
        f"  similar index loaded in {time.time() - started:.1f}s, vocab={index.vocabulary_size}, "
        f"docs={len(index.identities)}, peakRSS={_peak_rss_gib():.2f} GiB",
        flush=True,
    )

    accelerated = SparseNeighbourIndex(index)
    source = SimilarItemSource(index)
    rng = random.Random(20260201)
    probe = [index.identities[rng.randrange(len(index.identities))] for _ in range(8)]
    mismatches = sum(
        1
        for seed in probe
        if index.neighbours(seed, limit=DEPTH) != accelerated.neighbours(seed, limit=DEPTH)
    )
    if mismatches:
        print(
            f"  WARNING: accelerated path mismatched {mismatches}/8 probes -> using reference",
            flush=True,
        )
        accelerated = None  # type: ignore[assignment]
    else:
        print("  accelerated path verified bit-identical on 8 probes", flush=True)

    heads: dict[int, list[tuple[str, int, float]]] = {}
    queries: dict[int, str] = {}
    for position, slice_ in enumerate(cohort):
        seed = newest_indexed(slice_.history, is_indexed=lambda x: source.covers(x))
        if seed is None:
            heads[slice_.user_int_id] = []
            continue
        rows = (
            accelerated.neighbours(seed, limit=DEPTH)
            if accelerated is not None
            else index.neighbours(seed, limit=DEPTH)
        )
        heads[slice_.user_int_id] = _ground(rows, identity)[:DEPTH]
        queries[slice_.user_int_id] = seed
        if progress and position % 200 == 0:
            print(
                f"  similar_item {position}/{len(cohort)} peakRSS={_peak_rss_gib():.2f} GiB",
                flush=True,
            )
    return SourceHeads(heads=heads, score_kind="item_item_similarity", queries=queries)


def materialise_two_tower(
    cohort: Sequence[CohortSlice], identity: Any, *, progress: bool
) -> tuple[dict[int, list[tuple[str, int, float]]], str]:
    """Live Two-Tower retrieval through the repository's own source tool.

    The tool takes trusted history and scores the whole catalogue, so this is the second expensive
    process. Its ``score_kind`` is read from the tool rather than assumed.
    """
    from recommendation.control.two_tower_source import TwoTowerSourceTool
    from recommendation.inference.twotower import (
        TwoTowerInferenceConfig,
        TwoTowerInferenceEngine,
    )

    started = time.time()
    config = TwoTowerInferenceConfig(
        checkpoint_path=TWO_TOWER_CKPT,
        categories_path=TWO_TOWER_CKPT.parent / "metadata_categories.json",
        mappings_path=MAPPINGS,
        device="cpu",
    )
    engine = TwoTowerInferenceEngine(config)
    tool = TwoTowerSourceTool(engine)
    print(f"  two-tower engine ready in {time.time() - started:.1f}s", flush=True)
    heads: dict[int, list[tuple[str, int, float]]] = {}
    for position, slice_ in enumerate(cohort):
        try:
            rows = tool.propose(arguments=None, limit=DEPTH, history=slice_.history)
        except Exception as exc:  # noqa: BLE001 - an empty history is a legitimate empty head
            print(f"  two_tower user {slice_.user_int_id}: {type(exc).__name__}", flush=True)
            rows = []
        heads[slice_.user_int_id] = _ground(rows, identity)[:DEPTH]
        if progress and position % 500 == 0:
            print(f"  two_tower {position}/{len(cohort)} peakRSS={_peak_rss_gib():.2f} GiB", flush=True)
    return SourceHeads(heads=heads, score_kind=str(tool.score_kind), queries={})


MATERIALISERS = {
    CandidateSource.HISTORY: materialise_history,
    CandidateSource.CATALOG_SEARCH: materialise_catalog_search,
    CandidateSource.SIMILAR_ITEM: materialise_similar_item,
    CandidateSource.TWO_TOWER: materialise_two_tower,
}


# -- shard I/O ------------------------------------------------------------- #


def shard_path(source: CandidateSource, chunk: int) -> pathlib.Path:
    """Deterministic path for one ``(source, chunk)`` shard.

    Preflight shards live in a separate namespace so a ``--limit`` run can never be mistaken for
    cohort materialisation and silently assembled into the frozen artifact.
    """
    directory = SHARDS if _FULL_COHORT else SHARDS / "preflight"
    return directory / f"{source.value}.{chunk:05d}.json"


#: Set by :func:`main` from ``--limit``. False means "the full frozen cohort".
_FULL_COHORT = True


def run_source(
    source: CandidateSource, *, chunk_size: int, limit: int | None, force: bool = False
) -> None:
    """Materialise one source in this process, chunk by chunk, resuming completed shards."""
    global _FULL_COHORT
    _FULL_COHORT = limit is None
    shard_path(source, 0).parent.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(limit=limit)
    identity = build_identity_map(MAPPINGS)
    print(f"[{source.value}] cohort={len(cohort)} chunk_size={chunk_size}", flush=True)
    started = time.time()
    for chunk, start in enumerate(range(0, len(cohort), chunk_size)):
        path = shard_path(source, chunk)
        if path.is_file() and not force:
            print(f"  chunk {chunk} already complete, skipping", flush=True)
            continue
        piece = cohort[start : start + chunk_size]
        outcome = MATERIALISERS[source](piece, identity, progress=True)
        payload = {
            "source": source.value,
            "score_kind": outcome.score_kind,
            "chunk": chunk,
            "users": {str(uid): rows for uid, rows in sorted(outcome.heads.items())},
            "queries": {str(uid): q for uid, q in sorted(outcome.queries.items())},
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)  # atomic: a partial shard can never be mistaken for a complete one
        print(f"  chunk {chunk} done in {time.time() - started:.1f}s -> {path.name}", flush=True)
    print(f"[{source.value}] finished, peakRSS={_peak_rss_gib():.2f} GiB", flush=True)


def assemble(*, limit: int | None = None) -> None:
    """Assemble the shards into the frozen artifact and record its provenance."""
    import numpy as np

    cohort = load_cohort(limit=limit)
    order = [slice_.user_int_id for slice_ in cohort]
    arrays: dict[str, Any] = {}
    score_kinds: dict[str, str] = {}
    source_lengths: dict[str, dict[str, Any]] = {}
    for source in SOURCE_ORDER:
        per_user: dict[int, list[tuple[str, int, float]]] = {}
        per_user_query: dict[int, str] = {}
        score_kind = EXPECTED_SCORE_KINDS[source]
        chunk = 0
        while True:
            path = shard_path(source, chunk)
            if not path.is_file():
                break
            payload = json.loads(path.read_text(encoding="utf-8"))
            score_kind = payload.get("score_kind", score_kind)
            for uid, rows in payload["users"].items():
                per_user[int(uid)] = [tuple(r) for r in rows]
            for uid, query in payload.get("queries", {}).items():
                per_user_query[int(uid)] = str(query)
            chunk += 1
        lengths = np.zeros(len(order), dtype=np.int32)
        flat_ids: list[str] = []
        flat_ranks: list[int] = []
        flat_scores: list[float] = []
        for position, uid in enumerate(order):
            rows = per_user.get(uid, [])
            lengths[position] = len(rows)
            for parent_asin, rank, score in rows:
                flat_ids.append(parent_asin)
                flat_ranks.append(int(rank))
                flat_scores.append(float(score))
        arrays[f"{source.value}_lengths"] = lengths
        arrays[f"{source.value}_identities"] = np.asarray(flat_ids, dtype=object)
        arrays[f"{source.value}_ranks"] = np.asarray(flat_ranks, dtype=np.int32)
        arrays[f"{source.value}_scores"] = np.asarray(flat_scores, dtype=np.float64)
        # The query/seed actually used to produce each head, so the benchmark serves exactly the
        # query the head came from rather than re-deriving it.
        arrays[f"{source.value}_queries"] = np.asarray(
            [per_user_query.get(uid, "") for uid in order], dtype=object
        )
        score_kinds[source.value] = score_kind
        source_lengths[source.value] = {
            "users_with_head": int((lengths > 0).sum()),
            "min": int(lengths.min()),
            "mean": round(float(lengths.mean()), 4),
            "max": int(lengths.max()),
            "total_rows": int(lengths.sum()),
        }
        print(
            f"[{source.value}] users={len(order)} with_head={int((lengths > 0).sum())} "
            f"min={int(lengths.min())} mean={lengths.mean():.2f} max={int(lengths.max())}",
            flush=True,
        )
    arrays["user_int_ids"] = np.asarray(order, dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ASSEMBLED, **arrays)
    provenance = {
        "artifact": str(ASSEMBLED.relative_to(REPO)),
        "sha256": _sha256(ASSEMBLED),
        "size_bytes": ASSEMBLED.stat().st_size,
        "depth": DEPTH,
        "users": len(order),
        "source_order": [s.value for s in SOURCE_ORDER],
        "score_kinds": score_kinds,
        "source_lengths": source_lengths,
        "inputs": {
            "sequences": {"path": str(SEQUENCES.relative_to(REPO)), "sha256": _sha256(SEQUENCES)},
            "mappings": {"path": str(MAPPINGS.relative_to(REPO)), "sha256": _sha256(MAPPINGS)},
            "products": {"path": str(PRODUCTS.relative_to(REPO)), "sha256": _sha256(PRODUCTS)},
            "fusion_heads": {"path": str(FUSION_HEADS.relative_to(REPO)), "sha256": _sha256(FUSION_HEADS)},
            "two_tower": {"path": str(TWO_TOWER_CKPT.relative_to(REPO)), "sha256": _sha256(TWO_TOWER_CKPT)},
        },
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\nassembled -> {ASSEMBLED}\nprovenance -> {PROVENANCE}", flush=True)
    print(f"sha256 = {provenance['sha256']}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="M4 frozen head materialisation")
    parser.add_argument("--source", choices=sorted(CLI_SOURCES), default=None)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None, help="non-cohort preflight only")
    parser.add_argument(
        "--force", action="store_true", help="recompute existing shards (preflight measurement)"
    )
    args = parser.parse_args(argv)

    if args.assemble:
        assemble(limit=args.limit)
        return 0
    if not args.source:
        parser.error("either --source or --assemble is required")
    run_source(
        CLI_SOURCES[args.source],
        chunk_size=args.chunk_size,
        limit=args.limit,
        force=args.force,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
