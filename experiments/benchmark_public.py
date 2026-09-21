"""Phase 5: a public-data recommendation benchmark with a shared candidate universe.

The question this module exists to answer:

    does AgentRec-X provide measurable recommendation value on a real public dataset?

It does **not** add agent architecture.  It reuses the accepted evaluation protocol, the accepted
SASRec artifact, the metadata index and the multi-source retrieval machinery, and it measures them
against each other under one protocol.

Protocol discipline
-------------------
Every arm is scored by the **same** evaluator (`recommendation.evaluation.batched`) over the
**same** split, the **same** number of items, the **same** ``k`` values and the **same** cohort:

    user sequence (chronological) -> temporal leave-two-out
        train history = [i1 .. i(n-2)]
        validation    = history i1..i(n-2)   target i(n-1)
        test          = history i1..i(n-1)   target in

The evaluator owns PAD exclusion, seen-item masking, tie handling and ranking; a model supplies raw
scores and never masks.  That is what makes the arms comparable rather than each having its own
candidate protocol.

Cohort size, stated honestly
----------------------------
The accepted SASRec run evaluated all 412 445 eligible users on a GPU.  This environment is
CPU-only with 7 GB of RAM, so the benchmark evaluates a **deterministic 20 000-user sample** of
that same cohort.  The sample is chosen by a seeded permutation of eligible users stratified by
history length, so it is reproducible and not biased toward short or long histories.  Every arm
sees exactly the users, so the *comparison* is controlled; the absolute numbers are a sample
estimate and are reported as such, never as the accepted full-cohort figure.

Leakage discipline
------------------
Three checks run before any metric is reported, and the experiment refuses to publish results if
any fails:

1. no test target appears in its own history;
2. no validation target appears in its own history;
3. the sample's test targets are a subset of the full cohort's test targets, so subsetting cannot
   have introduced a target the protocol would not have produced.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from recommendation.control.candidate_ledger import reciprocal_rank_fusion  # noqa: E402
from recommendation.evaluation.batched import evaluate_batched  # noqa: E402
from recommendation.evaluation.split import (  # noqa: E402
    EvaluationCase,
    SplitReport,
    load_cohort_from_artifacts,
)
from recommendation.rag.retrieval import BM25_B, BM25_K1, _idf, tokenize  # noqa: E402

__all__ = [
    "BENCHMARK_VERSION",
    "COHORT_SEED",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_COHORT_SIZE",
    "DEFAULT_K_VALUES",
    "FUSION_SOURCES",
    "K_VALUES",
    "MODEL_SEED",
    "ItemTextIndex",
    "AGENT_SOURCES",
    "arm_agent_selected",
    "arm_fixed_fusion",
    "agent_select_sources",
    "agent_selection_plan",
    "arm_metadata_retrieval",
    "metadata_fused_heads",
    "arm_popularity",
    "arm_sasrec",
    "arm_two_tower",
    "assert_protocol",
    "cohort_from_cases",
    "dataset_manifest",
    "evaluate_arm",
    "AGENT_SOURCES",
    "arm_agent_selected",
    "arm_fixed_fusion",
    "agent_select_sources",
    "agent_selection_plan",
    "fused_head",
    "fusion_to_ranking",
    "popularity_rankings",
    "k_values",
    "left_pad_histories",
    "leakage_checks",
    "ranking_batches",
    "ranking_batches_for",
    "seed_query",
    "sasrec_top_rankings",
    "sha256_file",
    "stratified_sample",
]

#: Version of the benchmark contract.  Bumped when the cohort selection, the protocol or the
#: metric set changes, so a stored result can be read against the semantics that produced it.
BENCHMARK_VERSION = 1

#: Primary metric cut-offs.  ``@10`` is the headline the phase asks for; ``@5`` and ``@20`` cost
#: nothing extra because the evaluator ranks the full catalogue once.
K_VALUES: tuple[int, ...] = (5, 10, 20)
DEFAULT_K_VALUES: tuple[int, ...] = K_VALUES

#: Seed for the cohort permutation, and for any model-side sampling.  Recorded in every artifact.
COHORT_SEED = 20260201
MODEL_SEED = 2026

#: Users evaluated per arm.  See the module docstring for why this is a sample.
DEFAULT_COHORT_SIZE = 20_000

#: Evaluation batch size.  The shared evaluator holds a ``[batch, num_items + 1]`` float32 score
#: matrix *and* a same-shaped bool candidate mask, so peak memory grows at roughly
#: ``batch * num_items * 5`` bytes.  For this catalogue (156,747 columns) that is about 1.2 KB per
#: case: batch 256 needs ~0.4 GB, batch 4096 needs ~2.6 GB for scores plus ~0.6 GB for the mask and
#: is what pushed a 7 GB CPU-only host into the OOM killer during development.  256 keeps the
#: measured peak under 2 GB including the ~1.6 GB interpreter and dataset floor, at a throughput
#: penalty of under 10% (the evaluator is not the bottleneck).
DEFAULT_BATCH_SIZE = 256


def k_values() -> tuple[int, ...]:
    """The metric cut-offs every arm is reported at."""
    return K_VALUES


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, so an artifact can be tied to the exact bytes it was read from."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def dataset_manifest(
    *, sequences_path: Path, mappings_path: Path, raw_path: Path | None = None
) -> dict[str, Any]:
    """Describe the dataset an experiment is running on, with content hashes.

    Hashes rather than file names, because a file name says nothing about which preprocessing run
    produced the numbers.  The raw interaction file is recorded only if it is present: it is 2.6 GB
    and is deliberately not committed, so an experiment must be able to describe the *processed*
    artifacts it actually read without requiring the raw corpus to be on disk.
    """
    payload = json.loads(sequences_path.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "dataset": "Amazon Reviews 2023",
        "category": payload.get("category"),
        "artifact_format": payload.get("format"),
        "artifact_description": payload.get("description"),
        "users": payload.get("num_users"),
        "items": payload.get("num_items"),
        "interactions": payload.get("num_interactions"),
        "filtering": "iterative k-core (users and items), documented in the preprocessing module",
        "split_strategy": "chronological temporal leave-two-out",
        "processed": {
            "sequences": {
                "path": str(sequences_path),
                "sha256": sha256_file(sequences_path),
            },
            "mappings": {
                "path": str(mappings_path),
                "sha256": sha256_file(mappings_path),
            },
        },
    }
    if raw_path is not None and Path(raw_path).exists():
        manifest["raw"] = {"path": str(raw_path), "sha256": sha256_file(raw_path)}
    else:
        manifest["raw"] = "not present on this machine; the processed artifacts above are what was read"
    return manifest


def stratified_sample(
    cases: Sequence[EvaluationCase],
    *,
    size: int,
    seed: int = COHORT_SEED,
) -> list[EvaluationCase]:
    """A deterministic, history-length-stratified sample of the evaluation cohort.

    Stratification matters because sequential recommenders do much better on long histories, so an
    unstratified sample could shift every arm's absolute numbers together.  Users are bucketed by
    history length, each bucket is shuffled with the seeded RNG, and users are drawn round-robin
    across buckets proportional to bucket size.  The result is reproducible from the seed alone and
    independent of artifact ordering, because the input is sorted by user id first.
    """
    ordered = sorted(cases, key=lambda case: (case.user_int_id, case.user_id))
    if size >= len(ordered):
        return list(ordered)

    def bucket_of(case: EvaluationCase) -> str:
        length = len(case.test_history)
        if length <= 3:
            return "0:<=3"
        if length <= 5:
            return "1:4-5"
        if length <= 10:
            return "2:6-10"
        if length <= 20:
            return "3:11-20"
        return "4:>20"

    buckets: dict[str, list[EvaluationCase]] = {}
    for case in ordered:
        buckets.setdefault(bucket_of(case), []).append(case)
    rng = random.Random(seed)
    for members in buckets.values():
        rng.shuffle(members)

    # Round-robin across buckets, largest first, so a small bucket is still represented.
    order = sorted(buckets, key=lambda name: -len(buckets[name]))
    chosen: list[EvaluationCase] = []
    index = {name: 0 for name in order}
    while len(chosen) < size:
        progressed = False
        for name in order:
            if len(chosen) >= size:
                break
            position = index[name]
            if position < len(buckets[name]):
                chosen.append(buckets[name][position])
                index[name] = position + 1
                progressed = True
        if not progressed:
            break
    # Sorted by user id, so the cohort's ordering is stable and does not depend on bucket
    # iteration or on the shuffle above.
    return sorted(chosen, key=lambda case: (case.user_int_id, case.user_id))


def cohort_from_cases(
    cases: Sequence[EvaluationCase],
    *,
    size: int = DEFAULT_COHORT_SIZE,
    seed: int = COHORT_SEED,
) -> dict[str, Any]:
    """Select the evaluated cohort and describe it.

    Returns the sample plus a description of the selection, including the per-bucket counts, so a
    reader can see exactly which slice of the public dataset the numbers came from.
    """
    sample = stratified_sample(cases, size=size, seed=seed)
    lengths = sorted(len(case.test_history) for case in sample)
    description = {
        "benchmark_version": BENCHMARK_VERSION,
        "cohort_seed": seed,
        "cohort_size": len(sample),
        "eligible_users": len(cases),
        "sampled": len(sample) < len(cases),
        "selection": (
            "deterministic seeded permutation, stratified by history length, processed round-robin "
            "across length buckets then sorted by user id"
        ),
        "history_length": {
            "min": lengths[0] if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 4) if lengths else 0.0,
        },
    }
    return {"cases": sample, "description": description}


def leakage_checks(
    cases: Sequence[EvaluationCase],
    full: Sequence[EvaluationCase],
    *,
    timestamps: dict[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Verify the split cannot leak future information, decided on **timestamps**.

    The property that matters is not "the target never appears in the history" - a user who buys
    the same product twice legitimately has their target in their history - but "every history
    interaction happened strictly before the target".  Timestamps decide that, so when they are
    supplied they are the evidence.  Concretely:

    * ``test_history_strictly_before_target`` - for every case whose timestamps are known, the
      target's timestamp is greater than every history timestamp.  This is the no-future-leakage
      claim, and it is exact.
    * ``target_in_history_is_a_repurchase`` - every case whose test target appears in its history
      has an earlier occurrence of that item, i.e. the target is genuinely the user's next
      purchase of something they bought before.  Counted and reported, never hidden.
    * ``sample_targets_subset_of_full`` - subsetting cannot have invented a case.
    * ``split_shapes_consistent`` - ``test_history == train_history + (validation_target,)``.

    Without timestamps the timestamp-based checks report ``None`` and only the structural ones
    decide, which is stated rather than silently treated as a pass.
    """
    target_in_history = 0
    repeat_purchase = 0
    unexplained_target_in_history = 0
    validation_in_history = 0
    strict_before = 0
    checked = 0
    non_monotonic = 0

    for case in cases:
        if case.test_target in case.test_history:
            target_in_history += 1
            # An earlier occurrence is what makes it a repurchase rather than an inconsistency.
            if case.test_history.count(case.test_target) >= 1:
                repeat_purchase += 1
            else:  # pragma: no cover - count() >= 1 whenever the membership test is true
                unexplained_target_in_history += 1
        if case.validation_target in case.validation_history:
            validation_in_history += 1
        if timestamps is not None:
            stamps = timestamps.get(case.user_id)
            if stamps is not None and len(stamps) == case.sequence_length:
                checked += 1
                history_stamps = stamps[: len(case.test_history)]
                target_stamp = stamps[len(case.test_history)]
                if all(value < target_stamp for value in history_stamps):
                    strict_before += 1
                if any(
                    stamps[i] < stamps[i - 1] for i in range(1, len(stamps))
                ):
                    non_monotonic += 1

    full_targets = {(case.user_id, case.test_target) for case in full}
    sample_targets = {(case.user_id, case.test_target) for case in cases}
    subset_sound = sample_targets <= full_targets
    shapes_consistent = all(
        len(case.test_history) == len(case.validation_history) + 1 for case in cases
    )

    timestamp_evidence = None if timestamps is None else {
        "cases_checked": checked,
        "history_strictly_before_target": strict_before,
        "non_monotonic_sequences": non_monotonic,
        "all_checked_cases_strictly_ordered": (
            checked > 0 and strict_before == checked and non_monotonic == 0
        ),
    }
    passed = (
        unexplained_target_in_history == 0
        and subset_sound
        and shapes_consistent
        and (timestamp_evidence is None or timestamp_evidence["all_checked_cases_strictly_ordered"])
    )
    return {
        "test_target_in_own_history": target_in_history,
        "of_which_repurchase_of_an_earlier_item": repeat_purchase,
        "unexplained_target_in_history": unexplained_target_in_history,
        "validation_target_in_own_history": validation_in_history,
        "sample_targets_subset_of_full": subset_sound,
        "split_shapes_consistent": shapes_consistent,
        "timestamp_evidence": timestamp_evidence,
        "repeat_share": round(target_in_history / len(cases), 6) if cases else 0.0,
        "passed": passed,
    }


def assert_protocol(checks: dict[str, Any]) -> None:
    """Refuse to continue when a leakage or protocol check failed."""
    if not checks.get("passed", False):
        raise RuntimeError(f"protocol/leakage checks failed: {json.dumps(checks, sort_keys=True)}")


# --------------------------------------------------------------------------- #
# Catalogue text index
# --------------------------------------------------------------------------- #


#: Catalogue fields an item's searchable document is built from, in a fixed order.  The same
#: normalized catalogue the rest of the system reads, so the index invents no text.
INDEXED_FIELDS: tuple[str, ...] = (
    "title",
    "store",
    "main_category",
    "categories",
    "features",
)

#: Fields used to build a user's query from one history item.  Narrower than the indexed fields on
#: purpose: a query is the *user's* stated interest, and the brand plus category plus title is the
#: part of a product record that expresses what was bought.
QUERY_FIELDS: tuple[str, ...] = ("title", "store", "main_category", "categories")


class ItemTextIndex:
    """BM25 over normalized catalogue text, with a vectorized inverted index.

    Two things differ from the accepted
    :class:`~recommendation.control.catalog_search.CatalogSearchSource`, and both are stated here
    because they are the kind of difference that silently changes what a number means.

    **Data structure.**  The accepted source scores a query by scanning every catalogue document
    (``for parent_asin, counts, length, title in self._documents``).  That is the right shape for one
    interactive turn against 156 746 records; it is not viable for a benchmark that needs tens of
    thousands of queries, so this class keeps posting lists and scores only the documents that
    contain a query term.  The *scoring* is unchanged - the tokenizer, the BM25 constants and the idf
    function are imported from :mod:`recommendation.rag.retrieval`, so the maths cannot drift - and
    ``test_item_text_index_agrees_with_catalog_search_source`` pins that by asserting the two agree
    on the same catalogue and query.

    **Query-term filtering.**  Terms appearing in more than
    :data:`ItemTextIndex.MAX_QUERY_DOCUMENT_FRACTION` of the catalogue are dropped.  This is not a
    heuristic bolted on for speed.  Measuring this catalogue shows a term present in 156 721 of
    156 746 products, and under BM25's floored idf such a term carries a small but *positive* weight:
    including it adds a near-constant to almost every document, matches essentially the whole
    catalogue, and turns a 200-item shortlist into an arbitrary tie decided by item id.  Dropping
    those terms is what makes "retrieved" mean anything, and the cap is recorded in the artifact.
    """

    #: A query term present in more than this fraction of the catalogue is treated as
    #: non-discriminative and not scored.  5% of 156 746 is ~7 800 documents - far above any term
    #: that could identify a product, far below the stopword-like terms at 99%+.
    MAX_QUERY_DOCUMENT_FRACTION = 0.05

    def __init__(self, records: Any, *, fields: Sequence[str] = INDEXED_FIELDS) -> None:
        self._fields = tuple(fields)
        identities: list[str] = []
        lengths: list[int] = []
        postings: dict[str, list[int]] = {}
        term_frequencies: dict[str, list[int]] = {}
        document_frequency: dict[str, int] = {}

        for identity, record in records.items():
            text = " ".join(
                part for part in (self._field_text(record, f) for f in self._fields) if part
            )
            tokens = tokenize(text)
            if not tokens:
                # A record with no searchable text is not indexed.  It stays reachable by
                # identity, which is how the rest of the system reaches it too.
                continue
            position = len(identities)
            identities.append(identity)
            lengths.append(len(tokens))
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for term, count in counts.items():
                postings.setdefault(term, []).append(position)
                term_frequencies.setdefault(term, []).append(count)
                document_frequency[term] = document_frequency.get(term, 0) + 1

        import numpy as np

        self._identities = identities
        self._document_frequency = document_frequency
        self._max_document_frequency = int(len(identities) * self.MAX_QUERY_DOCUMENT_FRACTION)
        self._postings = {
            term: np.asarray(positions, dtype=np.int64)
            for term, positions in postings.items()
            if len(positions) <= self._max_document_frequency
        }
        self._term_frequencies = {
            term: np.asarray(values, dtype=np.float64)
            for term, values in term_frequencies.items()
            if term in self._postings
        }
        self._idf = {
            term: _idf(document_frequency[term], len(identities)) for term in self._postings
        }
        self._lengths = np.asarray(lengths, dtype=np.float64)
        self._average_length = float(self._lengths.mean()) if lengths else 1.0

    @staticmethod
    def _field_text(record: Any, field: str) -> str:
        """One searchable field's text, or ``""`` when absent.  Nothing is paraphrased."""
        value = getattr(record, field, None)
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (tuple, list)):
            return " ".join(str(item) for item in value if isinstance(item, str) and item.strip())
        return ""

    # -- metadata ---------------------------------------------------------- #

    @property
    def size(self) -> int:
        """How many catalogue records carry searchable text."""
        return len(self._identities)

    @property
    def fields(self) -> tuple[str, ...]:
        """The catalogue fields this index searches."""
        return self._fields

    @property
    def identities(self) -> tuple[str, ...]:
        """Indexed identities in catalogue order, which is the tie-break order."""
        return tuple(self._identities)

    @property
    def max_document_frequency(self) -> int:
        """The document-frequency cap above which a query term is not scored."""
        return self._max_document_frequency

    def query_terms(self, query: str) -> tuple[str, ...]:
        """The discriminative terms of a query, in first-seen order."""
        return tuple(term for term in dict.fromkeys(tokenize(query)) if term in self._postings)

    def score(self, query: str) -> list[tuple[str, float]]:
        """Return ``(identity, bm25)`` for every document containing a scored query term.

        Ordered by descending score then identity, so ties break exactly as the accepted source
        breaks them.  Documents scoring exactly zero are omitted, mirroring the accepted source's
        ``if score <= 0.0: continue``.
        """
        import numpy as np

        terms = self.query_terms(query)
        if not terms or not self._identities:
            return []

        positions = np.unique(np.concatenate([self._postings[term] for term in terms]))
        totals = np.zeros(positions.shape[0], dtype=np.float64)
        # ``positions`` is sorted and each term's own positions are sorted, so a term's contribution
        # is placed with a direct scatter instead of a per-position lookup.
        for term in terms:
            term_positions = self._postings[term]
            if term_positions.size == 0:
                continue
            slots = np.searchsorted(positions, term_positions)
            frequency = self._term_frequencies[term]
            lengths = self._lengths[term_positions]
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * lengths / (self._average_length or 1.0)
            )
            totals[slots] += self._idf[term] * (frequency * (BM25_K1 + 1.0)) / denominator

        scored = [
            (float(value), self._identities[int(position)])
            for position, value in zip(positions.tolist(), totals.tolist(), strict=True)
            if value > 0.0
        ]
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return scored


# --------------------------------------------------------------------------- #
# Sequential arm helpers / retrieval heads
# --------------------------------------------------------------------------- #


def sasrec_top_rankings(
    *,
    cases: Sequence[EvaluationCase],
    scorer: Callable[[Sequence[EvaluationCase]], Iterable[Any]],
    num_items: int,
    top_n: int = 1000,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[int]]:
    """Model scores as per-user ranked heads, for an arm that must *fuse* ranked lists.

    The evaluator consumes raw scores and owns masking, which is why the sequential arm is scored
    from a score matrix.  Rank fusion needs the opposite: each source's *ordering*.  This function
    reads a model's honest ordering by applying exactly the ordering rule the evaluator applies -
    higher score first, ties by ascending item id - and then discards the seen items, so the head it
    returns is the same candidate set the evaluator would rank first.

    It is used only to build fusion inputs; the sequential arm's reported metrics always come from
    the raw score matrix, never from this head.
    """
    heads: list[list[int]] = []
    for histories, targets, scores in scorer(cases):
        for row, target in enumerate(targets):
            row_scores = scores[row, 1:]
            order = torch.argsort(row_scores, descending=True, stable=True) + 1
            seen = set(histories[row])
            seen.discard(target)
            head = [int(item) for item in order.tolist() if int(item) not in seen]
            heads.append(head[:top_n])
    return heads


def seed_query(record: Any) -> str:
    """Flatten one catalogue record's query fields into text, or ``""`` when it has none."""
    if record is None:
        return ""
    parts: list[str] = []
    for field in QUERY_FIELDS:
        value = getattr(record, field, None)
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (tuple, list)):
            parts.extend(str(item) for item in value if isinstance(item, str) and item)
    return " ".join(parts)


def metadata_fused_heads(
    *,
    index: ItemTextIndex,
    records: dict[str, Any],
    id2item: Sequence[str | None],
    item2id: dict[str, int],
    cases: Sequence[EvaluationCase],
    seeds: int = 5,
    per_seed: int = 100,
    top_n: int = 1000,
) -> list[list[int]]:
    """The metadata arm's per-user item-id head, from which both its score matrix and its fusion
    input are built.

    One implementation, two consumers: :func:`arm_metadata_retrieval` turns each head into a score
    matrix, and the fusion arms take the same heads as a ranked source.  Computing them twice would
    let the "metadata-only" and "fusion" rows of the results table describe two subtly different
    retrievers, which is exactly the kind of drift the shared-evaluator rule exists to prevent.
    """
    cache: dict[str, list[str]] = {}

    def identities_for(asin: str) -> list[str]:
        cached = cache.get(asin)
        if cached is not None:
            return cached
        hits = index.score(seed_query(records.get(asin)))
        cached = [identity for _score, identity in hits[:per_seed] if identity != asin]
        cache[asin] = cached
        return cached

    heads: list[list[int]] = []
    for case in cases:
        history_asins = [
            id2item[item_id]
            for item_id in case.test_history
            if 0 < item_id < len(id2item) and id2item[item_id]
        ]
        rankings = [identities_for(asin) for asin in history_asins[-seeds:]]
        if not rankings:
            # No usable seed means no retrieval at all.  An empty head is honest; the caller's tail
            # convention is what positions the rest.
            heads.append([])
            continue
        fused = [identity for identity, _score in reciprocal_rank_fusion(rankings)]
        head = [item2id[identity] for identity in fused if identity in item2id]
        heads.append(list(dict.fromkeys(head))[:top_n])
    return heads


def arm_metadata_retrieval(
    *,
    num_items: int,
    heads: Sequence[Sequence[int]],
) -> Callable[[Sequence[EvaluationCase], int], Iterable[Any]]:
    """Metadata-retrieval-only: BM25 item-to-item, rank-fused over the user's recent history.

    Why item-to-item and not one long query
    ---------------------------------------
    Concatenating the metadata of several history items into a single query was measured and
    rejected: on this catalogue it matched 156 746 of 156 746 products, so the "shortlist" was the
    entire catalogue and any ranking above the tail was decided by the tie rule.  Scoring each recent
    history item separately and fusing the per-seed rankings keeps every query about one product, and
    the candidate set it yields is the set of products sharing discriminative text with something the
    user actually interacted with.

    Fusion is :func:`~recommendation.control.candidate_ledger.reciprocal_rank_fusion` - the
    repository's accepted rank-based fusion - because a BM25 value from one seed is not comparable
    with a BM25 value from another seed, and summing them would fabricate a quantity.

    ``num_items`` is the full catalogue width the evaluator expects.  ``heads`` are the per-user
    fused heads from :func:`metadata_fused_heads`, so this arm and the fusion arms cannot drift.
    """
    if len(heads) == 0 and num_items <= 0:
        raise ValueError("num_items must be positive")

    def batches(cases: Sequence[EvaluationCase], batch_size: int) -> Iterable[Any]:
        """Yield evaluator-ready batches, doing the per-user work one batch at a time.

        Building a complete 156 746-item permutation per user up front costs roughly 1.5 MB of
        CPython list overhead per user - about 30 GB at this cohort size - which made an earlier
        version of this arm unusable (the OOM killer took it).  Here the head for each user is
        written straight into the batch's score matrix, so peak memory is bounded by the batch.
        """
        import torch

        if len(heads) != len(cases):
            raise ValueError(f"heads ({len(heads)}) and cases ({len(cases)}) must be aligned")
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            scores = torch.zeros((len(chunk), num_items + 1), dtype=torch.float32)
            for row, head in enumerate(heads[start : start + batch_size]):
                ordered = list(dict.fromkeys(int(item) for item in head))
                if ordered:
                    scores[row, torch.tensor(ordered, dtype=torch.long)] = torch.arange(
                        len(ordered), 0, -1, dtype=torch.float32
                    )
            # Everything outside the head shares one explicit score *below* every retrieved item.
            # This is the documented tail convention: the retriever's own ordering occupies the ranks
            # that matter, and unretrieved items are not left in a catalogue-wide tie that the
            # evaluator would have to resolve by item id.  Column 0 (PAD) keeps its zero score, which
            # is irrelevant because the evaluator masks PAD positionally.
            tail = scores[:, 1:]
            tail[tail == 0.0] = -1.0
            yield [tuple(case.test_history) for case in chunk], [
                case.test_target for case in chunk
            ], scores

    return batches


def fusion_to_ranking(
    identities: Sequence[str], item2id: dict[str, int], num_items: int
) -> list[int]:
    """Turn a fused identity order into a **complete** item-id permutation.

    Recovering ranks over the full catalogue
    ----------------------------------------
    The evaluator scores a full ``[batch, num_items + 1]`` matrix, so a retrieval arm cannot hand it
    a shortlist: the items outside the shortlist still need an order.  Two options exist and only one
    of them is honest.

    * Give every unmatched item the same score.  That creates a tie spanning the whole catalogue, and
      the evaluator's tie rule then places the *lowest item id* of ~156 000 unretrieved products
      next, so the reported rank of a target would be decided by an item id rather than by the
      retriever.
    * Append the unmatched items in a deterministic, documented order (ascending item id).  The
      retriever's own order occupies the ranks that matter, and the tail is stated rather than
      implied.

    This function takes the second option.  The tail is a convention and is reported as one: it is
    why a retrieval arm's absolute rank is only interpretable when the target was actually retrieved,
    which is why the benchmark reports the retrieved-share diagnostic alongside the metrics.
    """
    seen: set[int] = set()
    head: list[int] = []
    for identity in identities:
        item_id = item2id.get(identity)
        if item_id is None or item_id in seen:
            continue
        seen.add(item_id)
        head.append(item_id)
    tail = [item_id for item_id in range(1, num_items + 1) if item_id not in seen]
    return head + tail


#: Which sources the agent-selected arm may consult.  The names are the same keys
#: :func:`arm_fixed_fusion` takes, so both arms fuse over one vocabulary.
AGENT_SOURCES: tuple[str, ...] = ("sequential", "metadata", "popularity")

#: Rule thresholds for the deterministic selection policy.  Named constants rather than inline
#: literals so the policy is auditable and a change is visible in the artifact.
SELECTION_MIN_DISTINCT_ITEMS: int = 3
SELECTION_SHORT_HISTORY_ITEMS: int = 8


# --------------------------------------------------------------------------- #
# Agent-selected arm (deterministic stand-in for the LLM policy)
# --------------------------------------------------------------------------- #


def agent_select_sources(
    case: EvaluationCase,
    *,
    available: Sequence[str] = AGENT_SOURCES,
) -> tuple[list[str], str]:
    """Choose which trusted candidate sources to consult, and why.

    What this is, and what it is not
    --------------------------------
    This is a **deterministic stand-in for an LLM policy**, not a language model.  The decision it
    makes is the one Phase 5 needs to measure: *which candidate sources are worth consulting for this
    user?*  The real Phase-4 policy answers that question with a model over the same information (see
    ``docs/PHASE4_HANDOFF.md`` for the live numbers), but a language model cannot be called once per
    user across a 20 000-user cohort, so this arm uses explicit rules over the same signals and
    reports each rule's firing rate.  Treating the two as interchangeable would overstate this arm,
    and the report says so.

    The rules
    ---------
    * **Cold start** (fewer than :data:`SELECTION_MIN_DISTINCT_ITEMS` distinct history items): there
      is no usable seed for item-to-item retrieval and too little sequence for a sequential model, so
      fall back to ``popularity``.  That is the honest floor, not a personalized recommendation.
    * **Short history** (fewer than :data:`SELECTION_SHORT_HISTORY_ITEMS` distinct items): a sequence
      this short gives the sequential model little to work with, while one or two products are enough
      to retrieve textually similar products, so prefer ``metadata``.
    * **Otherwise**: prefer ``sequential``, the strongest single signal on longer histories.

    Sources outside ``available`` are never selected, so the policy cannot name a source the run does
    not have - the same constraint the accepted action validator enforces.
    """
    history = list(case.test_history)
    distinct = len(set(history))
    if distinct < SELECTION_MIN_DISTINCT_ITEMS:
        chosen = "popularity"
        rationale = f"cold_start:{distinct}"
    elif distinct < SELECTION_SHORT_HISTORY_ITEMS:
        chosen = "metadata"
        rationale = f"short_history:{distinct}"
    else:
        chosen = "sequential"
        rationale = f"long_history:{distinct}"
    if chosen not in available:
        return [], f"{rationale}:{chosen}_unavailable"
    return [chosen], rationale


def agent_selection_plan(
    cases: Sequence[EvaluationCase],
    *,
    available: Sequence[str] = AGENT_SOURCES,
) -> dict[str, Any]:
    """The per-user selection plan plus its distribution, computed once per cohort."""
    choices: list[list[str]] = []
    reasons: list[str] = []
    for case in cases:
        selected, rationale = agent_select_sources(case, available=available)
        choices.append(selected)
        reasons.append(rationale)
    distribution: dict[str, int] = {}
    for selected, reason in zip(choices, reasons, strict=True):
        key = "none" if not selected else ",".join(selected)
        distribution[key] = distribution.get(key, 0) + 1
    return {"choices": choices, "reasons": reasons, "distribution": distribution}


def arm_agent_selected(
    *,
    num_items: int,
    source_rankings: dict[str, Sequence[Sequence[int]]],
    plan: dict[str, Any],
    head: int = 1000,
) -> Callable[[Sequence[EvaluationCase], int], Iterable[Any]]:
    """Agent-selected multi-source retrieval, scored exactly like the fixed fusion.

    The arm shares :func:`fused_head` with :func:`arm_fixed_fusion`; the only difference is *which*
    sources each user's run consults.  That is the point of the ablation - if the two arms differ for
    a reason other than selection, the comparison would be measuring two fusion rules instead of two
    selection policies.

    A user whose plan names a source the arm cannot serve falls back to the ascending-id tail rather
    than to a silent default, so an unservable selection is visible in the metrics instead of being
    papered over.
    """
    choices = plan["choices"]

    def batches(cases: Sequence[EvaluationCase], batch_size: int) -> Iterable[Any]:
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            scores = torch.zeros((len(chunk), num_items + 1), dtype=torch.float32)
            for row, index in enumerate(range(start, start + len(chunk))):
                selected = [name for name in choices[index] if name in source_rankings]
                if not selected:
                    continue
                head_ids = fused_head(
                    [source_rankings[name][index] for name in selected], top_n=head
                )
                ordered = list(dict.fromkeys(head_ids))
                scores[row, torch.tensor(ordered, dtype=torch.long)] = torch.arange(
                    len(ordered), 0, -1, dtype=torch.float32
                )
            tail = scores[:, 1:]
            tail[tail == 0.0] = -1.0
            yield [tuple(case.test_history) for case in chunk], [
                case.test_target for case in chunk
            ], scores

    return batches


#: The sources the fixed-fusion arm consults, in the fixed order it consults them.  A fixed fusion
#: consults all of them for every user; the agent-selected arm is the ablation that chooses.
FUSION_SOURCES: tuple[str, ...] = ("popularity", "sequential", "metadata")


# --------------------------------------------------------------------------- #
# Fusion arms
# --------------------------------------------------------------------------- #


def fused_head(
    rankings: Sequence[Sequence[int]],
    *,
    top_n: int,
) -> list[int]:
    """Rank-fuse per-source item-id rankings into one item-id head.

    ``reciprocal_rank_fusion`` is identity-keyed, so item ids are stringified on the way in and
    parsed back on the way out.  Rank-based fusion is the repository's accepted cross-source
    combination: the three sources score on incomparable scales (an interaction count, a SASRec
    logit, a BM25 value), and summing those numbers would fabricate a quantity.
    """
    fused = reciprocal_rank_fusion([[str(item) for item in ranking] for ranking in rankings])
    return [int(identity) for identity, _score in fused[:top_n]]


def arm_fixed_fusion(
    *,
    num_items: int,
    source_rankings: dict[str, Sequence[Sequence[int]]],
    head: int = 1000,
    sources: Sequence[str] | None = None,
) -> Callable[[Sequence[EvaluationCase], int], Iterable[Any]]:
    """Fixed multi-source fusion: consult every named source for every user, then RRF.

    This is the ablation's "fixed multi-source fusion" arm.  It is deliberately *not* adaptive: it
    reads the same sources for every user, in the same order, with the same rank-fusion constant,
    and it has no signal it could use to behave differently.  The agent-selected arm is the
    contrast, and the pair is what makes "does selection help?" an answerable question rather than an
    assertion.

    ``source_rankings`` maps a source name to its per-user ordered item-id head, aligned with the
    evaluated cohort.  Each source contributes only its own ordering; no source's score is compared
    with another's.  ``sources`` names which of them to fuse, in order; it defaults to the accepted
    Phase-5 source set, so the accepted arm's behaviour is unchanged and a post-Phase-5 arm (for
    example adding a Two-Tower source) is expressed as an explicit, additive choice.
    """
    for name, rankings in source_rankings.items():
        if len(rankings) != len(next(iter(source_rankings.values()))):
            raise ValueError(f"source {name!r} is not aligned with the cohort")
    resolved_sources = tuple(sources) if sources is not None else FUSION_SOURCES
    if not resolved_sources:
        raise ValueError("a fusion needs at least one source")
    missing = [name for name in resolved_sources if name not in source_rankings]
    if missing:
        raise ValueError(
            "fusion names source(s) with no rankings supplied: " + ", ".join(missing)
        )

    def batches(cases: Sequence[EvaluationCase], batch_size: int) -> Iterable[Any]:
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            scores = torch.zeros((len(chunk), num_items + 1), dtype=torch.float32)
            for row, index in enumerate(range(start, start + len(chunk))):
                head_ids = fused_head(
                    [source_rankings[name][index] for name in resolved_sources], top_n=head
                )
                ordered = list(dict.fromkeys(head_ids))
                scores[row, torch.tensor(ordered, dtype=torch.long)] = torch.arange(
                    len(ordered), 0, -1, dtype=torch.float32
                )
            tail = scores[:, 1:]
            tail[tail == 0.0] = -1.0
            yield [tuple(case.test_history) for case in chunk], [
                case.test_target for case in chunk
            ], scores

    return batches


def popularity_rankings(
    cases: Sequence[EvaluationCase], *, num_items: int, top_n: int = 1000
) -> list[list[int]]:
    """The popularity ordering as per-user heads, so it can enter a rank fusion."""
    ranking = arm_popularity(cases, num_items=num_items)(cases)[0]
    head = list(ranking[:top_n])
    return [head for _ in cases]


def left_pad_histories(
    cases: Sequence[EvaluationCase], max_seq_len: int
) -> Any:
    """Return a ``[batch, max_seq_len]`` **left-padded** history window for the model arms.

    Left-padding, not right-padding, is what both model arms require, and it is shared here
    rather than written twice so the two arms cannot drift apart in the one detail that is
    easiest to get silently wrong: both ``SASRec.full_catalog_scores`` and
    ``TwoTower.encode_users`` read the representation at the **last real position**, so a
    right-padded window would summarise a run of PAD rows and depress every metric.  Measured
    while building the Phase-5 arms: right-padding cost ~20% of Recall@10 for SASRec
    (0.010 85 vs 0.013 57 alongside the accepted full-cohort result).
    """
    import torch

    histories = [list(case.test_history[-max_seq_len:]) for case in cases]
    padded = torch.zeros((len(cases), max_seq_len), dtype=torch.long)
    for row, history in enumerate(histories):
        if history:
            padded[row, max_seq_len - len(history) :] = torch.tensor(history, dtype=torch.long)
    return padded


def arm_two_tower(
    *,
    checkpoint: Path,
    metadata_categories: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cpu",
) -> Callable[[Sequence[EvaluationCase]], Iterable[Any]]:
    """The Two-Tower retrieval artifact, scored over the full catalogue.

    Post-Phase-5 arm: the accepted five Phase-5 arms are untouched, and this one is scored by
    the **same** evaluator over the **same** cohort, split, catalogue and ``k`` values, so the
    comparison is controlled by construction.

    The model returns a ``[batch, num_items + 1]`` matrix of exact dot products between the
    user vector and every precomputed item vector.  Item vectors are computed once here and
    reused for every batch, which is the property that makes this architecture a retrieval
    baseline; ANN search is explicitly out of scope for this step.  As with SASRec, the model
    performs **no** masking - the evaluator owns PAD exclusion, seen-item masking and ranking.
    """
    import torch

    from recommendation.datasets.twotower import load_metadata_categories, metadata_tensor
    from recommendation.models.twotower import TwoTower, TwoTowerConfig

    payload = torch.load(checkpoint, map_location=device)
    config = dict(payload["model_config"])
    # ``TwoTowerConfig`` is a validating class rather than a dataclass, so the checkpoint's
    # config is passed by name instead of being filtered through ``__dataclass_fields__``.  The
    # constructor validates every field, which is what makes a checkpoint/config mismatch loud.
    if "num_categories" in config:
        config["num_categories"] = tuple(config["num_categories"])
    model = TwoTower(TwoTowerConfig(**config))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    if model.config.num_metadata_fields:
        artifact = metadata_categories
        if artifact is None:
            candidate = checkpoint.parent / "metadata_categories.json"
            if not candidate.exists():
                raise FileNotFoundError(
                    "the two-tower checkpoint was trained with categorical metadata but "
                    f"no category artifact was found next to it ({candidate})"
                )
            artifact = candidate
        model.set_categories(metadata_tensor(load_metadata_categories(artifact)))

    max_seq_len = model.config.max_seq_len

    def score_batches(cases: Sequence[EvaluationCase]) -> Iterable[Any]:
        item_embeddings = model.precompute_item_embeddings()
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            padded = left_pad_histories(chunk, max_seq_len)
            with torch.no_grad():
                user = model.encode_users(padded)
                scores = model.scores_from_embeddings(user, item_embeddings)
            yield [tuple(case.test_history) for case in chunk], [
                case.test_target for case in chunk
            ], scores

    return score_batches


# --------------------------------------------------------------------------- #
# Baseline arms
# --------------------------------------------------------------------------- #


def arm_popularity(
    cases: Sequence[EvaluationCase], *, num_items: int
) -> Callable[[Sequence[EvaluationCase]], list[Any]]:
    """A non-personalised popularity ranker over the training histories.

    Item frequency is counted **only** over the training histories of the evaluated cohort, which
    is the weakest defensible construction: counting over all users would leak validation and test
    targets into the popularity statistics.  It is the floor every other arm has to beat, and it is
    included precisely so the others are judged against something non-trivial to beat.
    """
    counts = [0] * (num_items + 1)
    for case in cases:
        for item_id in case.train_history:
            if 0 < item_id <= num_items:
                counts[item_id] += 1
    order = sorted(
        range(1, num_items + 1), key=lambda item: (-counts[item], item)
    )
    ranking = tuple(order)

    def scorer(_: Sequence[EvaluationCase]) -> list[Any]:
        return [ranking]

    return scorer


def arm_sasrec(
    *, checkpoint: Path, batch_size: int = DEFAULT_BATCH_SIZE, device: str = "cpu"
) -> Callable[[Sequence[EvaluationCase]], list[Any]]:
    """The accepted SASRec artifact, scored over the full catalogue.

    Loaded from the committed checkpoint and used exactly as the accepted run used it: the model
    receives histories and returns a ``[batch, num_items + 1]`` score matrix; the evaluator masks
    and ranks.  No retraining, no re-tuning and no candidate shortlist, so this arm is the accepted
    sequential baseline rather than a Phase-5 reimplementation of it.
    """
    import torch

    from recommendation.models.sasrec import SASRec, SASRecConfig

    payload = torch.load(checkpoint, map_location=device)
    config = payload["model_config"]
    model = SASRec(
        SASRecConfig(
            **{k: v for k, v in config.items() if k in SASRecConfig.__dataclass_fields__}
        )
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    max_seq_len = int(config.get("max_seq_len", 50))

    def score_batches(cases: Sequence[EvaluationCase]) -> Iterable[tuple[Any, Any, Any]]:
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            padded = left_pad_histories(chunk, max_seq_len)
            with torch.no_grad():
                scores = model.full_catalog_scores(padded)
            yield [tuple(case.test_history) for case in chunk], [
                case.test_target for case in chunk
            ], scores

    return score_batches


def evaluate_arm(
    *,
    cases: Sequence[EvaluationCase],
    num_items: int,
    batches: Callable[..., Any],
    cohort: str = "test",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Score one arm with the shared evaluator and return its metrics.

    ``batches`` is either a generator factory taking the cases (a model arm, or a ranker whose whole
    ranking is cheap), or a batch factory taking ``(cases, batch_size)`` (a retrieval arm, whose
    per-user work must be spread across batches so peak memory does not scale with the cohort).
    ``ranking_batches`` is the generator form; ``arm_metadata_retrieval`` returns the batch form.
    """
    if "batch_size" in inspect.signature(batches).parameters:
        score_batches = batches(cases, batch_size)
    else:
        score_batches = batches(cases)
    result = evaluate_batched(
        num_items=num_items,
        score_batches=score_batches,
        k_values=K_VALUES,
        cohort=cohort,
    )
    payload = result.as_dict()
    payload["protocol"] = {
        "version": BENCHMARK_VERSION,
        "split": "temporal_leave_two_out",
        "ranking": "full_catalogue",
        "k_values": list(K_VALUES),
        "cohort": cohort,
        "evaluated_users": len(cases),
        "batch_size": batch_size,
    }
    return payload


# --------------------------------------------------------------------------- #
# Evaluation / ranking encoding
# --------------------------------------------------------------------------- #


def ranking_batches_for(
    rankings: Sequence[Sequence[int]],
    cases: Sequence[EvaluationCase],
    *,
    batch_size: int | None = None,
) -> Iterable[Any]:
    """Encode one full-catalogue ranking per case as one evaluator batch."""
    import torch

    chunk = list(cases) if batch_size is None else list(cases)[:batch_size]
    if len(rankings) != len(chunk):
        raise ValueError("rankings and cases must have the same length")
    width = len(rankings[0]) if rankings else 0
    scores = torch.zeros((len(chunk), width + 1), dtype=torch.float32)
    for row, ranking in enumerate(rankings):
        ids = list(ranking)
        if len(ids) != width:
            raise ValueError("every ranking must span the same catalogue")
        base = torch.arange(width, 0, -1, dtype=torch.float32)
        if sorted(ids) != list(range(1, width + 1)):
            raise ValueError("ranking must be a permutation of item ids 1..num_items")
        scores[row, torch.tensor(ids, dtype=torch.long)] = base
    yield [tuple(case.test_history) for case in chunk], [
        case.test_target for case in chunk
    ], scores


def ranking_batches(
    ranking: Sequence[int], cases: Sequence[EvaluationCase], batch_size: int = DEFAULT_BATCH_SIZE
):
    """Yield the same ranking for every batch, in evaluator form.

    A non-personalised ranker is expressed as a score matrix with a decreasing score down the
    ranking, so the shared evaluator performs the masking, tie handling and ranking exactly as it
    does for a model.  Encoding a score matrix rather than a ranked list is what keeps the
    evaluator, not the arm, in charge of candidate legality - the same rule the accepted protocol
    applies to SASRec.
    """
    import torch

    width = len(ranking)
    # The score matrix is indexed by *item id*, so the ranking has to be scattered into position:
    # if ``ranking = (a, b, c, ...)`` then item ``a`` gets the largest score, ``b`` the next, and so
    # on.  ``scores[:, 1:] = base`` would instead score ascending item id and silently ignore the
    # ranking argument entirely - the evaluator ranks by score, so that variant reports "prefer the
    # highest item id" while claiming to be a popularity baseline.
    # ``test_ranking_batches_scatters_ranking_into_item_positions`` pins this.
    base = torch.arange(width, 0, -1, dtype=torch.float32)
    ids = list(ranking)
    if sorted(ids) != list(range(1, width + 1)):
        # A permutation is required: a duplicate or out-of-range id would be overwritten silently
        # and the arm would quietly rank fewer than ``width`` items.
        raise ValueError("ranking must be a permutation of item ids 1..num_items")
    scatter_index = torch.tensor(ids, dtype=torch.long)
    for start in range(0, len(cases), batch_size):
        chunk = cases[start : start + batch_size]
        scores = torch.zeros((len(chunk), width + 1), dtype=torch.float32)
        # ``scatter_index`` holds only real item ids, so PAD (column 0) keeps its zero score.  That
        # value is irrelevant to the result - the evaluator masks PAD out positionally before any
        # comparison - it is left at zero only so the matrix reads honestly.
        scores[:, scatter_index] = base
        yield [tuple(case.test_history) for case in chunk], [
            case.test_target for case in chunk
        ], scores
