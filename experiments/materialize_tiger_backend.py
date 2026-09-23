"""Materialise the item-id-only TIGER backend handoff (L1 of the four-layer model).

This is the **AgentRec-X-side** layer, and the only place ``parent_asin`` is read.  It uses
canonical identity for exactly three things:

1. joining the frozen catalogue to product metadata so item text can be composed;
2. nothing else - identity is re-attached by AgentRec-X outside the adapter, never handed to
   the backend;
3. provenance and digest bookkeeping.

Everything it writes is keyed by the **opaque integer item id**.  The adapter and the backend
never see a ``parent_asin``, and an AST guard asserts that.

Two modes:

``--mode handoff`` (default, CPU, seconds)
    Load the frozen artifacts, select the cohort, compose text, and write
    ``catalogue.json`` / ``catalogue_items.jsonl`` / ``products_text.jsonl`` /
    ``train_exposure.json`` / ``train_exposure.jsonl`` / ``eval_cohort.json`` /
    ``eval_cohort.jsonl`` / ``score_request.json`` / ``manifest.json``.

    ``eval_cohort.jsonl`` carries ``case_id``, ``history`` and ``required_frontier`` and
    **no target**.  The test targets stay in this process, because the frozen evaluator needs
    them here and the backend must never see them.

``--mode stub-run`` (opt-in)
    Additionally run the Step-2.3 backend placeholder stages by subprocess and score the
    cohort, so the process and score-batch boundary is exercised end to end.  No model, no
    training and no GPU are involved.

Usage::

    .venv/bin/python -m experiments.materialize_tiger_backend --limit 2000 \
        --out /tmp/tiger_handoff
    .venv/bin/python -m experiments.materialize_tiger_backend --limit 64 \
        --out /tmp/tiger_handoff --mode stub-run --status approximate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.backends.tiger_backend import (  # noqa: E402
    DEFAULT_COHORT_SEED,
    DEFAULT_K_VALUES,
    TigerBackendAdapter,
)
from recommendation.evaluation.split import load_cohort_from_artifacts  # noqa: E402

SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
PRODUCTS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_products.jsonl"
DEFAULT_OUT = REPO_ROOT / "runs" / "tiger_backend_handoff"

#: The text template.  Composed here, in AgentRec-X, from the accepted catalogue artifact;
#: trimmed to this many characters per item.
TEXT_TEMPLATE = ("title", "categories", "store", "features", "description")
MAX_TEXT_CHARS = 1000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compose_item_text(record: Any) -> str:
    """Join the frozen catalogue fields of a real metadata record into one string.

    Structural normalisation only: fields are concatenated, never paraphrased, summarised or
    generated, matching the source-fidelity rule the catalogue package already documents.  A
    field the source does not supply contributes nothing, and an item with no text at all still
    receives an (empty) entry, so catalogue coverage stays total.
    """
    parts: list[str] = []
    for name in TEXT_TEMPLATE:
        value = getattr(record, name, None)
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            joined = " ".join(str(entry).strip() for entry in value if str(entry).strip())
            if joined:
                parts.append(joined)
        else:
            text = str(value).strip()
            if text:
                parts.append(text)
    return ". ".join(parts)[:MAX_TEXT_CHARS]


def load_item_ids_and_texts(
    *, item_ids: Sequence[int] | None = None
) -> tuple[list[int], list[str], dict[str, Any]]:
    """Return ``(item_ids, texts, coverage)`` for the frozen catalogue, item-id ordered.

    The join key never leaves this function: canonical identity is read from the mappings
    artifact, used to look up metadata, converted into text, and dropped.  What is returned is
    the integer id and a string.

    ``item_ids`` restricts and *reorders* the catalogue.  The ids stay real AgentRec-X ids -
    they are simply presented in a different ``backend_row`` order - so a smoke run can use a
    small catalogue without inventing a synthetic id space.
    """
    import json as _json

    from recommendation.catalog import MetadataIndex

    payload = _json.loads(Path(MAPPINGS).read_text(encoding="utf-8"))
    id2item = payload["id2item"]
    num_items = int(payload["num_items"])
    if len(id2item) != num_items + 1:
        raise SystemExit(
            f"mappings id2item holds {len(id2item)} entries, expected {num_items + 1}"
        )
    wanted = list(range(1, num_items + 1)) if item_ids is None else [int(i) for i in item_ids]

    index = MetadataIndex.load(PRODUCTS)
    ids: list[int] = []
    texts: list[str] = []
    missing = 0
    empty = 0
    for item_id in wanted:
        if not 1 <= item_id <= num_items:
            raise SystemExit(f"item id {item_id} is outside 1..{num_items}")
        identifier = id2item[item_id]
        if identifier is None:
            raise SystemExit(f"mappings id2item[{item_id}] is None; real ids must map")
        record = index.lookup(identifier)
        # ``lookup`` returns ProductMetadata or the explicit MissingMetadata marker; the marker
        # carries no text fields, so a missing record yields an empty (never fabricated) string.
        if hasattr(record, "has_searchable_text") is False and type(record).__name__ == "MissingMetadata":
            missing += 1
            text = ""
        else:
            text = compose_item_text(record)
        if not text:
            empty += 1
        ids.append(item_id)
        texts.append(text)
    coverage = {
        "num_items": len(ids),
        "items_missing_metadata": missing,
        "items_empty_text": empty,
        "max_chars": MAX_TEXT_CHARS,
        "template": list(TEXT_TEMPLATE),
    }
    return ids, texts, coverage


def load_all_cases():
    """Load every eligible case from the frozen artifacts, once.

    ``train_exposure`` and the evaluation cohort are two different populations drawn from this
    one load, which is why it is separated from the selection step.
    """
    return load_cohort_from_artifacts(SEQUENCES, MAPPINGS)


def select_eval_cohort(cases: Sequence[Any], *, cohort_size: int):
    """Select the frozen deterministic evaluation cohort from ``cases``.

    Thin delegation to the accepted algorithm in ``experiments.benchmark_public`` - same seed,
    same bucket definition, same ordering.  This function exists so a caller can select the
    evaluation cohort *without* also truncating the training population, which is exactly the
    coupling that produced a 2,000-user training exposure for a 412,445-user catalogue.
    """
    import experiments.benchmark_public as bench

    selection = bench.cohort_from_cases(cases, size=cohort_size)
    # ``cohort_from_cases`` returns {"cases", "description"}; the description is what carries
    # the seed, the eligible-user count and the per-bucket provenance, so it is unwrapped here
    # rather than making every caller reach through the envelope.
    return list(selection["cases"]), selection["description"]


@dataclass(frozen=True)
class RemappedCase:
    """A cohort case renumbered onto a smoke-run item space.

    It presents the same interface the handoff builders need (``train_history`` and
    ``test_history``) so the production code path is exercised unchanged.
    """

    user_id: str
    user_int_id: int
    train_history: tuple[int, ...]
    validation_target: int
    test_target: int
    sequence_length: int

    @property
    def test_history(self) -> tuple[int, ...]:
        return self.train_history + (self.validation_target,)


def subsample_items(
    cases: Sequence[Any], *, num_items: int
) -> tuple[list[int], list[RemappedCase]]:
    """Pick the ``num_items`` most frequent items in the cohort and remap the cases onto them.

    Item ids are assigned by **preprocessing frequency**, not by catalogue order, so a prefix of
    the id range is a random subset of the catalogue - which is why a smoke run cannot simply
    truncate it.  This instead chooses the most-used items, renumbers them to a dense
    ``1..num_items`` span, and keeps only the cases that remain usable (a train history of at
    least two items and a test target) after the remap.

    The remapped ids are *internal to the smoke run*: they are real AgentRec-X item ids drawn
    from the frozen catalogue, presented in a different ``backend_row`` order.  Nothing about the
    production path changes, and no synthetic item is invented.
    """
    from collections import Counter

    counts: Counter[int] = Counter()
    for case in cases:
        counts.update(int(item) for item in case.train_history)
        counts.update((int(case.validation_target), int(case.test_target)))
    if not counts:
        raise SystemExit("cannot subsample items: the cohort holds no interactions")
    chosen = sorted(item for item, _ in counts.most_common(num_items))
    if not chosen:
        raise SystemExit("cannot subsample items: no item was observed")
    remap = {item: index + 1 for index, item in enumerate(chosen)}
    if len(remap) != len(chosen):
        raise SystemExit("item subsampling produced duplicate ids")
    catalogue_ids = [remap[item] for item in chosen]

    kept: list[Any] = []
    for case in cases:
        history = tuple(remap[item] for item in case.train_history if item in remap)
        if len(history) < 2:
            continue
        if case.validation_target not in remap or case.test_target not in remap:
            continue
        kept.append(
            RemappedCase(
                user_id=case.user_id,
                user_int_id=case.user_int_id,
                train_history=history,
                validation_target=remap[case.validation_target],
                test_target=remap[case.test_target],
                sequence_length=case.sequence_length,
            )
        )
    return catalogue_ids, kept


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="materialise the TIGER backend handoff")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "SMOKE ONLY. Remap the cohort onto its N most-used item ids and derive both the "
            "training exposure and the evaluation cohort from the remapped, usable cases. "
            "Omit it for production, where training exposure is every eligible user."
        ),
    )
    parser.add_argument(
        "--cohort",
        type=int,
        default=2000,
        help=(
            "size of the deterministic EVALUATION cohort. On the production path (no --limit) "
            "this sizes the evaluation cohort only and does NOT truncate training exposure, "
            "which is always every eligible user. In smoke mode (--limit given) the cohort also "
            "defines the source cases used for coherent item remapping. The canonical "
            "production evaluation cohort is 20000."
        ),
    )
    parser.add_argument("--mode", choices=("handoff", "stub-run"), default="handoff")
    parser.add_argument("--status", choices=("approximate", "certified"), default="approximate")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-items", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend-python", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    def say(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    started = time.time()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loaded once.  Two populations are drawn from it and they are deliberately NOT the same:
    #
    #   training exposure  every eligible user's train_history   -> the model's training corpus
    #   evaluation cohort  a deterministic --cohort subset       -> what the evaluator scores
    #
    # They were coupled, so `--cohort 2000` (the default) silently produced a 2,000-user
    # training corpus for a 412,445-user catalogue.  `--cohort` now sizes ONLY the evaluation
    # cohort; training exposure is all eligible users and does not depend on it.
    all_cases, split_report = load_all_cases()
    if not all_cases:
        raise SystemExit("the frozen artifacts yielded no eligible cases")

    eval_cases, selection = select_eval_cohort(all_cases, cohort_size=args.cohort)
    if not eval_cases:
        raise SystemExit("the evaluation cohort is empty; raise --cohort")

    catalogue_ids: Sequence[int] | None = None
    if args.limit is None:
        # PRODUCTION: full training exposure.  Every eligible case, independent of --cohort.
        training_cases: Sequence[Any] = all_cases
    else:
        # SMOKE: the full histories reference item ids outside the smoke catalogue, so a
        # full-history artifact would be incoherent.  The remap is applied to the evaluation
        # selection and the training exposure comes from the same remapped, usable cases -
        # which is why in smoke mode the cohort also defines the remapping source.
        catalogue_ids, smoke_cases = subsample_items(eval_cases, num_items=args.limit)
        if not smoke_cases:
            raise SystemExit(
                f"--limit {args.limit} leaves no usable case; raise --cohort or --limit"
            )
        eval_cases = smoke_cases
        training_cases = smoke_cases
        if not args.quiet:
            print(
                f"[handoff] --limit {args.limit}: remapped onto {len(catalogue_ids)} most-used "
                f"items; kept {len(smoke_cases)} usable cases for both exposure and evaluation",
                flush=True,
            )
    item_ids, texts, coverage = load_item_ids_and_texts(item_ids=catalogue_ids)
    say(f"[handoff] catalogue: {len(item_ids)} items, {coverage['items_empty_text']} with no text")
    say(
        f"[handoff] eligible users: {split_report.num_users_eligible}; "
        f"training exposure: {len(training_cases)} cases; "
        f"evaluation cohort: {len(eval_cases)} cases"
    )

    adapter = TigerBackendAdapter(
        out_dir, backend_python=args.backend_python, timeout_seconds=1800.0
    )
    catalogue = adapter.materialise_catalogue(
        item_ids=item_ids,
        num_users=split_report.num_users_eligible,
        mappings_sha256=sha256_file(MAPPINGS),
        sequences_sha256=sha256_file(SEQUENCES),
        products_sha256=sha256_file(PRODUCTS),
    )
    adapter.materialise_products_text(texts, catalogue=catalogue)
    exposure = adapter.materialise_train_exposure(
        train_histories=[case.train_history for case in training_cases],
        catalogue=catalogue,
    )
    selected, stats = adapter.materialise_eval_cohort(
        test_histories=[case.test_history for case in eval_cases],
        catalogue=catalogue,
        cohort_seed=DEFAULT_COHORT_SEED,
        k_values=DEFAULT_K_VALUES,
    )
    adapter.write_score_request(
        status=args.status.upper(),
        batch_size=args.batch_size,
        sample_items=args.sample_items,
        seed=args.seed,
    )
    # Additive manifest metadata so the exposure/evaluation distinction is observable by
    # inspection of the handoff itself, without changing any artifact schema.
    manifest = adapter.write_manifest(
        extra={
            "populations": {
                "eligible_users": split_report.num_users_eligible,
                "train_exposure_users": exposure.users,
                "train_exposure_examples": exposure.examples,
                "eval_cohort_cases": selected.cohort_size,
                "catalogue_items": catalogue.num_items,
                "protocol": exposure.protocol,
                "protocol_version": exposure.protocol_version,
                "exposure_field_source": exposure.field_source,
                "cohort_seed": selected.cohort_seed,
                "smoke_remapped": args.limit is not None,
            }
        }
    )

    summary: dict[str, Any] = {
        "out": str(out_dir),
        "mode": args.mode,
        # The four counts that make the exposure/evaluation distinction observable, using the
        # names from the frozen protocol so a production run can be verified by inspection:
        #   eligible_users 412445 / train_exposure_users 412445 / eval_cohort_cases 20000
        #   / catalogue_items 156746
        "eligible_users": split_report.num_users_eligible,
        "train_exposure_users": exposure.users,
        "eval_cohort_cases": selected.cohort_size,
        "catalogue_items": catalogue.num_items,
        "items": catalogue.num_items,
        "cohort_cases": selected.cohort_size,
        "exposure_users": exposure.users,
        "exposure_examples": exposure.examples,
        "smoke_remapped": args.limit is not None,
        "required_frontier": stats.as_dict(),
        "coverage": coverage,
        "manifest_files": sorted(manifest["files"]),
        "seconds": round(time.time() - started, 3),
    }
    say(f"[handoff] required_frontier min={stats.minimum} max={stats.maximum} "
        f"mean={round(stats.mean, 3)} p95={stats.p95} p99={stats.p99}")

    if args.mode == "stub-run":
        # Placeholder stages.  Step 2.4 replaces features and semantic ids; Step 2.6 replaces
        # the scorer.  Nothing here trains, and nothing here touches a GPU.
        sid_dir = out_dir / "sid"
        features_dir = out_dir / "features"
        checkpoint_dir = out_dir / "checkpoint"
        adapter.run_stage(
            "build-features", "--catalogue", str(out_dir), "--out", str(features_dir)
        )
        adapter.run_stage(
            "fit-sid",
            "--catalogue",
            str(out_dir),
            "--features",
            str(features_dir),
            "--out",
            str(sid_dir),
        )
        adapter.run_stage(
            "train",
            "--catalogue",
            str(out_dir),
            "--exposure",
            str(adapter.paths.exposure),
            "--sid",
            str(sid_dir),
            "--out",
            str(checkpoint_dir),
        )
        # The backend reads semantic_ids.json / tiger.json from the directory it is given, so
        # the placeholder artifacts are copied next to the handoff for this smoke run only.
        for name in ("semantic_ids.json", "layout.json", "tiger.json"):
            source = sid_dir / name if (sid_dir / name).is_file() else checkpoint_dir / name
            if source.is_file():
                (out_dir / name).write_bytes(source.read_bytes())
        adapter.write_manifest()
        result = adapter.run_stage(
            "score",
            "--cohort",
            str(out_dir),
            "--sid",
            str(out_dir),
            "--ckpt",
            str(out_dir),
            "--out",
            str(out_dir),
            "--status",
            args.status,
            "--batch-size",
            str(args.batch_size),
            "--sample-items",
            str(args.sample_items),
            "--seed",
            str(args.seed),
        )
        say(f"[stub-run] backend: {result.stdout.decode('utf-8', 'replace').strip()}")
        batches = list(adapter.read_score_shards())
        summary["stub_run"] = {
            "batches": len(batches),
            "cases": sum(len(batch.case_ids) for batch in batches),
            "status": batches[0].status if batches else None,
            "min_reachable_real": min(
                (int(batch.reachable_real_count.min()) for batch in batches), default=0
            ),
            "max_reachable_real": max(
                (int(batch.reachable_real_count.max()) for batch in batches), default=0
            ),
        }

    (out_dir / "materialize_run.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    say(f"[handoff] wrote {out_dir} in {summary['seconds']} s")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
