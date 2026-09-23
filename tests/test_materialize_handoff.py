"""Materializer training-exposure / evaluation-cohort decoupling (Step 2.5 preflight).

The bug these tests pin: ``--cohort`` sized the *training* population as well as the evaluation
cohort, so the canonical production handoff (`--cohort 2000`, the default) shipped a
**2,000-user** training corpus for a **412,445-user** catalogue.  Nothing about that is visible
in a metrics table; it is only visible in the handoff populations.

The tests run the real CLI end to end against a small synthetic handoff, because the property
under test is the *materialized artifact*, not an intermediate value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.evaluation.split import EvaluationCase  # noqa: E402

import experiments.materialize_tiger_backend as materialize  # noqa: E402

#: The evaluation cohort used by the synthetic fixtures.
SYNTHETIC_COHORT = 12

#: Disjoint in-catalogue item ranges.  Real targets must be catalogue items, so they cannot be
#: sentinel ids; separating the ranges is what makes a leak identifiable *by item id* while
#: keeping every id valid for the catalogue the backend is handed.
TRAIN_BASE = 1
VALIDATION_BASE = 1000
TEST_BASE = 2000
CATALOGUE_ITEMS = 3000


def make_cases(count: int) -> list[EvaluationCase]:
    """Synthetic eligible cases with disjoint history / validation / test item ranges.

    ``train_history`` draws from one range; ``validation_target`` and ``test_target`` draw from
    two others.  Note that ``test_history`` is defined by the frozen protocol as
    ``train_history + (validation_target,)``, so the eval cohort legitimately contains the
    validation target - and must not contain the test target.
    """
    cases: list[EvaluationCase] = []
    for index in range(count):
        history = tuple(TRAIN_BASE + index * 4 + offset for offset in range(4))
        cases.append(
            EvaluationCase(
                user_id=f"user-{index:04d}",
                user_int_id=index + 1,
                train_history=history,
                validation_target=VALIDATION_BASE + index,
                test_target=TEST_BASE + index,
                sequence_length=6,
            )
        )
    return cases


@pytest.fixture()
def synthetic_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Point the materializer at a tiny self-contained workspace.

    ``load_all_cases`` is patched because building a real Amazon artifact is neither possible nor
    relevant here; **cohort selection is not patched**, so the accepted deterministic algorithm
    runs unmodified on the synthetic cases.
    """
    cases = make_cases(SYNTHETIC_COHORT)
    num_items = CATALOGUE_ITEMS

    mappings_path = tmp_path / "mappings.json"
    mappings_path.write_text(
        json.dumps(
            {
                "num_users": len(cases),
                "num_items": num_items,
                "item2id": {f"asin-{i}": i for i in range(1, num_items + 1)},
                "id2item": [None] + [f"asin-{i}" for i in range(1, num_items + 1)],
            }
        ),
        encoding="utf-8",
    )
    sequences_path = tmp_path / "sequences.json"
    sequences_path.write_text(json.dumps({"sequences": []}), encoding="utf-8")

    products_path = tmp_path / "products.jsonl"
    with products_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"format": "agentrecx.catalog_products.v1", "source": "test"}) + "\n"
        )
        for index in range(1, num_items + 1):
            handle.write(
                json.dumps(
                    {
                        "parent_asin": f"asin-{index}",
                        "source": "McAuley-Lab/Amazon-Reviews-2023",
                        "title": f"product {index}",
                        "categories": ["Sports & Outdoors"],
                    }
                )
                + "\n"
            )

    monkeypatch.setattr(materialize, "MAPPINGS", mappings_path)
    monkeypatch.setattr(materialize, "SEQUENCES", sequences_path)
    monkeypatch.setattr(materialize, "PRODUCTS", products_path)
    monkeypatch.setattr(materialize, "load_all_cases", lambda: (list(cases), _SplitReport(len(cases), num_items)))
    return {"cases": cases, "num_items": num_items, "tmp_path": tmp_path}


class _SplitReport:
    """The fields the materializer reads from a real ``SplitReport``."""

    def __init__(self, eligible: int, catalog_size: int) -> None:
        self.num_users_eligible = eligible
        self.num_users_total = eligible
        self.catalog_size = catalog_size
        self.protocol = "temporal_leave_two_out"
        self.protocol_version = "agentrecx.eval_protocol.v1"


def run_materializer(out_dir: Path, *extra: str) -> dict:
    exit_code = materialize.main(
        ["--out", str(out_dir), "--quiet", *extra]
    )
    assert exit_code == 0
    return json.loads((out_dir / "materialize_run.json").read_text(encoding="utf-8"))


def read_exposure_rows(out_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (out_dir / "train_exposure.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_cohort_rows(out_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (out_dir / "eval_cohort.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# Production path: the populations are separate
# --------------------------------------------------------------------------- #


def test_production_exposure_is_every_eligible_user_not_the_cohort(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    """M eligible cases, --cohort N, no --limit => exposure M and cohort N, with M > N."""
    total = len(synthetic_workspace["cases"])
    cohort = 5
    assert total > cohort

    out_dir = tmp_path / "handoff"
    summary = run_materializer(out_dir, "--cohort", str(cohort))

    assert summary["eligible_users"] == total
    assert summary["train_exposure_users"] == total, "training exposure was truncated to the cohort"
    assert summary["eval_cohort_cases"] == cohort
    assert summary["smoke_remapped"] is False

    rows = read_exposure_rows(out_dir)
    assert len(rows) == total
    assert len(read_cohort_rows(out_dir)) == cohort


def test_production_changing_cohort_does_not_change_training_exposure(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    """The decoupling itself: `--cohort` moves only the evaluation population."""
    total = len(synthetic_workspace["cases"])
    small = run_materializer(tmp_path / "small", "--cohort", "4")
    large = run_materializer(tmp_path / "large", "--cohort", "10")

    assert small["eval_cohort_cases"] == 4
    assert large["eval_cohort_cases"] == 10
    # Identical training populations, because --cohort does not touch them.
    assert small["train_exposure_users"] == large["train_exposure_users"] == total
    assert len(read_exposure_rows(tmp_path / "small")) == total
    assert len(read_exposure_rows(tmp_path / "large")) == total


def test_training_exposure_contains_only_train_history_items(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    """No validation target and no test target may reach a backend-readable training artifact.

    Asserted **by item id**, not only by count: target ids live in a disjoint high range, so a
    leak is identifiable rather than merely suspected.
    """
    cases = synthetic_workspace["cases"]
    history_items = {item for case in cases for item in case.train_history}
    validation_items = {case.validation_target for case in cases}
    test_items = {case.test_target for case in cases}
    assert history_items.isdisjoint(validation_items | test_items)

    out_dir = tmp_path / "handoff"
    run_materializer(out_dir, "--cohort", "6")

    rows = read_exposure_rows(out_dir)
    seen = [item for row in rows for item in row["items"]]
    assert seen, "the exposure artifact is empty"
    assert set(seen) <= history_items, "a non-history item reached the training exposure"
    assert not (set(seen) & validation_items), "a validation target reached the exposure"
    assert not (set(seen) & test_items), "a TEST target reached the exposure"

    # Every row is exactly one case's train_history, so nothing was appended to a history.
    histories = {case.train_history for case in cases}
    for row in rows:
        assert tuple(row["items"]) in histories

    # And the exposure artifact itself declares its provenance.
    record = json.loads((out_dir / "train_exposure.json").read_text(encoding="utf-8"))
    assert record["field_source"] == "EvaluationCase.train_history"


def test_production_cohort_carries_no_target_field(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    """The backend-readable cohort artifact must not declare a target key."""
    out_dir = tmp_path / "handoff"
    run_materializer(out_dir, "--cohort", "6")
    for row in read_cohort_rows(out_dir):
        assert set(row) == {"case_id", "history", "required_frontier"}
    payload = json.loads((out_dir / "eval_cohort.json").read_text(encoding="utf-8"))
    assert "target" not in json.dumps(payload).lower()


def test_manifest_records_both_populations(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    """The distinction must be observable from the handoff manifest alone."""
    total = len(synthetic_workspace["cases"])
    out_dir = tmp_path / "handoff"
    run_materializer(out_dir, "--cohort", "5")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    populations = manifest["populations"]
    assert populations["eligible_users"] == total
    assert populations["train_exposure_users"] == total
    assert populations["eval_cohort_cases"] == 5
    assert populations["catalogue_items"] == synthetic_workspace["num_items"]
    assert populations["exposure_field_source"] == "EvaluationCase.train_history"
    assert populations["smoke_remapped"] is False


# --------------------------------------------------------------------------- #
# Deterministic evaluation cohort: the accepted algorithm is reused, not replaced
# --------------------------------------------------------------------------- #


def test_eval_cohort_selection_is_the_existing_deterministic_algorithm(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    """The selected cohort must equal ``benchmark_public.cohort_from_cases`` on the same input."""
    import experiments.benchmark_public as bench

    cases = synthetic_workspace["cases"]
    expected, expected_description = materialize.select_eval_cohort(cases, cohort_size=7)
    reference = bench.cohort_from_cases(cases, size=7)["cases"]
    assert [case.user_int_id for case in expected] == [case.user_int_id for case in reference]
    assert expected_description["cohort_seed"] == bench.COHORT_SEED
    assert expected_description["eligible_users"] == len(cases)


def test_eval_cohort_selection_is_stable_across_calls(
    synthetic_workspace: dict,
) -> None:
    """Same seed and same input => same cohort, and independent of input ordering."""
    cases = synthetic_workspace["cases"]
    first, _ = materialize.select_eval_cohort(cases, cohort_size=6)
    second, _ = materialize.select_eval_cohort(list(reversed(cases)), cohort_size=6)
    assert [case.user_int_id for case in first] == [case.user_int_id for case in second]


# --------------------------------------------------------------------------- #
# Smoke path: remapping stays coherent
# --------------------------------------------------------------------------- #


def test_smoke_path_remaps_items_and_keeps_exposure_inside_the_smoke_catalogue(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    """With --limit, no training history may reference an item outside the smoke catalogue."""
    limit = 200
    out_dir = tmp_path / "smoke"
    summary = run_materializer(out_dir, "--cohort", "10", "--limit", str(limit))

    assert summary["smoke_remapped"] is True
    # The smoke catalogue is the most-used items of the SELECTED evaluation cohort, so it is
    # bounded above by the limit but need not reach it.
    assert 0 < summary["catalogue_items"] <= limit
    limit = summary["catalogue_items"]
    # Both populations are drawn from the remapped, usable cases and stay inside the catalogue.
    for row in read_exposure_rows(out_dir):
        assert all(1 <= item <= limit for item in row["items"]), row
    for row in read_cohort_rows(out_dir):
        assert all(1 <= item <= limit for item in row["history"]), row
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["populations"]["smoke_remapped"] is True


def test_smoke_path_stays_internally_coherent(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    """Exposure and cohort describe the same remapped item space."""
    limit = 200
    out_dir = tmp_path / "smoke"
    summary = run_materializer(out_dir, "--cohort", "10", "--limit", str(limit))
    limit = summary["catalogue_items"]

    catalogue_rows = [
        json.loads(line)
        for line in (out_dir / "catalogue_items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    item_ids = {row["item_id"] for row in catalogue_rows}
    assert item_ids == set(range(1, limit + 1))
    assert [row["backend_row"] for row in catalogue_rows] == list(range(limit))

    exposure_items = {item for row in read_exposure_rows(out_dir) for item in row["items"]}
    cohort_items = {item for row in read_cohort_rows(out_dir) for item in row["history"]}
    assert exposure_items <= item_ids
    assert cohort_items <= item_ids


# --------------------------------------------------------------------------- #
# Boundary invariants that must survive the change
# --------------------------------------------------------------------------- #


def test_pad_is_never_a_catalogue_item_or_a_history_item(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    out_dir = tmp_path / "handoff"
    run_materializer(out_dir, "--cohort", "6")
    catalogue = json.loads((out_dir / "catalogue.json").read_text(encoding="utf-8"))
    assert catalogue["pad_id"] == 0
    assert catalogue["first_real_id"] == 1
    for row in read_exposure_rows(out_dir):
        assert 0 not in row["items"]
    for row in read_cohort_rows(out_dir):
        assert 0 not in row["history"]


def test_canonical_identity_never_reaches_a_handoff_artifact(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    """The materializer reads canonical identity and must not write it into the handoff.

    Checked precisely rather than by substring: an Amazon product's own *text* can legitimately
    contain an ASIN-shaped token (a model number in a description), and that is verbatim source
    text rather than leaked identity.  What must never appear is identity used **as identity** -
    a ``parent_asin`` key, or the canonical value carried in its own field.
    """
    out_dir = tmp_path / "handoff"
    run_materializer(out_dir, "--cohort", "6")
    def identity_fields(record: dict) -> list[str]:
        return [key for key in record if "asin" in key.lower()]

    offenders: list[str] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "parent_asin" in text:
            offenders.append(f"{path.name}: carries the parent_asin key")
            continue
        records: list[dict] = []
        stripped = text.lstrip()
        if stripped.startswith("{"):
            try:
                # A whole-file JSON object (catalogue.json, manifest.json, ...).
                records = [json.loads(text)]
            except json.JSONDecodeError:
                # A JSONL file.
                records = [
                    json.loads(line) for line in text.splitlines() if line.strip()
                ]
        for record in records:
            if not isinstance(record, dict):
                continue
            found = identity_fields(record)
            # `populations` and `coverage` are nested objects; check them too.
            for value in record.values():
                if isinstance(value, dict):
                    found += [f"{key} (nested)" for key in identity_fields(value)]
            if found:
                offenders.append(f"{path.name}: identity field(s) {found}")
                break
    assert offenders == [], f"canonical identity reached a handoff artifact: {offenders}"


def test_products_text_carries_exactly_int_id_and_text(
    synthetic_workspace: dict, tmp_path: Path
) -> None:
    """The one artifact that carries product text carries no identity field beside it."""
    out_dir = tmp_path / "handoff"
    run_materializer(out_dir, "--cohort", "6")
    with (out_dir / "products_text.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                assert set(json.loads(line)) == {"item_id", "text"}


def test_manifest_metadata_cannot_overwrite_the_digest_block(tmp_path: Path) -> None:
    """Additive metadata is additive: a caller cannot forge `files` or the format."""
    from recommendation.backends.tiger_backend import (
        ContractViolation,
        TigerBackendAdapter,
    )

    adapter = TigerBackendAdapter(tmp_path, timeout_seconds=30.0)
    adapter.materialise_catalogue(
        item_ids=[1, 2],
        num_users=1,
        mappings_sha256="a" * 64,
        sequences_sha256="b" * 64,
        products_sha256="c" * 64,
    )
    with pytest.raises(ContractViolation):
        adapter.write_manifest(extra={"files": {}})
    with pytest.raises(ContractViolation):
        adapter.write_manifest(extra={"format": "forged"})
    # A legitimate additive key is accepted.
    payload = adapter.write_manifest(extra={"populations": {"eligible_users": 2}})
    assert payload["populations"] == {"eligible_users": 2}
    assert "catalogue.json" in payload["files"]
