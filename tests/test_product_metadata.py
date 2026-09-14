"""Milestone 8-A tests: catalogue metadata normalization and lookup.

Fully offline.  No real metadata artifact, no checkpoint and no catalogue: the
fixtures are small synthetic records, so this suite stays lightweight and runs in
normal pytest.

The properties proved here are the ones the rest of M8 depends on:

* normalization is deterministic and never invents a field the source lacks;
* lookup is aligned, duplicate-preserving and read-only;
* ``parent_asin`` is opaque -- nothing assumes a ``B`` prefix or any other shape;
* a catalogue item without metadata is an explicit result, not an error;
* this layer contains no candidate generation, no ranking and no SASRec knowledge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.product_catalog_fixture import (  # noqa: E402
    OPAQUE_ASINS,
    RAW_RECORDS,
    index,
    raw_records,
    records,
)
from recommendation.catalog import (  # noqa: E402
    ARTIFACT_FORMAT,
    DUPLICATE_POLICY,
    CatalogCoverage,
    MetadataIndex,
    MissingMetadata,
    ProductMetadata,
    build_metadata_artifact,
    catalog_item_ids,
    normalize_product_record,
)
from recommendation.catalog.schemas import NORMALIZATION_VERSION  # noqa: E402


# --------------------------------------------------------------------------- #
# 1 / 2. Known and missing lookups
# --------------------------------------------------------------------------- #


def test_known_identifier_returns_normalized_metadata() -> None:
    idx = index()
    record = idx.lookup("boot-001")
    assert isinstance(record, ProductMetadata)
    assert record.parent_asin == "boot-001"
    assert record.title == "Waterproof Hiking Boots"
    assert record.store == "Acme Outdoors"
    assert record.categories == ("Sports & Outdoors", "Outdoor Recreation", "Hiking")
    assert record.features == ("waterproof membrane", "vibram sole", "ankle support")
    assert record.average_rating == 4.6
    assert record.rating_number == 1200
    assert record.details_dict()["Color"] == "brown"


def test_missing_identifier_returns_explicit_absence_not_an_error() -> None:
    idx = index()
    record = idx.lookup("does-not-exist")
    assert isinstance(record, MissingMetadata)
    assert record.status == "missing"
    assert record.parent_asin == "does-not-exist"
    assert "does-not-exist" not in idx


def test_missing_metadata_is_not_fabricated() -> None:
    """An absent record must not gain invented fields."""
    absence = index().lookup("absent-item")
    assert not hasattr(absence, "title")
    assert absence.as_dict()["status"] == "missing"


# --------------------------------------------------------------------------- #
# 3 / 4 / 5. Alignment, duplicates, non-mutation
# --------------------------------------------------------------------------- #


def test_lookup_many_preserves_order() -> None:
    idx = index()
    requested = ["_tent-003", "boot-001", "9mat-002"]
    results = idx.lookup_many(requested)
    assert [result.parent_asin for result in results] == requested


def test_lookup_many_preserves_duplicates() -> None:
    idx = index()
    results = idx.lookup_many(["boot-001", "boot-001", "missing-x"])
    assert [type(result).__name__ for result in results] == [
        "ProductMetadata",
        "ProductMetadata",
        "MissingMetadata",
    ]
    assert len(results) == 3


def test_lookup_does_not_mutate_its_input() -> None:
    idx = index()
    requested = ["boot-001", "missing-x", "boot-001"]
    snapshot = list(requested)
    idx.lookup_many(requested)
    assert requested == snapshot


def test_lookup_rejects_a_non_string_identifier() -> None:
    """A non-string is a programming error, not a lookup miss."""
    with pytest.raises(TypeError):
        index().lookup(123)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 6. Deterministic normalization
# --------------------------------------------------------------------------- #


def test_normalization_is_deterministic() -> None:
    first = [normalize_product_record(record) for record in raw_records()]
    second = [normalize_product_record(record) for record in raw_records()]
    assert first == second
    assert [record.model_dump() for record in first] == [
        record.model_dump() for record in second
    ]


def test_normalization_is_idempotent() -> None:
    """Re-validating a normalized record must not change it."""
    for record in records():
        assert ProductMetadata(**record.model_dump()) == record


def test_whitespace_is_normalized_but_text_is_never_reworded() -> None:
    record = normalize_product_record(
        {"parent_asin": "  sp-1  ", "title": "  Two  spaces  kept  ", "features": ["  a  b  "]}
    )
    assert record.parent_asin == "sp-1"
    # Interior spacing is data; only surrounding whitespace is structural.
    assert record.title == "Two  spaces  kept"
    assert record.features == ("a  b",)


def test_list_order_is_preserved() -> None:
    record = normalize_product_record(
        {"parent_asin": "ord-1", "features": ["z", "a", "m"], "categories": ["c3", "c1", "c2"]}
    )
    assert record.features == ("z", "a", "m")
    assert record.categories == ("c3", "c1", "c2")


def test_details_keep_source_attribute_names_and_order() -> None:
    record = normalize_product_record(
        {"parent_asin": "d-1", "details": {"Brand Name": "X", "Color": "red", "Material": "steel"}}
    )
    assert record.details == (("Brand Name", "X"), ("Color", "red"), ("Material", "steel"))


def test_price_text_is_canonical_text_of_the_parsed_value() -> None:
    """`price_text` is the canonical text of the PARSED number, not the raw token.

    The source field is a JSON number, so `json.loads` has already produced a Python
    number before normalization runs.  This test pins that distinction explicitly so
    the documentation cannot drift back into claiming raw-token preservation.
    """
    assert normalize_product_record({"parent_asin": "p1", "price": 89.99}).price_text == "89.99"
    assert normalize_product_record({"parent_asin": "p2", "price": 55.0}).price_text == "55.0"
    assert normalize_product_record({"parent_asin": "p3", "price": 55}).price_text == "55"
    # A source exponent form does NOT survive lexically: it is the parsed value.
    assert normalize_product_record({"parent_asin": "p4", "price": 1e2}).price_text == "100.0"
    assert normalize_product_record({"parent_asin": "p5", "price": 0.1 + 0.2}).price_text == (
        "0.30000000000000004"
    )


def test_price_text_keeps_source_strings_verbatim_including_placeholders() -> None:
    """Real source data uses an em-dash placeholder for an unknown price."""
    assert normalize_product_record({"parent_asin": "s1", "price": "159.00"}).price_text == "159.00"
    assert normalize_product_record({"parent_asin": "s2", "price": "—"}).price_text == "—"
    assert normalize_product_record({"parent_asin": "s3", "price": "  9.99  "}).price_text == "9.99"
    assert normalize_product_record({"parent_asin": "s4", "price": "   "}).price_text is None


def test_price_is_never_parsed_as_currency_or_inferred() -> None:
    """No currency conversion, no inference from other fields."""
    assert normalize_product_record({"parent_asin": "c1", "price": True}).price_text is None
    assert normalize_product_record({"parent_asin": "c2"}).price_text is None
    # A price-like string elsewhere must not populate price_text.
    record = normalize_product_record({"parent_asin": "c3", "title": "$19.99 deal"})
    assert record.price_text is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("average_rating", "not-a-number"),
        ("average_rating", float("nan")),
        ("average_rating", float("inf")),
        ("average_rating", True),
        ("rating_number", -5),
        ("rating_number", 1.5),
        ("rating_number", "many"),
        ("rating_number", True),
    ],
)
def test_unusable_numeric_values_become_missing_not_guessed(field: str, value: object) -> None:
    record = normalize_product_record({"parent_asin": "n-1", field: value})
    assert getattr(record, field) is None


def test_absent_source_fields_stay_missing() -> None:
    """No field is inferred: missing means missing."""
    record = normalize_product_record({"parent_asin": "bare-1"})
    assert record.title is None
    assert record.store is None
    assert record.main_category is None
    assert record.categories == ()
    assert record.features == ()
    assert record.description == ()
    assert record.price_text is None
    assert record.average_rating is None
    assert record.rating_number is None
    assert record.details == ()
    assert record.has_searchable_text is False


def test_brand_is_not_inferred_from_title() -> None:
    """A title that looks like it contains a brand must not populate ``store``."""
    record = normalize_product_record(
        {"parent_asin": "t-1", "title": "Acme Waterproof Boots", "details": {}}
    )
    assert record.store is None
    assert "Brand Name" not in record.details_dict()


def test_extra_source_fields_are_ignored_not_invented_into_the_schema() -> None:
    """An unknown source field does not silently become a normalized field."""
    record = normalize_product_record(
        {"parent_asin": "x-1", "colour_from_source": "blue", "bought_together": ["other"]}
    )
    assert "colour_from_source" not in record.model_dump()
    assert "bought_together" not in record.model_dump()


# --------------------------------------------------------------------------- #
# 7 / 8. Duplicate and malformed source policy
# --------------------------------------------------------------------------- #


def test_record_without_parent_asin_is_rejected() -> None:
    for bad in ({"title": "no key"}, {"parent_asin": None}, {"parent_asin": "   "}):
        with pytest.raises(ValueError):
            normalize_product_record(bad)


def test_duplicate_policy_is_first_record_wins(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"parent_asin": "dup-1", "title": "first"},
                {"parent_asin": "dup-1", "title": "second"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    outcome = build_metadata_artifact(raw, tmp_path / "out.jsonl", category="T")
    assert DUPLICATE_POLICY == "first"
    assert outcome.report.duplicate_keys == 1
    assert outcome.report.duplicate_examples[0]["parent_asin"] == "dup-1"

    loaded = MetadataIndex.load(tmp_path / "out.jsonl")
    assert loaded.lookup("dup-1").title == "first"


def test_malformed_records_are_counted_and_skipped(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps({"parent_asin": "ok-1", "title": "good"}),
                "{not valid json",
                json.dumps(["not", "an", "object"]),
                json.dumps({"title": "no key at all"}),
                "",
                json.dumps({"parent_asin": "ok-2", "title": "also good"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outcome = build_metadata_artifact(raw, tmp_path / "out.jsonl", category="T")
    assert outcome.report.parse_errors == 1
    assert outcome.report.non_object_records == 1
    assert outcome.report.missing_parent_asin == 1
    assert outcome.report.normalized_records == 2
    reasons = {example["reason"] for example in outcome.report.rejected_examples}
    assert {"invalid_json", "not_a_json_object", "missing_parent_asin"} <= reasons

    loaded = MetadataIndex.load(tmp_path / "out.jsonl")
    assert loaded.size == 2


# --------------------------------------------------------------------------- #
# 9. Catalog coverage
# --------------------------------------------------------------------------- #


def test_coverage_arithmetic() -> None:
    idx = index()
    coverage = idx.coverage_against(["boot-001", "9mat-002", "unknown-a", "unknown-b"])
    assert isinstance(coverage, CatalogCoverage)
    assert coverage.num_catalog_items == 4
    assert coverage.num_metadata_records == idx.size
    assert coverage.num_catalog_items_with_metadata == 2
    assert coverage.num_catalog_items_missing_metadata == 2
    assert coverage.coverage_percentage == pytest.approx(50.0)


def test_coverage_counts_metadata_outside_the_catalog() -> None:
    idx = index()
    coverage = idx.coverage_against(["boot-001"])
    assert coverage.num_catalog_items == 1
    assert coverage.num_catalog_items_with_metadata == 1
    assert coverage.num_catalog_items_missing_metadata == 0
    # The other three synthetic records are not in this small catalogue scope.
    assert coverage.num_metadata_records_not_in_catalog == idx.size - 1


def test_coverage_of_empty_catalog_is_defined() -> None:
    coverage = index().coverage_against([])
    assert coverage.coverage_percentage == 0.0
    assert coverage.num_catalog_items_with_metadata == 0


def test_catalog_item_ids_reads_the_accepted_mapping(tmp_path: Path) -> None:
    mapping = tmp_path / "mappings.json"
    mapping.write_text(
        json.dumps({"num_items": 2, "item2id": {"a-1": 1, "b-2": 2}, "id2item": [None, "a-1", "b-2"]}),
        encoding="utf-8",
    )
    num_items, asins = catalog_item_ids(mapping)
    assert num_items == 2
    assert asins == frozenset({"a-1", "b-2"})


def test_catalog_only_artifact_keeps_only_catalog_items(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"parent_asin": "in-1", "title": "in catalog"},
                {"parent_asin": "out-1", "title": "not in catalog"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "mappings.json"
    mapping.write_text(
        json.dumps({"num_items": 1, "item2id": {"in-1": 1}, "id2item": [None, "in-1"]}),
        encoding="utf-8",
    )
    outcome = build_metadata_artifact(
        raw, tmp_path / "out.jsonl", mappings_path=mapping, category="T"
    )
    assert outcome.report.normalized_records == 1
    assert outcome.report.records_outside_catalog == 1
    assert outcome.coverage.num_catalog_items == 1
    assert outcome.coverage.num_catalog_items_with_metadata == 1
    assert outcome.coverage.num_catalog_items_missing_metadata == 0
    assert MetadataIndex.load(tmp_path / "out.jsonl").size == 1


# --------------------------------------------------------------------------- #
# 10 / 11. Opacity and identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("parent_asin", OPAQUE_ASINS)
def test_opaque_identifiers_are_not_assumed_to_start_with_b(parent_asin: str) -> None:
    """Identifiers are opaque strings; only exact matching matters."""
    assert parent_asin in index()
    record = index().lookup(parent_asin)
    assert isinstance(record, ProductMetadata)
    assert record.parent_asin == parent_asin


def test_metadata_never_exposes_or_requires_an_item_id() -> None:
    """The metadata contract has no integer item id: the mapping stays authoritative."""
    fields = set(ProductMetadata.model_fields)
    assert "item_id" not in fields
    assert fields.isdisjoint({"id", "int_id", "user_id"})


def test_no_item_id_remapping_occurs_during_normalization() -> None:
    """Whatever the source calls the item, the normalized key is the same string."""
    record = normalize_product_record({"parent_asin": "B0CXYZ", "title": "t"})
    assert record.parent_asin == "B0CXYZ"
    # A mapping artifact is never consulted or produced by normalization.
    assert not hasattr(record, "item_id")


def test_normalizing_does_not_require_a_mapping_file(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"parent_asin": "solo-1", "title": "x"}) + "\n", encoding="utf-8")
    outcome = build_metadata_artifact(raw, tmp_path / "out.jsonl", category="T")
    assert outcome.report.normalized_records == 1
    assert outcome.coverage.num_catalog_items == 0


# --------------------------------------------------------------------------- #
# 12. No candidate-generation logic in the metadata layer
# --------------------------------------------------------------------------- #


def _executable_source(module: object) -> str:
    """Return only executable source lines, with comments and docstrings removed.

    Documentation is allowed to *say* that the catalogue layer contains no ranking
    logic; only code may not actually reference those concepts.
    """
    import ast

    source = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]
    tree = ast.parse(source)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                doc_lines.update(
                    range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1)
                )
    return "\n".join(
        line
        for number, line in enumerate(source.splitlines(), start=1)
        if number not in doc_lines and line.strip() and not line.lstrip().startswith("#")
    )


def test_metadata_layer_has_no_candidate_generation_or_ranking_logic() -> None:
    """Guard: the catalogue layer must never score or select recommendable items.

    Matches are word-bounded so the project's own ``recommendation.`` package path is
    not mistaken for ranking logic.
    """
    import re

    import recommendation.catalog.metadata as metadata_module
    import recommendation.catalog.schemas as schemas_module

    forbidden_patterns = (
        r"\brecommend(s|ed|ing)?\b",
        r"\bscore(s|d|r)?\b",
        r"\branking\b|\brank\w*\(",
        r"\btop_k\b",
        r"\brerank\w*\b",
        r"\bsasrec\b",
        r"\bembedding\w*\b",
        r"\btorch\b",
    )
    for module in (metadata_module, schemas_module):
        code = _executable_source(module)
        for pattern in forbidden_patterns:
            match = re.search(pattern, code, flags=re.IGNORECASE)
            assert match is None, (
                f"{module.__name__} must not reference {pattern!r}; "
                f"found {match.group(0)!r}"  # type: ignore[union-attr]
            )


def test_metadata_package_does_not_import_inference_or_tools() -> None:
    """The catalogue layer must not depend on the recommender at all."""
    import recommendation.catalog.metadata as metadata_module
    import recommendation.catalog.schemas as schemas_module

    for module in (metadata_module, schemas_module):
        code = _executable_source(module)
        for forbidden in ("recommendation.inference", "recommendation.tools", "torch", "fastapi"):
            assert forbidden not in code, f"{module.__name__} must not import {forbidden}"


# --------------------------------------------------------------------------- #
# Artifact round-trip and reuse
# --------------------------------------------------------------------------- #


def test_artifact_round_trips_and_is_self_describing(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(json.dumps(record) for record in raw_records()) + "\n", encoding="utf-8"
    )
    outcome = build_metadata_artifact(raw, tmp_path / "out.jsonl", category="T")
    assert outcome.envelope["format"] == ARTIFACT_FORMAT
    assert outcome.envelope["normalization_version"] == NORMALIZATION_VERSION
    assert outcome.envelope["source"]["raw_sha256"]
    assert outcome.envelope["counts"]["normalized_records"] == len(RAW_RECORDS)

    loaded = MetadataIndex.load(tmp_path / "out.jsonl")
    assert loaded.size == len(RAW_RECORDS)
    for record in records():
        assert loaded.lookup(record.parent_asin) == record


def test_rebuilding_from_identical_input_is_byte_identical(tmp_path: Path) -> None:
    """Same raw input -> identical artifact bytes, independent of run or hash seed."""
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(json.dumps(record) for record in raw_records()) + "\n", encoding="utf-8"
    )
    first = build_metadata_artifact(raw, tmp_path / "a.jsonl", category="T")
    second = build_metadata_artifact(raw, tmp_path / "b.jsonl", category="T")
    assert first.artifact_sha256 == second.artifact_sha256


def test_overwrite_is_refused_by_default(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"parent_asin": "a", "title": "x"}) + "\n", encoding="utf-8")
    build_metadata_artifact(raw, tmp_path / "out.jsonl", category="T")
    with pytest.raises(FileExistsError):
        build_metadata_artifact(raw, tmp_path / "out.jsonl", category="T", overwrite=False)


def test_source_and_processed_population_accounting_is_internally_consistent(
    tmp_path: Path,
) -> None:
    """`matching + outside == total`, and the processed artifact is catalog-only.

    Guards the distinction between *raw source metadata outside the catalog* (a real,
    large number that is deliberately not written) and *processed metadata outside the
    catalog* (0 by construction), which the M8 report must never conflate.
    """
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"parent_asin": "in-1", "title": "in"},
                {"parent_asin": "in-2", "title": "in"},
                {"parent_asin": "out-1", "title": "out"},
                {"parent_asin": "out-2", "title": "out"},
                {"parent_asin": "out-3", "title": "out"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "mappings.json"
    mapping.write_text(
        json.dumps(
            {"num_items": 2, "item2id": {"in-1": 1, "in-2": 2}, "id2item": [None, "in-1", "in-2"]}
        ),
        encoding="utf-8",
    )
    outcome = build_metadata_artifact(
        raw, tmp_path / "out.jsonl", mappings_path=mapping, category="T"
    )
    report = outcome.report
    coverage = outcome.coverage

    total = report.records_seen
    matching = coverage.num_catalog_items_with_metadata
    outside = report.records_outside_catalog

    # Every source record is either in the catalog or outside it.
    assert matching + outside == total == 5
    assert matching == 2
    assert outside == 3

    # The processed artifact is catalog-only...
    assert report.normalized_records == coverage.num_catalog_items == 2
    assert coverage.num_metadata_records_not_in_catalog == 0
    # ...which is exactly why "processed outside catalog" cannot describe the source.
    assert coverage.num_metadata_records_not_in_catalog != outside

    loaded = MetadataIndex.load(tmp_path / "out.jsonl")
    assert loaded.size == report.normalized_records
    assert "out-1" not in loaded


def test_non_catalog_only_build_writes_every_source_record(tmp_path: Path) -> None:
    """`catalog_only=False` writes all source records; the accounting still balances."""
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"parent_asin": "in-1", "title": "in"},
                {"parent_asin": "out-1", "title": "out"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "mappings.json"
    mapping.write_text(
        json.dumps({"num_items": 1, "item2id": {"in-1": 1}, "id2item": [None, "in-1"]}),
        encoding="utf-8",
    )
    outcome = build_metadata_artifact(
        raw, tmp_path / "out.jsonl", mappings_path=mapping, category="T", catalog_only=False
    )
    assert outcome.report.normalized_records == 2
    # The source-level out-of-catalog count is still reported even when written.
    assert outcome.report.records_outside_catalog == 1
    assert outcome.coverage.num_metadata_records_not_in_catalog == 1
    assert MetadataIndex.load(tmp_path / "out.jsonl").size == 2


def test_artifact_load_rejects_a_foreign_format(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"format": "something-else"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        MetadataIndex.load(bad)


def test_index_is_loaded_once_and_reused() -> None:
    """Lookups are dictionary probes; the artifact is not reparsed per call."""
    idx = index()
    assert idx.load_seconds >= 0.0
    first = idx.lookup("boot-001")
    for _ in range(50):
        assert idx.lookup("boot-001") is first


def test_index_metadata_reports_source_identity() -> None:
    idx = index()
    info = idx.metadata()
    assert info["records"] == idx.size
    assert "load_seconds" in info
