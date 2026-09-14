"""Deterministic catalogue-metadata normalization and lookup (Milestone 8-A).

This module turns the official Amazon Reviews 2023 product-metadata file into a
normalized, ``parent_asin``-keyed artifact and exposes a narrow read-only lookup
over it.

What this layer does **not** do
-------------------------------
It contains no candidate generation, no scoring of recommendable items, no ranking
between products, no reranking, no network access and no knowledge of SASRec.  The
accepted mapping stays authoritative for ``parent_asin <-> item_id``; this layer only
attaches descriptive facts to the external identity.

Determinism
-----------
Given the same raw input the pipeline produces the same artifact:

* records are consumed in file order and written in a deterministic key order;
* the duplicate policy is explicit (first record in file order wins) and counted;
* malformed records are rejected per an explicit policy and counted;
* record fields are emitted in a fixed schema order, so no dict/hash iteration
  order can affect the bytes;
* no network access happens after the raw file has been acquired;
* the destination is written atomically and never silently overwritten without
  explicit intent.

Missing metadata is a first-class outcome
-----------------------------------------
A catalog item the source does not cover is not an error and is not a reason to
drop a recommendation candidate.  :class:`~recommendation.catalog.schemas.MissingMetadata`
represents that case explicitly.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recommendation.io_utils import file_fingerprint

from .schemas import (
    MISSING_METADATA_STATUS,
    NORMALIZATION_VERSION,
    MetadataRecord,
    MissingMetadata,
    ProductMetadata,
)

__all__ = [
    "ARTIFACT_FORMAT",
    "DUPLICATE_POLICY",
    "CatalogCoverage",
    "MetadataIndex",
    "NormalizationOutcome",
    "NormalizationReport",
    "artifact_envelope",
    "build_metadata_artifact",
    "catalog_item_ids",
    "iter_metadata_records",
    "load_metadata_index",
    "normalize_product_record",
]

#: Format tag written into the artifact envelope.
ARTIFACT_FORMAT = "agentrecx.catalog_products.v1"

#: Explicit duplicate policy: the first record for a key in file order wins.
DUPLICATE_POLICY = "first"

#: Provenance label attached to every normalized record.
SOURCE_LABEL = "amazon_reviews_2023:meta_categories"

#: Fixed emission order for product records.  Writing fields in schema order means
#: the artifact bytes cannot depend on dict iteration order.
_FIELD_ORDER: tuple[str, ...] = (
    "parent_asin",
    "title",
    "subtitle",
    "author",
    "store",
    "main_category",
    "categories",
    "features",
    "description",
    "price_text",
    "average_rating",
    "rating_number",
    "details",
    "source",
)


# --------------------------------------------------------------------------- #
# Field-level coercion (missing stays missing)
# --------------------------------------------------------------------------- #


def _as_text(value: Any) -> str | None:
    """Return stripped text, or ``None`` when absent/blank/not a string."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_text_list(value: Any) -> tuple[str, ...]:
    """Return an order-preserving, whitespace-cleaned text tuple."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return ()
    seen: set[str] = set()
    cleaned: list[str] = []
    for entry in value:
        text = _as_text(entry)
        if text is None or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return tuple(cleaned)


def _as_details(value: Any) -> tuple[tuple[str, str], ...]:
    """Return the source detail bag as ordered, cleaned key/value pairs."""
    if not isinstance(value, dict):
        return ()
    pairs: list[tuple[str, str]] = []
    for key, raw in value.items():
        text_key = _as_text(key)
        text_value = _as_text(raw)
        if text_key is None or text_value is None:
            continue
        pairs.append((text_key, text_value))
    return tuple(pairs)


def _as_price_text(value: Any) -> str | None:
    """Return the parsed price value as canonical text, or ``None`` when unusable.

    Precise semantics: the real source carries this field as a JSON number, so
    ``json.loads`` has already converted it to a Python number before this function
    runs.  A number is therefore rendered with ``repr`` of its **parsed value**: this
    is a canonical representation, *not* the exact raw lexical token (a source ``1e2``
    becomes ``"100.0"``).  Text is used rather than a float so the field stays
    lossless relative to the parsed value without implying decimal or currency
    semantics that were never parsed.

    A source string is stripped and kept verbatim, which matters because the real data
    uses placeholder strings (an em dash) where the price is unknown.  Booleans and
    nulls are dropped.  Nothing is converted into a currency and nothing is inferred
    from another field.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return repr(value)
    return None


def _as_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` when absent/unparseable/non-finite."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        candidate = float(value)
    elif isinstance(value, str):
        try:
            candidate = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if math.isnan(candidate) or math.isinf(candidate):
        return None
    return candidate


def _as_int(value: Any) -> int | None:
    """Return a non-negative int, or ``None`` when absent/unparseable."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value) or value != int(value):
            return None
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = int(text)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def normalize_product_record(raw: dict[str, Any]) -> ProductMetadata:
    """Normalize one raw source record into :class:`ProductMetadata`.

    Raises
    ------
    ValueError
        If ``parent_asin`` is absent, not a string, or blank.  Such a record cannot
        be joined to the recommendation catalog and is therefore rejected by policy
        rather than guessed at.
    """
    if not isinstance(raw, dict):
        raise ValueError("record must be a JSON object")
    parent_asin = _as_text(raw.get("parent_asin"))
    if parent_asin is None:
        raise ValueError("record has no usable parent_asin")

    return ProductMetadata(
        parent_asin=parent_asin,
        title=_as_text(raw.get("title")),
        subtitle=_as_text(raw.get("subtitle")),
        author=_as_text(raw.get("author")),
        store=_as_text(raw.get("store")),
        main_category=_as_text(raw.get("main_category")),
        categories=_as_text_list(raw.get("categories")),
        features=_as_text_list(raw.get("features")),
        description=_as_text_list(raw.get("description")),
        # The parsed price value is rendered as canonical text; see _as_price_text
        # for why this is not the raw lexical token.
        price_text=_as_price_text(raw.get("price")),
        average_rating=_as_float(raw.get("average_rating")),
        rating_number=_as_int(raw.get("rating_number")),
        details=_as_details(raw.get("details")),
        source=SOURCE_LABEL,
    )


def record_to_ordered_dict(record: ProductMetadata) -> dict[str, Any]:
    """Return the record as JSON-ready values in the fixed schema order."""
    payload: dict[str, Any] = {}
    for name in _FIELD_ORDER:
        value = getattr(record, name)
        if isinstance(value, tuple):
            if name == "details":
                # Preserve key order in a plain object; the source itself is a map.
                payload[name] = {key: text for key, text in value}
            else:
                payload[name] = list(value)
        else:
            payload[name] = value
    return payload


# --------------------------------------------------------------------------- #
# Source streaming and catalog scope
# --------------------------------------------------------------------------- #


def iter_raw_metadata_records(path: str | Path) -> Iterator[tuple[int, Any]]:
    """Yield ``(line_number, parsed_or_None)`` for every non-empty source line.

    Unparseable lines are yielded as ``None`` so the caller can count them without
    the reader guessing at intent.
    """
    path = Path(path)
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("rt", encoding="utf-8")
    with handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield number, json.loads(text)
            except json.JSONDecodeError:
                yield number, None


def catalog_item_ids(mappings_path: str | Path) -> tuple[int, frozenset[str]]:
    """Return ``(num_items, parent_asins)`` from the accepted SASRec mapping.

    This reads the accepted mapping artifact; it never writes to it.  The mapping
    remains the single authority for catalogue membership.
    """
    with Path(mappings_path).open("rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    item2id = payload["item2id"]
    num_items = int(payload["num_items"])
    if len(item2id) != num_items:
        raise ValueError(
            f"mapping is inconsistent: item2id has {len(item2id)} entries but "
            f"num_items is {num_items}"
        )
    return num_items, frozenset(item2id)


# --------------------------------------------------------------------------- #
# Normalization report / outcome
# --------------------------------------------------------------------------- #


@dataclass
class NormalizationReport:
    """Counts describing one normalization run."""

    records_seen: int = 0
    parse_errors: int = 0
    non_object_records: int = 0
    missing_parent_asin: int = 0
    duplicate_keys: int = 0
    normalized_records: int = 0
    #: Records outside the accepted recommendation catalog (only counted when a
    #: catalog scope is supplied).
    records_outside_catalog: int = 0
    #: Small audit sample of rejected records (line number + reason).
    rejected_examples: list[dict[str, Any]] = field(default_factory=list)
    #: Small audit sample of duplicate keys (key + first/later line numbers).
    duplicate_examples: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "records_seen": self.records_seen,
            "parse_errors": self.parse_errors,
            "non_object_records": self.non_object_records,
            "missing_parent_asin": self.missing_parent_asin,
            "duplicate_keys": self.duplicate_keys,
            "normalized_records": self.normalized_records,
            "records_outside_catalog": self.records_outside_catalog,
            "rejected_examples": list(self.rejected_examples),
            "duplicate_examples": list(self.duplicate_examples),
        }


@dataclass
class CatalogCoverage:
    """Coverage of the accepted recommendation catalog by normalized metadata."""

    num_catalog_items: int
    num_metadata_records: int
    num_catalog_items_with_metadata: int
    num_catalog_items_missing_metadata: int
    num_metadata_records_not_in_catalog: int

    @property
    def coverage_percentage(self) -> float:
        """Percentage of catalog items that have metadata (0 when catalog is empty)."""
        if self.num_catalog_items == 0:
            return 0.0
        return 100.0 * self.num_catalog_items_with_metadata / self.num_catalog_items

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "num_catalog_items": self.num_catalog_items,
            "num_metadata_records": self.num_metadata_records,
            "num_catalog_items_with_metadata": self.num_catalog_items_with_metadata,
            "num_catalog_items_missing_metadata": self.num_catalog_items_missing_metadata,
            "num_metadata_records_not_in_catalog": self.num_metadata_records_not_in_catalog,
            "coverage_percentage": round(self.coverage_percentage, 6),
        }


@dataclass
class NormalizationOutcome:
    """Result of :func:`build_metadata_artifact`."""

    report: NormalizationReport
    coverage: CatalogCoverage
    artifact_path: Path
    artifact_sha256: str
    artifact_bytes: int
    manifest_path: Path | None
    elapsed_seconds: float
    envelope: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "report": self.report.as_dict(),
            "coverage": self.coverage.as_dict(),
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path),
            "elapsed_seconds": self.elapsed_seconds,
            "envelope": dict(self.envelope),
        }


def _load_catalog_scope(mappings_path: str | Path | None) -> tuple[int, frozenset[str] | None]:
    """Return ``(num_catalog_items, catalog_parent_asins or None)``."""
    if mappings_path is None:
        return 0, None
    return catalog_item_ids(mappings_path)


# --------------------------------------------------------------------------- #
# Artifact builder
# --------------------------------------------------------------------------- #


def build_metadata_artifact(
    raw_path: str | Path,
    artifact_path: str | Path,
    *,
    mappings_path: str | Path | None = None,
    category: str = "Sports_and_Outdoors",
    source_url: str | None = None,
    manifest_path: str | Path | None = None,
    catalog_only: bool = True,
    max_rejected_examples: int = 20,
    max_duplicate_examples: int = 20,
    overwrite: bool = True,
    progress_every: int = 500_000,
) -> NormalizationOutcome:
    """Normalize ``raw_path`` into a deterministic JSONL artifact at ``artifact_path``.

    Parameters
    ----------
    mappings_path:
        Accepted SASRec mapping artifact.  When supplied it defines the catalogue
        scope and the coverage statistics, and the mapping itself is only read.
    catalog_only:
        When true (default) only records whose ``parent_asin`` is in the accepted
        catalog are written.  Records outside the catalog are counted, never
        written and never used; this keeps the served artifact small while keeping
        the coverage arithmetic auditable.
    overwrite:
        Refuse to replace an existing artifact unless true, so an accidental rerun
        cannot silently clobber a verified artifact.

    The write is atomic: a temporary file in the destination directory is renamed
    into place only after the whole artifact has been written successfully.
    """
    raw_path = Path(raw_path)
    artifact_path = Path(artifact_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw metadata source not found: {raw_path}")
    if artifact_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing artifact without overwrite=True: {artifact_path}"
        )

    started = time.perf_counter()
    num_catalog_items, catalog_scope = _load_catalog_scope(mappings_path)
    report = NormalizationReport()

    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys: set[str] = set()
    first_seen_line: dict[str, int] = {}
    metadata_keys_in_scope: set[str] = set()

    # Phase 1: normalize into a records-only temporary file while accumulating the
    # counts the envelope needs.
    records_fd, records_name = tempfile.mkstemp(
        dir=str(artifact_path.parent), prefix=f".{artifact_path.name}.records.", suffix=".tmp"
    )
    try:
        with os.fdopen(records_fd, "w", encoding="utf-8") as handle:
            for line_number, raw in iter_raw_metadata_records(raw_path):
                report.records_seen += 1
                if raw is None:
                    report.parse_errors += 1
                    if len(report.rejected_examples) < max_rejected_examples:
                        report.rejected_examples.append(
                            {"line": line_number, "reason": "invalid_json"}
                        )
                    continue
                if not isinstance(raw, dict):
                    report.non_object_records += 1
                    if len(report.rejected_examples) < max_rejected_examples:
                        report.rejected_examples.append(
                            {"line": line_number, "reason": "not_a_json_object"}
                        )
                    continue

                parent_asin = _as_text(raw.get("parent_asin"))
                if parent_asin is None:
                    report.missing_parent_asin += 1
                    if len(report.rejected_examples) < max_rejected_examples:
                        report.rejected_examples.append(
                            {"line": line_number, "reason": "missing_parent_asin"}
                        )
                    continue

                # Explicit duplicate policy: first record in file order wins.
                if parent_asin in seen_keys:
                    report.duplicate_keys += 1
                    if len(report.duplicate_examples) < max_duplicate_examples:
                        report.duplicate_examples.append(
                            {
                                "parent_asin": parent_asin,
                                "first_line": first_seen_line[parent_asin],
                                "duplicate_line": line_number,
                            }
                        )
                    continue

                if catalog_scope is not None and parent_asin not in catalog_scope:
                    # Real source metadata that the catalog does not contain.  It is
                    # always counted; it is only *written* when catalog_only is false.
                    # This is a distinct quantity from "processed metadata outside the
                    # catalog", which is 0 for the default catalog-only artifact.
                    report.records_outside_catalog += 1
                    if catalog_only:
                        # The key is still recorded so a later duplicate of an
                        # out-of-scope record is not miscounted as in-scope data.
                        seen_keys.add(parent_asin)
                        first_seen_line[parent_asin] = line_number
                        continue

                record = normalize_product_record(raw)
                seen_keys.add(parent_asin)
                first_seen_line[parent_asin] = line_number
                metadata_keys_in_scope.add(parent_asin)

                # Every record reaching this point is written.  When a catalogue scope
                # is active, out-of-scope records were already filtered above, so
                # `catalog_only` expresses "restrict to the catalogue when one is
                # supplied" rather than a second, independent filter here.
                handle.write(json.dumps(record_to_ordered_dict(record), ensure_ascii=False))
                handle.write("\n")
                report.normalized_records += 1

                if progress_every and report.records_seen % progress_every == 0:
                    print(
                        f"  ... {report.records_seen:,} source records "
                        f"({report.normalized_records:,} in scope)",
                        flush=True,
                    )

            elapsed = time.perf_counter() - started
            # Coverage is derived from what the source actually supplied, by
            # intersecting the normalized keys with the catalog scope.  Deriving it
            # this way (rather than accumulating during the loop) keeps it correct for
            # every write mode, including catalog_only=False where out-of-catalog
            # records are also written.
            covered_in_catalog = (
                metadata_keys_in_scope & catalog_scope
                if catalog_scope is not None
                else frozenset()
            )
            coverage = CatalogCoverage(
                num_catalog_items=num_catalog_items,
                num_metadata_records=len(metadata_keys_in_scope),
                num_catalog_items_with_metadata=len(covered_in_catalog),
                num_catalog_items_missing_metadata=(
                    num_catalog_items - len(covered_in_catalog)
                    if catalog_scope is not None
                    else 0
                ),
                num_metadata_records_not_in_catalog=len(
                    metadata_keys_in_scope - covered_in_catalog
                ),
            )
            _assert_accounting_consistent(report, coverage, catalog_scope is not None)

            fingerprint = file_fingerprint(raw_path)
            envelope = artifact_envelope(
                category=category,
                report=report,
                coverage=coverage,
                raw_fingerprint=fingerprint,
                source_url=source_url,
                catalog_only=catalog_only,
            )

        # Phase 2: assemble the final artifact as `<envelope line>` + the records
        # file, streamed copy so a multi-hundred-megabyte artifact never lands in
        # memory.  Writing the envelope first (it is now fully known) means no
        # seek-back or truncate is needed, which removes a class of partial-write bug.
        assemble_fd, assemble_name = tempfile.mkstemp(
            dir=str(artifact_path.parent), prefix=f".{artifact_path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(assemble_fd, "w", encoding="utf-8") as out_handle:
                out_handle.write(json.dumps(envelope, ensure_ascii=False) + "\n")
                with open(records_name, "rt", encoding="utf-8") as in_handle:
                    shutil.copyfileobj(in_handle, out_handle, length=1 << 20)
            os.chmod(assemble_name, 0o644 & ~_umask())
            os.replace(assemble_name, artifact_path)
        except BaseException:
            Path(assemble_name).unlink(missing_ok=True)
            raise
    except BaseException:
        Path(records_name).unlink(missing_ok=True)
        raise
    Path(records_name).unlink(missing_ok=True)

    artifact_fingerprint = file_fingerprint(artifact_path)

    if manifest_path is not None:
        manifest = dict(envelope)
        manifest["manifest_format"] = "agentrecx.catalog_manifest.v1"
        manifest["artifact"] = artifact_fingerprint
        # Wall-clock timing belongs to the manifest, never to the reproducible artifact.
        manifest["elapsed_seconds"] = round(elapsed, 3)
        from recommendation.io_utils import write_json

        write_json(manifest_path, manifest)

    return NormalizationOutcome(
        report=report,
        coverage=coverage,
        artifact_path=artifact_path,
        artifact_sha256=artifact_fingerprint["sha256"],
        artifact_bytes=artifact_fingerprint["size_bytes"],
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
        elapsed_seconds=time.perf_counter() - started,
        envelope=envelope,
    )


def _assert_accounting_consistent(
    report: NormalizationReport,
    coverage: CatalogCoverage,
    has_catalog_scope: bool,
) -> None:
    """Fail loudly rather than write an artifact whose statistics cannot be true.

    Guards the identities the M8 report relies on:

    * every source record is either catalog-matched or outside the catalog;
    * catalog coverage never exceeds the catalog size and never goes negative;
    * the recorded in-catalog count is not greater than the records normalized.
    """
    if not has_catalog_scope:
        return
    matching = coverage.num_catalog_items_with_metadata
    if matching + report.records_outside_catalog != report.records_seen:
        raise AssertionError(
            "source accounting is inconsistent: "
            f"{matching} matched + {report.records_outside_catalog} outside "
            f"!= {report.records_seen} seen"
        )
    if not 0 <= matching <= coverage.num_catalog_items:
        raise AssertionError(
            f"catalog coverage {matching} is outside 0..{coverage.num_catalog_items}"
        )
    if coverage.num_catalog_items_missing_metadata < 0:
        raise AssertionError(
            f"negative missing-metadata count: {coverage.num_catalog_items_missing_metadata}"
        )


def _umask() -> int:
    """Return the process umask without leaving it changed."""
    value = os.umask(0)
    os.umask(value)
    return value


def artifact_envelope(
    *,
    category: str,
    report: NormalizationReport,
    coverage: CatalogCoverage,
    raw_fingerprint: dict[str, Any],
    source_url: str | None,
    catalog_only: bool,
) -> dict[str, Any]:
    """Build the self-describing header written as the artifact's first line.

    Deliberately **excludes** any wall-clock timing: the artifact must be
    byte-reproducible from the same raw input, so a duration would make its digest
    unstable for no benefit.  Run timings live in the manifest instead.
    """
    return {
        "format": ARTIFACT_FORMAT,
        "category": category,
        "normalization_version": NORMALIZATION_VERSION,
        "duplicate_policy": DUPLICATE_POLICY,
        "source": {
            "dataset": "Amazon Reviews 2023",
            "domain": "meta_categories",
            "category": category,
            "url": source_url,
            "raw_path": raw_fingerprint["path"],
            "raw_sha256": raw_fingerprint["sha256"],
            "raw_size_bytes": raw_fingerprint["size_bytes"],
            "raw_mtime_ns": raw_fingerprint["mtime_ns"],
        },
        "counts": report.as_dict(),
        "coverage": coverage.as_dict(),
        "catalog_only": catalog_only,
    }


# --------------------------------------------------------------------------- #
# Index / lookup
# --------------------------------------------------------------------------- #


@dataclass
class MetadataIndex:
    """Reusable, read-only index over a normalized catalog-metadata artifact.

    Loading happens once per instance; :meth:`lookup` is a dictionary probe, so a
    graph invocation never reparses the artifact.
    """

    records: dict[str, ProductMetadata]
    envelope: dict[str, Any]
    source_path: Path
    load_seconds: float = 0.0

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_records(
        cls,
        records: Iterable[ProductMetadata],
        *,
        envelope: dict[str, Any] | None = None,
        source_path: Path | None = None,
        load_seconds: float = 0.0,
    ) -> MetadataIndex:
        """Build an index from in-memory records (used by tests and callers)."""
        table: dict[str, ProductMetadata] = {}
        for record in records:
            # Explicit, deterministic duplicate policy mirroring the artifact
            # builder: the first record for a key wins.
            table.setdefault(record.parent_asin, record)
        return cls(
            records=table,
            envelope=dict(envelope or {}),
            source_path=source_path or Path("<in-memory>"),
            load_seconds=load_seconds,
        )

    @classmethod
    def load(cls, artifact_path: str | Path) -> MetadataIndex:
        """Load a normalized JSONL artifact, verifying its format envelope."""
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"catalog metadata artifact not found: {path}")

        started = time.perf_counter()
        table: dict[str, ProductMetadata] = {}
        envelope: dict[str, Any] = {}
        with path.open("rt", encoding="utf-8") as handle:
            first = handle.readline()
            if not first.strip():
                raise ValueError(f"artifact {path} has no envelope line")
            envelope = json.loads(first)
            if envelope.get("format") != ARTIFACT_FORMAT:
                raise ValueError(
                    f"artifact {path} has format {envelope.get('format')!r}, "
                    f"expected {ARTIFACT_FORMAT!r}"
                )
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                record = ProductMetadata(**json.loads(text))
                table.setdefault(record.parent_asin, record)

        return cls(
            records=table,
            envelope=envelope,
            source_path=path,
            load_seconds=time.perf_counter() - started,
        )

    # -- lookup ------------------------------------------------------------ #

    @property
    def size(self) -> int:
        """Number of metadata records held."""
        return len(self.records)

    def __contains__(self, parent_asin: object) -> bool:
        """True when a metadata record exists for ``parent_asin``."""
        return isinstance(parent_asin, str) and parent_asin in self.records

    def lookup(self, parent_asin: str) -> MetadataRecord:
        """Return metadata for one opaque identifier, or explicit absence.

        The argument is treated as an opaque string: it is only ever used as a
        dictionary key.  A non-string argument is a programming error, not a
        lookup miss, and raises ``TypeError``.
        """
        if not isinstance(parent_asin, str):
            raise TypeError(
                f"parent_asin must be a string, got {type(parent_asin).__name__}"
            )
        record = self.records.get(parent_asin)
        if record is None:
            return MissingMetadata(parent_asin=parent_asin or "<empty>")
        return record

    def lookup_many(self, parent_asins: Sequence[str]) -> tuple[MetadataRecord, ...]:
        """Return results positionally aligned with ``parent_asins``.

        Order and duplicates in the input are preserved exactly; the input sequence
        is never mutated.
        """
        return tuple(self.lookup(asin) for asin in parent_asins)

    def lookup_many_present(
        self, parent_asins: Sequence[str]
    ) -> tuple[ProductMetadata | None, ...]:
        """Aligned lookup returning ``None`` instead of :class:`MissingMetadata`."""
        results: list[ProductMetadata | None] = []
        for asin in parent_asins:
            record = self.records.get(asin) if isinstance(asin, str) else None
            results.append(record)
        return tuple(results)

    def coverage_against(self, catalog_parent_asins: Iterable[str]) -> CatalogCoverage:
        """Compute coverage of an arbitrary catalog scope against this index."""
        catalog = list(catalog_parent_asins)
        catalog_set = set(catalog)
        covered = [asin for asin in catalog if asin in self.records]
        return CatalogCoverage(
            num_catalog_items=len(catalog),
            num_metadata_records=self.size,
            num_catalog_items_with_metadata=len(covered),
            num_catalog_items_missing_metadata=len(catalog) - len(covered),
            num_metadata_records_not_in_catalog=sum(
                1 for asin in self.records if asin not in catalog_set
            ),
        )

    # -- diagnostics ------------------------------------------------------- #

    def field_population(self) -> dict[str, int]:
        """Count how many records populate each text-bearing field."""
        counts = {
            "title": 0,
            "store": 0,
            "main_category": 0,
            "categories": 0,
            "features": 0,
            "description": 0,
            "price_text": 0,
            "average_rating": 0,
            "rating_number": 0,
            "details": 0,
        }
        for record in self.records.values():
            for name in counts:
                if getattr(record, name):
                    counts[name] += 1
        return counts

    def metadata(self) -> dict[str, Any]:
        """Return a small JSON-serialisable identity block."""
        return {
            "format": self.envelope.get("format"),
            "normalization_version": self.envelope.get("normalization_version"),
            "source": self.envelope.get("source", {}),
            "records": self.size,
            "load_seconds": round(self.load_seconds, 4),
        }


def load_metadata_index(artifact_path: str | Path) -> MetadataIndex:
    """Convenience wrapper around :meth:`MetadataIndex.load`."""
    return MetadataIndex.load(artifact_path)
