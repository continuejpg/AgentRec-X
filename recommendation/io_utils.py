"""Input/output and validation helpers for the recommendation pipeline.

Responsibilities:
  * read raw Amazon Reviews 2023 interaction files (``.jsonl`` / ``.jsonl.gz``,
    plus optional parquet) as *unmodified* raw records;
  * validate and normalise one raw record into the pipeline schema;
  * write JSON artifacts atomically.

Design notes
------------
Raw records are treated as read-only: normalisation constructs a brand new dict
and never mutates the object handed in.  Files opened for reading are opened in
text mode only; this module contains no code path that writes to an input file.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

# --------------------------------------------------------------------------- #
# Raw reading
# --------------------------------------------------------------------------- #


def iter_raw_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield raw records from ``path`` one at a time.

    Supported formats: ``.jsonl``, ``.jsonl.gz`` (Amazon Reviews 2023 layout)
    and ``.parquet`` (requires the optional ``pandas`` + ``pyarrow`` extras).

    Blank lines are skipped.  A line that is not valid JSON is yielded as
    ``{"__parse_error__": <message>, "__line_number__": <n>}`` so the caller can
    count and report it instead of crashing on a single corrupt row.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"raw input not found: {path}")

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        yield from _iter_parquet_records(path)
        return

    opener = gzip.open if suffixes.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"__parse_error__": str(exc), "__line_number__": line_number}
                continue
            if not isinstance(record, Mapping):
                yield {
                    "__parse_error__": f"expected JSON object, got {type(record).__name__}",
                    "__line_number__": line_number,
                }
                continue
            yield dict(record)


def _iter_parquet_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a parquet file (optional dependency)."""
    try:
        import pandas as pd  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "reading .parquet requires the optional dependencies 'pandas' and "
            "'pyarrow'; install them or convert the file to .jsonl(.gz). "
            "The Amazon Reviews 2023 raw release is .jsonl.gz, so this is only "
            "needed if you supplied your own parquet export."
        ) from exc

    frame = pd.read_parquet(path)
    for record in frame.to_dict(orient="records"):
        yield {str(key): value for key, value in record.items()}


# --------------------------------------------------------------------------- #
# Validation / normalisation
# --------------------------------------------------------------------------- #

#: Machine-readable reasons a raw record can be rejected.
REASON_NOT_OBJECT = "not_an_object"
REASON_MISSING_FIELDS = "missing_fields"
REASON_BAD_USER = "bad_user_id"
REASON_BAD_ITEM = "bad_parent_asin"
REASON_BAD_TIMESTAMP = "bad_timestamp"
REASON_BAD_RATING = "bad_rating"


@dataclass
class NormalizationReport:
    """Counters describing what happened while normalising raw records."""

    total_records: int = 0
    parse_errors: int = 0
    rejected: Counter = field(default_factory=Counter)
    #: Records that passed validation (before duplicate collapsing).
    valid_records: int = 0
    duplicate_records_dropped: int = 0
    #: True when at least one timestamp looked like seconds rather than
    #: milliseconds (both are accepted; this is informational only).
    saw_second_precision_timestamp: bool = False

    @property
    def accepted(self) -> int:
        """Number of usable interactions, i.e. valid records minus duplicates.

        This is exactly the length of the list returned by
        :func:`load_interactions`, so the report can never disagree with the data.
        """
        return self.valid_records - self.duplicate_records_dropped

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the report."""
        return {
            "total_records": self.total_records,
            "parse_errors": self.parse_errors,
            "rejected_total": sum(self.rejected.values()),
            "rejected_by_reason": dict(sorted(self.rejected.items())),
            "valid_records": self.valid_records,
            "duplicate_records_dropped": self.duplicate_records_dropped,
            "accepted_records": self.accepted,
            "timestamp_precision": (
                "seconds" if self.saw_second_precision_timestamp else "milliseconds"
            ),
        }


@dataclass(frozen=True)
class Interaction:
    """One validated user-item interaction.

    ``timestamp`` is stored exactly as found in the raw file (the Amazon
    Reviews 2023 release uses unix *milliseconds*).  The pipeline never rescales
    it - only the ordering matters - which keeps the raw values auditable.
    """

    user_id: str
    item_id: str
    timestamp: int
    rating: float


def _coerce_timestamp(value: Any) -> int | None:
    """Coerce a raw timestamp to ``int``; return ``None`` if impossible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return None
    return None


def _coerce_rating(value: Any) -> float | None:
    """Coerce a raw rating to ``float``; return ``None`` if impossible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None
    if number != number:  # NaN
        return None
    return number


def _clean_identifier(value: Any) -> str | None:
    """Return a usable identifier string, or ``None`` if the value is unusable."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def normalize_record(
    record: Mapping[str, Any],
    report: NormalizationReport | None = None,
) -> Interaction | None:
    """Validate one raw record and convert it to an :class:`Interaction`.

    Returns ``None`` (and records the reason in ``report``) when the record is
    malformed, incomplete or semantically unusable.  The input mapping is never
    modified.
    """
    if report is not None:
        report.total_records += 1

    if not isinstance(record, Mapping):
        if report is not None:
            report.rejected[REASON_NOT_OBJECT] += 1
        return None

    if "__parse_error__" in record:
        if report is not None:
            report.parse_errors += 1
        return None

    missing = [name for name in config.REQUIRED_FIELDS if name not in record]
    if missing:
        if report is not None:
            report.rejected[REASON_MISSING_FIELDS] += 1
        return None

    user_id = _clean_identifier(record.get(config.FIELD_USER))
    if user_id is None:
        if report is not None:
            report.rejected[REASON_BAD_USER] += 1
        return None

    item_id = _clean_identifier(record.get(config.FIELD_ITEM))
    if item_id is None:
        if report is not None:
            report.rejected[REASON_BAD_ITEM] += 1
        return None

    timestamp = _coerce_timestamp(record.get(config.FIELD_TIMESTAMP))
    if timestamp is None:
        if report is not None:
            report.rejected[REASON_BAD_TIMESTAMP] += 1
        return None

    rating = _coerce_rating(record.get(config.FIELD_RATING))
    if rating is None:
        if report is not None:
            report.rejected[REASON_BAD_RATING] += 1
        return None

    if report is not None and timestamp < config.MIN_PLAUSIBLE_TIMESTAMP_SECONDS * 1000:
        # A value this small is either a seconds-precision timestamp from a
        # different mirror, or genuinely corrupt.  Either way it is accepted;
        # ordering within a user's history is what matters downstream.
        report.saw_second_precision_timestamp = True

    if report is not None:
        report.valid_records += 1

    return Interaction(
        user_id=user_id,
        item_id=item_id,
        timestamp=timestamp,
        rating=rating,
    )


def load_interactions(
    path: str | Path,
    deduplicate: str = config.DEDUPLICATE_POLICY,
) -> tuple[list[Interaction], NormalizationReport]:
    """Read, validate and deduplicate interactions from a raw file.

    Parameters
    ----------
    path:
        Raw ``.jsonl`` / ``.jsonl.gz`` / ``.parquet`` interaction file.
    deduplicate:
        ``"last"``, ``"first"`` or ``"keep"`` - how to treat records repeating
        the same ``(user_id, item_id, timestamp)`` triple.

    Returns
    -------
    (interactions, report)
        ``interactions`` are validated and independent of the raw objects;
        ``report`` describes every rejection.
    """
    if deduplicate not in config.VALID_DEDUPLICATE_POLICIES:
        raise ValueError(
            f"deduplicate must be one of {config.VALID_DEDUPLICATE_POLICIES}, got {deduplicate!r}"
        )

    report = NormalizationReport()

    if deduplicate == "keep":
        # No dedup bookkeeping at all: the result is just the validated stream.
        kept = [x for x in _normalized_stream(path, report)]
        return kept, report

    # Only one dict is held for the deduplicating paths.  A separate list would
    # double the memory of the largest category (~20M interactions), and it is
    # unnecessary: Python dicts preserve insertion order, so the keys are already
    # in first-seen order, which is the order we want to preserve.
    by_key: dict[tuple[str, str, int], Interaction] = {}

    for interaction in _normalized_stream(path, report):
        key = (interaction.user_id, interaction.item_id, interaction.timestamp)
        if key not in by_key:
            by_key[key] = interaction
            continue

        report.duplicate_records_dropped += 1
        if deduplicate == "last":
            # The most recent payload wins; because the key is identical only
            # the rating can differ, and re-assigning keeps the original
            # insertion position.
            by_key[key] = interaction
        # "first": keep the existing entry untouched.

    return list(by_key.values()), report


def _normalized_stream(path: str | Path, report: NormalizationReport):
    """Yield one validated :class:`Interaction` per usable raw record."""
    for raw in iter_raw_records(path):
        interaction = normalize_record(raw, report)
        if interaction is not None:
            yield interaction


# --------------------------------------------------------------------------- #
# Writing artifacts
# --------------------------------------------------------------------------- #


def _default_file_mode() -> int:
    """Return the mode a normally created file would get (0666 minus umask)."""
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as pretty JSON to ``path`` atomically.

    Writing to a temporary file in the same directory and then replacing the
    destination avoids leaving a half-written artifact behind if the process is
    interrupted.  ``json.dump`` streams, so large sequence files do not require
    a second full copy in memory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
        # mkstemp creates 0600; processed artifacts are data, not secrets, so
        # give them the permissions a normal file creation would have produced.
        os.chmod(temp_name, _default_file_mode())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    """Stream ``rows`` to a JSON Lines file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        os.chmod(temp_name, _default_file_mode())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def file_fingerprint(path: str | Path, chunk_size: int = 1 << 20) -> dict[str, Any]:
    """Return an identity record for a file: path, size, mtime and SHA-256.

    Used to prove in verification that the raw input was not modified: capture
    the fingerprint before and after a pipeline run and compare.  The digest is
    streamed in ``chunk_size`` blocks so a multi-GB raw archive never lands in
    memory, and the stat fields are included so a *moved or replaced* file is
    distinguishable from an unchanged one.
    """
    path = Path(path)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
            "sha256": None,
        }

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)

    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def read_json(path: str | Path) -> Any:
    """Read a JSON artifact produced by this package."""
    with open(Path(path), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def text_preview(path: str | Path, max_chars: int = 800) -> str:
    """Return the first ``max_chars`` characters of a file (for inspection)."""
    with open(Path(path), "rt", encoding="utf-8") as handle:
        return handle.read(max_chars)
