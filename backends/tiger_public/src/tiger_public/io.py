"""Artifact I/O for the public-TIGER backend boundary (Step 2.3).

Three jobs, kept apart:

1. **Hashing.**  Every artifact directory carries a ``manifest.json`` listing the sha256 of
   each file it wrote.  :func:`verify_manifest` re-hashes them, and
   :func:`validate_declared_files` additionally refuses a *missing* file and a hash that does
   not look like a digest.
2. **Reading.**  :func:`read_catalogue`, :func:`read_train_exposure` and
   :func:`read_eval_cohort` reconstruct the frozen dataclasses in
   :mod:`tiger_public.contracts` and run their ``validate()``, so a malformed handoff fails
   at the boundary instead of producing a plausible, wrong score matrix.
3. **Forbidden-key refusal.**  :func:`assert_no_forbidden_keys` is the mechanical form of the
   "no target is visible to the backend" rule: a handoff record carrying ``target``, ``label``,
   ``valid`` or ``test`` is refused before it is parsed into anything.

Step 2.3 writes no features, no semantic IDs and no checkpoint: those artifacts belong to
Steps 2.4-2.6.  What exists here is the reader and the digest discipline they will both reuse.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from tiger_public.contracts import (
    CONTRACT_VERSION,
    CatalogueHandoff,
    ContractError,
    EvalCohortHandoff,
    ScoreBatch,
    TrainExposureHandoff,
)

__all__ = [
    "ArtifactError",
    "assert_no_forbidden_keys",
    "iter_jsonl",
    "read_catalogue",
    "read_eval_cohort",
    "read_exposure_rows",
    "read_manifest",
    "read_semantic_ids",
    "read_train_exposure",
    "sha256_bytes",
    "sha256_file",
    "validate_declared_files",
    "verify_manifest",
    "write_json",
    "write_jsonl",
    "write_manifest",
]

#: Keys a handoff record may never carry.  Their presence means a target leaked across the
#: boundary, which is a refusal rather than a warning.
FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {"target", "targets", "label", "labels", "valid", "validation", "test", "test_target",
     "validation_target", "seen"}
)


class ArtifactError(ContractError):
    """Raised when an artifact is missing, malformed, or fails its digest check."""


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Stream a file's sha256 so a 300 MB artifact never has to be held in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write deterministic JSON and return its sha256.

    ``sort_keys=True`` plus a trailing newline makes the bytes a function of the payload, so
    two runs with the same configuration produce the same digest.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    target.write_text(text, encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> str:
    """Write one canonical JSON object per line and return the content's sha256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(record), sort_keys=True, ensure_ascii=True) for record in records]
    text = "".join(line + "\n" for line in lines)
    target.write_text(text, encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))


def write_manifest(
    directory: str | Path,
    *,
    extra: Mapping[str, Any] | None = None,
    contract_version: str = CONTRACT_VERSION,
) -> dict[str, Any]:
    """Hash every file a stage wrote and record the digests in ``manifest.json``.

    The manifest is written last and excludes itself.  Everything it lists is re-hashable by
    :func:`verify_manifest`, which the adapter runs before it trusts a single byte.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files[str(path.relative_to(root))] = sha256_file(path)
    payload: dict[str, Any] = {
        "format": "agentrecx.tiger.manifest.v3",
        "contract_version": contract_version,
        "files": files,
    }
    if extra:
        payload.update(dict(extra))
    write_json(root / "manifest.json", payload)
    return payload


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each JSONL record, refusing a blank or non-object line with its line number."""
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArtifactError(f"{target.name} line {number} is not JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ArtifactError(
                f"{target.name} line {number} must be a JSON object, "
                f"got {type(payload).__name__}"
            )
        yield payload


def assert_no_forbidden_keys(
    payload: Mapping[str, Any], *, context: str, allow: Sequence[str] = ()
) -> None:
    """Refuse a record carrying a target-shaped key.

    This is the mechanical form of "no target is visible to the backend".  A permitted key is
    still refused if it is *declared* forbidden, so the allow-list can only narrow.
    """
    permitted = set(allow)
    for key in payload:
        if key in permitted:
            continue
        if str(key).lower() in FORBIDDEN_KEYS:
            raise ArtifactError(
                f"{context} declares forbidden key {key!r}; a handoff artifact must not carry "
                "a target, a label, a split name, or a seen set"
            )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactError(f"missing required artifact {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactError(f"{path.name} must hold a JSON object")
    return payload


def _require(payload: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in payload:
        raise ArtifactError(f"{context} is missing required field {key!r}")
    return payload[key]


def read_manifest(directory: str | Path) -> dict[str, Any]:
    """Read ``manifest.json``, refusing a manifest with no ``files`` block."""
    path = Path(directory) / "manifest.json"
    payload = _read_json(path)
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ArtifactError(f"{path} must declare a non-empty 'files' object")
    return payload


def validate_declared_files(directory: str | Path, *, required: Sequence[str] = ()) -> dict[str, Any]:
    """Refuse a missing file and a declared digest that is not a sha256.

    This is the gate the adapter applies before it reads a single score: a handoff that claims
    a blank or truncated digest is rejected rather than trusted.
    """
    root = Path(directory)
    manifest = read_manifest(root)
    for name, digest in manifest["files"].items():
        if not _looks_like_sha256(digest):
            raise ArtifactError(
                f"manifest declares a non-sha256 digest for {name!r}: {digest!r}"
            )
        if not (root / name).is_file():
            raise ArtifactError(f"manifest declares {name!r}, which does not exist")
    for name in required:
        if name not in manifest["files"]:
            raise ArtifactError(f"manifest does not declare the required file {name!r}")
    return manifest


def verify_manifest(directory: str | Path, *, required: Sequence[str] = ()) -> dict[str, Any]:
    """Validate declared files, then re-hash each one and compare."""
    root = Path(directory)
    manifest = validate_declared_files(root, required=required)
    mismatched: list[str] = []
    for name, digest in manifest["files"].items():
        if sha256_file(root / name) != str(digest).lower():
            mismatched.append(name)
    if mismatched:
        raise ArtifactError(f"artifact digests do not match: {sorted(mismatched)}")
    return manifest


# --------------------------------------------------------------------------- #
# Handoff readers
# --------------------------------------------------------------------------- #


def read_catalogue(directory: str | Path, *, verify: bool = True) -> CatalogueHandoff:
    """Rebuild and validate :class:`CatalogueHandoff` from a handoff directory."""
    root = Path(directory)
    if verify:
        verify_manifest(root, required=("catalogue.json", "catalogue_items.jsonl"))
    payload = _read_json(root / "catalogue.json")
    assert_no_forbidden_keys(payload, context="catalogue.json")
    item_ids: list[int] = []
    for number, record in enumerate(iter_jsonl(root / "catalogue_items.jsonl"), start=1):
        assert_no_forbidden_keys(record, context=f"catalogue_items.jsonl line {number}")
        if set(record) != {"backend_row", "item_id"}:
            raise ArtifactError(
                "catalogue_items.jsonl records must hold exactly "
                f"{{'backend_row', 'item_id'}}, got {sorted(record)}"
            )
        if int(record["backend_row"]) != number - 1:
            raise ArtifactError(
                f"catalogue_items.jsonl line {number} declares backend_row "
                f"{record['backend_row']}; rows must be ascending and contiguous from 0"
            )
        item_ids.append(int(record["item_id"]))
    span = payload.get("item_id_span")
    handoff = CatalogueHandoff(
        format=str(_require(payload, "format", "catalogue.json")),
        contract_version=str(
            payload.get("contract_version", CONTRACT_VERSION)
        ),
        num_items=int(_require(payload, "num_items", "catalogue.json")),
        num_users=int(_require(payload, "num_users", "catalogue.json")),
        pad_id=int(_require(payload, "pad_id", "catalogue.json")),
        first_real_id=int(_require(payload, "first_real_id", "catalogue.json")),
        item_id_span=(int(span[0]), int(span[1])) if span else (0, 0),
        mappings_sha256=str(_require(payload, "mappings_sha256", "catalogue.json")),
        sequences_sha256=str(_require(payload, "sequences_sha256", "catalogue.json")),
        products_sha256=str(_require(payload, "products_sha256", "catalogue.json")),
        item_ids=tuple(item_ids),
    )
    handoff.validate()
    return handoff


def read_exposure_rows(path: str | Path) -> tuple[tuple[int, ...], ...]:
    """Read only the item-id rows of an exposure file.

    The function has no parameter for, and no access to, a target: it reads ``items`` and
    nothing else, which is this layer's half of the train-history-only guarantee.
    """
    rows: list[tuple[int, ...]] = []
    for number, record in enumerate(iter_jsonl(path), start=1):
        assert_no_forbidden_keys(record, context=f"train_exposure.jsonl line {number}")
        if set(record) - {"case_id", "items"}:
            raise ArtifactError(
                "train_exposure.jsonl records may hold only 'case_id' and 'items', got "
                f"{sorted(record)}"
            )
        items = record.get("items")
        if not isinstance(items, list) or not items:
            raise ArtifactError(
                f"train_exposure.jsonl line {number} must hold a non-empty 'items' list"
            )
        rows.append(tuple(int(item) for item in items))
    return tuple(rows)


def read_train_exposure(
    directory: str | Path,
    *,
    catalogue: CatalogueHandoff | None = None,
    verify: bool = True,
) -> TrainExposureHandoff:
    """Rebuild and validate :class:`TrainExposureHandoff` from a handoff directory."""
    root = Path(directory)
    if verify:
        verify_manifest(root, required=("train_exposure.jsonl",))
    payload = _read_json(root / "train_exposure.json")
    assert_no_forbidden_keys(payload, context="train_exposure.json")
    rows = read_exposure_rows(root / "train_exposure.jsonl")
    handoff = TrainExposureHandoff(
        format=str(_require(payload, "format", "train_exposure.json")),
        contract_version=str(payload.get("contract_version", CONTRACT_VERSION)),
        protocol=str(_require(payload, "protocol", "train_exposure.json")),
        protocol_version=str(_require(payload, "protocol_version", "train_exposure.json")),
        field_source=str(_require(payload, "field_source", "train_exposure.json")),
        users=int(_require(payload, "users", "train_exposure.json")),
        examples=int(_require(payload, "examples", "train_exposure.json")),
        rows=rows,
        examples_sha256=str(_require(payload, "examples_sha256", "train_exposure.json")),
        catalogue_sha256=str(_require(payload, "catalogue_sha256", "train_exposure.json")),
    )
    handoff.validate(catalogue=catalogue)
    return handoff


def read_eval_cohort(
    directory: str | Path,
    *,
    catalogue: CatalogueHandoff | None = None,
    verify: bool = True,
) -> EvalCohortHandoff:
    """Rebuild and validate :class:`EvalCohortHandoff` from a handoff directory.

    The reader requires ``history`` and ``required_frontier`` and has no notion of a target, so
    a cohort file that carried one would have to also carry a target-shaped key and would be
    refused by :func:`assert_no_forbidden_keys`.
    """
    root = Path(directory)
    if verify:
        verify_manifest(root, required=("eval_cohort.json", "eval_cohort.jsonl"))
    payload = _read_json(root / "eval_cohort.json")
    assert_no_forbidden_keys(payload, context="eval_cohort.json")
    case_ids: list[int] = []
    histories: list[tuple[int, ...]] = []
    frontiers: list[int] = []
    for number, record in enumerate(iter_jsonl(root / "eval_cohort.jsonl"), start=1):
        assert_no_forbidden_keys(record, context=f"eval_cohort.jsonl line {number}")
        if set(record) != {"case_id", "history", "required_frontier"}:
            raise ArtifactError(
                "eval_cohort.jsonl records must hold exactly "
                f"{{'case_id', 'history', 'required_frontier'}}, got {sorted(record)}"
            )
        history = record["history"]
        if not isinstance(history, list) or not history:
            raise ArtifactError(
                f"eval_cohort.jsonl line {number} must hold a non-empty 'history' list"
            )
        case_ids.append(int(record["case_id"]))
        histories.append(tuple(int(item) for item in history))
        frontiers.append(int(record["required_frontier"]))
    k_values = payload.get("k_values")
    handoff = EvalCohortHandoff(
        format=str(_require(payload, "format", "eval_cohort.json")),
        contract_version=str(payload.get("contract_version", CONTRACT_VERSION)),
        cohort_seed=int(_require(payload, "cohort_seed", "eval_cohort.json")),
        cohort_size=int(_require(payload, "cohort_size", "eval_cohort.json")),
        protocol_version=str(_require(payload, "protocol_version", "eval_cohort.json")),
        k_values=tuple(int(value) for value in (k_values or ())),
        case_ids=tuple(case_ids),
        test_histories=tuple(histories),
        required_frontier=tuple(frontiers),
        catalogue_sha256=str(_require(payload, "catalogue_sha256", "eval_cohort.json")),
    )
    handoff.validate(catalogue=catalogue)
    return handoff


def read_semantic_ids(directory: str | Path, *, verify: bool = True):
    """Read a ``semantic_ids.json`` produced by stage 2.

    Step 2.3 has no stage-2 producer, so this reader exists to keep the *contract* testable:
    the adapter and the backend both consume exactly this shape when Step 2.4 lands.
    """
    from tiger_public.contracts import SemanticIdArtifact

    root = Path(directory)
    path = root / "semantic_ids.json"
    if verify and (root / "manifest.json").is_file():
        verify_manifest(root)
    payload = _read_json(path)
    assert_no_forbidden_keys(payload, context="semantic_ids.json")
    assignment = payload.get("assignment")
    if not isinstance(assignment, list):
        raise ArtifactError("semantic_ids.json must hold an 'assignment' list")
    audit = payload.get("collision_audit") or {}
    over_pre = audit.get("pre_dedup") or {}
    over = over_pre.get("largest_group")
    dedup_size = int(payload.get("dedup_vocab_size", payload.get("codebook_size", 0)))
    if over is not None and int(over) > dedup_size:
        raise ArtifactError(
            f"largest pre-dedup collision group {over} exceeds dedup_vocab_size "
            f"{dedup_size}; re-fit with a larger codebook, more levels, or a wider dedup level"
        )
    artifact = SemanticIdArtifact(
        format=str(_require(payload, "format", "semantic_ids.json")),
        contract_version=str(payload.get("contract_version", CONTRACT_VERSION)),
        num_items=int(_require(payload, "num_items", "semantic_ids.json")),
        levels=int(_require(payload, "levels", "semantic_ids.json")),
        dedup_levels=int(_require(payload, "dedup_levels", "semantic_ids.json")),
        codebook_size=int(_require(payload, "codebook_size", "semantic_ids.json")),
        dedup_vocab_size=dedup_size,
        pad_row=tuple(int(value) for value in _require(payload, "pad_row", "semantic_ids.json")),
        assignment=tuple(tuple(int(value) for value in row) for row in assignment),
        collision_audit=audit,
        utilisation=tuple(payload.get("utilisation") or ()),
        content_features=payload.get("content_features") or {},
    )
    artifact.validate()
    return artifact


def merge_score_batches(batches: Iterable[ScoreBatch]) -> dict[str, Any]:
    """Summarise a stream of batches for a report, without touching the score values."""
    statuses: set[str] = set()
    batches_seen = 0
    cases = 0
    reachable_real: list[int] = []
    for batch in batches:
        batches_seen += 1
        cases += len(batch.case_ids)
        statuses.add(batch.status)
        reachable_real.extend(int(value) for value in batch.reachable_real_count)
    return {
        "batches": batches_seen,
        "cases": cases,
        "statuses": sorted(statuses),
        "min_reachable_real": min(reachable_real) if reachable_real else 0,
        "max_reachable_real": max(reachable_real) if reachable_real else 0,
    }
