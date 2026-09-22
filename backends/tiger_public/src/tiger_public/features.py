"""Frozen item-content encoding (Step 2.4, stage 1).

```text
products_text.jsonl   ->   frozen sentence encoder   ->   item_features.npy [N, 768]
```

The backend reads **only** ``products_text.jsonl``: the text was composed by AgentRec-X's
materialisation layer, and neither the original catalogue nor canonical identity is available
here.  That is what makes the encoder stage unable to leak an identity it never receives.

Two encoders are available, and the artifact always says which one produced it:

``sentence-t5``
    The audited baseline, ``sentence-transformers/sentence-t5-base``, frozen.  This is the
    encoder a real run uses.

``smoke``
    A deterministic, network-free stand-in: character-ngram hashing projected to 768
    dimensions.  It exists because the full pipeline - batching, hashing, the artifact chain,
    the RQ-VAE, the dedup pass, every gate - must be exercisable without a GPU and without
    downloading weights.  It is labelled ``step-2.4-smoke-encoder-not-sentence-t5`` in the
    artifact, so no measurement taken with it can be mistaken for a Sentence-T5 result.  It is
    **not** a model and carries no semantics.

Streaming is the default: rows are encoded in batches and written straight into a memory-mapped
array, so peak host memory does not scale with ``num_items``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from tiger_public.contracts import CONTRACT_VERSION

__all__ = [
    "ENCODER_SMOKE",
    "ENCODER_SENTENCE_T5",
    "EncoderError",
    "FeatureBuildOutcome",
    "TextBatch",
    "build_item_features",
    "iter_text_batches",
    "resolve_encoder",
]

#: The audited baseline encoder.
ENCODER_SENTENCE_T5 = "sentence-transformers/sentence-t5-base"

#: The network-free diagnostic stand-in.  Never a model; see the module docstring.
ENCODER_SMOKE = "smoke"

#: Written into every artifact the smoke encoder produces.
SMOKE_ENCODER_MARKER = "step-2.4-smoke-encoder-not-sentence-t5"

#: Embedding width of the audited baseline, and of the smoke stand-in.
SENTENCE_T5_DIM = 768


class EncoderError(RuntimeError):
    """Raised when the encoder cannot be constructed or the feature build cannot complete."""


@dataclass(frozen=True)
class TextBatch:
    """One batch of ``(item_ids, texts)``, in catalogue order."""

    item_ids: tuple[int, ...]
    texts: tuple[str, ...]


@dataclass
class FeatureBuildOutcome:
    """What the build observed, for the run report."""

    num_items: int = 0
    dim: int = 0
    empty_text_items: int = 0
    truncated_items: int = 0
    encoded_items: int = 0
    batches: int = 0
    seconds: float = 0.0
    peak_allocated_bytes: int | None = None
    encoder_id: str = ""
    encoder_revision: str = ""
    device: str = "cpu"
    non_finite_rows: int = 0
    replacements: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Reading the materialised text
# --------------------------------------------------------------------------- #


def iter_text_batches(path: str | Path, *, batch_size: int) -> Iterator[TextBatch]:
    """Stream ``products_text.jsonl`` in batches, refusing a malformed record.

    The reader requires exactly ``item_id`` and ``text``: a record carrying anything else (a
    target-shaped key, an identifier) is refused rather than ignored, so the feature stage
    cannot silently consume a handoff that grew an unexpected field.
    """
    if batch_size < 1:
        raise EncoderError(f"batch_size must be >= 1, got {batch_size}")
    source = Path(path)
    if not source.is_file():
        raise EncoderError(f"missing {source}")
    ids: list[int] = []
    texts: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EncoderError(f"{source.name} line {number} is not JSON: {error}") from error
            if not isinstance(record, dict) or set(record) != {"item_id", "text"}:
                raise EncoderError(
                    f"{source.name} line {number} must hold exactly 'item_id' and 'text', got "
                    f"{sorted(record) if isinstance(record, dict) else type(record).__name__}"
                )
            item_id = int(record["item_id"])
            text = record["text"]
            if item_id == 0:
                raise EncoderError(f"{source.name} line {number} carries item id 0 (PAD)")
            if not isinstance(text, str):
                raise EncoderError(f"{source.name} line {number} 'text' must be a string")
            ids.append(item_id)
            texts.append(text)
            if len(ids) >= batch_size:
                yield TextBatch(tuple(ids), tuple(texts))
                ids, texts = [], []
    if ids:
        yield TextBatch(tuple(ids), tuple(texts))


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #


def _smoke_encode(texts: Sequence[str], *, dim: int) -> np.ndarray:
    """Deterministic character-ngram hashing.  A plumbing stand-in, not a model.

    Nothing here is learned or semantic: it maps text to a fixed-width vector by hashing
    character trigrams, so two similar strings land near each other only incidentally.  Its
    purpose is that the rest of Step 2.4 can be exercised on a machine with no GPU and no
    network, and that the result is byte-reproducible.
    """
    rows = np.zeros((len(texts), dim), dtype=np.float32)
    for index, text in enumerate(texts):
        payload = (text or "").encode("utf-8")
        if not payload:
            continue
        for start in range(max(1, len(payload) - 2)):
            trigram = payload[start : start + 3]
            digest = hashlib.blake2b(trigram, digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            rows[index, bucket] += sign
        norm = float(np.linalg.norm(rows[index]))
        if norm > 0.0:
            rows[index] /= norm
    return rows


def resolve_encoder(
    name: str,
    *,
    device: str,
    batch_size: int,
    local_files_only: bool = False,
    max_chars: int | None = None,
    revision: str | None = None,
) -> tuple[Callable[[Sequence[str]], np.ndarray], dict[str, Any]]:
    """Return ``(encode, info)`` for an encoder name, optionally pinned to a revision.

    A repository *name* is mutable: the same name resolves to different weights over time, so a
    production artifact must record the immutable snapshot it actually used.  ``revision`` is
    therefore passed through to the loader, and the resolved commit is recorded separately from
    the requested identifier.  When no revision is supplied the loader's own resolution is used
    and reported honestly - the artifact never claims a pin it does not have.

    ``sentence-transformers`` is imported lazily, so the smoke encoder works in an environment
    where the ML stack is absent - which is exactly the situation the smoke gate is designed to
    be runnable in.
    """
    if name == ENCODER_SMOKE:
        def encode(texts: Sequence[str]) -> np.ndarray:
            return _smoke_encode(texts, dim=SENTENCE_T5_DIM)

        return encode, {
            "id": SMOKE_ENCODER_MARKER,
            "requested_id": ENCODER_SMOKE,
            "requested_revision": revision,
            "revision": None,
            "revision_resolved": False,
            "weights_pinned": False,
            "dim": SENTENCE_T5_DIM,
            "pooling": "character-trigram-hash",
            "normalize": True,
            "frozen": True,
            "is_model": False,
            "note": (
                "deterministic plumbing stand-in; NOT sentence-t5 and not a trained encoder. "
                "Any gate passed with it exercises the pipeline, not the representation."
            ),
        }

    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except Exception as error:  # pragma: no cover - depends on the backend venv
        raise EncoderError(
            f"encoder {name!r} needs sentence-transformers, which is not importable: {error}. "
            "Install it into the backend venv, or use --encoder smoke for a plumbing-only run."
        ) from error

    load_kwargs: dict[str, Any] = {
        "device": device,
        "local_files_only": bool(local_files_only),
    }
    if revision:
        load_kwargs["revision"] = revision
    try:
        # ``local_files_only`` must reach the loader: without it a missing cache silently becomes
        # a network fetch, which on an offline host hangs instead of failing.
        model = SentenceTransformer(name, **load_kwargs)
    except Exception as error:  # pragma: no cover - network / cache dependent
        raise EncoderError(
            f"could not load the encoder {name!r} at revision {revision!r} on device "
            f"{device!r}: {error}. Check that the weights are cached locally (the host may have "
            "no route to huggingface.co)."
        ) from error

    dimension = _encoder_dim(model)
    resolved = _encoder_revision(model, requested=name)
    if revision and not is_immutable_revision(resolved):
        # The loader did not report the commit we asked for.  Recording the requested value
        # verbatim would overstate the pin, so the resolved commit is taken from the cache when
        # it can be read, and otherwise reported as unresolved.
        resolved = _revision_from_hub_cache(name) or revision
    limit = int(max_chars or 0)

    def encode(texts: Sequence[str]) -> np.ndarray:
        payload = [text[:limit] if limit > 0 else text for text in texts]
        vectors = model.encode(
            payload,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    immutable = is_immutable_revision(resolved)
    return encode, {
        "id": f"{name}@{resolved}" if immutable else name,
        "requested_id": name,
        "requested_revision": revision,
        "revision": resolved if immutable else None,
        "revision_resolved": immutable,
        "weights_pinned": bool(revision) and immutable,
        "dim": dimension,
        "pooling": "model default",
        "normalize": False,
        "frozen": True,
        "is_model": True,
        "note": "frozen pretrained text encoder; weights are never updated",
    }


def _encoder_dim(model: Any) -> int:
    """Read the embedding width across sentence-transformers API generations.

    ``get_sentence_embedding_dimension`` is the documented call, but it was replaced by
    ``get_embedding_dimension`` in v5, and some builds carry the width only on the underlying
    transformer config.  Reading all three is what keeps this module working on both the pinned
    v3 line and the current release.
    """
    for attribute in ("get_sentence_embedding_dimension", "get_embedding_dimension"):
        reader = getattr(model, attribute, None)
        if callable(reader):
            value = reader()
            if isinstance(value, int) and value > 0:
                return int(value)
    for path in (("config", "hidden_size"), ("config", "d_model"),
                 ("config_dict", "hidden_size"), ("config_dict", "d_model")):
        node: Any = model
        for part in path:
            node = getattr(node, part, None) if not isinstance(node, dict) else node.get(part)
            if node is None:
                break
        if isinstance(node, int) and node > 0:
            return int(node)
    raise EncoderError(
        "could not determine the encoder's embedding width; refusing to guess, because a wrong "
        "width would silently reshape every downstream artifact"
    )


#: A HuggingFace snapshot identifier is a 40-character lowercase hex SHA-1.
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def is_immutable_revision(value: Any) -> bool:
    """True when ``value`` is a commit hash rather than a mutable branch or tag.

    ``"main"``, a tag, and the repository name itself are all *mutable* identifiers; only a
    snapshot SHA-1 pins the weights.  The distinction is what stops an artifact from claiming a
    reproducibility guarantee it does not have.
    """
    return isinstance(value, str) and bool(_REVISION_RE.match(value))


def _encoder_revision(model: Any, *, requested: str) -> str:
    """The loader's own resolved commit, when it exposes one.

    Newer ``sentence-transformers`` builds no longer set ``_commit_hash``, so this often returns
    the requested name; callers use :func:`is_immutable_revision` to tell the two apart and fall
    back to the Hub cache.
    """
    for attribute in ("_model_card_vars", "config"):
        node = getattr(model, attribute, None)
        if isinstance(node, dict):
            for key in ("__version__", "_commit_hash", "revision"):
                value = node.get(key)
                if isinstance(value, str) and value and is_immutable_revision(value):
                    return value
    commit = getattr(model, "_commit_hash", None)
    if isinstance(commit, str) and is_immutable_revision(commit):
        return commit
    return requested


def _revision_from_hub_cache(repo_id: str) -> str | None:
    """Read the snapshot SHA the local Hub cache holds for ``repo_id``.

    The cache stores it in ``refs/main`` and names the snapshot directory after it, so the pin
    can be recovered *after* a run that did not pass one - which is how the revision used by an
    earlier run is established without guessing.  Returns ``None`` when no cache entry exists.
    """
    import os

    roots = []
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value))
    home = os.environ.get("HF_HOME")
    if home:
        roots.append(Path(home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    folder = "models--" + repo_id.replace("/", "--")
    for root in roots:
        ref = root / folder / "refs" / "main"
        try:
            value = ref.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if is_immutable_revision(value):
            return value
    return None


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #


def build_item_features(
    *,
    products_text: str | Path,
    out_dir: str | Path,
    encoder: str = ENCODER_SENTENCE_T5,
    dim: int | None = None,
    batch_size: int = 256,
    device: str = "cpu",
    max_chars: int = 1000,
    expected_items: int | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[FeatureBuildOutcome, dict[str, Any]]:
    """Encode every item and write ``item_features.npy`` plus its metadata.

    The array holds **one row per real item and no PAD row**: row ``r`` belongs to the
    ``r + 1``-th entry of the catalogue handoff.  Every row is written, including the ones whose
    text is empty, so coverage stays ``1.000`` by construction and an item can never become
    unretrievable because its metadata was sparse.

    A non-finite vector is replaced with the batch mean rather than written through, and the
    substitution is counted: a NaN that reaches the quantizer would silently poison a codebook,
    so it is contained here and reported.
    """
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    source = Path(products_text)
    products_sha256 = _sha256_file(source)

    encode, info = resolve_encoder(
        encoder,
        device=device,
        batch_size=batch_size,
        local_files_only=local_files_only,
        max_chars=max_chars,
        revision=revision,
    )
    if revision and not info["revision_resolved"]:
        raise EncoderError(
            f"the requested encoder revision {revision!r} could not be confirmed; refusing to "
            "record a pin the run cannot prove. Check that the snapshot is cached, or pass "
            "--local-files-only to make a missing snapshot fail instead of resolving elsewhere."
        )
    resolved_dim = int(dim or info["dim"])
    if encoder != ENCODER_SMOKE and resolved_dim != int(info["dim"]):
        raise EncoderError(
            f"--dim {resolved_dim} disagrees with the encoder's width {info['dim']}; refusing, "
            "because the mismatch would reshape every downstream artifact"
        )

    # Count rows first so the memmap can be created at the right size without holding the
    # features in host memory.
    num_items = sum(len(batch.item_ids) for batch in iter_text_batches(source, batch_size=100_000))
    if expected_items is not None and num_items != expected_items:
        raise EncoderError(
            f"products_text holds {num_items} items but the catalogue handoff declares "
            f"{expected_items}; materials are inconsistent"
        )

    array_path = target_dir / "item_features.npy"
    # Written to a temporary path and renamed only on success.  An in-place memmap was the
    # earlier design and it was wrong: a run killed part-way through left a *complete-looking*
    # file whose unwritten rows were zeros, and a zero vector is a valid (if meaningless)
    # embedding, so nothing downstream noticed.  With the rename, an interrupted build leaves
    # no artifact at all.
    partial_path = target_dir / "item_features.npy.partial"
    partial_path.unlink(missing_ok=True)
    array = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float32, shape=(num_items, resolved_dim)
    )
    outcome = FeatureBuildOutcome(
        num_items=num_items,
        dim=resolved_dim,
        encoder_id=str(info["id"]),
        encoder_revision=str(info["revision"] or ""),
        device=device,
    )
    expected_row = 0
    started = time.perf_counter()
    completed = False
    if device != "cpu" and torch_available():
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    progress = log or (lambda message: None)
    try:
        for batch in iter_text_batches(source, batch_size=batch_size):
            item_ids = batch.item_ids
            if item_ids[0] != expected_row + 1:
                raise EncoderError(
                    f"products_text is not a contiguous 1..N span: expected item id "
                    f"{expected_row + 1}, got {item_ids[0]}"
                )
            ids = np.asarray(item_ids, dtype=np.int64)
            if not np.array_equal(ids, np.arange(expected_row + 1, expected_row + 1 + len(ids))):
                raise EncoderError("products_text item ids must ascend contiguously from 1")
            vectors = np.asarray(encode(list(batch.texts)), dtype=np.float32)
            if vectors.shape != (len(batch.texts), resolved_dim):
                raise EncoderError(
                    f"encoder returned shape {vectors.shape}, expected "
                    f"({len(batch.texts)}, {resolved_dim})"
                )
            outcome.empty_text_items += sum(1 for text in batch.texts if not text.strip())
            outcome.truncated_items += sum(1 for text in batch.texts if len(text) > max_chars)
            finite = np.isfinite(vectors).all(axis=1)
            if not finite.all():
                bad = int((~finite).sum())
                outcome.non_finite_rows += bad
                outcome.replacements += bad
                good = vectors[finite]
                replacement = good.mean(axis=0) if good.shape[0] else np.zeros(resolved_dim, np.float32)
                vectors = np.where(finite[:, None], vectors, replacement[None, :]).astype(np.float32)
            array[expected_row : expected_row + len(batch.texts)] = vectors
            expected_row += len(batch.texts)
            outcome.batches += 1
            if outcome.batches % 20 == 0 or expected_row == num_items:
                elapsed = time.perf_counter() - started
                rate = expected_row / elapsed if elapsed > 0 else 0.0
                progress(
                    f"encoded {expected_row}/{num_items} in {elapsed:.0f}s "
                    f"({rate:.1f} items/s)"
                )
    finally:
        # Closed here rather than left to interpreter teardown, which stalled on this host
        # after every row had already been written.
        array.flush()
        del array
    if expected_row != num_items:
        partial_path.unlink(missing_ok=True)
        raise EncoderError(
            f"encoded {expected_row} of {num_items} items; the partial file was discarded so an "
            "incomplete build cannot be mistaken for a complete one"
        )
    partial_path.replace(array_path)
    completed = True
    outcome.encoded_items = expected_row
    outcome.seconds = time.perf_counter() - started
    if device != "cpu" and torch_available():
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            outcome.peak_allocated_bytes = int(torch.cuda.max_memory_allocated())

    if not completed:
        raise EncoderError(f"encoded {expected_row} items but the file declares {num_items}")
    # A zero row is legitimate only when the item genuinely has no text, so the count is
    # reported rather than rejected.  A *contiguous trailing block* of them is not legitimate:
    # it is the signature of a truncated encode.  The rename above makes that unreachable, and
    # this check states the same rule a second time so the invariant is explicit.
    stored = np.load(array_path, mmap_mode="r", allow_pickle=False)
    norms = np.linalg.norm(np.asarray(stored), axis=1)
    zero_rows = int((norms == 0.0).sum())
    if zero_rows:
        positions = np.flatnonzero(norms == 0.0)
        contiguous = bool(np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)))
        if contiguous and positions[-1] == num_items - 1:
            raise EncoderError(
                f"the feature matrix ends with a contiguous block of {zero_rows} zero rows; that "
                "is a truncated encode, not a set of empty-text items"
            )
    del stored

    features_sha256 = _sha256_file(array_path)
    record: dict[str, Any] = {
        "format": "agentrecx.tiger.item_features.v3",
        "contract_version": CONTRACT_VERSION,
        "num_items": num_items,
        "dim": resolved_dim,
        "dtype": "float32",
        "order": "C",
        "endianness": "little",
        "encoder": info,
        "text_template": "composed by AgentRec-X in products_text.jsonl",
        "max_chars": int(max_chars),
        "empty_text_items": outcome.empty_text_items,
        "truncated_items": outcome.truncated_items,
        "nan_inf_scan": "clean" if outcome.non_finite_rows == 0 else "replaced",
        "non_finite_rows_replaced": outcome.non_finite_rows,
        "products_text_sha256": products_sha256,
        "sha256": features_sha256,
        "device": device,
        "seconds": round(outcome.seconds, 3),
        "batches": outcome.batches,
        "pad_row_present": False,
        "coverage": 1.0,
        "zero_norm_items": zero_rows,
        "partial_file_retained": partial_path.exists(),
    }
    _write_json(target_dir / "item_features.json", record)
    outcome.metadata = record
    return outcome, record


def torch_available() -> bool:
    """True when torch can be imported.  Guards the optional peak-VRAM accounting."""
    try:
        import torch  # noqa: F401, PLC0415

        return True
    except Exception:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )

