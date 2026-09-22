"""Backend-local tests for the Step-2.4 text-encoding stage.

The ``smoke`` encoder is deterministic and needs no network, so these tests exercise the whole
stage - streaming batches, the contiguous-id check, the atomic write, the corruption guards -
without any model weights.  The real encoder's arithmetic is checked by the smoke gate, which
runs on a GPU host.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tiger_public.contracts import CONTRACT_VERSION
from tiger_public.features import (
    ENCODER_SENTENCE_T5,
    SMOKE_ENCODER_MARKER,
    EncoderError,
    build_item_features,
    iter_text_batches,
    resolve_encoder,
)


def write_texts(path, texts: list[str]) -> None:
    path.write_text(
        "".join(
            json.dumps({"item_id": index + 1, "text": text}, sort_keys=True) + "\n"
            for index, text in enumerate(texts)
        ),
        encoding="utf-8",
    )


def test_iter_text_batches_is_contiguous_and_complete(tmp_path) -> None:
    path = tmp_path / "products_text.jsonl"
    write_texts(path, [f"item {index}" for index in range(7)])
    batches = list(iter_text_batches(path, batch_size=3))
    assert [len(batch.item_ids) for batch in batches] == [3, 3, 1]
    assert [item for batch in batches for item in batch.item_ids] == list(range(1, 8))


def test_iter_text_batches_refuses_a_pad_item_id(tmp_path) -> None:
    path = tmp_path / "products_text.jsonl"
    path.write_text(json.dumps({"item_id": 0, "text": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(EncoderError):
        list(iter_text_batches(path, batch_size=8))


def test_iter_text_batches_refuses_an_unexpected_field(tmp_path) -> None:
    """A record that grew a field is refused, not silently ignored."""
    path = tmp_path / "products_text.jsonl"
    path.write_text(
        json.dumps({"item_id": 1, "text": "x", "extra": 1}) + "\n", encoding="utf-8"
    )
    with pytest.raises(EncoderError):
        list(iter_text_batches(path, batch_size=8))


def test_smoke_encoder_is_deterministic_and_normalised() -> None:
    encode, info = resolve_encoder("smoke", device="cpu", batch_size=8)
    assert info["is_model"] is False
    assert info["id"] == SMOKE_ENCODER_MARKER
    first = encode(["a red bicycle", "a blue tent"])
    second = encode(["a red bicycle", "a blue tent"])
    assert first.shape == (2, 768)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    # The stand-in normalises, so each row has unit length.
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)


def test_build_writes_a_clean_float32_matrix_with_no_pad_row(tmp_path) -> None:
    path = tmp_path / "products_text.jsonl"
    write_texts(path, ["alpha", "", "gamma"])
    outcome, record = build_item_features(
        products_text=path, out_dir=tmp_path / "features", encoder="smoke", batch_size=2
    )
    array = np.load(tmp_path / "features" / "item_features.npy", mmap_mode="r")
    assert array.shape == (3, 768)
    assert array.dtype == np.float32
    assert np.isfinite(np.asarray(array)).all()
    assert record["pad_row_present"] is False
    assert record["coverage"] == 1.0
    assert record["contract_version"] == CONTRACT_VERSION
    # The empty-text item is still embedded, which is what keeps coverage total.
    assert record["empty_text_items"] == 1
    assert outcome.encoded_items == 3
    assert outcome.batches == 2


def test_build_records_the_source_digest(tmp_path) -> None:
    path = tmp_path / "products_text.jsonl"
    write_texts(path, ["alpha", "beta"])
    _outcome, record = build_item_features(
        products_text=path, out_dir=tmp_path / "features", encoder="smoke", batch_size=8
    )
    import hashlib

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["products_text_sha256"] == expected
    assert len(record["sha256"]) == 64


def test_build_is_reproducible(tmp_path) -> None:
    path = tmp_path / "products_text.jsonl"
    write_texts(path, ["alpha", "beta", "gamma", "delta"])
    first, record_a = build_item_features(
        products_text=path, out_dir=tmp_path / "a", encoder="smoke", batch_size=2
    )
    second, record_b = build_item_features(
        products_text=path, out_dir=tmp_path / "b", encoder="smoke", batch_size=2
    )
    assert record_a["sha256"] == record_b["sha256"]
    assert first.num_items == second.num_items


def test_build_leaves_no_partial_file(tmp_path) -> None:
    """An interrupted build must not leave an artifact that looks complete."""
    path = tmp_path / "products_text.jsonl"
    write_texts(path, ["alpha", "beta"])
    build_item_features(
        products_text=path, out_dir=tmp_path / "features", encoder="smoke", batch_size=8
    )
    assert not (tmp_path / "features" / "item_features.npy.partial").exists()
    record = json.loads((tmp_path / "features" / "item_features.json").read_text())
    assert record["partial_file_retained"] is False


def test_build_refuses_a_catalogue_size_mismatch(tmp_path) -> None:
    path = tmp_path / "products_text.jsonl"
    write_texts(path, ["alpha", "beta"])
    with pytest.raises(EncoderError):
        build_item_features(
            products_text=path,
            out_dir=tmp_path / "features",
            encoder="smoke",
            batch_size=8,
            expected_items=5,
        )


def test_build_refuses_a_non_contiguous_item_id_span(tmp_path) -> None:
    path = tmp_path / "products_text.jsonl"
    path.write_text(
        json.dumps({"item_id": 1, "text": "a"}) + "\n"
        + json.dumps({"item_id": 3, "text": "c"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EncoderError):
        build_item_features(
            products_text=path, out_dir=tmp_path / "features", encoder="smoke", batch_size=8
        )


def test_build_refuses_a_dimension_that_disagrees_with_the_encoder(tmp_path) -> None:
    path = tmp_path / "products_text.jsonl"
    write_texts(path, ["alpha"])
    with pytest.raises(EncoderError):
        build_item_features(
            products_text=path,
            out_dir=tmp_path / "features",
            encoder="smoke",
            dim=128,
            batch_size=8,
        )


def test_real_encoder_requires_huggingface_weights(tmp_path) -> None:
    """A missing encoder must fail loudly, never silently fall back to the stand-in."""
    path = tmp_path / "products_text.jsonl"
    write_texts(path, ["alpha"])
    if ENCODER_SENTENCE_T5 == "smoke":  # pragma: no cover - defensive
        pytest.skip("the baseline encoder id is unexpectedly the smoke stand-in")
    with pytest.raises(Exception) as error:
        resolve_encoder(
            "definitely-not-a-real-encoder-identifier",
            device="cpu",
            batch_size=1,
            local_files_only=True,
        )
    assert "encoder" in str(error.value).lower() or "could not load" in str(error.value).lower()
