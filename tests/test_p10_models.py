"""P10 model/data invariants: LightGCN graph provenance and BGE feature alignment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.p10_train_lightgcn import build_graph
from recommendation.datasets.bge_features import BGEFeatureError, build_bge_item_matrix
from recommendation.models.bge_sasrec import BGEEnhancedSASRec
from recommendation.models.lightgcn import LightGCN, LightGCNConfig
from recommendation.models.sasrec import SASRec, SASRecConfig


def test_lightgcn_graph_uses_train_history_only_and_offsets_items() -> None:
    cases = [
        SimpleNamespace(user_int_id=1, train_history=(1, 2)),
        SimpleNamespace(user_int_id=2, train_history=(2, 3)),
    ]
    adjacency, edge_index, num_users = build_graph(cases, num_items=3)
    assert num_users == 2
    assert tuple(adjacency.shape) == (5, 5)
    assert edge_index.tolist() == [[0, 1], [0, 2], [1, 2], [1, 3]]
    # The validation targets are not part of the graph edges.
    assert 4 not in edge_index[:, 1].tolist()


def test_lightgcn_propagates_and_has_pad_zero() -> None:
    config = LightGCNConfig(num_users=3, num_items=4, embedding_size=8, num_layers=2)
    model = LightGCN(config)
    indices = torch.tensor([[0, 3, 1, 4, 2, 3], [3, 0, 4, 1, 3, 2]])
    values = torch.ones(indices.shape[1])
    adjacency = torch.sparse_coo_tensor(indices, values, (7, 7)).coalesce()
    users, items = model.propagate(adjacency)
    assert users.shape == (3, 8)
    assert items.shape == (5, 8)
    assert torch.count_nonzero(items[0]) == 0
    assert torch.isfinite(users).all()


def _write_bge_fixture(tmp_path: Path) -> tuple[Path, Path]:
    vectors = np.zeros((3, 768), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    vectors[2, 2] = 1.0
    index = tmp_path / "index.npz"
    np.savez(
        index,
        format=np.asarray(["agentrecx.dense_catalog.v3"]),
        metadata=np.asarray(["{}"]),
        identities=np.asarray(["A", "B", "C"]),
        vectors=vectors,
        titles=np.asarray(["a", "b", "c"]),
    )
    mapping = tmp_path / "mappings.json"
    mapping.write_text(json.dumps({"item2id": {"A": 1, "B": 2, "C": 3}}), encoding="utf-8")
    return index, mapping


def test_bge_matrix_aligns_by_item_id_and_rejects_missing(tmp_path: Path) -> None:
    index, mapping = _write_bge_fixture(tmp_path)
    matrix = build_bge_item_matrix(index, mapping, num_items=3)
    assert matrix.shape == (4, 768)
    assert np.array_equal(matrix[0], np.zeros(768))
    assert matrix[1, 0] == 1.0
    assert matrix[3, 2] == 1.0

    bad_mapping = tmp_path / "bad.json"
    bad_mapping.write_text(json.dumps({"item2id": {"A": 1, "B": 2, "C": 3, "D": 4}}), encoding="utf-8")
    with pytest.raises(BGEFeatureError):
        build_bge_item_matrix(index, bad_mapping, num_items=4)


def test_bge_enhanced_sasrec_uses_same_representation_everywhere() -> None:
    config = SASRecConfig(
        num_items=6, max_seq_len=5, hidden_size=8, num_blocks=1, num_heads=2, dropout=0.0
    )
    bge = torch.zeros((7, 768), dtype=torch.float32)
    bge[1:] = torch.randn(6, 768) * 0.05
    base = SASRec(config)
    enhanced = BGEEnhancedSASRec(config, bge)
    ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    assert enhanced.item_representations().shape == (7, 8)
    assert enhanced.full_catalog_scores(ids).shape == (1, 7)
    assert base.full_catalog_scores(ids).shape == (1, 7)
    assert torch.isfinite(enhanced.full_catalog_scores(ids)).all()
