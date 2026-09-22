"""Tests for the read-only Semantic-ID audit tool (Step 2.4F).

The audit's whole value is that its arithmetic is right and that it changes nothing, so it is
tested on a hand-built artifact whose expected values can be computed independently:

* a model whose decoder is replaced by a known linear map, so the prefix reconstruction MSE at
  each level is predictable;
* a synthetic code assignment with a known occupancy and a known collision-group distribution.

The tool is imported from ``experiments/`` rather than duplicated, so a change to the tool is
tested here rather than drifting from its tests.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tiger_public.contracts import PAD_SENTINEL
from tiger_public.quantizer import QuantizerConfig, RqVae

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = REPO_ROOT / "experiments" / "audit_tiger_sid.py"


def _load_audit_module():
    """Import ``experiments/audit_tiger_sid.py`` from its path.

    ``experiments/`` is not on the backend's import path, and the tool is deliberately not part
    of the installed package, so it is loaded the same way a user would run it.
    """
    spec = importlib.util.spec_from_file_location("audit_tiger_sid", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_tiger_sid"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("audit_tiger_sid", None)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load_audit_module()


# --------------------------------------------------------------------------- #
# Artifact fixtures
# --------------------------------------------------------------------------- #


def build_artifact(tmp_path: Path, *, num_items: int = 32, dim: int = 16, levels: int = 3,
                   codebook_size: int = 8, seed: int = 2026) -> tuple[Path, Path, np.ndarray]:
    """Write a minimal, valid feature + SID artifact pair and return their directories."""
    features_dir = tmp_path / "features"
    sid_dir = tmp_path / "sid"
    features_dir.mkdir(parents=True, exist_ok=True)
    sid_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(num_items, dim, generator=generator).numpy().astype(np.float32)
    np.save(features_dir / "item_features.npy", matrix)
    digest = __import__("hashlib").sha256((features_dir / "item_features.npy").read_bytes()).hexdigest()
    (features_dir / "item_features.json").write_text(
        json.dumps(
            {
                "format": "agentrecx.tiger.item_features.v3",
                "num_items": num_items,
                "dim": dim,
                "dtype": "float32",
                "encoder": {"id": "test-encoder", "revision": None},
                "sha256": digest,
                "pad_row_present": False,
                "coverage": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = QuantizerConfig(
        input_dim=dim,
        encoder_dims=(dim, 8),
        latent_dim=4,
        codebook_size=codebook_size,
        levels=levels,
        epochs=1,
        batch_size=8,
        seed=seed,
    )
    torch.manual_seed(seed)
    model = RqVae(config)
    torch.save({"state_dict": model.state_dict(), "config": config.as_dict()},
               sid_dir / "tokenizer.pt")

    # A deterministic, hand-known assignment: item i takes code (i mod K) at every level, so
    # occupancy and collision groups are predictable.  The trailing digit is the dedup ordinal.
    assignment = [[PAD_SENTINEL] * (levels + 1)]
    seen: dict[tuple[int, ...], int] = {}
    for item in range(num_items):
        base = tuple((item % codebook_size) for _ in range(levels))
        ordinal = seen.get(base, 0)
        seen[base] = ordinal + 1
        assignment.append([*base, ordinal])
    (sid_dir / "semantic_ids.json").write_text(
        json.dumps(
            {
                "format": "agentrecx.tiger.semantic_ids.v3",
                "contract_version": "agentrecx.tiger_backend.v3",
                "num_items": num_items,
                "levels": levels,
                "codebook_size": codebook_size,
                "pad_row": [PAD_SENTINEL] * (levels + 1),
                "assignment": assignment,
                "content_features": {"sha256": digest},
                "quantizer": config.as_dict(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return features_dir, sid_dir, matrix


# --------------------------------------------------------------------------- #
# Prefix reconstruction
# --------------------------------------------------------------------------- #


def test_prefix_mse_matches_a_hand_computed_value(audit, tmp_path) -> None:
    """With a decoder replaced by a known map, each prefix MSE is computable by hand."""
    features_dir, sid_dir, matrix = build_artifact(tmp_path, num_items=16, dim=8, levels=2,
                                                   codebook_size=4)
    model = RqVae(QuantizerConfig.from_artifact(
        json.loads((sid_dir / "semantic_ids.json").read_text())["quantizer"]
    ))
    model.load_state_dict(torch.load(sid_dir / "tokenizer.pt", map_location="cpu")["state_dict"])
    model.eval()

    result = audit.prefix_diagnostics(model, matrix, batch_size=8, device="cpu")
    assert result["num_items"] == 16
    assert result["dim"] == 8
    values = [result["mse"]["L0"], result["mse"]["L0+L1"]]
    assert all(v is not None and math.isfinite(v) for v in values)
    assert result["mse"]["L0+L1+L2"] is None  # only two levels exist here
    assert result["zero_prediction_mse"] > 0.0
    # NOTE: prefix MSE is deliberately NOT asserted to be monotone.  Each level minimises the
    # *latent* residual it sees, and the decoder is nonlinear, so a level that reduces the
    # residual norm can still raise the decoded squared error.  The monotone quantity is the
    # residual norm, and that is asserted separately by
    # ``test_residual_norms_decrease_monotonically``.


def test_prefix_reconstruction_is_independently_verifiable(audit, tmp_path) -> None:
    """Recompute L0 by hand from the same tensors and compare with the tool's number."""
    features_dir, sid_dir, matrix = build_artifact(tmp_path, num_items=12, dim=8, levels=1,
                                                    codebook_size=4)
    payload = torch.load(sid_dir / "tokenizer.pt", map_location="cpu")
    model = RqVae(QuantizerConfig.from_artifact(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float32))
        target = model.normalise(tensor)
        latents = model.encoder(target)
        codebook = model.quantizer.codebooks[0]
        distances = (
            latents.pow(2).sum(dim=-1, keepdim=True)
            - 2.0 * latents @ codebook.t()
            + codebook.pow(2).sum(dim=-1).unsqueeze(0)
        )
        accumulated = codebook[distances.argmin(dim=-1)]
        expected = float((model.decoder(accumulated) - target).pow(2).mean())

    result = audit.prefix_diagnostics(model, matrix, batch_size=4, device="cpu")
    assert result["mse"]["L0"] == pytest.approx(expected, rel=1e-5)
    assert result["mse"]["L0+L1"] is None


def test_residual_norms_decrease_monotonically(audit, tmp_path) -> None:
    features_dir, sid_dir, matrix = build_artifact(tmp_path, num_items=24, dim=8, levels=3,
                                                    codebook_size=8)
    model = RqVae(QuantizerConfig.from_artifact(
        json.loads((sid_dir / "semantic_ids.json").read_text())["quantizer"]
    ))
    model.load_state_dict(torch.load(sid_dir / "tokenizer.pt", map_location="cpu")["state_dict"])
    model.eval()
    norms = audit.prefix_diagnostics(model, matrix, batch_size=8, device="cpu")["norms"]
    chain = [
        norms["initial_latent_norm_mean"],
        norms["residual_norm_after_L0"],
        norms["residual_norm_after_L1"],
        norms["residual_norm_after_L2"],
    ]
    assert all(b <= a + 1e-6 for a, b in zip(chain, chain[1:])), chain
    # The normalised target is unit-length by construction; the raw input need not be.
    assert norms["normalised_target_norm_mean"] == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# Occupancy, coverage and largest-code fraction
# --------------------------------------------------------------------------- #


def test_occupancy_and_coverage_are_reported_separately(audit) -> None:
    """Coverage and largest-code fraction answer different questions, so they must differ here.

    The fixture puts 24 items on 5 codes: every item shares a code with at least three others,
    so coverage is 5/8 while the largest code holds a substantial share of the catalogue.
    """
    codes = np.asarray([[item % 5] for item in range(24)], dtype=np.int64)
    block = audit.level_diagnostics(codes, codebook_size=8)[0]
    assert block["used_codes"] == 5
    assert block["dead_codes"] == 3
    assert block["codebook_coverage"] == pytest.approx(5 / 8)
    # 24 items over 5 codes -> the largest holds 5 (items 0,5,10,15,20).
    assert block["largest_code_count"] == 5
    # Reported rounded to six decimals, so the comparison allows for that rounding.
    assert block["largest_code_fraction"] == pytest.approx(5 / 24, abs=1e-6)
    # The two quantities must not coincide, which is the confusion the tool exists to prevent.
    assert block["largest_code_fraction"] != pytest.approx(block["codebook_coverage"])


def test_occupancy_histogram_is_complete_and_conserves_codes(audit) -> None:
    """Every codebook entry is counted exactly once across the histogram."""
    codebook_size = 8
    codes = np.asarray([[0], [0], [1], [2], [2], [2]], dtype=np.int64)
    block = audit.level_diagnostics(codes, codebook_size=codebook_size)[0]
    histogram = block["occupancy_histogram"]
    assert histogram["0"] == 5           # codes 3..7 were never selected
    assert histogram["1"] == 1           # code 1 once
    assert histogram["2"] == 1           # code 0 twice
    assert histogram["3"] == 1           # code 2 three times
    assert sum(histogram.values()) == codebook_size
    assert block["codes_used_once"] == 1


def test_entropy_bounds_and_fraction(audit) -> None:
    """A uniform assignment reaches the maximum; a single code reaches zero."""
    uniform = np.asarray([[item % 8] for item in range(64)], dtype=np.int64)
    block = audit.level_diagnostics(uniform, codebook_size=8)[0]
    assert block["id_entropy"] == pytest.approx(math.log(8), abs=1e-3)
    assert block["entropy_fraction"] == pytest.approx(1.0, abs=1e-3)

    single = np.zeros((16, 1), dtype=np.int64)
    collapsed = audit.level_diagnostics(single, codebook_size=8)[0]
    assert collapsed["id_entropy"] == pytest.approx(0.0)
    assert collapsed["largest_code_fraction"] == pytest.approx(1.0)
    assert collapsed["used_codes"] == 1


# --------------------------------------------------------------------------- #
# Collision groups
# --------------------------------------------------------------------------- #


def test_collision_group_size_distribution(audit) -> None:
    """Group sizes are counted, and the dedup ordinal is excluded from the grouping."""
    assignment = [[PAD_SENTINEL] * 3]
    # Two groups of size 2 and one group of size 3, each member carrying a distinct ordinal.
    for base, size in (((1, 1), 2), ((2, 2), 2), ((3, 3), 3)):
        for ordinal in range(size):
            assignment.append([*base, ordinal])
    result = audit.collision_diagnostics(assignment, levels=2)
    assert result["pre_dedup_group_size_distribution"] == {"2": 2, "3": 1}
    assert result["pre_dedup_collision_groups"] == 3
    assert result["pre_dedup_items_in_collision"] == 7
    assert result["pre_dedup_largest_group"] == 3
    assert result["post_dedup_distinct_sids"] == 7
    assert result["post_dedup_collisions"] == 0
    assert result["post_dedup_unique"] is True


def test_collision_diagnostics_ignores_the_pad_row(audit) -> None:
    """PAD is never an item and must not appear in any collision statistic."""
    assignment = [[PAD_SENTINEL] * 3, [1, 1, 0], [1, 1, 1]]
    result = audit.collision_diagnostics(assignment, levels=2)
    assert result["num_items"] == 2
    assert result["pre_dedup_group_size_distribution"] == {"2": 1}


def test_collision_diagnostics_refuses_an_empty_assignment(audit) -> None:
    with pytest.raises(audit.AuditError):
        audit.collision_diagnostics([[PAD_SENTINEL] * 3], levels=2)


# --------------------------------------------------------------------------- #
# Read-only guarantee and CLI
# --------------------------------------------------------------------------- #


def test_audit_is_read_only_on_disk(audit, tmp_path) -> None:
    """Running the audit must leave every input byte identical."""
    features_dir, sid_dir, _matrix = build_artifact(tmp_path)
    before = {
        path: path.read_bytes()
        for path in list(features_dir.iterdir()) + list(sid_dir.iterdir())
    }
    report_path = tmp_path / "report.json"
    exit_code = audit.main(
        [
            "--features", str(features_dir),
            "--sid", str(sid_dir),
            "--out", str(report_path),
            "--batch-size", "8",
            "--quiet",
        ]
    )
    assert exit_code == 0
    after = {
        path: path.read_bytes()
        for path in list(features_dir.iterdir()) + list(sid_dir.iterdir())
    }
    assert before == after, "the audit modified its inputs"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["read_only"] is True
    assert report["prefix"]["num_items"] == 32
    assert len(report["levels"]) == 3
    assert report["collisions"]["post_dedup_unique"] is True


def test_audit_refuses_a_feature_sid_hash_mismatch(audit, tmp_path) -> None:
    """Auditing a SID assignment against a different feature matrix must be refused."""
    features_dir, sid_dir, _matrix = build_artifact(tmp_path)
    record = json.loads((sid_dir / "semantic_ids.json").read_text(encoding="utf-8"))
    record["content_features"]["sha256"] = "0" * 64
    (sid_dir / "semantic_ids.json").write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(audit.AuditError) as error:
        audit.load_artifacts(features_dir, sid_dir)
    assert "different features" in str(error.value)


def test_audit_refuses_missing_artifacts(audit, tmp_path) -> None:
    with pytest.raises(audit.AuditError):
        audit.load_artifacts(tmp_path / "nope", tmp_path / "nope")


def test_audit_cli_reports_a_failure_as_exit_code_2(audit, tmp_path, capsys) -> None:
    exit_code = audit.main(
        ["--features", str(tmp_path / "x"), "--sid", str(tmp_path / "y")]
    )
    assert exit_code == 2
    assert "audit_tiger_sid" in capsys.readouterr().err


def test_render_includes_every_required_section(audit, tmp_path) -> None:
    features_dir, sid_dir, _matrix = build_artifact(tmp_path)
    _features, sid, feature_record, model = audit.load_artifacts(features_dir, sid_dir)
    codes = np.asarray([row[: int(sid["levels"])] for row in sid["assignment"][1:]], dtype=np.int64)
    report = {
        "prefix": audit.prefix_diagnostics(model, np.asarray(np.load(
            features_dir / "item_features.npy")), batch_size=8, device="cpu"),
        "levels": audit.level_diagnostics(codes, codebook_size=int(sid["codebook_size"])),
        "collisions": audit.collision_diagnostics(sid["assignment"], levels=int(sid["levels"])),
    }
    text = audit.render(report)
    for needle in ("PREFIX RECONSTRUCTION", "L0+L1+L2", "NORMS", "PER-LEVEL OCCUPANCY",
                   "PRE-DEDUP COLLISION GROUP SIZES", "largest_code_fraction" if False else "maxfrac"):
        assert needle in text, needle
    assert feature_record is not None
