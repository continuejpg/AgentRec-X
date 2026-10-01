"""Offline tests for the B0 artifact recoverability infrastructure.

Scope: the canonical manifest's ``recoverability`` section, the fetch/verify tooling in
``scripts/fetch_artifacts.py`` and the human-readable view in ``docs/ARTIFACT_STORAGE.md``.

Everything here is offline and fast.  No test downloads anything, and two tests actively prove
that the offline code paths cannot reach the network: they replace the single transport entry
point with something that fails, then exercise ``list`` and ``verify``.

What is deliberately **not** tested: whether an artifact is fetchable today.  Every entry is
``PENDING``, and a test that asserted otherwise would be asserting that an upload has happened.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from recommendation import local_demo

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "config" / "demo_runtime_artifacts.json"
STORAGE_DOC = REPO_ROOT / "docs" / "ARTIFACT_STORAGE.md"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_artifacts.py"

#: The digests that were established before B0 and must not drift.
KNOWN_SHA256 = {
    "two_tower_checkpoint": "0e56be01f872ed0cc46eca69ef54770b1ea7dbd3e4ef92aa6eb290b605d8b1ab",
    "mappings": "dca7815a2482ab200f2cf1e3dd62eec9035bf6c5386e76cce920325360f01912",
}

KNOWN_SIZES = {
    "two_tower_checkpoint": 162221222,
    "mappings": 39670810,
}


def _load_module() -> Any:
    """Load ``scripts/fetch_artifacts.py`` by path (``scripts`` is not a package).

    The module is registered in :data:`sys.modules` before execution because the script uses
    ``from __future__ import annotations``: on Python 3.10 the dataclass machinery resolves
    those string annotations through ``sys.modules``, and an unregistered module raises there.
    """
    spec = importlib.util.spec_from_file_location("agentrecx_fetch_artifacts", FETCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recovery() -> Any:
    return _load_module()


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(recovery: Any, manifest: dict[str, Any]) -> tuple[Any, ...]:
    return recovery.load_recovery_entries(manifest)


@pytest.fixture()
def pending_entries(recovery: Any, entries: tuple[Any, ...]) -> tuple[Any, ...]:
    """The same artifacts forced back to PENDING.

    The real entries are now UPLOADED_UNVERIFIED with a recorded remote path, so a test that wants
    the "nothing is published yet" path must build it explicitly — otherwise it would reach the
    network for real, which the offline suite must never do.
    """
    import dataclasses

    return tuple(
        dataclasses.replace(entry, durable_status=recovery.STATUS_PENDING, remote_path=None)
        for entry in entries
    )


def _location(recovery: Any, repo_type: str = "dataset", **overrides: Any) -> Any:
    """Build a DurableLocation for URI-shape tests."""
    fields = {
        "endpoint": "https://huggingface.co",
        "repo_type": repo_type,
        "repo_id": "owner/name",
        "revision": "main",
        "remote_path": "artifacts/file.bin",
    }
    fields.update(overrides)
    return recovery.DurableLocation(**fields)


# --------------------------------------------------------------------------- #
# 1. the manifest stays the launcher's manifest
# --------------------------------------------------------------------------- #


def test_canonical_manifest_is_still_a_valid_demo_runtime_manifest() -> None:
    """Adding a recoverability section must not disturb the demo launcher's contract.

    ``recommendation.local_demo`` rejects a manifest that adds or drops an artifact role, so
    this test is the guard against turning a recoverability edit into a launcher breakage.
    """
    specs = local_demo.load_artifact_specs(MANIFEST)
    assert {spec.role for spec in specs} == set(local_demo.EXPECTED_ARTIFACT_ROLES)


def test_recoverability_section_declares_the_b0_scope(entries: tuple[Any, ...]) -> None:
    """The registered scope, which grew as Serving V1's dependencies were identified.

    B0 registered the Two-Tower pair plus mappings; B0-c added the categorical metadata (which
    carries a scope_note); P2-B added the SASRec checkpoint, P2-D the catalogue and P2-E the demo
    sequences artifact.
    """
    assert {entry.role for entry in entries} == {
        "two_tower_checkpoint",
        "two_tower_metadata_categories",
        "mappings",
        "sasrec_checkpoint",
        "catalogue",
        "sequences",
    }


def test_recoverability_agrees_with_the_runtime_artifacts(manifest: dict[str, Any]) -> None:
    """Where a path appears in both lists, identity must not drift between them."""
    runtime = {entry["path"]: entry for entry in manifest["artifacts"]}
    for entry in manifest["recoverability"]["artifacts"]:
        other = runtime.get(entry["path"])
        if other is None:
            continue
        assert entry["size_bytes"] == other["size_bytes"], entry["path"]
        assert entry["sha256"] == other["sha256"], entry["path"]


# --------------------------------------------------------------------------- #
# 2. schema and known values
# --------------------------------------------------------------------------- #


def test_every_entry_carries_the_required_fields(
    recovery: Any, entries: tuple[Any, ...]
) -> None:
    for entry in entries:
        assert entry.role and entry.path and entry.description
        assert entry.size_bytes > 0
        assert len(entry.sha256) == 64 and entry.sha256 == entry.sha256.lower()
        assert entry.required_for, entry.role
        assert entry.durable_status in recovery.ALLOWED_STATUS


def test_recorded_digests_and_sizes_are_the_known_values(entries: tuple[Any, ...]) -> None:
    """The B0 scope pins two identities; silently editing either one must fail here."""
    by_role = {entry.role: entry for entry in entries}
    for role, digest in KNOWN_SHA256.items():
        assert by_role[role].sha256 == digest, role
        assert by_role[role].size_bytes == KNOWN_SIZES[role], role


def test_only_pending_entries_lack_a_location(recovery: Any, entries: tuple[Any, ...]) -> None:
    """A location exists exactly when the status says so; PENDING is the only addressless state."""
    for entry in entries:
        has_location = bool(entry.repo_id and entry.revision and entry.remote_path)
        assert has_location == (entry.durable_status in recovery.FETCHABLE_STATUS), entry.role


def test_manifest_never_contains_a_credential_value(manifest: dict[str, Any]) -> None:
    text = json.dumps(manifest).lower()
    assert "hf_" not in text
    assert "authorization" not in text
    assert "bearer" not in text


# --------------------------------------------------------------------------- #
# 3. selection and configuration parsing
# --------------------------------------------------------------------------- #


def test_select_entries_by_role_and_by_consumer(recovery: Any, entries: tuple[Any, ...]) -> None:
    assert [e.role for e in recovery.select_entries(entries, ["mappings"])] == ["mappings"]
    smoke = recovery.select_entries(entries, (), consumer=recovery.SMOKE_CONSUMER)
    assert {e.role for e in smoke} == {
        "two_tower_checkpoint",
        "mappings",
        "two_tower_metadata_categories",
    }
    assert len(recovery.select_entries(entries, ())) == len(entries)


def test_select_entries_rejects_an_unknown_role(recovery: Any, entries: tuple[Any, ...]) -> None:
    with pytest.raises(recovery.RecoveryError) as excinfo:
        recovery.select_entries(entries, ["not_a_role"])
    assert "unknown artifact role" in str(excinfo.value)


def test_env_file_parsing_is_minimal_and_non_destructive(
    recovery: Any, tmp_path: Path
) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# comment\n\nHF_REPO_ID=owner/agentrecx-artifacts\nHF_TOKEN='quoted'\n"
        "HF_REVISION=\"v1\"\nHF_REPO_ID=ignored-second-value\n",
        encoding="utf-8",
    )
    values = recovery.read_env_file(path)
    assert values["HF_REPO_ID"] == "owner/agentrecx-artifacts"
    assert values["HF_TOKEN"] == "quoted"
    assert values["HF_REVISION"] == "v1"
    assert recovery.read_env_file(tmp_path / "absent") == {}


def test_resolve_durable_prefers_the_environment_and_defaults_the_revision(
    recovery: Any, manifest: dict[str, Any], entries: tuple[Any, ...]
) -> None:
    ready = recovery.RecoveryEntry(
        **{
            **entries[0].__dict__,
            "durable_status": "READY",
            "remote_path": "artifacts/checkpoint.pt",
            "revision": None,
        }
    )
    location = recovery.resolve_durable(
        manifest, ready, {"HF_REPO_ID": "owner/repo", "HF_TOKEN": "secret-value"}
    )
    assert location.repo_id == "owner/repo"
    assert location.revision == "main"
    # The real manifest freezes repo_type = dataset, so the derived URI carries /datasets/.
    assert location.uri == (
        "https://huggingface.co/datasets/owner/repo/resolve/main/artifacts/checkpoint.pt"
    )
    assert "secret-value" not in location.uri


def test_hf_revision_overrides_the_manifest(
    recovery: Any, manifest: dict[str, Any], entries: tuple[Any, ...]
) -> None:
    ready = recovery.RecoveryEntry(
        **{
            **entries[0].__dict__,
            "durable_status": "READY",
            "remote_path": "a/b.pt",
            "revision": "pinned-sha",
        }
    )
    location = recovery.resolve_durable(manifest, ready, {"HF_REPO_ID": "o/r", "HF_REVISION": "v2"})
    assert location.revision == "v2"
    assert "/resolve/v2/" in location.uri


# --------------------------------------------------------------------------- #
# 4. blocked states fail clearly, never silently
# --------------------------------------------------------------------------- #


def test_pending_entry_is_refused_with_the_reason(
    recovery: Any, manifest: dict[str, Any], pending_entries: tuple[Any, ...]
) -> None:
    with pytest.raises(recovery.BlockedError) as excinfo:
        recovery.resolve_durable(manifest, pending_entries[0], {})
    message = str(excinfo.value)
    assert "PENDING" in message
    assert pending_entries[0].role in message
    assert "will not fall back to a local copy" in message


def test_missing_repository_is_refused(
    recovery: Any, manifest: dict[str, Any], entries: tuple[Any, ...]
) -> None:
    ready = recovery.RecoveryEntry(
        **{
            **entries[0].__dict__,
            "durable_status": "READY",
            "remote_path": "a/b.pt",
            "repo_id": None,
        }
    )
    with pytest.raises(recovery.BlockedError) as excinfo:
        recovery.resolve_durable(manifest, ready, {})
    assert recovery.ENV_REPO_ID in str(excinfo.value)


def test_missing_remote_path_is_refused(
    recovery: Any, manifest: dict[str, Any], entries: tuple[Any, ...]
) -> None:
    ready = recovery.RecoveryEntry(
        **{**entries[0].__dict__, "durable_status": "READY", "remote_path": None}
    )
    with pytest.raises(recovery.BlockedError) as excinfo:
        recovery.resolve_durable(manifest, ready, {"HF_REPO_ID": "o/r"})
    assert "remote_path" in str(excinfo.value)


def test_fetch_requires_an_explicit_selection(recovery: Any) -> None:
    """A bare `fetch` must not start a ~200 MB download."""
    assert recovery.main(["fetch"]) == recovery.EXIT_BLOCKED


# --------------------------------------------------------------------------- #
# 5. verification fails closed
# --------------------------------------------------------------------------- #


def test_digest_mismatch_fails_verification(recovery: Any, tmp_path: Path) -> None:
    entry = recovery.RecoveryEntry(
        role="synthetic",
        path="synthetic.bin",
        size_bytes=4,
        sha256="0" * 64,
        description="",
        required_for=("two_tower_serving_smoke",),
        durable_status="PENDING",
        repo_id=None,
        revision=None,
        remote_path=None,
    )
    target = tmp_path / "synthetic.bin"
    target.write_bytes(b"nope")
    with pytest.raises(recovery.VerificationError) as excinfo:
        recovery.verify_file(target, entry)
    assert "sha256 mismatch" in str(excinfo.value)


def test_size_mismatch_fails_before_hashing(recovery: Any, tmp_path: Path) -> None:
    entry = recovery.RecoveryEntry(
        role="synthetic",
        path="synthetic.bin",
        size_bytes=99,
        sha256="0" * 64,
        description="",
        required_for=("two_tower_serving_smoke",),
        durable_status="PENDING",
        repo_id=None,
        revision=None,
        remote_path=None,
    )
    target = tmp_path / "synthetic.bin"
    target.write_bytes(b"nope")
    with pytest.raises(recovery.VerificationError) as excinfo:
        recovery.verify_file(target, entry)
    assert "size mismatch" in str(excinfo.value)


def test_a_bad_download_is_deleted_and_never_installed(
    recovery: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The atomic-install rule: a wrong payload leaves nothing behind but a failure."""
    entry = recovery.RecoveryEntry(
        role="synthetic",
        path="artifacts/synthetic.bin",
        size_bytes=4,
        sha256="0" * 64,
        description="",
        required_for=("two_tower_serving_smoke",),
        durable_status="READY",
        repo_id="o/r",
        revision="main",
        remote_path="artifacts/synthetic.bin",
    )

    def fake_download(uri: str, destination: Path, token: str | None = None) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(b"wrong-bytes")
        return partial

    monkeypatch.setattr(recovery, "download", fake_download)
    location = _location(recovery, repo_id="o/r", remote_path="a/b")
    with pytest.raises(recovery.VerificationError):
        recovery.fetch_entry(entry, location, tmp_path, None)
    assert not (tmp_path / "artifacts" / "synthetic.bin").exists()
    assert not (tmp_path / "artifacts" / "synthetic.bin.part").exists()


def test_fetch_never_short_circuits_on_an_existing_local_copy(
    recovery: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A correct local file must not be used as a substitute for a durable source."""
    payload = b"good"
    import hashlib

    entry = recovery.RecoveryEntry(
        role="synthetic",
        path="artifacts/synthetic.bin",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        description="",
        required_for=("two_tower_serving_smoke",),
        durable_status="READY",
        repo_id="o/r",
        revision="main",
        remote_path="artifacts/synthetic.bin",
    )
    target = tmp_path / "artifacts" / "synthetic.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)  # already correct on disk

    calls: list[str] = []

    def fake_download(uri: str, destination: Path, token: str | None = None) -> Path:
        calls.append(uri)
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(payload)
        return partial

    monkeypatch.setattr(recovery, "download", fake_download)
    location = _location(recovery, repo_id="o/r", remote_path="a/b")
    recovery.fetch_entry(entry, location, tmp_path, None)
    assert calls == [location.uri], "fetch must download even when a valid copy is present"


# --------------------------------------------------------------------------- #
# 6. smoke contract
# --------------------------------------------------------------------------- #


def test_smoke_refuses_when_the_artifacts_already_exist(
    recovery: Any, manifest: dict[str, Any], entries: tuple[Any, ...]
) -> None:
    """Restoring from the development machine's own disk is not recoverability evidence."""
    with pytest.raises(recovery.BlockedError) as excinfo:
        recovery.run_smoke(entries, manifest, REPO_ROOT)
    message = str(excinfo.value)
    assert "ABSENT" in message
    assert "not recoverability evidence" in message


def test_smoke_in_a_clean_workspace_is_blocked_by_pending_not_by_a_local_copy(
    recovery: Any, manifest: dict[str, Any], pending_entries: tuple[Any, ...], tmp_path: Path
) -> None:
    """With nothing on disk and nothing published, the run stops at the durable location."""
    with pytest.raises(recovery.BlockedError) as excinfo:
        recovery.run_smoke(pending_entries, manifest, tmp_path)
    assert "PENDING" in str(excinfo.value)


def test_smoke_is_constructed_from_the_manifest_consumer(
    recovery: Any, entries: tuple[Any, ...]
) -> None:
    selected = recovery.select_entries(entries, (), consumer=recovery.SMOKE_CONSUMER)
    assert [e.role for e in selected][0] in {"two_tower_checkpoint", "mappings"}


# --------------------------------------------------------------------------- #
# 7. the offline paths cannot reach the network
# --------------------------------------------------------------------------- #


def test_offline_subcommands_never_open_a_connection(
    recovery: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the offline path attempted a network call")

    monkeypatch.setattr(recovery, "_open", explode)
    monkeypatch.setattr(recovery.urllib.request, "build_opener", explode)

    assert recovery.main(["list"]) == recovery.EXIT_OK
    assert recovery.main(["verify", "--repo-root", str(tmp_path)]) == recovery.EXIT_VERIFY
    capsys.readouterr()


def test_list_shows_the_derived_dataset_uri_and_never_a_token(
    recovery: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = "hf_LIST_MUST_NOT_PRINT_THIS"
    monkeypatch.setenv(recovery.ENV_TOKEN, sentinel)
    assert recovery.main(["list"]) == recovery.EXIT_OK
    out = capsys.readouterr().out
    assert "repo_type          : dataset" in out
    assert "READY" in out
    assert "https://huggingface.co/datasets/" in out
    assert sentinel not in out
    assert "present (value never printed)" in out


def test_a_credential_is_never_printed(
    recovery: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = "hf_THIS_MUST_NEVER_BE_PRINTED"
    monkeypatch.setenv(recovery.ENV_TOKEN, sentinel)
    monkeypatch.setenv(recovery.ENV_REPO_ID, "owner/repo")
    for argv in (["list"], ["fetch"]):
        recovery.main(argv)
        captured = capsys.readouterr()
        assert sentinel not in captured.out
        assert sentinel not in captured.err


# --------------------------------------------------------------------------- #
# 8. docs stay consistent with the canonical manifest
# --------------------------------------------------------------------------- #


def test_storage_doc_view_matches_the_canonical_manifest(manifest: dict[str, Any]) -> None:
    """The human-readable view must not drift from the machine-readable source."""
    text = STORAGE_DOC.read_text(encoding="utf-8")
    start = text.index("## B0 recoverability promotion")
    section = text[start : text.index("\n## ", start + 1)]

    rows: dict[str, dict[str, str]] = {}
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        rows[cells[0].strip("`")] = {
            "path": cells[1].strip("`"),
            "size": cells[2],
            "sha256": cells[3].strip("`"),
            "status": cells[4].strip("`"),
            "required_for": cells[5],
        }

    recorded = manifest["recoverability"]["artifacts"]
    assert set(rows) == {entry["role"] for entry in recorded}
    for entry in recorded:
        row = rows[entry["role"]]
        assert row["path"] == entry["path"]
        assert int(row["size"]) == entry["size_bytes"]
        assert row["sha256"] == entry["sha256"]
        assert row["status"] == entry["durable"]["status"]
        for consumer in entry["required_for"]:
            assert consumer in row["required_for"]


def test_storage_doc_states_the_no_local_copy_rule() -> None:
    # Normalise line wrapping: the rule is prose and may span two source lines.
    text = " ".join(STORAGE_DOC.read_text(encoding="utf-8").split())
    assert "not recoverability evidence" in text
    assert "PENDING" in text
    assert "scripts/fetch_artifacts.py" in text


def test_usage_documents_the_entrypoint_without_credentials() -> None:
    text = (REPO_ROOT / "docs" / "USAGE.md").read_text(encoding="utf-8")
    assert "scripts/fetch_artifacts.py" in text
    assert "HF_TOKEN=" not in text


def test_env_example_declares_the_three_variables_without_values() -> None:
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    for name in ("HF_REPO_ID", "HF_REVISION", "HF_TOKEN"):
        matches = [line for line in lines if line.startswith(f"{name}=")]
        assert matches == [f"{name}="], (name, matches)


# --------------------------------------------------------------------------- #
# 9. repository kind -> URI shape
# --------------------------------------------------------------------------- #


def test_dataset_repository_uri_carries_the_datasets_segment(recovery: Any) -> None:
    location = _location(recovery, "dataset")
    assert location.uri == (
        "https://huggingface.co/datasets/owner/name/resolve/main/artifacts/file.bin"
    )
    assert location.api_uri == "https://huggingface.co/api/datasets/owner/name"
    assert location.tree_uri() == "https://huggingface.co/api/datasets/owner/name/tree/main"


def test_model_repository_uri_has_no_collection_segment(recovery: Any) -> None:
    """A model file lives at /<repo_id>/resolve/... — the /models/ segment belongs to the API."""
    location = _location(recovery, "model")
    assert location.uri == "https://huggingface.co/owner/name/resolve/main/artifacts/file.bin"
    assert "/models/" not in location.uri
    assert location.api_uri == "https://huggingface.co/api/models/owner/name"


def test_revision_override_changes_both_uris(recovery: Any) -> None:
    location = _location(recovery, "dataset", revision="v2")
    assert "/resolve/v2/" in location.uri
    assert location.tree_uri("pinned") == "https://huggingface.co/api/datasets/owner/name/tree/pinned"


def test_canonical_manifest_declares_the_dataset_kind(manifest: dict[str, Any]) -> None:
    assert manifest["recoverability"]["repo_type"] == "dataset"


def test_a_prefixed_repo_id_is_rejected_to_avoid_a_doubled_segment(recovery: Any) -> None:
    for prefixed in ("datasets/owner/name", "models/owner/name"):
        with pytest.raises(recovery.BlockedError) as excinfo:
            recovery.check_repo_id(prefixed)
        assert "bare" in str(excinfo.value)
    assert recovery.check_repo_id("owner/name") == "owner/name"


def test_resolve_durable_rejects_a_prefixed_repo_id_from_the_environment(
    recovery: Any, manifest: dict[str, Any], entries: tuple[Any, ...]
) -> None:
    import dataclasses

    ready = dataclasses.replace(
        entries[0], durable_status="READY", remote_path="artifacts/file.bin"
    )
    with pytest.raises(recovery.BlockedError) as excinfo:
        recovery.resolve_durable(manifest, ready, {"HF_REPO_ID": "datasets/owner/name"})
    assert "doubled path" in str(excinfo.value)


def test_invalid_repo_type_fails_closed(recovery: Any) -> None:
    for bad in ("space", "DATASET", "", None, 7):
        with pytest.raises(recovery.RecoveryError) as excinfo:
            recovery.manifest_repo_type({"recoverability": {"repo_type": bad}})
        assert "repo_type" in str(excinfo.value)
    for good in ("model", "dataset"):
        assert recovery.manifest_repo_type({"recoverability": {"repo_type": good}}) == good


def test_resolve_durable_uses_the_manifest_kind(
    recovery: Any, manifest: dict[str, Any], entries: tuple[Any, ...]
) -> None:
    import dataclasses

    ready = dataclasses.replace(
        entries[0], durable_status="READY", remote_path="artifacts/file.bin"
    )
    location = recovery.resolve_durable(manifest, ready, {"HF_REPO_ID": "owner/name"})
    assert location.repo_type == "dataset"
    assert "/datasets/owner/name/resolve/main/" in location.uri


# --------------------------------------------------------------------------- #
# 10. configuration precedence
# --------------------------------------------------------------------------- #


def test_precedence_is_process_over_local_over_env(
    recovery: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "HF_REPO_ID=from-dot-env\nHF_REVISION=from-dot-env\nONLY_IN_ENV=yes\n", encoding="utf-8"
    )
    (tmp_path / ".env.local").write_text(
        "HF_REPO_ID=from-dot-env-local\nONLY_IN_LOCAL=yes\n", encoding="utf-8"
    )
    monkeypatch.setenv("HF_REPO_ID", "from-process")
    for absent in ("HF_REVISION", "ONLY_IN_ENV", "ONLY_IN_LOCAL"):
        monkeypatch.delenv(absent, raising=False)

    env = recovery.load_environment(tmp_path)
    assert env["HF_REPO_ID"] == "from-process"      # real environment wins
    assert env["HF_REVISION"] == "from-dot-env"     # .env fills what nothing higher sets
    assert env["ONLY_IN_LOCAL"] == "yes"            # .env.local is read
    assert env["ONLY_IN_ENV"] == "yes"


def test_env_local_overrides_env(
    recovery: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("HF_REVISION=from-dot-env\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("HF_REVISION=from-dot-env-local\n", encoding="utf-8")
    monkeypatch.delenv("HF_REVISION", raising=False)
    assert recovery.load_environment(tmp_path)["HF_REVISION"] == "from-dot-env-local"


def test_an_empty_value_is_unset(
    recovery: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env.local").write_text("HF_TOKEN=\nHF_REVISION=main\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_REVISION", raising=False)
    env = recovery.load_environment(tmp_path)
    assert "HF_TOKEN" not in env
    assert env["HF_REVISION"] == "main"


def test_an_empty_file_value_does_not_shadow_a_real_variable(
    recovery: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env.local").write_text("HF_REPO_ID=\n", encoding="utf-8")
    monkeypatch.setenv("HF_REPO_ID", "owner/name")
    assert recovery.load_environment(tmp_path)["HF_REPO_ID"] == "owner/name"


# --------------------------------------------------------------------------- #
# 11. probe: classification, size logic, and no bodies
# --------------------------------------------------------------------------- #


def test_http_status_classification_never_conflates_auth_and_not_found(recovery: Any) -> None:
    assert recovery.classify_http_status(401) == recovery.PROBE_AUTH
    assert recovery.classify_http_status(403) == recovery.PROBE_AUTH
    assert recovery.classify_http_status(404) == recovery.PROBE_NOT_FOUND
    assert recovery.classify_http_status(500) == recovery.PROBE_HTTP
    assert recovery.classify_http_status(200) == recovery.PROBE_HTTP


def test_transport_error_classification_separates_network_from_auth(recovery: Any) -> None:
    import socket as socket_module
    import urllib.error

    timeout = recovery.classify_transport_error(
        urllib.error.URLError(socket_module.timeout("timed out"))
    )
    assert timeout[0] == recovery.PROBE_TIMEOUT

    connection = recovery.classify_transport_error(
        urllib.error.URLError(OSError(101, "Network is unreachable"))
    )
    assert connection[0] == recovery.PROBE_CONNECTION

    auth = recovery.classify_transport_error(
        urllib.error.HTTPError("https://x/y", 401, "Unauthorized", None, None)
    )
    assert auth[0] == recovery.PROBE_AUTH and "401" in auth[1]

    missing = recovery.classify_transport_error(
        urllib.error.HTTPError("https://x/y", 404, "Not Found", None, None)
    )
    assert missing[0] == recovery.PROBE_NOT_FOUND

    other = recovery.classify_transport_error(
        urllib.error.HTTPError("https://x/y", 503, "Unavailable", None, None)
    )
    assert other[0] == recovery.PROBE_HTTP and "503" in other[1]


def test_size_header_preference_and_unavailable_reporting(recovery: Any) -> None:
    assert recovery.size_from_headers({"x-linked-size": "123"}) == (123, "x-linked-size")
    assert recovery.size_from_headers({"content-length": "123"}) == (123, "content-length")
    assert recovery.size_from_headers({"x-linked-size": "9", "content-length": "123"}) == (
        9,
        "x-linked-size",
    )
    assert recovery.size_from_headers({"etag": "abc"}) == (None, "unavailable")
    assert recovery.size_from_headers({"x-linked-size": "not-a-number"}) == (None, "unavailable")


@pytest.fixture()
def ready_entry(entries: tuple[Any, ...]) -> Any:
    """One entry promoted to READY with a recorded path, for probe tests."""
    import dataclasses

    return dataclasses.replace(
        entries[0], durable_status="READY", remote_path="artifacts/two_tower/checkpoint.pt"
    )


def _fake_json(repo_id: str = "owner/name"):  # type: ignore[no-untyped-def]
    def call(uri: str, token: str | None, timeout: int):  # noqa: ARG001
        if "/tree/" in uri:
            return 200, [{"type": "file", "path": "README.md"}]
        return 200, {"id": repo_id}

    return call


def test_probe_without_a_repository_touches_nothing(
    recovery: Any,
    manifest: dict[str, Any],
    pending_entries: tuple[Any, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(recovery.ENV_REPO_ID, raising=False)

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the probe attempted a network call with no repository configured")

    monkeypatch.setattr(recovery, "_http_get_json", explode)
    monkeypatch.setattr(recovery, "_http_head", explode)

    assert recovery.run_probe(pending_entries, manifest, tmp_path) == recovery.EXIT_BLOCKED
    out = capsys.readouterr().out
    assert "skipped" in out


def test_probe_skips_the_artifact_check_until_a_path_is_recorded(
    recovery: Any,
    manifest: dict[str, Any],
    pending_entries: tuple[Any, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A/B run; C reports a configuration gap instead of a fabricated result."""
    monkeypatch.setenv(recovery.ENV_REPO_ID, "owner/name")
    monkeypatch.setattr(recovery, "_http_get_json", _fake_json())

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("C must not probe a path that is not recorded")

    monkeypatch.setattr(recovery, "_http_head", explode)

    code = recovery.run_probe(pending_entries, manifest, tmp_path)
    assert code == recovery.EXIT_BLOCKED
    out = capsys.readouterr().out
    assert "remote_path is not recorded yet" in out
    assert "configuration gaps, not failures" in out


def test_probe_passes_and_pulls_no_body(
    recovery: Any,
    manifest: dict[str, Any],
    ready_entry: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(recovery.ENV_REPO_ID, "owner/name")
    monkeypatch.setattr(recovery, "_http_get_json", _fake_json())
    monkeypatch.setattr(
        recovery,
        "_http_head",
        lambda uri, token, timeout: (302, {"x-linked-size": str(ready_entry.size_bytes)}),
    )

    def no_download(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the probe downloaded an artifact body")

    monkeypatch.setattr(recovery, "download", no_download)

    assert recovery.run_probe([ready_entry], manifest, tmp_path) == recovery.EXIT_OK
    out = capsys.readouterr().out
    assert "PROBE PASS" in out
    assert "MATCH" in out


def test_probe_reports_a_size_mismatch_as_a_verification_failure(
    recovery: Any,
    manifest: dict[str, Any],
    ready_entry: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(recovery.ENV_REPO_ID, "owner/name")
    monkeypatch.setattr(recovery, "_http_get_json", _fake_json())
    monkeypatch.setattr(
        recovery, "_http_head", lambda uri, token, timeout: (302, {"x-linked-size": "1"})
    )

    assert recovery.run_probe([ready_entry], manifest, tmp_path) == recovery.EXIT_VERIFY
    assert "MISMATCH" in capsys.readouterr().out


def test_probe_reports_an_unavailable_size_without_faking_either_verdict(
    recovery: Any,
    manifest: dict[str, Any],
    ready_entry: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(recovery.ENV_REPO_ID, "owner/name")
    monkeypatch.setattr(recovery, "_http_get_json", _fake_json())
    monkeypatch.setattr(recovery, "_http_head", lambda uri, token, timeout: (302, {"etag": "x"}))

    assert recovery.run_probe([ready_entry], manifest, tmp_path) == recovery.EXIT_OK
    out = capsys.readouterr().out
    assert "unavailable" in out
    assert "neither a pass nor a failure" in out


def test_probe_classifies_a_blocked_egress_as_a_network_problem(
    recovery: Any,
    manifest: dict[str, Any],
    ready_entry: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The current environment has no egress; that must not read as a credential failure."""
    import socket as socket_module
    import urllib.error

    monkeypatch.setenv(recovery.ENV_REPO_ID, "owner/name")

    def unreachable(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.URLError(socket_module.timeout("timed out"))

    monkeypatch.setattr(recovery, "_http_get_json", unreachable)
    monkeypatch.setattr(recovery, "_http_head", unreachable)

    assert recovery.run_probe([ready_entry], manifest, tmp_path) == recovery.EXIT_NETWORK
    out = capsys.readouterr().out
    assert "timeout" in out
    assert "not a credential problem" in out


def test_probe_reports_authentication_separately_from_not_found(
    recovery: Any,
    manifest: dict[str, Any],
    ready_entry: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import urllib.error

    monkeypatch.setenv(recovery.ENV_REPO_ID, "owner/name")

    def unauthorized(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.HTTPError("https://x/y", 401, "Unauthorized", None, None)

    monkeypatch.setattr(recovery, "_http_get_json", unauthorized)
    monkeypatch.setattr(recovery, "_http_head", unauthorized)

    assert recovery.run_probe([ready_entry], manifest, tmp_path) == recovery.EXIT_VERIFY
    out = capsys.readouterr().out
    assert "authentication/permission" in out
    assert "HF_TOKEN" in out


def test_probe_never_prints_the_token(
    recovery: Any,
    manifest: dict[str, Any],
    ready_entry: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "hf_PROBE_MUST_NOT_PRINT_THIS"
    monkeypatch.setenv(recovery.ENV_REPO_ID, "owner/name")
    monkeypatch.setenv(recovery.ENV_TOKEN, sentinel)
    monkeypatch.setattr(recovery, "_http_get_json", _fake_json())
    monkeypatch.setattr(
        recovery, "_http_head", lambda uri, token, timeout: (302, {"x-linked-size": "1"})
    )

    recovery.run_probe([ready_entry], manifest, tmp_path)
    captured = capsys.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert "present (value never printed)" in captured.out


# --------------------------------------------------------------------------- #
# 12. durable status lifecycle (PENDING -> UPLOADED_UNVERIFIED -> READY)
# --------------------------------------------------------------------------- #


def test_the_lifecycle_has_three_distinct_states(recovery: Any) -> None:
    assert recovery.ALLOWED_STATUS == ("PENDING", "UPLOADED_UNVERIFIED", "READY")
    assert set(recovery.STATUS_MEANING) == set(recovery.ALLOWED_STATUS)


def test_only_a_pending_entry_is_refused_as_not_fetchable(recovery: Any) -> None:
    """An uploaded-but-unverified location must be fetchable — that is what B0-c verifies."""
    assert recovery.FETCHABLE_STATUS == ("UPLOADED_UNVERIFIED", "READY")
    assert recovery.STATUS_PENDING not in recovery.FETCHABLE_STATUS


def test_uploaded_unverified_is_resolvable_but_carries_no_claim(
    recovery: Any, manifest: dict[str, Any], entries: tuple[Any, ...]
) -> None:
    import dataclasses

    entry = dataclasses.replace(
        entries[0],
        durable_status=recovery.STATUS_UPLOADED_UNVERIFIED,
        remote_path="serving/twotower/checkpoint.pt",
    )
    location = recovery.resolve_durable(manifest, entry, {"HF_REPO_ID": "owner/name"})
    assert location.uri.endswith("/datasets/owner/name/resolve/main/serving/twotower/checkpoint.pt")
    # The state itself is the disclaimer; it must not read as READY.
    meaning = recovery.STATUS_MEANING[recovery.STATUS_UPLOADED_UNVERIFIED]
    assert "NOT been proven" in meaning
    assert "B0-c" in meaning
    assert "proven" in recovery.STATUS_MEANING[recovery.STATUS_READY]


def test_ready_is_reserved_for_a_proven_restore(recovery: Any) -> None:
    """READY is claimed only by a passing B0-c round-trip, never by an upload."""
    assert "B0-c passed" in recovery.STATUS_MEANING[recovery.STATUS_READY]
    assert "no remote location" in recovery.STATUS_MEANING[recovery.STATUS_PENDING]


def test_an_unknown_status_is_refused(recovery: Any, manifest: dict[str, Any]) -> None:
    import copy

    broken = copy.deepcopy(manifest)
    broken["recoverability"]["artifacts"][0]["durable"]["status"] = "READY_ISH"
    with pytest.raises(recovery.RecoveryError) as excinfo:
        recovery.load_recovery_entries(broken)
    assert "durable.status" in str(excinfo.value)


def test_pending_refusal_names_the_next_state(
    recovery: Any, manifest: dict[str, Any], pending_entries: tuple[Any, ...]
) -> None:
    with pytest.raises(recovery.BlockedError) as excinfo:
        recovery.resolve_durable(manifest, pending_entries[0], {"HF_REPO_ID": "owner/name"})
    message = str(excinfo.value)
    assert "UPLOADED_UNVERIFIED" in message
    assert "will not fall back to a local copy" in message


# --------------------------------------------------------------------------- #
# 13. B0-b registration: an upload must never claim READY
# --------------------------------------------------------------------------- #


def test_every_registered_entry_declares_its_remote_identity(manifest: dict[str, Any]) -> None:
    """Whatever its status, a registered artifact names the repository and path it would come from."""
    for entry in manifest["recoverability"]["artifacts"]:
        durable = entry["durable"]
        assert durable["repo_id"] == "autorec/agent-rec-x-artifacts", entry["role"]
        assert durable["revision"] == "main", entry["role"]
        # A PENDING artifact records where it *would* go without claiming a location exists.
        assert durable["remote_path"] or durable.get("intended_remote_path"), entry["role"]


def test_ready_entries_carry_restore_verification_and_pending_ones_do_not(
    manifest: dict[str, Any],
) -> None:
    """READY is a claim and must point at the restore that justified it; PENDING makes no claim.

    The sequences artifact is deliberately PENDING: its upload never committed, and the manifest
    records the blocker rather than a placeholder success.
    """
    for entry in manifest["recoverability"]["artifacts"]:
        durable = entry["durable"]
        status = durable["status"]
        if status == "READY":
            assert durable["verified_at"], entry["role"]
            assert durable["restore_sha256"] == entry["sha256"], entry["role"]
            assert durable["smoke_result"].startswith("PASS"), entry["role"]
        elif status == "UPLOADED_UNVERIFIED":
            # Uploaded and remotely queryable, but no restore has been proven yet.
            assert durable["commit_oid"], entry["role"]
            assert not durable["verified_at"], entry["role"]
        else:
            assert status == "PENDING", entry["role"]
            assert durable["commit_oid"] is None, entry["role"]
            assert durable["blocker"], entry["role"]


def test_recorded_remote_paths_are_the_uploaded_layout(
    entries: tuple[Any, ...],
) -> None:
    observed = {entry.role: entry.remote_path for entry in entries}
    assert observed["two_tower_checkpoint"] == "serving/twotower/checkpoint.pt"
    assert observed["mappings"] == "serving/catalog/Sports_and_Outdoors_mappings.json"
    assert observed["two_tower_metadata_categories"] == "serving/twotower/metadata_categories.json"


def test_registered_locations_derive_dataset_shaped_uris(
    recovery: Any, manifest: dict[str, Any], entries: tuple[Any, ...]
) -> None:
    environ = {"HF_REPO_ID": "autorec/agent-rec-x-artifacts"}
    for entry in entries:
        if entry.durable_status == recovery.STATUS_PENDING:
            # PENDING has no location by definition; its absence is asserted elsewhere.
            continue
        location = recovery.resolve_durable(manifest, entry, environ)
        assert location.uri.startswith(
            "https://huggingface.co/datasets/autorec/agent-rec-x-artifacts/resolve/main/"
        )


def test_b1_durability_record_is_storage_metadata_only() -> None:
    """The post-execution record must defer the verdict and must not be READY."""
    record = json.loads(
        (REPO_ROOT / "runs" / "b1_evidence" / "B1_DURABILITY.json").read_text(encoding="utf-8")
    )
    assert record["format"] == "agentrecx.b1.durability.v1"
    assert record["storage_status"] == "UPLOADED_UNVERIFIED"
    assert "storage metadata only" in record["statement"]
    assert record["verdict_source_of_truth"].endswith("B1_EXECUTION_MANIFEST.json -> verdicts")
    assert "READY is NOT claimed" in record["readiness"]
    b1_artifacts = [a for a in record["artifacts"] if a["kind"] == "b1"]
    assert len(b1_artifacts) == 7
    assert len(record["artifacts"]) == 9, "the record covers both runtime artifacts as well"
    for artifact in record["artifacts"]:
        assert artifact["upload_status"] == "UPLOADED_UNVERIFIED"
        assert len(artifact["sha256"]) == 64
        assert artifact["remote_lfs_oid"] == artifact["sha256"], artifact["file"]
        assert artifact["remote_tree_size"] == artifact["size_bytes"]
    for artifact in b1_artifacts:
        assert artifact["remote_path"].startswith("evidence/b1/")
