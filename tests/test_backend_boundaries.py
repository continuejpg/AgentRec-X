"""Boundary guards for the public-TIGER backend split (Step 2.3).

These tests are the mechanical form of ``docs/TIGER_BACKEND.md`` and ``AGENTS.md`` section 19.
They assert **absences**, which is the only way a boundary survives contact with future work:
a later milestone cannot quietly import torch into the adapter, mention canonical identity on
the backend side, or paste a vendored file in without a test failing.

Every check reads source text or artifact schemas.  None of them needs the backend virtual
environment, a model, a GPU or the 307 MB catalogue artifact to be present, so
``pytest -q`` stays runnable exactly as it is today.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKEND_ROOT = REPO_ROOT / "backends" / "tiger_public"
BACKEND_SRC = BACKEND_ROOT / "src" / "tiger_public"
ADAPTER_ROOT = REPO_ROOT / "recommendation" / "backends"

#: Modules the adapter must never import: the backend's ML stack, and the backend itself.
FORBIDDEN_IN_ADAPTER = {
    "torch",
    "transformers",
    "sentence_transformers",
    "lightning",
    "pytorch_lightning",
    "hydra",
    "tensorflow",
    "sklearn",
    "scipy",
    "tiger_public",
}

#: Modules the backend must never import from AgentRec-X.
FORBIDDEN_IN_BACKEND_PREFIXES = ("recommendation",)

#: A handoff record may never declare one of these; their presence means a target, a split
#: name, a seen set, or canonical identity crossed the boundary.
FORBIDDEN_ARTIFACT_KEYS = (
    "target",
    "targets",
    "label",
    "labels",
    "validation_target",
    "test_target",
    "seen",
    "parent_asin",
)


def _python_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _module_level_imports(path: Path) -> set[str]:
    """Top-level module names imported at *module scope*, via the AST.

    Only module scope counts.  An import deferred inside a function is a deliberate choice -
    the adapter defers ``torch`` so that its own contract stays NumPy-only - and treating it as
    a violation would forbid the very technique that keeps the boundary testable.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as error:  # pragma: no cover - a syntax error fails the suite elsewhere
        raise AssertionError(f"{path} does not parse: {error}") from error
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def _imported_modules_anywhere(path: Path) -> set[str]:
    """Every module name imported anywhere in a file, including inside functions."""
    names = set(_module_level_imports(path))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


# --------------------------------------------------------------------------- #
# T1 / T2 / T3 - import direction
# --------------------------------------------------------------------------- #


def test_t1_backend_never_imports_agentrecx() -> None:
    """The backend is a separate project; it may not reach into AgentRec-X."""
    offenders: list[str] = []
    for path in _python_files(BACKEND_ROOT):
        for name in _imported_modules_anywhere(path):
            if name.startswith(FORBIDDEN_IN_BACKEND_PREFIXES):
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports {name}")
    assert offenders == [], f"backend imports AgentRec-X: {offenders}"


def test_t2_adapter_has_no_ml_import() -> None:
    """The adapter's *own* contract is NumPy-only.

    A module-scope import of the backend's ML stack would make AgentRec-X depend on the
    backend venv, so it is forbidden.  ``torch`` is deferred inside one function on purpose -
    the evaluator needs it, and AgentRec-X already depends on it - and that deferral is
    asserted separately below rather than being treated as a violation.
    """
    offenders: list[str] = []
    for path in _python_files(ADAPTER_ROOT):
        for name in _module_level_imports(path):
            if name in FORBIDDEN_IN_ADAPTER:
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports {name}")
    assert offenders == [], f"adapter imports a forbidden module: {offenders}"


def test_t3_adapter_does_not_reuse_the_frozen_genrec_v0_or_evaluator() -> None:
    """The adapter must not *import* GenRec v0 or the evaluator.

    Importing ``recommendation.semantic_id`` would silently resurrect the historical baseline
    inside the new backend, and importing the evaluator would blur who owns ranking.  Prose may
    name both - the module explains why it does neither - so only imports are checked.
    """
    offenders: list[str] = []
    for path in _python_files(ADAPTER_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(
                    ("recommendation.semantic_id", "recommendation.evaluation")
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)} imports {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(
                        ("recommendation.semantic_id", "recommendation.evaluation")
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)} imports {alias.name}")
    assert offenders == [], f"adapter imports a frozen package: {offenders}"


def test_adapter_is_importable_in_a_numpy_only_environment() -> None:
    """Importing the adapter must not require torch.

    The evaluator hands the adapter's output to ``evaluate_batched`` in AgentRec-X's own
    process, where torch exists.  Deferring that import keeps the adapter's *own* contract
    NumPy-only, which is what lets the boundary be tested without the ML stack.
    """
    source = (ADAPTER_ROOT / "tiger_backend.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_level = {
        name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in (
            [alias.name.split(".", 1)[0] for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module.split(".", 1)[0]] if node.module else []
        )
    }
    assert "torch" not in module_level
    assert "numpy" in module_level


# --------------------------------------------------------------------------- #
# T4 - file allow-list (no ported split / evaluator / metric code)
# --------------------------------------------------------------------------- #


def test_t4_backend_contains_no_split_or_evaluator_module() -> None:
    """An external repo's split or evaluator must not be ported in.

    ``retrieve.py`` and ``scoring.py`` are permitted: scoring is the backend's own job, and
    retrieval is a search strategy.  A file named for an evaluator, a metric, or a split is
    not, because the backend owns none of those.
    """
    forbidden_fragments = ("eval", "metric", "split", "leave_one", "leave_two", "ndcg", "recall")
    allowed = {"scoring.py", "retrieve.py"}
    offenders: list[str] = []
    for path in _python_files(BACKEND_ROOT):
        if path.name in allowed:
            continue
        name = path.name.lower()
        if any(fragment in name for fragment in forbidden_fragments):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"backend holds split/evaluator/metric modules: {offenders}"


# --------------------------------------------------------------------------- #
# T5 - no target-shaped field, and no seen set, in any handoff schema
# --------------------------------------------------------------------------- #


def test_t5_handoff_schemas_declare_no_target_or_seen_field() -> None:
    """The handoff dataclasses must not carry a target, a split name, a seen set, or identity.

    A field would be enough: if the type has nowhere to put a target, no caller can pass one.
    """
    offenders: list[str] = []
    for module, classes in (
        (ADAPTER_ROOT / "tiger_backend.py", ("CatalogueHandoffData", "TrainExposureHandoffData",
                                             "EvalCohortHandoffData")),
        (BACKEND_SRC / "contracts.py", ("CatalogueHandoff", "TrainExposureHandoff",
                                        "EvalCohortHandoff")),
    ):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name not in classes:
                continue
            for statement in node.body:
                if not isinstance(statement, ast.AnnAssign) or statement.target is None:
                    continue
                field_name = getattr(statement.target, "id", "")
                for forbidden in FORBIDDEN_ARTIFACT_KEYS:
                    if forbidden in field_name:
                        offenders.append(f"{module.name}:{node.name}.{field_name}")
    assert offenders == [], f"handoff schemas declare forbidden fields: {offenders}"


def test_t5_eval_cohort_handoff_has_exactly_the_permitted_fields() -> None:
    """The cohort handoff is pinned to histories plus an integer frontier request."""
    from recommendation.backends.tiger_backend import EvalCohortHandoffData

    fields = set(EvalCohortHandoffData.__dataclass_fields__)
    assert fields == {
        "cohort_seed",
        "cohort_size",
        "protocol_version",
        "k_values",
        "case_ids",
        "test_histories",
        "required_frontier",
        "catalogue_sha256",
    }


def test_t5_no_handoff_artifact_writer_emits_a_target_key() -> None:
    """The adapter's own writers are scanned for a target-shaped JSON key."""
    source = (ADAPTER_ROOT / "tiger_backend.py").read_text(encoding="utf-8")
    # The only permitted occurrences are the *refusal* list and the docstrings that explain
    # them; a writer that emitted one would have to appear as a quoted key elsewhere.
    tree = ast.parse(source)
    emitted: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if any(f == key.value.lower() for f in FORBIDDEN_ARTIFACT_KEYS):
                        emitted.append(key.value)
    assert emitted == [], f"the adapter emits a forbidden JSON key: {sorted(set(emitted))}"


# --------------------------------------------------------------------------- #
# T6 - provenance
# --------------------------------------------------------------------------- #


def test_t6_every_backend_file_is_declared_in_provenance() -> None:
    """A pasted file cannot go unnoticed: every backend source file must be labelled."""
    provenance = BACKEND_ROOT / "PROVENANCE.md"
    assert provenance.is_file(), "backends/tiger_public/PROVENANCE.md is required"
    text = provenance.read_text(encoding="utf-8")
    missing: list[str] = []
    for path in sorted(BACKEND_ROOT.rglob("*")):
        if not path.is_file() or path.name == "PROVENANCE.md":
            continue
        if any(part in {".venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        relative = path.relative_to(BACKEND_ROOT)
        if str(relative) not in text:
            missing.append(str(relative))
    assert missing == [], f"PROVENANCE.md does not label: {missing}"


# --------------------------------------------------------------------------- #
# T7 - canonical identity is absent from the adapter and the backend
# --------------------------------------------------------------------------- #

#: Prose is permitted to name the rules it states; these are the prose suffixes.
_PROSE_SUFFIXES = {".md", ".txt", ".rst"}


def _code_identifiers(path: Path) -> set[str]:
    """Every *identifier* a Python file contains: names, attributes, arguments, class fields.

    Docstrings and comments are deliberately excluded here.  Prose in a module docstring that
    names the rule it forbids is the documentation doing its job; an *identifier* is the name
    actually being meaningful to the code, and that is what must never appear.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.arg):
            tokens.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            tokens.add(node.arg)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            tokens.add(node.target.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            tokens.add(node.name)
    return tokens


def test_t7a_adapter_never_names_canonical_identity() -> None:
    """``recommendation/backends/`` is item-id only: the name is not even an identifier.

    The adapter's docstrings state the rule (so a reader learns it) but the name must not be a
    parameter, field, local, keyword, class attribute or JSON key anywhere in the package -
    there is no channel through which canonical identity could arrive.
    """
    offenders: list[str] = []
    for path in _python_files(ADAPTER_ROOT):
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        for token in _code_identifiers(path):
            if "parent_asin" in token:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{token}")
    assert offenders == [], (
        f"canonical identity is an identifier under recommendation/backends/: {offenders}"
    )


def test_t7a_the_adapter_writes_no_canonical_identity_json_key() -> None:
    """Every JSON key the adapter emits is checked, and none is canonical identity."""
    tree = ast.parse((ADAPTER_ROOT / "tiger_backend.py").read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and "parent_asin" in str(key.value):
                    offenders.append(str(key.value))
    assert offenders == [], f"the adapter emits a canonical-identity key: {offenders}"


def test_t7b_backend_code_never_names_canonical_identity() -> None:
    """Under ``backends/`` the name may appear only in prose that states the rule."""
    offenders: list[str] = []
    for path in _python_files(BACKEND_ROOT):
        if any(part in {".venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        for token in _code_identifiers(path):
            if "parent_asin" in token:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{token}")
    assert offenders == [], f"canonical identity is an identifier in backend code: {offenders}"


#: Prose that is expected to name the rule it states.  Deliberately an explicit file list
#: rather than "any markdown": a future artifact directory containing a `.md` file must still
#: be scanned, so allowing a whole suffix would open the hole this guard exists to close.
_PROSE_ALLOWLIST = {"README.md", "PROVENANCE.md"}


def test_t7b_no_backend_data_file_carries_canonical_identity() -> None:
    """A committed data file under ``backends/`` must not carry the name either.

    ``README.md`` and ``PROVENANCE.md`` are exempt because naming the canonical key is how they
    document what is deliberately *not* absorbed; every other non-Python file is scanned as raw
    text, so an emitted artifact record would be caught.
    """
    offenders: list[str] = []
    for path in sorted(BACKEND_ROOT.rglob("*")):
        if not path.is_file() or path.suffix == ".py" or path.name in _PROSE_ALLOWLIST:
            continue
        if any(part in {".venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if "parent_asin" in path.read_text(encoding="utf-8", errors="replace"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"canonical identity appears in backend data files: {offenders}"


def test_t7_handoff_artifacts_carry_only_opaque_item_ids() -> None:
    """A materialised catalogue record holds exactly ``backend_row`` and ``item_id``."""
    source = (ADAPTER_ROOT / "tiger_backend.py").read_text(encoding="utf-8")
    assert '"backend_row": row, "item_id": item_id' in source
    # And the record's schema is pinned by the reader.
    assert "{'backend_row', 'item_id'}" in source


# --------------------------------------------------------------------------- #
# T8 - the frozen score rule is single-sourced
# --------------------------------------------------------------------------- #


def test_t8_score_rule_lives_in_exactly_one_module() -> None:
    """``scoring.py`` is the only *source* module where the item-score rule is defined.

    Splitting it across beam search, branch-and-bound and exhaustive scoring is precisely how
    two algorithms come to disagree about what "score" means and the certified comparison
    silently stops being comparable.  ``cli.py`` may only *record* it; the backend's own tests
    may reference it.
    """
    definitions = [
        path
        for path in _python_files(BACKEND_SRC)
        if "SCORE_RULE" in path.read_text(encoding="utf-8")
    ]
    names = sorted(path.name for path in definitions)
    assert names == ["cli.py", "scoring.py"], f"the score rule is declared in {names}"
    scoring_source = (BACKEND_SRC / "scoring.py").read_text(encoding="utf-8")
    assert '"eos_in_score": False' in scoring_source
    assert '"child_renormalisation": False' in scoring_source


def test_t8_the_rule_values_are_declared_only_in_scoring() -> None:
    """The rule's *values* live in ``scoring.py`` alone.

    Re-declaring them in a validator is how two modules eventually disagree about a frozen
    rule, so ``contracts.py`` checks only that a checkpoint *records* a rule and leaves the
    comparison to ``scoring.validate_score_rule``.
    """
    contracts = (BACKEND_SRC / "contracts.py").read_text(encoding="utf-8")
    assert "full_vocabulary" not in contracts
    assert '"eos_in_score": False' not in contracts
    scoring = (BACKEND_SRC / "scoring.py").read_text(encoding="utf-8")
    assert '"full_vocabulary"' in scoring
    assert '"eos_in_score": False' in scoring
    assert '"child_renormalisation": False' in scoring


def test_t8_no_module_renormalises_probabilities_over_valid_children() -> None:
    """A renormalisation would make a score depend on which other items were in the frontier."""
    offenders: list[str] = []
    for path in _python_files(BACKEND_SRC):
        if path.name == "scoring.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and "renormal" in node.id.lower():
                offenders.append(f"{path.name}:{node.id}")
            elif isinstance(node, ast.Attribute) and "renormal" in node.attr.lower():
                offenders.append(f"{path.name}:{node.attr}")
    assert offenders == [], f"a module renormalises over valid children: {offenders}"


def test_t8_adapter_and_backend_layout_arithmetic_agree() -> None:
    """The two sides must derive the same token layout, or SIDs would not round-trip."""
    from recommendation.backends.tiger_backend import build_token_layout

    layout = build_token_layout(levels=3, codebook_size=64, dedup_levels=1)
    assert layout["vocab_size"] == (3 + 1) * 64 + 3
    assert layout["special"] == {"pad": 256, "bos": 257, "eos": 258}
    assert layout["level_offsets"] == [0, 64, 128, 192]
    assert layout["sentinel_tokenisable"] is False


def test_t8_both_sides_compute_identical_collision_audits() -> None:
    """Duplication is only safe while it is *checked*: run both implementations on one input.

    The adapter and the backend each declare ``collision_audit`` because the backend must not be
    imported by AgentRec-X.  That duplication is deliberate, so this test keeps it honest: the
    two must agree field-for-field, including the pre/post split and the overflow flag.  The
    backend module is loaded from its file path, so the backend package never has to be
    importable from AgentRec-X's environment.
    """
    import importlib.util

    from recommendation.backends.tiger_backend import collision_audit as adapter_audit

    path = BACKEND_SRC / "contracts.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("tiger_public_contracts_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before execution: a frozen dataclass resolves its own module through
    # ``sys.modules`` while the decorator runs, so an unregistered probe module fails to build.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    codes = [(1, 1)] * 5 + [(2, 2)] + [(3, 4)]
    for dedup in (0, 1):
        for size in (4, 256):
            assert adapter_audit(
                codes, dedup_levels=dedup, dedup_vocab_size=size
            ) == module.collision_audit(codes, dedup_levels=dedup, dedup_vocab_size=size)


def test_contract_versions_match_on_both_sides() -> None:
    """A schema drift between the two re-declarations is refused at read time; assert it here."""
    backend = (BACKEND_SRC / "contracts.py").read_text(encoding="utf-8")
    adapter = (ADAPTER_ROOT / "tiger_backend.py").read_text(encoding="utf-8")
    for source, label in ((backend, "backend"), (adapter, "adapter")):
        assert '"agentrecx.tiger_backend.v3"' in source, label
    assert "agentrecx.tiger.catalogue.v3" in backend and "agentrecx.tiger.catalogue.v3" in adapter
    assert "agentrecx.tiger.eval_cohort.v3" in backend and "agentrecx.tiger.eval_cohort.v3" in adapter
    assert "agentrecx.tiger.scores.v3" in backend and "agentrecx.tiger.scores.v3" in adapter


def test_no_training_or_model_artifact_exists_yet() -> None:
    """Step 2.3 produces no checkpoint, no embedding and no measurement."""
    heavy = [
        path
        for path in BACKEND_ROOT.rglob("*")
        if path.is_file() and path.suffix in {".pt", ".pth", ".ckpt", ".npy", ".npz"}
    ]
    assert heavy == [], f"Step 2.3 must not ship model artifacts: {heavy}"


def test_backend_entrypoint_exposes_exactly_the_four_stages() -> None:
    """The CLI is the production crossing; its stage set is part of the contract."""
    source = (BACKEND_SRC / "cli.py").read_text(encoding="utf-8")
    for stage in ("build-features", "fit-sid", "train", "score"):
        assert f'"{stage}"' in source, stage
    for absent in ("evaluate", "compute-recall", "split", "ndcg"):
        assert f'"{absent}"' not in source, f"the CLI exposes a non-contract stage {absent!r}"


def test_stub_marker_is_unmistakable() -> None:
    """A placeholder must be labelled so it can never be read as a measurement."""
    source = (BACKEND_SRC / "__init__.py").read_text(encoding="utf-8")
    assert '"step-2.3-placeholder-no-ml"' in source
    cli = (BACKEND_SRC / "cli.py").read_text(encoding="utf-8")
    assert 'STUB_MARKER = "step-2.3-placeholder-no-ml"' in cli
    assert '"validation_used": False' in cli


def test_handoff_json_schema_is_target_free_when_materialised(tmp_path: Path) -> None:
    """Materialise a tiny handoff and assert no target-shaped field reached any artifact.

    This is the end-to-end form of T5: it inspects the bytes, not just the declarations.
    """
    from recommendation.backends.tiger_backend import TigerBackendAdapter

    adapter = TigerBackendAdapter(tmp_path, timeout_seconds=30.0)
    catalogue = adapter.materialise_catalogue(
        item_ids=[1, 2, 3],
        num_users=1,
        mappings_sha256="a" * 64,
        sequences_sha256="b" * 64,
        products_sha256="c" * 64,
    )
    adapter.materialise_products_text(["alpha", "beta", "gamma"], catalogue=catalogue)
    adapter.materialise_train_exposure(train_histories=[[1, 2, 3]], catalogue=catalogue)
    cohort, stats = adapter.materialise_eval_cohort(test_histories=[[1, 2]], catalogue=catalogue)
    adapter.write_score_request(status="APPROXIMATE", batch_size=2)
    manifest = adapter.write_manifest()
    assert stats.minimum >= 20  # k_max = 20, |seen| = 2 -> 22
    assert stats.maximum == stats.minimum

    offenders: list[str] = []
    for name in manifest["files"]:
        text = (tmp_path / name).read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_ARTIFACT_KEYS:
            if f'"{forbidden}"' in text:
                offenders.append(f"{name}:{forbidden}")
    assert offenders == [], f"a handoff artifact carries a forbidden key: {offenders}"

    # The target deliberately does not exist anywhere in the written cohort.
    cohort_payload = json.loads((tmp_path / "eval_cohort.json").read_text(encoding="utf-8"))
    assert "target" not in json.dumps(cohort_payload).lower()
    assert cohort.required_frontier == (22,)


def test_forged_pad_item_id_is_refused(tmp_path: Path) -> None:
    """``item_id = 0`` is PAD and must be refused rather than silently accepted."""
    from recommendation.backends.tiger_backend import ContractViolation, TigerBackendAdapter

    adapter = TigerBackendAdapter(tmp_path, timeout_seconds=30.0)
    with pytest.raises(ContractViolation):
        adapter.materialise_catalogue(
            item_ids=[0, 1, 2],
            num_users=1,
            mappings_sha256="a" * 64,
            sequences_sha256="b" * 64,
            products_sha256="c" * 64,
        )
