"""Lightweight documentation verification (documentation pack milestone).

These tests keep the prose honest without building a documentation framework. They check
four things that are cheap, deterministic and genuinely useful:

1. every relative link in the project documents resolves to a real file;
2. documented HTTP paths exist in the actual FastAPI application;
3. documented smoke module names are importable modules;
4. Mermaid fences are balanced, and the documents avoid unsupported marketing language.

They deliberately do **not** try to verify numeric claims. Numbers are checked by reading
the accepted artifact (`runs/sasrec_canonical_2026/run.json`,
`data/processed/Sports_and_Outdoors_products_manifest.json`) and by re-running the smokes,
which is what the milestone requires; a test that froze a metric into an assertion would
make the documentation harder to correct, not easier.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOC_PATHS: tuple[Path, ...] = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "ARCHITECTURE.md",
    REPO_ROOT / "docs" / "EXPERIMENTS.md",
    REPO_ROOT / "docs" / "USAGE.md",
    REPO_ROOT / "docs" / "PROJECT_HISTORY.md",
)

#: Marketing language the documentation must not use (unsupported by any artifact).
FORBIDDEN_MARKETING = (
    "revolutionary",
    "state-of-the-art",
    "state of the art",
    "industry-leading",
    "industry leading",
    "perfect personalization",
    "world-class",
    "best-in-class",
)

#: Endpoint shapes that never existed or were superseded; seeing them means stale docs.
#: Expressed as regexes with a negative lookbehind so the legitimate ``/v1/recommend``
#: is not flagged as a bare ``/recommend``.
OBSOLETE_ENDPOINT_PATTERNS = (
    r"(?<!v1)/recommend\b",
    r"/api/recommend\b",
    r"/v1/demo/chat\b",
    r"/v1/chat\b",
    r"POST /chat\b",
)

#: HTTP paths documented in the docs, mapped to the app path they must resolve to.
#: Each key must appear in the documented endpoint tables if it appears in the docs at all.
EXPECTED_APP_PATHS = (
    "/health",
    "/v1/model",
    "/v1/recommend",
    "/v1/demo/health",
    "/v1/demo/profiles",
    "/v1/demo/sessions",
    "/v1/demo/sessions/{session_id}",
    "/v1/demo/sessions/{session_id}/chat",
)

#: Smoke modules the usage document tells a reader to run.
DOCUMENTED_SMOKES = (
    "experiments.recommendation_tool_smoke",
    "experiments.agent_graph_smoke",
    "experiments.agent_tool_e2e_smoke",
    "experiments.product_rag_smoke",
    "experiments.agent_product_rag_smoke",
    "experiments.memory_smoke",
    "experiments.preference_matching_smoke",
    "experiments.preference_reranking_smoke",
    "experiments.reranking_evaluation_smoke",
    "experiments.agent_reranking_smoke",
    "experiments.web_demo_smoke",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", DOC_PATHS, ids=lambda p: p.name)
def test_document_exists_and_is_not_empty(path: Path) -> None:
    assert path.is_file(), f"missing document {path}"
    assert len(_text(path).strip()) > 500, f"{path.name} looks like a stub"


def test_all_documents_are_reachable_from_the_index() -> None:
    """Every project document is linked from README or the docs index."""
    index = _text(REPO_ROOT / "docs" / "README.md") + _text(REPO_ROOT / "README.md")
    for path in DOC_PATHS:
        if path.name in {"README.md"}:
            continue
        assert path.name in index, f"{path.name} is not linked from the documentation index"


@pytest.mark.parametrize("path", DOC_PATHS, ids=lambda p: p.name)
def test_relative_markdown_links_resolve(path: Path) -> None:
    """Every relative link target in a document exists on disk."""
    broken: list[str] = []
    for target in re.findall(r"\]\(([^)\s]+)\)", _text(path)):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        anchorless = target.split("#", 1)[0]
        if not anchorless:
            continue
        resolved = (path.parent / anchorless).resolve()
        if not resolved.exists():
            broken.append(target)
    assert broken == [], f"{path.name} links to missing paths: {broken}"


@pytest.mark.parametrize("path", DOC_PATHS, ids=lambda p: p.name)
def test_code_fences_are_balanced(path: Path) -> None:
    """An unclosed Mermaid or shell fence breaks GitHub rendering for the rest of the file."""
    fences = [line for line in _text(path).splitlines() if line.strip().startswith("```")]
    assert len(fences) % 2 == 0, f"{path.name} has an odd number of code fences"
    assert fences.count("```mermaid") <= len(fences) // 2


@pytest.mark.parametrize("path", DOC_PATHS, ids=lambda p: p.name)
def test_mermaid_blocks_are_conservative(path: Path) -> None:
    """Mermaid blocks use only the conservative diagram kinds and no stray fences."""
    text = _text(path)
    for block in re.findall(r"```mermaid\n(.*?)```", text, re.DOTALL):
        first = block.strip().splitlines()[0].strip()
        assert first.startswith(("flowchart ", "sequenceDiagram")), (
            f"{path.name} uses an unusual Mermaid diagram kind: {first!r}"
        )
        assert "```" not in block
        # Node labels must be quoted when they contain punctuation GitHub may mis-parse.
        for line in block.splitlines():
            for label in re.findall(r"\[([^\]]*)\]", line):
                if any(char in label for char in "()/:,"):
                    assert label.startswith('"') and label.endswith('"'), (
                        f"{path.name}: unquoted Mermaid label {label!r}"
                    )


def test_documented_http_paths_exist_in_the_application() -> None:
    """The endpoint tables match the real FastAPI application."""
    from recommendation.api.app import create_app

    documented = _text(REPO_ROOT / "README.md") + _text(REPO_ROOT / "docs" / "USAGE.md")
    for endpoint in EXPECTED_APP_PATHS:
        assert endpoint in documented, f"{endpoint} is not documented"

    # ``create_app`` builds the route table without touching artifacts or loading a model.
    paths = set(create_app(enable_demo=True).openapi()["paths"])
    assert set(EXPECTED_APP_PATHS) <= paths
    # Session paths are documented with the same parameter name the app uses.
    assert "/v1/model" in paths and "/v1/recommend" in paths


@pytest.mark.parametrize("name", DOCUMENTED_SMOKES)
def test_documented_smoke_modules_exist(name: str) -> None:
    """Every smoke the usage guide tells a reader to run is a real module."""
    module = REPO_ROOT / (name.replace(".", "/") + ".py")
    assert module.is_file(), f"documented smoke {name} does not exist"
    spec = importlib.util.spec_from_file_location(name, module)
    assert spec is not None and spec.loader is not None


def test_documented_package_readmes_exist() -> None:
    """Package READMEs referenced from the docs really exist."""
    packages = (
        "agent", "api", "baselines", "catalog", "datasets", "demo", "evaluation",
        "inference", "memory", "models", "preference_matching", "rag", "reranking",
        "tools", "training",
    )
    for package in packages:
        assert (REPO_ROOT / "recommendation" / package / "README.md").is_file(), package


@pytest.mark.parametrize("path", DOC_PATHS, ids=lambda p: p.name)
def test_no_unsupported_marketing_language(path: Path) -> None:
    lowered = _text(path).lower()
    for phrase in FORBIDDEN_MARKETING:
        assert phrase not in lowered, f"{path.name} contains unsupported claim {phrase!r}"


@pytest.mark.parametrize("path", DOC_PATHS, ids=lambda p: p.name)
def test_no_obsolete_endpoint_names(path: Path) -> None:
    text = _text(path)
    for pattern in OBSOLETE_ENDPOINT_PATTERNS:
        match = re.search(pattern, text)
        assert match is None, f"{path.name} references stale endpoint {match.group(0)!r}"


def test_docs_state_the_preference_relevance_limitation() -> None:
    """The no-relevance-claim limitation must be explicit in the two key documents."""
    for name in ("README.md", "docs/EXPERIMENTS.md"):
        lowered = _text(REPO_ROOT / name).lower()
        assert "preference-conditioned relevance labels" in lowered, name
        assert "satisfaction" in lowered, name


def test_docs_state_the_synthetic_fixture_caveat() -> None:
    """Preference fixtures used for diagnostics must be labelled synthetic."""
    for name in ("README.md", "docs/EXPERIMENTS.md"):
        lowered = _text(REPO_ROOT / name).lower()
        assert "synthetic" in lowered, name


def test_docs_do_not_present_itemcf_as_a_comparison() -> None:
    """The ItemCF non-comparability warning must survive in both key documents."""
    for name in ("README.md", "docs/EXPERIMENTS.md"):
        lowered = _text(REPO_ROOT / name).lower()
        assert "itemcf" in lowered, name
        assert "no same-artifact" in lowered or "not comparable" in lowered, name


def test_readme_stays_within_a_navigable_length() -> None:
    """README is the landing page, not a monolith; deep material lives in docs/."""
    lines = len(_text(REPO_ROOT / "README.md").splitlines())
    assert lines <= 560, f"README grew to {lines} lines; move detail into docs/"
    assert lines >= 200, f"README shrank to {lines} lines; it is the primary entry point"


def test_architecture_and_experiments_use_expected_sections() -> None:
    architecture = _text(REPO_ROOT / "docs" / "ARCHITECTURE.md")
    for heading in (
        "Architectural goals",
        "Component map",
        "Trust boundaries",
        "Failure behaviour",
        "Determinism and reproducibility",
        "Known limitations",
    ):
        assert heading in architecture, heading

    experiments = _text(REPO_ROOT / "docs" / "EXPERIMENTS.md")
    for heading in (
        "Research Questions",
        "Temporal Evaluation Protocol",
        "Accepted SASRec Results",
        "Reranking Policy Diagnostics",
        "Reproducibility",
        "Threats to Validity",
    ):
        assert heading in experiments, heading


def test_usage_document_covers_the_required_recipes() -> None:
    usage = _text(REPO_ROOT / "docs" / "USAGE.md")
    for heading in (
        "Environment assumptions",
        "Required artifacts",
        "Environment variables",
        "Running the Web Demo",
        "Opening the browser UI",
        "Example multi-turn conversation",
        "Calling the HTTP API directly",
        "Running tests",
        "Running smoke tests",
        "Troubleshooting",
    ):
        assert heading in usage, heading


def test_documented_environment_variables_are_read_by_the_code() -> None:
    """Every documented variable is actually read somewhere in the source."""
    usage = _text(REPO_ROOT / "docs" / "USAGE.md")
    variables = sorted(set(re.findall(r"\bAGENTRECX_[A-Z_]+\b", usage)))
    assert variables, "no environment variables documented"

    sources = "\n".join(
        _text(path)
        for path in (
            REPO_ROOT / "recommendation" / "config.py",
            REPO_ROOT / "recommendation" / "api" / "app.py",
            REPO_ROOT / "recommendation" / "memory" / "store.py",
        )
    )
    for variable in variables:
        assert variable in sources, f"{variable} is documented but never read by the code"
