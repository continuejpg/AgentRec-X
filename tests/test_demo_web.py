"""Milestone 11 frontend tests: served assets, structure, and client-side safety.

No browser-testing framework is introduced (none exists in the repository).  The checks
combine three cheap, deterministic layers:

* the assets are served by FastAPI over HTTP, with the right content;
* the HTML contains the structural regions the UI contract requires;
* the JavaScript source is guarded — it must talk to the documented API paths, must
  render text safely, and must contain **no** client-side recommendation policy.

The guards are AST-lite source checks over the shipped file, which is exactly what the
milestone asks for: proof that the browser cannot re-sort, re-score or re-filter the
backend's answer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.demo_fixture import build_harness, close_harness  # noqa: E402

WEB_ROOT = REPO_ROOT / "recommendation" / "web"
INDEX = WEB_ROOT / "index.html"
APP_JS = WEB_ROOT / "app.js"
STYLES = WEB_ROOT / "styles.css"


@pytest.fixture()
def harness(tmp_path: Path):
    built = build_harness(tmp_path)
    try:
        yield built
    finally:
        close_harness(built)


# --------------------------------------------------------------------------- #
# Assets are served
# --------------------------------------------------------------------------- #


def test_assets_exist_and_are_plain(harness) -> None:
    for path in (INDEX, APP_JS, STYLES):
        assert path.exists(), f"missing web asset {path.name}"
        assert path.stat().st_size > 0
    assert not list(WEB_ROOT.glob("*.ts"))
    assert not list(WEB_ROOT.glob("package*.json")), "no Node toolchain in the web assets"


def test_demo_page_is_served(harness) -> None:
    response = harness.client.get("/demo/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AgentRec-X" in response.text


def test_root_redirects_to_the_demo_page(harness) -> None:
    response = harness.client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/demo/"


def test_static_assets_are_served_with_the_expected_types(harness) -> None:
    js = harness.client.get("/demo/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    css = harness.client.get("/demo/styles.css")
    assert css.status_code == 200
    assert "css" in css.headers["content-type"]


def test_static_mount_does_not_shadow_the_api(harness) -> None:
    """``/v1/*`` must keep resolving to the API, not to the static mount."""
    assert harness.client.get("/v1/demo/health").status_code == 200
    assert harness.client.get("/health").status_code == 200
    assert harness.client.get("/demo/../v1/demo/health").status_code in (200, 404)


# --------------------------------------------------------------------------- #
# HTML structure
# --------------------------------------------------------------------------- #

REQUIRED_ELEMENT_IDS = (
    "transcript",
    "chat-form",
    "message-input",
    "send-button",
    "profile-select",
    "new-session",
    "reset-session",
    "preferences",
    "cards",
    "banner",
    "audit",
    "state-value",
    "session-value",
    "turn-value",
    "route-value",
    "candidate-value",
    "moved-value",
    "ranked-with-value",
    "original-order",
    "reranked-order",
)


@pytest.mark.parametrize("element_id", REQUIRED_ELEMENT_IDS)
def test_index_declares_the_required_ui_regions(element_id: str) -> None:
    assert f'id="{element_id}"' in INDEX.read_text(encoding="utf-8")


def test_index_loads_only_local_assets() -> None:
    """Same-origin only: no CDN, no remote script, no external stylesheet."""
    html = INDEX.read_text(encoding="utf-8")
    assert "http://" not in html
    assert "https://" not in html
    assert "./app.js" in html
    assert "./styles.css" in html


def test_index_documents_the_next_turn_preference_semantics() -> None:
    html = INDEX.read_text(encoding="utf-8").lower()
    assert "future" in html
    assert "following turn" in html


# --------------------------------------------------------------------------- #
# JavaScript guards
# --------------------------------------------------------------------------- #


def js() -> str:
    """The shipped JavaScript source."""
    return APP_JS.read_text(encoding="utf-8")


def js_code() -> str:
    """The JavaScript source with comments removed.

    Guards run over this, so the file may *describe* the patterns it avoids (for example
    explaining that ``innerHTML`` is never used) without tripping its own check -- the
    Milestone 10C lesson about docstrings matching source guards.
    """
    source = js()
    without_block = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$", "", without_block)


def test_js_uses_the_documented_api_paths() -> None:
    source = js()
    assert '"/v1/demo"' in source
    for path in ('"/sessions"', '"/chat"', '"/profiles"', '"/health"'):
        assert path in source, f"app.js does not call {path}"


def test_js_never_uses_inner_html() -> None:
    """All dynamic text must go through textContent or DOM construction."""
    source = js_code()
    assert "innerHTML" not in source
    assert "outerHTML" not in source
    assert "insertAdjacentHTML" not in source
    assert "document.write" not in source
    assert "eval(" not in source


def test_js_writes_dynamic_text_with_text_content() -> None:
    source = js_code()
    assert "textContent" in source
    assert "createElement" in source


def test_js_contains_no_client_side_recommendation_policy() -> None:
    """The frontend renders the API sequence verbatim: it never reorders or re-scores.

    Note what is *allowed*: reading ``card.violation_count`` to print a count is
    presentation.  What is forbidden is order-affecting work -- sorting, reversing,
    filtering, reducing, or comparing scores to derive a position.
    """
    source = js_code()
    for forbidden in (
        ".sort(",
        ".reverse(",
        ".filter(",
        ".reduce(",
        "sortKey",
        "Math.max",
        "Math.min",
        "localeCompare",
        "BM25",
    ):
        assert forbidden not in source, f"app.js contains {forbidden!r}"

    for comparison in re.findall(r"sasrec_score\s*[<>]", source):
        pytest.fail(f"app.js compares scores: {comparison!r}")


def test_js_takes_each_card_position_from_the_api() -> None:
    """The displayed rank is the backend's rank field, never a computed position."""
    source = js_code()
    assert "card.reranked_rank" in source
    assert "card.original_rank" in source


def test_js_renders_the_api_order_without_reordering() -> None:
    """Cards are appended in the received order, and the rank shown is the API's rank."""
    source = js_code()
    body = source.split("function renderCards", 1)[1].split("function buildCard", 1)[0]
    assert ".forEach(" in body
    assert "appendChild" in body
    assert ".sort(" not in body
    assert ".reverse(" not in body


def test_js_does_not_label_the_score_as_confidence_or_probability() -> None:
    source = js_code()
    assert "ranking score, not a probability" in source
    for mislabel in ("confidence", "probability score", "rating", "preference score"):
        assert mislabel not in source.lower().replace("not a probability", "")


def test_js_states_the_unknown_semantics_conservatively() -> None:
    source = js_code()
    assert "unknown (metadata cannot decide this preference)" in source
    assert "does not match" not in source.lower()
    assert "fails preference" not in source.lower()


def test_js_keeps_only_the_session_id_in_page_memory() -> None:
    """No trusted history, no preference database contents, no server internals."""
    source = js_code()
    assert "localStorage" not in source
    assert "sessionStorage" not in source
    assert "trusted_user_history" not in source
    assert "user_key" not in source
    assert "document.cookie" not in source


def test_js_handles_every_documented_ui_state() -> None:
    source = js_code()
    for state in (
        "initializing",
        "ready",
        "sending",
        "success",
        "session expired",
        "error",
    ):
        assert f'"{state}"' in source, f"app.js never sets the {state!r} state"
    # Empty recommendations get an explicit neutral message.
    assert "No eligible recommendations are available for this history." in source


def test_js_disables_duplicate_submission_while_in_flight() -> None:
    source = js_code()
    assert "state.sending" in source
    assert "if (state.sending" in source
    assert "els.send.disabled" in source


def test_js_never_claims_quality_or_relevance() -> None:
    source = js().lower()
    for forbidden in ("best for you", "most relevant", "better for you", "optimal choice",
                      "more personalized", "this is better"):
        assert forbidden not in source


# --------------------------------------------------------------------------- #
# CSS sanity
# --------------------------------------------------------------------------- #


def test_stylesheet_defines_the_layout_and_state_colours() -> None:
    css = STYLES.read_text(encoding="utf-8")
    assert ".layout" in css
    assert "grid-template-columns" in css
    for selector in (".card", ".transcript", ".preferences", ".audit", ".banner"):
        assert selector in css


def test_stylesheet_has_no_remote_imports() -> None:
    css = STYLES.read_text(encoding="utf-8")
    assert "@import" not in css
    assert "url(" not in css


# --------------------------------------------------------------------------- #
# XSS-relevant source checks
# --------------------------------------------------------------------------- #


def test_no_inline_event_handlers_in_the_page() -> None:
    html = INDEX.read_text(encoding="utf-8")
    for pattern in (r"\sonclick=", r"\sonerror=", r"\sonload=", r"\sonmouseover="):
        assert not re.search(pattern, html), f"inline handler {pattern} found"


def test_javascript_is_loaded_as_an_external_file_not_inline() -> None:
    """No inline <script> body: the CSP story stays simple and the JS is testable."""
    html = INDEX.read_text(encoding="utf-8")
    assert "<script src=" in html
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>.", html, re.DOTALL)


def test_hostile_text_is_rendered_through_text_content_only() -> None:
    """A metadata title or chat message containing markup cannot become DOM."""
    source = js_code()
    # ``el()`` is the single text-writing helper and it uses textContent.
    helper = source.split("function el(", 1)[1].split("function clear(", 1)[0]
    assert "textContent" in helper
    assert "innerHTML" not in helper
