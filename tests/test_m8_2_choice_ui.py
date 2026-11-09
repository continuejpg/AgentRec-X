"""M8.2 — demo frontend choice-feedback checks (static + served assets).

No browser framework is introduced.  The tests verify the shipped page/JS contract:
the optional choice panel exists, the documented choice endpoints are used, session
restoration uses the URL capability hash (no browser storage), the page defaults to
hidden when collection is disabled, and the server rejects the disabled read path.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.demo_fixture import PROFILE_A, build_harness, close_harness  # noqa: E402

WEB_ROOT = REPO_ROOT / "recommendation" / "web"
INDEX = WEB_ROOT / "index.html"
APP_JS = WEB_ROOT / "app.js"
STYLES = WEB_ROOT / "styles.css"


def test_choice_panel_and_optional_feedback_controls_exist() -> None:
    html = INDEX.read_text(encoding="utf-8")
    for element_id in (
        "shop-choice",
        "shop-choice-history",
        "shop-choice-status",
        "choice-satisfaction",
        "choice-confidence",
        "shop-no-suitable",
    ):
        assert f'id="{element_id}"' in html
    assert "No suitable product" in html
    assert "not a ground-truth target" in html
    assert "no authentication" in html.lower()
    assert "Satisfaction" in html
    assert "Choice confidence" in html
    # The panel starts hidden: default mode must look exactly like the accepted page.
    assert re.search(r'id="shop-choice"[^>]*\bhidden\b', html)


def test_js_uses_both_choice_endpoints_and_guards_duplicates() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert '"/agent/choice"' in source
    assert '"/agent/choices"' in source
    assert "Choose this product" in source
    assert "no_suitable_item" in source
    assert "state.shop.choiceSending" in source
    assert "state.shop.choiceRecord" in source
    assert "#session=" in source
    # M8.2 must not introduce browser storage; session restoration travels in the URL
    # capability hash only, which is also what keeps the demo account-free.
    assert "localStorage" not in source
    assert "sessionStorage" not in source


def test_js_restores_choice_panel_from_server_records_only() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    assert "loadChoiceHistory" in source
    assert "renderChoiceHistory" in source
    assert "renderChoiceFeedback" in source
    assert "choice_collection_disabled" in source
    # The UI renders the server's selected identity/flags; it never derives a target.
    assert "selected_parent_asin" in source
    assert "no_suitable_item" in source
    assert "target_ids" not in source


def test_js_source_has_no_corrupted_unicode_tokens() -> None:
    """Guard the browser asset against the encoding corruption fixed in M8.3."""
    source = APP_JS.read_text(encoding="utf-8")
    assert "\u9225" not in source
    assert "\ufffd" not in source
    assert "Sending…" in source


def test_choice_style_selectors_exist() -> None:
    css = STYLES.read_text(encoding="utf-8")
    for selector in (
        ".shop-choice",
        ".shop-choice-history",
        ".shop-choice-status",
        ".shop-card-choice",
        "#shop-no-suitable",
    ):
        assert selector in css


def test_disabled_collection_hides_page_capability_endpoint() -> None:
    """When M8.1 collection is off the read endpoint 404s, so the panel stays hidden."""
    import tempfile

    root = Path(tempfile.mkdtemp())
    harness = build_harness(root, memory_db=root / "memory.sqlite3")
    try:
        session = harness.client.post(
            "/v1/demo/sessions", json={"profile_id": PROFILE_A}
        ).json()["session_id"]
        response = harness.client.get(
            f"/v1/demo/sessions/{session}/agent/choices"
        )
        assert response.status_code == 404, response.text
        assert response.json()["error"] == "choice_collection_disabled"
        page = harness.client.get("/demo/")
        assert page.status_code == 200
        assert 'id="shop-choice"' in page.text
        script = harness.client.get("/demo/app.js")
        assert "/agent/choices" in script.text
    finally:
        close_harness(harness)
