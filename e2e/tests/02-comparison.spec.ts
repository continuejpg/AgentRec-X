/**
 * Scenario 2 — Comparison of the two verified real products (B004K6R43G vs B00008VF63).
 *
 * The UI path used is the real "Verified demo pair" action in the Comparison tab: it takes the
 * ids from the real GET /api/demo/config payload and calls the real POST /api/products/compare
 * (P16). The test cross-checks the DOM against the real HTTP JSON payload (no screenshots-only
 * claims) and then probes the real error path for invalid product refs at the BFF boundary.
 */
import { expect, test } from "@playwright/test";
import {
  ensureArtifactDirs,
  recordCheck,
  recordPayload,
  recordTiming,
  saveScreenshot,
} from "../fixtures/evidence";
import {
  bffGetJson,
  bffPostJson,
  openDemo,
  openPanelTab,
  waitForHealthReady,
} from "../fixtures/demo";

const DESKTOP = { width: 1440, height: 900 };
const VERIFIED = ["B004K6R43G", "B00008VF63"];

test.describe("Scenario 2 — Comparison", () => {
  test("verified pair comparison renders the real P16 payload; invalid refs return real errors", async ({
    page,
    request,
  }, testInfo) => {
    ensureArtifactDirs();
    const scenarioStarted = Date.now();
    await page.setViewportSize(DESKTOP);
    await openDemo(page);
    await waitForHealthReady(page, testInfo);

    const config = await bffGetJson(request, "/api/demo/config");
    expect(config.status).toBe(200);
    expect(config.body.verified_comparison_ids).toEqual(VERIFIED);
    recordCheck(
      testInfo,
      "network.config.verified_pair",
      `GET /api/demo/config verified_comparison_ids=${JSON.stringify(config.body.verified_comparison_ids)}`,
      JSON.stringify(config.body.verified_comparison_ids) === JSON.stringify(VERIFIED),
    );

    // ---- open the real Comparison tab and run the verified pair -----------------------
    await expect(page.getByTestId("right-panel")).toBeVisible({ timeout: 60_000 });
    await openPanelTab(page, "comparison");
    const selection = page.getByTestId("comparison-selection");
    await expect(selection).toBeVisible({ timeout: 60_000 });
    await expect(selection).toContainText("verified_comparison_ids");
    await expect(selection).toContainText("B004K6R43G");
    await expect(selection).toContainText("B00008VF63");

    const compareResponsePromise = page.waitForResponse(
      (response) =>
        response.url().includes("/api/products/compare") && response.request().method() === "POST",
      { timeout: 240_000 },
    );
    const compareStarted = Date.now();
    await page.getByTestId("compare-verified").click();
    const compareResponse = await compareResponsePromise;
    const comparePayload = await compareResponse.json();
    const compareRoundTrip = Date.now() - compareStarted;
    recordTiming(testInfo, "comparison.round_trip", compareRoundTrip, "real POST /api/products/compare (P16) round-trip");
    recordPayload(testInfo, "comparison", {
      status: comparePayload.status,
      products: (comparePayload.products ?? []).map((product: any) => ({
        product_id: product.product_id,
        title: product.title,
        role_status: product.role_status,
        unknown_fields: product.unknown_fields,
      })),
      dimension_attributes: (comparePayload.dimensions ?? []).map((dimension: any) => dimension.attribute),
      unknown_fields: comparePayload.unknown_fields,
      claims_summary: comparePayload.claims_summary,
      conditional_recommendation: comparePayload.conditional_recommendation,
      final_answer: comparePayload.final_answer,
      evidence_sources: comparePayload.evidence_sources,
    });

    const ids = (comparePayload.products ?? []).map((product: any) => product.product_id);
    const payloadChecks: Record<string, boolean> = {
      "http 200": compareResponse.status() === 200,
      "status comparison_ready": comparePayload.status === "comparison_ready",
      "two requested verified products": JSON.stringify(ids.sort()) === JSON.stringify([...VERIFIED].sort()),
      "dimensions returned": (comparePayload.dimensions ?? []).length > 0,
      "unknown fields returned": (comparePayload.unknown_fields ?? []).length > 0,
      "claims summary returned": comparePayload.claims_summary != null,
      // The live BFF returns the key but its value may legitimately be null for this pair.
      "conditional recommendation field present": Object.prototype.hasOwnProperty.call(
        comparePayload,
        "conditional_recommendation",
      ),
      "evidence sources returned": (comparePayload.evidence_sources ?? []).length > 0,
      "final answer returned": typeof comparePayload.final_answer === "string" && comparePayload.final_answer.length > 0,
    };
    const failing = Object.entries(payloadChecks)
      .filter(([, pass]) => !pass)
      .map(([name]) => name);
    recordCheck(
      testInfo,
      "network.compare.payload",
      `real P16 payload checks: ${Object.entries(payloadChecks)
        .map(([name, pass]) => `${name}=${pass ? "PASS" : "FAIL"}`)
        .join("; ")}`,
      failing.length === 0,
    );
    expect(compareResponse.status()).toBe(200);
    expect(failing, `comparison payload missing real fields: ${failing.join(", ")}`).toEqual([]);

    // ---- DOM rendering of the real payload -------------------------------------------
    const view = page.getByTestId("comparison-view");
    await expect(view).toBeVisible({ timeout: 120_000 });
    await expect(view).toHaveAttribute("data-status", "comparison_ready");
    const viewText = await view.innerText();
    // Section headings are rendered with CSS `text-transform: uppercase`, and innerText reflects
    // that transform (DIMENSIONS / CLAIMS SUMMARY), so compare headings case-insensitively.
    const upperViewText = viewText.toUpperCase();
    for (const id of VERIFIED) expect(viewText).toContain(id);
    for (const heading of [
      "Dimensions",
      "Preference relations",
      "Conflicts",
      "UNKNOWN fields",
      "Claims summary",
      "Conditional recommendation",
      "Evidence sources",
      "P16 final answer",
    ]) {
      expect(upperViewText, `comparison view must render "${heading}"`).toContain(heading.toUpperCase());
    }
    const dimensionRows = await view.locator("table tbody tr").count();
    expect(dimensionRows).toBeGreaterThan(0);
    const unknownSectionCount = await view.getByText(/UNKNOWN fields/i).count();
    expect(unknownSectionCount).toBeGreaterThanOrEqual(1);
    expect(upperViewText).toMatch(/UNKNOWN/);

    // claims summary numbers cross-checked against the real payload
    const claims = comparePayload.claims_summary ?? {};
    expect(upperViewText).toMatch(/CLAIMS SUMMARY[\s\S]{0,120}VALID[\s\S]{0,60}REJECTED[\s\S]{0,60}UNKNOWN/);
    recordCheck(
      testInfo,
      "comparison.claims_summary",
      `claims_summary valid=${claims.valid} rejected=${claims.rejected} unknown=${claims.unknown}; all real numbers rendered in the DOM`,
      claims.valid != null &&
        claims.rejected != null &&
        claims.unknown != null &&
        viewText.includes(String(claims.valid)) &&
        viewText.includes(String(claims.rejected)) &&
        viewText.includes(String(claims.unknown)),
    );

    // conditional recommendation text cross-checked verbatim (may legitimately be null)
    const conditional = page.getByTestId("conditional-recommendation");
    await expect(conditional).toBeVisible({ timeout: 60_000 });
    const conditionalText = await conditional.innerText();
    expect(conditionalText.toUpperCase()).toContain("SUPPORTED:");
    const conditionalRecommendationText = comparePayload.conditional_recommendation?.text ?? null;
    if (conditionalRecommendationText) {
      const normalise = (value: string) => value.replace(/\s+/g, " ").trim();
      expect(normalise(conditionalText)).toContain(normalise(String(conditionalRecommendationText)).slice(0, 40));
    } else {
      // Real BFF payload for this pair has conditional_recommendation=null; the UI must say so
      // instead of inventing a winner.
      expect(conditionalText).toMatch(/No conditional recommendation text was returned by P16/i);
    }
    recordCheck(
      testInfo,
      "comparison.conditional_recommendation",
      `conditional recommendation block: ${conditionalText.split("\n").slice(0, 3).join(" | ").slice(0, 240)}`,
      conditionalText.length > 20,
    );

    // P16 final answer / UNKNOWN / evidence source cross-checks
    if (comparePayload.final_answer) {
      const snippet = String(comparePayload.final_answer).replace(/\s+/g, " ").slice(0, 48);
      expect(viewText.replace(/\s+/g, " ")).toContain(snippet);
    }
    for (const source of comparePayload.evidence_sources ?? []) {
      expect(viewText).toContain(String(source));
    }
    const firstUnknownAttribute = comparePayload.unknown_fields?.[0]?.attribute ?? "";
    if (firstUnknownAttribute) {
      expect(viewText.toLowerCase()).toContain(String(firstUnknownAttribute).replace(/_/g, " ").toLowerCase());
    }
    recordCheck(
      testInfo,
      "comparison.dom_cross_check",
      `DOM renders ${ids.length} real products (${ids.join(", ")}), ${dimensionRows} dimension row(s), UNKNOWN/claims/conditional/final-answer/evidence-source blocks from the same HTTP JSON`,
      dimensionRows > 0 && ids.length === 2,
    );

    await saveScreenshot(page, testInfo, "desktop-comparison.png", DESKTOP);

    // ---- real error path: invalid product refs at the real BFF boundary ----------------
    const session = await bffPostJson(request, "/api/sessions", { profile_id: "profile_0049" });
    expect(session.status).toBe(200);
    const sessionId = session.body.session_id as string;

    const badCompare = await bffPostJson(request, "/api/products/compare", {
      session_id: sessionId,
      product_refs: ["B004K6R43G", "B000000000"],
    });
    const badCompareOk = [400, 422].includes(badCompare.status) && comparePayloadIsError(badCompare.body);
    recordPayload(testInfo, "invalid-ref-compare", {
      http_status: badCompare.status,
      body: badCompare.body,
    });
    recordCheck(
      testInfo,
      "comparison.invalid_ref.real_error",
      `POST /api/products/compare with a real + unknown ref → HTTP ${badCompare.status} code=${badCompare.body?.code} message=${String(
        badCompare.body?.message ?? "",
      ).slice(0, 160)} (no comparison_ready fallback)`,
      badCompareOk,
    );
    expect(
      [400, 422],
      `invalid compare ref must surface a real BFF error, got HTTP ${badCompare.status}: ${JSON.stringify(badCompare.body)}`,
    ).toContain(badCompare.status);
    expect(comparePayloadIsError(badCompare.body)).toBe(true);

    const badDetails = await bffPostJson(request, "/api/products/details", {
      session_id: sessionId,
      product_ref: "NOT_A_REAL_PRODUCT_REF",
    });
    const badDetailsOk =
      badDetails.status >= 400 &&
      typeof badDetails.body?.code === "string" &&
      String(badDetails.body?.message ?? "").length > 0;
    recordPayload(testInfo, "invalid-ref-details", { http_status: badDetails.status, body: badDetails.body });
    recordCheck(
      testInfo,
      "details.invalid_ref.real_error",
      `POST /api/products/details with an invalid ref → HTTP ${badDetails.status} code=${badDetails.body?.code} message=${String(
        badDetails.body?.message ?? "",
      ).slice(0, 160)}`,
      badDetailsOk,
    );
    expect(badDetails.status, "invalid details ref must not be a fabricated success").toBeGreaterThanOrEqual(400);
    expect(badDetailsOk).toBe(true);

    recordTiming(testInfo, "scenario2.total", Date.now() - scenarioStarted);
  });
});

/** The real error body must carry code/message and must never pretend a comparison succeeded. */
function comparePayloadIsError(body: any): boolean {
  if (!body || typeof body !== "object") return false;
  const hasErrorShape =
    typeof body.code === "string" && body.code.length > 0 && typeof body.message === "string" && body.message.length > 0;
  const looksLikeSuccess = body.status === "comparison_ready" || Array.isArray(body.products);
  return hasErrorShape && !looksLikeSuccess;
}
