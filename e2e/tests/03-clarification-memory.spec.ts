/**
 * Scenario 3 — Clarification & Memory (Demo B profile_0057, real M9 write through the real BFF).
 *
 * Real reference: P18_FINAL_REPORT.md §2 Demo 3 and runs/p18/demos/clarification_and_memory.json.
 * The vague query must trigger a real P16 clarification; the reply "I prefer noise cancelling and
 * wireless." must be persisted by the real MCP server (memory_update.added == 2), refresh the
 * preferences panel and continue the pending recommendation.
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
  hasPreference,
  lastProductsBlock,
  lastTurn,
  openDemo,
  preferenceValues,
  productCardIds,
  selectProfile,
  sendChat,
  waitForHealthReady,
} from "../fixtures/demo";

const DESKTOP = { width: 1440, height: 900 };
const VAGUE_QUERY = "I need some new headphones. Not sure what matters yet.";
const REPLY = "I prefer noise cancelling and wireless.";

test.describe("Scenario 3 — Clarification & Memory", () => {
  test("real clarification, real M9 write (added=2), refreshed preferences and continued recommendation", async ({
    page,
    request,
  }, testInfo) => {
    ensureArtifactDirs();
    const scenarioStarted = Date.now();
    await page.setViewportSize(DESKTOP);
    await openDemo(page);
    await waitForHealthReady(page, testInfo);

    const config = await bffGetJson(request, "/api/demo/config");
    const profile0057 = (config.body.profiles ?? []).find(
      (profile: any) => profile.profile_id === "profile_0057",
    );
    recordCheck(
      testInfo,
      "demo.config.write_profile",
      `profile_0057 memory_writes=${profile0057?.memory_writes} search_mode=${profile0057?.search_mode}`,
      Boolean(profile0057?.memory_writes),
    );
    expect(profile0057?.memory_writes).toBe(true);

    // ---- switch to the real write-enabled demo profile (new real BFF session) ---------
    const session = await selectProfile(page, testInfo, /Memory write/);
    expect(session.profile_id).toBe("profile_0057");

    // ---- vague request → real clarification ------------------------------------------
    await sendChat(page, testInfo, VAGUE_QUERY);
    // The real search products are emitted before the clarification and carry the verbatim Gate
    // decision (NEED_USER_CLARIFICATION / insufficient_decision_criteria).
    const gateCard = page.getByTestId("gate-card").last();
    await expect(gateCard).toBeVisible({ timeout: 120_000 });
    const gateAction = await gateCard.getAttribute("data-gate-action");
    recordCheck(
      testInfo,
      "gate.needs_clarification",
      `real P16 gate data-gate-action=${gateAction}; card text: ${(await gateCard.innerText()).split("\n").slice(0, 3).join(" | ")}`,
      gateAction === "NEED_USER_CLARIFICATION",
    );
    expect(gateAction, "the vague query must produce a real clarification gate").toBe("NEED_USER_CLARIFICATION");
    const card = page.getByTestId("clarification-card").last();
    await expect(card).toBeVisible({ timeout: 120_000 });
    const question = await page.getByTestId("clarification-question").last().innerText();
    const clarificationId = await card.getAttribute("data-clarification-id");
    const cardText = await card.innerText();
    recordCheck(
      testInfo,
      "clarification.question",
      `real P16 question="${question.trim()}" reason pill + handle ${clarificationId} + "${cardText
        .split("\n")
        .find((line) => line.includes("memory writes")) ?? "memory writes line"}" visible`,
      question.trim().length > 10 &&
        /what matters most/i.test(question) &&
        /^clar_[0-9a-f]+$/i.test(String(clarificationId)) &&
        /memory writes enabled/i.test(cardText),
    );
    expect(question.trim().length).toBeGreaterThan(10);
    expect(question).toMatch(/what matters most for this purchase/i);
    expect(String(clarificationId)).toMatch(/^clar_[0-9a-f]+$/i);
    expect(cardText).toMatch(/memory writes enabled/i);
    expect(await page.getByTestId("memory-update").count()).toBe(0);

    // ---- real explicit user reply → real M9 write ------------------------------------
    const replyInput = page.getByTestId("clarification-input").last();
    await expect(replyInput).toBeEnabled({ timeout: 60_000 });
    await replyInput.fill(REPLY);
    const replyResponsePromise = page.waitForResponse(
      (response) =>
        response.url().includes("/api/clarification/reply") && response.request().method() === "POST",
      { timeout: 240_000 },
    );
    const replyStarted = Date.now();
    await page.getByTestId("clarification-submit").last().click();
    const replyResponse = await replyResponsePromise;
    const replyRoundTrip = Date.now() - replyStarted;
    const replyBody = await replyResponse.json();
    recordTiming(testInfo, "clarification.reply.round_trip", replyRoundTrip, "real POST /api/clarification/reply (M9 write + prefs refresh + resumed recommendation)");
    recordPayload(testInfo, "clarification-reply", {
      http_status: replyResponse.status(),
      turn_id: replyBody.turn_id,
      memory_update: replyBody.memory_update,
      active_preferences: (replyBody.preferences?.active ?? []).map((item: any) => ({
        value: item.value,
        polarity: item.polarity,
        kind: item.kind,
        status: item.status,
        source_text: item.source_text,
      })),
      steps: replyBody.steps,
      recommendation_ids: (replyBody.recommendation?.items ?? []).map((item: any) => item.product_id),
      error: replyBody.error,
    });

    expect(replyResponse.status()).toBe(200);
    expect(replyBody.error).toBeNull();
    const memoryUpdate = replyBody.memory_update;
    expect(memoryUpdate, "real M9 memory_update payload is required").not.toBeNull();
    recordCheck(
      testInfo,
      "memory.update.counts",
      `real M9 result added=${memoryUpdate.added} superseded=${memoryUpdate.superseded} removed=${memoryUpdate.removed} skipped_duplicates=${memoryUpdate.skipped_duplicates} already_processed=${memoryUpdate.already_processed} confirmation_source=${memoryUpdate.confirmation_source}`,
      memoryUpdate.added === 2 && memoryUpdate.confirmation_source === "web_ui_explicit_user_reply",
    );
    expect(
      memoryUpdate.added,
      `expected the P18 verified real M9 write (added=2), got ${JSON.stringify(memoryUpdate)}`,
    ).toBe(2);
    expect(memoryUpdate.confirmation_source).toBe("web_ui_explicit_user_reply");
    expect(memoryUpdate.already_processed).toBe(false);

    const replyValues = preferenceValues(replyBody.preferences);
    const replyHasWireless = hasPreference(replyValues, /wireless/);
    const replyHasNoise = hasPreference(replyValues, /noise cancelling/);
    recordCheck(
      testInfo,
      "memory.preferences.refreshed",
      `refreshed active preferences: ${JSON.stringify(replyValues)} (source=${replyBody.preferences?.audit_source})`,
      replyHasWireless && replyHasNoise,
    );
    expect(replyHasWireless).toBe(true);
    expect(replyHasNoise).toBe(true);

    const stepTools = (replyBody.steps ?? []).map((step: any) => step.tool);
    expect(stepTools).toContain("submit_clarification_reply");
    expect(stepTools).toContain("get_active_preferences");
    expect(replyBody.recommendation, "the pending query must continue to a real recommendation").not.toBeNull();
    const continuedIds = (replyBody.recommendation?.items ?? []).map((item: any) => item.product_id);
    expect(continuedIds.length).toBeGreaterThanOrEqual(1);
    recordCheck(
      testInfo,
      "continuation.tools_and_recommendation",
      `continuation tools=${JSON.stringify(stepTools)}; real recommendation items=${JSON.stringify(continuedIds)}`,
      stepTools.includes("submit_clarification_reply") && continuedIds.length >= 1,
    );

    // ---- DOM: real memory update + refreshed preferences blocks ----------------------
    const memoryUpdateDom = page.getByTestId("memory-update").last();
    await expect(memoryUpdateDom).toBeVisible({ timeout: 120_000 });
    const memoryUpdateText = await memoryUpdateDom.innerText();
    expect(memoryUpdateText).toMatch(/Added\s+2/);
    expect(memoryUpdateText).toContain("written");
    expect(memoryUpdateText).toContain("web_ui_explicit_user_reply");
    const preferencesRefreshed = page.getByTestId("preferences-refreshed").last();
    await expect(preferencesRefreshed).toBeVisible({ timeout: 60_000 });
    const refreshedText = await preferencesRefreshed.innerText();
    expect(refreshedText.toLowerCase()).toMatch(/wireless/);
    expect(refreshedText.toLowerCase()).toMatch(/noise[ _-]?cancell?ing/);
    recordCheck(
      testInfo,
      "dom.memory_update",
      `DOM memory-update block shows Added=2 + written + web_ui_explicit_user_reply; preferences-refreshed shows wireless and noise cancelling`,
      /Added\s+2/.test(memoryUpdateText) && /wireless/i.test(refreshedText) && /noise[ _-]?cancell?ing/i.test(refreshedText),
    );

    // ---- DOM: sidebar preferences panel refreshed from the real M9 state -------------
    const sidebarPrefs = page.getByTestId("preferences-panel");
    await expect(sidebarPrefs).toContainText(/wireless/i, { timeout: 120_000 });
    await expect(sidebarPrefs).toContainText(/noise[ _-]?cancell?ing/i);
    const activePreferenceItems = await sidebarPrefs.getByTestId("preference-active-item").count();
    expect(activePreferenceItems).toBeGreaterThanOrEqual(2);
    expect(await sidebarPrefs.innerText()).toContain(replyBody.preferences?.audit_source ?? "");
    recordCheck(
      testInfo,
      "dom.preferences_panel",
      `sidebar preferences panel shows ${activePreferenceItems} real active preference item(s) incl. wireless + noise cancelling; audit source "${replyBody.preferences?.audit_source}" visible`,
      activePreferenceItems >= 2,
    );

    // ---- network cross-check: GET /api/preferences reflects the same real M9 state ---
    const prefsApi = await bffGetJson(request, `/api/preferences?session_id=${session.session_id}`);
    expect(prefsApi.status).toBe(200);
    const apiValues = preferenceValues(prefsApi.body);
    recordCheck(
      testInfo,
      "network.preferences",
      `GET /api/preferences?session_id=${session.session_id} → HTTP ${prefsApi.status}, active=${JSON.stringify(apiValues)}`,
      hasPreference(apiValues, /wireless/) && hasPreference(apiValues, /noise cancelling/),
    );
    expect(hasPreference(apiValues, /wireless/)).toBe(true);
    expect(hasPreference(apiValues, /noise cancelling/)).toBe(true);

    // ---- DOM: continued recommendation after the reply ------------------------------
    const turn = lastTurn(page);
    await expect(turn).toContainText("Continued recommendation after your reply", { timeout: 120_000 });
    const continuation = lastProductsBlock(page);
    await expect(continuation).toBeVisible({ timeout: 60_000 });
    const continuationDomIds = await productCardIds(continuation);
    expect(continuationDomIds.length).toBeGreaterThanOrEqual(1);
    expect(continuationDomIds).toContain(continuedIds[0]);
    const continuationText = await continuation.innerText();
    expect(continuationText).toMatch(/Price unavailable \/ UNKNOWN/);
    recordCheck(
      testInfo,
      "dom.continuation_recommendation",
      `continued recommendation block renders ${continuationDomIds.length} real card(s) (${continuationDomIds.join(
        ", ",
      )}) matching the reply payload; Gate/UNKNOWN visible`,
      continuationDomIds.length >= 1 && continuationDomIds.includes(continuedIds[0]),
    );

    // Bring the real M9 update block into the acceptance capture (the transcript is scrollable).
    await memoryUpdateDom.scrollIntoViewIfNeeded();
    await page.waitForTimeout(200);
    const memoryInView = await memoryUpdateDom.isVisible();
    recordCheck(
      testInfo,
      "screenshot.memory_update_visible",
      `desktop-clarification-memory.png captures the real memory-update block (Added=2 macro visible before capture=${memoryInView})`,
      memoryInView,
    );
    await saveScreenshot(page, testInfo, "desktop-clarification-memory.png", DESKTOP);
    recordTiming(testInfo, "scenario3.total", Date.now() - scenarioStarted);
  });
});
