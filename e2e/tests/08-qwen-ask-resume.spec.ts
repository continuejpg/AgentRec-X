/**
 * Real Qwen ASK → single real browser reply → task-scoped constraint priority → M9 history
 * retained → Qwen resume → fresh candidate re-evaluation → Gate final consistency.
 *
 * The isolated profile_0056_ask has historical wired/wireless conflict.  "Please avoid wired
 * headphones" is a current-purchase exclusion, not a permanent M9 retraction: the old wired
 * history must remain active in the audit trail, but the resumed task must exclude a product
 * with trustworthy wired catalogue evidence.  No second "Forget" message is required.
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
  hasPreference,
  lastTurn,
  openDemo,
  preferenceValues,
  selectProfile,
  sendChat,
  waitForHealthReady,
  waitForTurnIdle,
} from "../fixtures/demo";

const DESKTOP = { width: 1440, height: 900 };
const QUERY = "I need headphones. Can you recommend something suitable for me?";
const FIRST_REPLY = "For this purchase, I prefer wireless headphones for commuting. Please avoid wired headphones.";

test.describe("Qwen ASK → task constraint → resume (real GPU)", () => {
  test("one real reply changes current task, preserves history and prevents explicit-wired final", async ({
    page,
    request,
  }, testInfo) => {
    ensureArtifactDirs();
    await page.setViewportSize(DESKTOP);
    await openDemo(page);
    await waitForHealthReady(page, testInfo);

    const health = await bffGetJson(request, "/api/health");
    expect(health.body.qwen_live?.available).toBe(true);
    const config = await bffGetJson(request, "/api/demo/config");
    const testProfile = (config.body.profiles ?? []).find(
      (profile: any) => profile.profile_id === "profile_0056_ask",
    );
    expect(testProfile, "isolated ASK test profile must be enabled").toBeTruthy();
    expect(testProfile.memory_writes).toBe(true);
    const session = await selectProfile(page, testInfo, /ASK Test/);
    expect(session.profile_id).toBe("profile_0056_ask");

    const qwenLive = page.getByTestId("mode-qwen-live");
    await expect(qwenLive).toBeEnabled({ timeout: 60_000 });
    await qwenLive.click();
    await expect(qwenLive).toHaveAttribute("aria-pressed", "true");

    const scenarioStarted = Date.now();
    const prefsBefore = await bffGetJson(request, `/api/preferences?session_id=${session.session_id}`);
    const beforeValues = preferenceValues(prefsBefore.body);
    expect(hasPreference(beforeValues, /wired/)).toBe(true);
    expect(hasPreference(beforeValues, /wireless/)).toBe(true);

    // ---- real Qwen plan must produce a real ASK ---------------------------------------
    await sendChat(page, testInfo, QUERY);
    await waitForTurnIdle(page);
    const firstCard = page.getByTestId("clarification-card").last();
    await expect(firstCard).toBeVisible({ timeout: 180_000 });
    const firstId = (await firstCard.getAttribute("data-clarification-id")) ?? "";
    expect(firstId.length).toBeGreaterThan(0);
    const firstQuestion = await page.getByTestId("clarification-question").last().innerText();
    expect(firstQuestion).toMatch(/wired/i);
    expect(firstQuestion).toMatch(/wireless/i);
    await expect(lastTurn(page).getByTestId("turn-planner")).toHaveAttribute(
      "data-planner-kind",
      "qwen_base",
    );
    recordCheck(
      testInfo,
      "qwen_fix.real_ask",
      `real Qwen planner produced an ASK; handle=${firstId}; question="${firstQuestion}"`,
      /wired/i.test(firstQuestion) && /wireless/i.test(firstQuestion),
    );
    await saveScreenshot(page, testInfo, "qwen-fix-first-ask.png", DESKTOP);

    // ---- no-reply safety --------------------------------------------------------------
    const prefsAfterAsk = await bffGetJson(
      request,
      `/api/preferences?session_id=${session.session_id}`,
    );
    expect(preferenceValues(prefsAfterAsk.body)).toEqual(beforeValues);
    await expect(page.getByTestId("memory-update")).toHaveCount(0);
    const askGate = page.getByTestId("gate-card").last();
    await expect(askGate).toBeVisible({ timeout: 60_000 });
    expect(await askGate.getAttribute("data-gate-action")).toBe("NEED_USER_CLARIFICATION");
    recordCheck(
      testInfo,
      "qwen_fix.no_reply_safety",
      "after ASK and before any reply: no memory-update block, Gate=NEED_USER_CLARIFICATION, preference values unchanged",
      true,
    );

    // ---- one real browser-authored reply ---------------------------------------------
    const replyResponsePromise = page.waitForResponse(
      (response) =>
        response.url().includes("/api/clarification/reply") &&
        response.request().method() === "POST",
      { timeout: 240_000 },
    );
    await page.getByTestId("clarification-input").last().fill(FIRST_REPLY);
    await page.getByTestId("clarification-submit").last().click();
    const replyResponse = await replyResponsePromise;
    const replyBody = await replyResponse.json();
    expect(replyResponse.status()).toBe(200);
    expect(replyBody.memory_update?.confirmation_source).toBe("web_ui_explicit_user_reply");
    expect(replyBody.memory_update?.added).toBe(2);
    expect(replyBody.memory_update?.removed).toBe(0);
    expect(replyBody.clarification).toBeNull();

    const taskConstraints = replyBody.task_constraints ?? {};
    const hardExclusions = taskConstraints.hard_exclusions ?? [];
    expect(hardExclusions.some((item: any) => String(item.attribute) === "wired")).toBe(true);
    expect(hardExclusions.some((item: any) => String(item.source_text).toLowerCase().includes("avoid wired"))).toBe(true);
    const reevaluation = replyBody.candidate_reevaluation ?? {};
    expect(Number(reevaluation?.task_constraint_status?.violations ?? 0)).toBeGreaterThan(0);
    expect(Array.isArray(reevaluation?.excluded_candidate_ids)).toBe(true);
    recordPayload(testInfo, "qwen-fix-first-reply", replyBody);
    recordCheck(
      testInfo,
      "qwen_fix.task_constraints",
      `M9 added=${replyBody.memory_update?.added} removed=${replyBody.memory_update?.removed}; task hard exclusions=${JSON.stringify(hardExclusions.map((i: any) => [i.attribute, i.source_text]))}`,
      hardExclusions.length > 0 && replyBody.clarification == null,
    );
    recordCheck(
      testInfo,
      "qwen_fix.reevaluation",
      `resumed candidate re-evaluation: input=${reevaluation?.input_candidate_count}; eligible=${reevaluation?.eligible_count}; excluded=${JSON.stringify(reevaluation?.excluded_candidate_ids)}; violations=${reevaluation?.task_constraint_status?.violations}`,
      Number(reevaluation?.eligible_count ?? 0) > 0 &&
        Number(reevaluation?.task_constraint_status?.violations ?? 0) > 0,
    );
    await expect(page.getByTestId("memory-update").last()).toBeVisible({ timeout: 60_000 });
    await expect(page.getByTestId("memory-update").last()).toContainText(/Added\s*2/);
    // The answered clarification card remains as an audit row, but its reply control is gone;
    // there must be no second Qwen question/handle.
    await expect(page.getByTestId("clarification-card").last()).toHaveAttribute(
      "data-clarification-id",
      firstId,
    );
    await expect(page.getByTestId("clarification-input")).toHaveCount(0);
    await saveScreenshot(page, testInfo, "qwen-fix-after-reply.png", DESKTOP);

    // The old wired preference must still be in the M9 active/audit state: current-task
    // exclusion is not a permanent retraction.
    const prefsAfter = await bffGetJson(
      request,
      `/api/preferences?session_id=${session.session_id}`,
    );
    const active = Array.isArray(prefsAfter.body.active) ? prefsAfter.body.active : [];
    const retainedWired = active.filter(
      (item: any) => String(item.value ?? "").toLowerCase() === "wired" &&
        String(item.polarity ?? "").toLowerCase() === "prefer",
    );
    expect(retainedWired).toHaveLength(1);
    expect(preferenceValues(prefsAfter.body)).toEqual(
      expect.arrayContaining([expect.stringMatching(/wireless/)]),
    );
    recordCheck(
      testInfo,
      "qwen_fix.history_retained",
      `M9 active after one reply=${JSON.stringify(active.map((item: any) => [item.value, item.polarity]))}; historical wired was not permanantly deleted`,
      retainedWired.length === 1,
    );

    // ---- Qwen resume → Gate → final product consistency ------------------------------
    await expect
      .poll(async () => {
        const turn = lastTurn(page);
        if ((await turn.getByTestId("turn-meta").count()) === 0) return "no-meta";
        if ((await turn.getByTestId("products-block").count()) === 0) return "no-products";
        return "final";
      }, { timeout: 240_000, intervals: [300, 500, 1000, 2000] })
      .toBe("final");
    await expect(lastTurn(page).getByTestId("turn-planner")).toHaveAttribute(
      "data-planner-kind",
      "qwen_base",
    );
    const gate = lastTurn(page).getByTestId("gate-card").last();
    await expect(gate).toBeVisible({ timeout: 60_000 });
    const gateAction = await gate.getAttribute("data-gate-action");
    expect(gateAction).toBe("ALLOW_FINALIZE");
    const productCards = lastTurn(page).getByTestId("product-card");
    expect(await productCards.count()).toBeGreaterThan(0);
    const selectedProductId = await productCards.first().getAttribute("data-product-id");
    expect(selectedProductId, "final selected product id must be present").toBeTruthy();

    const details = await bffPostJson(request, "/api/products/details", {
      session_id: session.session_id,
      product_ref: selectedProductId,
    });
    expect(details.status).toBe(200);
    const evidenceRows = Array.isArray(details.body.evidence) ? details.body.evidence : [];
    const wiredEvidence = evidenceRows.filter(
      (row: any) => String(row.attribute ?? "").toLowerCase() === "wired",
    );
    expect(wiredEvidence, "final product must not have trustworthy wired evidence").toHaveLength(0);
    recordCheck(
      testInfo,
      "qwen_fix.final_product_not_wired",
      `final selected=${selectedProductId}; title=${details.body.title}; wired evidence rows=${wiredEvidence.length}; gate=${gateAction}`,
      wiredEvidence.length === 0,
    );
    recordCheck(
      testInfo,
      "qwen_fix.gate_final",
      `real resumed Qwen turn reached Gate action=${gateAction}; product=${selectedProductId}; products=${await productCards.count()}`,
      gateAction === "ALLOW_FINALIZE",
    );
    recordTiming(testInfo, "qwen_fix.total", Date.now() - scenarioStarted,
      "Qwen ASK → one reply → task constraints → re-evaluation → Gate");
    await saveScreenshot(page, testInfo, "qwen-fix-resume.png", DESKTOP);
  });
});
