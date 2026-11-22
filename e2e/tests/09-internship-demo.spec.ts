/**
 * Internship demo acceptance: the exact laptop multi-round path plus the two opt-in
 * personalization features through the real browser UI.
 *
 * This test is intentionally gated: the default launcher is BGE + flags OFF + strict prices.
 * Start the demo with the opt-in demo environment documented in docs/development/INTERNSHIP_DEMO.md,
 * then run it against the live stack.  It never mocks the BFF or fabricates a product.
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
  lastProductsBlock,
  lastTurn,
  openDemo,
  selectProfile,
  sendChat,
  waitForHealthReady,
} from "../fixtures/demo";

const DESKTOP = { width: 1440, height: 900 };
const LAPTOP_QUERY = "推荐一台笔记本";
const USE_CASE_REPLY = "游戏";
const BUDGET_REPLY = "8000 元";
const MEMORY_QUERY = "studio headphones";
const MEMORY_TEXT = "I prefer USB.";

async function replyToClarification(page: any, testInfo: any, text: string) {
  const input = page.getByTestId("clarification-input").last();
  await expect(input).toBeEnabled({ timeout: 60_000 });
  await input.fill(text);
  const responsePromise = page.waitForResponse(
    (response: any) =>
      response.url().includes("/api/clarification/reply") &&
      response.request().method() === "POST",
    { timeout: 240_000 },
  );
  const started = Date.now();
  await page.getByTestId("clarification-submit").last().click();
  const response = await responsePromise;
  const body = await response.json();
  recordTiming(testInfo, "clarification.reply", Date.now() - started, `real reply ${JSON.stringify(text)}`);
  expect(response.status()).toBe(200);
  return body;
}

async function waitForNewClarification(page: any, previousId: string) {
  await expect
    .poll(
      async () => {
        const card = page.getByTestId("clarification-card").last();
        if (!(await card.count())) return null;
        return card.getAttribute("data-clarification-id");
      },
      { timeout: 120_000, intervals: [250, 500, 1000, 2000] },
    )
    .not.toBe(previousId);
  return page.getByTestId("clarification-card").last();
}

test.describe("Internship demo — laptop M4 + opt-in personalization", () => {
  test.beforeEach(async ({ request }) => {
    const health = await bffGetJson(request, "/api/health");
    const enabled =
      health.body?.features?.memory_aware_retrieval === true &&
      health.body?.features?.behavior_aware_rerank === true &&
      health.body?.price_mode === "demo_historical_fx";
    test.skip(
      !enabled,
      "requires AGENTRECX_MEMORY_AWARE_RETRIEVAL=1, AGENTRECX_BEHAVIOR_AWARE_RERANK=1 and AGENTRECX_PRICE_MODE=demo_historical_fx",
    );
  });

  test("laptop ASK x2 with demo FX budget gate, then real memory and behavior evidence", async ({
    page,
    request,
  }, testInfo) => {
    ensureArtifactDirs();
    await page.setViewportSize(DESKTOP);
    await openDemo(page);
    await waitForHealthReady(page, testInfo);

    await page.waitForTimeout(750);
    let session = await selectProfile(page, testInfo, /Memory write/);
    if (session.profile_id !== "profile_0057") {
      // The first POST /api/sessions can still be the page's initial Demo A session; wait
      // for that request to settle and explicitly select Demo B again.
      await page.waitForTimeout(750);
      session = await selectProfile(page, testInfo, /Memory write/);
    }
    expect(session.profile_id).toBe("profile_0057");

    // ---- Scenario 1: real M4 ASK -> real reply -> ASK2 -> real final ----------------
    await sendChat(page, testInfo, LAPTOP_QUERY);
    const firstCard = page.getByTestId("clarification-card").last();
    await expect(firstCard).toBeVisible({ timeout: 120_000 });
    const firstId = String(await firstCard.getAttribute("data-clarification-id"));
    expect(firstId).toMatch(/^clar_[0-9a-f]+$/i);
    expect(await page.getByTestId("clarification-question").last().innerText()).toMatch(
      /what matters most|office|gaming|用途|场景|办公|游戏|最重要/i,
    );

    const firstReply = await replyToClarification(page, testInfo, USE_CASE_REPLY);
    // The BFF returns an empty Gate envelope carrying ASK2; it is not a product recommendation.
    expect(firstReply.recommendation?.gate?.action).toBe("NEED_USER_CLARIFICATION");
    expect(firstReply.recommendation?.items ?? []).toHaveLength(0);
    expect(firstReply.recommendation?.recommended_product_id).toBeNull();
    const secondCard = await waitForNewClarification(page, firstId);
    const secondId = String(await secondCard.getAttribute("data-clarification-id"));
    expect(secondId).not.toBe(firstId);
    expect(await page.getByTestId("clarification-question").last().innerText()).toMatch(
      /budget|预算|价格|spend/i,
    );

    const finalReply = await replyToClarification(page, testInfo, BUDGET_REPLY);
    expect(finalReply.recommendation).not.toBeNull();
    expect(finalReply.error).toBeNull();
    expect(finalReply.task_requirements?.budget?.max_amount).toBe(8000);
    expect(finalReply.task_requirements?.budget?.currency).toBe("CNY");

    const finalBlock = lastProductsBlock(page);
    await expect(finalBlock).toBeVisible({ timeout: 120_000 });
    await expect(page.getByTestId("gate-card").last()).toHaveAttribute(
      "data-gate-action",
      "ALLOW_FINALIZE",
      { timeout: 120_000 },
    );
    const finalText = await finalBlock.innerText();
    expect(finalText).toContain("¥6,293.00 CNY");
    expect(finalText).toMatch(/演示参考价|historical demo|reference/i);
    expect(finalText).not.toMatch(/\$899\.00 USD/);

    recordPayload(testInfo, "laptop-final", {
      recommendation: (finalReply.recommendation?.items ?? []).map((item: any) => ({
        product_id: item.product_id,
        price: item.price,
      })),
      requirements: finalReply.task_requirements,
      gate: finalReply.recommendation?.gate,
    });
    recordCheck(
      testInfo,
      "laptop.m4.final",
      `final items=${(finalReply.recommendation?.items ?? []).length}; price=¥6,293.00 CNY; budget=8000 CNY`,
      (finalReply.recommendation?.items ?? []).length >= 1,
    );

    // ---- Scenario 2: save a real M9 preference, then observe formal memory retrieval ----
    const memoryLab = page.getByTestId("memory-lab");
    await expect(memoryLab).toBeVisible({ timeout: 60_000 });
    await page.getByTestId("memory-lab-input").fill(MEMORY_TEXT);
    const rememberPromise = page.waitForResponse(
      (response: any) =>
        response.url().includes("/api/memory/remember") &&
        response.request().method() === "POST",
      { timeout: 120_000 },
    );
    await page.getByTestId("memory-lab-save").click();
    const remembered = await rememberPromise;
    expect(remembered.status()).toBe(200);
    const rememberedBody = await remembered.json();
    const added = rememberedBody.update?.added;
    const addedCount = Array.isArray(added) ? added.length : Number(added ?? 0);
    const activeValues = (rememberedBody.preferences?.active ?? []).map((item: any) =>
      String(item?.value ?? "").toLowerCase(),
    );
    expect(addedCount >= 1 || activeValues.includes("usb")).toBe(true);
    recordPayload(testInfo, "memory-lab", {
      update: rememberedBody.update,
      active: rememberedBody.preferences?.active,
    });

    await sendChat(page, testInfo, MEMORY_QUERY);
    const memoryBlock = lastProductsBlock(page);
    await expect(memoryBlock.getByTestId("personalization-evidence")).toBeVisible({
      timeout: 120_000,
    });
    const evidence = memoryBlock.getByTestId("personalization-evidence");
    const evidenceText = await evidence.innerText();
    expect(evidenceText).toMatch(/Memory retrieval/i);
    expect(evidenceText).toMatch(/new/i);
    expect(evidenceText).toContain("Behavior rerank");
    recordCheck(
      testInfo,
      "memory.ui.evidence",
      `browser-visible personalization evidence: ${evidenceText.split("\n").slice(0, 4).join(" | ")}`,
      /Memory retrieval/i.test(evidenceText) && /new/i.test(evidenceText),
    );

    // ---- Scenario 3: real Like in the UI -> next formal recommendation ----------------
    const firstCardInBlock = memoryBlock.getByTestId("product-card").nth(2);
    await expect(firstCardInBlock).toBeVisible({ timeout: 60_000 });
    const productId = await firstCardInBlock.getAttribute("data-product-id");
    expect(productId).toBeTruthy();
    const likeButton = firstCardInBlock.getByTitle(/^Like this item for this session/i);
    const eventPromise = page.waitForResponse(
      (response: any) =>
        response.url().includes("/api/behavior/events") &&
        response.request().method() === "POST",
      { timeout: 60_000 },
    );
    await likeButton.click();
    const eventResponse = await eventPromise;
    expect(eventResponse.status()).toBe(200);
    const eventBody = await eventResponse.json();
    expect((eventBody.accepted ?? []).length).toBeGreaterThanOrEqual(1);
    expect(JSON.stringify(eventBody.accepted)).toContain(String(productId));
    recordPayload(testInfo, "behavior-like", {
      accepted: eventBody.accepted,
      duplicates: eventBody.duplicates,
    });

    // The async flush is real; wait until the BFF profile contains the event before asking
    // the formal recommendation path to consume it.
    await expect
      .poll(
        async () => {
          const profile = await bffGetJson(
            request,
            `/api/behavior/profile?session_id=${encodeURIComponent(session.session_id)}`,
          );
          return (profile.body?.product_interests ?? []).length;
        },
        { timeout: 30_000, intervals: [250, 500, 1000] },
      )
      .toBeGreaterThanOrEqual(1);

    await sendChat(page, testInfo, MEMORY_QUERY);
    const behaviorBlock = lastProductsBlock(page);
    await expect(behaviorBlock.getByTestId("personalization-evidence")).toBeVisible({
      timeout: 120_000,
    });
    const behaviorText = await behaviorBlock.getByTestId("personalization-evidence").innerText();
    expect(behaviorText).toMatch(/Behavior rerank/i);
    expect(behaviorText).toMatch(/top\s+\+/i);
    recordCheck(
      testInfo,
      "behavior.ui.rerank",
      `browser-visible behavior contribution after real Like: ${behaviorText.split("\n").slice(0, 5).join(" | ")}`,
      /Behavior rerank/i.test(behaviorText) && /top\s+\+/i.test(behaviorText),
    );
    await expect(lastTurn(page).getByTestId("products-block")).toBeVisible();
    await behaviorBlock.getByTestId("personalization-evidence").scrollIntoViewIfNeeded();
    await page.waitForTimeout(250);
    await saveScreenshot(page, testInfo, "desktop-internship-personalization.png", DESKTOP);
  });
});
