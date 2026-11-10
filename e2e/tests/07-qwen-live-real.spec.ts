/**
 * REAL Qwen Base Live browser acceptance (GPU service, no doubles, no replay).
 *
 * The test is intentionally strict: it requires GET /api/health to report
 * qwen_live.available=true after a real /plan health probe, then drives the real browser
 * through the existing BFF → P17 MCP run_agent_turn → real Qwen3.5-9B Base planner →
 * ElectronicsPlannerExecutor → real catalogue/BGE/Gate/P15 path.
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
  lastTurn,
  openDemo,
  sendChat,
  waitForHealthReady,
} from "../fixtures/demo";

const DESKTOP = { width: 1440, height: 900 };
const RECOMMEND_QUERY = "I need earbuds I can use while working out.";
const NONFINAL_QUERY = "I need wireless earbuds under $1.";

async function selectQwenLive(page: import("@playwright/test").Page, testInfo: import("@playwright/test").TestInfo) {
  await page.setViewportSize(DESKTOP);
  await openDemo(page);
  await waitForHealthReady(page, testInfo);
  const qwen = page.getByTestId("mode-qwen-live");
  await expect(qwen).toBeEnabled({ timeout: 60_000 });
  await expect(qwen).toHaveAttribute("data-state", "available");
  await qwen.click();
  await expect(qwen).toHaveAttribute("aria-pressed", "true", { timeout: 30_000 });
  await expect(page.getByTestId("qwen-live-badge")).toBeVisible({ timeout: 30_000 });
}

async function assertRealPlannerTurn(turn: import("@playwright/test").Locator, testInfo: import("@playwright/test").TestInfo) {
  await expect(turn.getByTestId("turn-planner")).toHaveAttribute("data-planner-kind", "qwen_base", {
    timeout: 30_000,
  });
  await expect(turn.getByTestId("turn-planner")).toContainText("Qwen Base Live");
  const tools = await turn
    .locator('[data-testid="tool-row"]')
    .evaluateAll((rows) => rows.map((row) => row.getAttribute("data-tool") ?? ""));
  expect(tools).toContain("qwen_planner");
  expect(tools.some((tool) => tool === "planner_action")).toBeTruthy();
  expect(tools.some((tool) => tool.startsWith("retrieve_"))).toBeTruthy();
  recordCheck(
    testInfo,
    "qwen.real.planner",
    `real planner badge data-planner-kind=qwen_base; real tool rows=${JSON.stringify(tools)}`,
    tools.includes("qwen_planner") && tools.some((tool) => tool.startsWith("finalize_")),
  );
}

test.describe("REAL Qwen Base Live (GPU)", () => {
  test("real Qwen planner multi-step recommendation + products/evidence/Gate + screenshot", async ({
    page,
    request,
  }, testInfo) => {
    ensureArtifactDirs();
    const started = Date.now();
    await selectQwenLive(page, testInfo);

    const health = await bffGetJson(request, "/api/health");
    expect(health.status).toBe(200);
    expect(health.body.qwen_live?.available).toBe(true);
    expect(health.body.qwen_live?.status).toBe("available");
    const modes = health.body.modes ?? {};
    expect(modes.qwen_live?.available).toBe(true);
    expect(modes.qwen_live?.badge).toBe("Live Qwen3.5-9B Base");
    recordPayload(testInfo, "real-qwen-health", {
      qwen_live: health.body.qwen_live,
      qwen_mode: modes.qwen_live,
      tools: (health.body.servers ?? []).map((server: any) => ({
        profile_id: server.profile_id,
        tools: server.tools,
      })),
    });
    recordCheck(
      testInfo,
      "qwen.real.health",
      `GET /api/health qwen_live.status=${health.body.qwen_live?.status} available=${health.body.qwen_live?.available} planner_action=${health.body.qwen_live?.planner_action} latency=${health.body.qwen_live?.latency_ms}ms; tools=${JSON.stringify(
        (health.body.servers ?? []).map((server: any) => (server.tools ?? []).length),
      )}`,
      health.body.qwen_live?.available === true && health.body.qwen_live?.planner_action != null,
    );

    await sendChat(page, testInfo, RECOMMEND_QUERY);
    const turn = lastTurn(page);
    await assertRealPlannerTurn(turn, testInfo);

    const gate = turn.getByTestId("gate-card");
    await expect(gate).toBeVisible({ timeout: 30_000 });
    const gateAction = await gate.getAttribute("data-gate-action");
    expect(gateAction).toBe("ALLOW_FINALIZE");
    await expect(gate).toContainText("evidence_sufficient");
    recordCheck(
      testInfo,
      "qwen.real.gate",
      `real Qwen turn reached Gate data-gate-action=${gateAction}; Gate text is real`,
      gateAction === "ALLOW_FINALIZE",
    );

    const productCards = turn.getByTestId("product-card");
    await expect(productCards.first()).toBeVisible({ timeout: 60_000 });
    const productCount = await productCards.count();
    expect(productCount).toBeGreaterThan(0);
    const blockText = await turn.getByTestId("products-block").innerText();
    expect(blockText).toContain("Price unavailable / UNKNOWN");
    recordCheck(
      testInfo,
      "qwen.real.products",
      `real products rendered: ${productCount} cards; UNKNOWN price block present; Gate selected visible`,
      productCount > 0 && blockText.includes("Price unavailable / UNKNOWN"),
    );

    // Evidence is emitted for the selected product by the Qwen Live BFF flow.
    await turn.getByTestId("product-card").first().getByRole("button", { name: "View evidence" }).click();
    const evidencePanel = page.getByTestId("evidence-panel");
    await expect(evidencePanel).toBeVisible({ timeout: 120_000 });
    await expect(page.getByTestId("evidence-item").first()).toBeVisible({ timeout: 120_000 });
    const evidenceCount = await page.getByTestId("evidence-item").count();
    expect(evidenceCount).toBeGreaterThanOrEqual(1);
    recordCheck(
      testInfo,
      "qwen.real.evidence",
      `P15 evidence drawer rendered ${evidenceCount} real evidence items from the selected product`,
      evidenceCount >= 1,
    );

    recordTiming(testInfo, "qwen.real.turn", Date.now() - started, "page load + Qwen turn + evidence");
    await saveScreenshot(page, testInfo, "qwen-live-real.png", DESKTOP);
  });

  test("real Qwen non-FINALIZE: Gate blocks finalize; no fabricated recommendation", async ({
    page,
    request,
  }, testInfo) => {
    ensureArtifactDirs();
    const started = Date.now();
    await selectQwenLive(page, testInfo);
    const health = await bffGetJson(request, "/api/health");
    expect(health.body.qwen_live?.available).toBe(true);

    await sendChat(page, testInfo, NONFINAL_QUERY);
    const turn = lastTurn(page);
    await assertRealPlannerTurn(turn, testInfo);
    const gate = turn.getByTestId("gate-card");
    await expect(gate).toBeVisible({ timeout: 30_000 });
    const gateAction = await gate.getAttribute("data-gate-action");
    expect(gateAction).not.toBe("ALLOW_FINALIZE");
    recordCheck(
      testInfo,
      "qwen.real.non_finalize",
      `real Qwen planner executed; Gate did not allow finalize (data-gate-action=${gateAction}); no fabricated recommendation`,
      gateAction !== "ALLOW_FINALIZE",
    );
    recordTiming(testInfo, "qwen.real.nonfinal.turn", Date.now() - started, "Qwen non-FINALIZE turn");
    await saveScreenshot(page, testInfo, "qwen-live-nonfinal.png", DESKTOP);
  });
});
