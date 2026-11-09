/**
 * Shared E2E fixtures + helpers for the AgentRec-X Web Demo.
 *
 * These helpers drive the real browser UI against the real local BFF/MCP/BGE/M9 stack. They add
 * no fixture data and never stub network responses: every wait is on a real DOM/network signal.
 */
import { expect, type APIRequestContext, type Locator, type Page, type TestInfo } from "@playwright/test";
import { BFF_BASE, recordCheck, recordTiming } from "./evidence";

/** Chat turns include a real BGE search + recommend + evidence enrichment; be generous. */
export const TURN_TIMEOUT_MS = 240_000;

export function lastTurn(page: Page): Locator {
  return page.getByTestId("turn").last();
}

export async function openDemo(page: Page): Promise<void> {
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("app-shell")).toBeVisible({ timeout: 120_000 });
}

/**
 * The header readiness chip only switches to `data-state=ready` after GET /api/health reports
 * `ready=true`, i.e. after both real MCP servers finished their real catalogue + BGE cold start.
 */
export async function waitForHealthReady(page: Page, testInfo: TestInfo): Promise<void> {
  const started = Date.now();
  await expect(page.locator('[data-testid="readiness-chip"][data-state="ready"]')).toBeVisible({
    timeout: 300_000,
  });
  await expect(page.getByTestId("composer-input")).toBeEnabled({ timeout: 60_000 });
  recordTiming(testInfo, "ui.ready.wait", Date.now() - started, "page load → header readiness chip ready");
  recordCheck(
    testInfo,
    "health.ready",
    `readiness chip reports data-state=ready; composer enabled`,
    true,
  );
}

/**
 * Wait until the newest chat turn is really finished.
 *
 * Real completion signals (all must hold):
 *  - the expected new turn rendered (real `turn` elements),
 *  - the last turn reached a terminal view: `turn-meta` (done event), `error-bubble`, or
 *    `clarification-card`,
 *  - no tool row is still `running`, no streaming `activity-line`,
 *  - the composer is re-enabled (the app disables it for the whole SSE stream).
 *
 * The budget is large on purpose: each MCP server lazy-loads the BGE encoder on its *first*
 * search (~45-55 s measured on the real cold demo), and we must never cut a real turn short.
 */
export async function waitForTurnIdle(
  page: Page,
  timeout = TURN_TIMEOUT_MS,
  expectedTurns?: number,
): Promise<void> {
  await expect
    .poll(
      async () => {
        const turns = page.getByTestId("turn");
        const turnCount = await turns.count();
        if (expectedTurns != null && turnCount < expectedTurns) return `turns=${turnCount}/${expectedTurns}`;
        if (turnCount === 0) return "no-turn";
        const last = turns.last();
        const terminalMarkers =
          (await last.getByTestId("turn-meta").count()) +
          (await last.getByTestId("error-bubble").count()) +
          (await last.getByTestId("clarification-card").count());
        if (terminalMarkers === 0) return "turn-not-terminal";
        const runningTools = await last
          .locator('[data-testid="tool-row"][data-status="running"]')
          .count();
        if (runningTools > 0) return `running-tools=${runningTools}`;
        const activityLines = await page.getByTestId("activity-line").count();
        if (activityLines > 0) return `activity-line=${activityLines}`;
        if (await page.getByTestId("composer-input").isDisabled()) return "composer-disabled";
        return "idle";
      },
      { timeout, intervals: [250, 500, 1000, 2000, 3000] },
    )
    .toBe("idle");
}

/** Send a composer message and assert the real SSE response status + content type. */
export async function sendChat(
  page: Page,
  testInfo: TestInfo,
  text: string,
): Promise<{ status: number; contentType: string }> {
  const input = page.getByTestId("composer-input");
  await expect(input).toBeEnabled({ timeout: 60_000 });
  const turnsBefore = await page.getByTestId("turn").count();
  await input.fill(text);
  const started = Date.now();
  const responsePromise = page.waitForResponse(
    (response) =>
      response.url().includes("/api/chat/stream") && response.request().method() === "POST",
    { timeout: 120_000 },
  );
  await page.getByTestId("composer-send").click();
  const response = await responsePromise;
  const contentType = response.headers()["content-type"] ?? "";
  const ok = response.status() === 200 && contentType.includes("text/event-stream");
  recordCheck(
    testInfo,
    "chat.sse.response",
    `POST /api/chat/stream → HTTP ${response.status()} content-type=${contentType}`,
    ok,
  );
  await waitForTurnIdle(page, TURN_TIMEOUT_MS, turnsBefore + 1);
  recordTiming(testInfo, "chat.turn.latency", Date.now() - started, `real local turn: ${JSON.stringify(text.slice(0, 80))}`);
  if (!ok) throw new Error(`chat stream failed: HTTP ${response.status()} (${contentType})`);
  return { status: response.status(), contentType };
}

/** Select a demo profile in the real sidebar selector (creates a real new BFF session). */
export async function selectProfile(
  page: Page,
  testInfo: TestInfo,
  profileLabelPattern: RegExp,
): Promise<{ session_id: string; profile_id: string }> {
  const selector = page.getByTestId("profile-selector");
  await expect(selector).toBeVisible({ timeout: 60_000 });
  const responsePromise = page.waitForResponse(
    (response) => response.url().endsWith("/api/sessions") && response.request().method() === "POST",
    { timeout: 60_000 },
  );
  await selector.getByRole("button").filter({ hasText: profileLabelPattern }).click();
  const response = await responsePromise;
  const view = (await response.json()) as { session_id: string; profile_id: string };
  await expect(page.getByTestId("sidebar")).toContainText(view.session_id, { timeout: 60_000 });
  recordCheck(
    testInfo,
    "session.profile.select",
    `POST /api/sessions → HTTP ${response.status()} session=${view.session_id} profile=${view.profile_id}`,
    response.status() === 200,
  );
  recordCheck(
    testInfo,
    "session.profile.bound",
    `sidebar shows the new session id; profile_id=${view.profile_id}`,
    view.profile_id.length > 0,
  );
  return view;
}

/** Open a right-panel tab on desktop (the panel is a column at ≥1280 px). */
export async function openPanelTab(page: Page, tab: "details" | "evidence" | "comparison"): Promise<void> {
  const tabButton = page.getByTestId(`panel-tab-${tab}`);
  await expect(tabButton).toBeVisible({ timeout: 60_000 });
  await tabButton.click();
}

/** Fetch JSON from the real BFF with the Playwright API request context (no mocking). */
export async function bffGetJson(api: APIRequestContext, pathname: string): Promise<{ status: number; body: any }> {
  const response = await api.get(`${BFF_BASE}${pathname}`, { timeout: 30_000 });
  let body: any = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  return { status: response.status(), body };
}

export async function bffPostJson(
  api: APIRequestContext,
  pathname: string,
  data: unknown,
): Promise<{ status: number; body: any }> {
  const response = await api.post(`${BFF_BASE}${pathname}`, { data, timeout: 60_000 });
  let body: any = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  return { status: response.status(), body };
}

/** Case-insensitive normalisation for real M9 preference values ("noise cancelling" etc.). */
export function normalisePreferenceValue(value: unknown): string {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[_\-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function preferenceValues(state: any): string[] {
  const active = Array.isArray(state?.active) ? state.active : [];
  return active.map((item: any) => normalisePreferenceValue(item?.value));
}

/** Match a human-readable preference token against real preference values. */
export function hasPreference(values: string[], wanted: RegExp): boolean {
  return values.some((value) => wanted.test(value));
}

/** The last completed turn's products block (real BFF `products` event payload). */
export function lastProductsBlock(page: Page): Locator {
  return page.getByTestId("products-block").last();
}

export async function productCardIds(block: Locator): Promise<string[]> {
  return block
    .getByTestId("product-card")
    .evaluateAll((cards) => cards.map((card) => card.getAttribute("data-product-id") ?? ""));
}
