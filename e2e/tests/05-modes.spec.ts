/**
 * Modes acceptance: CPU Live is the default *real* mode; Qwen Replay is visibly labelled
 * "Historical Replay — Not a Live Model Call" and read-only; Qwen Live is disabled with its real
 * disabled_reason. Data comes from the real GET /api/health + GET /api/replay.
 */
import { expect, test } from "@playwright/test";
import { ensureArtifactDirs, recordCheck, recordPayload, saveScreenshot } from "../fixtures/evidence";
import { bffGetJson, openDemo, waitForHealthReady } from "../fixtures/demo";

const DESKTOP = { width: 1440, height: 900 };
const BADGE = "Historical Replay — Not a Live Model Call";

test.describe("Modes — CPU Live default, Qwen Replay labelled, Qwen Live disabled", () => {
  test("replay mode is visibly historical and never presented as a live model call", async ({
    page,
    request,
  }, testInfo) => {
    ensureArtifactDirs();
    await page.setViewportSize(DESKTOP);
    await openDemo(page);
    await waitForHealthReady(page, testInfo);

    // ---- real mode metadata from the BFF --------------------------------------------
    const health = await bffGetJson(request, "/api/health");
    expect(health.status).toBe(200);
    const modes = health.body.modes ?? {};
    recordPayload(testInfo, "modes", modes);
    expect(modes.qwen_live?.available).toBe(false);
    expect(modes.qwen_replay?.available).toBe(true);
    expect(modes.qwen_replay?.badge).toBe(BADGE);
    recordCheck(
      testInfo,
      "network.modes",
      `GET /api/health modes: cpu_live=${JSON.stringify(modes.cpu_live)} qwen_replay.badge="${modes.qwen_replay?.badge}" qwen_live.available=${modes.qwen_live?.available} disabled_reason="${modes.qwen_live?.disabled_reason}"`,
      modes.qwen_live?.available === false && modes.qwen_replay?.badge === BADGE,
    );

    // ---- CPU Live is the default mode ------------------------------------------------
    const modeGroup = page.getByRole("group", { name: "Assistant mode" });
    const cpuLive = modeGroup.getByRole("button", { name: "CPU Live", exact: true });
    const qwenReplay = modeGroup.getByRole("button", { name: "Qwen Replay", exact: true });
    const qwenLive = modeGroup.getByRole("button", { name: "Qwen Live", exact: true });
    await expect(cpuLive).toBeVisible({ timeout: 60_000 });
    await expect(cpuLive).toHaveAttribute("aria-pressed", "true");
    await expect(cpuLive).toBeEnabled();
    recordCheck(
      testInfo,
      "modes.cpu_live_default",
      `CPU Live mode control is active (aria-pressed=true) on first load; assistant is live (real BFF ready)`,
      (await cpuLive.getAttribute("aria-pressed")) === "true",
    );

    // ---- Qwen Live is disabled with the real reason ----------------------------------
    await expect(qwenLive).toBeDisabled();
    const qwenLiveTitle = (await qwenLive.getAttribute("title")) ?? "";
    expect(qwenLiveTitle).toContain("No live Qwen service detected");
    recordCheck(
      testInfo,
      "modes.qwen_live_disabled",
      `Qwen Live button disabled; title="${qwenLiveTitle}"`,
      qwenLiveTitle.includes("No live Qwen service detected"),
    );

    // ---- switch to Qwen Replay: real /api/replay payload + loud badge ----------------
    const replayResponsePromise = page.waitForResponse(
      (response) => response.url().includes("/api/replay") && response.request().method() === "GET",
      { timeout: 120_000 },
    );
    await qwenReplay.click();
    const replayResponse = await replayResponsePromise;
    const replayPayload = await replayResponse.json();
    recordPayload(testInfo, "replay", {
      http_status: replayResponse.status(),
      mode: replayPayload.mode,
      replay: replayPayload.replay,
      live_call: replayPayload.live_call,
      badge: replayPayload.badge,
      examples: (replayPayload.examples ?? []).length,
    });
    expect(replayResponse.status()).toBe(200);
    expect(replayPayload.replay).toBe(true);
    expect(replayPayload.live_call).toBe(false);
    expect(replayPayload.badge).toBe(BADGE);
    expect((replayPayload.examples ?? []).length).toBeGreaterThan(0);
    const examples = replayPayload.examples ?? [];
    const examplesWithoutTerminal = examples.filter((example: any) => example.terminal == null).length;
    recordCheck(
      testInfo,
      "network.replay.examples",
      `GET /api/replay examples=${examples.length}; ${examplesWithoutTerminal} saved trajectory/s carry final_answer instead of a terminal block (real P18/P14-C1 data)`,
      examples.length > 0,
    );

    // The mode must really switch (header control state), even before the replay sheet content.
    await expect(qwenReplay).toHaveAttribute("aria-pressed", "true", { timeout: 60_000 });
    const notice = page.getByTestId("replay-mode-notice");
    await expect(notice).toBeVisible({ timeout: 60_000 });
    await expect(notice).toContainText(BADGE);
    const replayPanel = page.getByTestId("replay-panel");
    await expect(replayPanel).toBeVisible({ timeout: 60_000 });
    await expect(replayPanel).toContainText(BADGE);
    await expect(replayPanel).toContainText("live_call=false");
    await expect(replayPanel).toContainText("replay=true");
    await expect(replayPanel).toContainText("Saved trajectories");
    await expect(page.getByTestId("replay-example")).toBeVisible({ timeout: 60_000 });
    recordCheck(
      testInfo,
      "modes.replay_labelled",
      `Qwen Replay controller aria-pressed=true; replay-mode-notice shows "${BADGE}"; replay panel lists ${examples.length} saved real trajectories with live_call=false/replay=true and a selected trajectory detail`,
      replayPayload.live_call === false && examples.length > 0,
    );

    // ---- the composer is read-only in replay mode -----------------------------------
    await expect(page.getByTestId("composer-input")).toBeDisabled();
    await expect(page.getByTestId("composer-input")).toHaveAttribute(
      "placeholder",
      /Qwen Replay is read-only historical trajectories/,
    );
    recordCheck(
      testInfo,
      "modes.replay_read_only",
      "composer is disabled and its hint says Qwen Replay is read-only historical trajectories — switch to CPU Live to chat",
      true,
    );

    await saveScreenshot(page, testInfo, "replay-mode.png", DESKTOP);

    // ---- switch back to the real live mode ------------------------------------------
    await page.getByRole("button", { name: "Switch back to CPU Live" }).click();
    await expect(notice).toBeHidden({ timeout: 30_000 });
    await expect(page.getByTestId("composer-input")).toBeEnabled({ timeout: 30_000 });
    await expect(cpuLive).toHaveAttribute("aria-pressed", "true");
    recordCheck(
      testInfo,
      "modes.back_to_cpu_live",
      "after 'Switch back to CPU Live': replay notice hidden, composer re-enabled, CPU Live pressed",
      (await cpuLive.getAttribute("aria-pressed")) === "true",
    );
  });
});
