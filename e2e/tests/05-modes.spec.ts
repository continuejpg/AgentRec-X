/**
 * Modes acceptance.
 *
 * CPU Live is the default *real* mode. Qwen Replay is visibly labelled
 * "Historical Replay — Not a Live Model Call" and read-only. Qwen Live is availability-gated by
 * docs/QWEN_LIVE_CONTRACT.md §2: it is selectable only when the backend actually reports
 * `modes.qwen_live.available === true`; otherwise the mode control is disabled with the backend's
 * real `disabled_reason` and an explicit Unavailable notice is rendered (never a silent fallback).
 *
 * Data comes from the real GET /api/health + GET /api/demo/config + GET /api/replay. No mocks and
 * no fabricated Qwen response: when the live mode is available the test only asserts the real UI
 * wiring, and a real chat turn is attempted only when the operator opts in with QWEN_LIVE_E2E=1.
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
  TURN_TIMEOUT_MS,
  waitForHealthReady,
} from "../fixtures/demo";

const DESKTOP = { width: 1440, height: 900 };
const BADGE = "Historical Replay — Not a Live Model Call";

test.describe("Modes — CPU Live default, Qwen Replay labelled, Qwen Live availability-gated", () => {
  test("replay mode is visibly historical and qwen live availability is honest", async ({
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
    const config = await bffGetJson(request, "/api/demo/config");
    expect(config.status).toBe(200);
    const healthModes = health.body.modes ?? {};
    const configModes = config.body.modes ?? {};
    const modes = { ...configModes, ...healthModes };
    recordPayload(testInfo, "modes", {
      health_modes: health.body.modes,
      config_modes: config.body.modes,
      health_qwen_live: health.body.qwen_live ?? null,
    });
    expect(modes.qwen_replay?.available).toBe(true);
    expect(modes.qwen_replay?.badge).toBe(BADGE);

    // Resolve Qwen Live exactly like web/lib/modes.ts: the current /api/health mode block first,
    // then the health snapshot, then /api/demo/config; anything else is unavailable.
    const qwenHealth = health.body.qwen_live ?? null;
    const healthQwenMode = healthModes.qwen_live ?? null;
    const configQwenMode = configModes.qwen_live ?? null;
    const qwenAvailable =
      typeof healthQwenMode?.available === "boolean"
        ? healthQwenMode.available
        : typeof qwenHealth?.available === "boolean"
          ? qwenHealth.available
          : configQwenMode?.available === true;
    const qwenMode = configQwenMode ?? healthQwenMode ?? {};
    const qwenReason: string =
      qwenMode.disabled_reason ?? healthQwenMode?.disabled_reason ?? qwenHealth?.detail ?? "";
    const qwenBadge: string =
      (qwenAvailable ? (qwenMode.badge ?? healthQwenMode?.badge) : null) ?? "Live Qwen mode";
    recordCheck(
      testInfo,
      "network.modes",
      `GET /api/health modes: cpu_live=${JSON.stringify(modes.cpu_live)} qwen_replay.badge="${modes.qwen_replay?.badge}" qwen_live.available=${String(qwenAvailable)} qwen_live.disabled_reason="${qwenReason}"`,
      modes.qwen_replay?.badge === BADGE && (qwenAvailable || qwenReason.length > 0),
    );

    // ---- CPU Live is the default mode ------------------------------------------------
    const modeGroup = page.getByRole("group", { name: "Assistant mode" });
    const cpuLive = modeGroup.getByRole("button", { name: "CPU Live", exact: true });
    const qwenReplay = modeGroup.getByRole("button", { name: "Qwen Replay", exact: true });
    const qwenLive = page.getByTestId("mode-qwen-live");
    const qwenLiveBadge = page.getByTestId("qwen-live-badge");
    const qwenUnavailableNotice = page.getByTestId("qwen-unavailable-notice");
    await expect(cpuLive).toBeVisible({ timeout: 60_000 });
    await expect(cpuLive).toHaveAttribute("aria-pressed", "true");
    await expect(cpuLive).toBeEnabled();
    recordCheck(
      testInfo,
      "modes.cpu_live_default",
      `CPU Live mode control is active (aria-pressed=true) on first load; assistant is live (real BFF ready)`,
      (await cpuLive.getAttribute("aria-pressed")) === "true",
    );

    // ---- Qwen Live: explicit available or explicit Unavailable, never ambiguous ------
    await expect(qwenLive).toBeVisible();
    if (!qwenAvailable) {
      await expect(qwenLive).toBeDisabled();
      await expect(qwenLive).toHaveAttribute("data-state", "unavailable");
      await expect(qwenLive).toHaveAttribute("aria-pressed", "false");
      const qwenLiveTitle = (await qwenLive.getAttribute("title")) ?? "";
      expect(qwenReason.length).toBeGreaterThan(0);
      expect(qwenLiveTitle).toBe(qwenReason);
      // docs/QWEN_LIVE_CONTRACT.md §2: with no live service the reason must say so.
      expect(qwenReason).toContain("No live Qwen service detected");
      // The explicit Unavailable state is rendered with the backend's real reason.
      await expect(qwenUnavailableNotice).toBeVisible({ timeout: 60_000 });
      await expect(qwenUnavailableNotice).toHaveAttribute("data-state", "unavailable");
      await expect(qwenUnavailableNotice).toContainText("Unavailable");
      await expect(qwenUnavailableNotice).toContainText(qwenReason);
      // No live badge may appear while the mode is unavailable.
      await expect(qwenLiveBadge).toHaveCount(0);
      // The default CPU mode is still live: availability is reported, not silently faked.
      await expect(page.getByTestId("composer-input")).toBeEnabled();
      recordCheck(
        testInfo,
        "modes.qwen_live_disabled",
        `Qwen Live button disabled with data-state=unavailable, title="${qwenLiveTitle}"; qwen-unavailable-notice shows the real disabled_reason; no qwen-live-badge; CPU Live remains the live default`,
        qwenLiveTitle.includes("No live Qwen service detected"),
      );
    } else {
      await expect(qwenLive).toBeEnabled();
      await expect(qwenLive).toHaveAttribute("data-state", "available");
      await expect(qwenUnavailableNotice).toHaveCount(0);
      await qwenLive.click();
      await expect(qwenLive).toHaveAttribute("aria-pressed", "true", { timeout: 30_000 });
      // The header badge is the backend's own live badge, not client copy.
      await expect(qwenLiveBadge).toBeVisible({ timeout: 30_000 });
      await expect(qwenLiveBadge).toContainText(qwenBadge);
      // Replay-only restrictions must not appear in the live Qwen mode.
      await expect(page.getByTestId("replay-mode-notice")).toHaveCount(0);
      await expect(page.getByTestId("composer-input")).toBeEnabled();
      recordCheck(
        testInfo,
        "modes.qwen_live_selectable",
        `Qwen Live button enabled (data-state=available, aria-pressed=true); qwen-live-badge="${qwenBadge}"; composer enabled; no replay-only notice`,
        true,
      );

      // Optional real live run: only when an operator explicitly opts in, and only if the
      // backend is actually ready. The test asserts a real outcome (planner metadata or an
      // explicit qwen_* error event) and never fabricates a response.
      if (process.env.QWEN_LIVE_E2E === "1" && health.body.ready === true) {
        const turnsBefore = await page.getByTestId("turn").count();
        const input = page.getByTestId("composer-input");
        await input.fill("I need wireless earbuds I can use while working out.");
        const started = Date.now();
        const responsePromise = page.waitForResponse(
          (response) =>
            response.url().includes("/api/chat/stream") && response.request().method() === "POST",
          { timeout: 120_000 },
        );
        await page.getByTestId("composer-send").click();
        const response = await responsePromise;
        let requestBody: any = null;
        try {
          requestBody = response.request().postDataJSON();
        } catch {
          requestBody = null;
        }
        expect(response.status()).toBe(200);
        expect(requestBody?.mode).toBe("qwen_live");
        await expect
          .poll(
            async () => {
              const turns = page.getByTestId("turn");
              if ((await turns.count()) < turnsBefore + 1) return "no-turn";
              const last = turns.last();
              const terminal =
                (await last.getByTestId("turn-meta").count()) +
                (await last.getByTestId("error-bubble").count());
              const running = await last.locator('[data-testid="tool-row"][data-status="running"]').count();
              return terminal > 0 && running === 0 ? "idle" : "busy";
            },
            { timeout: TURN_TIMEOUT_MS },
          )
          .toBe("idle");
        recordTiming(testInfo, "qwen.live.turn", Date.now() - started, "real Qwen Live SSE turn");
        const turn = lastTurn(page);
        const plannerRows = await turn.getByTestId("turn-planner").count();
        const errorBubbles = await turn.getByTestId("error-bubble").count();
        // Real outcome only: either reported planner metadata or a real error event.
        expect(plannerRows + errorBubbles).toBeGreaterThan(0);
        if (plannerRows > 0) {
          await expect(turn.getByTestId("turn-planner")).toContainText("Qwen Base Live");
          await expect(turn.getByTestId("turn-planner")).toHaveAttribute("data-planner-kind", "qwen_base");
          recordCheck(
            testInfo,
            "modes.qwen_live_turn",
            "real Qwen Live turn rendered backend planner metadata (Qwen Base Live) from the done/products payload",
            true,
          );
        } else {
          const errorText = await turn.getByTestId("error-bubble").innerText();
          expect(errorText.toLowerCase()).toContain("qwen");
          recordCheck(
            testInfo,
            "modes.qwen_live_turn",
            `real Qwen Live turn failed honestly with a qwen_* error event (no CPU fallback): ${errorText.slice(0, 180)}`,
            true,
          );
        }
      } else {
        recordCheck(
          testInfo,
          "modes.qwen_live_turn",
          "Qwen Live selected with the backend's real live badge; no chat turn was sent (set QWEN_LIVE_E2E=1 with a reachable live service to exercise a real turn — nothing is faked here)",
          true,
        );
      }

      // Back to the default live mode for the replay part of this test.
      await cpuLive.click();
      await expect(cpuLive).toHaveAttribute("aria-pressed", "true");
    }

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
