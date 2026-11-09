/**
 * Real failure visibility: real BFF error boundaries must surface real errors and never fabricate
 * a success.
 *
 * Verified product reality (probed against the live demo, recorded in the report):
 *  - profile_0049 is a real read-only MCP server (`memory_writes=false`);
 *  - its fixture already holds 3 active preferences, so even a deliberately vague query is
 *    really `ALLOW_FINALIZE`d — the shipped UI never reaches the clarification reply flow on this
 *    profile (measured with 7 vague queries; the writes-disabled notice is therefore unreachable
 *    in the UI with the current fixture, not faked);
 *  - the real error boundaries that ARE reachable are asserted here:
 *      1. read-only chat turn does not fabricate any memory update;
 *      2. POST /api/clarification/reply with a handle that is not pending → real 409
 *         `clarification_not_pending`;
 *      3. a browser can never inject identity/confirmation (`confirmed_by_user`) → real 422
 *         (Pydantic `extra="forbid"`), no chat stream is produced;
 *      4. invalid product refs return real 4xx (also covered in 02 for the compare path).
 */
import { expect, test } from "@playwright/test";
import { ensureArtifactDirs, recordCheck, recordPayload } from "../fixtures/evidence";
import { bffGetJson, bffPostJson, lastProductsBlock, openDemo, sendChat, waitForHealthReady } from "../fixtures/demo";

const DESKTOP = { width: 1440, height: 900 };
const VAGUE_QUERY = "I need some new headphones. Not sure what matters yet.";

test.describe("Real failure visibility — real BFF errors are surfaced, never faked", () => {
  test("read-only profile fabricates no memory write and real BFF error boundaries return errors", async ({
    page,
    request,
  }, testInfo) => {
    ensureArtifactDirs();
    await page.setViewportSize(DESKTOP);
    await openDemo(page);
    await waitForHealthReady(page, testInfo);

    const config = await bffGetJson(request, "/api/demo/config");
    const readOnlyProfile = (config.body.profiles ?? []).find(
      (profile: any) => profile.profile_id === "profile_0049",
    );
    recordCheck(
      testInfo,
      "demo.config.read_only_profile",
      `profile_0049 memory_writes=${readOnlyProfile?.memory_writes} (real read-only MCP server)`,
      readOnlyProfile?.memory_writes === false,
    );
    expect(readOnlyProfile?.memory_writes).toBe(false);

    // ---- 1) real read-only turn: works, but never fakes a memory write -----------------
    await sendChat(page, testInfo, VAGUE_QUERY);
    const block = lastProductsBlock(page);
    await expect(block).toBeVisible({ timeout: 60_000 });
    const gateAction = await page.getByTestId("gate-card").last().getAttribute("data-gate-action");
    const cardCount = await block.getByTestId("product-card").count();
    recordCheck(
      testInfo,
      "readonly.turn.real",
      `real read-only search/recommend turn: ${cardCount} cards, gate=${gateAction} (the current 0049 fixture finalizes vague queries; the writes-disabled clarification UI path is unreachable)`,
      cardCount >= 1 && gateAction === "ALLOW_FINALIZE",
    );
    expect(cardCount).toBeGreaterThanOrEqual(1);
    expect(gateAction).toBe("ALLOW_FINALIZE");

    const fakeMemoryUi =
      (await page.getByTestId("memory-update").count()) +
      (await page.getByTestId("preferences-refreshed").count());
    expect(fakeMemoryUi, "a read-only turn must not render any memory success UI").toBe(0);
    recordCheck(
      testInfo,
      "readonly.no_fake_memory_success",
      `read-only turn rendered no memory-update / preferences-refreshed blocks (count=${fakeMemoryUi}); no confirmation was fabricated`,
      fakeMemoryUi === 0,
    );

    // ---- shared real BFF session for the error-boundary probes -------------------------
    const session = await bffPostJson(request, "/api/sessions", { profile_id: "profile_0049" });
    expect(session.status).toBe(200);
    const sessionId = session.body.session_id as string;

    // ---- 2) real 409: clarification handle that is not pending -----------------------
    const bogusReply = await bffPostJson(request, "/api/clarification/reply", {
      session_id: sessionId,
      clarification_id: "clar_00000000000000000000000000000000",
      message: "I prefer wireless.",
    });
    recordPayload(testInfo, "not-pending-clarification", {
      http_status: bogusReply.status,
      body: bogusReply.body,
    });
    const bogusStatusOk = bogusReply.status === 409;
    const bogusCode = String(bogusReply.body?.code ?? "");
    const bogusMessage = String(bogusReply.body?.message ?? "");
    const bogusRealError =
      bogusStatusOk &&
      bogusCode === "clarification_not_pending" &&
      bogusMessage.length > 0 &&
      bogusReply.body?.memory_update == null;
    recordCheck(
      testInfo,
      "bff.not_pending_clarification.real_error",
      `POST /api/clarification/reply without a pending handle → HTTP ${bogusReply.status} code="${bogusCode}" message="${bogusMessage.slice(0, 160)}"; no memory_update success payload`,
      bogusRealError,
    );
    expect(bogusReply.status).toBe(409);
    expect(bogusCode).toBe("clarification_not_pending");
    expect(bogusMessage.length).toBeGreaterThan(0);
    expect(bogusReply.body?.memory_update ?? null).toBeNull();

    // ---- 3) real 422: browser cannot inject identity / confirmation -------------------
    const injectionResponse = await request.post(
      `${process.env.E2E_BFF_URL ?? "http://127.0.0.1:8787"}/api/chat/stream`,
      {
        data: {
          session_id: sessionId,
          message: "I need earbuds.",
          confirmed_by_user: true,
          user_id: "p14e2_lite_0049",
        },
        timeout: 30_000,
      },
    );
    let injectionBody: any = null;
    try {
      injectionBody = await injectionResponse.json();
    } catch {
      injectionBody = null;
    }
    recordPayload(testInfo, "identity-injection-rejected", {
      http_status: injectionResponse.status(),
      body: injectionBody,
    });
    const injectionRejected =
      injectionResponse.status() === 422 &&
      injectionBody != null &&
      !Array.isArray(injectionBody.products) &&
      injectionBody.text == null;
    recordCheck(
      testInfo,
      "bff.identity_injection_rejected",
      `POST /api/chat/stream with confirmed_by_user/user_id injected → HTTP ${injectionResponse.status()} (Pydantic extra=forbid); no tool/products success payload`,
      injectionRejected,
    );
    expect(injectionResponse.status()).toBe(422);
    expect(injectionRejected).toBe(true);

    // ---- 4) real 4xx: invalid product ref (compare path also covered in 02) -----------
    const badCompare = await bffPostJson(request, "/api/products/compare", {
      session_id: sessionId,
      product_refs: ["B004K6R43G", "B000000000"],
    });
    recordPayload(testInfo, "invalid-compare-ref", {
      http_status: badCompare.status,
      body: badCompare.body,
    });
    const invalidRefOk =
      [400, 422].includes(badCompare.status) &&
      typeof badCompare.body?.code === "string" &&
      String(badCompare.body?.message ?? "").length > 0 &&
      badCompare.body?.status !== "comparison_ready";
    recordCheck(
      testInfo,
      "bff.invalid_product_ref.real_error",
      `POST /api/products/compare with an unknown ref → HTTP ${badCompare.status} code="${badCompare.body?.code}" message="${String(
        badCompare.body?.message ?? "",
      ).slice(0, 160)}"; no comparison_ready fallback`,
      invalidRefOk,
    );
    expect([400, 422]).toContain(badCompare.status);
    expect(invalidRefOk).toBe(true);

    recordCheck(
      testInfo,
      "failure.no_fake_success",
      "all real error boundaries returned real HTTP errors (409 clarification_not_pending, 422 identity injection, 4xx invalid ref) and the read-only UI turn fabricated no memory success",
      true,
    );
  });
});
