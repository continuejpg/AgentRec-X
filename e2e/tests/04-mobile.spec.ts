/**
 * Mobile acceptance (390x844): single-column layout, real left drawer, composer docked at the
 * bottom, a real search, and the real comparison rendered inside the right sheet.
 *
 * Same real stack, no mocks. Screenshots `mobile-search.png` and `mobile-comparison-sheet.png`
 * are saved explicitly at exactly 390x844.
 */
import { expect, test } from "@playwright/test";
import {
  ensureArtifactDirs,
  recordCheck,
  recordTiming,
  saveScreenshot,
} from "../fixtures/evidence";
import {
  lastProductsBlock,
  openDemo,
  productCardIds,
  sendChat,
  waitForHealthReady,
} from "../fixtures/demo";

const MOBILE = { width: 390, height: 844 };
const SEARCH_QUERY = "I need earbuds I can use while working out.";

test.use({ viewport: MOBILE, isMobile: true, hasTouch: true, deviceScaleFactor: 1 });

test.describe("Mobile 390x844", () => {
  test("single column, real drawer, docked composer and a real search", async ({ page }, testInfo) => {
    ensureArtifactDirs();
    const scenarioStarted = Date.now();
    await openDemo(page);
    await waitForHealthReady(page, testInfo);

    const viewport = page.viewportSize();
    expect(viewport).toEqual(MOBILE);

    // ---- single-column shell: desktop sidebar and panel column are not rendered visible ---
    await expect(page.getByTestId("sidebar")).toBeHidden();
    await expect(page.getByTestId("panel-column")).toBeHidden();
    const shell = await page.getByTestId("app-shell").boundingBox();
    expect(shell).not.toBeNull();
    expect(Math.round(shell!.width)).toBe(MOBILE.width);
    const transcript = await page.getByTestId("transcript").boundingBox();
    expect(transcript).not.toBeNull();
    const singleColumn = transcript!.width >= MOBILE.width - 40;
    recordCheck(
      testInfo,
      "mobile.single_column",
      `app-shell width=${Math.round(shell!.width)}px at 390px viewport; desktop sidebar hidden; desktop panel column hidden; transcript width=${Math.round(transcript!.width)}px`,
      singleColumn,
    );
    expect(singleColumn).toBe(true);

    // ---- composer docked at the bottom (not part of the scrolling transcript) ----------
    const composerBefore = await page.getByTestId("composer").boundingBox();
    expect(composerBefore).not.toBeNull();
    const bottomGap = MOBILE.height - (composerBefore!.y + composerBefore!.height);
    const composerInBottomRegion = composerBefore!.y > MOBILE.height * 0.7 && bottomGap < 120;
    const composerOutsideTranscript = await page.evaluate(() => {
      const transcriptEl = document.querySelector('[data-testid="transcript"]');
      const composerEl = document.querySelector('[data-testid="composer"]');
      return Boolean(transcriptEl && composerEl && !transcriptEl.contains(composerEl));
    });
    recordCheck(
      testInfo,
      "mobile.composer_docked",
      `composer y=${Math.round(composerBefore!.y)} height=${Math.round(
        composerBefore!.height,
      )} bottom gap=${Math.round(bottomGap)}px; composer is outside the scrolling transcript=${composerOutsideTranscript}`,
      composerInBottomRegion && composerOutsideTranscript,
    );
    expect(composerInBottomRegion, "composer must sit in the bottom region on mobile").toBe(true);
    expect(composerOutsideTranscript, "composer must be docked outside the scroll area").toBe(true);

    // ---- real left drawer ------------------------------------------------------------
    await page.getByRole("button", { name: "Open conversations and preferences" }).click();
    const drawer = page.getByRole("dialog", { name: "Conversations and preferences" });
    await expect(drawer).toBeVisible({ timeout: 60_000 });
    await expect(drawer.getByTestId("sidebar")).toBeVisible();
    await expect(drawer).toContainText("Active preferences");
    await expect(drawer).toContainText("Demo profile");
    const drawerBox = await drawer.boundingBox();
    expect(drawerBox).not.toBeNull();
    // Round: fractional CSS pixels (340.0000019) must not fail a design-width check.
    const drawerWidth = Math.round(drawerBox!.width);
    expect(drawerWidth).toBeLessThanOrEqual(340);
    await saveScreenshot(page, testInfo, "mobile-drawer.png", MOBILE);
    recordCheck(
      testInfo,
      "mobile.drawer",
      `drawer opened via header button, width=${drawerWidth}px (<=340), contains sidebar/profile/preferences`,
      drawerWidth <= 340,
    );
    await drawer.getByRole("button", { name: "Close sidebar" }).click();
    await expect(drawer).toBeHidden({ timeout: 30_000 });

    // ---- real search on mobile -------------------------------------------------------
    await sendChat(page, testInfo, SEARCH_QUERY);
    const block = lastProductsBlock(page);
    await expect(block).toBeVisible({ timeout: 120_000 });
    const ids = await productCardIds(block);
    expect(ids.length).toBeGreaterThanOrEqual(2);
    await expect(block.getByTestId("product-card").first()).toBeVisible();
    const cardBox = await block.getByTestId("product-card").first().boundingBox();
    expect(cardBox).not.toBeNull();
    const cardWidth = Math.round(cardBox!.width);
    const blockText = await block.innerText();
    expect(blockText).toContain("Price unavailable / UNKNOWN");
    recordCheck(
      testInfo,
      "mobile.search.real_cards",
      `${ids.length} real cards on mobile (${ids.join(", ")}); first card width=${cardWidth}px; fixed UNKNOWN price text visible`,
      ids.length >= 2 && cardWidth <= 300 && blockText.includes("Price unavailable / UNKNOWN"),
    );

    const composerAfter = await page.getByTestId("composer").boundingBox();
    expect(composerAfter).not.toBeNull();
    const bottomGapAfter = MOBILE.height - (composerAfter!.y + composerAfter!.height);
    const dockedAfterSearch = composerAfter!.y > MOBILE.height * 0.7 && bottomGapAfter < 120;

    // Fixed-bottom semantics: scrolling the transcript must not move the docked composer.
    await page.getByTestId("transcript").evaluate((element) => {
      (element as HTMLElement).scrollTop = 0;
    });
    await page.waitForTimeout(250);
    const composerAfterScroll = await page.getByTestId("composer").boundingBox();
    expect(composerAfterScroll).not.toBeNull();
    const staysDockedOnScroll = Math.abs(composerAfterScroll!.y - composerAfter!.y) <= 2;
    const noHorizontalOverflow = await page.evaluate(
      () => Math.ceil(document.documentElement.scrollWidth) <= window.innerWidth + 1,
    );
    recordCheck(
      testInfo,
      "mobile.composer_after_search",
      `composer in bottom region after real search (y=${Math.round(composerAfter!.y)}, bottom gap=${Math.round(
        bottomGapAfter,
      )}px); y before=${Math.round(composerBefore!.y)}; transcript scroll does not move it (Δ=${Math.abs(
        Math.round(composerAfterScroll!.y - composerAfter!.y),
      )}px); no horizontal document overflow=${noHorizontalOverflow}`,
      dockedAfterSearch && staysDockedOnScroll && noHorizontalOverflow,
    );
    expect(dockedAfterSearch, "composer must stay in the bottom region after the real search").toBe(true);
    expect(staysDockedOnScroll, "scrolling the transcript must not move the docked composer").toBe(true);
    expect(noHorizontalOverflow).toBe(true);

    await saveScreenshot(page, testInfo, "mobile-search.png", MOBILE);
    recordTiming(testInfo, "mobile.scenario1.total", Date.now() - scenarioStarted);
  });

  test("real comparison rendered inside the mobile sheet", async ({ page }, testInfo) => {
    ensureArtifactDirs();
    const scenarioStarted = Date.now();
    await openDemo(page);
    await waitForHealthReady(page, testInfo);

    await page.getByRole("button", { name: "Open details, evidence and comparison panel" }).click();
    const sheet = page.getByTestId("panel-sheet");
    await expect(sheet).toBeVisible({ timeout: 60_000 });
    await expect(page.getByRole("dialog", { name: "Details, evidence and comparison" })).toBeVisible();
    const sheetBox = await sheet.boundingBox();
    expect(sheetBox).not.toBeNull();
    const sheetWidth = Math.round(sheetBox!.width);
    const sheetFits = sheetWidth >= 300 && sheetWidth <= 430;
    recordCheck(
      testInfo,
      "mobile.panel_sheet",
      `right panel sheet opened; width=${sheetWidth}px (<=430px), role=dialog aria-modal`,
      sheetFits,
    );
    expect(sheetFits).toBe(true);

    await sheet.getByTestId("panel-tab-comparison").click();
    await expect(sheet.getByTestId("comparison-selection")).toBeVisible({ timeout: 60_000 });
    const compareResponsePromise = page.waitForResponse(
      (response) =>
        response.url().includes("/api/products/compare") && response.request().method() === "POST",
      { timeout: 240_000 },
    );
    await sheet.getByTestId("compare-verified").click();
    const compareResponse = await compareResponsePromise;
    expect(compareResponse.status()).toBe(200);
    const comparePayload = await compareResponse.json();
    expect(comparePayload.status).toBe("comparison_ready");
    const comparison = sheet.getByTestId("comparison-view");
    await expect(comparison).toBeVisible({ timeout: 120_000 });
    await expect(comparison).toHaveAttribute("data-status", "comparison_ready");
    const text = await comparison.innerText();
    const upperText = text.toUpperCase();
    expect(text).toContain("B004K6R43G");
    expect(text).toContain("B00008VF63");
    // Headings use CSS text-transform: uppercase, so innerText is "CLAIMS SUMMARY" etc.
    expect(upperText).toContain("CLAIMS SUMMARY");
    expect(upperText).toContain("CONDITIONAL RECOMMENDATION");
    expect(upperText).toContain("DIMENSIONS");
    recordCheck(
      testInfo,
      "mobile.comparison_sheet",
      `real POST /api/products/compare HTTP ${compareResponse.status()} status=${comparePayload.status}; comparison rendered inside the sheet with both verified ids, claims summary and conditional recommendation`,
      compareResponse.status() === 200 && text.includes("B004K6R43G") && text.includes("B00008VF63"),
    );

    await saveScreenshot(page, testInfo, "mobile-comparison-sheet.png", MOBILE);
    recordTiming(testInfo, "mobile.comparison.total", Date.now() - scenarioStarted);
  });
});
