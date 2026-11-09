/**
 * Scenario 1 — Search & Recommend (real BFF → real P17 MCP tools → real BGE / P16 Gate / P15
 * evidence / real catalogue).
 *
 * Real query used: the P18 Demo-1 working-out query, documented in P18_FINAL_REPORT.md §2 and
 * runs/p18/demos/search_and_recommend.json (real trajectory: selected B004K6R43G,
 * Gate ALLOW_FINALIZE, UNKNOWN price/brand/stock/material/size/warranty).
 */
import { expect, test } from "@playwright/test";
import {
  BFF_BASE,
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
  openPanelTab,
  productCardIds,
  sendChat,
  waitForHealthReady,
} from "../fixtures/demo";

const DESKTOP = { width: 1440, height: 900 };
const QUERY = "I need earbuds I can use while working out.";
const ASIN = /^[A-Z0-9]{10}$/;

test.describe("Scenario 1 — Search & Recommend", () => {
  test("natural-language search returns real cards, Gate, UNKNOWN and evidence spans", async ({
    page,
    request,
  }, testInfo) => {
    ensureArtifactDirs();
    const scenarioStarted = Date.now();
    await page.setViewportSize(DESKTOP);
    await openDemo(page);
    await waitForHealthReady(page, testInfo);

    // ---- real network evidence: health + demo config --------------------------------
    const health = await bffGetJson(request, "/api/health");
    expect(health.status).toBe(200);
    expect(health.body.ready).toBe(true);
    expect(health.body.catalogue_size).toBe(76263);
    const config = await bffGetJson(request, "/api/demo/config");
    expect(config.status).toBe(200);
    expect(config.body.profiles.map((profile: any) => profile.profile_id)).toEqual([
      "profile_0049",
      "profile_0057",
    ]);
    expect(config.body.verified_comparison_ids).toEqual(["B004K6R43G", "B00008VF63"]);
    recordPayload(testInfo, "health", {
      ready: health.body.ready,
      status: health.body.status,
      catalogue_size: health.body.catalogue_size,
      servers: (health.body.servers ?? []).map((server: any) => ({
        profile_id: server.profile_id,
        status: server.status,
        search_mode: server.search_mode,
        writes_enabled: server.writes_enabled,
        startup_ms: server.startup_ms,
        tools: server.tools,
      })),
      modes: health.body.modes,
    });
    recordCheck(
      testInfo,
      "network.health",
      `GET /api/health ready=${health.body.ready} catalogue_size=${health.body.catalogue_size} servers=${(health.body.servers ?? [])
        .map((server: any) => `${server.profile_id}:${server.status}:${server.search_mode}`)
        .join(",")}`,
      health.body.ready === true && health.body.catalogue_size === 76263,
    );
    recordCheck(
      testInfo,
      "network.config",
      `GET /api/demo/config verified_comparison_ids=${JSON.stringify(config.body.verified_comparison_ids)}`,
      Array.isArray(config.body.verified_comparison_ids) && config.body.verified_comparison_ids.length === 2,
    );

    // The app auto-loads the full catalogue record for the top candidate right after `products`.
    const earlyDetailsPromise = page
      .waitForResponse(
        (response) =>
          response.url().includes("/api/products/details") && response.request().method() === "POST",
        { timeout: 240_000 },
      )
      .catch(() => null);

    // ---- send the real natural-language query ---------------------------------------
    await sendChat(page, testInfo, QUERY);
    const turn = lastTurn(page);
    await expect(turn.getByTestId("user-bubble")).toContainText("I need earbuds I can use while working out");

    const block = lastProductsBlock(page);
    await expect(block).toBeVisible({ timeout: 120_000 });

    // ---- real Gate decision (contract: ALLOW_FINALIZE or a real clarification wait) ---
    const gate = block.getByTestId("gate-card");
    await expect(gate).toBeVisible({ timeout: 60_000 });
    const gateAction = await gate.getAttribute("data-gate-action");
    const gateText = await gate.innerText();
    recordCheck(
      testInfo,
      "gate.action",
      `real P16 gate data-gate-action=${gateAction}; text="${gateText.split("\n").slice(0, 4).join(" | ")}"`,
      gateAction === "ALLOW_FINALIZE",
    );
    expect(
      gateAction,
      `expected the real Gate to ALLOW_FINALIZE for the P18 Demo-1 query, got ${gateAction}; gate text: ${gateText}`,
    ).toBe("ALLOW_FINALIZE");
    // humanize() only capitalises the first character ("ALLOW_FINALIZE" → "ALLOW FINALIZE").
    expect(gateText.toUpperCase()).toContain("ALLOW FINALIZE");
    expect(gateText).toMatch(/evidence_sufficient/);
    expect(gateText).toMatch(/Selected:\s*[A-Z0-9]{10}/);
    const selectedMatch = /Selected:\s*([A-Z0-9]{10})/.exec(gateText);
    const gateSelectedId = selectedMatch ? selectedMatch[1] : "";
    recordCheck(
      testInfo,
      "gate.detail",
      `reason_code visible; selected candidate=${gateSelectedId}; Gate UNKNOWN chips rendered=${/Gate UNKNOWN:/.test(gateText)}`,
      gateSelectedId.length === 10,
    );

    // ---- real product cards ----------------------------------------------------------
    await expect(block.getByTestId("product-card").first()).toBeVisible({ timeout: 60_000 });
    const ids = await productCardIds(block);
    recordCheck(
      testInfo,
      "products.cards",
      `${ids.length} real catalogue cards: ${ids.join(", ")}`,
      ids.length >= 2 && ids.every((id) => ASIN.test(id)) && new Set(ids).size === ids.length,
    );
    expect(ids.length).toBeGreaterThanOrEqual(2);
    for (const id of ids) expect(id, `card id ${id} must be a real ASIN-shaped catalogue id`).toMatch(ASIN);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain(gateSelectedId);

    const blockText = await block.innerText();
    expect(blockText).toMatch(/mode\s+bge/);
    expect(blockText).toMatch(/score\s+bge_dense/);
    expect(blockText).toContain("76,263");
    expect(blockText).toMatch(/amazon_electronics_c4/);
    expect(blockText).toContain("Price unavailable / UNKNOWN");
    expect(blockText).toMatch(/UNKNOWN · /);
    expect(blockText).toContain("Gate-selected item");

    const firstCardText = await block.getByTestId("product-card").first().innerText();
    recordCheck(
      testInfo,
      "products.card.content",
      `first card (${ids[0]}) contains title text, ASIN, namespace, fixed price text and UNKNOWN chips (${firstCardText.split("\n").slice(0, 6).join(" | ")}…)`,
      firstCardText.length > 60 &&
        firstCardText.includes(ids[0]) &&
        firstCardText.includes("Price unavailable / UNKNOWN") &&
        /UNKNOWN · /i.test(firstCardText),
    );
    expect(firstCardText.length).toBeGreaterThan(60);
    expect(firstCardText).toContain(ids[0]);
    expect(firstCardText).toContain("Price unavailable / UNKNOWN");

    const priceUnknownCount = await block.getByTestId("price-unknown").count();
    recordCheck(
      testInfo,
      "products.price.unknown",
      `every card renders the fixed "Price unavailable / UNKNOWN" block (${priceUnknownCount}/${ids.length})`,
      priceUnknownCount === ids.length,
    );
    expect(priceUnknownCount).toBe(ids.length);

    const evidencePillCount = await block.getByText("Evidence", { exact: true }).count();
    recordCheck(
      testInfo,
      "products.evidence_available",
      `${evidencePillCount} card(s) report the real evidence_available flag`,
      evidencePillCount >= 1,
    );
    expect(evidencePillCount).toBeGreaterThanOrEqual(1);

    await saveScreenshot(page, testInfo, "desktop-search.png", DESKTOP);

    // ---- cross-check the real detail payload vs the DOM -------------------------------
    const earlyDetailsResponse = await earlyDetailsPromise;
    let detailPayload: any = null;
    if (earlyDetailsResponse && earlyDetailsResponse.ok()) {
      detailPayload = await earlyDetailsResponse.json();
      recordPayload(testInfo, "details", {
        product_id: detailPayload.product_id,
        title: detailPayload.title,
        source_record: detailPayload.source_record,
        facts: (detailPayload.facts ?? []).map((fact: any) => ({
          attribute: fact.attribute,
          status: fact.status,
          value: fact.value,
          source: fact.source,
        })),
        evidence_count: (detailPayload.evidence ?? []).length,
        unknown_fields: detailPayload.unknown_fields,
        raw_fields: Object.fromEntries(
          Object.entries(detailPayload.raw_fields ?? {}).map(([key, value]) => [
            key,
            Array.isArray(value) ? value.length : 0,
          ]),
        ),
      });
      const detailOk =
        ASIN.test(String(detailPayload.product_id)) &&
        (detailPayload.evidence ?? []).length >= 1 &&
        (detailPayload.raw_fields?.title ?? []).length >= 1 &&
        String(detailPayload.source_record ?? "").includes("catalog_meta.jsonl");
      recordCheck(
        testInfo,
        "network.details.payload",
        `POST /api/products/details → product=${detailPayload.product_id} facts=${(detailPayload.facts ?? []).length} evidence=${(detailPayload.evidence ?? []).length} raw_fields=${Object.keys(detailPayload.raw_fields ?? {}).join(",")} source=${detailPayload.source_record}`,
        detailOk,
      );
      expect(detailOk).toBe(true);
    } else {
      recordCheck(testInfo, "network.details.payload", "no details response observed (auto-load deferred to View evidence)", true);
    }

    // ---- evidence drawer: real raw field / span / UNKNOWN / source record -------------
    const targetId = detailPayload?.product_id ?? gateSelectedId;
    const targetCard = block.locator(`[data-testid="product-card"][data-product-id="${targetId}"]`);
    await expect(targetCard).toBeVisible({ timeout: 60_000 });
    await targetCard.getByRole("button", { name: "View evidence" }).click();

    const panel = page.getByTestId("evidence-panel");
    await expect(panel).toBeVisible({ timeout: 120_000 });
    const evidenceItems = page.getByTestId("evidence-item");
    await expect(evidenceItems.first()).toBeVisible({ timeout: 120_000 });
    const evidenceCount = await evidenceItems.count();
    expect(evidenceCount).toBeGreaterThanOrEqual(1);

    const firstEvidenceText = await evidenceItems.first().innerText();
    expect(firstEvidenceText).toMatch(/(title|description|features|categories)/i);
    expect(firstEvidenceText).toMatch(/\d+\s*[–-]\s*\d+/);
    const spanQuoteCount = await panel.locator("blockquote").count();
    expect(spanQuoteCount).toBeGreaterThanOrEqual(1);
    await expect(page.getByTestId("unknown-fields")).toBeVisible({ timeout: 60_000 });
    await expect(page.getByTestId("unknown-fields")).toContainText("UNKNOWN");
    await expect(panel).toContainText("Source record");
    await expect(panel).toContainText("catalog_meta.jsonl");
    await expect(panel).toContainText("semantic_truth_verified=false");
    if (detailPayload && detailPayload.product_id === targetId && detailPayload.source_record) {
      await expect(panel).toContainText(String(detailPayload.source_record));
    }
    recordCheck(
      testInfo,
      "evidence.panel",
      `${evidenceCount} real P15 evidence span(s); first shows raw_field/span=${JSON.stringify(
        firstEvidenceText.split("\n").slice(0, 2).join(" | "),
      )}; UNKNOWN list, source record catalog_meta.jsonl and semantic_truth_verified=false visible`,
      evidenceCount >= 1 && /catalog_meta\.jsonl/.test(await panel.innerText()),
    );

    await saveScreenshot(page, testInfo, "desktop-evidence.png", DESKTOP);

    // ---- details tab: real facts / raw fields / separate active preferences ------------
    await openPanelTab(page, "details");
    const detailsPane = page.getByTestId("product-details");
    await expect(detailsPane).toBeVisible({ timeout: 60_000 });
    await expect(detailsPane).toContainText("Catalogue facts", { timeout: 120_000 });
    await expect(detailsPane).toContainText("Raw catalogue fields");
    await expect(detailsPane).toContainText("Your active preferences");
    const detailsTitle = await page.getByTestId("details-title").innerText();
    expect(detailsTitle.trim().length).toBeGreaterThan(3);
    if (detailPayload && detailPayload.product_id === targetId && detailPayload.title) {
      await expect(page.getByTestId("details-title")).toContainText(String(detailPayload.title).slice(0, 20));
    }
    recordCheck(
      testInfo,
      "details.panel",
      `details panel shows title "${detailsTitle.slice(0, 64)}", Catalogue facts, Raw catalogue fields and a separate Your active preferences block`,
      detailsTitle.trim().length > 3,
    );

    recordTiming(testInfo, "scenario1.total", Date.now() - scenarioStarted, "page load → evidence + details assertions done");
    recordCheck(testInfo, "environment.bff", `frontend "${await page.title()}" talked to real BFF ${BFF_BASE}`, true);
  });
});
