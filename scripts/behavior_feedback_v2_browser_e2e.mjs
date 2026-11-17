// AgentRec-X Behavior Feedback V2 - Chrome Qwen Live + BGE + behavior E2E.
//
//   node scripts/behavior_feedback_v2_browser_e2e.mjs
//
// Requires: Next.js on http://127.0.0.1:3000, BFF on http://127.0.0.1:8787 configured for
// real BGE + Qwen Live, and a Chrome started with --remote-debugging-port=9222.
// The script drives real headless Chrome through one Qwen Live multi-turn shopping flow:
// real BGE products, browser click/Like, AUTOMATED_BROWSER_E2E provenance, and the next-turn
// eligible-only Behavior OFF/ON shadow.  It never marks automated events as human LIVE.
// Screenshots and JSON evidence are written under runs/behavior_feedback_v2.

import fs from "node:fs";
import path from "node:path";

const CHROME = process.env.CHROME_DEBUG_URL ?? "http://127.0.0.1:9222";
const APP = process.env.NEXT_PUBLIC_URL ?? "http://127.0.0.1:3000/";
const OUT_DIR = path.resolve(process.env.BEHAVIOR_E2E_OUT ?? "runs/behavior_feedback_v2");
const BFF = (process.env.BFF_URL ?? "http://127.0.0.1:8787").replace(/\/+$/, "");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

fs.mkdirSync(OUT_DIR, { recursive: true });

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
    ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve, reject } = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) reject(new Error(`${message.error.code}: ${message.error.message}`));
        else resolve(message.result ?? {});
        return;
      }
      if (message.method) {
        for (const listener of this.listeners.get(message.method) ?? []) {
          listener(message.params);
        }
      }
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  on(method, listener) {
    const list = this.listeners.get(method) ?? [];
    list.push(listener);
    this.listeners.set(method, list);
  }

  async evaluate(expression) {
    const result = await this.send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture: true,
    });
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.text ?? "Runtime.evaluate failed");
    }
    return result.result?.value;
  }

  async waitFor(expression, { timeoutMs = 90_000, intervalMs = 250, label = expression } = {}) {
    const deadline = Date.now() + timeoutMs;
    let last;
    while (Date.now() < deadline) {
      try {
        last = await this.evaluate(expression);
        if (last) return last;
      } catch (error) {
        last = String(error);
      }
      await sleep(intervalMs);
    }
    throw new Error(`Timed out waiting for ${label}; last=${JSON.stringify(last)}`);
  }

  async screenshot(name) {
    const { data } = await this.send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: false,
    });
    const file = path.join(OUT_DIR, name);
    fs.writeFileSync(file, Buffer.from(data, "base64"));
    return file;
  }

  async clickSelector(selector, { index = 0 } = {}) {
    const rect = await this.evaluate(`(() => {
      const nodes = Array.from(document.querySelectorAll(${JSON.stringify(selector)}));
      const node = nodes[${index}];
      if (!node) return null;
      node.scrollIntoView({ block: "center", inline: "center" });
      const r = node.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height };
    })()`);
    if (!rect) throw new Error(`Selector not found: ${selector}[${index}]`);
    await sleep(120);
    const updated = await this.evaluate(`(() => {
      const node = Array.from(document.querySelectorAll(${JSON.stringify(selector)}))[${index}];
      if (!node) return null;
      const r = node.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    })()`);
    const point = updated ?? rect;
    await this.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: point.x, y: point.y });
    await this.send("Input.dispatchMouseEvent", {
      type: "mousePressed", x: point.x, y: point.y, button: "left", clickCount: 1,
    });
    await this.send("Input.dispatchMouseEvent", {
      type: "mouseReleased", x: point.x, y: point.y, button: "left", clickCount: 1,
    });
  }

  async clickButtonByText(text, { selector = "button" } = {}) {
    const rect = await this.evaluate(`(() => {
      const node = Array.from(document.querySelectorAll(${JSON.stringify(selector)})).find(
        (candidate) => (candidate.textContent || "").trim() === ${JSON.stringify(text)}
      );
      if (!node) return null;
      node.scrollIntoView({ block: "center", inline: "center" });
      const r = node.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    })()`);
    if (!rect) throw new Error(`Button not found: ${text}`);
    await sleep(120);
    const updated = await this.evaluate(`(() => {
      const node = Array.from(document.querySelectorAll(${JSON.stringify(selector)})).find(
        (candidate) => (candidate.textContent || "").trim() === ${JSON.stringify(text)}
      );
      if (!node) return null;
      const r = node.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    })()`);
    const point = updated ?? rect;
    await this.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: point.x, y: point.y });
    await this.send("Input.dispatchMouseEvent", {
      type: "mousePressed", x: point.x, y: point.y, button: "left", clickCount: 1,
    });
    await this.send("Input.dispatchMouseEvent", {
      type: "mouseReleased", x: point.x, y: point.y, button: "left", clickCount: 1,
    });
  }

  async clickCardButton(cardIndex, text) {
    const rect = await this.evaluate(`(() => {
      const card = document.querySelectorAll('[data-testid="product-card"]')[${cardIndex}];
      if (!card) return null;
      const node = Array.from(card.querySelectorAll("button")).find(
        (candidate) => (candidate.textContent || "").trim() === ${JSON.stringify(text)}
      );
      if (!node) return null;
      node.scrollIntoView({ block: "center", inline: "center" });
      const r = node.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    })()`);
    if (!rect) throw new Error(`Card ${cardIndex} button not found: ${text}`);
    await sleep(150);
    const updated = await this.evaluate(`(() => {
      const card = document.querySelectorAll('[data-testid="product-card"]')[${cardIndex}];
      const node = Array.from(card?.querySelectorAll("button") ?? []).find(
        (candidate) => (candidate.textContent || "").trim() === ${JSON.stringify(text)}
      );
      if (!node) return null;
      const r = node.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    })()`);
    const point = updated ?? rect;
    await this.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: point.x, y: point.y });
    await this.send("Input.dispatchMouseEvent", {
      type: "mousePressed", x: point.x, y: point.y, button: "left", clickCount: 1,
    });
    await this.send("Input.dispatchMouseEvent", {
      type: "mouseReleased", x: point.x, y: point.y, button: "left", clickCount: 1,
    });
  }

  async clickCardSelector(cardIndex, selector) {
    const rect = await this.evaluate(`(() => {
      const card = document.querySelectorAll('[data-testid="product-card"]')[${cardIndex}];
      const node = card?.querySelector(${JSON.stringify(selector)});
      if (!node) return null;
      node.scrollIntoView({ block: "center", inline: "center" });
      const r = node.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    })()`);
    if (!rect) throw new Error(`Card ${cardIndex} selector not found: ${selector}`);
    await sleep(150);
    const updated = await this.evaluate(`(() => {
      const card = document.querySelectorAll('[data-testid="product-card"]')[${cardIndex}];
      const node = card?.querySelector(${JSON.stringify(selector)});
      if (!node) return null;
      const r = node.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    })()`);
    const point = updated ?? rect;
    await this.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: point.x, y: point.y });
    await this.send("Input.dispatchMouseEvent", {
      type: "mousePressed", x: point.x, y: point.y, button: "left", clickCount: 1,
    });
    await this.send("Input.dispatchMouseEvent", {
      type: "mouseReleased", x: point.x, y: point.y, button: "left", clickCount: 1,
    });
  }

  async type(selector, text) {
    await this.evaluate(`(() => {
      const node = document.querySelector(${JSON.stringify(selector)});
      if (!node) return false;
      node.focus();
      return true;
    })()`);
    await this.send("Input.insertText", { text });
    await sleep(200);
    const value = await this.evaluate(`document.querySelector(${JSON.stringify(selector)})?.value ?? ""`);
    if (value !== text) {
      // React's controlled textarea can ignore a synthetic insertText in some headless paths;
      // fall back to the native setter plus a real input event.
      await this.evaluate(`(() => {
        const node = document.querySelector(${JSON.stringify(selector)});
        const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
        setter.call(node, ${JSON.stringify(text)});
        node.dispatchEvent(new Event("input", { bubbles: true }));
      })()`);
    }
  }
}

async function httpJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = { raw: text };
  }
  if (!response.ok) throw new Error(`${url} -> ${response.status}: ${text.slice(0, 400)}`);
  return body;
}

function safeText(value) {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

async function api(path, options = {}) {
  return httpJson(`${BFF}${path}`, options);
}

async function getSession(sessionId) {
  return api(`/api/sessions/${encodeURIComponent(sessionId)}`);
}

async function getProfile(sessionId) {
  return api(`/api/behavior/profile?session_id=${encodeURIComponent(sessionId)}`);
}

async function getShadow(sessionId, recommendationId, topK = 10) {
  const params = new URLSearchParams({
    session_id: sessionId,
    recommendation_id: recommendationId,
    top_k: String(topK),
  });
  return api(`/api/behavior/shadow?${params.toString()}`);
}

async function latestRecommendationId(sessionId) {
  const view = await getSession(sessionId);
  const turns = Array.isArray(view.turns) ? view.turns : [];
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    const payload = turns[index]?.products;
    if (payload?.recommendation_id) return payload.recommendation_id;
  }
  return view.latest_recommendation_id ?? null;
}

async function waitComposerReady(cdp) {
  await cdp.waitFor(
    `(() => {
      const input = document.querySelector('[data-testid="composer-input"]');
      return Boolean(input && !input.disabled);
    })()`,
    { timeoutMs: 180_000, label: "composer enabled" },
  );
}

async function newSession(cdp) {
  const before = await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`);
  await cdp.clickSelector('[data-testid="new-conversation"]');
  await cdp.waitFor(
    `(() => {
      const value = window.localStorage.getItem("agentrecx.web.session_id");
      return Boolean(value && value !== ${JSON.stringify(before)});
    })()`,
    { timeoutMs: 30_000, label: "new session id" },
  );
  await waitComposerReady(cdp);
  await sleep(900);
  return cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`);
}

async function sendQuery(cdp, text) {
  await waitComposerReady(cdp);
  const beforeTurns = await cdp.evaluate(`document.querySelectorAll('[data-testid="turn"]').length`);
  await cdp.type('[data-testid="composer-input"]', text);
  await cdp.clickSelector('[data-testid="composer-send"]');
  await cdp.waitFor(
    `document.querySelectorAll('[data-testid="turn"]').length > ${beforeTurns}`,
    { timeoutMs: 120_000, label: "new turn" },
  );
  await cdp.waitFor(
    `(() => {
      const turns = Array.from(document.querySelectorAll('[data-testid="turn"]'));
      const last = turns[turns.length - 1];
      return Boolean(last && last.querySelector('[data-testid="products-block"]'));
    })()`,
    { timeoutMs: 180_000, label: "products in latest turn" },
  );
  await cdp.waitFor(`!document.querySelector('[data-testid="composer-stop"]')`, {
    timeoutMs: 180_000, label: "turn finished",
  });
  await sleep(1_200);
}

async function latestProducts(cdp) {
  return cdp.evaluate(`(() => {
    const block = Array.from(document.querySelectorAll('[data-testid="products-block"]')).at(-1);
    if (!block) return [];
    return Array.from(block.querySelectorAll('[data-testid="product-card"]')).map((card) => {
      const titleButton = card.querySelector('button[aria-label^="Open details for "]');
      return {
        product_id: card.getAttribute("data-product-id"),
        title: (titleButton?.getAttribute("aria-label") || "").replace("Open details for ", "").slice(0, 180),
      };
    });
  })()`);
}

async function scrollLatestCards(cdp, limit = 10) {
  const count = await cdp.evaluate(`(() => {
    const block = Array.from(document.querySelectorAll('[data-testid="products-block"]')).at(-1);
    return block ? block.querySelectorAll('[data-testid="product-card"]').length : 0;
  })()`);
  for (let index = 0; index < Math.min(count, limit); index += 1) {
    await cdp.evaluate(`(() => {
      const block = Array.from(document.querySelectorAll('[data-testid="products-block"]')).at(-1);
      const card = block?.querySelectorAll('[data-testid="product-card"]')[${index}];
      card?.scrollIntoView({ block: "center" });
      return true;
    })()`);
    await sleep(1_250);
  }
  await sleep(700);
  return count;
}

async function clickAt(cdp, point) {
  await cdp.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: point.x, y: point.y });
  await cdp.send("Input.dispatchMouseEvent", {
    type: "mousePressed", x: point.x, y: point.y, button: "left", clickCount: 1,
  });
  await cdp.send("Input.dispatchMouseEvent", {
    type: "mouseReleased", x: point.x, y: point.y, button: "left", clickCount: 1,
  });
}

async function clickLatestCardButton(cdp, cardIndex, text) {
  const expression = `(() => {
    const block = Array.from(document.querySelectorAll('[data-testid="products-block"]')).at(-1);
    const card = block?.querySelectorAll('[data-testid="product-card"]')[${cardIndex}];
    if (!card) return null;
    const node = Array.from(card.querySelectorAll("button")).find(
      (candidate) => (candidate.textContent || "").trim() === ${JSON.stringify(text)}
    );
    if (!node) return null;
    node.scrollIntoView({ block: "center", inline: "center" });
    const r = node.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  })()`;
  const rect = await cdp.evaluate(expression);
  if (!rect) throw new Error(`latest card ${cardIndex}: button ${text} not found`);
  await sleep(150);
  await clickAt(cdp, rect);
}

async function clickLatestCardSelector(cdp, cardIndex, selector) {
  const expression = `(() => {
    const block = Array.from(document.querySelectorAll('[data-testid="products-block"]')).at(-1);
    const card = block?.querySelectorAll('[data-testid="product-card"]')[${cardIndex}];
    const node = card?.querySelector(${JSON.stringify(selector)});
    if (!node) return null;
    node.scrollIntoView({ block: "center", inline: "center" });
    const r = node.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  })()`;
  const rect = await cdp.evaluate(expression);
  if (!rect) throw new Error(`latest card ${cardIndex}: selector ${selector} not found`);
  await sleep(150);
  await clickAt(cdp, rect);
}


async function mainV2() {
  const targets = await httpJson(`${CHROME}/json/list`);
  const page = targets.find((target) => target.type === "page");
  if (!page) throw new Error("No Chrome page target found on the debugging port");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, { once: true });
    ws.addEventListener("error", reject, { once: true });
  });
  const cdp = new Cdp(ws);
  const consoleErrors = [];
  cdp.on("Runtime.exceptionThrown", (params) => consoleErrors.push(params.exceptionDetails?.text ?? "Runtime exception"));
  cdp.on("Log.entryAdded", (params) => {
    if (params.entry?.level === "error") consoleErrors.push(params.entry.text);
  });
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  await cdp.send("Log.enable");
  await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });

  const evidence = {
    schema: "agentrecx.behavior_feedback_v2.browser_e2e.v1",
    started_at: new Date().toISOString(),
    app_url: APP,
    bff_url: BFF,
    chrome: (await httpJson(`${CHROME}/json/version`)).Browser,
    mode: "qwen_live",
    search_mode_expected: "bge",
    steps: [],
    screenshots: [],
    console_errors: consoleErrors,
  };
  await cdp.send("Page.navigate", { url: APP });
  await cdp.waitFor(
    `(() => {
      const chip = document.querySelector('[data-testid="readiness-chip"]');
      const input = document.querySelector('[data-testid="composer-input"]');
      return Boolean(chip && chip.getAttribute("data-state") === "ready" && input && !input.disabled);
    })()`,
    { timeoutMs: 240_000, label: "BFF ready and composer enabled" },
  );
  await sleep(800);
  // This flag is set only by this CDP automation; the product UI never sets it.
  await cdp.evaluate("window.__AGENTRECX_AUTOMATED_BROWSER_E2E__ = true");
  await cdp.clickSelector('[data-testid="mode-qwen-live"]');
  await cdp.waitFor(`Boolean(document.querySelector('[data-testid="qwen-live-badge"]'))`, {
    timeoutMs: 30_000, label: "Qwen Live badge",
  });
  evidence.qwen_live_badge = await cdp.evaluate(`document.querySelector('[data-testid="qwen-live-badge"]').textContent.trim()`);
  const sessionId = await newSession(cdp);
  evidence.session_id = sessionId;

  await sendQuery(cdp, "I need wireless headphones for the gym.");
  const firstProducts = await latestProducts(cdp);
  const firstRec = await latestRecommendationId(sessionId);
  const firstTurn = (await getSession(sessionId)).turns?.at(-1);
  evidence.turn1 = {
    recommendation_id: firstRec,
    product_count: firstProducts.length,
    products: firstProducts.slice(0, 5),
    search_mode: firstTurn?.products?.search_mode,
    score_kind: firstTurn?.products?.score_kind,
    gate_action: firstTurn?.products?.gate?.action,
    planner: firstTurn?.planner,
    planner_calls: firstTurn?.planner_calls,
    planner_actions: firstTurn?.planner_actions,
  };
  if (firstTurn?.planner !== "qwen_base") throw new Error(`turn1 planner is ${firstTurn?.planner}, expected qwen_base`);
  if (!firstTurn?.planner_actions?.length) throw new Error("turn1 has no raw Qwen planner actions");
  if (firstTurn?.products?.search_mode !== "bge" || firstTurn?.products?.score_kind !== "bge_dense") {
    throw new Error(`turn1 is not BGE: ${firstTurn?.products?.search_mode}/${firstTurn?.products?.score_kind}`);
  }
  await scrollLatestCards(cdp, 10);
  const targetIndex = 4;
  await clickLatestCardButton(cdp, targetIndex, "View details");
  await sleep(700);
  await clickLatestCardSelector(cdp, targetIndex, 'button[title^="Like this item"]');
  await sleep(1_800);
  evidence.interaction = { clicked_index: targetIndex, liked_product: firstProducts[targetIndex]?.product_id };
  const firstProfile = await getProfile(sessionId);
  evidence.profile_after_interaction = {
    counts: firstProfile.counts,
    source_counts: firstProfile.source_counts,
    profile_provenance: firstProfile.profile_provenance,
    attribute_interests: firstProfile.attribute_interests?.map((item) => ({
      attribute: item.attribute, score: item.score, distinct_products: item.distinct_products,
    })),
  };
  const browserEvents = (firstProfile.recent_events || []).filter((item) => item.enters_profile);
  if (!browserEvents.some((item) => item.event_source === "AUTOMATED_BROWSER_E2E")) {
    throw new Error("browser events were not recorded as AUTOMATED_BROWSER_E2E");
  }
  if ((firstProfile.counts?.human_live_events || 0) !== 0) {
    throw new Error("automated browser E2E leaked into human LIVE_USER_INTERACTION counts");
  }
  evidence.screenshots.push(await cdp.screenshot("v2-qwen-live-after-like.png"));

  await sendQuery(cdp, "I need headphones with a microphone for calls.");
  const secondProducts = await latestProducts(cdp);
  const secondRec = await latestRecommendationId(sessionId);
  const secondTurn = (await getSession(sessionId)).turns?.at(-1);
  evidence.turn2 = {
    recommendation_id: secondRec,
    product_count: secondProducts.length,
    search_mode: secondTurn?.products?.search_mode,
    score_kind: secondTurn?.products?.score_kind,
    gate_action: secondTurn?.products?.gate?.action,
    planner: secondTurn?.planner,
    planner_calls: secondTurn?.planner_calls,
    planner_actions: secondTurn?.planner_actions,
  };
  if (secondTurn?.planner !== "qwen_base") throw new Error(`turn2 planner is ${secondTurn?.planner}, expected qwen_base`);
  if (!secondTurn?.planner_actions?.length) throw new Error("turn2 has no raw Qwen planner actions");
  if (secondTurn?.products?.search_mode !== "bge") throw new Error("turn2 is not BGE");
  const profile = await getProfile(sessionId);
  const shadow = await getShadow(sessionId, secondRec, 10);
  evidence.profile = {
    counts: profile.counts,
    source_counts: profile.source_counts,
    profile_provenance: profile.profile_provenance,
    attribute_interests: profile.attribute_interests?.map((item) => ({
      attribute: item.attribute, score: item.score, distinct_products: item.distinct_products,
    })),
  };
  evidence.shadow = {
    recommendation_id: secondRec,
    eligible_candidate_count: shadow.eligible_candidate_count,
    eligible_ranking_changed: shadow.eligible_ranking_changed,
    eligible_changes: shadow.eligible_changes,
    eligible_base_ranking: shadow.eligible_base_ranking,
    eligible_behavior_ranking: shadow.eligible_behavior_ranking,
    gate_context: shadow.gate_context,
    hard_constraint_violations: shadow.hard_constraint_violations,
    eligibility_unchanged: shadow.eligibility_unchanged,
    ineligible_candidates_with_behavior_signal: shadow.ineligible_candidates_with_behavior_signal,
  };
  if (shadow.hard_constraint_violations !== 0) throw new Error("shadow hard constraint violation");
  if (shadow.eligibility_unchanged !== true) throw new Error("behavior shadow changed Gate eligibility");
  evidence.screenshots.push(await cdp.screenshot("v2-browser-e2e-final.png"));
  evidence.finished_at = new Date().toISOString();
  evidence.console_errors = consoleErrors;
  const outFile = path.join(OUT_DIR, "browser_e2e_results.json");
  fs.writeFileSync(outFile, JSON.stringify(evidence, null, 2));
  console.log(JSON.stringify({
    out_file: outFile,
    session_id: sessionId,
    qwen_turn1: evidence.turn1.planner_calls,
    qwen_turn2: evidence.turn2.planner_calls,
    search_modes: [evidence.turn1.search_mode, evidence.turn2.search_mode],
    profile_counts: evidence.profile.counts,
    eligible_count: evidence.shadow.eligible_candidate_count,
    eligible_changed: evidence.shadow.eligible_ranking_changed,
    console_errors: consoleErrors,
  }, null, 2));
  ws.close();
}

mainV2().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
