// AgentRec-X Behavior Feedback V1.1 - real Chrome browser E2E (scenarios A-F).
//
//   node scripts/behavior_feedback_browser_e2e.mjs
//
// Requires: Next.js on http://127.0.0.1:3000, BFF on http://127.0.0.1:8787, and a Chrome
// started with --remote-debugging-port=9222.  The script drives real headless Chrome via the
// DevTools Protocol: impression-only, click+Like, compare attempt/failure/success, ineligible
// diagnostic, session isolation and Personalization OFF.  Screenshots and JSON evidence are
// written under runs/behavior_feedback_v11.

import fs from "node:fs";
import path from "node:path";

const CHROME = process.env.CHROME_DEBUG_URL ?? "http://127.0.0.1:9222";
const APP = process.env.NEXT_PUBLIC_URL ?? "http://127.0.0.1:3000/";
const OUT_DIR = path.resolve(process.env.BEHAVIOR_E2E_OUT ?? "runs/behavior_feedback_v11");
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

async function openBehavior(cdp) {
  await cdp.clickSelector('[data-testid="open-behavior"]');
  await cdp.waitFor(`Boolean(document.querySelector('[data-testid="behavior-inspector"]'))`, {
    timeoutMs: 30_000, label: "behavior inspector",
  });
  await sleep(900);
}

async function inspectorText(cdp) {
  return safeText(await cdp.evaluate(
    `document.querySelector('[data-testid="behavior-inspector"]')?.innerText`,
  ));
}

async function scrollBehaviorShadow(cdp) {
  await cdp.evaluate(`document.querySelector('[data-testid="behavior-shadow"]')?.scrollIntoView({ block: "start" })`);
  await sleep(500);
}

async function compareSelectedPair(cdp) {
  await clickLatestCardButton(cdp, 0, "Select");
  await sleep(450);
  await clickLatestCardButton(cdp, 1, "Select");
  await sleep(450);
  await cdp.clickButtonByText("Compare selected (2)");
  await sleep(6_000);
  return cdp.evaluate(`(() => {
    const error = document.querySelector('[data-testid="comparison-error"]')?.innerText || "";
    const panel = document.querySelector('[data-testid="right-panel"]')?.innerText || "";
    return { error: error.slice(0, 800), panel_tail: panel.slice(-900) };
  })()`);
}

function shadowSummary(shadow) {
  return {
    schema: shadow.schema,
    recommendation_id: shadow.recommendation_id,
    query: shadow.query,
    gate_context: shadow.gate_context,
    eligible_candidate_count: shadow.eligible_candidate_count,
    eligible_recommendation_available: shadow.eligible_recommendation_available,
    behavior_signal_present: shadow.behavior_signal_present,
    eligible_behavior_applied: shadow.eligible_behavior_applied,
    diagnostic_behavior_applied: shadow.diagnostic_behavior_applied,
    eligible_ranking_changed: shadow.eligible_ranking_changed,
    diagnostic_ranking_changed: shadow.diagnostic_ranking_changed,
    eligible_base_ranking: shadow.eligible_base_ranking,
    eligible_behavior_ranking: shadow.eligible_behavior_ranking,
    eligible_changes: shadow.eligible_changes,
    diagnostic_ranking: shadow.diagnostic_ranking,
    diagnostic_candidates: shadow.diagnostic_candidates,
    ineligible_candidates_with_behavior_signal: shadow.ineligible_candidates_with_behavior_signal,
    hard_constraint_violations: shadow.hard_constraint_violations,
    eligibility_unchanged: shadow.eligibility_unchanged,
    active_preferences: shadow.active_preferences,
    notes: shadow.notes,
  };
}

function profileSummary(profile) {
  return {
    schema: profile.schema,
    semantics_version: profile.semantics_version,
    enabled: profile.enabled,
    counts: profile.counts,
    source_counts: profile.source_counts,
    signal_semantics: profile.signal_semantics,
    recent_impressions: profile.recent_impressions,
    recent_viewed: profile.recent_viewed,
    recent_compared: profile.recent_compared,
    recent_compare_attempts: profile.recent_compare_attempts,
    recent_explicit_feedback: profile.recent_explicit_feedback,
    attribute_interests: profile.attribute_interests,
    product_interests: profile.product_interests,
    latest_recommendation_id: profile.latest_recommendation_id,
    suggestion: profile.suggestion,
    explicit_memory: profile.explicit_memory,
  };
}

async function captureScenario(cdp, evidence, key, name, extra = {}) {
  const sessionId = await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`);
  const recommendationId = await latestRecommendationId(sessionId);
  const profile = await getProfile(sessionId);
  const shadow = await getShadow(sessionId, recommendationId);
  const text = await inspectorText(cdp);
  const record = {
    name,
    session_id: sessionId,
    recommendation_id: recommendationId,
    profile: profileSummary(profile),
    shadow: shadowSummary(shadow),
    inspector_text: text,
    captured_at: new Date().toISOString(),
    ...extra,
  };
  evidence.scenarios[key] = record;
  evidence.steps.push({ name: key, at: new Date().toISOString(), ...extra });
  return record;
}

async function scenarioImpressionOnly(cdp, evidence) {
  const sid = await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`);
  await sendQuery(cdp, "I need wireless headphones for the gym.");
  const products = await latestProducts(cdp);
  await scrollLatestCards(cdp, 10);
  await sleep(1_500);
  await openBehavior(cdp);
  await cdp.screenshot("A-impression-only.png");
  const record = await captureScenario(cdp, evidence, "A_impression_only", "impression only", {
    session_id: sid,
    products,
  });
  if (!record.profile.counts.product_impression) throw new Error("Scenario A: no real impression recorded");
  if (record.profile.counts.product_click) throw new Error("Scenario A: unexpected click");
  if (record.profile.product_interests.length) throw new Error("Scenario A: impression leaked into product interest");
  if (record.profile.attribute_interests.length) throw new Error("Scenario A: impression leaked into attribute interest");
  if (record.shadow.eligible_behavior_applied) throw new Error("Scenario A: behavior applied without interest");
  return record;
}

async function scenarioClickLike(cdp, evidence) {
  await newSession(cdp);
  await sendQuery(cdp, "I need wireless headphones for the gym.");
  const products = await latestProducts(cdp);
  await scrollLatestCards(cdp, 10);
  const targetIndex = Math.max(2, Math.min(products.length - 2, 4));
  await clickLatestCardButton(cdp, targetIndex, "View details");
  await sleep(700);
  await clickLatestCardSelector(cdp, targetIndex, 'button[title^="Like this item"]');
  await sleep(1_000);
  const firstRec = await latestRecommendationId(await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`));
  const firstSession = await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`);
  const oldShadow = shadowSummary(await getShadow(firstSession, firstRec));
  await sendQuery(cdp, "I need wireless headphones for the gym.");
  await openBehavior(cdp);
  await cdp.screenshot("B-click-like.png");
  await scrollBehaviorShadow(cdp);
  await cdp.screenshot("B-next-eligible-shadow.png");
  const record = await captureScenario(cdp, evidence, "B_click_like", "click + like", {
    products,
    target_product: products[targetIndex]?.product_id,
    point_in_time_shadow: oldShadow,
  });
  if (record.profile.counts.product_click !== 1) throw new Error("Scenario B: click not recorded exactly once");
  if (record.profile.counts.explicit_like !== 1) throw new Error("Scenario B: Like not recorded");
  if (!record.profile.attribute_interests.length) throw new Error("Scenario B: no weak attribute interest");
  if (record.profile.counts.product_compare_attempted) throw new Error("Scenario B: unexpected compare");
  return record;
}

async function scenarioCompareLifecycle(cdp, evidence) {
  await newSession(cdp);
  await sendQuery(cdp, "I need wireless earbuds for the gym.");
  const firstEarbudProducts = await latestProducts(cdp);
  await scrollLatestCards(cdp, 10);
  // Click/like an ineligible search card.  A second earbuds recommendation is then issued
  // so that the point-in-time shadow can legitimately use those pre-recommendation events.
  const diagIndex = Math.max(2, Math.min(firstEarbudProducts.length - 2, 8));
  await clickLatestCardButton(cdp, diagIndex, "View details");
  await sleep(600);
  await clickLatestCardSelector(cdp, diagIndex, 'button[title^="Like this item"]');
  await sleep(700);
  await sendQuery(cdp, "I need wireless earbuds for the gym.");
  const earbudProducts = await latestProducts(cdp);
  const failure = await compareSelectedPair(cdp);
  await sleep(1_000);
  const failureSession = await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`);
  const failureRec = await latestRecommendationId(failureSession);
  const failureProfile = await getProfile(failureSession);
  const failureShadow = await getShadow(failureSession, failureRec);
  await cdp.screenshot("C-compare-failure.png");
  await openBehavior(cdp);
  await scrollBehaviorShadow(cdp);
  await cdp.screenshot("D-diagnostic-vs-eligible.png");

  await sendQuery(cdp, "I need wireless headphones for the gym.");
  await scrollLatestCards(cdp, 10);
  const success = await compareSelectedPair(cdp);
  await sleep(1_000);
  await cdp.screenshot("C-compare-success.png");
  await openBehavior(cdp);
  await scrollBehaviorShadow(cdp);
  const successSession = await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`);
  const successRec = await latestRecommendationId(successSession);
  const successProfile = await getProfile(successSession);
  const successShadow = await getShadow(successSession, successRec);

  evidence.scenarios.C_compare_lifecycle = {
    name: "compare attempt / failure / success",
    session_id: successSession,
    failure: {
      recommendation_id: failureRec,
      ui: failure,
      profile: profileSummary(failureProfile),
      shadow: shadowSummary(failureShadow),
    },
    success: {
      recommendation_id: successRec,
      ui: success,
      profile: profileSummary(successProfile),
      shadow: shadowSummary(successShadow),
    },
    diag_product: earbudProducts[diagIndex]?.product_id,
    failure_products: earbudProducts,
    captured_at: new Date().toISOString(),
  };
  evidence.scenarios.D_eligibility = {
    name: "ineligible candidate diagnostic vs eligible ranking",
    recommendation_id: failureRec,
    gate_context: failureShadow.gate_context,
    eligible_candidate_ids: failureShadow.eligible_candidate_ids,
    diagnostic_candidates: failureShadow.diagnostic_candidates,
    ineligible_candidates_with_behavior_signal: failureShadow.ineligible_candidates_with_behavior_signal,
    eligible_ranking_ids: failureShadow.eligible_ranking_ids,
    hard_constraint_violations: failureShadow.hard_constraint_violations,
    eligibility_unchanged: failureShadow.eligibility_unchanged,
  };
  const counts = successProfile.counts;
  if (counts.product_compare_attempted !== 2) throw new Error("Scenario C: expected two compare attempts");
  if (counts.product_compare_completed !== 1) throw new Error("Scenario C: expected one completed compare");
  if (counts.product_compare_failed !== 1) throw new Error("Scenario C: expected one failed compare");
  if (failureShadow.eligible_candidate_count !== 1) throw new Error("Scenario D: earbuds gate should expose one eligible candidate");
  if (failureShadow.diagnostic_candidates.length <= failureShadow.eligible_candidate_count) {
    throw new Error("Scenario D: diagnostic list missing ineligible candidates");
  }
  if (failureShadow.ineligible_candidates_with_behavior_signal.length < 1) {
    throw new Error("Scenario D: ineligible clicked product has no diagnostic signal");
  }
  if (failureShadow.eligible_ranking_ids.some((id) => failureShadow.diagnostic_candidates.find((row) => row.product_id === id)?.gate_eligible === false)) {
    throw new Error("Scenario D: ineligible candidate entered eligible ranking");
  }
  return { successProfile, successShadow, failureShadow };
}

async function scenarioIsolation(cdp, evidence) {
  await newSession(cdp);
  await sendQuery(cdp, "I need wireless headphones for the gym.");
  await scrollLatestCards(cdp, 3);
  await sleep(1_500);
  await openBehavior(cdp);
  await cdp.screenshot("E-session-isolation.png");
  const record = await captureScenario(cdp, evidence, "E_isolation", "fresh session isolation");
  if (record.profile.counts.product_click || record.profile.counts.explicit_feedback) {
    throw new Error("Scenario E: fresh session leaked prior clicks/likes");
  }
  if (record.shadow.behavior_signal_present) throw new Error("Scenario E: shadow inherited another session profile");
  return record;
}

async function scenarioPersonalizationOff(cdp, evidence) {
  await openBehavior(cdp);
  await cdp.clickSelector('[data-testid="behavior-toggle"]');
  await cdp.waitFor(
    `(document.querySelector('[data-testid="behavior-toggle"]')?.innerText || "").includes("OFF")`,
    { timeoutMs: 20_000, label: "personalization OFF" },
  );
  const sessionId = await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`);
  const recBefore = await latestRecommendationId(sessionId);
  const rawEvent = {
    event_id: `evt-off-probe-${Date.now()}`,
    event_type: "product_impression",
    recommendation_id: recBefore,
    product_id: "B0C1NGPMVP",
    displayed_position: 1,
    event_source: "LIVE_USER_INTERACTION",
  };
  const probe = await fetch(`${BFF}/api/behavior/events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, events: [rawEvent] }),
  });
  const probeBody = await probe.text();
  await sendQuery(cdp, "I need wireless headphones for the gym.");
  await sleep(1_500);
  const profile = await getProfile(sessionId);
  const recAfter = await latestRecommendationId(sessionId);
  const shadow = await getShadow(sessionId, recAfter);
  await cdp.screenshot("F-personalization-off.png");
  const record = {
    name: "personalization off",
    session_id: sessionId,
    events_endpoint_status: probe.status,
    events_endpoint_body: probeBody.slice(0, 500),
    profile: profileSummary(profile),
    shadow: shadowSummary(shadow),
    products_still_available: (await latestProducts(cdp)).length > 0,
    captured_at: new Date().toISOString(),
  };
  evidence.scenarios.F_personalization_off = record;
  if (probe.status !== 409) throw new Error(`Scenario F: expected 409 behavior_disabled, got ${probe.status}`);
  if (profile.enabled) throw new Error("Scenario F: profile remained enabled");
  if (!record.products_still_available) throw new Error("Scenario F: normal recommendation broke");
  return record;
}

async function mainV11() {
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
  cdp.on("Runtime.exceptionThrown", (params) => {
    consoleErrors.push(params.exceptionDetails?.text ?? "Runtime exception");
  });
  cdp.on("Log.entryAdded", (params) => {
    if (params.entry?.level === "error") consoleErrors.push(params.entry.text);
  });
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  await cdp.send("Log.enable");
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false,
  });

  const evidence = {
    schema: "agentrecx.behavior_feedback_v11.browser_e2e.v1",
    started_at: new Date().toISOString(),
    app_url: APP,
    bff_url: BFF,
    chrome: (await httpJson(`${CHROME}/json/version`)).Browser,
    scenarios: {},
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
    { timeoutMs: 180_000, label: "BFF ready and composer enabled" },
  );
  await sleep(1_200);
  evidence.sessions = {
    initial: await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`),
  };
  evidence.screenshots.push(await cdp.screenshot("00-ready.png"));

  evidence.scenarios.A_impression_only = await scenarioImpressionOnly(cdp, evidence);
  evidence.sessions.A = evidence.scenarios.A_impression_only.session_id;
  evidence.scenarios.B_click_like = await scenarioClickLike(cdp, evidence);
  evidence.sessions.B = evidence.scenarios.B_click_like.session_id;
  const compare = await scenarioCompareLifecycle(cdp, evidence);
  evidence.sessions.C = evidence.scenarios.C_compare_lifecycle.session_id;
  evidence.sessions.D = evidence.scenarios.C_compare_lifecycle.session_id;
  const isolation = await scenarioIsolation(cdp, evidence);
  evidence.sessions.E = isolation.session_id;
  const off = await scenarioPersonalizationOff(cdp, evidence);
  evidence.sessions.F = off.session_id;

  evidence.finished_at = new Date().toISOString();
  evidence.console_errors = consoleErrors;
  const outFile = path.join(OUT_DIR, "browser_e2e.json");
  fs.writeFileSync(outFile, JSON.stringify(evidence, null, 2));
  fs.writeFileSync(
    path.join(OUT_DIR, "eligible_off_on_comparison.json"),
    JSON.stringify({
      schema: "agentrecx.behavior_feedback_v11.eligible_off_on.v1",
      generated_at: new Date().toISOString(),
      scenarios: {
        A: evidence.scenarios.A_impression_only.shadow,
        B: evidence.scenarios.B_click_like.shadow,
        C_failure: evidence.scenarios.C_compare_lifecycle.failure.shadow,
        C_success: evidence.scenarios.C_compare_lifecycle.success.shadow,
        E: evidence.scenarios.E_isolation.shadow,
        F: evidence.scenarios.F_personalization_off.shadow,
      },
    }, null, 2),
  );
  console.log(JSON.stringify({
    out_file: outFile,
    sessions: evidence.sessions,
    A_impressions: evidence.scenarios.A_impression_only.profile.counts.product_impression,
    A_interests: evidence.scenarios.A_impression_only.profile.attribute_interests.length,
    B_eligible_changed: evidence.scenarios.B_click_like.shadow.eligible_ranking_changed,
    C_counts: evidence.scenarios.C_compare_lifecycle.success.profile.counts,
    D_eligible_count: evidence.scenarios.D_eligibility.eligible_candidate_count,
    D_diagnostic_count: evidence.scenarios.D_eligibility.diagnostic_candidates.length,
    E_events: evidence.scenarios.E_isolation.profile.counts.live_events,
    F_off_status: evidence.scenarios.F_personalization_off.events_endpoint_status,
    console_errors: consoleErrors,
  }, null, 2));
  ws.close();
}

mainV11().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
