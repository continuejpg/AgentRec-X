// AgentRec-X Behavior Feedback V1 - real Chrome browser E2E.
//
//   node scripts/behavior_feedback_browser_e2e.mjs
//
// Requires: Next.js on http://127.0.0.1:3000, BFF on http://127.0.0.1:8787, and a Chrome
// started with --remote-debugging-port=9222 (the wrapper script starts it).
//
// The script drives a real headless Chrome via the DevTools Protocol: it types a query,
// scrolls product cards into view (IntersectionObserver impressions), clicks a card,
// likes an item, compares two cards, opens the Behavior Inspector, sends a second query
// and verifies the next Behavior Shadow, then creates a fresh session and verifies
// isolation.  Screenshots and a JSON evidence file are written under runs/behavior_feedback_v1.

import fs from "node:fs";
import path from "node:path";

const CHROME = process.env.CHROME_DEBUG_URL ?? "http://127.0.0.1:9222";
const APP = process.env.NEXT_PUBLIC_URL ?? "http://127.0.0.1:3000/";
const OUT_DIR = path.resolve(process.env.BEHAVIOR_E2E_OUT ?? "runs/behavior_feedback_v1");
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

async function main() {
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
    schema: "agentrecx.behavior_feedback.browser_e2e.v1",
    started_at: new Date().toISOString(),
    app_url: APP,
    chrome: (await httpJson(`${CHROME}/json/version`)).Browser,
    steps: [],
    screenshots: [],
    console_errors: consoleErrors,
  };

  const recordStep = (name, detail) => {
    evidence.steps.push({ name, at: new Date().toISOString(), ...detail });
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
  const sessionIdA = await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`);
  evidence.session_a = sessionIdA;
  recordStep("ready", { session_id: sessionIdA });
  evidence.screenshots.push(await cdp.screenshot("00-ready.png"));

  // 1. Real query through the chat UI.
  await cdp.type('[data-testid="composer-input"]', "I need wireless earbuds for the gym.");
  await cdp.clickSelector('[data-testid="composer-send"]');
  await cdp.waitFor(`Boolean(document.querySelector('[data-testid="products-block"]'))`, {
    timeoutMs: 120_000, label: "products block",
  });
  await cdp.waitFor(`!document.querySelector('[data-testid="composer-stop"]')`, {
    timeoutMs: 180_000, label: "turn finished",
  });
  await sleep(1_000);
  const products = await cdp.evaluate(`(() => {
    const cards = Array.from(document.querySelectorAll('[data-testid="product-card"]'));
    return cards.map((card) => {
      const titleButton = card.querySelector('button[aria-label^="Open details for "]');
      return {
        product_id: card.getAttribute("data-product-id"),
        title: (titleButton?.getAttribute("aria-label") || "").replace("Open details for ", "").slice(0, 180),
      };
    });
  })()`);
  recordStep("query_1", { products });
  evidence.products_first = products;
  evidence.screenshots.push(await cdp.screenshot("01-query-products.png"));

  // 2. Real visibility: scroll each card into view and let IntersectionObserver fire.
  const cardCount = products.length;
  for (let index = 0; index < Math.min(cardCount, 6); index += 1) {
    await cdp.evaluate(`(() => {
      const card = document.querySelectorAll('[data-testid="product-card"]')[${index}];
      card?.scrollIntoView({ block: "center" });
      return true;
    })()`);
    await sleep(1_350);
  }
  await sleep(800);

  // 3. Click a tail card (not the first already-promoted card) and like it.  Choosing a
  // lower-ranked item gives the next shadow actual rank headroom; clicking rank 1 could only
  // confirm an already-top recommendation.
  const targetIndex = Math.max(2, Math.min(products.length - 2, 4));
  await cdp.clickCardButton(targetIndex, "View details");
  await sleep(900);
  await cdp.clickCardSelector(targetIndex, 'button[title^="Like this item"]');
  await sleep(600);
  await cdp.screenshot("02-details-and-like.png");
  recordStep("click_and_like", { product_id: products[targetIndex]?.product_id, card_index: targetIndex });

  // 4. Compare the first two cards through the real UI.  Click sequentially: React state
  // updates are asynchronous, so two clicks in the same evaluate() would both read the
  // pre-click selection.
  await cdp.clickCardButton(0, "Select");
  await sleep(450);
  await cdp.clickCardButton(1, "Select");
  await sleep(500);
  await cdp.clickButtonByText("Compare selected (2)");
  recordStep("compare_started", { selected: 2 });
  await sleep(5_000);
  evidence.screenshots.push(await cdp.screenshot("03-compare.png"));

  // 5. Open the Behavior Inspector and read the real profile.
  await cdp.clickSelector('[data-testid="open-behavior"]');
  await cdp.waitFor(`Boolean(document.querySelector('[data-testid="behavior-inspector"]'))`, {
    timeoutMs: 30_000, label: "behavior inspector",
  });
  await cdp.waitFor(`Boolean(document.querySelector('[data-testid="behavior-counts"]'))`, {
    timeoutMs: 30_000, label: "behavior counts",
  });
  await sleep(1_200);
  const inspectorText = safeText(await cdp.evaluate(`document.querySelector('[data-testid="behavior-inspector"]')?.innerText`));
  recordStep("behavior_inspector", { text: inspectorText.slice(0, 1200) });
  evidence.inspector_text = inspectorText;
  evidence.screenshots.push(await cdp.screenshot("04-behavior-inspector.png"));

  // 6. Second recommendation: the next shadow uses only events recorded before it.
  await cdp.type('[data-testid="composer-input"]', "I need wireless earbuds for the gym.");
  await cdp.clickSelector('[data-testid="composer-send"]');
  await cdp.waitFor(`document.querySelectorAll('[data-testid="products-block"]').length >= 2`, {
    timeoutMs: 120_000, label: "second products block",
  });
  await cdp.waitFor(`!document.querySelector('[data-testid="composer-stop"]')`, {
    timeoutMs: 180_000, label: "second turn finished",
  });
  await sleep(1_500);
  // Re-open/focus the Behavior panel in case the panel tab changed.
  await cdp.clickSelector('[data-testid="open-behavior"]');
  await sleep(2_000);
  const shadowText = safeText(await cdp.evaluate(`document.querySelector('[data-testid="behavior-shadow"]')?.innerText`));
  evidence.shadow_text = shadowText;
  await cdp.evaluate(`document.querySelector('[data-testid="behavior-shadow"]')?.scrollIntoView({ block: "start" })`);
  await sleep(600);
  recordStep("shadow_after_second_query", { text: shadowText });
  evidence.screenshots.push(await cdp.screenshot("05-next-shadow.png"));

  // 7. API-level cross-check of the same session and a fresh isolated session.
  const profileA = await httpJson(`http://127.0.0.1:8787/api/behavior/profile?session_id=${encodeURIComponent(sessionIdA)}`);
  evidence.profile_a = {
    counts: profileA.counts,
    attribute_interests: profileA.attribute_interests,
    latest_recommendation_id: profileA.latest_recommendation_id,
    suggestion: profileA.suggestion,
  };

  // New conversation button: fresh opaque session, same browser.
  await cdp.clickSelector('[data-testid="new-conversation"]');
  await cdp.waitFor(
    `window.localStorage.getItem("agentrecx.web.session_id") && window.localStorage.getItem("agentrecx.web.session_id") !== ${JSON.stringify(sessionIdA)}`,
    { timeoutMs: 30_000, label: "new session id" },
  );
  await sleep(1_000);
  const sessionIdB = await cdp.evaluate(`window.localStorage.getItem("agentrecx.web.session_id")`);
  evidence.session_b = sessionIdB;
  await cdp.clickSelector('[data-testid="open-behavior"]');
  await sleep(1_500);
  const inspectorBText = safeText(await cdp.evaluate(`document.querySelector('[data-testid="behavior-inspector"]')?.innerText`));
  const profileB = await httpJson(`http://127.0.0.1:8787/api/behavior/profile?session_id=${encodeURIComponent(sessionIdB)}`);
  evidence.profile_b = {
    counts: profileB.counts,
    attribute_interests: profileB.attribute_interests,
    recent_events: profileB.recent_events.length,
  };
  recordStep("new_session_isolation", {
    session_id: sessionIdB,
    count_text: inspectorBText.slice(0, 500),
    profile_counts: profileB.counts,
  });
  evidence.screenshots.push(await cdp.screenshot("06-new-session-isolated.png"));

  // 8. Toggle personalization off in the new session and verify events are rejected.
  await cdp.clickSelector('[data-testid="behavior-toggle"]');
  await sleep(1_200);
  const toggleText = safeText(await cdp.evaluate(`document.querySelector('[data-testid="behavior-toggle"]')?.innerText`));
  recordStep("toggle_off", { toggle_text: toggleText });
  evidence.screenshots.push(await cdp.screenshot("07-personalization-off.png"));

  evidence.finished_at = new Date().toISOString();
  evidence.console_errors = consoleErrors;
  const outFile = path.join(OUT_DIR, "browser_e2e.json");
  fs.writeFileSync(outFile, JSON.stringify(evidence, null, 2));
  console.log(JSON.stringify({ out_file: outFile, session_a: sessionIdA, session_b: sessionIdB, products: products.length, inspector_text: inspectorText.slice(0, 700) }, null, 2));
  ws.close();
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
