/* AgentRec-X demo - four-source fusion and Agent dispatch views.
 *
 * Additive by design: this file is loaded after app.js and never touches the existing chat
 * demo's DOM, ids or listeners.  It inserts a tab bar and two new views, and toggles the
 * visibility of the page's original content when another tab is active.
 *
 * Every number on screen comes from a live API call:
 *   GET  /health                     - engine, fusion profile, TIGER worker state
 *   POST /v1/recommend               - four-source fused recommendations + provenance
 *   POST /v1/demo/agent/recommend    - Agent source dispatch (message + history + sources)
 *
 * Nothing is synthesised: no titles, images, confidences or reasons are invented.  Where the
 * API does not provide something (product titles today), the UI says so.
 */
(function () {
  "use strict";

  var ROOT_HEALTH = "/health";
  var FUSION = "/v1/recommend";
  var AGENT = "/v1/demo/agent/recommend";
  var TITLES = "/v1/catalogue/titles";

  // Serving source names (GET /health: fusion_sources) - what the fusion engine actually runs.
  var SERVING_SOURCES = [
    { id: "sequential", label: "SASRec", kind: "Sequential recommendation" },
    { id: "two_tower", label: "Two-Tower", kind: "Vector recall" },
    { id: "catalog_lexical", label: "Catalog Lexical", kind: "Text retrieval" },
    { id: "tiger", label: "TIGER", kind: "Semantic-ID generative retrieval" }
  ];

  // Agent plan names (POST /v1/demo/agent/recommend: sources[]) - a DIFFERENT vocabulary.
  // history/catalog_search/two_tower/tiger come from DemoSourcePlanPolicy.SOURCE_PLANS.
  var AGENT_SOURCES = [
    { id: "history", label: "history (SASRec)" },
    { id: "catalog_search", label: "catalog_search (BM25)" },
    { id: "two_tower", label: "two_tower" },
    { id: "tiger", label: "tiger (needs a ready worker)" }
  ];

  // A history verified over real HTTP during the T-02 acceptance rounds.  It is a real frozen
  // cohort history, not a fabricated identity.
  // The ten identities used for the verified real TIGER Agent HTTP request.  Real values from the
  // frozen cohort; no test target is included and no recommendation is pre-computed.
  var SAMPLE_HISTORY = [
    "B07C1D8DKL", "B01KI10HVM", "B0C4MNRN7R", "B077L2HYKH", "B016YLUXLE",
    "B08CTBYWFG", "B0BHNPVSQ3", "B071RB7QKX", "B000AO7JIW", "B075V9TG8V"
  ];

  var state = { health: null, view: "chat", busy: false };
  var originalNodes = [];
  var els = { titleNodes: [] };

  // ---------------------------------------------------------------- helpers

  function h(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function num(value) {
    return typeof value === "number" && isFinite(value) ? value : null;
  }

  function shortHash(value) {
    return typeof value === "string" && value.length > 12 ? value.slice(0, 12) + "..." : value;
  }

  function request(url, options) {
    var settings = options || {};
    return fetch(url, {
      method: settings.method || "GET",
      headers: settings.body ? { "content-type": "application/json" } : undefined,
      body: settings.body ? JSON.stringify(settings.body) : undefined
    }).then(function (response) {
      var parse = response.headers.get("content-type") &&
        response.headers.get("content-type").indexOf("application/json") >= 0
        ? response.json().catch(function () { return null; })
        : response.text().then(function (text) { return { detail: text }; });
      return parse.then(function (payload) {
        if (!response.ok) {
          var detail = payload && payload.detail ? payload.detail : "HTTP " + response.status;
          throw new Error(String(detail));
        }
        return payload;
      });
    });
  }

  function parseHistory(text) {
    return String(text || "")
      .split(/[\s,;]+/)
      .map(function (part) { return part.trim(); })
      .filter(function (part) { return part.length > 0; });
  }

  /** Render any API object as a definition list: unknown fields are shown, never invented. */
  function renderObject(parent, value, depth) {
    var level = depth || 0;
    if (value === null || value === undefined) {
      parent.appendChild(h("div", "p4-kv", "-"));
      return;
    }
    if (typeof value !== "object") {
      parent.appendChild(h("div", "p4-kv", String(value)));
      return;
    }
    if (Array.isArray(value)) {
      if (!value.length) {
        parent.appendChild(h("div", "p4-kv", "(empty)"));
        return;
      }
      var list = h("ul", "p4-list");
      value.slice(0, 50).forEach(function (entry) {
        var item = h("li");
        if (entry !== null && typeof entry === "object") {
          renderObject(item, entry, level + 1);
        } else {
          item.textContent = String(entry);
        }
        list.appendChild(item);
      });
      parent.appendChild(list);
      return;
    }
    var table = h("dl", "p4-dl");
    Object.keys(value).sort().forEach(function (key) {
      var dt = h("dt", null, key);
      var dd = h("dd");
      var entry = value[key];
      if (entry !== null && typeof entry === "object") {
        renderObject(dd, entry, level + 1);
      } else {
        dd.textContent = entry === null || entry === undefined ? "-" : String(entry);
      }
      table.appendChild(dt);
      table.appendChild(dd);
    });
    parent.appendChild(table);
  }

  function banner(kind, title, detail) {
    els.banner.className = "p4-banner p4-" + kind;
    els.banner.hidden = false;
    clear(els.banner);
    els.banner.appendChild(h("strong", null, title));
    if (detail) els.banner.appendChild(h("div", "p4-detail", detail));
  }

  function hideBanner() { els.banner.hidden = true; }

  function setBusy(node, busy, label) {
    state.busy = busy;
    node.disabled = busy;
    if (label) node.textContent = busy ? "Working..." : label;
  }

  // ---------------------------------------------------------------- health

  function tigerBlock() {
    var health = state.health || {};
    return health.tiger || null;
  }

  function tigerState() {
    var tiger = tigerBlock();
    if (!tiger) return "disabled";
    return String(tiger.state || "not_started");
  }

  function fusionSources() {
    var health = state.health || {};
    return Array.isArray(health.fusion_sources) ? health.fusion_sources : [];
  }

  function renderHealth() {
    var health = state.health || {};
    var sources = fusionSources();
    var tiger = tigerBlock();
    var state4 = tigerState();

    clear(els.health);
    var chips = [
      ["engine", health.active_engine || "unknown"],
      ["status", health.status || "unknown"],
      ["model_loaded", String(health.model_loaded)],
      ["device", health.device || "-"],
      ["sources", sources.length ? sources.join(" + ") : "-"],
      ["RRF K", health.rrf_k === undefined || health.rrf_k === null ? "-" : String(health.rrf_k)],
      ["source head depth", health.source_head === undefined || health.source_head === null
        ? "-" : String(health.source_head)]
    ];
    if (health.serving_fusion_profile) chips.push(["profile", health.serving_fusion_profile]);
    chips.forEach(function (pair) {
      var chip = h("span", "p4-chip");
      chip.appendChild(h("span", "p4-chip-key", pair[0]));
      chip.appendChild(h("span", "p4-chip-value", pair[1]));
      els.health.appendChild(chip);
    });

    // Four source cards.  A source is "in the running fusion" only when /health lists it, and
    // TIGER is additionally gated on its own worker state - a configured source is not a ready one.
    clear(els.sources);
    SERVING_SOURCES.forEach(function (source) {
      var card = h("div", "p4-source");
      var inFusion = sources.indexOf(source.id) >= 0;
      var status = inFusion ? "in fusion profile" : "not in this profile";
      var cls = inFusion ? "p4-ok" : "p4-muted";
      if (source.id === "tiger") {
        status = state4;
        cls = state4 === "ready" ? "p4-ok" : (state4 === "loading" ? "p4-warn" :
          (state4 === "failed" ? "p4-bad" : "p4-muted"));
        if (state4 !== "ready") inFusion = false;
      }
      card.className = "p4-source " + cls;
      card.appendChild(h("div", "p4-source-name", source.label));
      card.appendChild(h("div", "p4-source-kind", source.kind));
      card.appendChild(h("div", "p4-source-state", status));
      if (source.id === "tiger" && tiger) {
        var bits = [];
        if (tiger.retrieval_mode) bits.push("mode " + tiger.retrieval_mode);
        if (tiger.checkpoint_sha256) bits.push("ckpt " + shortHash(tiger.checkpoint_sha256));
        if (tiger.sid_artifact_sha256) bits.push("sid " + shortHash(tiger.sid_artifact_sha256));
        if (tiger.error) bits.push("error: " + tiger.error);
        if (bits.length) card.appendChild(h("div", "p4-source-detail", bits.join(" | ")));
      }
      els.sources.appendChild(card);
    });

    var active4 = sources.indexOf("tiger") >= 0 && state4 === "ready";
    els.profileNote.textContent = active4
      ? "Serving four sources: " + sources.join(" + ")
      : "Serving " + (sources.length || 0) + " source(s): " + (sources.join(" + ") || "-") +
        ". TIGER is " + state4 + " and is NOT counted as serving.";
    els.profileNote.className = "p4-note " + (active4 ? "p4-ok" : "p4-warn");
  }

  function refreshHealth() {
    return request(ROOT_HEALTH).then(function (payload) {
      state.health = payload;
      renderHealth();
      syncAgentSourceAvailability();
      return payload;
    }).catch(function (error) {
      banner("bad", "Could not read /health", error.message);
    });
  }

  function syncAgentSourceAvailability() {
    var state4 = tigerState();
    if (els.agentSourceInputs && els.agentSourceInputs.tiger) {
      var box = els.agentSourceInputs.tiger;
      var ready = state4 === "ready";
      box.disabled = !ready;
      if (!ready) box.checked = false;
      box.parentNode.title = ready
        ? "The TIGER worker completed its handshake"
        : "TIGER worker state: " + state4 + " - cannot be selected";
    }
  }

  // ---------------------------------------------------------------- fusion view

  function loadTitles() {
    var wanted = els.titleNodes.map(function (entry) { return entry.asin; });
    if (!wanted.length) return Promise.resolve();
    return request(TITLES + "?asins=" + encodeURIComponent(wanted.join(",")))
      .then(function (payload) {
        var titles = (payload && payload.titles) || {};
        els.titleNodes.forEach(function (entry) {
          var value = titles[entry.asin];
          entry.node.textContent = value
            ? value
            : "no title in the catalogue for this identity";
          entry.node.className = value ? "p4-title" : "p4-title p4-muted";
        });
      })
      .catch(function () {
        els.titleNodes.forEach(function (entry) {
          entry.node.textContent = "title lookup unavailable - identity only";
          entry.node.className = "p4-title p4-muted";
        });
      });
  }

  function renderFusionResult(payload) {
    clear(els.fusionResults);
    els.titleNodes = [];
    var recommendations = payload.recommendations || [];
    var provenance = payload.provenance || {};
    var scoreKind = payload.score_kind || "unknown";
    var meta = h("div", "p4-note");
    meta.textContent = "score_kind: " + scoreKind +
      " | requested_k: " + payload.requested_k + " | returned_k: " + payload.returned_k +
      " | eligible_candidates: " + payload.eligible_candidates +
      " | history_length: " + payload.history_length;
    els.fusionResults.appendChild(meta);
    if (scoreKind === "rrf_rank_fusion") {
      els.fusionResults.appendChild(h("div", "p4-note p4-warn",
        "The score is an RRF rank-fusion value (sum of 1/(K+rank)); it is NOT a probability."));
    }
    if (!recommendations.length) {
      els.fusionResults.appendChild(h("div", "p4-empty", "The API returned no candidates."));
      return;
    }

    recommendations.forEach(function (item) {
      var entry = provenance[item.parent_asin] || {};
      var sources = Array.isArray(entry.sources) ? entry.sources : [];
      var card = h("article", "p4-card");
      var head = h("div", "p4-card-head");
      head.appendChild(h("span", "p4-rank", "#" + item.rank));
      head.appendChild(h("span", "p4-asin", item.parent_asin));
      head.appendChild(h("span", "p4-itemid", "item_id " + item.item_id));
      card.appendChild(head);

      // The title is fetched from the read-only catalogue projection for exactly the returned
      // identities.  Until it arrives - or if the catalogue has none - the ASIN stands alone.
      var title = h("div", "p4-title", "loading product title...");
      card.appendChild(title);
      els.titleNodes.push({ asin: item.parent_asin, node: title });

      var facts = h("div", "p4-facts");
      facts.appendChild(h("span", "p4-fact", "score " + Number(item.score).toFixed(6) +
        " (" + scoreKind + ")"));
      facts.appendChild(h("span", "p4-fact", sources.length + " source(s)"));
      if (sources.indexOf("tiger") >= 0) {
        facts.appendChild(h("span", "p4-fact p4-ok", "includes TIGER"));
      } else if (sources.length) {
        facts.appendChild(h("span", "p4-fact p4-muted", "no TIGER contribution"));
      }
      card.appendChild(facts);

      var ranks = entry.source_ranks || {};
      var kinds = entry.source_score_kinds || {};
      if (Object.keys(ranks).length) {
        var table = h("table", "p4-table");
        var thead = h("thead");
        var headRow = h("tr");
        ["source", "source_rank", "source_score_kind"].forEach(function (label) {
          headRow.appendChild(h("th", null, label));
        });
        thead.appendChild(headRow);
        table.appendChild(thead);
        var tbody = h("tbody");
        Object.keys(ranks).sort().forEach(function (source) {
          var row = h("tr");
          row.appendChild(h("td", null, source));
          row.appendChild(h("td", null, String(ranks[source])));
          row.appendChild(h("td", null, kinds[source] || "-"));
          tbody.appendChild(row);
        });
        table.appendChild(tbody);
        card.appendChild(table);
      }

      var toggle = h("button", "p4-toggle", "Show provenance");
      var detail = h("div", "p4-detail-body");
      detail.hidden = true;

      var tiger = entry.tiger;
      if (tiger) {
        var box = h("div", "p4-tiger");
        box.appendChild(h("div", "p4-tiger-head", "TIGER's own view of this candidate"));
        var rows = [
          ["retrieval_mode", tiger.retrieval_mode],
          ["semantic_id", Array.isArray(tiger.semantic_id) ? "[" + tiger.semantic_id.join(", ") + "]" : null],
          ["source rank", tiger["rank"]],
          ["score (sequence log-probability, not a probability)",
            tiger.score === undefined ? null : Number(tiger.score).toFixed(6)],
          ["score_kind", tiger.score_kind],
          ["item_id", tiger.item_id]
        ];
        rows.forEach(function (pair) {
          if (pair[1] === null || pair[1] === undefined) return;
          box.appendChild(h("div", "p4-kv", pair[0] + ": " + pair[1]));
        });
        detail.appendChild(box);
      } else {
        detail.appendChild(h("div", "p4-muted", "This candidate has no TIGER provenance."));
      }
      // The table above already shows ranks and score kinds, and the TIGER box shows the model's
      // own view; the untouched API object goes behind a second fold so the default view stays
      // readable without discarding any evidence.
      var rawToggle = h("button", "p4-toggle", "Show raw technical evidence");
      var raw = h("div", "p4-raw");
      raw.hidden = true;
      renderObject(raw, entry, 0);
      rawToggle.addEventListener("click", function () {
        raw.hidden = !raw.hidden;
        rawToggle.textContent = raw.hidden ? "Show raw technical evidence"
                                           : "Hide raw technical evidence";
      });
      detail.appendChild(rawToggle);
      detail.appendChild(raw);

      toggle.addEventListener("click", function () {
        detail.hidden = !detail.hidden;
        toggle.textContent = detail.hidden ? "Show provenance" : "Hide provenance";
      });
      card.appendChild(toggle);
      card.appendChild(detail);
      els.fusionResults.appendChild(card);
    });
  }

  function submitFusion(event) {
    if (event) event.preventDefault();
    hideBanner();
    var history = parseHistory(els.fusionHistory.value);
    if (!history.length) {
      banner("bad", "A history is required", "Enter at least one parent_asin.");
      return;
    }
    var k = parseInt(els.fusionK.value, 10);
    clear(els.fusionResults);
    els.fusionResults.appendChild(h("div", "p4-loading", "Requesting /v1/recommend ..."));
    setBusy(els.fusionSubmit, true, "Recommend");
    request(FUSION, { method: "POST", body: { history: history, k: k } })
      .then(function (payload) {
        renderFusionResult(payload);
        return loadTitles().then(function () { return refreshHealth(); });
      })
      .catch(function (error) {
        clear(els.fusionResults);
        banner("bad", "The recommendation request failed", error.message);
      })
      .then(function () { setBusy(els.fusionSubmit, false, "Recommend"); });
  }

  // ---------------------------------------------------------------- agent view

  function agentSteps(payload) {
    var trajectory = payload.trajectory || {};
    var steps = trajectory.tool_calls || trajectory.steps || [];
    return Array.isArray(steps) ? steps : [];
  }

  function renderAgentResult(payload) {
    clear(els.agentResults);
    els.titleNodes = [];
    if (payload.error) {
      banner("bad", "The Agent reported an error", String(payload.error.detail || payload.error));
      return;
    }
    var terminal = payload.terminal || {};
    var grounded = payload.grounded || {};
    var steps = agentSteps(payload);
    var consultations = steps.filter(function (step) { return step && step.action !== "finish"; });

    // ---- layer 1: execution summary -------------------------------------
    var summary = h("section", "p4-panel");
    summary.appendChild(h("h3", null, "1. Execution summary"));
    var chips = h("div", "p4-chips");
    function chip(key, value, cls) {
      var node = h("span", "p4-chip " + (cls || ""));
      node.appendChild(h("span", "p4-chip-key", key));
      node.appendChild(h("span", "p4-chip-value", value === undefined || value === null ? "-" : String(value)));
      chips.appendChild(node);
    }
    chip("succeeded", terminal.succeeded, terminal.succeeded ? "p4-ok" : "p4-warn");
    chip("source plan", (payload.requested_sources || payload.sources ||
      (consultations.length ? consultations.map(function (s) { return s.action; }).join(" -> ") : "-")));
    chip("completion", terminal.termination_reason || terminal.status || "finished");
    chip("elapsed", (payload.timing && (payload.timing.total_ms || payload.timing.total_seconds))
      ? (payload.timing.total_ms ? payload.timing.total_ms + " ms" : payload.timing.total_seconds + " s") : "-");
    chip("tool calls", consultations.length + (steps.length ? " of " + steps.length + " steps" : ""));
    chip("grounded", grounded.grounded_count);
    chip("ungrounded", grounded.ungrounded_count);
    summary.appendChild(chips);
    summary.appendChild(h("div", "p4-note p4-warn",
      "The source plan is the one the request declared; it is executed by the controller's " +
      "deterministic policy, not chosen autonomously by an LLM."));
    els.agentResults.appendChild(summary);

    // ---- layer 2: action timeline ---------------------------------------
    var timeline = h("section", "p4-panel");
    timeline.appendChild(h("h3", null, "2. Action timeline"));
    if (!steps.length) {
      timeline.appendChild(h("div", "p4-empty", "The response carried no execution steps."));
    }
    steps.forEach(function (step, index) {
      var row = h("div", "p4-step");
      row.appendChild(h("span", "p4-step-index", String(index + 1)));
      var body = h("span", "p4-step-body");
      var label = String(step.action || "action").toUpperCase();
      if (step.action === "select_source" && step.source) label += "(" + step.source + ")";
      body.appendChild(h("strong", null, label));
      var bits = [];
      if (step.validated !== undefined) bits.push("validated=" + step.validated);
      if (step.candidate_count !== undefined) bits.push("candidate_count=" + step.candidate_count);
      if (step.observation_type) bits.push("observation_type=" + step.observation_type);
      if (step.note) bits.push(String(step.note));
      if (step.refusal_codes && step.refusal_codes.length) bits.push("refusals=" + step.refusal_codes.join(","));
      if (bits.length) body.appendChild(h("div", "p4-small", bits.join(" · ")));
      row.appendChild(body);
      timeline.appendChild(row);
    });
    els.agentResults.appendChild(timeline);

    // ---- layer 3: product results ---------------------------------------
    var products = h("section", "p4-panel");
    var recommendations = payload.recommendations || [];
    products.appendChild(h("h3", null, "3. Recommendations (" + recommendations.length + ")"));
    var groundingLine = h("div", "p4-note");
    groundingLine.textContent = "Grounding: " + (grounded.grounded_count === undefined
      ? "not reported" : grounded.grounded_count + " grounded / " +
        (grounded.ungrounded_count === undefined ? "?" : grounded.ungrounded_count) + " ungrounded");
    products.appendChild(groundingLine);
    // Eligibility is reported only when it actually ran.
    var eligibility = payload.eligibility;
    var evaluated = eligibility && (eligibility.evaluated === true ||
      (eligibility.checked_count !== undefined && eligibility.checked_count !== null) ||
      (eligibility.feasible_candidate_count !== undefined));
    products.appendChild(h("div", "p4-note " + (evaluated ? "" : "p4-muted"),
      evaluated ? "Eligibility: evaluated" : "Eligibility: not evaluated"));
    if (!recommendations.length) {
      products.appendChild(h("div", "p4-empty", "The Agent returned no candidates."));
    }
    recommendations.forEach(function (item) {
      var card = h("article", "p4-card");
      var head = h("div", "p4-card-head");
      head.appendChild(h("span", "p4-rank", "#" + (item.rank === undefined ? "-" : item.rank)));
      head.appendChild(h("span", "p4-asin", item.parent_asin || item.identity || "(no identity)"));
      if (item.item_id !== undefined) head.appendChild(h("span", "p4-itemid", "item_id " + item.item_id));
      card.appendChild(head);
      var title = h("div", "p4-title", item.title || "loading product title...");
      card.appendChild(title);
      if (!item.title) els.titleNodes.push({ asin: item.parent_asin || item.identity, node: title });
      var facts = h("div", "p4-facts");
      var itemSources = item.sources || (item.provenance && item.provenance.sources) || [];
      facts.appendChild(h("span", "p4-fact", Array.isArray(itemSources) && itemSources.length
        ? itemSources.join(" + ") : "source not reported"));
      if (item.grounded !== undefined) {
        facts.appendChild(h("span", "p4-fact " + (item.grounded ? "p4-ok" : "p4-warn"),
          "grounded " + item.grounded));
      }
      if (item.score !== undefined) facts.appendChild(h("span", "p4-fact", "score " + item.score));
      card.appendChild(facts);
      var toggle = h("button", "p4-toggle", "Show technical details");
      var detail = h("div", "p4-detail-body");
      detail.hidden = true;
      renderObject(detail, item, 0);
      toggle.addEventListener("click", function () {
        detail.hidden = !detail.hidden;
        toggle.textContent = detail.hidden ? "Show technical details" : "Hide technical details";
      });
      card.appendChild(toggle);
      card.appendChild(detail);
      products.appendChild(card);
    });
    els.agentResults.appendChild(products);

    // ---- raw response, collapsed ----------------------------------------
    var rawPanel = h("section", "p4-panel");
    var rawToggle = h("button", "p4-toggle", "Show full response (raw evidence)");
    var rawBody = h("div", "p4-detail-body");
    rawBody.hidden = true;
    renderObject(rawBody, {
      terminal: payload.terminal, trajectory: payload.trajectory, grounded: payload.grounded,
      eligibility: payload.eligibility, timing: payload.timing, control_plane: payload.control_plane,
      route: payload.route, message: payload.message, api_version: payload.api_version
    }, 0);
    rawToggle.addEventListener("click", function () {
      rawBody.hidden = !rawBody.hidden;
      rawToggle.textContent = rawBody.hidden ? "Show full response (raw evidence)"
                                             : "Hide full response (raw evidence)";
    });
    rawPanel.appendChild(rawToggle);
    rawPanel.appendChild(rawBody);
    els.agentResults.appendChild(rawPanel);

    var wanted = els.titleNodes.map(function (entry) { return entry.asin; })
      .filter(function (asin) { return typeof asin === "string" && asin; });
    if (wanted.length) {
      var nodes = els.titleNodes;
      request(TITLES + "?asins=" + encodeURIComponent(wanted.join(",")))
        .then(function (titlesPayload) {
          var titles = (titlesPayload && titlesPayload.titles) || {};
          nodes.forEach(function (entry) {
            var value = titles[entry.asin];
            entry.node.textContent = value || "no title in the catalogue for this identity";
            entry.node.className = value ? "p4-title" : "p4-title p4-muted";
          });
        })
        .catch(function () {
          nodes.forEach(function (entry) {
            entry.node.textContent = "title lookup unavailable - identity only";
            entry.node.className = "p4-title p4-muted";
          });
        });
    }
  }

  function submitAgent(event) {
    if (event) event.preventDefault();
    hideBanner();
    var message = String(els.agentMessage.value || "").trim();
    var history = parseHistory(els.agentHistory.value);
    var sources = AGENT_SOURCES
      .filter(function (source) { return els.agentSourceInputs[source.id].checked; })
      .map(function (source) { return source.id; });
    if (!message) { banner("bad", "message is required", "The Agent endpoint needs a message."); return; }
    if (!history.length) { banner("bad", "history is required", "Enter at least one parent_asin."); return; }
    if (!sources.length) { banner("bad", "Select at least one source", "sources must be non-empty."); return; }

    clear(els.agentResults);
    els.agentResults.appendChild(h("div", "p4-loading", "Requesting /v1/demo/agent/recommend ..."));
    setBusy(els.agentSubmit, true, "Run agent");
    var body = { message: message, history: history, k: parseInt(els.agentK.value, 10), sources: sources };
    request(AGENT, { method: "POST", body: body })
      .then(function (payload) { renderAgentResult(payload); return refreshHealth(); })
      .catch(function (error) {
        clear(els.agentResults);
        banner("bad", "The Agent request failed", error.message);
      })
      .then(function () { setBusy(els.agentSubmit, false, "Run agent"); });
  }

  // ---------------------------------------------------------------- shell

  function buildShell() {
    originalNodes = Array.prototype.slice.call(document.body.children);

    var shell = h("div", "p4-shell");
    shell.appendChild(h("header", "p4-header")).appendChild(h("h1", null, "AgentRec-X"));
    shell.querySelector(".p4-header").appendChild(h("div", "p4-sub",
      "Multi-paradigm recommender: fused retrieval sources and an optional Agent source dispatcher"));

    var tabs = h("nav", "p4-tabs");
    [["chat", "Chat (existing demo)"], ["fusion", "Four-source fusion"],
     ["agent", "Agent dispatch"]].forEach(function (pair) {
      var button = h("button", "p4-tab", pair[1]);
      button.type = "button";
      button.addEventListener("click", function () { showView(pair[0]); });
      button.dataset.view = pair[0];
      tabs.appendChild(button);
    });
    shell.appendChild(tabs);

    els.banner = h("div", "p4-banner");
    els.banner.hidden = true;
    els.banner.setAttribute("role", "alert");
    shell.appendChild(els.banner);

    // view: fusion
    var fusion = h("section", "p4-view");
    fusion.dataset.view = "fusion";
    var form = h("form", "p4-form");
    form.appendChild(h("label", "p4-label", "User history (chronological parent_asin values)"));
    els.fusionHistory = h("textarea", "p4-input");
    els.fusionHistory.rows = 3;
    els.fusionHistory.value = SAMPLE_HISTORY.join(", ");
    form.appendChild(els.fusionHistory);
    var controls = h("div", "p4-controls");
    els.fusionK = h("select", "p4-select");
    [5, 10].forEach(function (value) {
      var option = h("option", null, "k=" + value);
      option.value = String(value);
      els.fusionK.appendChild(option);
    });
    controls.appendChild(els.fusionK);
    var sample = h("button", "p4-button", "Load sample history");
    sample.type = "button";
    sample.addEventListener("click", function () {
      els.fusionHistory.value = SAMPLE_HISTORY.join(", ");
      banner("info", "Sample history loaded",
        "These identities were verified over real HTTP in the T-02 acceptance rounds.");
    });
    controls.appendChild(sample);
    els.fusionSubmit = h("button", "p4-button p4-primary", "Recommend");
    els.fusionSubmit.type = "submit";
    controls.appendChild(els.fusionSubmit);
    form.appendChild(controls);
    form.addEventListener("submit", submitFusion);
    fusion.appendChild(form);

    var statusPanel = h("section", "p4-panel");
    statusPanel.appendChild(h("h3", null, "System and fusion profile (GET /health)"));
    els.health = h("div", "p4-chips");
    statusPanel.appendChild(els.health);
    els.profileNote = h("div", "p4-note");
    statusPanel.appendChild(els.profileNote);
    els.sources = h("div", "p4-sources");
    statusPanel.appendChild(els.sources);
    fusion.appendChild(statusPanel);

    els.fusionResults = h("div", "p4-results");
    fusion.appendChild(els.fusionResults);
    shell.appendChild(fusion);

    // view: agent
    var agent = h("section", "p4-view");
    agent.dataset.view = "agent";
    var agentForm = h("form", "p4-form");
    agentForm.appendChild(h("label", "p4-label", "message (required)"));
    els.agentMessage = h("input", "p4-input");
    els.agentMessage.value = "Recommend some cycling accessories based on my purchase history.";
    agentForm.appendChild(els.agentMessage);
    agentForm.appendChild(h("label", "p4-label", "history (required)"));
    els.agentHistory = h("textarea", "p4-input");
    els.agentHistory.rows = 2;
    els.agentHistory.value = SAMPLE_HISTORY.join(", ");
    agentForm.appendChild(els.agentHistory);
    var agentControls = h("div", "p4-controls");
    els.agentK = h("select", "p4-select");
    [5, 10].forEach(function (value) {
      var option = h("option", null, "k=" + value);
      option.value = String(value);
      els.agentK.appendChild(option);
    });
    agentControls.appendChild(els.agentK);
    els.agentSubmit = h("button", "p4-button p4-primary", "Run agent");
    els.agentSubmit.type = "submit";
    agentControls.appendChild(els.agentSubmit);
    agentForm.appendChild(agentControls);

    var plans = h("fieldset", "p4-fieldset");
    plans.appendChild(h("legend", null,
      "sources - Agent plan names (history / catalog_search / two_tower / tiger). " +
      "These are NOT the serving names (sequential / catalog_lexical)."));
    els.agentSourceInputs = {};
    AGENT_SOURCES.forEach(function (source) {
      var label = h("label", "p4-check");
      var box = h("input");
      box.type = "checkbox";
      box.value = source.id;
      box.checked = source.id === "history";
      label.appendChild(box);
      label.appendChild(h("span", null, " " + source.label));
      els.agentSourceInputs[source.id] = box;
      plans.appendChild(label);
    });
    agentForm.appendChild(plans);
    agentForm.addEventListener("submit", submitAgent);
    agent.appendChild(agentForm);
    agent.appendChild(h("div", "p4-note p4-warn",
      "The plan you tick is executed by the controller's deterministic policy. It is not an " +
      "LLM's autonomous choice, and no paid provider is called by this page."));
    els.agentResults = h("div", "p4-results");
    agent.appendChild(els.agentResults);
    shell.appendChild(agent);

    document.body.insertBefore(shell, document.body.firstChild);
    injectStyles();
    showView("chat");
  }

  function showView(name) {
    state.view = name;
    Array.prototype.forEach.call(document.querySelectorAll(".p4-tab"), function (tab) {
      tab.classList.toggle("p4-active", tab.dataset.view === name);
    });
    Array.prototype.forEach.call(document.querySelectorAll(".p4-view"), function (view) {
      view.hidden = view.dataset.view !== name;
    });
    // The original page is shown only on the chat tab; its DOM and listeners are untouched.
    originalNodes.forEach(function (node) {
      if (node.classList && node.classList.contains("p4-shell")) return;
      node.style.display = name === "chat" ? "" : "none";
    });
    if (name !== "chat") refreshHealth();
  }

  function injectStyles() {
    var css = document.createElement("style");
    css.textContent = [
      ".p4-shell{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1180px;margin:0 auto;padding:16px;color:#1b1f24}",
      ".p4-header h1{margin:0;font-size:22px}",
      ".p4-sub{color:#5b6470;font-size:13px;margin:4px 0 12px}",
      ".p4-tabs{display:flex;gap:8px;flex-wrap:wrap;border-bottom:1px solid #d8dee6;margin-bottom:12px}",
      ".p4-tab{background:none;border:0;border-bottom:2px solid transparent;padding:8px 12px;cursor:pointer;font-size:14px;color:#48505a}",
      ".p4-tab.p4-active{border-bottom-color:#2f6feb;color:#1b3a7a;font-weight:600}",
      ".p4-banner{border-radius:6px;padding:10px 12px;margin:10px 0;font-size:13px}",
      ".p4-bad{background:#fdecec;border:1px solid #f0b4b4}",
      ".p4-info{background:#eef4ff;border:1px solid #b9cdf7}",
      ".p4-form{background:#fff;border:1px solid #e3e8ee;border-radius:8px;padding:14px;margin-bottom:14px}",
      ".p4-label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#5b6470;margin:8px 0 4px}",
      ".p4-input{width:100%;box-sizing:border-box;padding:8px 10px;border:1px solid #cfd6de;border-radius:6px;font-size:14px;font-family:inherit}",
      ".p4-controls{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}",
      ".p4-select,.p4-button{padding:8px 12px;border:1px solid #cfd6de;border-radius:6px;background:#fff;font-size:14px;cursor:pointer}",
      ".p4-primary{background:#2f6feb;border-color:#2f6feb;color:#fff}",
      ".p4-button[disabled]{opacity:.6;cursor:progress}",
      ".p4-panel{background:#fff;border:1px solid #e3e8ee;border-radius:8px;padding:14px;margin-bottom:14px}",
      ".p4-panel h3,.p4-panel h4{margin:0 0 8px;font-size:14px}",
      ".p4-chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}",
      ".p4-chip{border:1px solid #e3e8ee;border-radius:999px;padding:3px 10px;font-size:12px;background:#f7f9fc}",
      ".p4-chip-key{color:#6b7480;margin-right:6px}",
      ".p4-sources{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}",
      ".p4-source{border:1px solid #e3e8ee;border-radius:8px;padding:10px;background:#fbfcfe}",
      ".p4-source-name{font-weight:600;font-size:14px}",
      ".p4-source-kind,.p4-small{font-size:12px;color:#6b7480}",
      ".p4-source-state{font-size:12px;margin-top:6px}",
      ".p4-source-detail{font-size:11px;color:#6b7480;margin-top:4px;word-break:break-all}",
      ".p4-ok{color:#1a7f45}.p4-warn{color:#9a6700}.p4-bad{color:#b42318}.p4-muted{color:#6b7480}",
      ".p4-note{font-size:13px;color:#48505a;margin:8px 0}",
      ".p4-results{display:flex;flex-direction:column;gap:10px}",
      ".p4-card{border:1px solid #e3e8ee;border-radius:8px;padding:12px;background:#fff}",
      ".p4-card-head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}",
      ".p4-rank{font-weight:700;color:#2f6feb}",
      ".p4-title{font-size:15px;font-weight:600;margin:6px 0 2px}",
      ".p4-asin{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px}",
      ".p4-itemid{font-size:12px;color:#6b7480}",
      ".p4-facts{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}",
      ".p4-fact{font-size:12px;background:#f4f6fa;border-radius:4px;padding:3px 8px}",
      ".p4-table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}",
      ".p4-table th,.p4-table td{text-align:left;border-bottom:1px solid #eef1f5;padding:4px 6px}",
      ".p4-toggle{margin-top:6px;background:none;border:0;color:#2f6feb;cursor:pointer;font-size:13px;padding:0}",
      ".p4-detail-body{margin-top:8px;border-top:1px dashed #e3e8ee;padding-top:8px}",
      ".p4-tiger{background:#f2f8ff;border:1px solid #cfe2ff;border-radius:6px;padding:8px;margin-bottom:8px}",
      ".p4-tiger-head{font-weight:600;font-size:13px;margin-bottom:4px}",
      ".p4-kv{font-size:12px;color:#39414c;margin:2px 0;word-break:break-word}",
      ".p4-dl{display:grid;grid-template-columns:minmax(120px,auto) 1fr;gap:2px 10px;font-size:12px;margin:0}",
      ".p4-dl dt{color:#6b7480}.p4-dl dd{margin:0}",
      ".p4-list{margin:4px 0 4px 16px;padding:0;font-size:12px}",
      ".p4-trace{border-left:3px solid #cfd6de;padding-left:10px;margin:10px 0}",
      ".p4-trace-head{font-size:12px;color:#6b7480;margin-bottom:6px}",
      ".p4-step{display:flex;gap:8px;font-size:12px;margin:3px 0}",
      ".p4-step-index{color:#2f6feb;font-weight:600}",
      ".p4-loading{font-size:13px;color:#48505a}",
      ".p4-empty{font-size:13px;color:#6b7480}",
      ".p4-fieldset{border:1px solid #e3e8ee;border-radius:6px;padding:8px 10px;margin-top:10px}",
      ".p4-fieldset legend{font-size:12px;color:#48505a;padding:0 4px}",
      ".p4-check{display:inline-flex;align-items:center;gap:4px;font-size:13px;margin-right:14px}",
      "@media (max-width:720px){.p4-shell{padding:10px}.p4-sources{grid-template-columns:1fr 1fr}",
      ".p4-dl{grid-template-columns:1fr}.p4-table{font-size:12px}}"
    ].join("");
    document.head.appendChild(css);
  }

  function start() {
    buildShell();
    refreshHealth();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
