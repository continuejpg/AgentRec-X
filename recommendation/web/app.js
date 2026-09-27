/* AgentRec-X Milestone 11 demo client.
 *
 * Thin renderer over the demo HTTP API.  It performs NO recommendation work:
 * it never scores, filters, ranks or re-orders anything, and it renders the
 * `recommendations` array in exactly the sequence the backend produced.
 *
 * Safety: every piece of dynamic text (user messages, agent prose, catalogue metadata)
 * is written with textContent / createElement.  `innerHTML` is not used anywhere, so a
 * product title or a chat message containing markup is displayed literally.
 */
(function () {
  "use strict";

  var API = "/v1/demo";

  var els = {
    banner: document.getElementById("banner"),
    transcript: document.getElementById("transcript"),
    form: document.getElementById("chat-form"),
    input: document.getElementById("message-input"),
    send: document.getElementById("send-button"),
    profileSelect: document.getElementById("profile-select"),
    newSession: document.getElementById("new-session"),
    resetSession: document.getElementById("reset-session"),
    preferences: document.getElementById("preferences"),
    cards: document.getElementById("cards"),
    state: document.getElementById("state-value"),
    session: document.getElementById("session-value"),
    turn: document.getElementById("turn-value"),
    route: document.getElementById("route-value"),
    candidates: document.getElementById("candidate-value"),
    moved: document.getElementById("moved-value"),
    rankedWith: document.getElementById("ranked-with-value"),
    originalOrder: document.getElementById("original-order"),
    rerankedOrder: document.getElementById("reranked-order"),
    apiVersion: document.getElementById("api-version"),
    tracePanel: document.getElementById("trace-panel"),
    statusChips: document.getElementById("status-chips"),
    memoryNote: document.getElementById("memory-note")
  };

  var state = {
    sessionId: null,
    profileId: null,
    sending: false,
    expired: false,
    turns: 0,
    ready: false,
    lastRoute: null
  };

  /* ------------------------------------------------------------------ */
  /* small DOM helpers (no innerHTML)                                    */
  /* ------------------------------------------------------------------ */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function clear(node) {
    while (node.firstChild) { node.removeChild(node.firstChild); }
  }

  function setState(label) {
    els.state.textContent = label;
  }

  function showBanner(kind, title, detail) {
    clear(els.banner);
    els.banner.hidden = false;
    els.banner.setAttribute("data-kind", kind);
    els.banner.appendChild(el("strong", null, title));
    if (detail) {
      els.banner.appendChild(document.createTextNode(" — " + detail));
    }
  }

  function hideBanner() {
    els.banner.hidden = true;
    clear(els.banner);
  }

  function setBusy(busy) {
    state.sending = busy;
    els.send.disabled = busy || state.expired;
    els.input.disabled = busy || state.expired;
    els.newSession.disabled = busy;
    els.resetSession.disabled = busy || !state.sessionId;
    els.profileSelect.disabled = busy;
    els.send.textContent = busy ? "Sending…" : "Send";
  }

  /* ------------------------------------------------------------------ */
  /* HTTP                                                                */
  /* ------------------------------------------------------------------ */

  function request(path, options) {
    var opts = options || {};
    return fetch(API + path, {
      method: opts.method || "GET",
      headers: opts.body ? { "Content-Type": "application/json" } : undefined,
      body: opts.body ? JSON.stringify(opts.body) : undefined
    }).then(function (response) {
      var contentType = response.headers.get("content-type") || "";
      var parse = contentType.indexOf("application/json") >= 0
        ? response.json()
        : response.text().then(function (text) { return { detail: text }; });
      return parse.then(function (payload) {
        if (!response.ok) {
          var error = new Error((payload && payload.detail) || response.statusText);
          error.status = response.status;
          error.code = (payload && payload.error) || "http_" + response.status;
          throw error;
        }
        return payload;
      });
    });
  }

  function describeError(error) {
    if (error && error.code === "session_not_found") {
      return "This session no longer exists on the server. Start a new session to continue.";
    }
    if (error && error.code === "invalid_request") {
      return "The request was rejected: " + (error.message || "check the message and k value.");
    }
    if (error && error.code === "session_capacity_exceeded") {
      return "The demo has reached its live-session limit. Reset an existing session first.";
    }
    if (error && error.status >= 500) {
      return "The demo backend could not complete this turn. No partial result was fabricated.";
    }
    return (error && error.message) || "Unexpected error.";
  }

  /* ------------------------------------------------------------------ */
  /* startup                                                             */
  /* ------------------------------------------------------------------ */

  function loadProfiles() {
    return request("/profiles").then(function (payload) {
      els.apiVersion.textContent = "api " + payload.api_version;
      clear(els.profileSelect);
      (payload.profiles || []).forEach(function (profile) {
        var option = el("option", null, profile.display_name + " (" + profile.profile_id + ")");
        option.value = profile.profile_id;
        els.profileSelect.appendChild(option);
      });
      if (!payload.profiles || payload.profiles.length === 0) {
        throw new Error("the server advertises no demo profiles");
      }
      state.profileId = payload.profiles[0].profile_id;
      els.profileSelect.value = state.profileId;
      els.profileSelect.disabled = false;
    });
  }

  function checkHealth() {
    /* Readiness probe: keeps the "initializing" state honest and surfaces an
       unready backend before the user types anything. */
    return request("/health").then(function (payload) {
      if (!payload.demo_ready || !payload.model_loaded) {
        throw new Error(payload.detail || "the demo backend is not ready");
      }
      state.ready = true;
      renderStatusChips();
      return payload;
    });
  }

  function createSession() {
    hideBanner();
    setState("initializing");
    setBusy(true);
    var chosen = els.profileSelect.value || state.profileId;
    return request("/sessions", { method: "POST", body: { profile_id: chosen } })
      .then(function (payload) {
        state.sessionId = payload.session_id;
        state.profileId = payload.profile.profile_id;
        state.turns = payload.turn || 0;
        state.expired = false;
        els.session.textContent = payload.session_id;
        els.turn.textContent = String(state.turns);
        els.route.textContent = "—";
        els.candidates.textContent = "—";
        els.moved.textContent = "—";
        els.rankedWith.textContent = "—";
        clear(els.originalOrder);
        clear(els.rerankedOrder);
        clear(els.transcript);
        clear(els.cards);
        els.cards.appendChild(el("p", "muted", "Send a message that asks for recommendations to see cards here."));
        renderPreferences([]);
        clear(els.memoryNote);
        els.memoryNote.hidden = true;
        renderStatusChips();
        setBusy(false);
        setState("ready");
        els.input.focus();
      })
      .catch(function (error) {
        setBusy(false);
        setState("error");
        state.sessionId = null;
        showBanner("error", "Could not start a session", describeError(error));
      });
  }

  function resetSession() {
    if (!state.sessionId) { return; }
    hideBanner();
    setBusy(true);
    var target = state.sessionId;
    request("/sessions/" + encodeURIComponent(target), { method: "DELETE" })
      .then(function () {
        showBanner("info", "Session reset", "Only that session was removed. Starting a new one.");
        state.sessionId = null;
        return createSession();
      })
      .catch(function (error) {
        setBusy(false);
        setState("error");
        showBanner("error", "Reset failed", describeError(error));
      });
  }

  /* ------------------------------------------------------------------ */
  /* chat                                                                */
  /* ------------------------------------------------------------------ */

  function appendTurn(kind, meta, body, extra) {
    var turn = el("div", "turn " + kind);
    turn.appendChild(el("div", "turn-meta", meta));
    turn.appendChild(el("div", "turn-body", body));
    if (extra) { turn.appendChild(extra); }
    els.transcript.appendChild(turn);
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  function send(event) {
    if (event) { event.preventDefault(); }
    if (state.sending || state.expired || !state.sessionId) { return; }
    var message = els.input.value.trim();
    if (!message) {
      showBanner("error", "Message required", "Type a shopping message before sending.");
      return;
    }

    hideBanner();
    setBusy(true);
    setState("sending");
    appendTurn("user", "You", message);
    els.input.value = "";

    request("/sessions/" + encodeURIComponent(state.sessionId) + "/chat", {
      method: "POST",
      body: { message: message, k: 5 }
    })
      .then(function (payload) {
        state.turns = payload.turn;
        els.turn.textContent = String(payload.turn);
        els.route.textContent = payload.route;
        state.lastRoute = payload.route;
        appendTurn("agent", "Agent · " + payload.turn_id, payload.message);
        renderTrace(payload.trace);
        renderStatusChips();
        renderMemoryUpdate(payload.memory_update);
        renderCards(payload.recommendations);
        renderAudit(payload.audit);
        renderPreferences(payload.active_preferences);
        setState(payload.audit && payload.audit.reranking_applied ? "success" : "success");
        setBusy(false);
        els.input.focus();
      })
      .catch(function (error) {
        setBusy(false);
        if (error && error.code === "session_not_found") {
          state.expired = true;
          setState("session expired");
          setBusy(false);
          showBanner("error", "Session expired", describeError(error));
          return;
        }
        setState("error");
        showBanner("error", "Turn failed", describeError(error));
      });
  }

  /* ------------------------------------------------------------------ */
  /* rendering                                                           */
  /* ------------------------------------------------------------------ */

  /* Recommendation Trace: a read-only view of what this turn already decided.  Every value
   * comes from the response the backend produced; nothing is recomputed here.  Sections with
   * no authoritative value for the turn are omitted rather than shown as placeholders. */
  function traceSection(title, lines) {
    if (!lines || lines.length === 0) { return null; }
    var wrap = el("div", "trace-section");
    wrap.appendChild(el("h4", null, title));
    var ul = el("ul");
    lines.forEach(function (line) { ul.appendChild(el("li", null, line)); });
    wrap.appendChild(ul);
    return wrap;
  }

  function traceRows(title, rows) {
    if (!rows || rows.length === 0) { return null; }
    var wrap = el("div", "trace-section");
    wrap.appendChild(el("h4", null, title));
    var ul = el("ul", "trace-list");
    rows.forEach(function (row) {
      var li = el("li", row.className || null);
      li.appendChild(el("span", "k", row.key));
      li.appendChild(el("span", "v", row.value));
      ul.appendChild(li);
    });
    wrap.appendChild(ul);
    return wrap;
  }

  function renderStatusChips() {
    clear(els.statusChips);
    if (state.ready) {
      els.statusChips.appendChild(el("span", "chip ok", "backend ready"));
    }
    if (state.profileId) {
      els.statusChips.appendChild(el("span", "chip", "profile " + state.profileId));
    }
    if (state.sessionId) {
      els.statusChips.appendChild(el("span", "chip", "session " + state.sessionId.slice(0, 8)));
    }
    if (state.turns) {
      els.statusChips.appendChild(el("span", "chip", "turn " + state.turns));
    }
    if (state.lastRoute) {
      els.statusChips.appendChild(el("span", "chip",
        state.lastRoute === "recommend" ? "recommendation" : "direct"));
    }
  }

  function renderTrace(trace) {
    /* Renders into the permanent panel, from the response's own trace object only. */
    clear(els.tracePanel);
    if (!trace) {
      els.tracePanel.appendChild(el("p", "muted", "No trace for this turn."));
      return;
    }
    var sections = [];

    var route = el("div", "trace-section");
    route.appendChild(el("h4", null, "Route"));
    route.appendChild(el("div", "trace-value",
      trace.route === "recommend" ? "RECOMMENDATION" : "DIRECT"));
    sections.push(route);

    if (typeof trace.candidate_count === "number") {
      sections.push(traceRows("Candidates", [
        { key: "returned", value: String(trace.candidate_count) }
      ]));
    }

    var active = trace.active_preferences || [];
    var prefs = el("div", "trace-section");
    prefs.appendChild(el("h4", null, "Active preferences"));
    var chips = el("div", "trace-chips");
    if (active.length === 0) {
      chips.appendChild(el("span", "chip empty", "none"));
    }
    active.forEach(function (item) {
      chips.appendChild(el("span", "chip pref", item.value));
    });
    prefs.appendChild(chips);
    sections.push(prefs);

    var changes = trace.memory_changes || {};
    var memRows = [];
    if ((changes.added || []).length) {
      memRows.push({ key: "+ added", value: changes.added.join(", ") });
    }
    if ((changes.removed || []).length) {
      memRows.push({ key: "\u2212 removed", value: changes.removed.join(", ") });
    }
    if ((changes.superseded || []).length) {
      memRows.push({ key: "\u21bb replaced", value: changes.superseded.join(", ") });
    }
    if (memRows.length) { sections.push(traceRows("Memory changes", memRows)); }

    if (trace.evidence) {
      var ev = trace.evidence;
      sections.push(traceRows("Evidence", [
        { key: "\u2713 MATCH", value: String(ev.match_candidates) },
        { key: "! VIOLATION", value: String(ev.violation_candidates) },
        { key: "? UNKNOWN", value: String(ev.unknown_candidates) }
      ]));
    }

    var moves = trace.ranking_changes || [];
    if (moves.length) {
      var wrap = el("div", "trace-section");
      wrap.appendChild(el("h4", null, "Rank movement"));
      var ul = el("ul", "trace-list");
      moves.forEach(function (row) {
        var li = el("li", "trace-move");
        var up = row.moved && row.final_rank < row.original_rank;
        var arrow = row.moved ? (up ? "\u2191" : "\u2193") : "\u2014";
        li.appendChild(el("span", "k", "#" + row.original_rank + " \u2192 #" + row.final_rank));
        li.appendChild(el("span", "v " + (row.moved ? (up ? "up" : "down") : "same"), arrow));
        ul.appendChild(li);
      });
      wrap.appendChild(ul);
      sections.push(wrap);
    }

    /* Deliberately absent on the browser path: source and grounding.  Rendered ONLY if a
     * future path supplies an authoritative value, so no placeholder can look factual. */
    if (trace.source) {
      sections.push(traceRows("Source", [{ key: "source", value: trace.source }]));
    }
    if (trace.grounding) {
      sections.push(traceRows("Grounding", [
        { key: "verified", value: trace.grounding.verified + " / " + trace.grounding.total }
      ]));
    }

    sections.forEach(function (section) {
      if (section) { els.tracePanel.appendChild(section); }
    });
  }

  function renderPreferences(preferences) {
    clear(els.preferences);
    var list = preferences || [];
    if (list.length === 0) {
      els.preferences.appendChild(el("span", "chip empty", "none yet"));
      return;
    }
    list.forEach(function (item) {
      var label;
      if (item.kind === "price_max") { label = "\u2264 " + item.value; }
      else if (item.kind === "price_min") { label = "\u2265 " + item.value; }
      else if (item.polarity === "avoid") { label = "avoid " + item.value; }
      else { label = item.value; }
      var chip = el("span", "chip pref", label);
      chip.setAttribute("title", item.kind);
      els.preferences.appendChild(chip);
    });
  }

  function renderMemoryUpdate(update) {
    if (!update || !update.changed) { return; }
    var parts = [];
    (update.added || []).forEach(function (item) {
      parts.push("Preference saved for future turns: " + item.polarity + " " + item.value);
    });
    (update.superseded || []).forEach(function (item) {
      parts.push("Replaced earlier preference: " + item.value);
    });
    (update.removed || []).forEach(function (item) {
      parts.push("Removed preference: " + item.value);
    });
    if (parts.length === 0) { return; }
    showBanner("info", "Memory updated", parts.join(" \u00b7 "));

    clear(els.memoryNote);
    var onlyRemoved = (update.removed || []).length > 0
      && (update.added || []).length === 0
      && (update.superseded || []).length === 0;
    els.memoryNote.hidden = false;
    els.memoryNote.appendChild(el("div", "memory-note-title",
      onlyRemoved ? "Preference removed" : "Preference updated"));
    var chips = el("div", "chips");
    (update.added || []).forEach(function (item) {
      chips.appendChild(el("span", "chip ok", "+ " + item.value));
    });
    (update.superseded || []).forEach(function (item) {
      chips.appendChild(el("span", "chip", "\u21bb " + item.value));
    });
    (update.removed || []).forEach(function (item) {
      chips.appendChild(el("span", "chip bad", "\u2212 " + item.value));
    });
    els.memoryNote.appendChild(chips);
  }

  function mark(status) {
    if (status === "match") { return "\u2713"; }
    if (status === "violation") { return "\u2717"; }
    return "?";
  }

  function evidenceLabel(record) {
    var prefix = record.polarity === "avoid" ? "avoid " : "";
    var text = prefix + record.value;
    if (record.status === "unknown") {
      return text + " \u2014 unknown (metadata cannot decide this preference)";
    }
    if (record.status === "violation") {
      return text + " \u2014 supported violation";
    }
    return text + " \u2014 supported match";
  }

  function renderCards(cards) {
    clear(els.cards);
    var list = cards || [];
    if (list.length === 0) {
      els.cards.appendChild(el(
        "p",
        "muted",
        "No eligible recommendations are available for this history."
      ));
      return;
    }
    list.forEach(function (card) {
      els.cards.appendChild(buildCard(card));
    });
  }

  function buildCard(card) {
    var finalRank = card.reranked_rank || card.original_rank;
    var reranked = card.reranked_rank !== null && card.reranked_rank !== undefined;
    var moved = reranked && card.reranked_rank !== card.original_rank;
    var node = el("article", "card" + (moved ? " moved" : ""));

    var head = el("div", "card-head");
    head.appendChild(el("span", "card-rank", "#" + finalRank));
    head.appendChild(el("span", "card-title", cardTitle(card)));
    node.appendChild(head);
    node.appendChild(el("div", "card-asin", card.parent_asin));

    if (moved) {
      var up = card.reranked_rank < card.original_rank;
      node.appendChild(el("span", "card-move " + (up ? "up" : "down"),
        (up ? "\u2191" : "\u2193") + " from #" + card.original_rank + " (SASRec)"));
    } else if (reranked) {
      node.appendChild(el("span", "card-move same", "\u2014 unchanged"));
    }

    if (card.evidence && card.evidence.length) {
      var list = el("ul", "evidence");
      card.evidence.forEach(function (record) {
        var item = el("li", "status-" + record.status);
        item.appendChild(el("span", "mark", mark(record.status)));
        item.appendChild(el("span", null,
          (record.polarity === "avoid" ? "avoid " : "") + record.value));
        item.setAttribute("title", evidenceLabel(record));
        list.appendChild(item);
      });
      node.appendChild(list);
    }

    var snippets = evidenceSnippets(card);
    if (snippets.length) {
      var ul = el("ul", "snippets");
      snippets.forEach(function (text) { ul.appendChild(el("li", null, text)); });
      node.appendChild(ul);
    } else {
      node.appendChild(el("p", "muted", "No supported metadata evidence for this item."));
    }

    var details = el("details");
    details.appendChild(el("summary", null, "Show more catalogue details"));

    var ranks = el("div", "card-ranks");
    ranks.appendChild(el("div", null, "Original SASRec rank: #" + card.original_rank));
    ranks.appendChild(el("div", null, "Raw SASRec ranking score " + formatScore(card.sasrec_score)
      + " (a ranking score, not a probability)"));
    if (card.movement_summary) { ranks.appendChild(el("div", null, card.movement_summary)); }
    details.appendChild(ranks);

    var facts = el("div", "card-facts");
    facts.appendChild(factRow("Store", card.metadata && card.metadata.store, "Store unavailable"));
    facts.appendChild(factRow("Category", category(card), "Category unavailable"));
    facts.appendChild(factRow("Price", card.metadata && card.metadata.price_text, "Price unavailable"));
    details.appendChild(facts);

    var counts = el("div", "card-counts");
    counts.appendChild(el("span", "pill match", card.match_count + " match"));
    counts.appendChild(el("span", "pill violation", card.violation_count + " violation"));
    counts.appendChild(el("span", "pill unknown", card.unknown_count + " unknown"));
    details.appendChild(counts);

    var rest = remainingMetadata(card);
    if (rest.length) {
      var meta = el("ul", "snippets");
      rest.forEach(function (text) { meta.appendChild(el("li", null, text)); });
      details.appendChild(meta);
    }
    node.appendChild(details);
    return node;
  }

  function cardTitle(card) {
    if (card.metadata && card.metadata.title) { return card.metadata.title; }
    return "Title unavailable";
  }

  function category(card) {
    if (!card.metadata) { return null; }
    if (card.metadata.main_category) { return card.metadata.main_category; }
    if (card.metadata.categories && card.metadata.categories.length) {
      return card.metadata.categories.join(" > ");
    }
    return null;
  }

  function factRow(label, value, fallback) {
    var row = el("div");
    row.appendChild(el("span", "label", label + ":"));
    row.appendChild(el("span", null, value === null || value === undefined || value === "" ? fallback : value));
    return row;
  }

  function formatScore(score) {
    if (typeof score !== "number") { return String(score); }
    return (score >= 0 ? "+" : "") + score.toFixed(4);
  }

  var RELEVANT_KEYS = /description|feature|material|color|colour|brand|item|product|size|weight/i;

  /* The card view carries exactly one free-text field: metadata.details.  No .filter/.sort is
     used anywhere in this client -- the frontend renders the API sequence verbatim. */
  function detailPairs(card) {
    var out = [];
    if (!card.metadata) { return out; }
    var details = card.metadata.details || [];
    for (var i = 0; i < details.length; i += 1) {
      var pair = details[i];
      if (pair && pair.length === 2) { out.push(pair); }
    }
    return out;
  }

  /* Show the 1-2 metadata pairs most likely to explain a preference verdict. */
  function evidenceSnippets(card) {
    var pairs = detailPairs(card);
    var relevant = [];
    for (var i = 0; i < pairs.length; i += 1) {
      if (RELEVANT_KEYS.test(String(pairs[i][0]))) { relevant.push(pairs[i]); }
    }
    var chosen = relevant.length ? relevant : pairs;
    var out = [];
    for (var j = 0; j < chosen.length && out.length < 2; j += 1) {
      out.push(chosen[j][0] + ": " + chosen[j][1]);
    }
    return out;
  }

  function remainingMetadata(card) {
    var shown = {};
    var snippets = evidenceSnippets(card);
    for (var i = 0; i < snippets.length; i += 1) { shown[snippets[i]] = true; }
    var pairs = detailPairs(card);
    var out = [];
    for (var j = 0; j < pairs.length && out.length < 10; j += 1) {
      var text = pairs[j][0] + ": " + pairs[j][1];
      if (!shown[text]) { out.push(text); }
    }
    return out;
  }

  function renderAudit(audit) {
    if (!audit) { return; }
    els.candidates.textContent = String(audit.candidate_count);
    els.moved.textContent = String(audit.moved_count);
    els.rankedWith.textContent = audit.ranked_with_preference_count + " preference(s)";
    fillOrder(els.originalOrder, audit.original_order);
    fillOrder(els.rerankedOrder, audit.reranked_order);
  }

  function fillOrder(node, asins) {
    clear(node);
    (asins || []).forEach(function (asin) { node.appendChild(el("li", null, asin)); });
  }

  /* ------------------------------------------------------------------ */
  /* wiring                                                              */
  /* ------------------------------------------------------------------ */

  els.form.addEventListener("submit", send);
  els.newSession.addEventListener("click", function () { createSession(); });
  els.resetSession.addEventListener("click", function () { resetSession(); });
  els.profileSelect.addEventListener("change", function () {
    state.profileId = els.profileSelect.value;
  });

  setBusy(false);
  setState("initializing");
  checkHealth()
    .then(function () { return loadProfiles(); })
    .then(function () { return createSession(); })
    .catch(function (error) {
      setState("error");
      showBanner("error", "Demo unavailable", describeError(error));
    });
})();
