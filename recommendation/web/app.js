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
    apiVersion: document.getElementById("api-version")
  };

  var state = {
    sessionId: null,
    profileId: null,
    sending: false,
    expired: false,
    turns: 0
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

  function appendTurn(kind, meta, body) {
    var turn = el("div", "turn " + kind);
    turn.appendChild(el("div", "turn-meta", meta));
    turn.appendChild(el("div", "turn-body", body));
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
        appendTurn("agent", "Agent · " + payload.turn_id, payload.message);
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

  function renderPreferences(preferences) {
    clear(els.preferences);
    var list = preferences || [];
    if (list.length === 0) {
      els.preferences.appendChild(el("p", "muted", "No active preferences yet."));
      return;
    }
    var groups = { Avoid: [], Prefer: [], Budget: [] };
    list.forEach(function (item) {
      if (item.kind === "price_max") {
        groups.Budget.push("\u2264 " + item.value);
      } else if (item.kind === "price_min") {
        groups.Budget.push("\u2265 " + item.value);
      } else if (item.polarity === "avoid") {
        groups.Avoid.push(item.value + " (" + item.kind + ")");
      } else {
        groups.Prefer.push(item.value + " (" + item.kind + ")");
      }
    });
    Object.keys(groups).forEach(function (name) {
      if (groups[name].length === 0) { return; }
      els.preferences.appendChild(el("h3", null, name));
      var ul = el("ul");
      groups[name].forEach(function (text) { ul.appendChild(el("li", null, text)); });
      els.preferences.appendChild(ul);
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
    showBanner("info", "Memory updated", parts.join(" · "));
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
    var node = el("article", "card");

    var head = el("div", "card-head");
    head.appendChild(el("span", "card-rank", "#" + (card.reranked_rank || card.original_rank)));
    head.appendChild(el("span", "card-title", cardTitle(card)));
    node.appendChild(head);
    node.appendChild(el("div", "card-asin", card.parent_asin));

    var facts = el("div", "card-facts");
    facts.appendChild(factRow("Store", card.metadata && card.metadata.store, "Store unavailable"));
    facts.appendChild(factRow("Category", category(card), "Category unavailable"));
    facts.appendChild(factRow("Price", card.metadata && card.metadata.price_text, "Price unavailable"));
    node.appendChild(facts);

    var ranks = el("div", "card-ranks");
    var rankText = "Original SASRec rank " + card.original_rank;
    if (card.reranked_rank !== null && card.reranked_rank !== undefined) {
      rankText += " · final rank " + card.reranked_rank;
    }
    ranks.appendChild(el("div", null, rankText));
    ranks.appendChild(el("div", null, "Raw SASRec ranking score " + formatScore(card.sasrec_score)
      + " (a ranking score, not a probability)"));
    node.appendChild(ranks);

    if (card.movement_summary) {
      node.appendChild(el("div", "card-move", card.movement_summary));
    }

    var counts = el("div", "card-counts");
    counts.appendChild(el("span", "pill match", card.match_count + " match"));
    counts.appendChild(el("span", "pill violation", card.violation_count + " violation"));
    counts.appendChild(el("span", "pill unknown", card.unknown_count + " unknown"));
    node.appendChild(counts);

    if (card.evidence && card.evidence.length) {
      var list = el("ul", "evidence");
      card.evidence.forEach(function (record) {
        var item = el("li", "status-" + record.status);
        item.appendChild(el("span", "mark", mark(record.status)));
        item.appendChild(el("span", null, evidenceLabel(record)));
        list.appendChild(item);
      });
      node.appendChild(list);
    }

    var snippets = evidenceSnippets(card);
    if (snippets.length) {
      var ul = el("ul", "snippets");
      snippets.forEach(function (text) { ul.appendChild(el("li", null, text)); });
      node.appendChild(ul);
    } else if (card.metadata_status === "missing") {
      node.appendChild(el("p", "muted", "No supported metadata evidence for this item."));
    } else {
      node.appendChild(el("p", "muted", "No supported metadata evidence for this item."));
    }

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

  function evidenceSnippets(card) {
    var snippets = [];
    if (!card.metadata) { return snippets; }
    var details = card.metadata.details || [];
    details.forEach(function (pair) {
      if (pair && pair.length === 2) {
        snippets.push(pair[0] + ": " + pair[1]);
      }
    });
    return snippets.slice(0, 6);
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
