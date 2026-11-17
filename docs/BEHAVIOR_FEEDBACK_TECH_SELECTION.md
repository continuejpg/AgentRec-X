# Behavior Feedback V1 — Public Reference Review & Technical Selection

Status: implemented selection for AgentRec-X Behavior Feedback V1 (2026-11-17).  
Scope: first-party shopping-event collection, a short-term behavior profile and a switchable
Behavior-Aware Ranking Shadow.  No recommendation platform, no model retraining and no change to
the production default ranker or Decision Gate.

## 0. Decision in one paragraph

AgentRec-X already owns a Next.js/React 19 client, a FastAPI BFF, a session registry, an MCP tool
chain and an event-sourced M9 memory.  Behavior feedback therefore uses a **small first-party
collector inside the existing BFF**, backed by a local SQLite database and a React
`IntersectionObserver`; it does **not** add PostHog, Snowplow, Kafka, Flink, Redis or a separate
analytics service.  The public projects below were used as design references only.  Their event
vocabulary, recommendation identity, impression rules and privacy boundaries were reimplemented in
AgentRec-X's own terms.

## 1. Public references actually consulted

### 1.1 Amazon Personalize — Recording events

- <https://docs.aws.amazon.com/personalize/latest/dg/recording-events.html>

Borrowed concepts:

- user / session / item interaction as the unit of feedback;
- a recommendation identity that lets a later interaction be attributed to the list that produced
  it;
- impression context, including the item position;
- decoupling real-time event collection from the next recommendation request.

Not adopted: managed event trackers, AWS IAM/roles, datasets/schemas, automatic model retraining.
AgentRec-X is a local demo and must run without cloud credentials.

### 1.2 Snowplow — Ecommerce events

- <https://docs.snowplow.io/docs/events/ootb-data/ecommerce-events/>

Borrowed vocabulary: product-list view vs product-list click, product detail view, item position,
product/list metadata.  The four V1 event names are intentionally narrower:
`product_impression`, `product_click`, `product_compare`, `explicit_item_feedback`.

Not adopted: Snowplow trackers, enrichments, warehouse loaders or its full ecommerce taxonomy.

### 1.3 AWS Retail Demo Store

- <https://github.com/aws-samples/retail-demo-store>

Used only for architecture and interaction ideas (product list/detail, recommendation identity,
analytics separation).  The repository is archived; no code was migrated and no infrastructure was
created.

### 1.4 PostHog — Next.js SDK

- <https://posthog.com/docs/libraries/next-js>

Evaluated as an optional analytics SDK.  **Not used.**  AgentRec-X keeps a first-party endpoint so
that:

- only whitelisted structured events leave the page (no autocapture, no DOM/input capture);
- session recording is disabled by construction, not by an SDK switch;
- no raw shopping conversation or M9 memory text is uploaded;
- the Ranker never queries an analytics report at request time;
- the demo runs fully offline with no external API key.

### 1.5 NVIDIA Transformers4Rec / RecBole

- <https://github.com/NVIDIA-Merlin/Transformers4Rec>
- <https://github.com/RUCAIBox/RecBole>

Used as sequence/behavior-modeling references only.  V1 does **not** train a new sequential model;
the profile is a transparent fixed aggregation so it can be explained from individual events.

### 1.6 Open Bandit Pipeline

- <https://zr-obp.readthedocs.io/en/latest/>

Used as a future off-policy evaluation reference only.  There is no propensity/logged policy in
this milestone, so **no IPS/SNIPS/DR metric is computed**.  Fake propensities would be worse than
reporting the gap.

## 2. What already existed and was reused

| Need | Existing component reused | Change |
|---|---|---|
| Session identity/isolation | `web_demo.sessions.SessionStore` and opaque `web-...` ids; M9 already session-scoped through MCP `session_id` | Added per-session behavior switch and frozen recommendation snapshots |
| Browser UI | Next.js app, product cards, right-side context panel, chat composer | Added Behavior tab/header entry and item feedback buttons; no second React/chat framework |
| Backend | FastAPI BFF, existing exception mapping, SSE and request-validation conventions | Added five behavior endpoints; no new service |
| Persistence | No database existed; Python `sqlite3` is stdlib | One small SQLite file under `runs/`, not Kafka/Redis |
| Memory writes | Existing MCP `remember_preference` / M9 `PreferenceMemoryService` | No new memory store; suggestion confirmation calls the existing writer |
| Identity safety | Server-bound profile + session; browser never sends `user_id` | Behavior payload forbids extra fields and never accepts a user id |
| Product truth | Existing `ProductsPayload` / `get_product_info` / P15 evidence | Added server-assigned positions, recommendation id and ranker version |
| Replay/synthetic separation | Existing Replay mode is read-only and never a live model call | Event `event_source` tag; LIVE and AUTOMATED_BROWSER_E2E may enter the profile with separate provenance; replay/synthetic excluded |

## 3. Why not a heavier platform

| Candidate | Why not in V1 |
|---|---|
| PostHog / Snowplow / Segment SDK | External dependency, key management, autocapture/session-recording risk, and the local demo must run offline; the required events are small and structured |
| Kafka / Flink | No multi-node ingestion, no streaming joins, no operational need; SQLite handles a local single-user demo |
| Redis | Adds a daemon and lifecycle problem for a feature that needs only a tiny durable append log |
| AWS Retail Demo Store | Archived, full-stack retail platform; licence/maintenance/migration cost far exceeds the behavior-feedback scope |
| Transformers4Rec / RecBole model | Requires new training data, labels and a serving path; the milestone explicitly forbids retraining |
| Open Bandit IPS | No logged propensities and no stochastic logging policy; reporting an IPS number would be fabricated |

## 4. Event contract summary

All events carry: `event_id`, `event_type`, opaque `session_id` (envelope), server-derived
`request_id`, `recommendation_id`, `product_id`, `displayed_position`, `ranker_version`,
server timestamp and `event_source`.

Four types:

1. `product_impression` — card entered the browser viewport, at least 60% visible for 1000 ms;
   deduped by `(session_id, recommendation_id, product_id)`.
2. `product_click` — card opened details/evidence; repeated clicks are capped by the profile rule.
3. `product_compare` — a real comparison action over 2–5 products from one issued list; the primary
   product is the event `product_id`, all compared ids are in metadata.
4. `explicit_item_feedback` — Like/Dislike for one item; explicitly not an M9 preference.

Sources: `LIVE_USER_INTERACTION`, `RECORDED_REPLAY`, `SYNTHETIC_TEST`.  Replay/synthetic rows are
stored with provenance but excluded from every profile count and score.

## 5. Visibility and attribution rules

- The backend Top-K is **not** an impression.  The frontend uses `IntersectionObserver`; the BFF
  only accepts an impression whose `(recommendation_id, product_id, displayed_position)` match a
  snapshot it issued to that same session.
- Every displayed list receives a new `recommendation_id`; a re-render that does not change the
  list keeps the id, while a new/ordered list gets a new one.
- Events are validated against the frozen snapshot; a browser cannot invent a product, position or
  recommendation id.
- The production `ProductsPayload.items` order is never modified by the behavior path.  The
  Behavior OFF/ON list is a separately labelled shadow endpoint over the same frozen pool.

## 6. Privacy and data policy

Collected: opaque ids, event type, display position, server timestamp, Like/Dislike label,
compare id list, and a **server-derived** snapshot of verified product attributes.  
Never collected: names, `user_id`, chat text, M9 memory text, free-form notes, page URLs, keystrokes,
session recordings, IP geolocation or third-party analytics ids.

Controls:

- per-session personalization switch; when off the BFF returns HTTP 409 for new events;
- delete-one-session endpoint (does not touch M9 or other sessions);
- retention window (`AGENTRECX_BEHAVIOR_RETENTION_DAYS`, default 7);
- the SQLite file lives under git-ignored `runs/`; no event data is committed;
- normal `SessionStore` scoping and opaque session ids remain the authorization boundary; the
  browser cannot supply a `user_id`.

## 7. What is explicitly deferred

- propensity-logged policies, IPS/SNIPS/DR and unbiased evaluation;
- production multi-user identity/auth (the demo remains local and session-capability based);
- a trained short-term interest model;
- automatic promotion of the shadow ranker into the default path;
- dashboards/BI beyond the in-product Behavior Inspector;
- click/cart/purchase conversion goals (the current events are impression/click/compare/like, not a
  checkout funnel).

The recommendation-quality status stays **NOT ESTABLISHED**.  V1 demonstrates a real event chain and
an explainable shadow, not an online CTR or conversion lift.

## 8. V1.1 addendum (signal correction and eligibility boundary)

The original V1 selection above remains valid.  V1.1 changes three semantics, not the platform
selection:

1. **Impression is exposure context only.**  It has zero interest weight; it is retained for
   visibility, click association and future exposure-bias work.  It is not negative feedback.
2. **Compare is an operation lifecycle**, not a preference: `attempted → succeeded|failed`, with
   `compare_id` and real failure provenance.  A completed compare does not imply liking every
   compared product.
3. **Formal shadow ranking is eligible-only.**  `gate_eligible` means “in Gate ALLOW_FINALIZE and
   in `candidate_presentable_ids`”; ineligible products appear only in a clearly labelled
   diagnostic ranking and can never enter the eligible Top-K.

A minimal adapter fix now passes the session's current query into the existing P16 comparison tool;
this corrects target/role inference without relaxing any P16 role rule.  See
`AGENTREC_X_BEHAVIOR_FEEDBACK_V11_REPORT.md`.
