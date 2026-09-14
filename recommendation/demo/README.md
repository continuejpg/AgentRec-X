# Multi-turn web demo (Milestone 11)

A browser-usable, multi-turn shopping-agent demo over the **already accepted** backend.
Milestone 11 is a productization/integration milestone: it adds a web layer, a session
model and a composition root, and **no** recommendation logic.

```
browser (recommendation/web: plain HTML + CSS + JS)
   │  same-origin fetch
   ▼
FastAPI demo endpoints            recommendation/api/demo_routes.py
   │
   ▼
DemoSessionManager                recommendation/demo/sessions.py
   │  session_id → user_key, trusted_user_history, turn sequence, lock
   ▼
AgentGraph                        recommendation/agent (M7B → M10D, unchanged)
   │
   ├─ RecommendationTool → SASRecInferenceEngine        (M7A / M6)
   ├─ ProductEnricher    → MetadataIndex                (M8)
   ├─ PreferenceMemoryService → SQLitePreferenceStore   (M9)
   ├─ PreferenceCandidateMatcher                        (M10A)
   └─ PreferenceReranker                                (M10B)
   │
   ▼
ChatResponse (explicit whitelist) + grounded agent text
```

## Three separate states

These are different things, and the demo never conflates them.

| State | Where it lives | Who may change it | Lifetime |
| --- | --- | --- | --- |
| **Trusted behavioural history** | the session, copied from a `DemoProfile` | nobody during a session — application-owned and read-only | process (rebuilt from the accepted sequences artifact) |
| **Preference memory** | the accepted Milestone 9 store, per `user_key` | only the user's own explicit statements, extracted by the accepted M9 extractor | persists across requests **and** server restarts |
| **Browser transcript** | the page (JS memory only) | the page | the page |

Chat text is *never* interaction history. `"I bought B0BX5QFWQN."` does not append
anything to `trusted_user_history`; it is ordinary untrusted text that may (or may not)
produce an explicit preference. Interaction tracking, if ever wanted, belongs to a
separate trusted application-event pipeline.

## Sessions

```
DemoSession
    session_id            opaque UUID4 capability token (the only client-visible handle)
    user_key              "demo-session:<session_id>" — the M9 memory namespace
    profile_id            which server-owned demo profile seeded this session
    trusted_user_history  read-only tuple of parent_asin values
    created_at            creation time (also used for optional TTL expiry)
    creation_index        logical creation order
    next_turn_sequence    server-owned turn counter
    lock                  per-session threading.Lock
```

* **Session ids are opaque.** They are UUID4 values, not paths, database keys or user
  ids. A session id is a demo capability token; there is no authentication.
* **`user_key` is derived from the session, never from the profile.** Two sessions on
  the same demo profile therefore have completely separate preference memory.
* **Turn ids are server-owned**: `<session_id>:<sequence>`, allocated while holding the
  session's lock. A client cannot supply one, and a failed turn does not free its id.
* **One lock per session.** Simultaneous messages to one session are serialised
  (distinct ids, no interleaved memory); different sessions run independently. There is
  no global lock.
* **Bounded registry.** `max_sessions` (default 64) caps live sessions; reaching it is an
  explicit `503`, never an eviction of a live session. Optional `ttl_seconds` expiry is
  swept when a session is created.

### Reset semantics

`DELETE /v1/demo/sessions/{session_id}`:

1. takes the session's own turn lock, so an in-flight turn finishes first;
2. removes the session from the live registry — the id stops working immediately (`404`);
3. **retires** its preference-memory `user_key`, which is never reissued;
4. touches no other session in any way;
5. does **not** erase Milestone 9 rows.

Point 5 is deliberate. Accepted M9 semantics retain provenance for superseded *and*
removed entries, `SQLitePreferenceStore` exposes no user-scoped delete, and Milestone 11
forbids store-mutation logic in the web layer. A reset session's rows therefore remain as
**unreachable audit provenance**: no API path can read them, and no later session can ever
receive that namespace. A new session must be created explicitly — the server never
creates one silently.

### Persistence, stated precisely

| Concern | Survives an HTTP request | Survives a server restart |
| --- | --- | --- |
| Preference memory (SQLite) | yes | **yes** |
| Live session registry | yes | **no** — process-local, and the session id becomes unknown |

The browser therefore treats a `404 session_not_found` as "start a new session", which is
also what happens after a restart. This is a local research demo, not a durable identity
system.

## Demo profiles

`DemoProfile(profile_id, display_name, trusted_user_history, source_user_int_id,
source_length)`.

Profiles are built deterministically from the accepted processed sequences artifact using
the same rule as the accepted Milestone 7C integration:

1. walk stored user records in order (ascending `user_int_id`, so the result is
   independent of dict/hash ordering);
2. take the first records that pass a minimum-length filter;
3. supply `parent_asins[:-2] + [parent_asins[-2]]` — the training prefix plus the
   validation target, with the final leave-one-out **test target excluded**.

So no future evaluation target is ever exposed to the demo or fed to the model. Selection
never inspects scores or candidates, so a profile cannot be tuned toward a preferred
output. The public profile view carries only `profile_id`, `display_name` and the history's
*shape* — never the history itself.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/demo/health` | demo readiness (`model_loaded`, `metadata_loaded`, `demo_ready`, profiles, sessions) |
| `GET` | `/v1/demo/profiles` | the server-owned demo profiles on offer |
| `POST` | `/v1/demo/sessions` | create an isolated session (`{"profile_id": "demo-user-1"}`) → `201` |
| `GET` | `/v1/demo/sessions/{id}` | session metadata, turn count, ACTIVE preferences |
| `POST` | `/v1/demo/sessions/{id}/chat` | one turn (`{"message": "...", "k": 5}`) |
| `DELETE` | `/v1/demo/sessions/{id}` | reset that session |
| `GET` | `/demo/` | the browser demo (static assets) |
| `GET` | `/` | redirect to `/demo/` |

The three accepted Milestone 6 endpoints — `GET /health`, `GET /v1/model`,
`POST /v1/recommend` — keep their exact paths, schemas and semantics. They are never
routed through the agent. The demo API is opt-in (`create_app(..., enable_demo=True)`),
which is why `create_app()` used directly — including by the Milestone 6 tests — stays
exactly the Milestone 6 service.

### Request validation

* `message`: non-blank string, ≤ 2000 characters;
* `k`: **strict** integer in `1..100` (reuses the accepted Recommendation Tool bounds);
* `profile_id`: non-blank string;
* `extra="forbid"` everywhere.

A client cannot supply `trusted_user_history`, `history`, `parent_asins`,
`preference_snapshot`, a reranking report, `user_key`, `session_id`, `turn_id` or `route`:
none of those fields exists in any request schema, so sending one is a `422`.

### Response shape

```
ChatResponse
    api_version, session_id, turn_id, turn, route
    message                 the agent's grounded text
    active_preferences      ACTIVE memory AFTER this turn's write (panel state)
    memory_update           the accepted M9 write summary for this turn
    recommendations[]       structured cards, in backend order
    audit                   reranking_applied, counts, original/reranked order,
                            ranked_with_preferences (what THIS turn ranked against)
```

Each card carries `reranked_rank`, `original_rank`, `parent_asin`, `item_id`,
`sasrec_score`, grounded metadata (`title`, `store`, `main_category`, `price_text`,
`categories`, `details`), `match_count` / `violation_count` / `unknown_count`, the M10A
`evidence` records, `metadata_status`, `fallback_reason` and a factual
`movement_summary`.

`recommendations[]` is authoritative: it is emitted in the Milestone 10B reranked order
when reranking ran, and in the original order otherwise. The frontend renders that
sequence and never re-sorts it.

### Error mapping

| Failure | Status | Code |
| --- | --- | --- |
| unknown / expired / reset session | 404 | `session_not_found` |
| unknown demo profile | 404 | `unknown_profile` |
| live-session capacity reached | 503 | `session_capacity_exceeded` |
| demo backend not composed | 503 | `demo_unavailable` |
| missing / unknown trusted history | 422 | `invalid_history` |
| invalid Tool request | 422 | `invalid_request` |
| RecommendationTool failure | 502 | `recommendation_failed` |
| matching / reranking failure | 502 | `preference_stage_failed` |
| agent orchestration failure | 502 | `agent_failed` |
| anything else | 502 | `demo_backend_failed` |

Every `5xx` detail is authored by the mapping layer, so an internal exception message,
filesystem path or stack trace never reaches a client. A failed turn never returns `200`
with fabricated recommendation content.

## Preference timing (unchanged from M9)

A preference stated in the current message is **stored during that turn but does not
affect that turn's ranking**: the graph loads its snapshot before `decide` and persists
after `finalize`. It takes effect from the **next** turn.

The API makes this auditable rather than implicit:

* `active_preferences` is the memory state *after* the write (what the panel shows);
* `audit.ranked_with_preferences` is the snapshot that actually ranked *this* turn;
* `memory_update` reports exactly what changed, so the UI can say
  "Preference saved for future turns: avoid red" instead of implying it was applied.

## Reranking

Reranking uses the frozen Milestone 10B policy (`violation_count ASC, match_count DESC,
original_rank ASC, item_id ASC`). The web layer neither reimplements nor re-derives it.
`item_id` is unreachable for valid input with unique original ranks, so no tie-break
explanation is ever produced, and `movement_summary` states only the two ranks the
reranker actually produced.

No quality or relevance claim is made anywhere: policy adherence is not converted into
"best", "most relevant", "better" or "more personalized". The raw SASRec score is labelled
a ranking score and disclaimed as not a probability, confidence value, rating or
preference score.

## Serialization boundary

`recommendation/demo/serialization.py` is an explicit **whitelist** adapter from graph
state to the public response. There is no `dict(state)`, no `model_dump()` of internal
models and no splat of an internal mapping anywhere in it (AST-guarded by tests).

Never exposed: the trusted history (or any of it), the memory `user_key`, memory store
paths, checkpoint paths, the profile's internal `source_user_int_id`, internal
`memory_id` values, raw extraction payloads, and exception text.

Card metadata and evidence are attached by `(parent_asin, item_id)` **identity**, never by
list position, so a reordered candidate can never inherit its neighbour's facts.

## Frontend

Plain HTML, CSS and JavaScript served by FastAPI as static files. No Node toolchain, no
build step, no npm, no framework, no CDN — the page loads only same-origin assets.

* every piece of dynamic text is written with `textContent` / `createElement`;
  `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write` and `eval` appear
  nowhere in the shipped file (test-guarded over comment-stripped source);
* the client contains **no** scoring, sorting, filtering or re-ranking. It renders
  `recommendations[]` exactly as received and shows the API's rank fields;
* UI states are explicit: `initializing`, `ready`, `sending`, `success`, empty
  recommendations, validation error, session expired, server error;
* the send button is disabled while a request is in flight, so a double click cannot
  produce two submissions;
* only the session id and the selected profile id are ever held in page memory — no
  `localStorage`, no `sessionStorage`, no cookies, and never trusted history or preference
  database contents.

## Runtime composition

`DemoRuntime` / `build_demo_runtime()` / `DemoRuntime.from_env()` construct everything
once per process:

```
ServiceSettings → SASRecInferenceEngine → RecommendationTool
MetadataIndex   → ProductEnricher
SQLitePreferenceStore → PreferenceMemoryService
PreferenceCandidateMatcher, PreferenceReranker
DemoSessionManager
```

Endpoints never construct any of this: they read the runtime from the FastAPI application
state.

One value is genuinely per-request: `k`. The accepted Milestone 7B trust boundary routes
`k` exclusively through `AgentDecision`, so the runtime caches one compiled `AgentGraph`
per `(k, session user_key)`. Compiling a graph is pure Python over the *same* process-scoped
collaborators, the cache is bounded, and entries are released when a session is reset. No
model, index or service is ever rebuilt.

A missing accepted artifact fails **startup** with a clear error, not the first browser
request. (When the engine is injected — as in tests — the demo reuses that engine rather
than loading a second copy of the checkpoint.)

## Logging

Startup logs the profile count and session capacity. Nothing logs API keys, secrets,
environment variables, checkpoint bytes, store contents or SQLite rows, and raw user
messages are not logged.

## Running it

```bash
cd /root/AgentRec-X
export OMP_NUM_THREADS=8                 # required by this sandbox's libgomp

# 1. start the demo server (M6 API + M11 demo API + browser demo on one port)
.venv/bin/python -m recommendation.api.app --host 127.0.0.1 --port 8000

# 2. open the demo
#    http://127.0.0.1:8000/demo/            (the page)
#    http://127.0.0.1:8000/v1/demo/health   (readiness)
#    http://127.0.0.1:8000/docs             (OpenAPI for every endpoint)

# 3. run the formal M11 smoke (real checkpoint, real catalogue, real HTTP)
.venv/bin/python -m experiments.web_demo_smoke

# 4. run the demo test suites
.venv/bin/python -m pytest -q tests/test_demo_sessions.py tests/test_demo_api.py \
    tests/test_demo_web.py tests/test_demo_multiturn.py
```

`AGENTRECX_*` environment variables select the artifacts and the memory database:

| Variable | Meaning |
| --- | --- |
| `AGENTRECX_CHECKPOINT_PATH` | accepted checkpoint |
| `AGENTRECX_MAPPINGS_PATH` | id mappings |
| `AGENTRECX_MANIFEST_PATH` | run manifest |
| `AGENTRECX_DEVICE` | `cpu` / `cuda` |
| `AGENTRECX_HOST`, `AGENTRECX_PORT` | bind address |
| `AGENTRECX_MEMORY_DB` | preference-memory SQLite file (default `data/artifacts/demo/preference_memory.sqlite3`) |

Bind to `127.0.0.1` for local use. If a remote host needs access, forward the port
(for example an SSH tunnel); do not hard-code a provider-specific public URL.

### Browser walkthrough

1. the page loads and reports `ready`;
2. a session is created automatically for `demo-user-1`;
3. send `Recommend some products.` → cards appear, in the backend's order;
4. send `I don't want red.` → the banner says the preference was saved for future turns,
   and the audit block still shows `0` preferences ranked this turn;
5. send `Recommend again.` → the audit block now shows `1`, and the preferences panel
   lists it;
6. send `I don't care about color anymore.` → the panel empties;
7. press **Reset session** → that id is gone and a new session starts;
8. **New session** with the other profile → a separate preference panel;
9. the browser console shows no errors;
10. `GET /health`, `GET /v1/model` and `POST /v1/recommend` still answer as before.

## Limitations

* **local research demo** — bind to localhost; same-origin frontend and API;
* **no authentication or authorization** — a session id is a capability token and nothing
  more; no OAuth, no accounts, no production identity claims;
* **deterministic offline routing** — the route chooser is a small keyword rule, not an
  LLM. No provider SDK, API key or network call is involved. A provider-backed decision
  model can be added later behind the existing injected seam;
* **synthetic/demo profiles** — demo histories come from the accepted sequences artifact
  with the leave-one-out test target excluded; they are not a user directory;
* **no recommendation-quality claim** — the accepted M5 benchmark is sealed and is not
  recomputed. Preference adherence is measured, not relevance;
* **no Semantic IDs** — that is a later phase;
* **preference evidence coverage is sparse** in places — the accepted Milestone 10A
  conservative asymmetry means a readable non-match is `UNKNOWN`, so some preferences
  legitimately move nothing;
* **reset keeps audit rows** — see *Reset semantics* above;
* **the live session registry does not survive a restart** (preference memory does);
* **no server-side transcript** — the conversation lives in the page.
