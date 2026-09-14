# Preference memory (Milestone 9)

A trustworthy conversational memory layer that **does not touch the recommendation
trust boundary**.

```
TRUSTED INTERACTION MEMORY                     PREFERENCE MEMORY (this package)
──────────────────────────                     ────────────────────────────────
chronological parent_asin history              explicit user statements
owned by the application                       "I prefer lightweight hiking gear"
lives in graph/application input state         "I don't want red"
reaches SASRec via RecommendationContext       "my budget is under $100"
                                               "I don't care about color anymore"
        │                                              │
        │  no field, method or table                   │  injected extractor
        │  in this package can express it              ▼
        │                                       PreferenceCandidate  (untrusted)
        │                                              │  strict validation
        │                                              ▼
        │                                       PreferenceMemoryService
        │                                              │
        ▼                                              ▼
   RecommendationTool ──► SASRec              PreferenceStore (in-memory | SQLite)
```

The two domains are different types in different packages and are never collapsed into
one free-form profile.

---

## 1. Guarantees

> **Preference memory does not modify recommendation candidate order in M9.**

> **Conversational statements are never converted into trusted behavioral interactions.**

Concretely:

| Guarantee | How it is enforced |
| --- | --- |
| Only explicit user statements are stored | extraction runs on user-authored text only; the service's entry point takes `user_message` |
| Nothing is inferred | preferences are never derived from SASRec history, candidates, catalogue metadata, RAG evidence, ratings or model scores |
| The Agent cannot touch interaction history | no memory type has an interaction/`parent_asin`/`item_id` field; the store exposes no event API |
| Conversational text cannot become a behavioural event | there is no code path from `process_turn` to `RecommendationContext`; asserted by an adversarial regression |
| Candidates are never added, dropped or reordered | the memory layer never sees a `ToolResult`; asserted with/without preferences on the real chain |
| RAG stays candidate-scoped | preference text may only augment the *query*; the retrieval universe is unchanged |
| No provider API | extraction is injected and offline; the shipped extractor is deterministic rules |
| Users never see each other's memory | explicit `user_key` on every read and write; primary key is `(user_key, memory_id)` |
| Nothing is erased silently | supersession happens only on explicit correction intent, and removal only on explicit retraction; both tombstone the entry and retain provenance |
| Independent constraints are not conflated | "I don't want red" + "I don't want blue" keeps both; no preference kind is treated as a single-valued slot |
| M9 is not a conversation archive | credentials are rejected by schema; unrelated text yields no preference |

## 2. Schema

`PreferenceMemoryEntry` (strict, `extra="forbid"`, frozen):

```
memory_id       str            deterministic, derived from the entry's ORIGIN
user_key        str            owning user/session
kind            PreferenceKind
value           str            normalised, <=120 chars, credential-guarded
polarity        prefer | avoid
source_text     str            exact span of the user's message (provenance)
source_turn_id  str            which turn it came from
extractor       str            which extractor produced it
status          active | superseded | removed
logical_seq     int >= 1       monotonic per user; the stable sort key
created_at      float          unix timestamp
supersedes      str | null     memory_id this entry replaced
superseded_by   str | null     memory_id that replaced this entry
```

`PreferenceMemorySnapshot` is an immutable, deterministically ordered view
(`logical_seq`, then `memory_id`) produced by a read.

### Preference kinds

Deliberately narrow — each exists because a shopping constraint needs it:

`category`, `feature`, `brand`, `price_max`, `price_min`, `color`, `material`,
`free_form_constraint`.

`free_form_constraint` is the explicit escape hatch: an explicit statement that does not
fit the narrow kinds is stored under a truthful generic kind rather than being forced
into a wrong one. This is a shopping ontology, not a user-profile schema — no age,
gender, income, health, location or personality fields exist.

## 3. Extraction

```python
extract(user_message) -> PreferenceExtraction(preferences=(...), removals=(...))
```

The seam is injected, so the store and service contracts do not change if a future LLM
adapter replaces the shipped extractor. Two implementations ship:

* `RuleBasedPreferenceExtractor` — conservative, offline, dependency-free;
* `ScriptedPreferenceExtractor` — deterministic test double.

### Supported explicit syntax

| Wording | Extracted |
| --- | --- |
| "my budget is under $100" | `price_max = 100` |
| "my budget is at least $50" | `price_min = 50` |
| "I don't want anything over $80" | `price_max = 80` |
| "I prefer blue" / "I prefer a blue jacket" | `color = blue` |
| "I prefer lightweight hiking gear" | `category = lightweight hiking` |
| "I need a waterproof jacket" | `feature = waterproof` |
| "I don't want red" | `color = red` (avoid) |
| "I never want leather" / "I avoid plastic" | `material = …` (avoid) |
| "I prefer the brand Acme" | `brand = Acme` |
| "I'm looking for hiking boots" | `category = hiking` |
| "I prefer lightweight" | `free_form_constraint = lightweight` |
| "I don't care about color anymore" | retraction of `color` (no new entry) |
| "actually, I prefer blue instead" | `color = blue` with `mode=REPLACE` |
| "I prefer blue instead of black" | `color = blue` with `mode=REPLACE, replaces=black` |
| "make that blue" | `color = blue` with `mode=REPLACE` |

Direction markers are matched as whole phrases, so `"at least"` is never misread as an
upper bound, and a negated price ("don't want anything over $80") is stored as a price
ceiling rather than a product avoidance. Correction markers (`instead`, `instead of X`,
`make that`, `I meant`) set `mode=REPLACE`; everything else is `ADD`.

### What is deliberately not extracted

Behavioural and unrelated text yields nothing: `"I bought B0BX5QFWQN yesterday."`,
`"I clicked B0BBFB48YQ."`, `"You recommended a fishing line."`, `"The model scored it
5.8."`, greetings, email addresses, phone numbers, and any text carrying a credential.

The credential guard rejects API-key, token, AWS-key and card-number shapes, while
ordinary values (`nothing-like-this`, `waterproof membrane`, `non-slip`, `159.00`) pass
through — both directions are regression-tested.

## 4. Structured extraction is untrusted

`PreferenceCandidate` is what an extractor *proposes*; the service re-validates every
candidate before storage. A candidate has no field for an item id, an interaction event,
a score, a candidate list, a file path or a memory id, so a hostile or buggy extractor
has nowhere to put them, and `extra="forbid"` turns an attempt into a hard error.
Malformed output raises `PreferenceExtractionError` and leaves existing memory
untouched.

## 5. Lifecycle semantics

| Case | Behaviour |
| --- | --- |
| **Add** | a validated candidate becomes an `active` entry at the next `logical_seq` |
| **Duplicate** | an entry whose `(kind, value, polarity)` is already active is skipped (`skipped_duplicates`) |
| **Idempotency** | the same `(user_key, source_turn_id)` with the same candidate set is a no-op (`already_processed`); the identity is origin-keyed, not text-keyed |
| **Coexistence** | independent constraints coexist. "I don't want red" then "I don't want blue" keeps **both** avoidances, and "I prefer black" then "I prefer blue" keeps **both** preferences. No kind is a singleton slot. |
| **Replacement** | only an explicitly corrective statement retracts anything. `mode=REPLACE` (from "instead", "instead of X", "make that", "I meant") supersedes same-kind, same-polarity entries; the corrected entry becomes `superseded` with `superseded_by` set, and the newer entry records `supersedes`. `replaces` names the corrected value when the text provides one; otherwise the statement's own value is the target, which is how "actually I don't want red instead" resolves an earlier "I prefer red". |
| **Removal** | an explicit retraction (`"I don't care about color anymore"`) tombstones matching entries as `removed`; no new entry is created and no history is erased |
| **Multi-valued kinds** | `feature`, `category` and `free_form_constraint` accumulate, because two features are not competing values |

### ADD / REPLACE / REMOVE

The contract distinguishes three operations, so replacement is never inferred from the
preference kind:

| Mode | Meaning | Retracts |
| --- | --- | --- |
| `ADD` *(default)* | add an independent constraint | **nothing** |
| `REPLACE` | the statement corrects an earlier one | same kind + same polarity, narrowed to `replaces` when given (a named target is corrected whatever its polarity) |
| `REMOVE` | the statement retracts constraints | converts to a removal directive; stores no value |

Because `ADD` is the default and the only automatic retraction is for `REPLACE`, a
second same-kind statement is never treated as sufficient evidence of replacement. A
same-value, opposite-polarity pair stated without correction intent is kept as a
faithful contradiction — visible in the audit trail rather than silently resolved —
and is resolved deterministically once the user states correction intent.

## 6. Read surface

```python
get_active_preferences(user_key) -> PreferenceMemorySnapshot   # active only
get_memory_history(user_key)    -> PreferenceMemorySnapshot   # full audit trail
```

Ordering is deterministic (`logical_seq`, then `memory_id`). Reading never writes: the
Agent's read node issues no store mutation, so a read-only turn cannot change memory.
Only active entries are constraints; superseded and removed entries exist for audit.

## 7. Persistence

Two interchangeable backends behind one `PreferenceStore` contract:

* `InMemoryPreferenceStore` — unit tests and ephemeral runtimes;
* `SQLitePreferenceStore` — Python-standard-library `sqlite3`, no new dependency, no
  Redis/Postgres/vector database.

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE preference_memory (
    memory_id TEXT NOT NULL,      user_key TEXT NOT NULL,
    kind TEXT NOT NULL,           value TEXT NOT NULL,
    polarity TEXT NOT NULL,       source_text TEXT NOT NULL,
    source_turn_id TEXT NOT NULL, extractor TEXT NOT NULL,
    status TEXT NOT NULL,         logical_seq INTEGER NOT NULL,
    created_at REAL NOT NULL,     supersedes TEXT, superseded_by TEXT,
    PRIMARY KEY (user_key, memory_id)
);
CREATE INDEX idx_pref_user_status_seq ON preference_memory (user_key, status, logical_seq);
CREATE INDEX idx_pref_user_turn       ON preference_memory (user_key, source_turn_id);
```

The schema version is stored in `meta.schema_version` and verified on open, so a future
migration is explicit rather than silent. The `(user_key, memory_id)` primary key makes
cross-user leakage structurally impossible and lets identical ids exist per user.

* the database path is configurable and no path is hard-coded (`AGENTRECX_MEMORY_DB` is
  the conventional environment variable);
* `*.db` / `*.sqlite*` runtime files are git-ignored and never committed;
* tests use temporary directories and `:memory:`;
* mutating calls commit per statement; multi-step updates (supersede + add) run inside
  one `BEGIN IMMEDIATE` transaction, and nested blocks that fail roll the whole unit
  back rather than leaving a half-applied supersede.

## 8. Agent integration

Topology, conditional on what is injected:

```
no memory,  no enricher :  START → decide ─┬─ direct ──────────────► finalize → END
                                           └─ recommend ───────────► finalize → END
+ enricher (M8)         :  START → decide ─┬─ direct ──────────────► finalize → END
                                           └─ recommend → enrich ──► finalize → END
+ memory (M9)           :  START → load_memory → decide
                                  ├─ direct ───────────────────────► finalize
                                  └─ recommend → [enrich] ─────────► finalize
                                                                        ↓
                                                            persist_memory → END
```

* `load_memory` runs before `decide`, so both routes see the same snapshot, and it only
  reads.
* `persist_memory` runs after `finalize` on both routes, so the direct route can read
  and update conversational memory **without invoking the recommender at all**.
* The direct route never opens the metadata layer (asserted with a lookup that raises if
  touched) and performs no inference.
* Only `user_message` reaches the extractor: system prompts, tool output, RAG evidence
  and model reasoning are never passed to it.

### Turn semantics (documented deliberately)

A preference stated in a message is **persisted immediately** but takes effect from the
**next** turn. The retrieval query and the rendered preference block are built from the
snapshot loaded at the start of the turn, so a turn can never appear to have let its own
statement influence the candidates it returned. Configure with
`AgentGraph(..., memory_service=..., user_key=...)`; omitting them preserves the accepted
M7B/M7C/M8 behaviour exactly.

### Query augmentation (opt-in, documented format)

When memory is active, the retrieval query becomes:

```
<user query>
preferences:
- <kind>: <prefer|avoid> <value>
```

This changes only *which evidence fragments* are selected from the metadata of the
already-fixed candidate set. It cannot add, drop or reorder a candidate, and it never
touches interaction history. Disable with `augment_query_with_preferences=False`.

## 9. Grounded response

The deterministic formatter may print stored preferences alongside grounded metadata:

```
Top 3 candidate(s) from the sequential recommender, with catalogue facts where available:

Your stated preferences (stored from your own messages; not used to rank these candidates):
- category: prefers lightweight hiking

1. B0BX5QFWQN (ranking score +5.8085)
   Title: ...
```

The label is explicit that these did not rank the candidates. M9 presents a *stored
preference* and *metadata evidence*; it never invents a preference-match score, never
says "this product perfectly matches your preferences", and never describes the raw
SASRec score as a probability, confidence or rating.

## 10. Performance (engineering diagnostics)

Measured with SQLite in a temporary directory:

| Operation | Value |
| --- | --- |
| preference read (`get_active_preferences`) | ~0.03 ms |
| preference write (one turn) | ~0.4 ms |
| store reopen | ~0.3 ms |

No full-text or vector index is needed or present.

## 11. Running it

```bash
# tests (offline, no database artifact produced)
.venv/bin/python -m pytest -q tests/test_memory_store.py tests/test_memory_service.py tests/test_agent_memory.py

# smoke: multi-turn lifecycle across a reopened store, then the real M8 chain
.venv/bin/python -m experiments.memory_smoke
.venv/bin/python -m experiments.memory_smoke --skip-real      # memory part only
.venv/bin/python -m experiments.memory_smoke --k 5 --json /tmp/m9.json
```

## 12. Boundary

* **Replacement is explicit, not inferred.** A second statement of the same kind adds
  an independent constraint unless the user expressed correction intent, so preference
  memory never forgets a constraint the user did not retract.
* **M9 does not rerank.** Preference memory may inform the final grounded response, the
  retrieval query and structured context for later work, but it must not change
  candidate identity, count, rank or raw SASRec score. Preference-aware reranking,
  scoring and critique belong to **M10**.
* **M9 is not a general conversation archive** and not a user profile: only explicit
  shopping preferences, with provenance, are stored.
* **No hosted LLM** is required or used; a provider adapter can implement the same
  injected extraction interface later without changing this package's contracts.
