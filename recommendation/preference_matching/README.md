# Preference–candidate matching (Milestone 10A)

Produces **structured, attributable evidence** describing whether each recommendation
candidate's already-attached metadata satisfies, violates or is inconclusive about each
ACTIVE user preference.

```
trusted interaction history
        ↓
RecommendationTool / SASRec                 (M6 / M7A — unchanged)
        ↓
ordered candidate list
        ↓
M8 candidate-scoped metadata / RAG          (frozen boundary)
        ↓
M9 ACTIVE preference memory                 (read-only)
        ↓
M10A matcher                                (this package)
        ↓
evidence only  ──►  (later) M10B policy
```

## 1. What M10A is

An evidence layer. It answers exactly one question per `(candidate, active preference)`
pair:

> Does the metadata we actually hold for this candidate satisfy the preference, break
> it, or fail to tell us?

It is deterministic, offline, provider-free and framework-independent. It is a plain
domain component, not a LangGraph node, so it can be reasoned about and tested on its
own.

## 2. What M10A is not

> **M10A does not rerank recommendation candidates.**

* it does not add, drop, replace or reorder a candidate;
* it does not filter on hard constraints;
* it does not compute a preference score, weight, sign or any combined number;
* it does not decide how much a violation matters;
* it does not use an LLM, embeddings or a provider API;
* it does not retrieve: there is no catalogue handle, no retriever and no store in the
  matcher, so no product outside the current candidate list is reachable.

`M10B` owns policy: weighting evidence, reranking, filtering and explanation.

## 3. Three-state semantics

```python
class EvidenceStatus(str, Enum):
    MATCH = "match"
    VIOLATION = "violation"
    UNKNOWN = "unknown"
```

Deliberately **not** a boolean, because "we cannot tell" is a first-class and common
outcome.

| Status | Exact meaning |
| --- | --- |
| `MATCH` | The supplied metadata **explicitly satisfies** the active preference: the preferred value is present in a field that can express it, or a numeric bound is met. |
| `VIOLATION` | The supplied metadata **explicitly breaks** the active preference: the forbidden value is present for an `avoid` constraint, or a numeric value falls outside the stated bound. |
| `UNKNOWN` | The available metadata is **insufficient** to decide: no record, no such field, no such value, an unreadable number, or a preference kind no field can express. |

> **UNKNOWN means the available metadata is insufficient to determine whether the
> candidate satisfies or violates the preference.**

### The conservative asymmetry

| Situation | Result | Why |
| --- | --- | --- |
| `avoid red`, metadata `Color = red` | `VIOLATION` | Finding a forbidden value is proof. |
| `prefer red`, metadata `Color = red` | `MATCH` | Finding a wanted value is proof. |
| `prefer red`, metadata `Color = blue` | `UNKNOWN` | M8 metadata is **not** an exhaustive product specification; a different listed value does not prove red is absent. |
| `avoid red`, metadata `Color = blue` | `UNKNOWN` | Absence of the forbidden value is not proof of absence. |
| `avoid red`, no colour field | `UNKNOWN` | Missing attribute. |
| no metadata record at all | `UNKNOWN` | Nothing to read. |

**Positive preferences never produce a violation.** Only negative categorical
constraints and numeric bounds can.

## 4. Supported preference-kind matrix

Derived by inspecting the actual M9 ontology (`PreferenceKind`) against the actual M8
metadata schema (`ProductMetadata`). No kind was added, removed or reinterpreted.

| M9 kind | Support | Evidence source | Can produce |
| --- | --- | --- | --- |
| `color` | **SUPPORTED** | `details.Color`, `details.Colour` | `MATCH`, `VIOLATION` |
| `material` | **SUPPORTED** | `details.Material` | `MATCH`, `VIOLATION` |
| `brand` | **SUPPORTED** | `store`, then `details["Brand Name"]` / `details["Brand"]` | `MATCH`, `VIOLATION` |
| `price_max` | **SUPPORTED** | `price_text` (numeric) | `MATCH`, `VIOLATION` |
| `price_min` | **SUPPORTED** | `price_text` (numeric) | `MATCH`, `VIOLATION` |
| `feature` | *PARTIALLY_SUPPORTED* | `features`, `title`, `description` | `MATCH` / `VIOLATION` on occurrence, else `UNKNOWN` |
| `category` | *PARTIALLY_SUPPORTED* | `categories`, `main_category`, `title` | `MATCH` / `VIOLATION` on occurrence, else `UNKNOWN` |
| `free_form_constraint` | **UNSUPPORTED_FOR_MATCHING** | — | always `UNKNOWN` |

Why `feature`/`category` are only partial: a positive occurrence in free text is real
evidence, but their absence in text proves nothing, so a miss is never a violation.
`free_form_constraint` is arbitrary text with no defined metadata slot, so it can never
be decided — it is retained as an explicit `UNKNOWN` rather than silently dropped, so a
later stage knows evidence was unavailable.

`UNSUPPORTED_FOR_MATCHING` is evaluated **before** metadata availability: no metadata
could ever resolve such a kind, so reporting `metadata_missing` would wrongly suggest
that better data would help.

## 5. Matching rules

```python
PreferenceCandidateMatcher().match(candidates=..., preferences=...)
# equivalently
match_candidates(candidates=..., preferences=...)
```

* `candidates` — the accepted M8 enrichment items (`EnrichedRecommendation`), already
  in SASRec order;
* `preferences` — either an M9 `PreferenceMemorySnapshot` (only its **ACTIVE** entries
  are used) or an explicit entry sequence (inactive entries are filtered out anyway).

The matcher accepts already-loaded data. It never opens SQLite, reloads metadata or
constructs a graph, and it takes no `user_id`, so memory lifecycle stays separate from
evidence semantics.

### Categorical kinds

Values are compared as **whole normalised tokens**, never substrings:
`red` does not match `hundred`, and `blue` does not match `blueberry`. A multi-token
preference requires all of its tokens.

### Numeric kinds

`price_text` is parsed defensively; M8 performs no currency normalisation, so M10A
performs none either.

| Preference | Metadata | Result |
| --- | --- | --- |
| `price_max 100` | `99.99` / `100` | `MATCH` (`numeric_within_limit`) |
| `price_max 100` | `100.01` | `VIOLATION` (`numeric_exceeds_limit`) |
| `price_min 50` | `49.99` | `VIOLATION` (`numeric_below_minimum`) |
| `price_min 50` | `50` / `50.01` | `MATCH` (`numeric_within_limit`) |
| any bound | absent | `UNKNOWN` (`insufficient_metadata`) |
| any bound | non-numeric (`—`) | `UNKNOWN` (`metadata_unparseable`) |

A missing price is **never** treated as zero.

## 6. Reason codes

Machine-readable and deterministic; a short `detail` string may accompany them but is
never the contract.

```
exact_value_match          explicit_value_conflict    preferred_value_absent
numeric_within_limit       numeric_exceeds_limit      numeric_below_minimum
metadata_missing           metadata_unparseable       unsupported_preference_kind
insufficient_metadata
```

## 7. Evidence schema

```python
PreferenceEvidence(
    preference_id, preference_kind, preference_polarity, preference_value,
    preference_source_text, preference_source_turn_id, preference_logical_seq,
    status, reason_code, support,
    metadata_field, metadata_value, metadata_present, detail,
)

CandidatePreferenceEvidence(
    original_rank, item_id, parent_asin, sasrec_score, evidence=(...),
)

PreferenceEvidenceReport(
    candidates=(...), active_preference_count, counts,
)
```

Every record answers *"why was candidate X marked this way?"* without guessing: it
carries the preference's id, kind, polarity, value, exact source span, source turn and
logical sequence, plus the metadata field and value the decision read.

There is deliberately **no** `reranked_rank`, `final_score`, `critic_score` or
`preference_score` anywhere in the contract. `counts` holds `match_count`,
`violation_count` and `unknown_count` only — descriptive integers with no weighting.

## 8. Preservation guarantees

| Guarantee | How |
| --- | --- |
| Candidate identity | `parent_asin` copied verbatim |
| Candidate count | one output record per input candidate, always |
| Original rank | `original_rank` copied from the Tool rank |
| SASRec score | copied exactly; never normalised, rounded, shifted or combined |
| Input order | output candidates are produced in the exact input iteration order |
| Preference universe | only ACTIVE entries; superseded and removed entries produce nothing |
| No candidate expansion | a preference naming a non-candidate product adds nothing |
| No mutation | candidates, metadata, preferences and the input list are untouched |

## 9. Active-preference-only rule

Matching consumes `PreferenceMemorySnapshot` ACTIVE entries. Superseded and removed
entries are excluded, and inactive entries are also filtered out defensively if a
caller passes a raw sequence. This ties M10A directly to the M9 semantics:

```text
"I prefer black."              → active
"Actually, I prefer red instead."  → black superseded, red active
   ⇒ matching uses red only; black produces zero evidence
```

```text
"I don't want red." "I don't want blue." "I don't care about color anymore."
   ⇒ no active colour constraints, so no colour evidence at all
```

A `REMOVE` directive is never itself an evidence-producing preference.

## 10. Determinism and complexity

Identical inputs produce structurally identical reports (no timestamps, no random tie
breaks, no timing fields). Complexity is `O(candidates × active_preferences)`; each pair
reads a bounded set of fields and needs no indexing.

Measured on the real chain: see `experiments/preference_matching_smoke.py`.

## 11. Running it

```bash
# tests (offline, synthetic fixtures, no checkpoint or metadata artifact)
.venv/bin/python -m pytest -q tests/test_preference_matching.py

# real candidates + real M8 metadata + real M9 memory
.venv/bin/python -m experiments.preference_matching_smoke
.venv/bin/python -m experiments.preference_matching_smoke --k 5 --json /tmp/m10a.json
```

## 12. Boundary summary

| Milestone | Scope |
| --- | --- |
| M8 | Candidate-scoped metadata and evidence retrieval (frozen universe) |
| M9 | Explicit preference memory with ADD / REPLACE / REMOVE lifecycle |
| **M10A** | **Evidence only**: MATCH / VIOLATION / UNKNOWN per candidate and active preference |
| M10B | Out of scope here: weighting that evidence, reranking, filtering, explanation |

M10A changes no candidate order, no SASRec score and no recommendation metric. The
formal M5 benchmark remains frozen and is not recomputed by this milestone.
