# AgentRec-X — Recommendation module

**Milestone 1: Amazon Reviews 2023 preprocessing pipeline**

This module converts raw Amazon Reviews 2023 interaction data into
chronological, k-core-filtered, deterministically indexed user sequences that
later milestones (ItemCF, SASRec, leave-one-out evaluation) can consume directly.

Milestone 1 contains **no models**. ItemCF, SASRec, agents, RAG, memory,
semantic IDs, FastAPI and training are explicitly out of scope here.

---

## 1. Purpose

Sequential recommenders are extremely sensitive to two things that are easy to
get wrong when preparing data:

1. **Time.** A user's history must be ordered oldest → newest. If it is not,
   leave-one-out evaluation ("hold out the last item") leaks the future into the
   training history and every reported metric becomes meaningless.
2. **Frequency.** Users and items with a handful of interactions make a
   benchmark noisy and make an embedding table mostly dead weight. The standard
   remedy is a *k-core*: repeatedly drop users and items below a minimum
   interaction count until the remaining graph is stable.

This module implements exactly those two steps, plus schema validation and
reproducible integer id assignment, and nothing else.

---

## 2. Expected input format

Amazon Reviews 2023 ships one gzipped JSON-Lines file per category, at
`review_categories/<Category>.jsonl.gz` (the target category is
**Sports and Outdoors**, 19.6M ratings / 10.3M users / 1.6M items). Each line is
one review. The documented "User Reviews" schema is:

| Field | Type | Meaning | Used here |
| --- | --- | --- | --- |
| `rating` | float | 1.0 – 5.0 | ✅ **label** |
| `title` | str | review title | ❌ ignored |
| `text` | str | review body | ❌ ignored |
| `images` | list | user-posted images | ❌ ignored |
| `asin` | str | variant-level product id | ❌ ignored |
| `parent_asin` | str | parent product id | ✅ **item id** |
| `user_id` | str | reviewer id | ✅ **user id** |
| `timestamp` | int | unix time (milliseconds) | ✅ **time** |
| `verified_purchase` | bool | purchase verification | ❌ ignored |
| `helpful_vote` | int | helpfulness votes | ❌ ignored |

Example line (synthetic, fields match the real schema):

```json
{"rating": 5.0, "title": "synthetic review 1", "text": "synthetic review body 1 for u1/i1",
 "images": [], "asin": "A000001", "parent_asin": "i1", "user_id": "u1",
 "timestamp": 1700000100000, "helpful_vote": 1, "verified_purchase": false}
```

Supported file types: `.jsonl`, `.jsonl.gz` (both read with the standard
library), and `.parquet` (optional, needs `pandas` + `pyarrow`).

### Why `parent_asin` is the canonical item id

Taken from the official dataset card: products that differ only by colour,
style or size share a single **parent** id, and the `asin` column of *earlier*
Amazon datasets was in fact the parent id — the card's own instruction is
"Please use parent ID to find product meta".

Choosing `parent_asin` therefore (a) collapses variants of one product into one
catalogue item instead of fragmenting its interactions, and (b) keeps this
pipeline directly comparable with prior Amazon benchmarks that were built on
parent-level item ids. The `asin` field is deliberately ignored.

### Validation rules

A record is rejected (counted, never silently repaired) when it:

* is not valid JSON, or is valid JSON but not an object → `parse_errors`;
* lacks any of `user_id`, `parent_asin`, `timestamp`, `rating` → `missing_fields`;
* has an empty/non-string `user_id` or `parent_asin` → `bad_user_id` / `bad_parent_asin`;
* has a non-numeric, NaN or infinite `rating` → `bad_rating`;
* has a non-coercible `timestamp` → `bad_timestamp`.

Raw records are treated as **read-only**: normalisation returns a new immutable
`Interaction` object and never edits the input. Timestamps are preserved exactly
as found (the real release uses *milliseconds*); only their ordering is used.

### Duplicate handling

Records repeating the same `(user_id, parent_asin, timestamp)` triple are
collapsed according to `--deduplicate`:

* `last` *(default)* — keep the last occurrence in file order (a re-imported or
  corrected review row wins);
* `first` — keep the first occurrence;
* `keep` — disable deduplication entirely.

---

## 3. Output artifacts

Three JSON files are written to `data/processed/` as
`<Category>_sequences.json`, `<Category>_mappings.json` and
`<Category>_metadata.json`. JSON is used rather than pickle so artifacts are
inspectable, diffable and language-independent; the sequence list is written
with a streaming `json.dump` so a large category does not need a second full
copy in memory.

### `<Category>_sequences.json` — the data models read

One entry per user, arrays aligned by index, oldest first:

```json
{
  "format": "agentrecx.sequences.v1",
  "category": "Synthetic",
  "num_users": 5,
  "num_items": 5,
  "num_interactions": 25,
  "sequences": [
    {
      "user_id": "u1",
      "user_int_id": 1,
      "item_ids": [1, 2, 3, 4, 5],
      "parent_asins": ["i1", "i2", "i3", "i4", "i5"],
      "unix_ms": [1700000100000, 1700000200000, 1700000300000,
                  1700000400000, 1700000500000],
      "ratings": [5.0, 5.0, 3.0, 5.0, 5.0],
      "length": 5
    }
  ]
}
```

`item_ids[k]` was reviewed at `unix_ms[k]` with `ratings[k]`. `item_ids` holds
1-based integers; `parent_asins` keeps the original strings alongside so a
predicted id can always be traced back to an Amazon product without joining
against the mapping file.

### `<Category>_mappings.json` — id tables

```json
{
  "padding": {"pad_id": 0, "first_real_id": 1, "note": "..."},
  "num_users": 5,
  "num_items": 5,
  "user2id": {"u1": 1, "u2": 2, "...": 0},
  "item2id": {"i1": 1, "i2": 2, "...": 0},
  "id2user": [null, "u1", "u2", "u3", "u4", "u5"],
  "id2item": [null, "i1", "i2", "i3", "i4", "i5"]
}
```

`id2user` / `id2item` are inverse arrays whose index *is* the integer id, with
`null` at 0 — handy for decoding model output during error analysis.

### `<Category>_metadata.json` — config, statistics, provenance

Contains the resolved configuration (`min_user_interactions`,
`min_item_interactions`, `pad_id`, dedup policy, ordering rule), the schema
field mapping and its rationale, the full normalisation report (records seen,
parse errors, rejections by reason, duplicates dropped), the round-by-round
k-core history, a before/after statistics block, and the raw input's path, size
and mtime so a run can be audited or reproduced.

---

## 4. Filtering rules

Options: `--min-user-interactions` and `--min-item-interactions`, both
defaulting to **5**. Thresholds are inclusive: an entity with *exactly* 5
interactions survives.

Filtering is a true **iterative k-core**, not a single users-then-items pass:

```
repeat:
    remove every user with fewer than min_user_interactions interactions
    remove every item with fewer than min_item_interactions interactions
until a full pass removes nothing
```

Iteration is mandatory because the two sides interact. Removing an item can
drop a user below the user threshold, and removing that user can orphan further
items. The synthetic dataset demonstrates exactly this: user `u6` enters the
loop with precisely 5 interactions (so the first user pass keeps it), loses
three of them when items `i6`/`i7`/`i8` are removed by the item pass, and is
only dropped in round 2 — which is why the run reports 3 rounds, not 1. Counts
are updated incrementally, so each pass costs `O(#interactions)`.

Interaction counts follow the usual implicit-feedback convention: each review
row counts once, so two reviews of the same item by the same user count twice.
The loop is capped by `MAX_K_CORE_ROUNDS` (100) purely as a safety net; the
graph strictly shrinks each round, so convergence is guaranteed.

---

## 5. ID conventions

* **Item id `0` is reserved for padding.** No real item may receive it, so a
  downstream embedding table can safely use index 0 as PAD.
* **Real item ids start at `1` and are contiguous**: `1 … N` for `N` surviving
  items. (Padding is reserved for the user table too, for symmetry.)
* **Assignments are deterministic**: ids are handed out in ascending order of
  the raw string id. The mapping therefore does not depend on input file order,
  dictionary iteration order, or `PYTHONHASHSEED`.
* **Mappings cover surviving entities only.** Filtered-out users and items are
  absent from `user2id` / `item2id`.
* Only the *filtered survivors* are numbered, so ids are dense — convenient for
  embedding lookups, at the cost of changing if the thresholds change.

---

## 6. Running it

Everything runs on CPU; the synthetic path needs no downloads and no GPU.

### Tests

```bash
# preferred (needs pytest; see requirements-dev.txt)
.venv/bin/python -m pytest tests/ -v

# dependency-free fallback, same 27 tests
python tests/test_preprocess.py
```

### Synthetic end-to-end smoke test

```bash
# 1. write the tiny synthetic dataset (8 users, 8 items, 35 rows, 3 malformed lines)
python -m tests.sample_data --out data/raw/synthetic_reviews.jsonl

# 2. run the pipeline
python -m recommendation.preprocess \
    --input data/raw/synthetic_reviews.jsonl \
    --category Synthetic

# 3. inspect the artifacts
ls -l data/processed/
python -c "import json;d=json.load(open('data/processed/Synthetic_sequences.json'));print(d['num_users'],d['num_interactions']);print(d['sequences'][0])"
```

Expected synthetic outcome (defaults, min = 5/5):

```
Before filtering: 33 interactions, 8 users, 8 items
After filtering : 25 interactions, 5 users, 5 items
k-core rounds   : 3  -> [(1, 2, 3), (2, 1, 0), (3, 0, 0)]
```

### CLI reference

```bash
python -m recommendation.preprocess --help
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-i, --input` | *(required)* | raw `.jsonl` / `.jsonl.gz` / `.parquet` file |
| `-c, --category` | `Sports_and_Outdoors` | name used in artifact filenames |
| `--min-user-interactions` | `5` | k-core threshold for users |
| `--min-item-interactions` | `5` | k-core threshold for items |
| `-o, --output-dir` | `data/processed` | where artifacts are written |
| `--deduplicate` | `last` | `last` \| `first` \| `keep` |
| `-q, --quiet` | off | suppress progress output |

### Processing a real (small) Amazon Reviews 2023 sample

`data/` is git-ignored, so the raw archive stays local. The category files live
on the McAuley Lab mirror (verified reachable; the older
`datarepo.eng.ucsd.edu/mcauley_group/...` path now returns 404):

```
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz
```

The full Sports and Outdoors file is **~2.6 GB compressed**, so do not download
it for an integration test. Fetch a bounded byte range instead: a gzip stream
decompresses correctly from offset 0, so the first N bytes yield valid records
(only the final, truncated line has to be dropped).

```bash
mkdir -p data/raw
URL=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz

# 1. bounded prefix: first 40 MB of the gzip stream (~264k review lines)
curl -sS -r 0-39999999 -o data/raw/_chunk.jsonl.gz "$URL"
gunzip -c data/raw/_chunk.jsonl.gz > data/raw/_chunk.jsonl   # exits 1 at the truncated tail

# 2. keep the first 100,000 complete JSON records, discard the partial last line
python - <<'PY'
import json
kept = 0
with open("data/raw/_chunk.jsonl") as fin, open("data/raw/Sports_and_Outdoors_sample.jsonl", "w") as fout:
    for line in fin:
        if kept >= 100_000:
            break
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue                      # the truncated tail line
        fout.write(line + "\n")
        kept += 1
print("kept", kept)
PY
rm -f data/raw/_chunk.jsonl data/raw/_chunk.jsonl.gz

# 3. preprocess. A 100k *prefix* is not the full 5-core benchmark, so the default
#    thresholds of 5 would discard most of it; 2/2 is the integration-test setting.
python -m recommendation.preprocess \
    --input data/raw/Sports_and_Outdoors_sample.jsonl \
    --category Sports_and_Outdoors_sample \
    --min-user-interactions 2 --min-item-interactions 2
```

Verified on 2026-09-13: 100,000 raw records → 99,941 interactions (0 parse
errors, 0 rejections, 59 duplicate triples collapsed) → 8,081 users / 11,907
items / 39,838 interactions after a 6-round k-core. See AGENTS.md for the rule
that prefix-sample figures such as sparsity and survival rate must **not** be
read as properties of the full category.

For the full category, pass the `.jsonl.gz` file directly — the loader streams
gzip, so peak memory scales with the *filtered* graph rather than the file size.

### Evaluation

The unified, model-independent evaluation protocol that later consumes these
artifacts (temporal leave-two-out split, full-ranking candidate semantics,
seen-item masking, HR@K / Recall@K / NDCG@K) lives in
[`evaluation/README.md`](evaluation/README.md). It is deliberately separate from
preprocessing and contains no recommender.

The SASRec data contract and architecture live in
[`datasets/README.md`](datasets/README.md), and the Milestone 4 training
objective/trainer in [`training/README.md`](training/README.md). Milestone 5 is the
first stage intended for full-scale RTX 4090 training.

### Environment

`recommendation/` imports only the Python standard library, so the pipeline
runs on a bare interpreter. `pytest` is the sole development dependency
(`requirements-dev.txt`); `pandas`/`pyarrow` are needed only for `.parquet`
input. In this repository a workspace virtualenv is used:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

---

## 7. How this connects to later milestones

**ItemCF.** Use `parent_asins` (or `item_ids`) per user as the "basket": item
co-occurrence is counted within a user's history. Dense ids from `item2id` index
the similarity matrix directly, and id 0 is free for padding. The k-core means
every item has ≥ 5 interactions, so no similarity row is estimated from a single
observation.

**SASRec.** The item-id sequences are already the model input; `item2id` is the
vocabulary (size `num_items + 1` with PAD at 0). `unix_ms` is available if you
want time-based position features. Because each sequence is chronological, the
standard pipeline applies cleanly: train on `item_ids[:-2] → item_ids[-2]`,
validate on the second-to-last item, and test on the last item. No shuffling of
histories may ever be introduced before that split.

**Leave-one-out evaluation.** Hold out `item_ids[-1]` as the test target and
`item_ids[-2]` as validation. Never re-sort or re-filter *after* splitting:
that would move the held-out item's timestamp relative to the history and leak
the future. Filtering happens here, before any split, which is exactly the
ordering this milestone guarantees.

**Agent layer (M7A–M7C).** The accepted Tool (`recommendation/tools/`), the
offline LangGraph graph (`recommendation/agent/`) and the real-chain integration
(`tests/test_agent_tool_e2e.py`, `experiments/agent_tool_e2e_smoke.py`) consume
these artifacts without changing any recommender semantics. Milestone 7C validates
composition only — it does not recompute the formal benchmark, and it makes no
recommendation-quality claim.

**Product metadata and candidate-scoped RAG (M8).** `recommendation/catalog/` builds
a deterministic `parent_asin`-keyed artifact from the official Amazon Reviews 2023
product-metadata file, and `recommendation/rag/` retrieves attributable evidence for
**only** the candidates the recommender already produced. Neither layer changes
candidate generation, ranking or the accepted mapping. See
[`catalog/README.md`](catalog/README.md) and [`rag/README.md`](rag/README.md).

**Preference memory (M9).** `recommendation/memory/` stores *explicit* conversational
preferences, user-scoped, with provenance and a deterministic lifecycle. It is a
separate domain from the trusted interaction history SASRec consumes: it holds no
behavioural events, and conversational statements are never converted into interactions.
Preference memory does **not** modify recommendation candidate order in M9 — candidate
identity, count, rank and raw SASRec score are untouched. See
[`memory/README.md`](memory/README.md).

**Preference–candidate evidence (M10A).** `recommendation/preference_matching/` reports,
for each candidate and each ACTIVE preference, whether the candidate's already-attached
metadata satisfies it (`MATCH`), breaks it (`VIOLATION`) or cannot decide
(`UNKNOWN`). It is an evidence layer only: it does **not** rerank, filter, add or drop
candidates, and it computes no combined score. Missing metadata is always `UNKNOWN`,
never a match or a violation. See
[`preference_matching/README.md`](preference_matching/README.md).

**Deterministic reranking (M10B).** `recommendation/reranking/` reorders those same
candidates under an explicit lexicographic policy — fewer explicit violations first,
then more explicit matches, then the original SASRec rank, then `item_id` as a final
deterministic fallback. It never adds, removes or filters candidates, never modifies the
raw SASRec score and computes no weighted score; `UNKNOWN` evidence is neutral. See
[`reranking/README.md`](reranking/README.md). The same package carries the Milestone 10C
**policy evaluation** (displacement, top-k overlap, adherence, coverage, consistency and
movement attribution) — an observational layer that reuses the production reranker and
reports no relevance or quality metric.

---

## 8. Module layout

| Path | Responsibility |
| --- | --- |
| `recommendation/config.py` | paths, schema field mapping, thresholds, id conventions |
| `recommendation/io_utils.py` | raw reading, validation, dedup, atomic JSON writing |
| `recommendation/preprocess.py` | ordering, iterative k-core, id mapping, statistics, CLI |
| `recommendation/evaluation/` | unified model-independent evaluation protocol (see its README) |
| `recommendation/baselines/` | reference recommenders — currently ItemCF (see its README) |
| `recommendation/datasets/` | SASRec training dataset + inference encoder (see its README) |
| `recommendation/models/` | SASRec architecture (see its README) |
| `recommendation/training/` | SASRec loss + minimal deterministic trainer (see its README) |
| `recommendation/inference/` | serving inference engine + deterministic top-k ranking (see its README) |
| `recommendation/api/` | FastAPI recommendation service (see its README) |
| `recommendation/tools/` | Agent-facing Recommendation Tool contract (see its README) |
| `recommendation/agent/` | Minimal LangGraph agent orchestration over the Tool (see its README) |
| `recommendation/catalog/` | `parent_asin`-keyed product metadata: normalization, artifact, coverage, lookup (see its README) |
| `recommendation/rag/` | Candidate-scoped product evidence retrieval (see its README) |
| `recommendation/memory/` | Explicit conversational preference memory: schema, stores, lifecycle (see its README) |
| `recommendation/preference_matching/` | Preference–candidate evidence: MATCH / VIOLATION / UNKNOWN (see its README) |
| `recommendation/reranking/` | Deterministic preference-aware reranking, plus its policy evaluation (see its README) |
| `tests/sample_data.py` | deterministic synthetic dataset + expected results |
| `tests/test_preprocess.py` | 27 tests (pytest-compatible, dependency-free runner included) |
| `tests/test_agent_tool_e2e.py` | Milestone 7C real-chain E2E: graph -> Tool -> real engine -> accepted checkpoint |
