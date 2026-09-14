# Catalog metadata layer (Milestone 8-A)

Turns the official Amazon Reviews 2023 **product metadata** file into a deterministic,
`parent_asin`-keyed artifact, and exposes a narrow read-only lookup over it.

```
official meta_<Category>.jsonl.gz  (raw, immutable, git-ignored)
        │  normalize (deterministic, offline)
        ▼
<Category>_products.jsonl  (self-describing envelope + one record per product)
        │  load once
        ▼
MetadataIndex  ──lookup(parent_asin)──►  ProductMetadata | MissingMetadata
```

Metadata attaches descriptive facts to the identity the recommender already uses. It
never redefines identity and it is never a recommendation signal.

---

## 1. Source and provenance

| Item | Value |
| --- | --- |
| Dataset | Amazon Reviews 2023 (McAuley Lab) — the same provenance as this project's review interactions |
| Domain | `meta_categories` (product metadata), **not** `review_categories` |
| URL | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz` |
| Local raw path | `data/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz` (git-ignored) |
| Raw bytes | `1,037,418,105` |
| Raw SHA-256 | `f75abcf0af21db0a6c6701f29d1f9b94480557ecbdc238dfc39ebf83a3c9fc3a` |
| Source records | `1,587,421` |
| License / terms | Governed by the Amazon Reviews 2023 dataset card; the raw file is never committed and never modified |

The raw file is acquired **once**, manually, before any normalization:

```bash
mkdir -p data/raw/meta_categories
curl -sS -o data/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz
sha256sum data/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz
```

No scraped pages, third-party mirrors, synthetic records or generated descriptions are
used. Normalization performs **no network access** — the raw file is the only input.

## 2. Actual source schema

Determined by inspecting the real file, not from memory. Every record carries
`parent_asin`, `title`, `main_category`, `average_rating`, `rating_number`,
`features`, `description`, `price`, `images`, `videos`, `store`, `categories`,
`details`, `bought_together`, and rarely `subtitle`/`author`.

| Source field | Type seen | M8-A treatment |
| --- | --- | --- |
| `parent_asin` | `str` | **join key**; opaque, stripped, whitespace-clean, never pattern-matched |
| `title` | `str` | `title` |
| `subtitle`, `author` | `str` (rare) | `subtitle`, `author` |
| `store` | `str` | `store` — kept under its source name, **not** renamed to "brand" |
| `main_category` | `str \| null` | `main_category` |
| `categories` | `list[str]` | `categories` (order preserved) |
| `features` | `list[str]` | `features` (order preserved) |
| `description` | `list[str]` | `description` (order preserved) |
| `price` | `float` (rarely a placeholder `str` such as `—`) | `price_text`, canonical text of the **parsed** value |
| `average_rating` | `float` | `average_rating` |
| `rating_number` | `int` | `rating_number` |
| `details` | `dict[str,str]` | `details`, source attribute names kept (`"Brand Name"`, `"Color"`, `"Material"`, …) |
| `images`, `videos`, `bought_together` | `list` | **ignored** — not needed for text grounding, and keeping them would bloat the artifact |

There is no `item_id` in the schema, and M8-A never invents one.

## 3. Normalized schema

`ProductMetadata` (`schemas.py`), `extra="forbid"`, `frozen=True`:

```
parent_asin     str            required, opaque
title           str | None
subtitle        str | None
author          str | None
store           str | None     source 'store', not renamed
main_category   str | None
categories      tuple[str,...] order preserved
features        tuple[str,...] order preserved
description     tuple[str,...] order preserved
price_text      str | None     canonical text of the PARSED value (not the raw token)
average_rating  float | None
rating_number   int | None
details         tuple[(str,str),...]  source key order preserved
source          str            provenance label
```

`MissingMetadata(parent_asin, status="missing", reason=...)` is the first-class
representation of "the source does not cover this item".

### What is never inferred

`brand` is not read out of `title`; `category` is not classified from free text;
`price` is not parsed out of a description; `material`/`color`/`gender` are not
guessed from a name. If the source does not carry a value, the field stays
`None`/empty. A `title` that happens to start with a brand name leaves `store` empty —
asserted by test.

### Normalization rules

* surrounding whitespace is stripped; **interior** spacing is preserved;
* text is never paraphrased, summarised, truncated for meaning or generated;
* list order is preserved; blank entries are dropped; exact duplicates *within one
  field of one record* are collapsed (they would otherwise be scored twice);
* `details` keys and values are stripped; blanks dropped; source key order preserved;
* `price_text` is the **canonical text of the parsed value**, not the raw lexical
  token: the source field is a JSON number, so `json.loads` has already produced a
  Python number and it is rendered with `repr` (`55.0` → `"55.0"`, and a source
  exponent form `1e2` → `"100.0"`). A source *string* is kept stripped and verbatim,
  which matters because the real data stores an em dash `"—"` as the placeholder for
  an unknown price. Booleans and `null` are dropped. No currency is ever parsed and no
  value is inferred from another field;
* non-finite/negative/unparseable numeric values become `None`, never guesses.

## 4. Duplicate and malformed policy

| Situation | Policy | Counted as |
| --- | --- | --- |
| Same `parent_asin` appears again | **first record in file order wins** (`DUPLICATE_POLICY = "first"`), matching the accepted preprocessing `--deduplicate` vocabulary | `duplicate_keys` + an audited example |
| Line is not valid JSON | reject | `parse_errors` |
| Valid JSON but not an object | reject | `non_object_records` |
| `parent_asin` missing/blank | reject — it cannot be joined to the catalog | `missing_parent_asin` |
| Field has the wrong type | that field becomes `None`; the record survives | — |
| `parent_asin` outside the accepted catalog | not written; counted | `records_outside_catalog` |

Nothing is silently repaired, and no record is dropped for a *field* problem.

## 5. Artifact and manifest

`build_metadata_artifact()` writes a self-describing JSONL artifact:

```
line 1 : {"format": "agentrecx.catalog_products.v1", "category", "normalization_version",
          "duplicate_policy", "source": {...raw sha256, bytes, mtime, url...},
          "counts": {...}, "coverage": {...}, "catalog_only": true}
line 2+: one normalized product record per line, fields in a fixed schema order
```

Records are emitted in a fixed field order and the envelope is written **before** the
records (assembled from a temporary records file), so:

* the artifact bytes depend only on the input, not on dict/hash iteration order;
* there is no seek-back or truncate step, removing a partial-write failure mode;
* a run timed out or interrupted leaves no half-written artifact at the destination
  (the final file is renamed into place only after the whole thing is written);
* wall-clock timing lives in the **manifest**, never in the artifact, so the artifact
  digest is reproducible.

`<Category>_products_manifest.json` records the source URL, raw SHA-256, raw bytes,
domain/category, normalization version, record counts, duplicate and malformed counts,
catalog coverage statistics, the processed artifact's SHA-256/size, and run timing.

### Population accounting (raw source vs processed artifact)

Two different "outside catalog" quantities exist and must not be confused:

```
raw source metadata outside the catalog     = 1,430,675   (real source records, deliberately not written)
processed metadata outside the catalog      =         0   (the artifact is catalog-only)
```

| Quantity | Value |
| --- | --- |
| `source_records_total` | 1,587,421 |
| `source_records_matching_catalog` | 156,746 |
| `source_records_outside_catalog` | 1,430,675 |
| `processed_catalog_only_records` | 156,746 |
| `processed_records_outside_catalog` | **0** |

with the identities `matching + outside == total` (`156,746 + 1,430,675 = 1,587,421`)
and `processed_catalog_only_records == source_records_matching_catalog`.

Because the source has no duplicate `parent_asin` and no malformed record in this
category, *source records* and *distinct valid source keys* coincide here. In general
they would not, and the identity above is stated over valid unique source records.

`records_outside_catalog` is the manifest field for the raw-source figure; the
processed figure is `0` by construction whenever the artifact is catalog-only (the
default), which is exactly why "processed metadata outside catalog = 0" says nothing
about how much source metadata was skipped.

### Measured result (Sports & Outdoors, normalization version 1)

```
source records seen         : 1,587,421
parse errors                : 0
non-object records          : 0
missing parent_asin         : 0
duplicate keys              : 0
source records outside catalog : 1,430,675
normalized records written  : 156,746
artifact bytes              : 307,500,147
artifact SHA-256            : 175c83caa3523704319f5a5adf1531f0d839d26ba0604be646c9dfdfb21e71dd
```

Rebuilding from the same raw input twice produces **byte-identical** artifacts
(verified with `cmp`), and an independent streaming reclassification of the raw source
against the accepted mapping reproduces these counts exactly.

## 6. Catalog coverage audit

Joined against the accepted SASRec mapping (`item2id`, 156,746 items):

| Metric | Value |
| --- | --- |
| catalog items | 156,746 |
| metadata records (in scope) | 156,746 |
| catalog items with metadata | 156,746 |
| catalog items missing metadata | 0 |
| coverage | **100.0000 %** |
| metadata records not in catalog | 0 |

Field population across the covered catalog:

| Field | Records |
| --- | --- |
| `average_rating`, `rating_number` | 156,746 |
| `title` | 156,737 |
| `details` | 155,398 |
| `store` | 155,343 |
| `categories` | 152,893 |
| `features` | 144,766 |
| `main_category` | 141,850 |
| `price_text` | 94,490 |
| `description` | 91,196 |

> Coverage is a **descriptive data statistic** about metadata availability. It is not
> a recommendation-quality metric, and no recommendation claim may be based on it.

Missing metadata is explicitly allowed by design: a candidate without metadata remains
a valid candidate with `metadata_status="missing"`. It is never dropped, replaced, or
filled with a fabricated description.

## 7. Lookup contract

```python
lookup(parent_asin) -> ProductMetadata | MissingMetadata
lookup_many(parent_asins) -> tuple[ProductMetadata | MissingMetadata, ...]
parent_asin in index -> bool
```

* results are **positionally aligned** with the input;
* duplicates in the input stay duplicated in the output;
* a non-string identifier raises `TypeError` (a programming error, not a miss);
* the input sequence is never mutated;
* unknown identifiers return `MissingMetadata`, never `None` and never an exception;
* lookup is a dictionary probe — **0.27 µs** measured — and the artifact is parsed once
  per `MetadataIndex` instance, never per candidate or per graph invocation.

`MetadataIndex.from_records(...)` builds an index from in-memory records for tests, so
normal pytest needs no real artifact.

## 8. What this layer does not do

* no candidate generation, scoring, ranking, reranking or filtering of recommendable
  items — enforced by a source guard test;
* no knowledge of SASRec, `item_id`, tensors, checkpoints or inference;
* no inference of missing attributes;
* no network access at any point after raw acquisition;
* no mutation of the accepted mapping artifacts.

## 9. Usage

```bash
# build the artifact (raw file must already be downloaded)
.venv/bin/python -m experiments.prepare_product_metadata
.venv/bin/python -m experiments.prepare_product_metadata --all-records      # every source record
.venv/bin/python -m experiments.prepare_product_metadata --json /tmp/m8a.json
```

```python
from recommendation.catalog import MetadataIndex, MissingMetadata, ProductMetadata

index = MetadataIndex.load("data/processed/Sports_and_Outdoors_products.jsonl")  # once
record = index.lookup("B0BX5QFWQN")
if isinstance(record, MissingMetadata):
    ...  # explicit absence
else:
    print(record.title, record.store, record.features)
```

## 10. Tests

```bash
.venv/bin/python -m pytest -q tests/test_product_metadata.py
```

47 tests covering known/missing lookup, alignment, duplicate and order preservation,
input non-mutation, deterministic and idempotent normalization, the duplicate and
malformed policies, coverage arithmetic, opaque identifiers, absence of item-id
remapping, absence of candidate-generation logic, artifact round-trip,
byte-reproducibility and overwrite protection. Fully offline with synthetic fixtures.

## 11. Boundary

M8-A covers the metadata layer only. Candidate-scoped retrieval lives in
[`../rag/`](../rag/README.md): **M8-A never retrieves, and M8-B never redefines the
metadata contract.** Product metadata and semantic enrichment beyond what this
artifact contains remain out of scope.
