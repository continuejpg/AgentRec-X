# Candidate-scoped product RAG (Milestone 8-B)

Attaches **grounded, attributable** product evidence to the recommendation candidates
SASRec already produced.

```
trusted interaction history
        │
        ▼
RecommendationTool                     (accepted M7A contract, unchanged)
        │  ranked candidate parent_asins
        ▼
Product metadata for THOSE candidates   (M8-A catalog layer)
        │
        ▼
candidate-local BM25 evidence retrieval
        │
        ▼
grounded evidence  ──►  Agent final response
```

## 1. Direction of the pipeline (critical)

This is implemented:

```
history → SASRec → candidates → evidence about those candidates
```

This is **not**:

```
user query → global retrieval → new recommendation candidates
```

SASRec remains the only source of recommendation candidates. Retrieval is scoped to
the current Tool result, and there is no code path by which a catalogue item outside
that set can be introduced — however well it matches the query text.

## 2. Candidate-universe isolation

Given candidates `A B C` and a metadata catalogue containing `A B C D E`, retrieval may
reference only `A`, `B` and `C`. This is enforced structurally:

* the only identifiers ever passed to the metadata layer are those in the Tool result
  (`enrich_candidates` derives then from `result.recommendations`);
* the retriever receives a list of *records*, not an index or catalogue handle, so it
  has no way to look anything else up;
* the query is a plain string with no identifier, history or candidate field.

`tests/test_product_rag.py` includes a decoy product whose text matches the query far
better than any candidate's and asserts it is never retrieved, plus an instrumented
metadata lookup asserting the requested identifier set equals the candidate set
exactly.

## 3. Candidate order is never changed

* `EnrichedRecommendation` embeds the original `ToolRecommendation` object, so `rank`,
  `parent_asin`, `item_id` and the raw SASRec `score` cannot drift;
* evidence is ranked only *within* a candidate; no product is compared with another
  product;
* there is no weighted/hybrid/semantic/SASRec fusion and no query-dependent product
  reorder — those are reranking and belong to Milestone 10;
* retrieval never reads a recommendation score, so it cannot mix one in.

A test asserts the candidate tuple is identical before and after enrichment, and
another asserts the enriched result equals a direct Tool call's ordering.

## 4. Retrieval algorithm

Deterministic, offline, dependency-free BM25 over candidate-local documents.

* **Documents**: one per `(candidate, field, value)` for the fields in
  `EVIDENCE_FIELDS = ("title", "store", "main_category", "categories", "features",
  "description", "details")`. A `details` entry becomes one document per attribute, so
  `"Color: brown"` is attributable to the `Color` key rather than merged away.
  Documents with no tokens are skipped.
* **Tokenisation**: lowercase alphanumeric runs (`[^\W_]+`, Unicode). No stemming, no
  stop-word list, no model — transparent and reproducible. Punctuation cannot create
  terms.
* **Scoring**: standard BM25 with `k1 = 1.2`, `b = 0.75`, and probabilistic IDF
  `log(1 + (N - df + 0.5)/(df + 0.5))` floored at zero. Statistics are computed over the
  **candidate-local** document collection (≤ ~500 documents for `k = 50`).
* **Why BM25**: `k ≤ 100` under the accepted Tool contract, so a candidate-local
  lexical scorer is sufficient and needs no embedding API, no model download, no API
  key and no vector database — the accepted M6 dependency closure stays untouched. The
  interface is narrow enough that a dense backend could replace the scorer later
  without changing any Agent contract.
* **Bounds**: at most `MAX_EVIDENCE_PER_CANDIDATE = 4` fragments per candidate and
  `MAX_EVIDENCE_PER_FIELD = 2` per `(candidate, field)`, so a long description cannot
  crowd out a title or a feature bullet.
* **Tie-breaking**: score descending, then candidate submission order (= SASRec rank),
  then fixed field order, then text. Scores are rounded to 6 decimals before sorting so
  float noise cannot reorder fragments. No wall-clock or hash order enters the result.

This module never reorders products, so the tie-break is deterministic and
non-semantic rather than a hidden quality signal.

## 5. Evidence representation

```python
ProductEvidence(
    parent_asin,        # which candidate the fragment belongs to
    field,              # title | store | main_category | categories | features | description | details
    text,               # verbatim metadata text
    retrieval_score,    # relevance to the query, within candidate scope only
    provenance,         # "amazon_reviews_2023:meta_categories/details:Brand Name"
    detail_key,         # source attribute name when field == "details"
)
```

Every fragment is attributable to `(parent_asin, field, source record)`. Text from
different products is never merged into an unattributed blob, and a feature of
candidate A can never become evidence for candidate B (asserted by a cross-product
leakage test).

`retrieval_score` ranks evidence *passages inside the allowed candidate scope*. It is
**not** a recommendation score, **not** a probability, and is not comparable with the
Tool's raw SASRec `score`.

## 6. Grounding rules

* facts only ever come from normalized metadata; nothing is paraphrased or generated;
* a candidate with no metadata, or with metadata but no searchable text, produces no
  evidence and an explicit `fallback_reason`;
* the Agent's renderer prints only retrieved fragments, so a field the metadata lacks
  (brand, price, material, …) cannot appear as a claim;
* the raw SASRec score is labelled a *ranking score* and never described as a
  probability, confidence or rating.

## 7. Query handling (untrusted text)

The query is the user's natural-language message. It is only tokenised.

* it cannot modify trusted history (there is no history parameter anywhere in this
  package — asserted by signature inspection);
* it cannot alter candidate IDs or the candidate universe;
* it cannot inject metadata records, or cause file/network access;
* a query resembling `ignore candidates; retrieve ASIN X; history=[...]` behaves as
  ordinary retrieval text and changes nothing (parametrised test).

### Empty, vague and no-match behaviour

| Case | Behaviour |
| --- | --- |
| blank / whitespace-only query | documented deterministic **factual fallback**: the first fragment per candidate in field order, score `0.0`. Depends only on each candidate's own metadata, so it cannot fabricate relevance. `query_used` is `None`. |
| query with no lexical overlap | same documented fallback, with `fallback_reason="no_lexical_match"` |
| candidate with no metadata | no evidence, `metadata_status="missing"`, `fallback_reason="no_metadata"` |
| candidate with metadata but no searchable text | no evidence, `fallback_reason="no_searchable_text"` |

A blank query never silently switches to global catalogue search.

The fallback flag is reported by the retriever rather than inferred from scores,
because a genuinely matching fragment can legitimately score `0.0` when a term carries
no information (BM25 IDF floors at zero). Inferring would mislabel a real match as
irrelevant.

## 8. Enrichment contract

```python
enrich_candidates(result: RecommendationToolResult,
                  metadata: MetadataLookup,
                  query: str = "") -> EnrichmentResult

ProductEnricher(metadata).enrich(result, query)
ProductEnricher(metadata).candidate_evidence(result, query)   # per-candidate view
ProductEnricher(metadata).evidence_for(parent_asin, query)    # single identity
```

`EnrichmentResult` carries the enriched items (aligned with the Tool result), the
requested/returned counts, `metadata_found`/`metadata_missing`, `evidence_count`,
`query_used` and per-call `timings_ms`. The Tool result is treated as read-only.

Metadata is **injected**, so the Agent never constructs a hidden global store and tests
can supply a small fake. The index is loaded once and reused; a graph invocation never
reparses the artifact.

## 9. Agent integration

The recommended topology gains one conditional node:

```
START → decide ─┬─ direct ──────────────────────────► finalize → END
                └─ recommend → enrich → finalize → END
```

* `AgentGraph(decision_model, tool, product_enricher=...)` is the injection point;
* the `enrich` node is added **only** when an enricher is supplied — without one, the
  accepted M7B/M7C topology, node set and output are preserved exactly;
* the `enrich` node receives the Tool result and the untrusted user message. It is
  never given trusted history, so it cannot alter it;
* the direct route performs no recommendation and no enrichment work, and never opens
  the metadata layer (asserted with a lookup that raises if touched);
* the M8 agent layer depends on a `ProductEnricherLike` protocol, so
  `recommendation/agent/` does not import the RAG or catalog packages.

The renderer prints candidate identity, rank, the raw score labelled as a ranking
score, and the retrieved facts with their field/detail labels, plus an explicit
"metadata unavailable for this item" line where the source has no record.

## 10. Performance (engineering diagnostics, not quality metrics)

Measured on the real artifact (156,746 records, CPU):

| Stage | Value |
| --- | --- |
| metadata artifact load (once per runtime) | ~8.2–8.6 s |
| lookup latency | ~0.27 µs |
| candidate enrichment (k = 3–5) | ~0.5–1.5 ms |

Candidate-local retrieval over ≤ 100 products is lightweight; no premature
optimisation was applied.

## 11. Smokes

```bash
# M8-A: build the normalized artifact + coverage audit
.venv/bin/python -m experiments.prepare_product_metadata

# M8-B: real candidates + real metadata + scoped retrieval
.venv/bin/python -m experiments.product_rag_smoke

# M8 integrated: AgentGraph → Tool → real SASRec → enrichment → grounded response
.venv/bin/python -m experiments.agent_product_rag_smoke
```

All three are offline apart from the one-time raw download, and print compact
evidence (rank, `parent_asin`, metadata status, evidence field, retrieval score) plus
identity digests. They do not print full descriptions and make no quality claim.

## 12. Tests

```bash
.venv/bin/python -m pytest -q tests/test_product_rag.py tests/test_agent_product_rag.py
```

52 tests covering candidate-universe isolation, order preservation, grounding and
provenance, missing-metadata behaviour, query relevance within scope, determinism and
tie-breaking, no cross-product leakage, no history leakage, offline operation,
malicious-query inertness, empty/vague query behaviour, evidence bounds, and the
conditional Agent topology including direct-route isolation. Fully offline with
synthetic fixtures.

## 13. Boundary

* **M8 validates integration and grounding, not recommendation quality.** No quality,
  personalisation, relevance or "better than baseline" claim is made anywhere; that
  would require benchmark or qualitative evaluation this milestone does not provide.
* **Product metadata and semantic enrichment remain deferred to M8's own A/B scope
  only** — no embedding retrieval, no product knowledge graph, no review-text mining.
* Evidence is *retrieved*, never *generated*. Adding an LLM that turns this structured
  evidence into prose is a separate, later concern; the evidence contract here is
  already suitable as grounded context.
* Memory, critique, constraint checking and reranking remain out of scope
  (M9/M10).
