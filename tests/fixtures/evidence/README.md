# Phase 4 evidence fixtures

A deliberately separate, reproducible local artifact that a trusted `EvidenceSource` reads.

**This is not the catalogue projection.** A fact in this file is unknown to a run until an
`ACQUIRE_EVIDENCE` action reads it. That separation is the point: it is what makes the
`UNKNOWN -> acquire new trusted evidence -> SATISFIED / VIOLATED` trajectory prove the Phase-4
capability rather than re-reading facts the run already had.

Format: one JSON object per line.

```json
{"parent_asin": "cand-red", "attributes": {"material": "full-grain leather"}, "source_note": "vendor spec sheet"}
```

* `parent_asin` — the product the facts are about. One line per product; a duplicate is an error
  rather than a silent precedence decision.
* `attributes` — attribute name to stated value. Keys must be from the evidence vocabulary; an
  unknown key is an error, because a silently ignored fact would look like absence.
* `source_note` — the provenance note recorded on every item read from this line.

Synthetic fixture data for deterministic offline tests and evaluation. It is **not** real vendor
documentation and makes no claim about any real product.
