# Electronics Metadata Enrichment MVP

Status: **local MVP, additive metadata; NOT a new catalogue or search backend.**

## What it is

An optional sidecar built from Amazon Reviews 2023 `raw_meta_Electronics` and keyed by the
same `parent_asin` as the current C4 Electronics catalogue.  It stores title, features,
description, categories, store, parsed `details`, historical crawl-time price, historical
average rating and rating count for the **76,263 accepted C4 Electronics products**.

The sidecar is not part of the BGE index.  BM25/BGE still search the original C4 text, so
the frozen retrieval identity and index are unchanged.

## Build

```bash
# metadata shards (not committed) live under a temporary directory
python scripts/build_electronics_metadata_sidecar.py \
  --metadata-dir "$TEMP/amazon_reviews_2023_raw_meta_Electronics" \
  --out-dir data/enrichment
```

Outputs:

- `data/enrichment/electronics_metadata_sidecar.jsonl`
- `data/enrichment/manifest.json`

Both are data artifacts and are git-ignored.

## Use

Environment switch:

```bash
AGENTRECX_METADATA_MODE=original   # default, C4 only
AGENTRECX_METADATA_MODE=enriched   # attach matching sidecar records
AGENTRECX_METADATA_SIDECAR=/path/to/electronics_metadata_sidecar.jsonl
```

The MCP server also exposes explicit CLI flags:

```bash
python -m recommendation.mcp.server \
  --user-key p14e2_lite_0049 \
  --memory-fixture data/p14e2/benchmark/dev.agent.jsonl \
  --search-mode bge \
  --metadata-mode enriched
```

## Semantics

- Missing sidecar record: **fall back to the original C4 product**; no fabricated metadata.
- Sidecar fields are quoted as upstream metadata, not real-world truth.
- `price` is historical crawl-time price; it remains **UNKNOWN for live price/stock**.
- `average_rating` / `rating_number` are historical upstream aggregates.
- Structured connectivity/form-factor facts are added only when they do not contradict a
  C4 attribute.  Mutually exclusive form-factor disagreements clear the derived form factor
  and are recorded under `metadata_conflicts`; they never pass a hard constraint.
- P15 evidence uses distinct `metadata_*` raw fields and keeps exact spans.  The original
  C4 evidence remains available.
- Field coverage is not a claim of factual correctness.

## Smoke / before-after

`experiments/p20_metadata/run_cpu_e2e.py` starts the existing MCP server in Original and
Enriched modes and records the same tasks under `runs/p20_metadata/`.

`experiments/p20_metadata/run_qwen_live.py` runs the real Qwen path through
`run_agent_turn`.  The verified run used:

- AutoDL RTX 4090 24 GB, existing `Qwen3.5-9B-ms` `/plan` + `/explain` server on port 8009;
- SSH tunnel `127.0.0.1:18029 -> remote:8009`;
- MCP server in the local WSL `.venv`, `AGENTRECX_METADATA_MODE=enriched`;
- `--metadata-mode enriched --search-mode bge`.

Result files:

- `runs/p20_metadata/qwen_live_original.json`
- `runs/p20_metadata/qwen_live_enriched.json`
- `runs/p20_metadata/summary.json`

The live run selected `B08D32JM5F` in both modes.  Enriched mode produced 12 product
evidence items including `metadata_details`, and the final answer cited `pev_05`
(`metadata_details` form-factor `In Ear`) and treated brand as no longer UNKNOWN; Original
mode had 6 evidence items, no metadata fields, and still reported brand as UNKNOWN.

The CPU-only Qwen experiment was a fallback during setup; it is not the acceptance path
because Qwen3.5-9B Q4 on CPU was slow and needed 8192 context to fit the explain prompt.
