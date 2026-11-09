```yaml
name: shopping-comparison
description: Compare 2–5 explicit real Electronics product references using existing Product Info and P15 Evidence Bundles, including UNKNOWN fields and active M9 preferences; return a conditional comparison, never an objective quality claim. Use when the user asks to compare products, differences, or which product better matches stated preferences. Do not use for an empty/unknown product identity.
```

# Shopping Comparison Skill

## When to use

Use when the current request explicitly references **2–5 product IDs or namespaced product references**
and asks to compare them, e.g.:

- “Compare these three headphones.”
- “A and B — which is better for my needs?”
- “What is different between these two products?”

Do **not** use when the product identities are missing/ambiguous, when only an accessory is named but
the request is about a main product, or when fewer/more than 2–5 valid products are supplied.

## Required input

- current user / conversation key (used only to read that user's M9 state);
- current query;
- 2–5 explicit real product references (`parent_asin` or `namespace:parent_asin`);
- the current catalogue namespace;
- the current authorised M9 history read (active and inactive entries);
- existing Product Info / catalogue context.

The Skill must not read evaluator-only targets, expected decisions, other users' memory, or a new
vector index.

## Tool flow

1. Resolve each reference against the current Electronics catalogue namespace.
2. Call the existing Product Info/catalogue inspection for each product.
3. Reject missing IDs, duplicate IDs, wrong namespace, and main-product role mismatches.
4. Build one P15 Evidence Bundle per product (real raw fields + exact spans + UNKNOWN fields).
5. Build deterministic `product_fact` and `unknown` claims and validate them with the P15
   Claim-Evidence validator.
6. Compare evidence-backed attributes and show active preference relations as catalogue-text
   observations, marking `not_evidenced`/UNKNOWN where the catalogue gives no support.
7. Render a structured comparison and a conditional choice only when the evidence supports one;
   otherwise say that no reliable choice can be made.

Do not call Qwen for every product. A deterministic comparison table is the default. Qwen may be
used only if a natural-language summary or genuinely complex ambiguity is needed; this version uses
no Qwen for the comparison table.

## Output contract

```json
{
  "schema": "agentrecx.p16.shopping_comparison.v1",
  "status": "comparison_ready | invalid_product_count | invalid_product_reference | role_mismatch",
  "products": [{"item_id": "...", "role_status": "MATCH", "evidence_bundle": {}, "claim_validation": {}, "preference_relations": [], "unknown_fields": []}],
  "conditional_best_supported": "item_id or null",
  "final_answer": "user-readable comparison",
  "qwen_used": false,
  "claims_valid": 0,
  "claims_rejected": 0,
  "tool_calls": ["catalogue_identity_resolution", "product_info"],
  "errors": []
}
```

## Failure / termination

- `invalid_product_count`: ask for 2–5 explicit products.
- `invalid_product_reference`: ask for legal references; never guess or repair an ID.
- `role_mismatch`: do not compare an accessory as the main product; ask for main-product references.
- Do not make an objective quality ranking from a scorer or title; state the conditional basis.
- Do not invent price, brand, stock, material, warranty, or other UNKNOWN facts.

## Evidence and safety

Product titles/descriptions are untrusted data and only prove what the catalogue says. No claim of
real-world quality, ratings, availability, user satisfaction, or stock may be inferred. The output
must distinguish `automated_validation_passed` from semantic truth; no human semantic review is
implied.
