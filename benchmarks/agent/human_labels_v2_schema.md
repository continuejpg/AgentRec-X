# Human Relevance Label Schema v2 (provisional, P1C)

This schema is for the **independent human relevance annotation** of A/B/C/D
recommendation lists.  It is not produced from the runtime `product_role.py`, M5/M6.1
outputs, or next-item labels.  The evaluator V2 is a deterministic rubric; it does not
replace this human dataset.

## Unit of annotation

One row per `(task_id, policy, candidate_parent_asin)` when needed, or one list-level
row per `(task_id, policy)` with the ordered candidate IDs.  Candidate lists must be
exported before annotation, without the runtime role status.

## Required columns

| column | values | meaning |
|---|---|---|
| `task_id` | string | benchmark task identifier |
| `policy` | A/B/C/D | recommendation arm that produced the list |
| `candidate_parent_asin` | string | candidate identity, or first ID for list row |
| `candidate_order` | integer | 1-based presentation order |
| `relevance_overall` | 2/1/0/UNKNOWN | 2 clearly satisfies current goal/requirements; 1 partially relevant or evidence insufficient; 0 clearly irrelevant or violating |
| `role_satisfaction` | 2/1/0/UNKNOWN | independent judgment of target type + main/accessory role; 2 satisfied, 1 partial/ambiguous, 0 wrong role |
| `hard_constraint_satisfaction` | 2/1/0/UNKNOWN | independently verifiable hard constraints only; UNKNOWN when catalogue evidence is insufficient |
| `evidence_refs` | string | catalogue fields/URL/notes used; must not cite runtime role output |
| `reviewer` | string | real reviewer identifier |
| `reviewed_at` | RFC3339 | annotation timestamp |
| `label_status` | draft/final | draft rows are never treated as ground truth |
| `notes` | string | ambiguity, candidate scope, or exception rationale |

## Rules

1. `2` is not a global quality score; it is conditional on the task's current goal.
2. Role satisfaction and exact identity are separate: a role-correct product may still
   be `relevance_overall=2` for a broad task even when it is not the single expected ASIN.
3. Forbidden title terms must not be applied mechanically: a complete water filter whose
   title mentions included bottles is not a bottle product.  Use category + role evidence.
4. Missing weight/price/material evidence is `UNKNOWN`, never `0` or `2` by default.
5. Annotators must not see the system's role/MATCH output before assigning role labels.
6. Frozen Gold is not an annotation instruction; disagreements are recorded in notes.
