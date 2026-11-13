# Dev Evaluation Checklist (Planner / Agent Baseline)

Purpose: a reusable 25-task initial Dev checklist assembled from existing tasks. It is not a frozen test and must not be used to tune on P18 Reserved.

## Sources and label trust

| Source | Tasks | Label status | Trust |
|---|---:|---|---|
| P14-E2 Lite dev | 20 | `template_generation_v1`, `review_status=automated_checked` | provisional; no human verification |
| P14-B3 Eval100 (sample) | 5 | `review_status=pending_human_review`, `human_verified=false` | provisional; requires human review |

P18 Reserved was **not** included and must not be used for tuning.

## Metrics

- Recommendation success: selected item is in the evaluator acceptable set (independent label).
- Correct ASK: expected clarification and terminal `awaiting_clarification`.
- Correct abstain: expected `no_verified_match`/`insufficient_evidence` and terminal `no_verified_candidate_in_scope`.
- Hard constraint violation: selected item violates a stated hard constraint.
- Efficiency: budget exhausted, redundant RETRIEVE, tool calls, wall time.
- Evidence quality: P15 claim validation status; this is citation-level validation, not semantic truth.

## 25-task list

| # | Task | Split | Type | Expected | Coverage | Human review |
|---:|---|---|---|---|---|---|
| 1 | `p14e2_lite_0041` | dev | no_history_recommend | recommend | recommend/acceptable-set, hard constraint | medium |
| 2 | `p14e2_lite_0042` | dev | no_history_recommend | recommend | recommend/acceptable-set, hard constraint | medium |
| 3 | `p14e2_lite_0043` | dev | no_history_recommend | recommend | recommend/acceptable-set, hard constraint | medium |
| 4 | `p14e2_lite_0044` | dev | no_history_recommend | recommend | recommend/acceptable-set, hard constraint | medium |
| 5 | `p14e2_lite_0045` | dev | history_influenced | recommend | recommend/acceptable-set, long-term preference | medium |
| 6 | `p14e2_lite_0046` | dev | history_influenced | recommend | recommend/acceptable-set, long-term preference | medium |
| 7 | `p14e2_lite_0047` | dev | history_influenced | recommend | recommend/acceptable-set, long-term preference | medium |
| 8 | `p14e2_lite_0048` | dev | history_influenced | recommend | recommend/acceptable-set, long-term preference | medium |
| 9 | `p14e2_lite_0049` | dev | history_influenced | recommend | recommend/acceptable-set, long-term preference | medium |
| 10 | `p14e2_lite_0050` | dev | same_query_persona_pair | recommend | recommend/acceptable-set, long-term preference | medium |
| 11 | `p14e2_lite_0051` | dev | same_query_persona_pair | recommend | recommend/acceptable-set, long-term preference | medium |
| 12 | `p14e2_lite_0052` | dev | same_query_persona_pair | recommend | recommend/acceptable-set, long-term preference | medium |
| 13 | `p14e2_lite_0053` | dev | same_query_persona_pair | recommend | recommend/acceptable-set, long-term preference | medium |
| 14 | `p14e2_lite_0054` | dev | preference_update | recommend | recommend/acceptable-set, memory update/withdrawal | medium |
| 15 | `p14e2_lite_0055` | dev | preference_withdrawal | recommend | recommend/acceptable-set, memory update/withdrawal, hard constraint | medium |
| 16 | `p14e2_lite_0056` | dev | preference_conflict | clarification_required | correct ASK, memory conflict | high |
| 17 | `p14e2_lite_0057` | dev | insufficient_information | clarification_required | correct ASK, ambiguity / ASK | high |
| 18 | `p14e2_lite_0058` | dev | insufficient_information | clarification_required | correct ASK, ambiguity / ASK | high |
| 19 | `p14e2_lite_0059` | dev | insufficient_information | clarification_required | correct ASK, ambiguity / ASK | high |
| 20 | `p14e2_lite_0060` | dev | no_verified_match | no_verified_match | bounded abstain, hard constraint | high |
| 21 | `p14b3_eval100_0001` | eval100 | no_history_recommend | recommend | recommend/acceptable-set, hard constraint | high |
| 22 | `p14b3_eval100_0065` | eval100 | preference_update | recommend | recommend/acceptable-set, memory update/withdrawal | high |
| 23 | `p14b3_eval100_0089` | eval100 | preference_conflict | clarification_required | correct ASK, memory conflict | high |
| 24 | `p14b3_eval100_0093` | eval100 | no_verified_match | no_verified_match | bounded abstain, hard constraint | high |
| 25 | `p14b3_eval100_0109` | eval100 | unverifiable_constraint | insufficient_evidence | bounded abstain, unverifiable constraint / insufficient evidence, hard constraint | high |

## Highest-priority human review

- `p14e2_lite_0056` — memory preference conflict (wired vs wireless); expected clarification.
- `p14e2_lite_0057` — insufficient information / ASK; check the query is genuinely non-specific.
- `p14e2_lite_0060` — composite hard constraints, no verified candidate; check the label and the composite constraint semantics.
- `p14b3_eval100_0089` — second corpus conflict case; pending human review.
- `p14b3_eval100_0093` — second-corpus no-verified case; pending human review.
- `p14b3_eval100_0109` — unverifiable price constraint / insufficient evidence; pending human review.
- All P14-B3 Eval100 sample tasks need independent human review before their labels can be called gold.

## Not yet sufficient for Corrective RAG

Before Corrective RAG, we still need:

1. Human-reviewed expected decisions and hard constraints for the 25-task Dev set (or a documented subset).
2. Independent gold evidence/relevance labels per task, not only `expected_decision`.
3. Retrieval-failure labels: which query/evidence gap would a corrective step fix, and what is the correct recovery?
4. A held-out baseline score for the same policy/model with confidence intervals.
5. A semantic answer-quality judge or human rubric; P15 claim validation is not a truth judge.
6. A frozen evaluation manifest/version and no reuse of P18 Reserved.
