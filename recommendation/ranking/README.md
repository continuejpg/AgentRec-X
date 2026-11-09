# `recommendation.ranking` — M6.1 belief-aware offline ranking foundation

**Offline lab only. Not imported by the serving path.** This package joins a frozen trusted
candidate snapshot, the M5 `for_ranker` projection and the accepted M10A/M10B explicit
preference evidence into typed features, ranks them with three preregistered policies
(A0/A1/A2), and evaluates them under the frozen paired protocol in
[`docs/M6_1_RANKING_PROTOCOL.md`](../../docs/M6_1_RANKING_PROTOCOL.md). It never changes
Serving V1, RRF, the accepted reranker, M9 memory or model checkpoints, and it makes no
recommendation-quality claim unless a documented label protocol exists.

## Module map

| Module | Responsibility |
| --- | --- |
| `schemas.py` | frozen contracts: candidate snapshot/records, eligibility/evidence states, feature schema, tasks, arm rankings, report models |
| `errors.py` | typed rejections (`CandidateSnapshotError`, `FeatureBuildError`, `RankingPolicyError`, `RankingProtocolError`) |
| `resolver.py` | trusted candidate feature resolver: identity/metadata join, snapshot construction and ownership/staleness/count/digest verification, eligibility projection |
| `features.py` | `BeliefAwareFeatureBuilder`: real M10A explicit evidence + grounded inferred soft evidence (ordinal, UNKNOWN-neutral) + catalogue coverage |
| `policies.py` | shared eligibility projection; A0 original order, A1 real M10B reranker, A2 explicit priority + ordinal inferred tie-break |
| `evaluation.py` | `EvaluationProtocol`, `FrozenPairedEvaluator`: contract metrics, optional labeled quality metrics, paired deltas, report digest |

## Invariants

1. **Same candidates for every arm.** A snapshot is frozen once; eligibility is applied
   once; no arm may add, drop or replace a candidate.
2. **Identity boundary.** `parent_asin` plus trusted `item_id`; PAD (0) is never a
   candidate; digests verify, never reconstruct.
3. **Ownership/staleness.** User/task/turn/count/ref/digest mismatches are rejected.
4. **UNKNOWN is neutral.** Missing catalogue data is never a violation or a positive match.
5. **Explicit priority is preserved.** A2's first two sort keys are exactly M10B's
   violations/matches keys; inferred evidence can only break remaining ties.
6. **No score mixing.** Raw SASRec logits, Two-Tower cosines, BM25 and RRF scores are
   provenance only; no policy reads them numerically.
7. **Ordinal strength only.** M5 belief strength is a 1–3 ordering rank, never a
   probability.
8. **No label leakage.** Task labels are consumed only by the evaluator; the feature
   builder cannot accept them.
9. **Default Serving V1 untouched.** This package is offline; no production import, flag or
   ranking change is introduced.

## Minimal usage

```python
from recommendation.ranking import FrozenRankingTask
from recommendation.ranking.resolver import TrustedCandidateFeatureResolver
from recommendation.ranking.features import BeliefAwareFeatureBuilder
from recommendation.ranking.policies import rank_all_arms
from recommendation.ranking.evaluation import FrozenPairedEvaluator, DEFAULT_PROTOCOL

from tests.m6_1_fixture import build_task  # synthetic fixture (offline tests only)

task = build_task()  # FrozenRankingTask with a trusted snapshot + for_ranker projection
feature_set = BeliefAwareFeatureBuilder().build(task=task)
a0, a1, a2 = rank_all_arms(feature_set, task.snapshot)
report = FrozenPairedEvaluator(protocol=DEFAULT_PROTOCOL).evaluate([task])
```

## Tests and smoke

```bash
.venv/bin/python -m pytest -q tests/test_m6_1_resolver.py tests/test_m6_1_features.py \
    tests/test_m6_1_policies.py tests/test_m6_1_evaluation.py tests/test_m6_1_contract.py
.venv/bin/python -m experiments.m6_1_ranking_smoke
.venv/bin/python -m experiments.m6_1_belief_aware_evaluation
```
