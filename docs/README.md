# AgentRec-X — Documentation Index

> **Status: CURRENT index.** Authority order: worktree code → the **Current** documents
> below → historical/frozen records. A historical document describes the state at the time
> it was frozen; it never overrides current code.

## Status legend

| Marker | Meaning |
| --- | --- |
| **Current** | Maintained against the worktree; safe to use for onboarding and operation |
| **Current evidence** | Frozen evaluation/acceptance record, still the authoritative evidence for its stated protocol |
| **Experimental** | Implemented and tested, but not part of the default serving profile or not quality-claimed |
| **Historical** | Retained for chronology, reasoning and reproducibility; superseded for current architecture |
| **Audit** | Read-only snapshot of a past audit; not a maintained contract |

## Current

| Document | Read it when you want to… |
| --- | --- |
| [../README.md](../README.md) | project overview, quick start, model/Agent boundaries, evidence summary |
| [ARCHITECTURE.md](ARCHITECTURE.md) | **the current architecture**: Serving V1, `/v1/recommend`, the shopping Agent, adaptive preferences, evaluation boundaries, directory map |
| [ONBOARDING.md](ONBOARDING.md) | current coordinates, reading order, hard rules and a short self-check |
| [USAGE.md](USAGE.md) | install, run the demo, call the API, run tests/smokes, troubleshoot |
| [EXPERIMENTS.md](EXPERIMENTS.md) | the formal evaluation protocol, accepted metrics and threats to validity |

## Current evidence — frozen acceptance and evaluation records

| Document | Read it when you want to… |
| --- | --- |
| [V2_FINAL_ACCEPTANCE.md](V2_FINAL_ACCEPTANCE.md) | the frozen engineering/quality/completeness/delivery verdicts |
| [V2_KNOWN_LIMITATIONS.md](V2_KNOWN_LIMITATIONS.md) | what is explicitly **not** proven |
| [V2_REPRODUCE.md](V2_REPRODUCE.md) | reproduce the environment, artifact checks and regression commands |
| [B1_PREREGISTRATION.md](B1_PREREGISTRATION.md) | the frozen paired-evidence protocol for the Two-Tower/fusion study |
| [M3_PREREGISTRATION.md](M3_PREREGISTRATION.md) / [M4_RESULT.md](M4_RESULT.md) | the bounded LLM policy experiment and its formal result |
| [V2_ROADMAP_COVERAGE.md](V2_ROADMAP_COVERAGE.md) | roadmap item status (PASS / PARTIAL / UNKNOWN) |

## Experimental / optional

| Document | Read it when you want to… |
| --- | --- |
| [LEARNED_RANKER_DEPLOYMENT.md](LEARNED_RANKER_DEPLOYMENT.md) | operate `serving_v1_learned_ranker`: artifacts, fallback, agent boundary; not a default or quality claim |
| [TIGER_BACKEND.md](TIGER_BACKEND.md) | the optional external TIGER backend boundary and PAD/provenance rules |
| [SEMANTIC_ID.md](SEMANTIC_ID.md) | the frozen historical GenRec v0 / Semantic-ID baseline |
| [TWOTOWER.md](TWOTOWER.md) | the Two-Tower study, complementarity and fusion controls |
| [MODEL_EXPANSION_HANDOFF.md](MODEL_EXPANSION_HANDOFF.md) | the model-expansion handoff and its evidence boundaries |

## Historical

| Document | Read it when you want to… |
| --- | --- |
| [HISTORICAL_ARCHITECTURE.md](HISTORICAL_ARCHITECTURE.md) | the detailed architecture as it was frozen before the current shopping Agent; **superseded** |
| [ARCHITECTURE_TEXT_DIAGRAM.md](ARCHITECTURE_TEXT_DIAGRAM.md) | old text-only diagrams; **superseded** |
| [PROJECT_HISTORY.md](PROJECT_HISTORY.md) | milestone sequence and commit anchors |
| [SESSION_MILESTONES.md](SESSION_MILESTONES.md) | the regenerable request/milestone index; historical source material, not a current contract |
| [PHASE4_HANDOFF.md](PHASE4_HANDOFF.md) | M11-era state and invariants |
| [PHASE5_HANDOFF.md](PHASE5_HANDOFF.md) | public-data benchmark and leakage checks |
| [PROJECT_STATE.md](PROJECT_STATE.md) | a frozen 2026-09-27 project-state snapshot |
| [../HANDOFF_AGENTRECX.md](../HANDOFF_AGENTRECX.md) | an explicitly historical handoff; not a current serving description |

## Audit snapshots

| Document | Read it when you want to… |
| --- | --- |
| [MEMORY_AGENT_CAPABILITY_AUDIT.md](MEMORY_AGENT_CAPABILITY_AUDIT.md) | the L1–L4 Memory/Agent capability matrix |
| [MEMORY_AGENT_CALL_CHAINS.md](MEMORY_AGENT_CALL_CHAINS.md) | three traced call chains and where one stopped |
| [MEMORY_AGENT_RESUME_CLAIMS.md](MEMORY_AGENT_RESUME_CLAIMS.md) | supported vs unsupported resume claims |

## Where things live

* **Architecture and request paths** — [ARCHITECTURE.md](ARCHITECTURE.md).
* **Per-package contracts** — one README per package under `recommendation/`, e.g.
  [`agent/`](../recommendation/agent/README.md),
  [`demo/`](../recommendation/demo/README.md),
  [`inference/`](../recommendation/inference/README.md),
  [`memory/`](../recommendation/memory/README.md),
  [`evaluation/`](../recommendation/evaluation/README.md),
  [`api/`](../recommendation/api/README.md).
* **Formal evaluation** — [`EXPERIMENTS.md`](EXPERIMENTS.md), `recommendation/evaluation/`,
  `experiments/benchmark_public.py`, `experiments/benchmark_twotower.py`.
* **Constructed scenario / engineering validation** — `experiments/p3_scenario_eval.py`,
  `experiments/p4_multi_turn_eval.py`, `tests/`, startup/readiness scripts.
* **Frozen evidence** — selected small JSON/report files under `runs/` and `docs/reports/`.
  Large checkpoints, data, caches and generated `runs/` output are git-ignored.

## Documentation conventions

1. **Current vs historical.** Current documents are maintained against the worktree. A
   historical document states its status at the top and is never silently treated as current.
2. **Evidence layers stay separate.** Engineering correctness, offline recommendation
   quality, constructed-scenario behaviour and delivery/acceptance are different claims.
   Quality that is not established is written `NOT ESTABLISHED`.
3. **No unsupported claims.** Numbers cite a committed artifact or executed report. There is
   no preference-conditioned relevance or satisfaction claim, no online A/B claim, and no
   claim that the Learned Ranker beats RRF.
