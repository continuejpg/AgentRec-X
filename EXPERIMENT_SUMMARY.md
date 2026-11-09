# AgentRec-X — Experiment Summary (P18)

> This document separates what was **confirmed**, what **failed**, what is **not established**, and what is only a **local demo**. It is not a claim of real-user recommendation accuracy.

## 1. Confirmed engineering successes

| Area | Evidence |
|---|---|
| Real Electronics catalogue + identity binding | 76,263 accepted products, BGE index identities exactly equal to catalogue; P14-D2 |
| BGE dense retrieval improves target reach | Original-query Recall@20 **10 → 27/135**, Recall@100 **20 → 66/135** on Dev (P14-D2) |
| P15 Evidence Bundle | Real raw field/span/source, UNKNOWN preservation, automated claim validation; 9 finalized E2E tasks |
| P16 reusable Skills | Comparison and Clarification actually invoked; memory update/withdrawal/idempotency/cross-user tests |
| P17 MCP | Official `mcp==2.3.0`, protocol 2026-07-28, stdio; independent client discovers 7 tools |
| P18 CPU MCP demos | Search/recommend, 2-product comparison, clarification + M9 update; complete real tool traces |
| Gate/UNKNOWN/Memory isolation | ASK/no-verified tasks not forced to recommend; writes disabled by default; cross-user handles rejected |

## 2. Algorithm improvements confirmed

- **BGE retrieval** is a real, measured improvement in candidate reach versus BM25 on the frozen Dev protocol (P14-D2).
- The MCP/Agent engineering stack is functional and reproducible under CPU MCP mode.
- **No confirmed improvement** from real long-history personalized reranking or Planner SFT: both were tested and rejected as default choices in earlier phases.

## 3. Experiments that did not succeed

| Experiment | Result |
|---|---|
| Real-history reranking after metadata recovery | Real history did not beat Query Only or Shuffled controls (`HISTORY_ABLATION FAIL`, P14-D1-R) |
| Qwen Planner SFT | Adapter trained (134 samples) but Dev20 Top-1 **12/15 vs Base 15/15**; not adopted (P14-C1) |
| Hybrid RRF BM25+BGE | **22/135 Recall@20 / 47/100** vs BGE 27/66; no increment (P14-D2) |
| Evidence semantic grounding as human truth | Not established; automated structural validation only (P15/P16) |
| Qwen as MCP planner | Not implemented; P17 MCP is CPU Rule Planner, Qwen Agent Mode is separate/replay |

## 4. Not established

- Real-user recommendation accuracy or satisfaction;
- Semantic truth of natural-language explanations (no human review; `semantic_truth_verified=false`);
- Independent long-term personalization gain from real history;
- Production authentication/authorization, public deployment security, remote MCP HTTP;
- Generalization beyond the simulated P14-E2 benchmark labels.

## 5. CPU MCP vs Qwen Agent Mode

- **CPU MCP Mode:** local stdio, BGE/BM25, Rule Planner + Gate, P15/P16, no GPU/Qwen;
- **Qwen Agent Mode:** existing Qwen3.5-9B Base `/plan` and P15 `/explain`; requires external GPU/model service; P18 uses **Replay** of saved real trajectories, clearly marked `live_call=false`.

## 6. P18 final evidence

- Three CPU MCP demos with real tool trajectories;
- Cold startup ~34–40 s; BGE first search ~18.2 s (CPU encoder load included); hot BGE search ~0.15 s;
- One-time Reserved Test: 20/20 tasks completed; 12 recommend tasks, 11 Top-1 correct; 8 non-recommend, 4 correct clarifications, 4 correct unknown terminations; false ASK/refusal/hard violations 0; errors 0. **Simulated labels only; workflow generalization check, not real-user accuracy.**
- 20 human-review materials produced and left **pending_human_review**; no reviewer completed them.

## 7. Final project position

AgentRec-X is a **local, reproducible engineering demo** with honest boundaries. It is not a production public recommendation service, and its explanation quality is not yet human-validated.
