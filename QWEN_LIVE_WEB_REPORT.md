# QWEN_LIVE_WEB_REPORT — real GPU Qwen Base Live acceptance (2026-11-09)

> **Update 2026-11-10:** the Qwen ASK → browser reply → M9 update → Qwen resume branch has now been
> exercised with the real Base model on an isolated wired/wireless conflict profile. That follow-up
> acceptance is **LIMITED**, not GO: the prescribed single reply added an AVOID entry but did not
> retract the old wired preference under current M9 semantics, a second real browser retraction was
> required, and the final Gate-selected product had a wired catalogue title. See
> [`QWEN_ASK_RESUME_REPORT.md`](QWEN_ASK_RESUME_REPORT.md). The 2026-11-09 `GO` below remains a
> historical verdict for the earlier, narrower scope and its ASK caveat is superseded by this
> update.

> **Overall: `GO`** for the requested Qwen Base Live scope. This is now backed by **real
> Qwen3.5-9B-Base inference on the user's GPU instance**, the real P17 MCP `run_agent_turn` tool,
> the real `ElectronicsPlannerExecutor`, the real catalogue/BGE/Gate/P15 path, and two real
> browser Qwen Live turns. No HTTP double and no Replay trajectory was used for this verdict.
>
> Caveat: the real model did **not** choose `ASK` in the prompts tested, so the Qwen ASK →
> browser-authored reply → M9 update → Qwen resume branch was **not triggered** in this round.
> The existing CPU clarification/M9 continuation path remained green; no ASK was fabricated.
>
> The previous report (2026-11-09 earlier) correctly recorded `BLOCKED` when the instance had no
> GPU/model service. This file supersedes that status with the evidence below.

## 0. Status matrix

| Gate | Result | Evidence |
|---|---|---|
| `GPU_AVAILABLE` | **PASS** | RTX 4090, 24,564 MiB total, driver 570.124.04; CUDA true; peak observed 19,845 MiB under Qwen; model load 12.203 s. |
| `QWEN_BASE_INFERENCE` | **PASS** | Real `POST /plan` responses from `Qwen3.5-9B-Base` on `/root/autodl-tmp/models/Qwen3.5-9B-ms`, base only (`adapter=None`), valid PlannerAction(s), no parser errors. |
| `REAL_QWEN_MCP_EXECUTION` | **PASS** | Real MCP `run_agent_turn` (8 tools with Qwen configured) → real executor → catalogue/BGE/Gate/P15; 3/3 real recommendation queries finalized, plus a real non-FINALIZE Gate rejection. |
| `REAL_QWEN_WEB_E2E` | **PASS** | Real browser (system Chrome) Qwen Live tests **2/2 passed**; full Playwright suite **9/9 passed** (7 CPU/Replay + 2 Qwen). Real screenshots saved. |
| `MEMORY_CONTINUATION` | **PARTIAL / NOT_TRIGGERED** | CPU clarification → real M9 `added=2` → resume passed. Qwen did not emit ASK for the tested prompts, so no false Qwen-memory-continuation claim is made. |
| `CPU_RULE_REGRESSION` | **PASS** | Original seven tests in the full 9-test suite all passed; P17/P16/P15 targeted: **20 passed**; P17 default 7 tools unchanged. |
| `REPLAY_SEPARATION` | **PASS** | Qwen Replay remains historical (`live_call=false`, exact badge); never used as a live result. |
| **Overall** | **GO** (Qwen ASK branch not triggered) | All explicit GO gates in the brief are satisfied by real model + real browser evidence. |

## 1. GPU and model service

| Item | Value |
|---|---|
| Instance | `autodl-container-06fa4ab99e-6a73b008`, SSH `connect.cqa1.seetacloud.com:25405` |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB total, driver 570.124.04 |
| Python env | `/root/autodl-tmp/agentrecx-sft/bin/python` — torch 2.11.0+cu128, transformers 5.22.1, CUDA true |
| Model | `/root/autodl-tmp/models/Qwen3.5-9B-ms` (19 GB, 16 files, Qwen3.5 text CausalLM path) |
| Adapter | **None** — base model only; no SFT adapter loaded |
| Server | existing `training/agent_sft/model_server.py`, bind `127.0.0.1:8009`, max_new_tokens 256 |
| Tunnel | temporary WSL SSH `127.0.0.1:18009 → remote 127.0.0.1:8009` (no instance lifecycle operation) |
| Model load | **12.203 s** measured on a clean restart; `proposal server on 127.0.0.1:8009 adapter=None` |
| GPU memory | 17,549 MiB during load; **19,845 MiB peak observed** during real serving; released to 1 MiB after cleanup |
| Lifecycle | model server and tunnel stopped after collecting evidence; the AutoDL GPU instance itself was not created/stopped/released |

Artifacts: [`runs/web_demo/qwen_live/gpu_service_info.json`](runs/web_demo/qwen_live/gpu_service_info.json),
`model_load_measurement.txt`, `model_server.log`, `gpu_service_info_raw.txt`.
The server log also shows optional `causal_conv1d` / `flash-linear-attention` kernels are absent and
the model uses reference PyTorch implementations; this affects speed, not the returned action.

## 2. Real `/plan` inference evidence

Direct real calls (no double): request/action/token/latency saved in
`real_plan_probe_remote.json`, `real_plan_probe_tunnel.json`, and
`real_plan_ask_probe.json`.

| Probe | Action | Latency | Input/Output tokens |
|---|---|---:|---:|
| Remote `/plan`, working-out query | `RETRIEVE` | 3,580 ms | 388 / 69 |
| Through WSL tunnel | `RETRIEVE` | 3,001 ms | 388 / 69 |
| After measured model restart, through tunnel | `RETRIEVE` | 2,621 ms | 388 / 69 |
| BFF health probe | `RETRIEVE` | 2,716 ms | (health call) |

Five additional ambiguous prompts (`something for my ears`, `recommend something`, `a gift`,
`good for music`, `refrigerator`) all produced valid `RETRIEVE` actions; **no ASK was emitted by
the model** in this round. No action was coerced or rewritten.

## 3. Real MCP `run_agent_turn` + existing executor

The BFF started both MCP servers with Qwen configured: both exposed **8 tools** including
`run_agent_turn` (original 7 intact). Real model PlannerActions were executed by the existing
`ElectronicsPlannerExecutor` over the real catalogue/BGE, with Scorer, CandidateLedger,
Eligibility, Decision Gate and P15 evidence reused unchanged.

| Query (profile_0049, read-only) | Qwen actions | Selected | Gate | Evidence | Elapsed |
|---|---|---|---|---:|---:|
| `I need earbuds I can use while working out.` | RETRIEVE → RERANK → FINALIZE (3 calls) | `B08D32JM5F` | `ALLOW_FINALIZE / evidence_sufficient` | P15 bundle yes | 36.3 s cold |
| `I need wireless earbuds for running with good battery life.` | RETRIEVE → RERANK → FINALIZE (3 calls) | `B07GPH9WF3` | `ALLOW_FINALIZE / evidence_sufficient` | yes | 18.2 s |
| `I need something good for music.` | RETRIEVE → RERANK → FINALIZE (3 calls) | `B00B93N0PO` | `ALLOW_FINALIZE / evidence_sufficient` | yes | 17.3 s |
| `I need wireless earbuds under $1.` | RETRIEVE → RERANK → FINALIZE → RETRIEVE → RERANK → FINALIZE → RETRIEVE (7 calls) | none | `NEED_MORE_PRODUCT_EVIDENCE / hard_constraint_unknown`; no finalize | no finalize | 73.0 s |

The last row is the required real **non-FINALIZE** case. The model received a real observation and
kept choosing actions; the existing Gate refused to finalize because the hard requirement was
UNKNOWN. The executor hit its bounded step/budget path (`budget_exhausted`), not a fake success.
Transcript: `real_qwen_mcp_transcript.json`, `real_qwen_nonfinal.json`.

- No parser errors or invalid PlannerActions.
- No Rule fallback or Rule-rewritten answer.
- P15 `evidence_bundle` and `/explain` output were present for finalized turns (`rag` true).
- Qwen planner latencies per action: approx. 2.6–4.7 s; input tokens 388–874; output 69–107 per
  call; total input 1,892–5,108 and output 241–618 per turn.

## 4. Real browser E2E

Full Playwright run against the real local stack (system Chrome, real BFF/MCP/BGE/M9/GPU Qwen):

- **9/9 tests passed**: 7 original CPU/Replay tests + **2 real Qwen Live tests**.
- Qwen-mode browser test 1: real health showed `qwen_live.status=available`,
  `planner_action=RETRIEVE`, 8 tools per profile; the UI showed **Qwen Live Available**,
  `data-planner-kind=qwen_base`, real tool rows (`run_agent_turn`, `qwen_planner`,
  `planner_action`, `retrieve_1`, `rerank_2`, `finalize_3`, `decision_gate`,
  `evidence_preparation`), 10 product cards, Gate `ALLOW_FINALIZE`, and 7 P15 evidence items.
  Screenshot: `qwen-live-real.png`.
- Qwen-mode browser test 2: real non-FINALIZE query; UI showed real repeated actions and Gate
  `NEED_MORE_PRODUCT_EVIDENCE`, no fabricated recommendation. Screenshot:
  `qwen-live-nonfinal.png`.
- Browser turn timings: first Qwen recommendation **58.6 s** (mostly real BGE cold search),
  subsequent Qwen non-FINALIZE turn **25–30 s**; model calls and tools are separately measured.
- Evidence artifacts: `runs/web_demo/qwen_live/e2e_real/` (`summary.json`, `evidence.jsonl`,
  `full_suite_results.json`, screenshots).

## 5. ASK / Memory continuation

The real model did not return `ASK` in any tested prompt; therefore the real Qwen path did not
create a clarification handle in this round. Per the brief, this is recorded honestly rather than
forcing an ASK. The existing CPU path remains real and passed:

- CPU E2E scenario 3: real Gate clarification → browser-authored reply → M9
  `web_ui_explicit_user_reply` with `added=2` → refreshed preferences → continued recommendation.
- No model-generated reply was accepted, and no Qwen memory write was fabricated.

`MEMORY_CONTINUATION` is therefore **PARTIAL (CPU path PASS; Qwen ASK branch not triggered)**.
This is a coverage gap, not a fake success.

## 6. Performance and stability

| Measurement | Value |
|---|---|
| Model server ready after clean restart | **12.203 s** (`proposal server on 127.0.0.1:8009 adapter=None`) |
| `/plan` latency | 2.6–3.6 s direct/tunnel; 2.6–4.7 s inside the multi-step MCP loop |
| Planner calls per finalized turn | 3 |
| Planner calls on the non-FINALIZE under-$1 task | 7 (bounded by existing Gate/step budget) |
| GPU peak observed | 19,845 MiB / 24,564 MiB |
| Browser first real Qwen recommendation | 58.6 s including cold BGE; subsequent Qwen turns 25–30 s |
| Invalid actions / JSON parse errors | 0 |
| Fake FINALIZE / Rule fallback | 0 (Gate rejected the impossible query) |

Stability observation: on the impossible-price query, Qwen repeated a
RETRIEVE/RERANK/FINALIZE cycle until the existing budget ended. This is bounded and honestly
visible in the UI/tool timeline; it is a model-quality/loop tendency, not a Gate or contract
change. No relaxation of Gate/action rules was made.

## 7. Real vs contract double vs replay (explicit)

| Artifact | Meaning |
|---|---|
| `runs/web_demo/qwen_live/real_*`, `e2e_real/` | **REAL_QWEN_LIVE** — actual Qwen3.5-9B Base on the GPU and real browser/API path; basis of the GO verdict. |
| `runs/web_demo/qwen_live/mcp_contract_transcript.json`, `e2e_contract_fake/` | Historical **CONTRACT_DOUBLE** evidence from the previous round (local HTTP planner double). Retained for provenance; **not** used for this GO. |
| `runs/p18/qwen_replay.json`, `/api/replay` | **HISTORICAL_REPLAY** — saved trajectories only, `live_call=false`; never presented as live. |

## 8. Regression commands/results

- Full browser suite: `pnpm exec playwright test` → **9/9 passed**, 103.5 s, no unexpected/failures.
- P17/P16/P15 targeted: `.venv/bin/python -m pytest -q tests/test_p17_mcp.py tests/test_p16_skills.py tests/test_p15_rag.py` → **20 passed** (122.98 s).
- P17 original 7 tools unchanged when Qwen is not configured; with Qwen configured both servers expose 8 tools (verified by `/api/health` and browser evidence).
- Frontend typecheck/build and the previous BFF/Qwen contract tests remain green.
- Reserved Test not run; no training/SFT/adapter, no V0/Gate/Memory semantic change, no BGE rebuild, no commit/push/deploy.

## 9. Artifacts

```
runs/web_demo/qwen_live/
  gpu_service_info.json            GPU/model/service facts, load and memory
  real_plan_probe_remote.json      real /plan request/response
  real_plan_probe_tunnel.json      real /plan through WSL tunnel
  real_plan_ask_probe.json         5 ambiguous real model actions (all RETRIEVE)
  real_qwen_mcp_transcript.json    real MCP run_agent_turn transcripts (3 queries)
  real_qwen_nonfinal.json          real non-FINALIZE Gate evidence
  e2e_real/                        real browser E2E: 9-test JSON/results/evidence
  qwen-live-real.png               real Qwen Live browser screenshot
  qwen-live-nonfinal.png           real Gate-blocked Qwen screenshot
  model_server.log, model_load_measurement.txt
```

## 10. Limitations / final verdict

- The local demo remains a localhost-only engineering demo: no auth/TLS/rate limiting, no
  production deployment, no user-satisfaction/A/B/online-quality claim.
- Qwen ASK and Qwen-mode memory continuation were not triggered by the real model; the CPU memory
  path passed. This must be stated whenever GO is cited.
- The model's rationale is not treated as truth; Gate/P15 remain authoritative. The non-FINALIZE
  query demonstrated a bounded repeated-action tendency that future model-quality work may address.
- The provided GPU instance and the temporary model server/tunnel were managed only for this
  validation; the instance lifecycle was not touched. Service and tunnel are stopped now.
- The worktree also retains an earlier in-progress BGE cold-start single-flight/warmup change from
  the interrupted performance round. It was not evaluated as part of this Qwen verdict and no
  commit was made; all real Qwen tests above passed with the current worktree.

**Final verdict: `GO`** for real GPU Qwen Base Live (real inference, real MCP executor, real Gate,
real browser, saved traces/screenshots, CPU/Replay regression), with the explicit caveat that the
Qwen ASK/memory-continuation branch was not observed.
