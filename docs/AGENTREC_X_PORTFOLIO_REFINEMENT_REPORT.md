# AgentRec-X — GitHub Portfolio Refinement Report

Status: **COMPLETED with one conditional public-release caveat** (2026-11-18).
Scope: presentation and documentation only. No recommendation/Agent/BFF/MCP/Memory/Gate code changed.

## 0. Result at a glance

| Item | Result |
|---|---|
| README changed incrementally from the existing structure | **Yes** |
| Existing sections, badges, screenshots and links preserved | **Yes** |
| V2 GPU Live evidence added | **Yes** |
| Root directories moved or renamed | **No** |
| Production code changed | **No** |
| Private keys / API tokens tracked | **No** |
| All README relative links and images resolve | **Yes** (after this report was added) |
| Python tests | **231 passed** |
| TypeScript / Next build | **PASS** |
| Push | **Not performed** |
| Publish readiness | **Conditional** — operational AutoDL endpoint details remain in historical tracked docs; see §5 |

## 1. Original README structure

`README.md` (354 lines before this round) was a Chinese-first project README with this
existing order:

```text
# AgentRec-X  (title, tagline, 4 badges, EN link)
## 目录
## 项目是什么
## 核心能力
## 界面预览
## 系统架构
## 运行模式
## 快速开始
### 1. 本地 Web Demo（推荐展示入口）
### Legacy Serving Demo 的主要 HTTP 接口
### 2. CPU MCP Server
### 3. 定向测试
## 关键结果（口径必须一起读）
### V2.1 Demo Verification（单次真实运行 / 测试，不是离线平均指标）
## 目录结构
## 文档导航
## 质量与安全边界
```

The README already had a strong evidence-honesty structure: real catalogue size, BGE/ESCI/C4
results with separate protocols, a “cannot be added horizontally” warning, Qwen Live vs Replay
separation, and explicit `NOT ESTABLISHED` quality boundaries.

## 2. What was preserved

- The title, tagline, badges, English README link and Chinese-first voice.
- The full section order and every existing section heading.
- All existing demo screenshots and their order:
  `desktop-search.png`, `desktop-comparison.png`, `desktop-evidence.png`,
  `mobile-search.png`, `qwen-live.png`, `memory-timeline-memory-lab.png`,
  `ranker-replay.png`, `desktop-home-v21.png`.
- The architecture text diagram, Qwen Live / Replay distinction, existing ESCI/C4/P18 result
  table, documentation navigation and safety-boundary text.
- Existing startup commands (`scripts/run_web_demo.ps1`, BFF, MCP, pytest, Next.js).
- All current docs and historical experiment records; nothing was deleted or moved.

## 3. README diff summary

The diff is intentionally small and limited to five areas:

1. **Project positioning (2 lines added).** Added the default Rule Planner and the optional
   verified Qwen3.5-9B Base Live Planner, explicitly stating that failure is reported rather than
   silently downgraded.
2. **Interface preview (6 lines added).** Added a real V2 screenshot row:
   - `behavior-v2-qwen-live-browser.png` — Qwen Live + BGE + Gate;
   - `behavior-v11-eligible-shadow.png` — eligible-only OFF/ON shadow.
3. **Behavior V2 preview paragraph (shortened/replaced).** Kept the capability statement but
   removed duplicated numeric detail; the full figures now live once in the key-results section.
   Added a link to the new key-result anchor and the V2 report.
4. **Key results (22 lines added).** Added one subsection under the existing results section:
   100/100 scenarios, 21 Qwen Live / 79 CPU-BGE, 31 Qwen turns, 136 BGE turns,
   42 eligible-comparable cases, 2 rank changes, latency p50/p95, zero hard-constraint
   violations, zero human LIVE events and `preference_lift = unavailable_no_human_labels`.
5. **Docs navigation and safety boundary (3 lines added/updated).** Added V2 report and this
   report to the existing navigation; added the V2 automated-event/human-label boundary; fixed a
   broken footer link from `AGENTRECX_HANDOFF.md` to
   `docs/handoff/AGENTRECX_HANDOFF.md`.

No full-file formatting, no heading renaming, no translation and no section reordering was done.

## 4. V2 information added and verified against artifacts

Source of truth:
[`AGENTREC_X_BEHAVIOR_FEEDBACK_V2_GPU_E2E_REPORT.md`](AGENTREC_X_BEHAVIOR_FEEDBACK_V2_GPU_E2E_REPORT.md)
and `runs/behavior_feedback_v2/` (local, ignored).

| Claim | Value | Evidence |
|---|---:|---|
| Automated scenarios | 100 / 100 passed | `evaluation_summary.json` |
| Qwen GPU Live / CPU-BGE | 21 / 79 | `evaluation_summary.json` |
| Qwen Live turns / raw actions | 31 / 31 | `qwen_live_trace.jsonl` |
| Online BGE retrieval turns | 136 | `bge_retrieval_trace.jsonl` |
| BGE mode | all `bge` / `bge_dense` | same |
| Eligible-comparable OFF/ON | 42 | `eligible_off_on_comparison.json` |
| Eligible ranking changes | 2 (`v2-022`, `v2-028`) | same |
| Hard-constraint violations | 0 | `gate_validation.json` |
| Human LIVE events | 0 | profile snapshots |
| Qwen Planner latency | p50 11.17 s / p95 14.23 s | `latency_breakdown.json` |
| Qwen turn latency | p50 17.98 s / p95 24.77 s | same |
| BGE retrieval latency | p50 159.64 ms / p95 187.18 ms | same |
| Human labels | 0 | `evaluation_summary.json` |

The README does **not** claim all 100 scenarios are GPU Planner tests. It states that Qwen runs
on the remote RTX 4090 while BGE/BFF/MCP run on local CPU/WSL. It also states that automated
browser events are not human preference labels and that no CTR/satisfaction/quality lift is claimed.

## 5. Public-repository security audit

### P0 result

- **No tracked private key, API token, `.env`, `.env.local` or `.m4_credential`.**
- `git log --all -- .env .env.local .m4_credential` returned no commits.
- Local `.env` and `.env.local` contain provider/artifact credentials but are ignored and were not
  printed, copied, committed or cited. `.m4_credential` is empty.
- The only `ghp_…`, `sk-…` and `AKIA…` strings in tracked files are explicit synthetic fixtures in
  `tests/test_memory_service.py`, `tests/test_setup_readiness.py` and `tests/test_start_demo_env.py`;
  they are not live credentials.
- No model weights, runtime SQLite stores, browser profiles or behavior `runs/` directories are
  tracked (`data/`, `models/`, `runs/`, `*.sqlite*`, `.venv*` and env files are ignored).

### Conditional issue — operational AutoDL endpoint details

Several **historical tracked docs** contain AutoDL SSH hostnames/ports, key filenames and container
IDs. Representative files:

```text
docs/external-eval/ESCI_EXTERNAL_DATA_MANIFEST.md
docs/external-eval/ESCI_EXTERNAL_EVAL_REPORT.md
docs/handoff/NEXT_SESSION_HANDOFF.md
docs/milestones/P14/P14C0_REPORT.md
docs/milestones/P2-P8/P7_A_ARM_WIRING_REPORT.md
docs/milestones/P2-P8/P7_B_SOFT_STATE_AND_GPU_BLOCKER_REPORT.md
docs/qwen/QWEN_LIVE_WEB_REPORT.md
experiments/p14c0/*.py
```

These are **not private-key material** and do not by themselves grant access, but they are
internal/operational endpoint identifiers. They are already in Git history; a `.gitignore` cannot
remove them. This audit does not rewrite history.

Recommendation before a fully public release:

1. redact current endpoint/container-ID strings in the representative docs above;
2. decide whether the historical endpoint exposure is acceptable for the intended audience;
3. if not, perform a separate, authorized history rewrite and rotate/disable the old instance
   endpoints. Do not force-push or rewrite as part of this documentation round.

### Other findings

- Behavior V2 local evidence (`runs/behavior_feedback_v2/gpu_health.json`,
  `experiment_manifest.json`) contains a remote container label. It is ignored and not in the
  public tree; sanitize it if run evidence is ever published separately.
- The largest tracked files are a legitimate frozen ESCI evidence gzip (8.08 MB) and historical
  source diffs under `docs/`; no model or raw data is tracked.
- Root-level `P3*` / `P4*` JSONL traces are local ignored experiment artifacts, not public tree
  content; they were retained rather than deleted.
- README does not contain SSH commands, keys, tokens or full session dumps.

## 6. Root-directory policy

No tracked file was moved, renamed or deleted. No production directory was touched:

```text
recommendation/   web/   web_demo/   tests/   scripts/   docs/   experiments/   data/   models/   runs/
```

Only one new documentation file was added:

```text
docs/AGENTREC_X_PORTFOLIO_REFINEMENT_REPORT.md
```

The existing `docs/` structure was reused. No new asset directory was created; the two new README
screenshots already existed under `docs/assets/web_demo/`.

## 7. Code-impact statement

Changed tracked files in this round:

```text
README.md
docs/README.md
docs/AGENTREC_X_PORTFOLIO_REFINEMENT_REPORT.md   (new)
```

No changes to:

```text
recommendation/, web/, web_demo/, tests/, scripts/, Planner, MCP, BGE,
PreferenceAwareScorer, M9, Eligibility, DecisionGate, Behavior Profile,
Ranking Shadow, P18/P20/P100, AutoDL inference configuration, model weights or indexes.
```

## 8. Link and image verification

A relative-link checker was run over `README.md`, `README_EN.md`, `docs/README.md` and the V2
report. All Markdown targets and all HTML `href`/`src` targets resolve after this report was added.
The previous broken footer link `AGENTRECX_HANDOFF.md` now points to
`docs/handoff/AGENTRECX_HANDOFF.md`.

## 9. Test summary

Baseline and post-change commands:

```text
.venv/bin/python -m pytest -q \
  tests/test_behavior_feedback.py tests/test_web_demo_unit.py \
  tests/test_qwen_live_bff.py tests/test_demo_api.py \
  tests/test_demo_sessions.py tests/test_docs.py
# Baseline: 231 passed in 94.73s
# Post-change: 231 passed in 77.15s

pnpm exec tsc --noEmit        # PASS
pnpm exec next build          # PASS
git diff --check              # PASS
```

External AutoDL/Qwen Live services were not running during this documentation round, so no new
live GPU E2E was executed. The V2 results cited here are from the already completed and frozen
V2 run, not from a new run.

## 10. Git status and pending push

Local branch versus remote:

```text
## master...origin/master [ahead 4]
```

The four local commits not yet pushed before this documentation commit:

```text
9a39119 feat: add behavior feedback v2 gpu live e2e evidence
8add5a7 chore: record clean v1.1 evidence manifest
8d0009c fix: tighten behavior semantics and eligible-only shadow
c9c1f0e feat: add behavior feedback v1 tracking and shadow
```

This round adds one local documentation commit. No push, force-push, branch deletion, release, or
history rewrite was performed.

## 11. Publish readiness

| Area | Status |
|---|---|
| Private keys / API tokens in tracked tree | **PASS — none** |
| `.env` / local credential files excluded | **PASS** |
| Model/data/runtime artifacts excluded | **PASS** |
| Links and screenshots | **PASS** |
| Tests/build unchanged | **PASS** |
| AutoDL SSH endpoint/container-ID hygiene in tracked historical docs | **ATTENTION** |
| Fully public release | **CONDITIONAL** |

The repository is suitable for a **private/internal GitHub portfolio review** now. Before a fully
public release, decide on the AutoDL endpoint/container-ID redaction/rotation issue above. This
report does not claim that issue is resolved by `.gitignore`.

## 12. Answers to the required twelve questions

1. **Original README structure:** the `目录` block plus ten main Chinese content sections in the
   order documented in §1, with V2.1 results and docs navigation already present.
2. **Preserved:** title/tagline/badges, section order, all existing screenshots, architecture
   diagram, result table, mode table, startup commands, docs map and safety boundaries.
3. **Modified paragraphs:** project positioning, V2 preview paragraph, docs navigation, safety
   boundary and one broken footer link.
4. **V2 additions:** one screenshot row and one key-results subsection with the verified
   100-scenario/BGE/Qwen/latency/human-label numbers.
5. **Duplication:** the V2 preview paragraph was shortened so the numeric table appears once;
   V1/V1.1/V2 are not three repeated long sections.
6. **Root changes:** no tracked root file moved/renamed/deleted; one new report under `docs/`.
7. **Production code:** not modified.
8. **Images/links:** all relative targets resolve.
9. **Public-content findings:** no secret material tracked; conditional AutoDL endpoint exposure
   in historical docs; behavior evidence is ignored/local.
10. **Tests:** 231 passed; TypeScript/build PASS; link check PASS; no live E2E rerun.
11. **Pending push:** four pre-existing commits listed in §10 plus the documentation commit.
12. **Safe to publish:** conditional, as described in §11.
