# AgentRec-X — Final Engineering Polish Report

**Date:** 2026-11-10 (Asia/Shanghai)  
**Scope:** Git archival, Web/MCP candidate consistency, BGE first-query stability/performance  
**Final verdict:** **PASS** for all three requested phases.

All work is local. No push, release, deployment, model training, benchmark expansion, Reserved Test,
catalogue/index rebuild, V0 weight change, M9 permanent-history semantic change, new MCP tool or
AutoDL lifecycle operation was performed.

---

## 1. Git archive

### 1.1 Audited starting state

- Branch: `archive/agentrecx-web-demo-v1`
- Baseline HEAD: `c24b138bbc003689e3d75cd1f8dcae35d9dcad41`
- Uncommitted at start: **32 modified + 18 untracked = 50 entries**
- The only BGE performance implementation WIP was
  `experiments/p14c1/search_backend.py` (single-flight/warmup draft).
- `.gitignore` correctly excludes `data/`, `models/`, `runs/`, checkpoints, `.env*`, SQLite DBs and
  large adapters. No large model/data/index artifact was staged.
- Staged review found no secret-like patterns. Licence/notice files were untouched and retained.

### 1.2 Local commits

All three Qwen layers touch the same MCP/BFF/Web files. Per the brief's allowance for file overlap,
they were archived as one coherent verified Qwen commit rather than unsafe hunk splitting.

| Commit | Purpose |
|---|---|
| `3c89e7e62335fb9f5c0e526b5a2b46871e69136c` | Qwen Base Live, real ASK→M9→Resume, task-scoped clarification consistency, tests, E2E, contracts and reports (46 files) |
| `93360debe004282f29aaffc549d5a6b0ae2eaa49` | Historical Qwen acceptance evidence source-diff snapshots (documentation only) |
| `b7303e89498e9d5262a2886f49ba02cf80307acc` | Corrected Gate-selected E2E assertion and candidate diagnosis; no runtime ranking change |
| `a40ec881d986bb0d9ae4c28ff83544ccb21fe6ff` | Validated BGE single-flight/warm-up/Search Ready performance implementation (6 files) |

The BGE implementation was excluded from the Qwen commit while it was unverified, kept intact in
the worktree, then validated and committed separately as `a40ec88`.

### 1.3 Safety

- No `git reset --hard`, `git clean`, forced checkout or blind `git add .`.
- Staged file list and diff were inspected before each commit.
- No model weights, datasets, index files, credentials or secrets were committed.
- No push/release/deploy and no old history rewritten.
- The final working tree is clean after the documentation commit.

---

## 2. Web vs Direct MCP candidate diagnosis

### 2.1 Actual result

The earlier apparent difference was **not a candidate-quality or ranking defect**.

- Web and Direct MCP both had **15 input candidates**, **14 eligible**, with
  `B091SF1545` excluded by the current `avoid wired` hard exclusion.
- Both used the **same eligible candidate ID list**:
  `B00M0KUF44, B0BVZKKYBS, B00NFL0E2A, B00B93N0PO, B015M9NKUK, B07ZF8893F,
  B0C3LYY9NL, B017ILTARW, B000P93MYM, B007ZG32I4, B01N7S0IPR, B00FCUCMLI,
  B09XCWX4JG, B01LS2IG6I`.
- Both Gate-selected the **same final product: `B00M0KUF44`**
  (`RunPhones Wireless Headphones (Graphite Gray, Medium).`).
- `B00NFL0E2A` (`Philmore Headphones.`) was the **first initial catalogue-result card** on the
  shared turn, not the Gate-selected item. The old E2E assertion read the first
  `product-card`, producing the incorrect "Web selected Philmore" conclusion.
- The E2E now reads the dedicated `gate-selected-item` banner. The runtime V0/Scorer/Gate code
  required **no change** for candidate ordering.

Evidence: `runs/web_demo/final_polish/candidate_diagnosis.json`, plus the corrected E2E
`qwen_fix.reevaluation_pool` and `qwen_fix.final_product_not_wired` checks.

### 2.2 Soft preference behavior

- `avoid wired` remains a task-scoped hard exclusion.
- `prefer wireless` is a soft preference.
- In the same pool, `B00M0KUF44` has explicit catalogue `wireless` evidence and ranks before
  `B00NFL0E2A`, whose connection evidence is UNKNOWN. UNKNOWN is not treated as a positive match.
- Hard exclusion is applied before Gate; no ASIN special case and no weight tuning.

**Diagnosis:** PASS. **Runtime repair required:** none; only the E2E/reporting assertion was fixed.

---

## 3. BGE first-query performance and initialization stability

### 3.1 Implementation

`experiments/p14c1/search_backend.py` + `recommendation/control/catalog_search.py`:

- Single-flight encoder initialization with a lock + ready event.
- Double-checked ready path; model/tokenizer published only after both load and `eval()` succeed.
- Clear attempt-limited failure state (2 attempts) and no half-initialized model publication.
- One controlled recovery retry in `warmup()`.
- A real warm-up query plus a real dense retrieval check before MCP readiness.
- Encoder status/timings exposed to the MCP manifest/log: catalogue, configure, warm-up,
  pre-handshake and Search Ready.
- Injectable encoder loader for deterministic failure/single-flight tests; no production behavior
  change.

`recommendation/mcp/server.py` + `web_demo/mcp_manager.py`:

- BGE profiles receive `--warmup` from the Web BFF; startup synchronously reaches Search Ready even
  if the encoder is still cold.
- `AGENTRECX_WEB_DEMO_BGE_WARMUP=0` is available only for controlled before/after measurements.
- No fake readiness: if warm-up fails after the bounded retries, MCP startup fails explicitly; the
  BFF shows the error and can be restarted.

`scripts/run_web_demo.ps1` forwards the controlled BGE warm-up switch.

No BGE model weights or BGE index were changed or rebuilt.

### 3.2 Measured cold starts (three Web runs each, same host/assets/query)

Query: `wireless headphones for commuting`.

| Metric | Baseline (no prewarm) | Warmed BGE | Change |
|---|---:|---:|---:|
| BFF ready (`/api/health`) | 39.60–62.20 s, avg **49.53 s** | 45.73–58.47 s, avg **50.14 s** | +0.61 s avg (ranges overlap) |
| MCP per-server startup | 30.0–48.9 s | 36.0–49.1 s | includes encoder warm-up |
| **First real user query** | 15.63–17.12 s, avg **16.28 s** | 0.54–0.56 s, avg **0.552 s** | **−15.73 s / −96.6%** |
| **Ready → first result** | 55.69–79.32 s, avg **65.81 s** | 46.28–59.04 s, avg **50.69 s** | **−15.12 s / −23.0%** |
| Warm query | 0.283–0.300 s, avg **0.288 s** | 0.280–0.298 s, avg **0.288 s** | unchanged |

MCP internal warm segments (current manifests):

- Catalogue load: **~7.95–8.33 s**
- Index/configure: **~5.19–5.79 s**
- Encoder warm-up: **~8.91–18.42 s** (OS/file-cache dependent)
- MCP pre-handshake total: **~13.38 s** baseline vs **~22.95–32.01 s** warmed
- MCP startup including transport/client handshake: **~36–49 s** warmed.

Direct source measurement (same fixed query):

- No prewarm: first query **17.56 s**, total process **31.94 s**
- Prewarm: warmup **15.95 s**, first query **0.092 s**, total process **31.40 s**
- First 10 hit IDs identical in both runs; search results/evidence unchanged.

Frontend: the final warmed CPU browser suite's first real turn measured **~1.7 s**
(`01-search-recommend` chat.turn.latency), after page readiness.

### 3.3 Concurrency / cancellation / failure

- **4 concurrent real first queries**: all completed in **17.41 s**; `encoder_status.attempts=1`;
  all four hit lists identical.
- **Abandoned requester then retry**: retry **233 ms** after initialization, identical hits,
  `attempts=1`; no half-initialized state.
- **Real BFF client abort then retry**: retry **570 ms** after cancellation/25 s wait; same query
  completed. This test intentionally closed the SSE stream after the initial event; it does not
  claim a full mid-model-load cancellation was measured.
- **Failure/recovery unit tests**: first load failure → second attempt succeeds; repeated failure
  stops after 2 attempts with explicit error; ready event only after success.
- Failure at MCP warm-up is a visible startup error, not fake ready.

### 3.4 Resource and isolation cost

At Search Ready with two independent profiles:

| Item | Baseline ready (encoder not yet loaded) | Warmed ready |
|---|---:|---:|
| Each MCP RSS | ~1.124 GB | ~1.548 GB |
| Two MCP RSS | ~2.25 GB | ~3.10 GB |
| BFF RSS | ~70 MB | ~70 MB |
| Host RAM | 7.7 GB total | 7.7 GB total |

The encoder costs **~0.42 GB per profile**; two profiles pay **~0.85 GB** extra at readiness.
This is the same encoder that the baseline would load on first user query; the cost is shifted
into startup and reported, not hidden. No shared-encoder/public arbitrary-user process was created;
P17 profile isolation remains one MCP process per fixed user key.

---

## 4. Regression

| Check | Result |
|---|---|
| `tests/test_bge_performance.py` | **5 passed** (single-flight, retry, repeated failure, warm-up/retrieval, BM25 no-op) |
| Targeted BGE/P14-C1/P14-D2/catalog/P17/P16/web/Qwen contracts | **74 passed** (152.98 s) |
| Final post-commit CPU/Replay Playwright `01`–`06` | **7/7 passed**, 13.1 s |
| Final warmed browser first turn | **~1.7 s** |
| BGE first-hit IDs baseline vs warmed | **identical, 10/10** |
| Warm-query regression | unchanged (avg 0.288 s both) |
| Qwen Live interface contracts (`test_qwen_live_unit/bff`) | passed; Qwen GPU service not restarted in this phase per brief |
| CPU Live / Qwen Replay | passed in 7/7 |
| P16 comparison / M9 clarification write / Gate | passed in 7/7 |
| Search product IDs / evidence spans | unchanged in direct BGE comparison and browser scenarios |

The final browser run was against the committed code (HEAD `a40ec88`), after the BGE changes.
The real Qwen GPU service was deliberately not restarted for phase 3; its previously verified
ASK→Memory→Resume evidence remains archived, and the Qwen contract tests pass.

---

## 5. Status matrix

| Required status | Result |
|---|---|
| `GIT_ARCHIVE` | **PASS** |
| `QWEN_CHANGES_ARCHIVED` | **PASS** |
| `BGE_WIP_PRESERVED` | **PASS** (preserved uncommitted until validated; committed as `a40ec88`) |
| `WEB_MCP_CANDIDATE_DIAGNOSIS` | **PASS** |
| `SOFT_PREFERENCE_EVIDENCE_ORDERING` | **PASS** |
| `GATE_CONSISTENCY` | **PASS** |
| `BGE_SINGLE_FLIGHT` | **PASS** |
| `FIRST_QUERY_LATENCY` | **IMPROVED** (−96.6% first BFF query; −23.0% ready→first result) |
| `WARM_QUERY_REGRESSION` | **PASS** |
| `MULTI_PROFILE_ISOLATION` | **PASS** |
| `WEB_E2E` | **PASS** |
| **Overall** | **PASS** |

---

## 6. Required final answers

1. **Are all ~50 modifications safely saved?**
   Yes. The verified Qwen, ASK/Resume, clarification and BGE changes are local commits; the
   pre-existing uncommitted work was preserved and committed, and the BGE WIP was not lost. The
   final worktree is clean.
2. **What is committed vs WIP?**
   All requested Qwen and BGE implementation work is committed in the four SHAs above. There is no
   implementation WIP left; the only non-Git content is intentionally ignored runtime evidence
   under `runs/`.
3. **Why did Web and Direct MCP pick different products?**
   They did not. Both selected `B00M0KUF44` from the same 14 eligible IDs after task re-evaluation.
   The earlier `B00NFL0E2A` claim came from an E2E assertion reading the first initial product card
   instead of the Gate-selected banner. The assertion/report were corrected; no runtime change.
4. **Is an evidence-backed wireless product correctly preferred in the same pool?**
   Yes. `B00M0KUF44` has explicit wireless catalogue evidence and ranks ahead of the UNKNOWN
   `B00NFL0E2A`; UNKNOWN is not treated as MATCH. No ASIN special case or weight change.
5. **How much faster is the first search?**
   BFF first real user query: **16.28 s avg (15.63–17.12) → 0.552 s avg (0.540–0.563)**,
   about **96.6%** less wait. Direct BGE first query: **17.56 s → 0.092 s**. Ready→first result
   average: **65.81 s → 50.69 s** (**23.0%** lower). Warm queries are unchanged at ~0.288 s.
6. **Did resource/startup total cost increase?**
   The encoder now loads during startup: two-profile MCP RSS at ready rises **~2.25 GB → ~3.10 GB**
   (**~0.85 GB extra**, ~0.42 GB per isolated profile). BFF ready average changed only
   **+0.61 s** in these three-run samples (ranges **39.6–62.2 s → 45.7–58.5 s**, overlapping), while
   ready→first result dropped ~15.1 s. The startup warm-up is included in the reported cost; no
   calculation was hidden.
7. **Did BGE changes break Qwen ASK → Memory → Resume?**
   No contract change was made; Qwen unit/BFF contract tests pass and the previously archived real
   ASK/Resume evidence remains. The real Qwen GPU service was not restarted for this phase, so no new
   real GPU ASK test is claimed; CPU Live/Qwen Replay and the shared MCP/search/Gate paths passed.
8. **Can the project move to final demo/resume polish?**
   Yes for the engineering scope: three phases PASS, local Git archive complete, candidate
   consistency explained, first search substantially faster and stable. The remaining external
   requirement is to start the GPU Qwen service and tunnel for a live Qwen demo; those are
   intentionally stopped now.

---

## 7. Evidence map

- Git commit list/status: `runs/web_demo/final_polish/git_commits.json`, final `git status`.
- Candidate diagnosis: `runs/web_demo/final_polish/candidate_diagnosis.json`
- BGE performance summary: `runs/web_demo/final_polish/performance_summary.json`
- Raw startup/query logs: `runs/web_demo/final_polish/{startup,health,bff_query,rss}_*.json/txt`
- Concurrency: `runs/web_demo/final_polish/bge_concurrency.json`
- Cancellation/retry: `runs/web_demo/final_polish/bff_cancel_retry.json`
- Direct BGE baseline/warm: `bge_baseline.jsonl`, `bge_prewarm.jsonl`
- Final CPU/Replay E2E log: `runs/web_demo/final_polish/final_warm_cpu_e2e.log`
- Targeted tests: `runs/web_demo/final_polish/pytest_final.log`
- Qwen acceptance evidence: `QWEN_LIVE_WEB_REPORT.md`, `QWEN_ASK_RESUME_REPORT.md`,
  `QWEN_CLARIFICATION_FIX_REPORT.md`, `runs/web_demo/qwen_*`
- Source diff: `FINAL_POLISH_SOURCE_DIFF.txt`

Current services are stopped; GPU memory is **1 MiB**; the existing AutoDL instance was not
lifecycle-managed. No commit/push was sent to any remote.
