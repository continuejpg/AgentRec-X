# GIT_ARCHIVE_REPORT — AgentRec-X local Git archive

> **Question this report answers:** *Is this archive enough to reproduce the Web Demo on another
> machine that has the lawful data and environment dependencies?* — **No from Git alone. Yes with
> the external assets documented in [`docs/ARCHIVE_MANIFEST.md`](../ARCHIVE_MANIFEST.md) plus a
> compatible WSL/Python/Node environment.** The source, tests, docs, locks, licences and run
> commands are complete in Git; the multi-hundred-MB catalogue/model/index are intentionally not
> in Git and must be obtained or rebuilt as described there.

## 1. Archive coordinates

| Item | Value |
|---|---|
| Repository | `D:\IT\CODE\PROJECTION\agentrec\X` |
| Baseline branch | `feat/agent-source-limit-ablation` |
| Baseline HEAD | `4dec6d2` (`fix(test): resolve remaining suite failures`) |
| Archive branch | **`archive/agentrecx-web-demo-v1`** |
| Archive commit A (P14–P18 snapshot) | **`8eca2c6a0946a6dc0f52c925c6ccaa5d7e42cc42`** |
| Archive commit B (P19 Web Demo + README) | **`ce23f7ab00b6645c11bb8c72528e483baa80e510`** |
| Archive metadata commits | **C + D** (branch tip; see `git log -1 --format=%H` on the archive branch) |
| Remote / push / release | **none** — local only; no remote was contacted |
| Existing history | unchanged; archive commits are new descendants of `4dec6d2` |
| Pre-existing worktree changes | **408 status entries** at baseline; none deleted or reset; they were committed in A |

The baseline status snapshot is preserved at `runs/web_demo/baseline/` (git-ignored evidence,
kept locally for audit): `head.txt`, `branch.txt`, `git_status_before.txt`,
`git_diff_stat_before.txt`, `captured_at.txt`, and the pre-edit copy of
`recommendation/mcp/server.py`.

## 2. Commit structure

| Commit | Scope | Files | Insertions / deletions |
|---|---|---:|---|
| **A** `8eca2c6` | P14–P18 source, tests, reports, docs, benchmarks, training code/configs, legacy experiment archives, serving/agent changes already present in the worktree | **647** | +128,847 / −13,456 |
| **B** `ce23f7a` | P19 Web Demo: `web/`, `web_demo/`, `e2e/`, `third_party/`, Web docs, committed screenshots, run scripts, README entry, `.gitignore` safety rules | **95** | +19,428 / −7 |
| **C** | `GIT_ARCHIVE_REPORT.md`, `GIT_ARCHIVE_SOURCE_DIFF.txt`, `docs/ARCHIVE_MANIFEST.md`, `docs/archive_python_freeze.txt`, `scripts/git_archive_source_diff.py`, refreshed `docs/assets/web_demo/*.png` | **13** | small |
| **D** (tip) | final metadata consistency commit: this report, the source-diff generator and the regenerated source diff | **3** | small |

A is deliberately a *snapshot* of mixed pre-existing work: the worktree did not contain a reliable
separation of each earlier milestone, and the brief explicitly permits one clearly-scoped local
milestone commit instead of inventing history. B and C are cleanly separated as the Web Demo and
archive-metadata layers.

## 3. What is in the archive

* **Serving / Agent / control-plane source:** `recommendation/` (including the P17 MCP server),
  `experiments/`, `benchmarks/`, `skills/`, `training/` code/configs/reports, `config/`.
* **P19 Web Demo:** `web/` (Next.js 16/React 19/TypeScript/Tailwind 4), `web_demo/` (FastAPI BFF
  over the official P17 MCP stdio client), `e2e/` (Playwright), launchers/stop/E2E scripts,
  `docs/WEB_DEMO_CONTRACT.md`, `WEB_DEMO_GUIDE.md`, `WEB_DEMO_REPORT.md`,
  `WEB_DEMO_SOURCE_DIFF.txt`, committed screenshots under `docs/assets/web_demo/`.
* **Third-party compliance:** `third_party/` (Apache-2.0 licence copy, upstream reuse guide,
  dependency manifest, header-audit tool/result) and `THIRD_PARTY_NOTICES.md`.
* **Tests:** existing suite + `tests/test_web_demo_unit.py`, `tests/test_web_demo_integration.py`,
  plus the E2E suite under `e2e/tests/`.
* **Dependency locks:** `requirements*.txt`, `web/pnpm-lock.yaml`, `e2e/pnpm-lock.yaml`,
  `docs/archive_python_freeze.txt` (84 packages from the verified WSL venv).
* **Reports/docs:** P10–P18 reports and source diffs, architecture/onboarding/experiments docs,
  P18 final report, `GIT_ARCHIVE_*`, `docs/ARCHIVE_MANIFEST.md`.
* **Legacy experiment archives:** `archive/legacy_experiments/*.tar.gz` (small, P14-0 retention
  archives of removed scripts/scratch; no model weights).

## 4. What is intentionally NOT committed (and why)

| Excluded | Size / reason |
|---|---|
| `data/` (Amazon-C4/processed data, P13-R/P14-E2 fixtures) | multi-GB; licence/redistribution restrictions; rebuild from lawful raw data |
| `models/bge-base-en-v1.5/` | 438 MB model weights; fetch from HuggingFace with pinned revision |
| `runs/` (BGE index 213 MB, checkpoints, local reports) | generated/local evidence; 10 small evidence files already tracked remain; external assets documented by SHA |
| `training/agent_sft/output/*.tar.gz` | 2 × ~164 MB machine-local LoRA adapters |
| `.venv*`, `node_modules/`, `.next/`, caches | rebuildable environments/build output |
| `.env`, `.env.*`, `.m4_credential`, SQLite DBs, `webapp_accounts.json` | secrets/credentials/local state |
| `runs/web_demo/` logs/screenshots/traces | local E2E artifacts; 8 final screenshots are copied into `docs/assets/web_demo/` and committed |

Safety rules were added to `.gitignore`: root-anchored `/models/` (so source packages such as
`recommendation/models/` are **not** hidden — this was caught and fixed during staging),
`training/agent_sft/output/*.tar.gz`; `e2e/.gitignore` no longer hides `pnpm-lock.yaml`.
No file under `data/`, `models/`, `runs/`, `node_modules/`, `.venv` or any secret path is in the
archive commits.

## 5. Sensitive / large-file scan

* **Forbidden-path check** over the staged archive: no `data/`, `models/`, `runs/`,
  `node_modules/`, `.next/`, `.venv/`, `.env*`, `.m4_credential` or adapter tarballs.
* **Largest staged file:** 0.32 MB (`P3C2B_SOURCE_DIFF.txt`); the archive source payload is
  ~9.7 MB in commit A + ~2.2 MB in commit B (including screenshots).
* **Secret-pattern scan** of staged text files (and text members of the small legacy tar archives):
  two hits, both benign and reviewed:
  1. `tests/test_m5_1_state_adapter.py`: a **fake** `sk-abcdefghijklmnop` fixture used to assert
     secret-looking preference values are refused;
  2. `archive/legacy_experiments/p2b_agent_diagnostics.tar.gz::experiments/credential_audit.py`:
     a credential-auditor source file whose own regex/comment text matches the scan.
  No live API key, HF token, GitHub token, AWS key or private key was found. `.env`,
  `.env.local` and `.m4_credential` are ignored and were never staged.
* `web/pnpm-lock.yaml` and `e2e/pnpm-lock.yaml` contain registry URLs/integrity hashes only.

## 6. Post-archive verification (targeted, not full pytest)

Commands run after commit B:

| Check | Command | Result |
|---|---|---|
| P17/P16/P15 targeted regression | `.venv/bin/python -m pytest -q tests/test_p17_mcp.py tests/test_p16_skills.py tests/test_p15_rag.py -p no:cacheprovider` | **20 passed in 118.04 s** |
| MCP tool schema | read `runs/p17/tool_schemas.json` | protocol **2026-07-28**, **7/7 tools** (`search_products`, `get_product_info`, `recommend_products`, `get_active_preferences`, `compare_products`, `request_preference_clarification`, `submit_clarification_reply`) |
| Web E2E | `powershell -ExecutionPolicy Bypass -File scripts/run_web_demo_e2e.ps1` | **7/7 passed**, Playwright **68.5 s**, demo startup **33.8 s**, 8/8 screenshots, ports released after cleanup |
| Frontend build/typecheck | `cmd /c "pnpm --dir web exec tsc --noEmit"` and `cmd /c "pnpm --dir web build"` | exit 0 / build succeeded (verified in this round before the archive commits; source unchanged in C) |
| Legacy licence/header audit | stored `third_party/header-audit.json` + manual re-check of 19 adapted file headers | **PASS**, 0 problems; 19 files retain Anthropic copyright/SPDX/modification note, 15 pure-new modules have neutral project headers |
| README local-link check | `python` link scan over `README.md` | **PASS** — all relative file/image links resolve (excluding external URLs/anchors) |
| Git cleanliness after C | `git status --short` | expected: **empty** (all tracked archive content committed; only git-ignored local assets/logs remain) |

The Web E2E re-ran after the archive commits and confirmed: real BGE search/recommend, P15
evidence spans + UNKNOWN, P16 comparison + real 400 on invalid refs, real clarification with M9
`added=2` and `web_ui_explicit_user_reply`, refreshed preferences, resumed recommendation,
mobile drawers/composer, Qwen Replay label/Qwen Live disabled, and real BFF error boundaries.

## 7. Reproducibility answer (explicit)

**Git archive alone: NO.** A fresh clone contains every source file, script, test, lock, report,
licence notice and screenshot, but **not** the four runtime assets in
`docs/ARCHIVE_MANIFEST.md`:

1. Electronics catalogue `data/p13r/electronics/catalog_meta.jsonl` (32.3 MB, SHA-256
   `14f27eed…`);
2. P14-D2 BGE index `runs/p14d2/bge_index.npz` (213.2 MB, `fce84987…`);
3. BGE model `models/bge-base-en-v1.5/` (weights 438 MB, `4b9d4a4b…`);
4. P17 memory fixture `data/p14e2/benchmark/dev.agent.jsonl` (27.9 KB, `a1bd72a7…`).

**With those assets + a compatible environment: YES for the Web Demo.** The archive includes the
one-command runner, BFF, frontend, E2E suite, dependency locks, hashes and run instructions. Known
gaps are stated in the manifest: raw Amazon data is not redistributable; the P14-D2 BGE-index
builder is not a standalone archived script (obtain the accepted index or re-encode with the
documented parameters and verify identity/hash); the CPU BGE warm-up is tens of seconds.

## 8. Residual / uncommitted state

After the metadata commits (C/D) the intended state is: **no uncommitted source or documentation files**. Remaining
local-only content is intentionally ignored (see §4), including `data/`, `models/`, `runs/`
assets, virtualenvs, node_modules and local logs. If `git status --short` is non-empty after
checkout, it should only be those ignored paths (which `git status` does not print by default).

There were **no files of uncertain provenance** left to force-commit: the pre-existing worktree
changes were source/report/test files; the only binary-ish items were the small reviewed
`archive/legacy_experiments/*.tar.gz` retention archives (committed) and screenshots (committed
under `docs/assets/web_demo/`).

## 9. Commands used for the archive

```powershell
# inspect (read-only) and baseline capture
git status --porcelain; git log --oneline -3
# create the local archive branch (no reset/clean/checkout)
git switch -c archive/agentrecx-web-demo-v1
# commits A/B/C were created from explicit staged path lists; never `git add -A` over the repo
# verification after archive
wsl -d Ubuntu-22.04 --cd /mnt/d/IT/CODE/PROJECTION/agentrec/X -- \
  .venv/bin/python -m pytest -q tests/test_p17_mcp.py tests/test_p16_skills.py tests/test_p15_rag.py
powershell -ExecutionPolicy Bypass -File scripts/run_web_demo_e2e.ps1
# no git push / no release / no remote operation was executed
```

## 10. Verdict

| Item | Result |
|---|---|
| Local commit(s) | **YES** — A `8eca2c6`, B `ce23f7a`, C/D metadata branch tip |
| Traceable archive | **YES** — baseline/parents, source diff, manifests and hashes recorded |
| README Web entry + screenshots | **YES** (committed) |
| Dependency/resource manifest | **YES** — `docs/ARCHIVE_MANIFEST.md` + freeze + locks |
| Third-party licence info | **YES** — `THIRD_PARTY_NOTICES.md` + `third_party/` |
| Reproducible demo instructions | **YES** (one command) — subject to external assets |
| Post-archive Web E2E | **7/7 PASS** |
| P17/P16/P15 regression | **20 passed** |
| Sensitive/large data committed | **NO** — deliberate exclusions documented |
| Push/release/remote | **NONE** |
| Git history rewritten | **NONE** |

**Bottom line:** the archive safely preserves the completed engineering work and gives a clear,
verified local entry point. It is **not** a self-contained artifact bundle; the external data/model
assets and their licensing remain the boundary. That boundary is stated here rather than hidden.
