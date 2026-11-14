# README Review — AgentRec-X (pre-edit snapshot `922a10b`)

> Review only. The README had not yet been modified when this snapshot was taken.
> The existing layout, image order, headings and anchors are treated as the baseline.

## A. Content to preserve

1. **Title, tagline and badges** — the one-sentence positioning “LLM Shopping Agent /
   能被验证、知道边界的购物 Agent” is better than a generic “AI recommender” claim.
2. **Original section order and anchors** — 项目是什么 → 核心能力 → 界面预览 → 系统架构 →
   运行模式 → 快速开始 → 关键结果 → 目录结构 → 文档导航 → 质量与安全边界.
3. **All existing images** — 4 image references in the Chinese README (plus `replay-mode.png`
   in the English README); all resolve under `docs/assets/web_demo/`:
   - `desktop-search.png`
   - `desktop-comparison.png`
   - `desktop-evidence.png`
   - `mobile-search.png`
   - `replay-mode.png`
   Keep their order and alt text. The broader asset folder also contains
   `desktop-clarification-memory.png`, `mobile-comparison-sheet.png`, `mobile-drawer.png`;
   these are retained but not currently referenced.
4. **Architecture text diagram** — useful as the “bounded deterministic backbone + thin MCP”
   story; should be extended, not replaced.
5. **CPU / Qwen Live / Qwen Replay mode table** — correct architectural distinction.
6. **ESCI / C4 / P18 result table and the “cannot be added horizontally” caveat** — this is the
   strongest evidence-honesty part of the README.
7. **Documentation navigation and quality/safety boundary** — keep.

## B. Content needing an incremental update

1. **核心能力 / Agent row** does not show the verified **Qwen3.5-9B live PlannerAction** path,
   nor **Browser → Next.js → FastAPI BFF → MCP → Qwen → Gate**.
2. **系统架构 text** still says `Rule Planner` only; it should show the optional Qwen Planner
   branch while keeping Rule as the CPU default.
3. **运行模式** says Qwen Live is optional, but the current V2.1 status is: **real browser turn
   verified**, with WSL-local loopback tunnel and the real `/plan` health check.
4. **MCP tool count** says 7 tools; the write-enabled local demo now has one thin
   `remember_preference` adapter (8 tools on that profile). The default/read-only profile still
   has 7.
5. **Session Memory Isolation** is missing: per-browser-session M9 namespaces, A/B isolation,
   read-only profile refusal, process-local persistence warning.
6. **界面预览** lacks Qwen Live, Memory Timeline/Memory Lab, and Ranker Replay screenshots.
7. **Quick start** does not link the new V2.1 Demo Guide, Setup/Deployment or Architecture docs.
8. **关键结果** has the correct existing ESCI / catalogue figures but no separate V2.1
   verification block for:
   - Qwen Live 20.58 s single real browser turn;
   - session isolation;
   - 38 WSL tests / 20 local tests / TypeScript PASS.
9. **质量与安全边界** needs the updated statement: Qwen Live is verified but requires a local
   tunnel; session memory is process-local and lost on BFF restart; ranker comparison remains
   recorded Replay.

## C. Recommended additions (README vs separate docs)

### Add directly to README

- A small **V2.1 Demo verification** subsection with real screenshot and measured facts.
- New screenshot row for Qwen Live / Memory Timeline / Ranker Replay.
- A Qwen Planner branch in the architecture text.
- Existing-results table additions, clearly separated from offline metrics.
- Links to the new guides.

### Keep in separate docs

- Full step-by-step UI walkthrough → `docs/DEMO_GUIDE.md`.
- Full CPU / remote GPU / SSH tunnel / env var / troubleshooting → `docs/SETUP_AND_DEPLOYMENT.md`.
- Full component and sequence architecture → `docs/ARCHITECTURE.md` (update existing).
- V2.1 exact raw evidence and logs → `docs/AGENTREC_X_DEMO_V21_VERIFICATION.md`.

### Do not add to README

- Raw tool JSON, full session IDs, SSH commands with private host/key paths, or all experiment
  failure histories. These belong in docs or run artifacts.

## D. Pre-edit consistency check

- **Images:** all referenced images exist at their current relative paths.
- **Links:** README references `README_EN.md`, `PROJECT_ARCHITECTURE.md`, `docs/ARCHITECTURE.md`,
  `DEMO_GUIDE.md`, `docs/web-demo/WEB_DEMO_GUIDE.md`, `P18_FINAL_REPORT.md`, `docs/USAGE.md`,
  `docs/REPRODUCE_AGENT_EVAL.md`, and test/report paths; the reviewed files exist.
- **V2.1 evidence:** `docs/AGENTREC_X_DEMO_V21_VERIFICATION.md`,
  `runs/final_demo/ui_qwen_live.png`, `runs/final_demo/qwen_live_turn_v21.json`,
  `runs/final_demo/qwen_raw_planner_action.json`, and
  `runs/final_demo/session_isolation.json` exist locally.
- **Remote:** the V2.1 commit `922a10b` was pushed to `origin/master` before editing the README;
  `origin/master` is `922a10b` and the local branch is not ahead/behind.
