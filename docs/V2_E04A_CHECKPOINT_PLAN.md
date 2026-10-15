# AgentRec-X V2 / E-04A — 阶段 1—5 Git 检查点最终审核清单（只读；未执行任何 Git 写操作）

> 生成时间：2026-10-15 16:46:10 · HEAD `faeff0b6f790d8c9d077d7d07ee26998aeecad2a`
> 依据：`git status --porcelain -uall` 的**实际输出**逐行分类（单一分类函数驱动清单与所有 `git add` 命令）
> 工作树实况：**32 modified + 62 untracked = 94 个候选文件**
> 本轮**未**创建分支、**未** `git add`、**未** `git commit`、**未**触碰 `.env` 与业务代码。

## 0. 本轮核验重点（按你的要求）

| 检查 | 结果 |
|---|---|
| E-04A 实验脚本是否已纳入清单 | ✓ `experiments/e04a_provider_smoke.py` 列于 **E 类**（待你确认） |
| `usage()` 修复是否正确 | ✓ 离线核对：`usage()` 为方法 → 现读取 `input_tokens/output_tokens/total_tokens/latency_ms/estimated_cost/request_id`；并用假 client 验证记录与调用上限（第 2 次调用**抛错不发送**） |
| E-04A 报告是否齐全 | ✓ `V2_E04A_REPORT.md`（含 §18 首轮真实联调、§19 R1 Replanning、四条边界）、`V2_E04A_PROVIDER_GUIDE.md`、`V2_E04A_CHECKPOINT_PLAN.md` 均在 **C 类** |
| 业务代码是否被本轮改动 | ✗ **未改动**（本轮只更新报告与清单文档） |
| Git 状态 | ✗ 无任何写操作（HEAD 与文件计数与你看到的一致） |


## A. 生产代码（应纳入）（35 个）

- `recommendation/api/app.py` (modified)
- `recommendation/api/demo_routes.py` (modified)
- `recommendation/api/schemas.py` (modified)
- `recommendation/control/context.py` (modified)
- `recommendation/control/loop.py` (modified)
- `recommendation/demo/__init__.py` (modified)
- `recommendation/demo/agent_chat_adapter.py` (untracked)
- `recommendation/demo/agent_policy.py` (modified)
- `recommendation/demo/agent_schemas.py` (modified)
- `recommendation/demo/agent_service.py` (modified)
- `recommendation/demo/agent_trajectory.py` (modified)
- `recommendation/demo/explanation.py` (untracked)
- `recommendation/demo/feedback.py` (untracked)
- `recommendation/demo/llm_agent.py` (untracked)
- `recommendation/demo/llm_policy.py` (modified)
- `recommendation/demo/llm_runtime.py` (modified)
- `recommendation/demo/memory_ranking.py` (untracked)
- `recommendation/demo/observation.py` (untracked)
- `recommendation/demo/product_knowledge.py` (untracked)
- `recommendation/demo/replan_policy.py` (untracked)
- `recommendation/demo/runtime.py` (modified)
- `recommendation/demo/schemas.py` (modified) ← **事故恢复限制文件（§4）**
- `recommendation/demo/serialization.py` (modified)
- `recommendation/demo/sessions.py` (modified)
- `recommendation/demo/tool_selection.py` (untracked)
- `recommendation/inference/fusion.py` (modified)
- `recommendation/memory/extraction.py` (modified)
- `recommendation/memory/query_adapter.py` (untracked)
- `recommendation/memory/schemas.py` (modified)
- `recommendation/memory/service.py` (modified)
- `recommendation/preference_matching/matcher.py` (modified)
- `recommendation/preference_matching/schemas.py` (modified)
- `recommendation/web/app.js` (modified)
- `recommendation/web/fusion.js` (untracked)
- `recommendation/web/index.html` (modified)


## B. 回归测试（应纳入）（32 个）

- `tests/demo_fixture.py` (modified)
- `tests/test_agent_chat_adapter.py` (untracked)
- `tests/test_agent_control_plane.py` (modified)
- `tests/test_agent_state_isolation.py` (untracked)
- `tests/test_api.py` (modified)
- `tests/test_demo_api.py` (modified)
- `tests/test_demo_fusion_view.py` (untracked)
- `tests/test_demo_llm_agent_memory.py` (untracked)
- `tests/test_demo_llm_agent_memory_acceptance.py` (untracked)
- `tests/test_demo_llm_agent_mode.py` (untracked)
- `tests/test_demo_llm_mode.py` (modified)
- `tests/test_demo_memory_ranking.py` (untracked)
- `tests/test_demo_recommendation_trace.py` (modified)
- `tests/test_fusion_tiger.py` (untracked)
- `tests/test_http_tiger_surface.py` (untracked)
- `tests/test_llm_agent_availability_guard.py` (untracked)
- `tests/test_llm_selected_controller.py` (untracked)
- `tests/test_llm_tool_selection.py` (untracked)
- `tests/test_memory_bm25_retrieval.py` (untracked)
- `tests/test_memory_query_adapter.py` (untracked)
- `tests/test_multi_source_fusion_presentation.py` (untracked)
- `tests/test_product_knowledge_rag.py` (untracked)
- `tests/test_tiger_agent_wiring.py` (untracked)
- `tests/test_tiger_engine.py` (untracked)
- `tests/test_tiger_engine_real.py` (untracked)
- `tests/test_tiger_source_contract.py` (untracked)
- `tests/test_v2_bounded_replanning.py` (untracked)
- `tests/test_v2_narration_consistency.py` (untracked)
- `tests/test_v2_observation_replan_gate.py` (untracked)
- `tests/test_v2_phase4_joint_acceptance.py` (untracked)
- `tests/test_v2_product_feedback.py` (untracked)
- `tests/test_v2_schema_integrity.py` (untracked)


## C. 技术文档（应纳入）（20 个）

- `docs/USAGE.md` (modified)
- `docs/V2_E01A3B_REPORT.md` (untracked)
- `docs/V2_E01A3_HANDOFF.md` (untracked)
- `docs/V2_E01A_CLOSE_REPORT.md` (untracked)
- `docs/V2_E01A_STATE_FIX_REPORT.md` (untracked)
- `docs/V2_E01B1_REPORT.md` (untracked)
- `docs/V2_E01B2_REPORT.md` (untracked)
- `docs/V2_E02A_REPORT.md` (untracked)
- `docs/V2_E02A_V_REPORT.md` (untracked)
- `docs/V2_E02B_REPORT.md` (untracked)
- `docs/V2_E02C_NARRATION_FIX_REPORT.md` (untracked)
- `docs/V2_E02C_REPORT.md` (untracked)
- `docs/V2_E03A_REPORT.md` (untracked)
- `docs/V2_E03B_REPORT.md` (untracked)
- `docs/V2_E03C_REPORT.md` (untracked)
- `docs/V2_E03C_SCHEMA_INTEGRITY_REPORT.md` (untracked)
- `docs/V2_E03C_SCHEMA_INVENTORY.md` (untracked)
- `docs/V2_E04A_CHECKPOINT_PLAN.md` (untracked)
- `docs/V2_E04A_PROVIDER_GUIDE.md` (untracked)
- `docs/V2_E04A_REPORT.md` (untracked) ← E-04A 最终报告（PASS + 四条边界）


## D. 不应提交（工件/凭据/缓存/临时文件）（0 个）

（本类为空 —— 检查后未发现此类文件）



## E. 需人工确认（实验性文件）（7 个）

- `backends/tiger_public/src/tiger_public/serve_worker.py` (untracked) ← 外部 backend 源码（AGENTS.md §19；其 venv 被忽略）
- `docs/TIGER_SERVING.md` (untracked)
- `experiments/e04a_provider_smoke.py` (untracked) ← E-04A 实验脚本（含 usage() 修复、调用上限、dry-run 默认）
- `experiments/t03lite_tiger_increment.py` (untracked)
- `experiments/t03lite_tiger_increment_verify.py` (untracked)
- `recommendation/backends/tiger_engine.py` (untracked) ← TIGER 实验路径（生产目录内的实验后端）
- `recommendation/control/tiger_source.py` (modified)


---

## 1. 逐文件 `git add` 白名单（**不使用 `git add -A`**；每条命令 ≤5 个路径）

### A 类（生产代码，35 个）

```bash
git add \
  'recommendation/api/app.py' \
  'recommendation/api/demo_routes.py' \
  'recommendation/api/schemas.py' \
  'recommendation/control/context.py' \
  'recommendation/control/loop.py'

git add \
  'recommendation/demo/__init__.py' \
  'recommendation/demo/agent_policy.py' \
  'recommendation/demo/agent_schemas.py' \
  'recommendation/demo/agent_service.py' \
  'recommendation/demo/agent_trajectory.py'

git add \
  'recommendation/demo/llm_policy.py' \
  'recommendation/demo/llm_runtime.py' \
  'recommendation/demo/runtime.py' \
  'recommendation/demo/schemas.py' \
  'recommendation/demo/serialization.py'

git add \
  'recommendation/demo/sessions.py' \
  'recommendation/inference/fusion.py' \
  'recommendation/memory/extraction.py' \
  'recommendation/memory/schemas.py' \
  'recommendation/memory/service.py'

git add \
  'recommendation/preference_matching/matcher.py' \
  'recommendation/preference_matching/schemas.py' \
  'recommendation/web/app.js' \
  'recommendation/web/index.html' \
  'recommendation/demo/agent_chat_adapter.py'

git add \
  'recommendation/demo/explanation.py' \
  'recommendation/demo/feedback.py' \
  'recommendation/demo/llm_agent.py' \
  'recommendation/demo/memory_ranking.py' \
  'recommendation/demo/observation.py'

git add \
  'recommendation/demo/product_knowledge.py' \
  'recommendation/demo/replan_policy.py' \
  'recommendation/demo/tool_selection.py' \
  'recommendation/memory/query_adapter.py' \
  'recommendation/web/fusion.js'
```

### B 类（回归测试，32 个）

```bash
git add \
  'tests/demo_fixture.py' \
  'tests/test_agent_control_plane.py' \
  'tests/test_api.py' \
  'tests/test_demo_api.py' \
  'tests/test_demo_llm_mode.py'

git add \
  'tests/test_demo_recommendation_trace.py' \
  'tests/test_agent_chat_adapter.py' \
  'tests/test_agent_state_isolation.py' \
  'tests/test_demo_fusion_view.py' \
  'tests/test_demo_llm_agent_memory.py'

git add \
  'tests/test_demo_llm_agent_memory_acceptance.py' \
  'tests/test_demo_llm_agent_mode.py' \
  'tests/test_demo_memory_ranking.py' \
  'tests/test_fusion_tiger.py' \
  'tests/test_http_tiger_surface.py'

git add \
  'tests/test_llm_agent_availability_guard.py' \
  'tests/test_llm_selected_controller.py' \
  'tests/test_llm_tool_selection.py' \
  'tests/test_memory_bm25_retrieval.py' \
  'tests/test_memory_query_adapter.py'

git add \
  'tests/test_multi_source_fusion_presentation.py' \
  'tests/test_product_knowledge_rag.py' \
  'tests/test_tiger_agent_wiring.py' \
  'tests/test_tiger_engine.py' \
  'tests/test_tiger_engine_real.py'

git add \
  'tests/test_tiger_source_contract.py' \
  'tests/test_v2_bounded_replanning.py' \
  'tests/test_v2_narration_consistency.py' \
  'tests/test_v2_observation_replan_gate.py' \
  'tests/test_v2_phase4_joint_acceptance.py'

git add \
  'tests/test_v2_product_feedback.py' \
  'tests/test_v2_schema_integrity.py'
```

### C 类（技术文档，20 个）

```bash
git add \
  'docs/USAGE.md' \
  'docs/V2_E01A3B_REPORT.md' \
  'docs/V2_E01A3_HANDOFF.md' \
  'docs/V2_E01A_CLOSE_REPORT.md' \
  'docs/V2_E01A_STATE_FIX_REPORT.md'

git add \
  'docs/V2_E01B1_REPORT.md' \
  'docs/V2_E01B2_REPORT.md' \
  'docs/V2_E02A_REPORT.md' \
  'docs/V2_E02A_V_REPORT.md' \
  'docs/V2_E02B_REPORT.md'

git add \
  'docs/V2_E02C_NARRATION_FIX_REPORT.md' \
  'docs/V2_E02C_REPORT.md' \
  'docs/V2_E03A_REPORT.md' \
  'docs/V2_E03B_REPORT.md' \
  'docs/V2_E03C_REPORT.md'

git add \
  'docs/V2_E03C_SCHEMA_INTEGRITY_REPORT.md' \
  'docs/V2_E03C_SCHEMA_INVENTORY.md' \
  'docs/V2_E04A_CHECKPOINT_PLAN.md' \
  'docs/V2_E04A_PROVIDER_GUIDE.md' \
  'docs/V2_E04A_REPORT.md'
```

### E 类（实验性文件，7 个 —— **需你逐项确认后才加入**）

```bash
# 仅在你确认这些 TIGER / 外部 backend / E-04A 实验文件应当入库后执行
git add \
  'recommendation/control/tiger_source.py' \
  'backends/tiger_public/src/tiger_public/serve_worker.py' \
  'docs/TIGER_SERVING.md' \
  'experiments/e04a_provider_smoke.py' \
  'experiments/t03lite_tiger_increment.py'

git add \
  'experiments/t03lite_tiger_increment_verify.py' \
  'recommendation/backends/tiger_engine.py'
```

### D 类（排除项）

**0 个**：检查后未发现应排除的候选文件。
上面四类之和（35+32+20+7）= 94 ✓。

## 2. 敏感信息与大型工件核查（本轮对全部 94 个候选实际执行）

| 检查 | 方法 | 结果 |
|---|---|---|
| 凭据字面量 | 逐文件正则 `sk-<16+>` / `api_key=` / `Bearer <token>` / `password=` | **0 命中** ✓ |
| 工件扩展名 | `*.pt/*.pth/*.ckpt/*.safetensors/*.onnx/*.pkl/*.npy/*.npz/*.db/*.sqlite3/*.log/*.zip/*.parquet/*.bin` | **0 命中** ✓ |
| 大文件（>200 KB） | `stat().st_size` | **0 命中** ✓ |
| `.env` | `git check-ignore -v .env` | 已忽略 ✓（且未被修改：md5 `b504f32fb81445b4a10c24f012457592`，mtime 2026-10-01） |
| `__pycache__` / `.pytest_cache` / venv | `.gitignore` 覆盖 | 未出现在候选清单 ✓ |

## 3. 阶段成果完整性（E-01A…E-03C + E-04A）

| 阶段 | 生产代码（A） | 测试（B） | 报告（C） |
|---|---|---|---|
| E-01A / 3B / CLOSE / STATE-FIX | `control/context.py`、`control/loop.py`、`demo/agent_service.py`、`demo/agent_trajectory.py`、`demo/agent_schemas.py`、`api/app.py`、`api/demo_routes.py`、`api/schemas.py` | `test_agent_control_plane.py`、`test_llm_selected_controller.py`、`test_llm_agent_policy.py`、`test_demo_agent_api.py`、`test_control_stage2_candidates.py`、`test_constraint_eligibility.py`、`test_candidate_reference_boundary.py` | 4 份 E-01A 报告 |
| E-01B（多轮 Memory） | `memory/query_adapter.py`、`demo/serialization.py`、`demo/runtime.py` | `test_demo_llm_agent_memory.py`、`test_memory_query_adapter.py`、`test_memory_bm25_retrieval.py` | `V2_E01B1/2_REPORT.md` |
| E-02A / E-02A-V | `demo/memory_ranking.py`、`inference/fusion.py`、`web/*` | `test_demo_memory_ranking.py`、`test_multi_source_fusion_presentation.py`、`test_demo_fusion_view.py` | `V2_E02A*_REPORT.md` |
| E-02B（Product RAG） | `demo/product_knowledge.py`、`demo/explanation.py`、`demo/agent_chat_adapter.py` | `test_product_knowledge_rag.py` | `V2_E02B_REPORT.md` |
| E-02C（联合验收 + 叙述） | `demo/agent_chat_adapter.py` | `test_v2_phase4_joint_acceptance.py`、`test_v2_narration_consistency.py` | `V2_E02C*_REPORT.md` |
| E-03A / E-03B / E-03C | `demo/observation.py`、`demo/replan_policy.py`、`demo/agent_policy.py`、`demo/feedback.py`、`memory/*`、`preference_matching/*` | `test_v2_observation_replan_gate.py`、`test_v2_bounded_replanning.py`、`test_v2_product_feedback.py` | `V2_E03A/B/C_REPORT.md` |
| E-03C-SCHEMA-INTEGRITY | `demo/schemas.py` | `test_v2_schema_integrity.py` | `V2_E03C_SCHEMA_INTEGRITY_REPORT.md`、`V2_E03C_SCHEMA_INVENTORY.md` |
| **E-04A（本轮）** | —（未改业务代码） | —（未改测试） | `V2_E04A_REPORT.md`、`V2_E04A_PROVIDER_GUIDE.md`、`V2_E04A_CHECKPOINT_PLAN.md` + **E 类** `experiments/e04a_provider_smoke.py` |
| TIGER（阶段 4 / 外部 backend） | `demo/llm_policy.py`、`demo/tool_selection.py`、`recommendation/backends/tiger_engine.py`、`control/tiger_source.py` | 7 个 `test_tiger_*` / `test_fusion_tiger.py` / `test_http_tiger_surface.py` / `test_llm_agent_availability_guard.py` | `docs/TIGER_SERVING.md`（E 类） |

**结论**：每个阶段的代码/测试/报告在清单中均有对应条目 ✓；E-04A 的产出（脚本 + 3 份报告）已全部纳入 ✓。

## 4. `recommendation/demo/schemas.py` 恢复限制（必须随检查点写入版本说明）

> 事故前的未提交版本**无法逐字恢复**；当前版本由 HEAD 基线 + 未受损消费者 + 测试契约重建，并由
> 1070 项测试（含 12 项 Schema 完整性专项）验证**可观察契约兼容**（字段/类型/required/默认值/
> 序列化/OpenAPI 形状）。**不可证明**：docstring 文字、字段书写顺序、注释与 `description` 措辞。
> 证据：`/tmp/destroyed_schemas.py.txt`（md5 `b695c793…`，为**错误覆盖内容**）、
> `/tmp/e03c_schema_integrity/schemas.reconstructed.py`（md5 `ed088de8…`）。
> 详见 [`docs/V2_E03C_SCHEMA_INTEGRITY_REPORT.md`](V2_E03C_SCHEMA_INTEGRITY_REPORT.md)。**不得**标记为可信历史基线。

## 5. untracked 变化核对（58 → 59 → 60 → 62）

| 时点 | untracked | 新增 |
|---|---|---|
| E-03C-SCHEMA-INTEGRITY 结束 | 58 | — |
| 你看到的上一轮状态 | 59 | `experiments/e04a_provider_smoke.py`（E 类） |
| Provider 指南写入后 | 60 | `docs/V2_E04A_PROVIDER_GUIDE.md`（C 类） |
| 检查点审计 + 报告后 | 62 | `docs/V2_E04A_CHECKPOINT_PLAN.md`、`docs/V2_E04A_REPORT.md`（C 类） |

除上述 4 个新增外，其余文件与上一轮**完全一致** ✓（无删除、无重命名、无内容改动 ✓）。

## 6. 建议的提交划分与提交信息

```text
分支：v2/phase5-e03c-checkpoint
```

**建议拆 4 个提交**（便于 bisect / 回滚；若你想一次到位，也可单提交，但必须保留 §4 的限制说明）：

1. `checkpoint(v2): phases 1-3 code and tests` —— A/B 类中 E-01/E-02 相关部分
2. `checkpoint(v2): phase 4-5 features (E-03A/B/C)` —— E-03 相关 A/B 类
3. `checkpoint(v2): TIGER backend and serving surface` —— TIGER 相关 A/B/E 类
4. `checkpoint(v2): docs, reports and E-04A evidence` —— C 类全部 + `experiments/e04a_provider_smoke.py`

```text
提交信息（主提交）：
checkpoint(v2): phases 1-5 (E-01A..E-03C) + E-04A provider acceptance

- E-01A/B: agent control plane, multi-turn memory, M9 read/write ordering
- E-02A/AV/B/C: memory ranking, multi-source fusion, product RAG, narration fix
- E-03A/B/C: observation+gate, bounded replanning, product feedback
- E-03C-SCHEMA-INTEGRITY: rebuilt demo/schemas.py validated against the observable
  contract (1070 tests).  CAVEAT: the pre-incident uncommitted source could not be
  recovered byte for byte; docstrings, field order and description wording are
  reconstructed.  See docs/V2_E03C_SCHEMA_INTEGRITY_REPORT.md.
- E-04A: real DeepSeek acceptance, 4/4 authorized calls spent (tool selection,
  product explanation, replan source choice x2), no retries; deterministic initial
  plan in R1; no online recommendation-quality or production-stability claim.
- tests: 1070 passed offline (fake model clients); no online calls in the suite.
```

## 7. 正式 Git 写操作前的离线可校验备份（强烈建议先做）

```bash
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X
BACKUP_DIR=/mnt/d/IT/CODE/PROJECTION/agentrec/backups          # 仓库之外
mkdir -p "$BACKUP_DIR"; STAMP=$(date +%Y%m%dT%H%M%S)
tar --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='backends/*/.venv' -czf "$BACKUP_DIR/agentrecx_worktree_$STAMP.tar.gz" .
( cd "$BACKUP_DIR" && sha256sum "agentrecx_worktree_$STAMP.tar.gz" > "agentrecx_worktree_$STAMP.tar.gz.sha256" )
( cd "$BACKUP_DIR" && sha256sum -c "agentrecx_worktree_$STAMP.tar.gz.sha256" )
tar -tzf "$BACKUP_DIR/agentrecx_worktree_$STAMP.tar.gz" | grep -c "recommendation/demo/schemas.py"
```

## 8. 风险与执行顺序（**待你第二次明确确认后才执行**）

1. **先确认 E 类 7 个文件**（含 E-04A 实验脚本与 TIGER 相关文件）；`git restore --staged` 只能撤回**暂存** ✓。
2. **绝不使用 `git add -A`**（`backends/` 下可能含被忽略的 venv/工件 ✗）——本清单逐路径添加 ✓。
3. **执行顺序**：离线备份（§7）→ `git switch -c v2/phase5-e03c-checkpoint` → 按 §6 的 4 个提交分别
   `git add <白名单路径…>`（用 `git status --short` 复核暂存集合）→ `git commit` → `git show --stat HEAD` 复核 →
   `git push` **需再次单独授权** ✓。
4. 本轮**未执行**任何写操作 ✓；下一项正式任务 **E-04B（真实 SASRec/TIGER 资产联调与质量评测）**
   尚未开始，且 **E-04B 不调用 DeepSeek 在线 API** ✓。
