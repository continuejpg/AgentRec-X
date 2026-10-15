# AgentRec-X V2 — E-01A-3B 交接说明（Handoff）

> 本文件是**给下一个会话的唯一入口**。新会话没有历史上下文，请只依赖本文件 + 仓库本身。
> 写于 E-01A-3A 完成时。仓库状态、已验证结论、待办与命令都在这里。

## 0. 仓库状态（写此文件时实测）

```text
repo   : /mnt/d/IT/CODE/PROJECTION/agentrec/X
branch : master
HEAD   : faeff0b6f790d8c9d077d7d07ee26998aeecad2a
status : 12 modified + 22 untracked   ← 全部为合法未提交成果，必须保留
```

**硬性规则**：禁止 `git reset / checkout / restore / clean / stash`、禁止自动 `commit`/`stage`、
禁止重训模型、禁止安装新依赖、禁止在未授权时调用在线付费 LLM、禁止修改
`runs/m4_evidence/*`、`runs/t03lite_*`、`runs/step26_*`、`runs/step28_*` 等既有实验产物、
禁止改动 `/v1/recommend` 的默认行为。

## 1. 已交付并验证的能力（可直接复用，不要重写）

| 能力 | 位置 | 验证证据 |
|---|---|---|
| LLM 工具选择（有界、可信校验） | `recommendation/demo/tool_selection.py` | `tests/test_llm_tool_selection.py` **16 passed** |
| 选择 → 真实 Controller 执行 | `recommendation/demo/agent_service.py` 的 `run_selection()` / `_execute_plan()` / `_controller_for(use_memory=)` | `tests/test_llm_selected_controller.py` **12 passed**（单源/多源真实执行、冲突、失败、legacy run） |
| 执行结果 → ChatResponse 适配 | `recommendation/demo/agent_chat_adapter.py` | `tests/test_agent_chat_adapter.py` **9 passed** |
| Memory → BM25 query（旧路径） | `recommendation/memory/query_adapter.py` + `agent_policy` 的 `use_memory` 开关 | M-02A/M-02A-V 测试；`use_memory=False` 供新路径隔离 |

关键接口（不要改签名）：
```python
ToolSelectionPlanner(client).plan(*, user_request, allowed_sources, budget) -> SourceSelection
DemoAgentService.available_sources -> tuple[str, ...]        # 运行时派生，含 TIGER ready 门控
DemoAgentService.run_selection(selection, AgentRecommendRequest) -> AgentRecommendResponse
build_agent_chat_response(resp, *, session_id, turn_id, turn, message=None, ...) -> ChatResponse
agent_execution_digest(resp) -> dict                        # terminal/trajectory/grounded/timing/titles
```

## 2. 本轮（E-01A-3B）唯一目标

打通并**用执行型测试证明**这条链：

```text
POST /v1/demo/sessions/{id}/chat  decision_mode="llm_agent"
 → ToolSelectionPlanner.plan()  （allowed_sources 取 available_sources，budget 取控制器预算）
 → DemoAgentService.run_selection()
 → LoopController → ActionValidator → CandidatePlane → GroundingVerifier → CandidateLedger
 → build_agent_chat_response()
 → ChatResponse（真实商品 + 真实 provenance + 可观察执行证据）
```

## 3. 已知接线点（file:line 均已核实，直接改这些地方）

| 要改的地方 | 位置 | 说明 |
|---|---|---|
| 模式常量 | `recommendation/demo/schemas.py:69-70` | 新增 `DECISION_MODE_LLM_AGENT = "llm_agent"`；同步 `DecisionMode` 字面量类型 |
| 请求 Schema | `schemas.py:125-144` | `ChatRequest{message, k, decision_mode?}` —— **没有 `sources` 字段**，所以"请求体覆盖 LLM 选择"结构上不可能 |
| 模式解析 | `recommendation/api/demo_routes.py:326-338` | 合法集在 `:334`；新增分支 |
| 模式下发 | `demo_routes.py:396,405` | `decision-modes` 端点，需把新模式加入可用列表 |
| Chat 分支 | `demo_routes.py:495-522` | 现有形态：`state, run = runtime.llm_turn(...)` / `state = runtime.turn(...)` → `TurnDecision(...)` → `build_chat_response(...)` |
| 客户端注入 | `recommendation/demo/runtime.py:161-165, 295-305` | `use_llm_client(client)` 目前只保存了 `LLMTurnPlanner`；**需同时保存原始 client** 给 ToolSelectionPlanner 用 |
| AgentService 获取 | `runtime.py:348` `agent_service()` | 从这里取 `available_sources` 与 `run_selection` |
| 响应适配 | `recommendation/demo/agent_chat_adapter.py`（已交付） | 新路径**不要**走 `build_chat_response(state)`，除非你能无损构造 graph state（不建议） |

**建议**：新增 `runtime.llm_agent_turn(k, *, user_key, agent_input) -> ChatResponse`（或返回 `(ChatResponse, digest)`），
内部完成 planner → run_selection → adapter；`demo_routes` 只做分支与错误映射。
`route="direct"` 时**不调用** `run_selection`、不建空计划、复用现有 direct 回答能力。

## 4. 必须遵守的语义约束（违反即算缺陷）

1. `allowed_sources` 必须来自 `available_sources`（含 TIGER 未 ready 时**不在**集合中）；不得硬编码四源。
2. 失败/无候选时 `route` 与空列表必须如实；**不得**把 `completion_refused` 包装成成功；不得伪造 audit/provenance。
3. `AgentRecommendResponse` 不含 session/turn/memory ⇒ 由调用方显式传入，不许猜。
4. 新路径 Catalog Search 走 `use_memory=False`（当前请求优先）；**Legacy 默认行为不变**。
5. LLM 的 `reason` 只是工具选择说明，**不是**商品事实或回答全文。

## 5. 必须补跑的测试（上一轮起一直未跑）

```bash
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X
.venv/bin/python -m pytest tests/test_agent_control_plane.py tests/test_demo_agent_api.py -q
```
新会话请把**新 Chat 集成测试**与下列文件合并到**尽量少的 pytest 调用**里（WSL `/mnt/d` 导入极慢，
实测单次导入 real ≈135 s 而 user+sys 仅 ≈8 s）：

```bash
.venv/bin/python -m pytest \
  tests/test_agent_chat_adapter.py tests/test_llm_selected_controller.py \
  tests/test_llm_tool_selection.py tests/test_demo_api.py -q 2>&1 | tail -5
```

## 6. 环境事实与踩坑记录（省时间用）

* 测试**导入**是主要开销（`/mnt/d` 挂载盘 I/O），不是测试本身；用 `tail -N` 截断输出、`--tb=line`。
* 长任务放后台 job，别 busy-poll；**不要**把 `A && B &` 整条链后台化。
* 清理进程用 `ss -ltnp | grep -w <port>`；`pkill -f` 会自匹配杀掉自己的 shell。
* 真实 TIGER 服务：`AGENTRECX_TIGER=1 .venv/bin/python -m recommendation.api.app --port 8000`，
  冷启动 5–8 分钟；`/health` 的 `tiger.ready` 才是就绪判据。LLM 配置由 `./scripts/start_demo.sh` 自动加载 `.env`
  （裸 `python -m ...` **不会**加载）。
* 真实 LLM Provider **未授权调用**；本轮只用 Fake client 联调。

## 7. 验收标准（E-01A-3B 的 PASS 条件）

1. `decision_mode="llm_agent"` 可经真实 Chat 路由调用；
2. 单源与多源请求都返回**真实商品**（来自 Controller/Ledger）与 provenance；
3. `direct` 路由不启动推荐 Controller；
4. 不可用来源 / 非法或超预算计划 / 空结果 / 工具失败均安全可见，且不回退为固定四源；
5. 旧 `deterministic`、旧 `llm`、Agent Recommend、`/v1/recommend` 行为不变（有测试证据）；
6. 结束时实测 `git status --short` 与 HEAD，并确认既有未提交改动完整。
