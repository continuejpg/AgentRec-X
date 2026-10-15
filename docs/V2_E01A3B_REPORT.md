# AgentRec-X V2 / E-01A-3B — 真实 LLM Recommendation Agent 接入 Chat API（完成报告）

> 入口文档：[`docs/V2_E01A3_HANDOFF.md`](V2_E01A3_HANDOFF.md)。
> 使用者文档（已同步）：[`docs/USAGE.md`](USAGE.md)。
> 结论：**PASS** —— Chat → LLM 决策 → 合法来源计划 → Controller 执行 → 真实候选 → 可信
> ChatResponse 全链路有执行型测试证据；recommend / direct / 失败状态可区分；旧模式兼容。
> 1 项既有失败与 1 项既有行为观察单独列出（§8、§13）。

---

## 1. 工作树开始状态（实测）

```text
branch : master
HEAD   : faeff0b6f790d8c9d077d7d07ee26998aeecad2a      （= 上一阶段结束状态 faeff0b）
status : 19 modified + 26 untracked
```

其中 12 modified + 23 untracked 属于上一阶段（TIGER / Demo / M-02A / E-01A-1/2/3A）未提交成果，
7 modified + 3 untracked 属于 E-01A-3B 上一轮（Chat `decision_mode="llm_agent"` 接线）。
本轮**全部保留**，未执行 `reset / checkout / restore / clean / stash`，未 commit，
未安装依赖，未训练模型，未调用任何在线付费 LLM。

---

## 2. 实际 `llm_agent` Chat 调用链

```text
POST /v1/demo/sessions/{id}/chat   {"message": …, "k": …, "decision_mode": "llm_agent"}
  demo_routes.chat                              recommendation/api/demo_routes.py:580
  → _resolve_decision_mode                      demo_routes.py:426   （provider 不可用即 503，不降级）
  → session.switch_decision_mode("llm_agent")   recommendation/demo/sessions.py:130
  → runtime.llm_agent_turn(...)                 recommendation/demo/runtime.py:396
      ├─ agent_service()                        runtime.py:513      （控制器缓存、available_sources）
      ├─ tool_planner() → ToolSelectionPlanner   runtime.py:321 / demo/tool_selection.py:151
      │    plan(user_request, allowed_sources=<available_sources ∩ SOURCE_PLANS>, budget=tool_budget)
      │      → 可信校验（未提供 / 重复 / 超预算 / 自相矛盾 ⇒ ToolSelectionError，且什么都没跑）
      ├─ route="recommend":
      │    LLMAgentTurn.run()                   recommendation/demo/llm_agent.py:215
      │    → DemoAgentService.run_selection()   recommendation/demo/agent_service.py:540
      │    → LoopController → ActionValidator → CandidatePlane → GroundingVerifier → CandidateLedger
      │    → AgentRecommendResponse
      │    → build_agent_chat_response()        recommendation/demo/agent_chat_adapter.py:233
      └─ route="direct":
           不建计划、不建控制器；runtime.turn() 走既有 direct 能力
           → build_chat_response(..., agent=direct_agent_execution(...))  demo/serialization.py:542
  → ChatResponse（真实商品 + 每卡 provenance + trace 决策事实 + agent 执行记录）
```

---

## 3. 修改文件、函数及 file:line

### 3.1 本轮新增

| 文件 | 内容 |
|---|---|
| `recommendation/demo/llm_agent.py` | `offered_sources()`:78；`LLMAgentTurn`:94（`plan()`:171 / `request()`:189 / `run()`:215） |
| `tests/test_demo_llm_agent_mode.py` | 30 个 Chat API 执行型集成测试（A–G 场景） |

### 3.2 本轮修改

| 文件:行 | 函数 / 常量 | 改动 |
|---|---|---|
| `recommendation/demo/schemas.py:84,94` | `DECISION_MODE_LLM_AGENT` / `DecisionMode` | 新模式进入合法集合与字面量 |
| `recommendation/demo/schemas.py:595` | `ChatResponse.agent` | **新增可选、默认 None** 的执行字段（§7.1） |
| `recommendation/demo/agent_schemas.py:451,462` | `AGENT_OUTCOMES` / `AgentExecutionView` | 封闭 outcome 词表 + 复用既有安全视图的执行记录 |
| `recommendation/demo/agent_chat_adapter.py:85` | `agent_outcome()` | 由**运行自身记录**推导 outcome（direct/recommended/no_candidates/refused/failed） |
| `…:112 / :142` | `agent_execution_view()` / `direct_agent_execution()` | 执行记录组装（含 `action_counts`，只数 `validated=True` 的步） |
| `…:161` | `_card_for()` | §6C：`metadata` 填 **title-only** 视图，其余属性保持不可用；保留 `score_kind` |
| `…:233 / :300` | `build_agent_chat_response()` / `_trace_for()` | 接收 `planned_sources`；空文案回落为 outcome 自有文案；执行事实移出旧 trace |
| `…:78` | `OUTCOME_MESSAGES` | refused / failed / no_candidates 的自撰用户消息 |
| `recommendation/demo/serialization.py:542` | `build_chat_response(..., agent=None)` | 仅 direct 分支使用，其他调用方行为不变 |
| `recommendation/demo/runtime.py:287,302,321,339` | `llm_client()/llm_planner()/tool_planner()/use_llm_client()` | 两个 LLM 模式**共用一个 provider client**；注入后两者同时生效 |
| `recommendation/demo/runtime.py:396` | `llm_agent_turn()` | 最小编排入口，返回 `(ChatResponse, evidence)` |
| `recommendation/demo/llm_runtime.py:79` | `build_llm_client()` | 从 `build_llm_planner()` 拆出构造，行为不变 |
| `recommendation/demo/agent_service.py:307` | `tool_budget` | 只读暴露控制器自身 `max_tool_calls` |
| `recommendation/demo/agent_policy.py:86` | `SOURCE_PURPOSES` | 与 `SOURCE_PLANS` 同处的来源说明（可描述集合 == 可执行集合） |
| `recommendation/demo/tool_selection.py:162` | `ToolSelectionPlanner.client` | 只读 provider 身份（与 `LLMTurnPlanner` 对称） |
| `recommendation/api/demo_routes.py:352,369` | `_llm_agent_availability()` / `_policy_availability()` | 按模式各自探测可用性 |
| `…:402` | `_tool_selection_failure()` | 工具选择失败的 HTTP 映射（超时/未配置/拒绝计划/无工具） |
| `…:446,496,520` | `_mode_view()` / `decision_modes()` | 第三个模式条目，各自上报各自可用性 |
| `…:533` | `create_session()` | 新会话不允许以不可用的策略模式启动 |
| `…:580,618` | `chat()` | 新模式分支：只做分支与错误映射，执行证据打一行 DEBUG |
| `recommendation/web/app.js:788` | `buildCard()` | §6C：非 `sasrec_logit` 的分数按其 `score_kind` 标注，不再一律写成 SASRec |
| `docs/USAGE.md` | — | 三模式对照表、`agent` 字段与三种"空"的区分、真实 provider smoke 步骤 |

### 3.3 测试文件

| 文件 | 改动 |
|---|---|
| `tests/test_demo_llm_agent_mode.py` | 新增 A–G 场景（30 个测试） |
| `tests/test_agent_chat_adapter.py` | E-01A-3B 回归：5 种 outcome、三种"空"的可区分性、空文案回落、`action_counts` 真实次数、title-only 元数据、执行记录不含 prose（15 个测试） |
| `tests/demo_fixture.py` | `DemoEngine` 补 `has_parent_asin` / `parent_asin_to_item_id` / `item_id_to_parent_asin`（生产引擎本就有；缺它 `GroundingVerifier` 拒绝构造） |
| `tests/test_demo_llm_mode.py`、`tests/test_demo_recommendation_trace.py` | 顶层键集期望 **+`agent`**（授权的新增可选字段；两处均同时断言该模式为 `null`，属强化而非放宽） |

---

## 4. Chat Request / Response 示例（实测，脚本跑真实 HTTP 栈）

请求（场景 A）：

```json
POST /v1/demo/sessions/<id>/chat
{"message": "find me a widget", "k": 3, "decision_mode": "llm_agent"}
```

响应（截断；`message` 为控制器渲染文本）：

```json
{
 "turn": 1,
 "route": "recommend",
 "recommendations": [
  {"original_rank": 1, "parent_asin": "cand-red",   "item_id": 101, "sasrec_score": 0.9,
   "score_kind": "sasrec_logit", "fusion_sources": [], "metadata_status": "found",
   "metadata": {"title": "RedWidget", "store": null, "price_text": null, "categories": [],
                "details": []}},
  {"original_rank": 2, "parent_asin": "cand-blue",  "item_id": 102, "sasrec_score": 0.8, …},
  {"original_rank": 3, "parent_asin": "cand-black", "item_id": 103, "sasrec_score": 0.7, …}
 ],
 "audit": {"candidate_count": 3},
 "trace": {"route": "recommend", "candidate_count": 3, "decision_mode": "llm_agent",
           "provider": "deepseek", "model": "deepseek-chat", "proposed_route": "recommend"},
 "agent": {
  "outcome": "recommended",
  "planned_sources": ["history"],
  "controller_executed": true,
  "action_counts": {"recommend_from_history": 1, "finish": 1},
  "terminal": {"status": "finished", "termination_reason": "completed", "succeeded": true,
               "steps": 2, "tool_calls": 1, "retries": 0,
               "limits": {"max_steps": 6, "max_tool_calls": 4, "max_retries": 1}},
  "grounded": {"grounded_count": 3, "ungrounded_count": 0, "sources_present": ["history"], …}
 }
}
```

---

## 5. 单源 / 多源 / direct 三个执行型测试轨迹（实测）

测试位置：`tests/test_demo_llm_agent_mode.py`
（`test_a_single_source_plan_returns_the_controllers_products`、
`test_a_multi_source_plan_executes_exactly_the_sources_the_model_chose`、
`test_a_direct_plan_starts_no_recommendation_controller`）。

```text
场景 A  planner 选择 ["history"]，消息 "find me a widget"
        planned_sources = ["history"]
        trajectory.actions = ["recommend_from_history", "finish"]
        action_counts = {"recommend_from_history": 1, "finish": 1}
        outcome = recommended        route = recommend        引擎调用 = 1

场景 B  planner 选择 ["catalog_search", "history"]，消息 "I want a RedWidget"
        planned_sources = ["catalog_search", "history"]
        trajectory.actions = ["search_catalog", "recommend_from_history", "finish"]
        action_counts = {"search_catalog": 1, "recommend_from_history": 1, "finish": 1}
        （断言中不存在 "select_source"，即未选来源未被执行）
        grounded.sources_present = ["catalog_search"]，provenance 记录
            {source: catalog_search, source_rank: 1, source_score: 1.311258,
             score_kind: bm25, source_query: "I want a RedWidget", item_id: 101, step_index: 1}
        卡片：cand-red 带 fusion_sources=["catalog_search"] 且 score_kind="bm25"

场景 C  planner 选择 route="direct"，消息 "hello there"
        planned_sources = []，controller_executed = false
        terminal / trajectory / grounded / timing 均为 null
        outcome = direct，route = "direct"，recommendations = []
        compiled_controller_count = 0（未编译任何控制器），引擎调用 = 0
        message 与 deterministic 模式对同一消息的回答逐字相同
```

**等价性回归**（`test_the_new_mode_is_not_a_second_pipeline`）：同一
`(message, history, plan)` 下，Chat 模式的卡片、分数、`score_kind`、`fusion_sources`、文案与
`agent.trajectory` / `agent.grounded` 与既有 `POST /v1/demo/agent/recommend` **逐项相同**，
`terminal` 除 `run_id`（每次运行新生成）外相同。

---

## 6. 实际执行来源、次数、Ledger provenance

* **来源集合**：`available_sources`（运行时派生：`history` / `catalog_search`，加上部署启用的
  `similar_item` / `two_tower` / `tiger`）∩ 计划词表 `SOURCE_PLANS`。TIGER 未 ready 时
  `available_sources` 不含它，因此**既不进入 Prompt 也不可被选中**
  （`test_the_menu_and_budget_come_from_the_deployment`、`test_an_unoffered_source_is_refused_and_runs_nothing`）。
* **预算**：`service.tool_budget == LoopLimits.max_tool_calls`（默认 4），计划不可能大于它授权的那次运行；
  预算不足时 `ToolSelectionError(over_budget)`，一行都不执行
  （`test_the_planner_budget_is_the_controllers_bound`）。
* **次数**：`agent.action_counts` 只统计 `TrajectoryStepView.validated=True` 的步，因此
  "控制器在预算内合法重复调用同一来源"会被如实显示 —— 拒绝场景实测
  `{"search_catalog": 4, "finish": 2}`、`{"recommend_from_history": 4, "finish": 2}`
  （`tests/.../test_a_failing_tool_yields_no_fabricated_products`）。
* **Ledger provenance**：卡片级 `fusion_sources` + `agent.grounded.candidates[].provenance`
  （来源名、该来源自己的排名、自己的原始分数、`score_kind`、`source_query`、item_id、step_index）。

---

## 7. Agent 执行状态如何传输到 ChatResponse

### 7.1 新增字段（向后兼容、默认 None）

```python
# recommendation/demo/schemas.py:595
agent: AgentExecutionView | None = Field(default=None, ...)
```

`AgentExecutionView`（`agent_schemas.py:462`）是**既有安全执行摘要的类型化投影**，字段全部来自
已执行运行或既有公开视图，且 `extra="forbid"`：

```text
outcome             direct | recommended | no_candidates | refused | failed
planned_sources     被校验计划点名的来源（模型选的，不是默认列表）
controller_executed 推荐控制器是否真的运行过（direct 回合为 false）
action_counts       真正执行过的动作及次数（只数 validated 步）
terminal            AgentTerminalView：status / termination_reason / termination_detail /
                    succeeded / steps / tool_calls / retries / limits
trajectory          TrajectorySummaryView：动作种类、观察类型、拒绝码（无推理文本）
grounded            GroundedCandidateView：计数 + 每候选每来源 provenance
timing              AgentTimingView
```

* 不暴露：Prompt、模型原文、错误堆栈、完整 Memory、任何内部对象；
* 不塞进旧 `RecommendationTrace`（该视图的字段各有确定含义，且既有测试钉住其字段集）；
  `RecommendationTrace.grounding` 因此保持 `null`，执行事实只有 `agent` 一个家
  （`test_the_response_adds_no_field_and_no_model_prose` 同时钉住 trace 与 agent 的字段集与禁词）。

### 7.2 recommend / direct / 失败三态可区分（§6B）

`agent_outcome()`（`agent_chat_adapter.py:85`）只读运行自身记录，顺序为：
有候选 ⇒ `recommended`；计划不是 recommend ⇒ `direct`；terminal 未成功且
`termination_reason == "completion_refused"` ⇒ `refused`；未成功且未拒绝 ⇒ `failed`；
其余（成功但无可展示）⇒ `no_candidates`。

| 情况 | `route` | `recommendations` | `agent.outcome` | `controller_executed` | 文案 |
|---|---|---|---|---|---|
| LLM 主动 direct | `direct` | `[]` | `direct` | false | 既有直接回答 |
| recommend 但零候选 | `direct` | `[]` | `refused`（或 `no_candidates`） | true | 控制器若未渲染文本，则用 outcome 自撰文案 |
| recommend 但 completion_refused | `direct` | `[]` | `refused` | true | 同上 |
| recommend 成功 | `recommend` | 真实卡片 | `recommended` | true | 控制器渲染文本 |

回归测试：`tests/test_agent_chat_adapter.py::test_the_three_empty_looking_cases_are_actually_distinguishable`
（三种情况 route 与卡片完全相同，仅 `agent` 能区分，且三者文案互不相同）。

### 7.3 商品展示与分数语义（§6C）

* `metadata` 只在响应确实带 title 时填 **title-only** `ProductMetadataView`，其余属性保持 `None`
  （前端渲染为 unavailable），绝不编造 store / 价格 / 属性 / 评分；
  `metadata_status` 仍表示"可信目录是否认识该身份"。
  `tests/test_agent_chat_adapter.py::test_a_single_source_result_keeps_identity_order_and_provenance`
  由"metadata 必须为 None"**强化**为"只有 title 可以出现，其余字段必须为 None"。
* `score_kind` 一律保留（`sasrec_logit` / `bm25` / `rrf_rank_fusion` / `agent_controller_score`）。
* 前端 `app.js:788` 据此标注：非 `sasrec_logit` 显示 `Ranking score (<kind>)`，不再把融合分数或
  agent 分数写成 "Raw SASRec ranking score"（`test_the_browser_labels_a_score_by_its_kind_rather_than_calling_it_sasrec`）。

---

## 8. 空候选、`completion_refused` 与失败处理（实测）

```text
零候选（消息 "zzzqqq nothing matches"，计划 ["catalog_search"]）
  HTTP 200, route="direct", recommendations=[], trace.proposed_route="recommend"
  agent.outcome = "refused", controller_executed = true
  agent.action_counts = {"search_catalog": 4, "finish": 2}
  agent.terminal = {status: "failed", termination_reason: "completion_refused",
                    termination_detail: "candidates_not_grounded: …", succeeded: false,
                    steps: 5, tool_calls: 4, retries: 1}
  message = "The recommendation run could not be completed: its completion guard refused to
             finish it, so no products are being recommended. …"

工具失败（引擎抛错，计划 ["history"]）
  HTTP 200（同上，不是 5xx：控制器把工具失败记为运行失败）
  agent.outcome = "refused"（控制器自己的 reason），
  termination_detail = "last_execution_failed: …"，
  trajectory.refusals = ["execution_failed", …]，
  agent.action_counts = {"recommend_from_history": 4, "finish": 2}，recommendations = []
```

三处都不生成任何商品，`route` / 空列表如实，且**与主动 direct 可区分**（§7.2）。控制器自身的
`termination_reason` / `termination_detail` 原样传达，不重新解释、不包装成成功。

---

## 9. Memory 冲突隔离结果（场景 F，实测）

```text
先经 deterministic 路径写入偏好："I prefer trail running shoes."
  → active_preferences = [{"kind": "category", "polarity": "prefer", "value": "trail running"}]

再走 llm_agent（计划 ["catalog_search"]），消息 "RedWidget basketball"：
  ledger provenance.source_query = "RedWidget basketball"      ← 只有本轮请求
  memory_update.changed = false                                ← 本路径不写记忆
  active_preferences 仍显示 "trail running"（只读安全投影）
  写入前后 store 均无本回合新增条目
```

即：新模式的 Catalog Search 走 E-01A-2 的 `use_memory=False`，历史 ACTIVE PREFER **不会**被
追加进 BM25 查询；旧模式 M-02A 行为未改（`tests/test_llm_selected_controller.py` 的两条策略测试
继续通过）。本轮未新增任何跨轮记忆写入流程（属 E-01B）。
测试：`test_a_stored_preference_is_not_folded_into_this_turns_query`。

---

## 10. 旧 Chat / Agent API 测试结果

单次 pytest 调用（16 个文件，含 handoff §9 点名的全部回归）：

```text
$ .venv/bin/python -m pytest tests/test_demo_llm_agent_mode.py tests/test_agent_chat_adapter.py \
    tests/test_llm_tool_selection.py tests/test_llm_selected_controller.py \
    tests/test_agent_control_plane.py tests/test_demo_agent_api.py tests/test_demo_api.py \
    tests/test_demo_llm_mode.py tests/test_demo_control_plane.py tests/test_demo_multiturn.py \
    tests/test_demo_preference_pipeline_fixes.py tests/test_demo_recommendation_trace.py \
    tests/test_demo_sessions.py tests/test_demo_web.py tests/test_llm_agent_availability_guard.py \
    tests/test_demo_fusion_view.py -q
→ 1 failed, 416 passed   （唯一失败见 §13.1，改动前基线即失败）
```

旧行为证据：

* `deterministic`、`llm` 两种模式的响应与语义未变（键集仅多出恒为 `null` 的 `agent`）；
  `test_demo_llm_mode.py`、`test_demo_recommendation_trace.py`、`test_demo_api.py`、
  `test_demo_multiturn.py`、`test_demo_sessions.py` 全通过；
* deterministic 回合**不会**触发新模式（`agent_service_composed == False`，`agent == None`）；
* `/v1/demo/agent/recommend` 未改动（`test_demo_agent_api.py` 全通过，且见 §5 等价性回归）；
* `/v1/recommend` 未改动（`test_demo_api.py::test_m6_endpoints_are_unchanged_with_the_demo_enabled`）；
* 未新建第二套路由或 Chat Session 服务：`demo_routes.py` 仍只有原 7 个端点。

---

## 11. 真实 Provider / 模型 Smoke Test 的实际执行情况

| 验证层级 | 本轮实际执行 | 说明 |
|---|---|---|
| Fake LLM + 真实 Chat/Controller/Plane/Ledger/Runtime | ✅ 已执行 | §10 的 416 passed；模型侧一律 `ScriptedModelClient` / 超时替身 |
| 生产 provider **接线**（未注入 client 时构造真实适配器） | ✅ 已执行，**零网络调用** | `test_the_production_provider_path_composes_the_same_planner`：`runtime.tool_planner()` 构造出 `ToolSelectionPlanner`，其 client 与 `llm_planner()` 的 client 是**同一个**，`planner.calls == 0` |
| 真实 Provider **调用**（云端付费） | ❌ **未执行** | 未获授权。可复现步骤已写入 `docs/USAGE.md`（"Real-provider smoke test (opt-in, manual)"）：`.env` 配置 → `./scripts/start_demo.sh` → `GET /v1/demo/decision-modes` → 带 `decision_mode="llm_agent"` 的 chat，并列出应检查的字段 |
| 真实 SASRec / TIGER 模型资产 | ❌ **未执行** | 未加载 349 MB checkpoint、未启动 TIGER worker（冷启动 5–8 分钟）；TIGER 只覆盖"未 ready ⇒ 不在菜单里" |

---

## 12. 结束时实际 HEAD 与 git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （与开始时一致：未提交、未 reset）

$ git status --short
24 modified + 26 untracked   （共 50 条）
```

**modified 的构成（24）**：

* 既有 12 个（TIGER / Demo / M-02A / E-01A-1/2/3A 的未提交改动）**全部保留**：
  `recommendation/api/{app,demo_routes,schemas}.py`、`recommendation/control/{context,loop,tiger_source}.py`、
  `recommendation/demo/{agent_policy,agent_service}.py`、`recommendation/inference/fusion.py`、
  `recommendation/web/index.html`、`tests/{test_api,test_demo_api}.py`；
* 本轮使另外 12 个文件进入 modified：
  `docs/USAGE.md`、`recommendation/demo/{__init__,agent_schemas,llm_policy,llm_runtime,runtime,schemas,serialization}.py`、
  `recommendation/web/app.js`、`tests/{demo_fixture,test_demo_llm_mode,test_demo_recommendation_trace}.py`；
* 其中 3 个文件（`recommendation/api/demo_routes.py`、`recommendation/demo/agent_policy.py`、
  `recommendation/demo/agent_service.py`）在**既有未提交改动之上**追加了本里程碑的最小改动。

**untracked 的构成（26）**：既有 23 个（含 handoff 自身）全部保留；本轮新增
`recommendation/demo/llm_agent.py`、`tests/test_demo_llm_agent_mode.py`、`docs/V2_E01A3B_REPORT.md`。

未执行任何 `commit / add / reset / checkout / restore / clean / stash`；未改动
`runs/m4_evidence/*`、`runs/t03lite_*`、`runs/step26_*`、`runs/step28_*` 等既有实验产物；
未安装依赖；未训练模型；未调用未授权的在线 LLM。

---

## 13. 遗留问题与最终结论

### 13.1 既有失败：已在 E-01A-CLOSE 关闭（Case B：偏好投影过宽）

`tests/test_agent_control_plane.py::test_policy_context_has_no_route_to_trusted_state`

当时判断"需要显式拍板"，现已在 E-01A-CLOSE 轮确定真实根因并修复：M-02A 把**整份 M9 snapshot**
（含 `user_key`、逐条 `source_text`、`source_turn_id`、时间戳、已撤销条目）挂到了 `PolicyContext`，
既违反该模块自述的边界（"a count, never the entries"、"no ``user_key``"），也违反守卫测试显式的
`assert not hasattr(context, "preference_snapshot")`。修复方式是把交给策略的对象收窄为**值级投影**
`ActivePreferenceProjection`（kind / polarity / value / 不透明 entry id / 排序键 + status），
M-02A 的查询适配器与其排序键一字未改，检索行为逐值不变。

完整根因、数据边界、file:line 与回归证据见
[`docs/V2_E01A_CLOSE_REPORT.md`](V2_E01A_CLOSE_REPORT.md)。

### 13.2 既有行为观察：已定位（本节原始描述有误，以下为更正）

**原始（错误）描述**：该 service 共享的 `CatalogSearchSource` / 元数据索引"首次使用"导致候选集不同。

**更正后的真实原因**（E-01A-CLOSE 实测定位）：`DemoSourcePlanPolicy` 的"已服务来源"集合
`self._served`（`recommendation/demo/agent_policy.py:173`）**跨 run 泄漏**——`choose()` 只在 step 0
重置 `_finished_plan`（`:211-213`），从未清空 `_served`，尽管类注释（`:171-174`）与 `choose` 文档
（`:200-204`）都声明该状态不得跨请求泄漏。因此**同一个被缓存的 controller** 的第 2 次及以后运行，
在已有 grounded 候选后会把计划中的后续来源判定为"已服务"而跳过（`_proposal_for` 的
`grounded and source in self._served` 守卫，`:247`）。

最小复现（同一 controller、同一请求、连续三次）：

```text
run 1: actions=[recommend_from_history, search_catalog, finish]  presented=[cand-red]                         (1 卡)
run 2: actions=[recommend_from_history, finish]                  presented=[cand-red, cand-blue, cand-black]  (3 卡)
run 3: 同 run 2                                    policy._served 始终为 ['catalog_search', 'history']
```

影响范围：凡 `(k, user_key, plan)` 命中同一 controller 的重复请求都会变（`/v1/demo/agent/recommend`
的 user_key=None 请求，以及新 `llm_agent` 模式——它同样以 user_key=None 运行，故所有会话共享同一
controller）。与 catalog 索引、冷启动延迟、TIGER ready 状态无关（`CatalogSearchSource.__init__`
即时建索引；同 service 下新建 controller 仍复现 1 卡行为）。

**已在 E-01A-STATE-FIX 修复**：`DemoSourcePlanPolicy` 在 run 的第一步重置全部 run 级状态，
`DemoAgentService` 改为每个请求组合一个 controller（policy + candidate plane + ledger 均为请求级），
同一输入连续两次 run 的动作与候选完全一致，归档回放未变化。完整对照与证据见
[`docs/V2_E01A_STATE_FIX_REPORT.md`](V2_E01A_STATE_FIX_REPORT.md)。

### 13.3 明确未做（scope）

浏览器**未**新增第三种模式的单选控件（handoff §3 的 8 个接线点不含前端；`/decision-modes` 已如实
上报新模式，旧 UI 不受影响），新模式目前经 HTTP 调用；`similar_item` 仍不可作为计划来源
（既有 422 行为与测试保持不变），因此它虽然会被 `available_sources` 上报，但**不会**被交给模型；
`llm_agent` 的推荐回合当时不写偏好记忆（原因见 `recommendation/demo/llm_agent.py` 模块文档：
响应契约不携带记忆写入、请求契约不携带 session turn id），direct 回合保持既有记忆行为。
**E-01B-1 已贯通这一点**：写操作改由 runtime 承担（每轮一次，键为 session 自己的 namespace 与
服务端 turn id），响应报告 Store 的真实摘要；推荐回合本身仍以 memory-free 方式运行
（`audit.ranked_with_preference_count == 0`，历史偏好只作为 Planner 的结构化上下文）。
见 [`docs/V2_E01B1_REPORT.md`](V2_E01B1_REPORT.md)。

### 13.4 结论

**PASS。** Chat → LLM 决策 → 合法来源计划 → Controller 执行 → 真实候选 → 可信 ChatResponse 的
完整链路已由 30 个 Chat API 执行型集成测试 + 15 个适配器回归测试证明；recommend / direct /
零候选 / completion_refused / 工具失败均可区分且如实上报；旧 `deterministic`、旧 `llm`、
Agent Recommend、`/v1/recommend` 行为不变。真实 provider 与真实模型资产按 §11 明确标注为未执行。
本轮到此停止，不进入 E-01B。
