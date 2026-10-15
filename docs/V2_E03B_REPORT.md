# AgentRec-X V2 / E-03B — Bounded Replanning（真实执行闭环）

> 前序：[`docs/V2_E03A_REPORT.md`](V2_E03A_REPORT.md)（Observation 与确定性 Gate，1011 项回归）。
> 本轮实现**阶段 5 第一次真实执行的 Replanning**：首次来源不足或为零 → 可信观察 → 确定性 Gate
> → 模型合法选择一个未执行来源 → **在同一次运行、同一份预算内实际执行** → 合并本轮合法候选 →
> Memory Ranking → Product RAG → ChatResponse。
> 结论：**PASS**。

---

## 0. 一句话结论

```text
一次真实请求（k=4，首次来源 history 只给 1 件）：
  step0 recommend_from_history  obs=recommendation  cand=1   validated ✓
        └ observation(initial) = {candidates:1, consulted:[history], outcomes:{history:candidates},
                                 remaining:tool 3 / step 5}
        └ gate: 1 < 4, 有未试来源, 预算够 2 步, 无拒绝, 未 replan 过  ⇒ 允许
        └ model(受限上下文) → {"replan":true,"source":"catalog_search"}
  step1 search_catalog          obs=candidate_set   cand=4   validated ✓
  step2 finish                                                    validated ✓
  terminal = finished/completed, steps 3/6, tool_calls 2/4, retries 0
  卡片 = [k-hist, k-black, k-blue, k-red]（首轮候选保留 + 追加来源候选并入同一融合池）
  observation(final) = {candidates:4, consulted:[history,catalog_search], remaining:2/3}
  audit.replan = {status:"appended_candidates", chosen:"catalog_search", model_called:true,
                  initial_candidate_count:1, final_candidate_count:4}
零候选恢复（k=4，首次 catalog_search 无匹配）：
  step0 search_catalog obs=candidate_set cand=0 → gate 允许 → 模型选 history
  step1 recommend_from_history cand=1 → step2 finish
  terminal = finished/completed（**不是 completion_refused**），outcome=recommended，
  卡片 = [k-hist]，message 由卡片渲染
```

## 1. 初始 HEAD、git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a
$ git status --short -uall
开始：26 modified + 49 untracked      （与 E-03A 报告结束状态一致）
结束：26 modified + 51 untracked
```

未 reset/checkout/restore/clean/stash，未删未跟踪文件，未装依赖，未训练模型，未调用在线 Provider，
未改 `/v1/recommend`。修改前已阅读相关文件的现有 diff。

## 2. 当前 Controller 中真正可用的 Replan 执行切入点

审计结论（§2 的 9 个问题，逐条实测）：

| 问题 | 事实 |
|---|---|
| 1 工具观察结果在哪里收到 | `LoopController.step` 执行动作后写入 `TrajectoryRecorder` 与 `state`，并把**已最小化**的 observation 交回策略：`PolicyContext.last_observation`（`control/context.py:37`） |
| 2 下一步策略何时 FINISH | `DemoSourcePlanPolicy.choose`：计划内来源服务完毕 ⇒ 提议 `FINISH`（`agent_policy.py:203`） |
| 3 CompletionGuard 何时拒绝零候选 | `control/completion.py`：无候选的完成提议被拒绝 ⇒ `completion_refused`（终态，E-03A 已证实） |
| 4 预算如何扣减 | 同一次 run 的 `ControlState`：每个动作 1 step、每个工具动作 1 tool call；`steps_remaining/tool_calls_remaining` = `max(0, limit-used)`（`control/schemas.py:540,545`） |
| 5 策略能否安全改变待执行来源 | **能**：策略是控制器唯一询问"下一步做什么"的组件，`ActionProposal` 本身不含预算/工具身份（`control/schemas.py:302`），所有提议仍过 `ActionValidator` |
| 6 新来源如何再次经过验证 | 提议 → `ActionValidator` → `CandidatePlane`（`CandidateLedger` + `GroundingVerifier`）；本轮的追加来源与计划内来源走**完全相同的路径**（实测同一个 policy 的 `proposal_for_source` 构造动作与参数） |
| 7 追加候选如何进入累计融合池 | 与既有完全一致：`presentable_tool_result` 累计 → `reciprocal_rank_fusion` → `limit=k`；本轮**未改**该链路 |
| 8 如何避免重复执行首次来源 | 策略以"运行自身记录的 observation 里的 source"判定已执行来源（`_fold_observation`），已执行的来源不再作为 eligible、也不再被重复提议 |
| 9 哪些终止原因绝对不可恢复 | `invalid_action` / `completion_refused` / `no_available_action` / `execution_failed`（E-03A `BLOCKING_TERMINATIONS`）；另加"上一次提议被拒绝"⇒ 不再 replan |

**没有**采用 `Controller.run() → Controller.run()` 的外层重试 ✗。使用的是控制器既有扩展点——
**policy**：它在每个工具执行之后被调用，拿到真实 run 状态，因此追加决策发生在 `FINISH` 提议**之前**，
也在 CompletionGuard 之前（这正是 §5 的关键）。Controller 状态机、`ActionValidator`、CompletionGuard、
CandidatePlane、GroundingVerifier、RRF、M10A/M10B、RAG 均未修改。

## 3. 零候选为什么曾不可恢复，以及本轮如何合法处理

E-03A 已实测：零候选的推荐 run 由 accepted `CompletionGuard` 直接判为 `completion_refused`；终态 Gate
按 §4G 契约**不能**绕过该终态。因此本轮不去改终态，而是把决策点前移到**提议之前**：

```text
旧路径：来源返回 0 → 策略再提议同一来源（直到工具预算耗尽）→ 提议 FINISH → Guard 拒绝 → completion_refused
新路径：来源返回 0 → 策略看到 observation(status=empty) → 判定该来源"已咨询"（不再重复）
        → 若 0 < k 且仍有未执行合法来源与 ≥2 步预算 → 模型选一个来源 → 同一 run 内执行
        → 有候选后再 FINISH → Guard 接受 → completed
```

实测（§9C 关键场景）：zero-candidate run 的终态是 `completed`、`outcome=recommended`、返回真实商品；
**不是**把 `completion_refused` 改名，也没有重启 controller 刷新预算。如果追加来源仍无候选
（§9D），run 维持既有拒绝语义（`completion_refused`，`status=appended_empty`，只做一次决策、不重试）。
真正的工具执行失败仍服从既有终止契约（`execution_failed` 属阻断集合）；非法动作、安全拒绝与未授权工具
不能借 Replan 放行（§9F/G 实测）。

## 4. 追加规划前后完整 Controller Action/Observation 轨迹（实测）

```text
step 0: action=recommend_from_history  obs=recommendation  cand=1  validated=True  refusals=[]
step 1: action=search_catalog          obs=candidate_set   cand=4  validated=True  refusals=[]
step 2: action=finish                  obs=None            cand=4  validated=True  refusals=[]
action_counts = {recommend_from_history:1, search_catalog:1, finish:1}
terminal = {status:finished, termination_reason:completed, succeeded:true,
            steps:3, tool_calls:2, retries:0,
            limits:{max_steps:6, max_tool_calls:4, max_retries:1}}
```

对比零候选恢复：`{search_catalog:1, recommend_from_history:1, finish:1}`，`steps=3, tool_calls=2`
（同一份预算内），终态 `completed`。

## 5. 首次 Observation、Replan Decision、最终 Observation

三类状态彼此可区分（§7）。运行时记录（agent 响应 `replan_execution`，`agent_schemas.py:395,415`）：

```text
status=appended_candidates | model_called=True | chosen_source=catalog_search | decision_step=1
initial = {candidates:1, sources_consulted:["history"],
           source_outcomes:{"history":"candidates"}, remaining_tool_calls:3, remaining_steps:5}
final   = {candidates:4, sources_consulted:["history","catalog_search"],
           source_outcomes:{"history":"candidates","catalog_search":"candidates"},
           remaining_tool_calls:2, remaining_steps:3}
```

* `initial` 在**决策时刻冻结**（`_RunState.initial_consulted/initial_outcomes`），因此"首次来源的结果"
  与"请求最终结果"永远不会被读成同一个数；
* `final` 的预算是**运行结束时**的真实剩余（来自 terminal，而非决策时）；
* E-03A 的终态 Observation 语义保持不变，仍然描述整个 run：`executed_sources=[history,catalog_search],
  grounded_count=4, presented_count=4, candidates_sufficient=true`；
* 聊天侧紧凑块：`audit.replan.{status, model_called, chosen_source, initial_candidate_count,
  final_candidate_count, reason_code, continue_recommended, eligible_sources, remaining_*, decision_source}`
  （`demo/schemas.py:493-520`）。`status` 闭集见 §7。

## 6. 模型真实收到的受限上下文及结构化输出

实测 payload（`build_replan_request_payload`，`replan_policy.py:245`）：

```json
{"user_request": "Widget",
 "stated_preferences": [],
 "observation": {"requested_k": 4, "verified_candidates": 1,
                 "consulted_sources": ["history"], "source_outcomes": {"history": "candidates"},
                 "remaining_tool_calls": 3, "remaining_steps": 5, "step_index": 1},
 "eligible_sources": [{"name": "catalog_search",
                       "purpose": "lexical full-catalogue search over normalized product text"}],
 "instructions": "Add at most one of the eligible sources, or decline. You cannot change budgets, tools, candidates or queries."}
```

结构化契约：`{"replan": true|false, "source": "<eligible name>|null", "reason": "..."}`，按
`parse_replan_answer`（`replan_policy.py:291`）解析并**逐项校验**：名字必须恰好在 `eligible_sources`
内，否则拒绝执行。全部测试断言该 payload 中不出现商品身份、标题、分数、`source_text`、`memory_id`、
`user_key` 或 history。模型**不能**：改预算、发明工具、选已执行来源、改候选、伪造成功、改 user_key、
绕 `ActionValidator`、要求第二次 Replan（§9F 逐项实测：越权/已执行/非法 JSON/重复请求/调用失败全部安全
终止，且不产生额外工具调用）。

## 7. 来源合法性与总预算校验

```text
eligible_sources = 本部署 SOURCE_PLANS ∩ 控制器当前 available_actions − 已咨询来源
                   （未注册/未就绪的来源，例如未就绪的 tiger，根本不在 available_actions 中）
Gate 顺序（首个命中，replan_policy.py:537）：
  replan_already_used            —— 每个请求最多一次
  run_not_running                —— 运行状态不是 running
  not_a_recommendation           —— 一次来源都没咨询过
  blocked_by_termination         —— 上一次提议被拒绝（安全边界不被绕过）
  candidates_sufficient          —— 已持有 ≥ k（有硬约束时只算 verified eligible）⇒ 不追加
  tool_budget_exhausted          —— 剩余工具预算 < 1
  sources_exhausted              —— 没有未执行的合法来源
  not_enough_steps_to_finish     —— 剩余步骤 < 2（一步只能终止，不能"执行+终止"）
```

总预算实测：追加来源与非追加来源共用同一 `ControlState`（`steps 3/6`、`tool_calls 2/4`、`retries 0`，
`limits` 与部署一致）；**没有任何路径创建第二个 controller 或刷新预算**。`max_tool_calls=1` 时不许追加
（`tool_budget_exhausted`）；`max_steps=2` 时不许追加（`not_enough_steps_to_finish`）；`max_steps=3`
恰好够（实测成功）。

## 8. 首次、追加和最终候选 ID 对照（实测）

| 时刻 | 候选 |
|---|---|
| 首次来源（history，逐行） | `k-hist`（1 件） |
| 追加来源（catalog_search，逐字检索 "Widget"） | `k-red`, `k-blue`, `k-black`, `k-hist`（4 件，其中 `k-hist` 也被命中） |
| 最终展示（k=4，融合 + Memory Ranking 后） | `k-hist, k-black, k-blue, k-red` |
| 零候选恢复 | 首次 `[]` → 追加 history → 最终 `[k-hist]` |
| 两次都零候选 | 最终 `[]`，`completion_refused`（无第三轮） |

首轮候选**没有**被丢弃，也没有用第二个来源的 `tool_result` 覆盖：最终 4 件 = 首轮 1 件 ∪ 追加 4 件
（去重后 4 件）。provenance/分数来自 ledger 自身记录，未伪造（`k-hist` 因被词法来源命中而带
`fusion_sources=["catalog_search"]`、`score_kind="bm25"`；`multi_source_count=0`，因为 capability 路径的
候选不是 ledger sighting——E-02A-V 的既有边界）。

## 9. RRF、Memory Ranking、RAG 与 ChatResponse 的最终行为

```text
RRF：未改公式/常数/并列规则；追加来源的候选进入同一累计池后再融合（`k` 在融合后生效）。
Memory Ranking：只在**最终**池上执行。实测：先声明 "I prefer black."，
  最终池 = {k-hist, k-red, k-blue, k-black} → 结果 [k-black, k-hist, k-blue, k-red]，
  audit.moved_count=2, reranking_applied=true, ranked_with_preferences=[color/prefer/black]，
  k-black: match_count=1, violation_count=0。
RAG：只解释最终展示的商品（audit.rag.products_examined = 展示数）。
ChatResponse：卡片数量/顺序/分数/证据/理由与上述一致；message 由卡片渲染，
  实测每行 "<n>. <parent_asin> (item <id>) - <reason>" 的顺序与 recommendations 逐项一致。
```

## 10. 非法计划、失败、零候选与预算用尽结果（实测）

| 场景 | 结果 |
|---|---|
| 候选足够（首个来源已 ≥ k 或两来源计划） | 不调用模型决策（`replan_calls=0`）、不追加、`status=not_triggered`（-sufficient）|
| 候选不足（1/4） | 模型被调用一次、追加来源真实执行、`status=appended_candidates` |
| 零候选 + 可用来源 | 同一 run 内恢复为 `completed`，返回真实商品 |
| 两次都零候选 | `completion_refused`、`status=appended_empty`、只决策一次、不重试来源 |
| 工具预算不足（`max_tool_calls=1`） | `model_called=false`、`status=gate_blocked`、工具调用 1 次 |
| 步骤预算不足（`max_steps=2`） | 同上；`max_steps=3` 恰好可行 |
| 模型要求不存在的来源 | `status=invalid_output`，无额外工具调用 |
| 模型要求已执行来源 / 未授权工具 | `invalid_output` / 被忽略（`eligible` 之外一律不执行）|
| 非法 JSON / 无答案 | `status=invalid_output`，无重试 |
| 调用失败/超时 | `status=model_failed`，无重试，答案不变 |
| 重复要求 Replan | 每请求只一次决策（实测 1 次调用、1 个追加来源）|
| 未授权计划（tiger） | HTTP 502 `llm_invalid_plan`，根本不产生 Replan |

## 11. 两 Session 与并发隔离

```text
session A 声明 "I prefer black." → 下一轮：audit.replan 的模型上下文 stated_preferences=[color/prefer/black]，
                                   最终顺序 k-black 第一；
session B 未声明             → stated_preferences=[]，顺序 k-hist 第一；
A 的活跃偏好 = {(color, prefer, black)}，B = {}（互不可见）。
并发：4 个 session 同时各发起一次推荐（共享一个 provider 客户端），四次都拿到
      {k-hist,k-red,k-blue,k-black} 且各发生 1 次 Replan 决策（共 4 次）；无异常、无串用。
```

## 12. Feature Flag 关闭时的逐项不变性

开关 `AGENTRECX_AGENT_REPLANNING`（`replan_policy.py:75`，**默认关闭**；`build_demo_runtime(replanning=...)`
可显式覆盖）。关闭时实测：**0 次额外模型调用**、**0 次额外工具调用**（`tool_calls` 比开启时少 1）、
答案与旧路径逐字一致（仅首次来源的 1 件）、`audit.replan.status=null`（没有执行记录）、
E-03A 终态观察与 Gate 契约不变；零候选时维持 `completion_refused`。
请求声明式 `/v1/demo/agent/recommend`、deterministic / llm Chat、Serving `/v1/recommend` 均不 Replan
（实测 `replan_execution=null`）。全部既有断言（含 E-02A/E-02A-V/E-02B/E-02C/E-03A）在默认关闭下通过。

## 13. 修改文件与 file:line

| 文件:行 | 内容 |
|---|---|
| `recommendation/demo/replan_policy.py`（**新增**） | `ENV_AGENT_REPLANNING:75`、`replanning_enabled:78`、`REPLAN_STATUSES:87`、Gate 码 `:99-107`、`ReplanRecord:135`、`normalise_source:192`、`build_replan_request_payload:245`、`parse_replan_answer:291`、`BoundedReplanPolicy:324`（`choose:414`、`_fold_observation:485`、`_eligible_sources:524`、`_gate:537`、`_decide_and_maybe_append:567`、`_ask_model:614`） |
| `recommendation/demo/agent_policy.py:291` | `DemoSourcePlanPolicy.proposal_for_source`（公开的"按来源构造提议"入口，参数与计划内来源完全同规则） |
| `recommendation/demo/agent_schemas.py:395,415,742` | `ReplanObservationView`、`AgentReplanView`、`AgentRecommendResponse.replan_execution`（可选） |
| `recommendation/demo/schemas.py:493-520` | 聊天侧 `ReplanAuditView` 扩展（`status/model_called/chosen_source/initial_candidate_count/final_candidate_count`） |
| `recommendation/demo/agent_service.py:255,500,555,646,671,916` | 开关读取；`run_selection/_execute_plan/_compose_controller/_build_controller` 传递 `replan_client` 与 `preference_snapshot`；LLM 选择路径上组装 `BoundedReplanPolicy`；`_replan_execution_view` 生成三类状态块 |
| `recommendation/demo/llm_agent.py:294,320` | `LLMAgentTurn.run(..., replan_client=)` |
| `recommendation/demo/runtime.py:186,554,767` | `replanning` 字段/`_replanning` 开关；仅 LLM 选择路径传入客户端；开关随 runtime 传给 service |
| `recommendation/demo/agent_chat_adapter.py:248,350,448` | 把执行记录并入 `audit.replan`（`None` 表示"未启用"） |
| `tests/test_v2_bounded_replanning.py`（**新增，24 项**） | §9 A–J 的执行型测试 + Gate 规则测试 |
| `tests/test_llm_selected_controller.py:120-140`（未跟踪文件） | 测试替身接受并记录新增的 `replan_client` 关键字 |
| `docs/USAGE.md` | Bounded replanning 章节（开关、Gate 条件、模型契约、三类状态、关闭时的行为） |

未改动：`LoopController` 状态机/预算语义、`ActionValidator`、`CompletionGuard`、`CandidatePlane`、
`GroundingVerifier`、`CandidateLedger`、RRF、M10A/M10B、RAG、`ChatResponse` 既有字段语义、
deterministic / llm Chat、Agent Recommend、Serving、以及 E-03A 的终态 Observation 与 Gate 契约。

## 14. 测试命令及通过/失败数量

```bash
.venv/bin/python -m pytest tests/test_v2_bounded_replanning.py \
  tests/test_v2_observation_replan_gate.py tests/test_v2_narration_consistency.py \
  tests/test_v2_phase4_joint_acceptance.py tests/test_multi_source_fusion_presentation.py \
  tests/test_demo_memory_ranking.py tests/test_product_knowledge_rag.py \
  tests/test_demo_llm_agent_memory_acceptance.py tests/test_agent_state_isolation.py \
  tests/test_agent_control_plane.py tests/test_llm_selected_controller.py \
  tests/test_demo_llm_agent_mode.py tests/test_provider_record_replay.py \
  tests/test_demo_llm_agent_memory.py tests/test_agent_chat_adapter.py \
  tests/test_llm_tool_selection.py tests/test_demo_agent_api.py tests/test_demo_api.py \
  tests/test_demo_multiturn.py tests/test_demo_llm_mode.py \
  tests/test_demo_preference_pipeline_fixes.py tests/test_memory_query_adapter.py \
  tests/test_memory_bm25_retrieval.py tests/test_preference_matching.py \
  tests/test_preference_reranking.py tests/test_agent_reranking.py tests/test_agent_memory.py \
  tests/test_demo_control_plane.py tests/test_demo_recommendation_trace.py \
  tests/test_demo_sessions.py tests/test_demo_web.py tests/test_demo_fusion_view.py \
  tests/test_llm_agent_availability_guard.py tests/test_control_stage2_candidates.py \
  tests/test_constraint_eligibility.py tests/test_candidate_reference_boundary.py \
  tests/test_llm_agent_policy.py tests/test_serving_fusion.py -q
```

```text
1035 passed in 83.71s        （38 个文件，0 failed；E-03A 基线 1011 + 本轮新增 24）
```

开发过程中自查修正的真实缺陷（均已修复并保留测试）：Gate 误判自己刚写的 `decided_at` 导致永不触发；
来源词表不一致（capability observation 用动作名、计划用 `history`）导致已咨询来源仍被当作 eligible；
runtime 未把开关传给 service 导致显式 `replanning=True` 无效；`initial` 快照在 run 结束后才取、与 final 混同。
测试侧修正：lexical 分词语料、会话复用（记忆）、并发共享单个 provider 客户端、以及若干我原先写错的期望值
（融合顺序、ledger provenance）——产品行为均经实测确认正确后才改写断言，未放宽任何安全断言。

## 15. 未验证的真实 Provider 与大型资产情况

* 所有测试与实测均使用 **fake/Scripted ModelClient**（按 payload 形状作答），**未调用任何在线 LLM**；
  真实 provider 的时延、限流与输出分布未验证。
* 未加载 349 MB SASRec checkpoint / 300 MB 元数据真实工件，也未启动真实 6.21 GB item-item 或 TIGER
  后端；`two_tower`/`tiger` 作为可追加来源的路径只在"未注册 ⇒ 不在 eligible"这一侧被覆盖。
* 未做端到端浏览器验证；`/v1/recommend` 未改动、未重测其真实模型输出（仅回归 `test_serving_fusion.py`）。

## 16. 结束时 HEAD、git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （未提交、未 reset）
$ git status --short -uall      →  26 modified + 51 untracked
```

本轮新增未跟踪文件：`recommendation/demo/replan_policy.py`、`tests/test_v2_bounded_replanning.py`、
`docs/V2_E03B_REPORT.md`。
本轮改动的已跟踪文件（均在既有 modified 集合内，故 modified 计数仍为 26）：
`recommendation/demo/agent_policy.py`、`recommendation/demo/agent_schemas.py`、
`recommendation/demo/schemas.py`、`recommendation/demo/agent_service.py`、
`recommendation/demo/runtime.py`、`docs/USAGE.md`。
本轮改动的**未跟踪**文件（此前轮次新增、尚未纳入版本控制，`git status` 显示为 `??`）：
`recommendation/demo/llm_agent.py`、`recommendation/demo/agent_chat_adapter.py`、
`tests/test_llm_selected_controller.py`——三者分别属于更早的 E-01/E-02 轮次，本轮只做了加法修改。

## 17. 最终结论

**PASS。** §13 的 PASS 标准逐项由真实执行型测试支持：

```text
同一次推荐请求：首次来源得到不足/零个合法候选
  → 可信 Observation（来自该 run 自己的 observation/候选/预算，决策时刻冻结）
  → 确定性 Gate 允许（来源合法、预算足够两步、无拒绝、未用过第二次机会）
  → 模型合法选择未执行来源（受限上下文 + 结构化校验）
  → Controller 在原总预算内执行追加来源（同一 run、steps/tool_calls 共享、validator/plane/grounding 全走）
  → 合并本轮合法候选（首轮不丢、去重、k 在最终池生效）
  → Memory Ranking、RAG、ChatResponse 正确
```

且：最多一次 Replan（实测每请求 1 次决策、1 个追加来源）；不绕过安全拒绝；不刷新工具与步骤预算；
不伪造来源与候选（未 grounded 商品被拒绝、越权名字不执行）；旧路径与 Serving 不回归
（1035 passed）。本轮未做无限 ReAct、多 Agent、新模型、新 RRF、新 Memory、新评分器、MCP 与生产身份。
本轮到此停止，不进入 E-03C。
