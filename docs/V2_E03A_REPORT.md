# AgentRec-X V2 / E-03A — Observation & Replan Gate

> 前序：[`docs/V2_E02C_NARRATION_FIX_REPORT.md`](V2_E02C_NARRATION_FIX_REPORT.md)（第 4 阶段收尾）。
> 本轮进入第 5 阶段：**只做 Observation 与 Replan Decision Gate，不执行第二轮计划**。
> 结论：**PASS** —— 真实 Controller 执行 → 生成与本轮工具/候选/预算一致的可信 Observation →
> 形成合法、有边界、可验证的 Replan Decision → 当前推荐结果（商品、顺序、分数、provenance、
> 偏好证据、解释）逐项不变；旧模式与 Serving 不回归。

---

## 0. 一句话结论

```text
同一次真实运行（两来源，k=4）：
  observation = { executed_sources:[history, catalog_search], calls:{history:1, catalog_search:1},
                  outcomes:{history:candidates, catalog_search:candidates},
                  grounded:4, ungrounded:0, presented:4, requested_k:4, sufficient:true,
                  termination:finished/completed, remaining:tool 2 / step 3, unexecuted:[] }
  replan      = { continue:false, reason_code:"candidates_sufficient", eligible_sources:[],
                  decision_source:"deterministic_rules" }
同一 fixture 的可达边界（实测）：
  k=9, 单来源   → continue:true,  "candidates_below_k"（eligible=[catalog_search]，预算 3/4）
  k=9, 两来源   → continue:false, "sources_exhausted"
  k=9, tool=1   → continue:false, "budget_exhausted"（remaining_tool_calls=0，仍有未试来源）
  空结果来源    → outcomes{catalog_search:"empty"} → "blocked_by_termination"（completion_refused）
  工具真失败    → outcomes{history:"failed"}       → "blocked_by_termination"（completion_refused）
  未 grounded   → grounded=4, ungrounded=1, presented=4（被拒 identity 绝不进入候选/答案）
  direct 回合   → audit.replan = null（未运行 controller）
```

## 1. 开始时 HEAD 和工作树

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a
$ git status --short -uall   →  26 modified + 46 untracked
```

与上一轮报告结束状态一致。未 reset/checkout/restore/clean/stash，未删未跟踪文件，未装依赖，
未训练模型，未调用在线 Provider，未改 Serving `/v1/recommend`。

## 2. 当前控制器实际终态与审计字段

| 事实 | 来源（真实字段） |
|---|---|
| 终止原因（8 种，闭集） | `recommendation/control/schemas.py:485` `TerminationReason`：`completed` / `max_steps` / `max_tool_calls` / `invalid_action` / `completion_refused` / `no_available_action` / `execution_failed` / `awaiting_user`；响应中为 `terminal.{status,termination_reason,termination_detail,succeeded}`（`AgentTerminalView`） |
| 预算与实际消耗 | `ControlState.steps_remaining` / `tool_calls_remaining`（`schemas.py:540,545`，即 `max(0, limit-used)`）；响应中为 `terminal.{steps,tool_calls,retries,limits{max_steps,max_tool_calls,max_retries}}` |
| 每个工具执行后的状态/错误/候选数 | `TrajectoryRecorder`（`control/trajectory.py:65`）逐 step 记录；投影为 `TrajectorySummaryView.steps[*]`：`action` / `validated` / `observation_type` / `candidate_count` / `eligibility_state` / `refusal_codes`（`demo/agent_trajectory.py:53`）。每步的 observation 本体是 `CandidateSetObservation`（`control/schemas.py:626`：`source`/`status:ok\|empty\|failed`/`requested_k`/`returned_k`/`ungrounded_count`/…）或 capability 的 `RecommendationObservation` |
| 候选"有"与"足够" | `grounded`（`GroundedCandidateView`：grounded/ungrounded/multi_source 计数 + 逐候选 provenance）、`eligibility`（`AgentEligibilityView`：`evaluated` + `verified_eligible_count`）、`recommendations`（最终展示数）——三者分别回答"通过验证""通过硬约束""最终展示" |
| 已执行 / 未执行来源 | 已执行：轨迹 step 的 `action`（本部署的候选来源动作映射为 `history` / `catalog_search`）；未执行：`DemoAgentService.available_sources` 减去已执行 |
| 为什么 finish | 顺序策略 `DemoSourcePlanPolicy` 服务完计划内来源后提议 `finish`，由 `CompletionGuard` 接受（有候选时）或拒绝（无候选时 ⇒ `completion_refused`） |
| 可安全给 LLM 的候选信息 | 仅计数与来源名（现有 `PolicyContext`/`CandidateState` 就是"计数 + 不透明引用"，从不给 identity/score）；本轮沿用这一边界 |
| 可复用的 Observation 结构 | 已有：`Observation` 家族（逐 step，loop 内部）、`CandidateState`（策略可见计数）、`TrajectorySummaryView`（逐 step 摘要）、`GroundedCandidateView`、`AgentEligibilityView`、`AgentTerminalView`。**本轮不新建控制抽象**，而是把这些既有投影组合成"运行级"观察 |

未使用 `ChatResponse.message`、商品标题或任何模型自述来推断工具是否成功。

## 3. Observation 构造位置与信任边界

```text
controller 运行结束（candidates/RRF/ranking/RAG 全部已定）
  └─ DemoAgentService._execute_plan（agent_service.py:676-706）
       ├─ recommendations = build_recommendations_view(...)      ← 已是最终权威答案
       ├─ grounded / eligibility / terminal / trajectory views   ← 已是冻结投影
       ├─ build_observation(terminal=…, grounded=…, eligibility=…, trajectory=…,
       │                    plan=…, available_sources=…, requested_k=payload.k,
       │                    presented_count=len(recommendations))  observation.py:153
       └─ decide_replan(observation)                              observation.py:224
```

信任边界：观察**只**由"该次运行自己的投影记录 + 请求的 k + 部署公开的来源清单"构成；函数是纯函数
（只读、无副作用）。它不调用任何工具、不重跑 Planner、不改查询、不写 controller 状态、不新增循环。
给决策层的投影（`observation_context`，`observation.py:317`）只含计数、来源名、预算与 reason code：
**没有**商品身份/标题/分数/查询、**没有** `source_text`/`memory_id`/`user_key`、**没有**跨会话内容、
**没有**未授权来源、**没有**未 grounded 商品的事实性宣称、**没有**成功率或质量分。

## 4. 各字段来自哪个真实执行记录

| Observation 字段 | 真实来源 |
|---|---|
| `run_ref` | `terminal.run_id`（controller 生成的 run-local 标识） |
| `requested_k` | 请求的 `k` |
| `plan` | 本轮通过校验的来源计划（agent endpoint：请求声明；llm_agent：LLM 选中并经校验） |
| `executed_sources` / `source_calls` | 轨迹 step 的 `action`（仅候选产出动作计入；`select_source` 无法从投影得知其来源，按动作名如实上报而非猜测） |
| `source_outcomes` | 每步的 `observation_type` + `candidate_count` + `refusal_codes`：`failed`（有 refusal code 或 failure 观察）/ `empty`（候选观察且计数 0）/ `candidates`（计数 > 0） |
| `failed_sources` / `empty_sources` | 上者的分类派生 |
| `grounded_count` | `grounded.grounded_count`（Ledger 以 identity 为键 ⇒ 已跨来源去重） |
| `ungrounded_count` | `grounded.ungrounded_count`（被 GroundingVerifier 拒绝的 identity） |
| `multi_source_count` | `grounded.multi_source_count` |
| `presented_count` | `len(recommendations)`（最终权威答案的条数） |
| `candidates_sufficient` | `presented_count >= requested_k`（纯计数） |
| `termination_status` / `termination_reason` | `terminal.status` / `terminal.termination_reason` |
| `steps_used`/`steps_limit`/`tool_calls_used`/`tool_calls_limit`/`retries_used` | `terminal.steps`/`tool_calls`/`retries`/`limits` |
| `remaining_steps` / `remaining_tool_calls` / `has_further_budget` | `max(0, limit-used)`，与 `ControlState.*_remaining` 同式 |
| `eligibility_evaluated` / `eligible_candidate_count` | `eligibility.evaluated` / `eligibility.verified_eligible_count`（false = 未评估，绝不等于"全部通过"） |

## 5. Replan Decision Gate 的具体规则

首个命中即返回（`observation.py:224`，reason code 闭集 `observation.py:82`）：

| # | 条件 | 结论 | 依据 |
|---|---|---|---|
| 1 | 没有任何来源被咨询（`executed_sources` 为空） | `not_a_recommendation`，不继续 | 无可 replan 的对象 |
| 2 | `termination_reason ∈ {invalid_action, completion_refused, no_available_action, execution_failed}` | `blocked_by_termination`，**不继续** | §4G：安全/非法/明确拒绝不得被绕过（即使预算与候选都充足） |
| 3 | `presented_count >= k` 且无失败来源 | `candidates_sufficient`，不继续 | §4A：答案已足，追加无意义（预算/来源的耗尽不该被当作它的理由） |
| 4 | `remaining_steps <= 0` 或 `remaining_tool_calls <= 0` | `budget_exhausted`，**不继续** | §4E |
| 5 | `unexecuted_sources` 为空 | `sources_exhausted`，**不继续** | §4F |
| 6 | 存在失败来源 | `tool_failed`，**建议继续** | §4D：换一个来源可能可行 |
| 7 | `presented_count == 0` | `no_candidates`，**建议继续** | §4B |
| 8 | 其它（`0 < presented < k`） | `candidates_below_k`，**建议继续** | §4C（说明句显式声明"只数数量、不评相关性"） |

边界：`eligible_sources` 恒为"部署提供 − 已执行"；`remaining_*` 与观察逐字一致；`decision_source` 恒为
`deterministic_rules`（本轮**未**接入模型判断，见 §12）。决策结构没有任何可以放宽预算或新增来源的字段
（测试断言其字段集合恰好为文档所列 7 个）。

## 6. 实测轨迹（全部来自真实 Demo API 执行）

| 场景 | 实测 |
|---|---|
| **有候选（k=4）** | executed=[history,catalog_search], calls={history:1,catalog_search:1}, outcomes 均 `candidates`, grounded=4, presented=4, sufficient=true, termination=finished/completed, remaining=2/3 ⇒ `candidates_sufficient`（不继续） |
| **有候选（k=2，grounded=3）** | presented=2=k ⇒ `candidates_sufficient` |
| **候选不足（k=9，单来源）** | presented=4 < 9，unexecuted=[catalog_search]，remaining=3/4 ⇒ `candidates_below_k`，eligible=[catalog_search]，说明句含"counts products, not relevance" |
| **工具真实失败** | `engine.error=RuntimeError` ⇒ outcomes={history:"failed"}, failed_sources=[history], empty_sources=[], grounded=0, termination=`completion_refused` ⇒ `blocked_by_termination` |
| **成功但零候选** | 请求无匹配 ⇒ outcomes={catalog_search:"empty"}, empty_sources=[catalog_search], failed_sources=[], grounded=0 ⇒ `blocked_by_termination` |
| **被 GroundingVerifier 拒绝** | 目录含 `j-ghost`（映射不认识）⇒ grounded=4, ungrounded=1, presented=4；`j-ghost` 不在任何卡片中；来源仍记为 `candidates`（"返回了但未通过验证"≠"没检索到"） ⇒ `candidates_sufficient` |
| **预算用尽** | `LoopLimits(max_steps=3,max_tool_calls=1)` + k=9 ⇒ remaining_tool_calls=0, has_further_budget=false, unexecuted=[catalog_search] ⇒ `budget_exhausted`（不继续） |
| **来源耗尽** | k=9 + 两来源 ⇒ unexecuted=[], budget 仍有 ⇒ `sources_exhausted`（不继续） |
| **拒绝/非法** | 无候选 run ⇒ `blocked_by_termination`；未授权来源的 LLM 计划 ⇒ HTTP 502 `llm_invalid_plan`（根本没有 observation 可 replan） |
| **direct 回合** | 未运行 controller ⇒ `audit.replan = null`（"没有决策"与"决策为否"不同） |

**边界如实标注**：§4B 的"零候选 + 预算 + 合法来源 ⇒ 建议追加"在本部署**无法经 Controller 到达**——
accepted `CompletionGuard` 对"无候选的推荐"直接拒绝（`completion_refused`），而本轮的 Gate 依 §4G
必须尊重该拒绝。该规则以规则级测试覆盖（构造观察 ⇒ `no_candidates`，continue=true），并在报告中
明确标注为"规则支持、当前控制器不可达"，而不是把缺失字段推测成成功。

## 7. 单来源与多来源对照

```text
单来源（history）：executed=[history], calls={history:1}, unexecuted=[catalog_search],
                   grounded=4（= 该来源自身顺序 [j-red,j-blue,j-black,j-plain]）, presented=4
多来源（history+catalog_search）：executed=[history,catalog_search], calls 各 1,
                   grounded=4（= 累计并集，Ledger 仅记 catalog_search）, presented=4,
                   answer=[j-red,j-black,j-blue,j-plain]（= 两来源 RRF 顺序）
```
即观察描述的是**本轮累计合法候选**（4 件、含仅由 capability 提出的商品），而不是最后一次工具快照
（Ledger 只有 catalog_search 的 2 条 sighting）。两者同时被断言，避免把架构边界写成"统一 Ledger 记账"。

## 8. 两个请求的状态隔离

```text
run1（history, k=4）：run_ref=A, calls={history:1}, unexecuted=[catalog_search], requested_k=4
run2（catalog_search, k=2）：run_ref=B≠A, calls={catalog_search:1}, unexecuted=[history], requested_k=2
```
四个字段两两不同，且 `replan.eligible_sources` 各自只列对方未用的来源 ⇒ 观察不跨请求串用。
（另有既有 `test_agent_state_isolation.py` 的 8 线程并发用例在全量回归中通过。）

## 9. Observation 前后最终商品与解释逐项不变的证据

| 证据 | 实测 |
|---|---|
| 观察是"派生"而非"另算" | `observation.presented_count == len(recommendations)`；`observation.grounded_count == grounded.grounded_count`；`steps_used == terminal.steps`；`tool_calls_used == terminal.tool_calls`（测试逐项断言） |
| 没有因为观察而多调用工具 | `sum(source_calls.values()) <= terminal.tool_calls`（候选调用是运行自身消耗的子集） |
| 观察不含商品泄漏 | 响应 `observation` 的 JSON 中不出现任何卡片的 `parent_asin` 或 `item_id` |
| 同输入同输出 | 同一 session 重复两次相同请求（含 Memory 重排轮）：商品顺序、`sasrec_score`、`fusion_sources`、偏好证据 `evidence`、RAG `reason` 逐项相同 |
| 全量回归 | 37 个文件 1011 passed / 0 failed（含 E-02A/E-02A-V/E-02B/E-02C/E-02C-NARRATION 的全部既有断言） |
| 其它路径 | deterministic / llm Chat 的 `audit.replan` 为 `null` 且旧字段不变；Serving `/v1/recommend` 与 provider replay 全绿 |

## 10. 修改文件与 file:line

| 文件:行 | 内容 |
|---|---|
| `recommendation/demo/observation.py`（**新增**） | `ACTION_SOURCES:64`、`BLOCKING_TERMINATIONS:77`、`REPLAN_REASON_CODES:82`、`DECISION_SOURCE_RULES:95`、`_step_facts:117`（逐 step → 每来源调用次数与三态结果）、`build_observation:153`、`decide_replan:224`（§5 规则表）、`observation_context:317`（LLM 侧受限投影） |
| `recommendation/demo/agent_schemas.py:295,369,683,690` | `AgentObservationView`、`ReplanDecisionView`；`AgentRecommendResponse.observation/.replan`（均可选） |
| `recommendation/demo/schemas.py:449,493` | `AuditView.replan`（可选）+ `ReplanAuditView`（chat 侧紧凑块） |
| `recommendation/demo/agent_service.py:676-706` | 在 `_execute_plan` 中把既有视图提升为局部变量、构建观察与决策并附到响应（纯读取，位于一切决策之后） |
| `recommendation/demo/agent_chat_adapter.py:248,350,448` | `_replan_audit`（`None` 表示"未运行 controller"）；`_audit_for(..., replan=)` |
| `tests/test_v2_observation_replan_gate.py`（**新增，16 项**） | §7 的 12 类执行型断言 + 规则级用例 |
| `tests/test_demo_memory_ranking.py:466-490` | 精确 `audit` 断言改为"旧字段逐项 + 单独断言 E-03A 块"（旧字段仍逐字节比较） |
| `docs/USAGE.md` | 记录观察与决策字段及其"只决策、不执行"的边界 |

未改动：Controller 状态机与验证规则、CandidatePlane / GroundingVerifier / CandidateLedger / RRF、
M10A/M10B、M9、RAG 检索与解释验证、`ChatResponse` 与 `AgentRecommendResponse` 既有字段语义、
deterministic / llm Chat、Agent Recommend、Serving。

## 11. 全部测试命令与通过/失败数

```bash
.venv/bin/python -m pytest tests/test_v2_observation_replan_gate.py \
  tests/test_v2_narration_consistency.py tests/test_v2_phase4_joint_acceptance.py \
  tests/test_product_knowledge_rag.py tests/test_demo_memory_ranking.py \
  tests/test_multi_source_fusion_presentation.py tests/test_demo_llm_agent_memory_acceptance.py \
  tests/test_demo_llm_agent_memory.py tests/test_agent_chat_adapter.py \
  tests/test_demo_llm_agent_mode.py tests/test_agent_state_isolation.py \
  tests/test_agent_control_plane.py tests/test_llm_tool_selection.py \
  tests/test_llm_selected_controller.py tests/test_demo_agent_api.py tests/test_demo_api.py \
  tests/test_demo_multiturn.py tests/test_demo_llm_mode.py \
  tests/test_demo_preference_pipeline_fixes.py tests/test_memory_query_adapter.py \
  tests/test_memory_bm25_retrieval.py tests/test_preference_matching.py \
  tests/test_preference_reranking.py tests/test_agent_reranking.py tests/test_agent_memory.py \
  tests/test_provider_record_replay.py tests/test_demo_control_plane.py \
  tests/test_demo_recommendation_trace.py tests/test_demo_sessions.py tests/test_demo_web.py \
  tests/test_demo_fusion_view.py tests/test_llm_agent_availability_guard.py \
  tests/test_control_stage2_candidates.py tests/test_constraint_eligibility.py \
  tests/test_candidate_reference_boundary.py tests/test_llm_agent_policy.py \
  tests/test_serving_fusion.py -q
```

```text
1011 passed in 117.86s        （37 个文件，0 failed；本轮新增 16 项）
```

开发过程中修正的 3 处**测试自身**假设（非产品缺陷，均已核实后改写）：单来源时答案保持该来源自身顺序
（不是融合顺序）；"同输入同输出"必须在同一 session 内重复（换 session 就没有同一份 Memory）；
以及一处真实产品缺陷（`_audit_for` 内引用未定义的 `response`）——该缺陷在本轮**自查**中被发现并当场修
正（`agent_chat_adapter.py:248/350/448` 改为显式参数），未进入最终交付。

## 12. 尚未实现的实际 Replanning 能力

1. **不执行第二轮**：`continue_recommended=True` 只是一个判断；本轮不调用任何额外来源、不重跑 Planner、
   不生成第二轮计划（§0/§8 明确要求）。执行侧属 E-03B。
2. **未接入模型判断**：`decision_source` 恒为 `deterministic_rules`。现有 `ModelClient`/Planner 完全可承载
   一个结构化决策（同样的 `ModelRequest` + JSON 校验 + `correction` 重试通道已存在），但本轮按"避免重复
   架构"的要求先完成确定性 Gate；`observation_context` 已把模型侧可见字段固定下来，供 E-03B 直接使用。
3. **无相关性评价**：`candidates_below_k` 只数数量；"少"不等于"不相关"，仓库当前没有可信的相关性评价器。
4. **§4B 的零候选建议经 Controller 不可达**（`completion_refused` 优先，且 Gate 必须尊重该拒绝）：规则已
   实现并被规则级测试覆盖，API 级可达性受 accepted CompletionGuard 约束，如实报告而不绕过。
5. `select_source` 的来源名不在轨迹投影中 ⇒ 观察按动作名上报（不猜测来源）；若要精确到来源，需要
   controller 只读暴露该动作的参数（本轮未做）。

## 13. 最终 HEAD、git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （未提交、未 reset）
$ git status --short -uall
开始：26 modified + 46 untracked
结束：26 modified + 48 untracked
```

本轮新增 2 个未跟踪文件：`recommendation/demo/observation.py`、`tests/test_v2_observation_replan_gate.py`
（本报告为第 3 个）。本轮改动的已跟踪文件：`recommendation/demo/agent_service.py`、
`recommendation/demo/agent_schemas.py`、`recommendation/demo/schemas.py`、`docs/USAGE.md`、
`tests/test_demo_memory_ranking.py`（均在既有 modified 集合内，故计数不变；`agent_chat_adapter.py` 为更早
轮次的未跟踪文件）。前序 TIGER / Demo / M9 / M10 / E-01 / E-02 成果与 13 份文档逐条确认仍在。

## 14. 最终结论

**PASS。** 完成标准逐项由真实执行型测试支持：

```text
真实 Controller 执行（capability + plane，真实轨迹/ledger/grounding/预算）
  → 生成与本轮工具、候选、预算一致的可信 Observation（13 类字段全部来自真实记录）
  → 形成合法、有边界的 Replan Decision（8 条规则、闭集 reason code、budget/source/safety 三重约束）
  → 不改变当前推荐结果（派生而非另算、不额外调用工具、同输入同输出、1011 项回归全绿）
```

同时：单来源与多来源观察均正确；两请求状态不串用；direct/refused/failed 分支不被误改；旧模式与 Serving
不回归；失败/空结果/被拒 identity 三态可区分；预算与来源耗尽、以及安全拒绝均**禁止**继续。
§12 列出的"实际第二轮执行、模型驱动决策、相关性评价"为下一里程碑范围，本轮未做亦未声称完成。
本轮到此停止，不进入 E-03B。
