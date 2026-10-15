# AgentRec-X V2 / E-01B-1 — 多轮 Chat Session 与 M9 Preference Memory 接线

> 前序：[`docs/V2_E01A_STATE_FIX_REPORT.md`](V2_E01A_STATE_FIX_REPORT.md)（跨 run 状态泄漏修复）、
> [`docs/V2_E01A_CLOSE_REPORT.md`](V2_E01A_CLOSE_REPORT.md)（偏好投影的边界）。
> 结论：**PASS** —— 同一 Session 的真实偏好更新 → M9 存储 → 下一轮 `llm_agent` 读取有效投影
> 已由执行型测试证明；不同 Session 不串用；旧 deterministic / llm Chat 与 E-01A 工具执行不回归。
> 本轮不新增 Memory 系统、不证明推荐质量提升，未进入 E-01B-2。

---

## 1. Preflight、HEAD、初始工作树状态

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a
$ git status --short | grep -c '^ M' ; grep -c '^??'
25 modified + 29 untracked          （与上轮结束一致，全部 TIGER / Demo / M-02A / E-01A 成果保留）
```

本轮开始前先读了 `recommendation/demo/runtime.py` 的 `llm_agent_turn`、
`recommendation/api/demo_routes.py` 的 chat 分支、`recommendation/demo/llm_agent.py`、
`recommendation/demo/tool_selection.py`、`recommendation/demo/agent_chat_adapter.py`、
M9 的 `PreferenceMemoryService` / `PreferenceStore` / `PreferenceMemorySnapshot` 与既有
`tests/test_demo_multiturn.py` 的 ADD / REPLACE / REMOVE 契约。未执行任何禁用的 git 操作。

## 2. 原有 Chat Memory 写入与新模式的实际调用关系（定向确认结论）

| 问题 | 事实（源码核对） |
|---|---|
| 1. deterministic / llm 如何写入 ADD / REMOVE / supersede？ | 两者都经由 accepted graph 的 memory 节点：`persist_memory_stage(state, memory_service, user_key)` → `PreferenceMemoryService.process_turn(user_key, user_message, turn_id)`。deterministic 用 `RuleBasedPreferenceExtractor`；`llm` 用同一份已验证 plan 构造的 `LLMPlanExtractor`（`runtime.llm_turn` 里 `run.memory_service(store)`）。supersede 由 M9 的 `_supersedes` 规则判定（同 kind/value 的显式替换），不是按 kind 推断。 |
| 2. 新模式的 direct 分支是否复用？ | **是**。`runtime.llm_agent_turn` 的 direct 分支调 `self.turn(...)`，即 accepted graph（按部署的 control plane），因此 load / extract / persist / report 全走既有路径。 |
| 3. 新模式的 recommend 分支是否读取 Memory？ | **修复前：不读**。`LLMAgentTurn.request()` 构造 `AgentRecommendRequest(user_key=None)`，`plan()` 也不传 `context`；只有响应面板做了一次 store 读取（`active_preferences` 是真值，`memory_update` 是硬编码空视图）。**本轮起：读 + 写**（见 §3、§4）。 |
| 4. Session ID 与 user_key 的绑定？ | `DemoSessionManager.create` 由 session UUID 派生：`user_key = f"{_USER_KEY_PREFIX}:{session_id}"`（`sessions.py:270`），**从不**取自 profile id；`delete()` 会 retire 该 namespace 且永不重发。chat 路由把它传给 `llm_agent_turn(user_key=session.user_key, ...)`（`demo_routes.py:621`）。 |
| 5. `user_key=None` 会不会导致跨 Session 共用？ | 会——但不是共用"记录"，而是共用**不带记忆的运行**：`run_selection` 以 `payload.user_key`（此前恒为 None）作为 controller 的 memory 开关，所以所有会话共享同一"无记忆"运行路径。本轮把**记忆身份**移到 session 层（runtime 用 `session.user_key` 读写），运行本身仍显式 memory-free，两者不再混淆。 |
| 6. `memory_update` / `active_preferences` 是真值吗？ | direct 分支：都是真值（来自 graph state 的 M9 写入结果与 store 快照）。recommend 分支（修复前）：`active_preferences` 是真值（真实 store 读取），`memory_update` 是硬编码 `MemoryUpdateView()`——因为该分支当时确实不写。**本轮起** recommend 分支的 `memory_update` 来自 M9 的 `TurnMemoryResult.update`，`active_preferences` 来自同一结果的 `active` 快照。 |

## 3. Session → M9 `user_key` 的可信绑定设计

```text
session_id (UUID4, 客户端可见的唯一句柄)
   └─ user_key = "demo-session:<session_id>"      ← 服务端派生，客户端无法指定，LLM 也看不到
        ├─ 读：runtime._active_preference_snapshot(user_key)          （每轮一次，先读后写）
        ├─ 写：runtime._process_turn_memory(user_key, message, turn_id)（每轮一次，键为服务端 turn_id）
        └─ 面板：M9 返回的 post-write 快照 active_preference_views(...)
   AgentRecommendRequest.user_key 仍为 None  →  推荐控制器保持 memory-free 运行
```

* **身份来源可信**：`user_key` 只由 session id 派生（`sessions.py:270`），表单/消息/模型都无法提供它；
  `ModelRequest` 里也不含任何 `user_key`（工具选择请求只带 `user_request`、`available_tools`、
  `tool_budget` 与受限偏好上下文）。
* **同 Session 一致**：读、写、面板、下一轮读取都用同一个 `session.user_key`。
* **不同 Session 隔离**：两个 session 的 id 不同 ⇒ namespace 不同；M9 的每个方法都以 `user_key` 为
  第一道过滤，跨 namespace 读取不可能发生（`tests/test_demo_llm_agent_memory.py` 的 C 场景实测）。
* **reset 语义不变**：`delete()` 仍 retire namespace（`sessions.py:302-320`，M9 行保留为不可达审计溯源）。
* **不改变 M9 语义**：ADD / REMOVE / supersede 仍由 M9 判定；本轮没有新解析器、没有新 Store、
  没有旁路写库；测试也不直接写 Store 冒充 Chat 更新。
* **边界（必须明说）**：这是**服务端管理的 Demo Session 身份绑定**，等价于"会话级隔离"，
  **不是生产鉴权**（任何人都能创建新会话，从而获得新 namespace），也**不是跨进程持久化**
  （默认 SQLite 文件在单进程内，`AGENTRECX_MEMORY_DB` 决定其位置）。

## 4. 修改文件、函数与 file:line

| 文件:行 | 函数 / 常量 | 改动 |
|---|---|---|
| `recommendation/demo/llm_agent.py:103` | `MAX_PLANNER_PREFERENCES = 10` | 交给 Planner 的偏好条数上限（有界投影） |
| `recommendation/demo/llm_agent.py:106-133` | `planner_preference_context(projection)` | 新增：把受限投影收窄为"工具决策所需"的结构化上下文（kind/polarity/value + 计数），空记忆返回 `{}`（不发 context） |
| `recommendation/demo/llm_agent.py:190,201,222,263` | `LLMAgentTurn.__init__(preference_context=)` / `.preference_context` / `plan()` | 上下文随 turn 携带，并通过既有的 `ToolSelectionPlanner.plan(context=...)` 通道下发；`plan()` 文档说明"结构化字段，非请求文本" |
| `recommendation/demo/llm_agent.py:40-77` | 模块文档 | 重写记忆章节：读为上下文、写由 runtime 每轮一次、运行仍 memory-free（附三条契约理由） |
| `recommendation/demo/tool_selection.py:59-62` | `_SYSTEM_PROMPT` 规则 8 | 明确"上下文里的偏好是背景，不是本轮请求；不得回述成请求、不得拼进请求文本" |
| `recommendation/demo/runtime.py:396-547` | `llm_agent_turn` | 新增步骤 1（读投影）与步骤 4（每轮一次写入），报表真值；direct 分支显式不重复写入 |
| `recommendation/demo/runtime.py:550-558` | `_active_preference_snapshot` | 无 namespace / 无 memory service 时返回 `None`（诚实缺省，不伪造空快照） |
| `recommendation/demo/runtime.py:560-577` | `_process_turn_memory` | 复用 `PreferenceMemoryService.process_turn`；非法身份返回 `None`；Store 失败原样抛出（不伪装成功） |
| `tests/test_demo_llm_agent_memory.py`（新增，16 项） | — | A–I 场景执行型测试 |
| `tests/test_demo_llm_agent_mode.py:765-806` | `test_a_recommendation_turn_writes_what_the_user_states_and_reports_it` | **契约更新**：原来断言"推荐回合不写记忆"，本轮改为"写入并如实上报，但运行仍 memory-free"（模块文档同步） |
| `docs/USAGE.md`（模式说明段） | — | 更新 `llm_agent` 的记忆行为描述 |

未改：`PreferenceMemoryService` / Store / Snapshot / `query_adapter.adapt_terms` / M9 的 ADD/REMOVE/
supersede 语义、`DemoSourcePlanPolicy`（`use_memory=False` 隔离不变）、`/v1/recommend`、
`agent_chat_adapter`（`memory_update` / `active_preferences` 入参本就存在）。

## 5. 两轮 ADD → 下一轮读取的真实执行轨迹

实测（真实 Chat 路由 + 真实 M9 + 真实 `ToolSelectionPlanner` 构造的 `ModelRequest`，
provider 为脚本化 client）：

```text
turn 1  session A  message="I prefer black."            模型路由 direct
        memory_update = {changed: true, added: [{kind: color, polarity: prefer, value: black}],
                         superseded: [], removed: [], skipped_duplicates: 0, ...}
        active_preferences（面板）= [{color, prefer, black}]
        agent.outcome = "direct"    agent.controller_executed = false
        store（真实 SQLite）= 1 条 ACTIVE 记录："black"（→ 证明只写一次，未重复提交）

turn 2  session A  message="Recommend something for me." 模型选择 ["history"]
        Planner 实际收到的 context：{"active_preferences": [{"kind": "color",
                                   "polarity": "prefer", "value": "black"}],
                                   "active_preference_count": 1}
        Plan 的 user_request = "Recommend something for me."（"black" 不在其中）
        memory_update = {changed: false, ...}（本轮未陈述任何偏好 → 如实为空）
        audit.ranked_with_preference_count = 0（运行仍 memory-free）
        active_preferences 面板 = [{color, prefer, black}]
```

对应测试：`test_a_stated_preference_is_written_and_reaches_the_next_turns_planner`（A 场景）；
另有 `test_a_preference_stated_on_a_recommendation_turn_is_written_once_and_reported`（推荐回合自身
陈述偏好：写入一次、`skipped_duplicates=1` 于重复轮次、`ranked_with_preference_count=0`）。

## 6. REMOVE / supersede 的实际状态变化

```text
supersede："I prefer black." → 面板 {color/prefer/black}
           "Actually, I prefer blue instead."
             memory_update = {changed: true, added: [blue], superseded: [black]}
             面板 = {color/prefer/blue}；store 中 black 变为 SUPERSEDED（保留审计溯源）
           下一轮 Planner context = [{"color","prefer","blue"}]   ← black 不再出现

remove：   "I don't care about color anymore."
             memory_update = {changed: true, removed: [<active 记录>]}
             面板 = []；store 中该条目变为 REMOVED
           下一轮 Planner context = None（无 context 字段）      ← 失效偏好不再进入
```

对应测试：`test_a_superseded_preference_stops_reaching_the_planner`、
`test_a_removed_preference_stops_reaching_the_planner`。

## 7. 两个不同 Session 的隔离测试

```text
session A：写入 "I prefer black."            → A 面板 {color/prefer/black}
session B：只发推荐请求                       → Planner context = None ／ B 面板 [] ／ B 的 store 为空
session B：写入 "I prefer blue."              → B 面板 {color/prefer/blue}
           复核 A 面板仍是 {color/prefer/black}（互不影响）
```

另外断言两个 session 的 `user_key` 不同（`harness.manager.get(...).user_key`）。
对应测试：`test_two_sessions_do_not_share_preferences`（C 场景）。

## 8. 传给 ToolSelectionPlanner 的真实安全投影示例

真实下发（A 场景 turn 2 的 `ModelRequest.context_payload`）：

```json
{
  "user_request": "Recommend something for me.",
  "available_tools": [
    {"name": "history", "purpose": "sequential recommendation from the shopper's own trusted purchase history"},
    {"name": "catalog_search", "purpose": "lexical full-catalogue search over normalized product text"}
  ],
  "tool_budget": 4,
  "context": {
    "active_preferences": [{"kind": "color", "polarity": "prefer", "value": "black"}],
    "active_preference_count": 1
  }
}
```

* **只有** kind / polarity / value（外加计数与上限）；**没有** `user_key`、`memory_id`、
  `source_text`、`turn_id`、时间戳，**没有** REMOVED / superseded 条目，**没有**完整 snapshot；
* 上限 `MAX_PLANNER_PREFERENCES = 10`，并同时给出 `active_preference_count`，被截断时不会被误读为全集
  （`test_the_planner_context_is_capped_and_counts_what_it_left_out`）；
* 当前请求与历史偏好是**两个字段**：`user_request` 是用户本轮原话，`context.active_preferences`
  是历史；历史值从不拼接进请求文本（`test_stored_preferences_never_enter_this_turns_search_query`
  同时断言 Planner 收到了偏好、而 ledger 的 `source_query` 只有本轮请求）；
* 无记忆时不发 `context` 字段（`test_no_memory_means_no_context_at_all`），因此无记忆部署的
  prompt 与本轮之前逐字一致；
* 该上下文确实参与决策：`test_the_disclosed_context_can_change_which_tool_is_chosen` 用一个"按上下文
  路由"的脚本化 client 证明，同一请求在有无该偏好的两种状态下，合法计划与执行轨迹随之改变
  （只证明通道可用，不主张质量提升——那属 E-01B-2）。

## 9. ChatResponse 的 `memory_update` / `active_preferences` 证据

| 情形 | 响应字段（实测） | 为什么是真值 |
|---|---|---|
| 推荐轮陈述了偏好 | `changed=true, added=[lightweight]`，`trace.memory_changes.changed=true` | 来自 M9 `TurnMemoryResult.update`（store 自己的写入摘要），不是模型的话 |
| 推荐轮没陈述任何偏好 | `changed=false`，`added=[]` | 同上，M9 摘要为空 |
| 重复陈述同一偏好 | `changed=false, skipped_duplicates=1` | M9 的去重计数 |
| direct 轮陈述偏好 | `changed=true, added=[...]` | 来自 graph state 的 M9 写入结果（既有路径） |
| 非法 / 缺失记忆身份 | `changed=false`，`active_preferences=[]`，`evidence.memory_write_attempted="none"` | runtime 显式跳过，未触碰 store（`test_a_blank_memory_identity_writes_nothing_and_pollutes_nothing`） |
| store 写入失败 | **整个 turn 5xx**，响应体不含任何 `changed` 字段；store 未变 | 不存在的写入绝不报成功（`test_a_failing_store_write_is_reported_as_a_failure_not_a_success`） |
| 计划被拒（未授权来源） | HTTP 502 `llm_invalid_plan`，store 未变 | 计划先于任何写入（`test_tool_execution_and_controller_isolation_do_not_regress`） |

另外两条"不越界"的证据：
* direct 轮写偏好不会把推荐控制器标记为执行过：`agent.outcome="direct"`、
  `agent.controller_executed=false`（E-01A-3B 既有测试 + A 场景断言）；
* 推荐轮的 `active_preferences` 是 store 的 post-write 快照，而 `audit.ranked_with_preference_count`
  仍为 0 —— 面板状态与"本轮排序是否用了偏好"是两个字段，各自如实。

## 10. 测试命令、通过/失败数量与旧模式回归

单次 pytest 调用（23 个文件：本轮新增 + 任务书点名的全部套件 + 相关内存/图/回放）：

```bash
.venv/bin/python -m pytest tests/test_demo_llm_agent_memory.py tests/test_demo_llm_agent_mode.py \
  tests/test_agent_state_isolation.py tests/test_agent_control_plane.py \
  tests/test_memory_query_adapter.py tests/test_memory_bm25_retrieval.py \
  tests/test_llm_selected_controller.py tests/test_llm_tool_selection.py \
  tests/test_agent_chat_adapter.py tests/test_demo_agent_api.py tests/test_demo_api.py \
  tests/test_demo_llm_mode.py tests/test_demo_recommendation_trace.py \
  tests/test_demo_control_plane.py tests/test_demo_multiturn.py \
  tests/test_demo_preference_pipeline_fixes.py tests/test_demo_sessions.py tests/test_demo_web.py \
  tests/test_demo_fusion_view.py tests/test_llm_agent_availability_guard.py \
  tests/test_agent_memory.py tests/test_agent_reranking.py tests/test_provider_record_replay.py -q
```

```text
662 passed in 690.18s      （最终一次全量复跑；本轮新增 16 项）
```

* **0 failed**；旧模式回归：`deterministic` 与 `llm` 两条 Chat 路径的记忆行为由
  `tests/test_demo_multiturn.py` / `tests/test_demo_llm_mode.py` 原样通过，
  本轮另加 `test_the_deterministic_chat_mode_keeps_its_memory_behaviour`（ADD → supersede →
  排序仍带偏好 `ranked_with_preference_count == 1`）与
  `test_the_plan_backed_chat_mode_keeps_its_memory_behaviour`；
* 归档回放 `tests/test_provider_record_replay.py` 全绿：已接受的 action 序列未变；
* **一处契约测试的显式更新**（不是放宽断言）：`tests/test_demo_llm_agent_mode.py` 原
  `test_a_recommendation_turn_changes_no_preference_memory` 断言"推荐回合不写记忆且 store 为空"，
  这正是本轮要改掉的旧边界；已重写为 `test_a_recommendation_turn_writes_what_the_user_states_and_reports_it`，
  新断言**更强**（store 里确实只有一条 ACTIVE 记录、摘要与面板一致、且运行仍 memory-free）。
  旧行为没有被固化为正确预期。

## 11. 未处理的持久化、鉴权与 Memory 相关问题（明确边界）

1. **不是生产鉴权**：隔离单位是"服务端管理的 Demo Session"。任何调用者都能新建会话从而获得新的
   namespace；会话创建本身不经验证。生产级身份与鉴权不在本轮范围（§8 明确排除）。
2. **不是跨进程持久化**：默认 M9 存储是进程内的 SQLite 文件（`AGENTRECX_MEMORY_DB` 决定位置）；
   多副本部署之间不会共享，也不会自动复制。reset 只 retire namespace，M9 行作为不可达审计溯源保留
   （accepted M9 语义，未改）。
3. **推荐排序尚未 memory-aware**：`llm_agent` 的推荐运行仍 memory-free
   （`audit.ranked_with_preference_count == 0`）。偏好只影响"选哪些工具"，不影响候选排序、
   不参与融合后重排——这是 E-01B-2 与后续阶段的交付。
4. **不做相关性筛选**：本轮把全部 ACTIVE 偏好（上限 10 条）交给 Planner，没有做"与当前请求相关"
   的筛选，也没有重新启用 Memory→BM25 拼接（§5 明确要求保持 `use_memory=False` 隔离）。
5. **偏好提取能力沿用旧规则**：recommend 轮用 `RuleBasedPreferenceExtractor`（deterministic 路径
   同一个），`llm` 模式的 `LLMPlanExtractor` 未被引入该路径；因此复杂的自然语言偏好（需要模型理解
   的）在 `llm_agent` 的推荐轮可能提取不到——本轮刻意不扩大解析能力。
6. **direct 轮的写入仍由 graph 负责**：因此两分支的写入实现不同源（graph 节点 vs runtime 调用），
   但都调用同一个 `PreferenceMemoryService.process_turn`，且都由服务端 turn_id 保证幂等；
   已用"store 中恰好一条记录"的断言钉住不重复写入。

## 12. 结束时实际 HEAD 与 git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （未提交、未 reset）

$ git status --short
开始：25 modified + 29 untracked
结束：25 modified + 31 untracked
```

* 新增 2 个未跟踪文件：`tests/test_demo_llm_agent_memory.py`、`docs/V2_E01B1_REPORT.md`；
* 其余都是既有 modified 文件上的原地编辑：`recommendation/demo/runtime.py`、
  `recommendation/demo/llm_agent.py`、`recommendation/demo/tool_selection.py`、
  `tests/test_demo_llm_agent_mode.py`、`docs/USAGE.md`、`docs/V2_E01A3B_REPORT.md`；
* 前序文件逐条确认仍在，无删除、无回退；未执行 `commit` / `add` / `reset` / `checkout` /
  `restore` / `clean` / `stash`；未安装依赖；未训练模型；未调用在线 LLM；
* 未改动 `/v1/recommend`（Serving）与 M9 的 ADD / REMOVE / supersede 语义。

## 13. 结论

**PASS。** 同一 Chat Session 内：

* 用户在任意轮次（direct 或 recommend）陈述的偏好由既有 M9 协议真实写入，响应报告 store 自己的
  摘要与 post-write 面板；
* 下一轮 `llm_agent` 把当前 ACTIVE 偏好作为**结构化、受限、有界**的上下文交给真实
  `ToolSelectionPlanner`（`user_request` 与 `active_preferences` 分离，历史值绝不拼进请求文本）；
* supersede / remove 之后，失效偏好不再进入 Planner 上下文；
* 两个 Session 的 namespace 完全隔离，且身份由服务端从 session id 派生、LLM 无从指定；
* 计划被拒、身份缺失、store 失败都不会产生假成功；
* 旧 deterministic / llm Chat、E-01A 的工具执行与请求级隔离、归档回放全部不回归（662 passed）。

第 11 节列出的边界（无生产鉴权、无跨进程持久化、排序尚未 memory-aware、未做相关性筛选、
提取能力沿用旧规则）是本轮有意保留的范围，不作为未完成项。本轮到此停止，不进入 E-01B-2。
