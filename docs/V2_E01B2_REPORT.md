# AgentRec-X V2 / E-01B-2 — 多轮 Memory 驱动推荐端到端验收

> 前序：[`docs/V2_E01B1_REPORT.md`](V2_E01B1_REPORT.md)（Session↔M9 接线）、
> [`docs/V2_E01A_STATE_FIX_REPORT.md`](V2_E01A_STATE_FIX_REPORT.md)（请求级隔离）。
> 结论：**PASS** —— Memory 已读到 → 影响 Planner 决策 → 影响真实执行与最终候选，
> 三个证据层级在**同一个受控多轮场景**中全部成立，且撤销后可回到基线；跨 Session 隔离与旧路径
> 兼容性通过。本轮**未修改任何生产代码**（验收未暴露产品 Bug，见 §9）。

---

## 0. 一句话结论与边界

在一个受控夹具（两个来源的商品 ID **完全不相交**）中，同一条用户请求
`"Recommend a BasketWidget"` 在偏好更新前后返回了**完全不同的真实商品**：

```text
偏好更新前：plan=[history]         → 执行 recommend_from_history → 商品 [hist-alpha, hist-beta, hist-gamma]
偏好更新后：plan=[catalog_search]  → 执行 search_catalog        → 商品 [cat-delta, cat-epsilon]
撤销偏好后：plan=[history]         → 执行 recommend_from_history → 商品回到 [hist-alpha, hist-beta, hist-gamma]
```

**边界**：驱动 Planner 的是一个受控的 Fake ModelClient（规则：有 ACTIVE 偏好→查目录，否则→用历史）。
它证明的是**通道与传播**（Memory 被读到、进入合法决策、被执行、改变候选），
**不是**在线大模型的推荐效果，也不主张任何质量（Recall/NDCG）提升。
融合后的 Memory-aware 重排属阶段 4，本轮不做（§10 允许如实声明该边界）。

---

## 1. 开始时 HEAD、工作树状态

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a
$ git status --short | grep -c '^ M' ; grep -c '^??'
25 modified + 31 untracked        （与上轮结束一致，未执行任何禁用 git 操作）
```

## 2. 多轮 Chat 的真实执行调用链

```text
POST /v1/demo/sessions/{sid}/chat  {"message": …, "k": 3, "decision_mode": "llm_agent"}
 ├─ demo_routes.chat                        recommendation/api/demo_routes.py:580
 ├─ _resolve_decision_mode → 可用性探测       demo_routes.py:426
 ├─ runtime.llm_agent_turn(session.user_key) recommendation/demo/runtime.py:396
 │   ├─ ① 读：_active_preference_snapshot(user_key)               runtime.py:550
 │   │      → project_active_preferences(snapshot)                memory/query_adapter.py:117
 │   ├─ ② ToolSelectionPlanner.plan(user_request, tools, budget, context=…)
 │   │      → planner_preference_context(projection)              demo/llm_agent.py:106
 │   │      → 真实 ModelRequest（受控 client 应答）
 │   ├─ ③ 写：_process_turn_memory(user_key, message, turn_id)    runtime.py:560
 │   │      → PreferenceMemoryService.process_turn（M9，ADD/REMOVE/supersede）
 │   ├─ ④ route="recommend" → DemoAgentService.run_selection（use_memory=False）
 │   │      → LoopController → ActionValidator → CandidatePlane
 │   │      → GroundingVerifier → CandidateLedger
 │   └─ ⑤ route="direct"  → accepted graph（graph 自己 load/extract/persist）
 └─ build_agent_chat_response / build_chat_response（真值 memory_update + 面板 + agent 执行记录）
```

顺序即语义：**先读、后写**，所以本轮写入的偏好只影响**后续轮次**（§7 有专门测试）。

## 3. ADD → 推荐 → supersede → 推荐 → REMOVE → 推荐 的完整轨迹（实测）

```text
① 基线  "Recommend a BasketWidget"   ACTIVE=[]                     context=None
        plan=[history]  counts={recommend_from_history:1, finish:1}  ledger=[hist-alpha,hist-beta,hist-gamma]
        cards=[hist-alpha,hist-beta,hist-gamma]  grounded.sources_present=["history"]  memory_update.changed=false

② 陈述  "I prefer black."（路由 direct）
        memory_update={changed:true, added:[{color,prefer,black}]}   面板=[{color,prefer,black}]
        agent.outcome=direct  controller_executed=false  store 恰好 1 条 ACTIVE 记录

③ 同请求 "Recommend a BasketWidget"
        context={"active_preferences":[{color,prefer,black}],"active_preference_count":1}
        plan=[catalog_search]  counts={search_catalog:1, finish:1}
        ledger=[cat-delta,cat-epsilon]  cards=[cat-delta,cat-epsilon]  grounded.sources_present=["catalog_search"]
        provenance=[{source:catalog_search, score_kind:bm25, source_query:"Recommend a BasketWidget"}] ×2
        memory_update.changed=false（本轮未陈述新偏好）

④ 更正  "Actually, I prefer blue instead."
        memory_update={changed:true, added:[blue], superseded:[black]}  面板=[{color,prefer,blue}]

⑤ 同请求 plan=[catalog_search]  context 只含 blue（"black" 不出现在 context 里）
        counts={search_catalog:1, finish:1}  cards=[cat-delta,cat-epsilon]

⑥ 撤销  "I don't care about color anymore."
        memory_update={changed:true, removed:[{color,prefer,blue}]}     面板=[]

⑦ 同请求 context=None  plan=[history]  counts={recommend_from_history:1, finish:1}
        cards=[hist-alpha,hist-beta,hist-gamma]   ← 与 ① 完全一致：撤销回到基线
```

## 4. Memory 前后 Planner Context 对照（真实 `ModelRequest.context_payload`）

| 轮次 | ACTIVE Memory | Planner 实际收到的 context |
|---|---|---|
| ① 基线 | `[]` | `None`（无 `context` 字段） |
| ② 陈述轮 | 写前 `[]` → 写后 `[{color,prefer,black}]` | `None`（**读在写之前**：该轮看不到自己的陈述） |
| ③ 同请求 | `[{color,prefer,black}]` | `{"active_preferences": [{"kind":"color","polarity":"prefer","value":"black"}], "active_preference_count": 1}` |
| ④ 更正轮 | 写前 `[black]` → 写后 `[blue]` | `{... "value": "black"}`（同样是写前状态） |
| ⑤ 同请求 | `[{color,prefer,blue}]` | `{"active_preferences": [{"kind":"color","polarity":"prefer","value":"blue"}], "active_preference_count": 1}` |
| ⑥ 撤销轮 | 写前 `[blue]` → 写后 `[]` | `{... "value": "blue"}` |
| ⑦ 同请求 | `[]` | `None` |

`user_request` 在 ①③⑤⑦ 恒为 `"Recommend a BasketWidget"`（未被记忆改写）；
投影里没有 `user_key` / `memory_id` / `source_text` / `turn_id`，也没有 superseded/removed 条目。

## 5. selected_sources / Controller Action / 候选 ID / 最终商品 前后对照

§8 要求的表格（全部为实测值，同一 Session、同一请求文本、同一夹具与预算）：

| 字段 | 偏好更新前 | 偏好更新后 |
|---|---|---|
| 用户当前请求 | `Recommend a BasketWidget` | `Recommend a BasketWidget`（相同） |
| ACTIVE Memory | `[]` | `[{color, prefer, black}]` |
| Planner 实际上下文 | `None` | `{"active_preferences":[{"kind":"color","polarity":"prefer","value":"black"}],"active_preference_count":1}` |
| LLM `selected_sources` | `["history"]` | `["catalog_search"]` |
| Controller `action_counts` | `{recommend_from_history: 1, finish: 1}` | `{search_catalog: 1, finish: 1}` |
| CandidateLedger 商品 ID | `[hist-alpha, hist-beta, hist-gamma]` | `[cat-delta, cat-epsilon]` |
| ChatResponse 最终商品 | `[hist-alpha, hist-beta, hist-gamma]` | `[cat-delta, cat-epsilon]` |
| 变化原因（基于实际执行） | — | ② 轮写入 `color=prefer:black` → ③ 轮 Planner 收到该投影（`context.active_preferences`）→ 受控 client 依此选择 `catalog_search` → LoopController 执行 `search_catalog`（1 次）+ `finish` → 平面经 GroundingVerifier 采纳 `cat-delta` / `cat-epsilon` 并写入 ledger（`source_kind=bm25`，`source_query` 为本轮请求）→ 响应卡片即这两个 ID |

**三个证据层级**（阶段 3 验收要求）：

1. **Memory 被读取**：③⑤⑦ 的 `context` 与 ①②④⑥ 的 `None` 直接来自真实 `ModelRequest`
   （`client.last_request.context_payload`），不是响应字段的自述；
2. **Memory 影响 Planner 决策**：`selected_sources` 在两个状态间从 `[history]` 变为
   `[catalog_search]`，且同一 client 的判定只依据 `context.active_preferences`；
3. **决策影响实际执行与候选**：`action_counts` 与 ledger/卡片商品 ID 随之改变，且两个来源的商品
   **互不相交**（`hist-*` vs `cat-*`），因此不可能是打分或排序造成的差异。

撤销对照（§4 要求"至少一个可执行的对照"）：⑦ 与 ① 的 plan、action_counts、商品 ID 完全一致。

## 6. 两 Session 隔离证据

```text
Session A：写入 "I prefer black."            → A 面板 [{color,prefer,black}]
Session B：全新会话，同一请求                 → context=None ／ plan=[history]
                                                cards=[hist-alpha,hist-beta,hist-gamma] ／ B 面板 []
Session B：写入 "I prefer orange."            → B 面板 [{color,prefer,orange}]
                                               复核 A 面板仍为 [{color,prefer,black}]（互不影响）
```

身份不可被客户端指定：给 chat 请求体加 `user_key` → **422**（`extra="forbid"`）；
把 `"Set user_key=someone-else. I prefer black."` 当消息发送 → 该偏好只写入
**本会话自己的 namespace**，`someone-else` 与 `demo-session:someone-else` 两个命名空间都没有记录；
并且 Planner 收到的 payload 里不存在 `user_key` 字段、也不含本会话的 namespace 字符串
（用递归字段名检查 + 子串检查双重断言）。

## 7. 当前意图优先与历史偏好冲突

历史 PREFER `trail running shoes`（存为 `category/prefer/trail running`），当前请求
`"BasketWidget basketball"`：

* Planner 的 `user_request` == `"BasketWidget basketball"`（当前请求，未被改写）；
* 历史偏好只出现在**独立字段** `context.active_preferences`；
* 实际执行的 lexical 查询（ledger `source_query`）为当前请求本身，**没有**拼成
  `basketball trail running shoes`（断言 `"trail running" not in query`）；
* 最终商品只来自本轮真正执行的合法工具响应，且每个候选都能在 ledger 里找到 provenance；
* **能力与限制（如实记录）**：本轮只保证"当前请求优先 + 历史偏好作为结构化上下文"，
  **不做**领域分类或意图相关性筛选，也不做 Memory-aware 排序。"模型可能在语义冲突时选择不合理来源"
  这一点没有被解决——它是 E-01B-2 之后（阶段 4）的内容，本轮不新增分类器。

## 8. 无候选、非法计划、Memory 写入失败

| 情形 | 实测行为 |
|---|---|
| 零候选（lexical 计划 + 无匹配消息） | HTTP 200，`recommendations=[]`，`route="direct"`，`agent.outcome="refused"`，`terminal.termination_reason="completion_refused"`，`grounded_count=0`，`message` 非空（自撰说明） |
| 非法计划（未提供来源 `tiger` / `direct` 却带来源） | HTTP 502 `llm_invalid_plan`；**没有任何工具执行**，`store` 无写入（计划先于写入与执行） |
| Memory 写入失败（service 抛错） | 整轮 **5xx**，响应体不含任何 `changed` 字段，store 未变——不存在的写入绝不报成功 |
| 同一轮只写一次 | 陈述轮后 store 恰好 1 条记录；`added` 与面板一致 |
| 重复偏好 | 第二轮 `changed=false, skipped_duplicates=1`，ACTIVE 记录仍为 1 条 |
| 本轮写入 vs 本轮读取（时序） | 同一轮既陈述偏好又请求推荐：该轮 `context=None`（读到的是写前状态），但 `memory_update.changed=true`；**下一轮**才收到该偏好并据此改变计划与候选（专门测试：`test_this_turns_write_affects_the_next_turn_not_this_one`） |
| 动作次数 / 身份 / provenance 一致 | `action_counts` 与 trajectory 的 validated 步一致；卡片 ID ⊆ ledger grounded；provenance 的 `source`/`score_kind`/`source_query` 与执行的来源一致 |

## 9. 修改文件、函数及 file:line

**本轮未修改任何生产代码。** 验收没有暴露产品 Bug：E-01B-1 的接线（读投影 + 每轮一次写入 +
Session 身份绑定）在本轮的真实端到端场景中直接成立。唯一的代码变更是**新增验收测试**：

| 文件:行 | 内容 |
|---|---|
| `tests/test_demo_llm_agent_memory_acceptance.py:53-67` | 受控夹具：`HISTORY_ROWS`（历史源商品）与 `CATALOG_ROWS`（仅词法可命中），两者 ID 不相交 |
| `…:76-91` | `_DisjointEngine`：历史源返回自己的行，但 grounding 身份映射覆盖两个来源 |
| `…:94-106` | `_catalogue_index()`：两个来源的目录记录（否则 lexical 候选会被正确判定为 ungroundable） |
| `…:109-139` | `_MemoryAwareClient`：受控的"按记忆上下文路由"的 client（本测试的规则，非线上模型行为） |
| `…:142-166` | `_ScriptedClient`：陈述轮使用的固定脚本 client |
| `…:169-231` | 真实 App 组装：`build_demo_runtime` + `create_app` + 真实 SQLite store |
| `…:232-256` | `_field_names` / `_provenance` 断言辅助 |
| `…:258-355` | **场景 A**：`test_memory_changes_the_plan_the_execution_and_the_products`（三层级 + supersede + remove 全流程） |
| `…:356-386` | 场景 B：`test_only_the_current_active_value_is_disclosed_after_a_correction` |
| `…:387-461` | 场景 C：`test_a_second_session_neither_reads_nor_influences_the_first`、`test_the_client_cannot_choose_or_override_the_memory_identity` |
| `…:462-502` | 场景 D：`test_a_stored_preference_never_rewrites_the_current_request_or_its_query` |
| `…:503-576` | 场景 E：时序、重复、写入失败 |
| `…:577-612` | 非法计划、零候选 |

**验收期间我第一版测试自身的 5 处问题（诊断结论，非产品缺陷）**，如实记录以免被误读为产品改动：

1. 基线请求原写作 `"I want a BasketWidget"`：accepted 规则提取器把 `want` 视为偏好陈述
   （`recommendation/memory/extraction.py:181` 的 `_POSITIVE` 允许省略主语），于是基线轮自己写入了一条
   `free_form_constraint` 偏好，"无记忆基线"不成立。改为祈使句 `"Recommend a BasketWidget"`（无 desire 动词）
   后基线确实无记忆。**这是既有 M9 语义，不是 Bug**（同文件 `:33-46` 就记录了这类句式），但它意味着：
   任何 Demo 脚本若用 "I want X" 提问，会在同一轮写入偏好——演示时需知。
2. 断言 `"user_key" not in json.dumps(payload)` 被我自己注入的消息文本 `"Set user_key=…"` 触发；
   改为**结构化字段名检查**（递归收集 JSON key）＋本会话 namespace 子串检查。
3. Session B 面板为空：同 (1) 的连带效应，基线改写后成立，并额外断言"不含 A 的黑"。
4. 同上 (1)。
5. 同上 (1)（重复陈述那条消息里含 `I want`，改写后只剩真正的重复偏好，`skipped_duplicates=1`）。

未改：任何 `recommendation/**` 生产文件、任何既有测试断言、`/v1/recommend`、M9 语义、
`DemoSourcePlanPolicy(use_memory=False)`、CandidatePlane/Ledger/Validator。

## 10. 测试命令、通过与失败数

单次 pytest 调用（24 个文件：本轮验收 + E-01B-1/E-01A 全部相关套件 + 内存/图/归档回放）：

```bash
.venv/bin/python -m pytest tests/test_demo_llm_agent_memory_acceptance.py \
  tests/test_demo_llm_agent_memory.py tests/test_demo_llm_agent_mode.py \
  tests/test_llm_tool_selection.py tests/test_llm_selected_controller.py \
  tests/test_agent_chat_adapter.py tests/test_agent_state_isolation.py \
  tests/test_agent_control_plane.py tests/test_demo_agent_api.py tests/test_demo_multiturn.py \
  tests/test_demo_llm_mode.py tests/test_memory_query_adapter.py tests/test_memory_bm25_retrieval.py \
  tests/test_demo_api.py tests/test_demo_recommendation_trace.py tests/test_demo_control_plane.py \
  tests/test_demo_preference_pipeline_fixes.py tests/test_demo_sessions.py tests/test_demo_web.py \
  tests/test_demo_fusion_view.py tests/test_llm_agent_availability_guard.py \
  tests/test_agent_memory.py tests/test_agent_reranking.py tests/test_provider_record_replay.py -q
```

```text
672 passed in 627.40s        （本轮新增 10 项验收测试；全套 0 failed）
```

回归确认项：Legacy deterministic / llm Chat 不回归（`test_demo_multiturn.py`、
`test_demo_llm_mode.py`、`test_demo_preference_pipeline_fixes.py` 全绿）；Agent Recommend 不回归
（`test_demo_agent_api.py`）；请求级 Controller/Policy/Ledger 隔离不回归
（`test_agent_state_isolation.py`，含 8 线程并发用例）；Serving `/v1/recommend` 未被修改且测试通过
（`test_demo_api.py` 的 M6 兼容用例）；M9 ADD/REMOVE/supersede 语义不回归
（`test_agent_memory.py`、`test_demo_multiturn.py`）。

## 11. 尚未完成的真实 Provider / 模型资产联调

* **真实在线 Provider**：未调用（未授权）。本轮的模型决策全部来自受控 Fake Client；
  真实 provider 下"模型是否会依据这些偏好选择不同工具"**未经验证**，也不构成本轮结论。
  可复现步骤已在 `docs/USAGE.md` 的 "Real-provider smoke test (opt-in, manual)"。
* **真实 SASRec checkpoint / TIGER worker**：未加载（夹具为确定性轻量 engine 与合成目录）；
  TIGER 侧只覆盖"未 ready ⇒ 不进菜单"。
* **融合后的 Memory-aware 重排（阶段 4）**：未实现——本轮的 Memory 影响路径是
  **工具选择（哪些可信来源被咨询）**，不是打分或融合后排序；`audit.ranked_with_preference_count`
  在 `llm_agent` 推荐轮仍为 0（如实上报）。
* **质量指标**：未做大规模 Recall/NDCG 实验，本轮不主张任何质量提升。

## 12. 结束时实际 HEAD、git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （未提交、未 reset）

$ git status --short
开始：25 modified + 31 untracked
结束：25 modified + 33 untracked
```

* 新增 2 个未跟踪文件：`tests/test_demo_llm_agent_memory_acceptance.py`、`docs/V2_E01B2_REPORT.md`；
* **modified 计数未变（25）**，且没有任何生产文件被本轮修改；
* 前序文件逐条确认仍在（TIGER / Demo / M-02A / E-01A / E-01B-1 成果与 6 份文档），无删除、无回退；
* 未执行 `commit` / `add` / `reset` / `checkout` / `restore` / `clean` / `stash`；
  未安装依赖；未训练模型；未调用在线 LLM。

## 13. 最终结论

**PASS。** 完成标准要求的闭环——用户通过 Chat 更新偏好 → M9 保存 → 后续轮次 Planner 读取 →
合法工具决策发生可解释变化 → Controller 实际执行 → 最终可信推荐候选可观察地变化——已由真实
API 级执行型测试支持（§3–§5，三个证据层级齐备，且候选集合互不相交）；
撤销/更正、跨 Session 隔离、身份不可由客户端指定、非法计划与写入失败的真值上报、旧路径兼容性
均通过（§6–§8、§10）。阶段 3 的允许边界（工具选择级影响、非排序级、无生产鉴权、无真实 Provider 联调）
已在 §7 与 §11 明确记录。本轮到此停止，不进入阶段 4。
