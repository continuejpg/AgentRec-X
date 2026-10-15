# AgentRec-X V2 / E-01A-STATE-FIX — 推荐 Policy 跨请求状态泄漏修复

> 前序：[`docs/V2_E01A_CLOSE_REPORT.md`](V2_E01A_CLOSE_REPORT.md)（§5 定位了本 bug 并给出最小复现，
> 当时按范围纪律未修）、[`docs/V2_E01A3B_REPORT.md`](V2_E01A3B_REPORT.md)（§13.2 已更正）。
> 结论：**PASS** —— 泄漏已按 run-local 语义修复，真实 Controller 执行型回归 13 项新增、
> 合并回归 646 passed / 0 failed，归档回放轨迹未发生变化。本轮不新增 Agent 能力，未进入 E-01B。

---

## 1. `_served` 泄漏的准确生命周期与根因

### 1.1 生命周期（逐处核对，含 file:line）

| 事件 | 位置 | 修复前 |
|---|---|---|
| 初始化 | `recommendation/demo/agent_policy.py:176`（`_served`）、`:177`（`_finished_plan`） | 随 policy 实例创建 |
| 读取（跳过判定） | `:247` `if context.candidate_state.grounded and source in self._served: return None` | 命中即"该来源已服务" |
| 写入 | `:250`（SEARCH_CATALOG）、`:263`（其余来源） | 只增不减 |
| `_finished_plan` 重置 | `:223`（`step_index == 0`） | **有**重置 |
| `_served` 重置 | —— | **从来没有**（即使类注释 `:171-174` 与 `choose` 文档都声明"每个 run 应重置"） |

### 1.2 复用链（为什么同一个 policy 会服务多个请求）

`DemoAgentService._controller_for`（修复前 `agent_service.py:438-465`）按
`(k, user_key, plan)` 缓存**整个 `LoopController`**，而 `_build_controller`（`:467`）在构造时创建
`DemoSourcePlanPolicy` 并绑定到 controller（`LoopController.__init__` 保存 `self._policy`，`:1719`）。
每个 run 只新建 `_LoopEngine`（`LoopController.new_engine`，`:1830-1866`），**policy 实例是被复用的**，
于是 `_served` 的写入跨 run 存活。

同一条复用链还带来第二个泄漏：`CandidatePlane` 及其 `CandidateLedger` 也在 controller 构造时创建，
随缓存跨请求存活。`build_grounded_view` 直接读 ledger 条目作为"本轮 grounded 全集"，所以第二个请求会
把第一个请求检索到的候选报成自己的（实测见 §5）。

### 1.3 触发条件与可观测后果

第 2 次运行中，当计划里**靠后**的来源在 `_served` 里仍有残留、且本轮已有 grounded 候选时，
`_proposal_for` 直接跳过它 → `_finished_plan=True` → FINISH。可观测后果有两个：

1. 计划来源被静默跳过（多源计划少跑一路）；
2. 呈现集合随之改变（第 1 次是 plane 采纳集，第 2 次退回 capability 集），两次相同输入结果不同。

### 1.4 审计同时确认的三件事（用于选择修复方案）

* **替换/新建 controller 的成本极低**：`LoopController.__init__` 只是赋值与协议校验
  （`loop.py:1681-1732`），其余协作者（engine、catalogue、词法索引、matcher、reranker、memory store）
  都由 service 持有并注入；且 loop **本来就每个 run** 编译一次驱动图（`_run_graph_driver` 内
  `build_loop_graph()`，`loop.py:1954-1958`），所以"每请求一个 controller"没有新增量级成本。
* **两个入口共用同一条链**：legacy `/v1/demo/agent/recommend` → `run()`；
  新 `llm_agent` → `run_selection()`；两者都走 `_execute_plan` → `_controller_for`。新路径传
  `user_key=None`（其记忆隔离决策），因此缓存键不含会话，**所有会话共享同一个 controller**。
* **并发没有被任何锁隔开**：`self._lock` 只保护缓存字典；`LoopController.invoke` 内没有锁；
  FastAPI 的同步端点在线程池中执行，聊天路径只按 session 串行（`runtime.sessions.turn`），
  不同会话可并发。因此"仅在第 0 步清空"只能修顺序场景，**不能**隔离并发共享同一 policy 的情形。

## 2. 修复方案与 file:line

采取**双层最小修复**，都是"让状态属于 run"而不是"让调用方记得重置"：

| 层 | 文件:行 | 改动 |
|---|---|---|
| **① Policy 自身 run-local** | `recommendation/demo/agent_policy.py:176-177`（字段注释）、`:194-226`（`choose` 文档 + 重置） | `step_index == 0`（run 的第一次决策）时同时清空 `_served` 与 `_finished_plan`：`self._served = set()` / `self._finished_plan = False`。文档说明了为什么放在这里：不依赖调用方如何组装 controller |
| **② 组合改为请求级隔离** | `recommendation/demo/agent_service.py:26-40`（模块文档）、`:236`（诊断计数器）、`:331`（`close`）、`:429-439`（`controllers_built`）、`:442-472`（`_compose_controller`）、`:475`（`_build_controller` 文档）、`:586`（调用点） | 删除 controller 缓存与 `max_cached_controllers` / `DEFAULT_MAX_CACHED_CONTROLLERS`（仅本文件引用），每个请求组合一个 controller：policy、CandidatePlane 与 CandidateLedger 全部请求级；`compiled_controller_count`（"当前缓存数"）替换为单调诊断 `controllers_built` |

**没有**改动：`LoopController`、`CandidatePlane`、`CandidateLedger`、`ActionValidator`、
`ToolSelectionPlanner`、`DemoSourcePlanPolicy` 的决策逻辑（只改状态重置时机）、
推荐模型/RRF/Memory Query Adapter/Fast Serving、工具选择策略。

## 3. 修复前后：同一 controller 连续两次 run 的动作与候选对照

白盒复现（同一 controller 实例、同一请求、连续三次；脚本用真实 `LoopController` + 真实
`CandidatePlane`）：

```text
修复前（E-01A-CLOSE 记录）
run 1: actions=[recommend_from_history, search_catalog, finish]  presented=[cand-red]                          (1 卡)
run 2: actions=[recommend_from_history, finish]                  presented=[cand-red, cand-blue, cand-black]   (3 卡)
run 3: 同 run 2

修复后（本轮实测）
run 1: actions=[recommend_from_history, search_catalog, finish]  presented=[cand-red]                          (1 卡)
run 2: actions=[recommend_from_history, search_catalog, finish]  presented=[cand-red]                          (1 卡)
run 3: actions=[recommend_from_history, search_catalog, finish]  presented=[cand-red]                          (1 卡)
```

Ledger 跨请求污染对照（同一 service，计划 `["catalog_search"]`，三次不同查询）：

```text
修复前：run1 grounded=[cand-red] / run2 grounded=[cand-red, cand-blue] / run3 grounded=[cand-red, cand-blue, cand-black]
修复后：run1 grounded=[cand-red] / run2 grounded=[cand-blue]           / run3 grounded=[cand-black]
```

## 4. Legacy Agent 与 `llm_agent` 的实际验证结果

HTTP 层（真实路由 + 真实 service，脚本用脚本化 model client，无网络）：

```text
Legacy /v1/demo/agent/recommend（计划 ["history","catalog_search"]，k=5/7，同一 payload 连发两次）
  修复前：run1 cards=1 steps=[history(4), catalog(1), finish] ；run2 cards=4 steps=[history(4), finish]
  修复后：run1 与 run2 均为 cards=1 steps=[history(4), catalog(1), finish]（k=5、k=7 皆然，fresh 与 repeat 一致）

新 llm_agent（同一计划、k=5、三个不同会话；修复前共享同一缓存 controller）
  修复前：会话 A/B/C 均 action_counts={recommend_from_history:1, finish:1} —— catalog_search 从未被咨询
  修复后：会话 A/B/C 均 action_counts={recommend_from_history:1, search_catalog:1, finish:1}，cards=1
```

执行型回归（`tests/test_agent_state_isolation.py`，13 项）里对应的用例：
`test_the_same_request_twice_executes_the_same_sources_both_times`、
`test_a_second_request_reports_only_its_own_grounded_candidates`、
`test_alternating_plans_do_not_contaminate_each_other`、
`test_the_legacy_entry_point_is_stable_across_requests`、
`test_the_llm_selected_entry_point_is_stable_across_requests`、
`test_a_failed_run_does_not_change_the_next_run`。

## 5. 同一 run 内合法重复调用来源仍然正常

修复**只**在 `step_index == 0` 重置，run 内的重试/重复语义未动：

* 工具失败时 run 内仍会重试到预算为止：`test_in_run_retries_still_happen_within_the_budget`
  断言 `recommend_from_history` 在一次 run 内被调用 >1 次（实测 4 次），且 `tool_calls ≤ max_tool_calls`；
* 一次成功 run 内每个来源只调用一次（不会被重置逻辑变成循环）：
  `test_a_plan_is_not_restarted_within_one_run` 断言各动作计数均为 1、`finish` 恰好 1 次；
* 预算语义不变（控制器掌管预算而非菜单，`loop.py:492-503`）：
  `test_budgets_still_bound_a_run_and_the_terminal_state_stays_truthful` 覆盖
  `max_tool_calls=1`（撤下工具动作，用已有候选完成，第二来源不被咨询）、
  `max_steps=1`（`termination_reason == "max_steps"`，`succeeded False`）、
  工具失败 + 小预算（有界停止且不呈现候选）；
* 计划推进/不重启的单元级语义：`test_the_run_reset_keeps_a_plan_advancing_and_restartable`
  （step0→来源1、step1→来源2、step2→FINISH、step3 仍 FINISH；下一 run 的 step0 重新从来源1开始）。

## 6. 并发共享情况与已验证的隔离边界

* **修复前的真实共享风险**：同一 `(k, user_key, plan)`（新模式下即所有会话与 legacy 无 user_key 请求）
  共用一个 controller → 一个 policy、一个 plane、一个 ledger；`self._lock` 不覆盖 run 执行，
  端点又是线程池中的同步函数，故并发 run 会交错读写 `_served`/`_finished_plan` 与 ledger。
  仅"第 0 步清空"无法隔离这种情况。
* **修复后的隔离边界**：`_compose_controller` 每请求新建 controller，因此 policy、plane、ledger
  都不再跨请求共享；进程级共享物只剩只读/内部同步的协作者（engine 的 id 映射、`MetadataIndex`、
  `CatalogSearchSource` 的只读索引、`PreferenceCandidateMatcher`、`PreferenceReranker`、
  memory store）——其中 `ProductEnricher` 无缓存（`rag/enrichment.py:170-179`），
  `CatalogSearchSource` 在 `__init__` 一次性建索引后只读（`control/catalog_search.py:159-172`）。
* **已执行的验证**：`test_concurrent_requests_do_not_share_run_state` —— 8 线程 × 2 次
  同一 service、同一计划，共享一个 `Barrier` 起跑，断言 16 个 run **全部**执行
  `{recommend_from_history:1, search_catalog:1, finish:1}` 且呈现同一候选集合。
* **边界声明（未宣称的）**：本轮的并发隔离是"不再共享可变 run 状态"，而不是给 service 引入
  并发框架；`DemoAgentService.close()` 仍可能在请求运行中释放一个自建的 TIGER worker，
  `_controllers_built` 计数器受锁保护但仅作诊断用。共享 store（preference memory）自身的并发语义
  由 M9 既有实现负责，本轮未改动也未重新验证——这一点按范围保持原样。

## 7. 回归命令与通过/失败数量

单次 pytest 调用（22 个文件：新增回归 + 任务书点名的全部套件 + 相关 Agent API/内存/图/回放）：

```bash
.venv/bin/python -m pytest tests/test_agent_state_isolation.py tests/test_agent_control_plane.py \
  tests/test_memory_query_adapter.py tests/test_memory_bm25_retrieval.py \
  tests/test_llm_selected_controller.py tests/test_llm_tool_selection.py \
  tests/test_agent_chat_adapter.py tests/test_demo_llm_agent_mode.py tests/test_demo_agent_api.py \
  tests/test_demo_api.py tests/test_demo_llm_mode.py tests/test_demo_recommendation_trace.py \
  tests/test_demo_control_plane.py tests/test_demo_multiturn.py \
  tests/test_demo_preference_pipeline_fixes.py tests/test_demo_sessions.py tests/test_demo_web.py \
  tests/test_demo_fusion_view.py tests/test_llm_agent_availability_guard.py \
  tests/test_agent_memory.py tests/test_agent_reranking.py tests/test_provider_record_replay.py -q
```

```text
646 passed in 424.26s
```

* **0 failed**；
* **归档回放未发生变化**：`tests/test_provider_record_replay.py` 全绿 —— 已接受的轨迹没有被本次修复改动
  （它们都是"第一个 run"的场景，修复对第一个 run 没有可观测影响）；
* 只更新了 3 处测试期望，全部是**接口重命名/契约变化**的显式同步，没有把旧 bug 固化为正确预期：
  `tests/test_demo_llm_agent_mode.py` 的 `compiled_controller_count → controllers_built`
  （断言语义不变："direct 计划不组合 controller"），以及本轮新增文件中我自己的断言修正（记录见提交说明）；
* 未重新训练模型，未运行大型离线推荐实验。

## 8. 工作树开始与结束状态

```text
开始：HEAD faeff0b6f790d8c9d077d7d07ee26998aeecad2a，25 modified + 27 untracked
结束：HEAD faeff0b6f790d8c9d077d7d07ee26998aeecad2a，25 modified + 29 untracked
```

* 新增 2 个未跟踪文件：`tests/test_agent_state_isolation.py`（本轮执行型回归）与
  `docs/V2_E01A_STATE_FIX_REPORT.md`（本文件）；modified 计数**未变**；
* 其余改动都是既有 modified 文件上的原地编辑：
  `recommendation/demo/agent_policy.py`、`recommendation/demo/agent_service.py`、
  `tests/test_demo_llm_agent_mode.py`（三者本就 modified）；
* 前序文件逐条确认仍在（TIGER / Demo / Memory / E-01A 全部未提交成果与 5 份文档），无删除、无回退；
* 未执行 `commit` / `add` / `reset` / `checkout` / `restore` / `clean` / `stash`；
  未安装依赖；未训练模型；未调用在线 LLM。

## 9. 结论

**PASS。** 相同输入、相同计划、相同确定性工具状态下，两次独立推荐 run 不再因上一次的 Policy
内部状态跳过计划来源：同一 controller 连续三次 run 的动作与候选完全一致（§3），legacy 与新
`llm_agent` 两条入口在重复请求与并发请求下都完整执行各自计划（§4、§6），同 run 内合法重试与预算
语义未变（§5），归档回放未变化、合并回归 646 passed / 0 failed。本轮到此停止，不进入 E-01B。
