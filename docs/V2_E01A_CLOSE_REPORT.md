# AgentRec-X V2 / E-01A-CLOSE — 安全守卫修复与 Chat Agent 最终验收

> 前序文档：[`docs/V2_E01A3_HANDOFF.md`](V2_E01A3_HANDOFF.md)、
> [`docs/V2_E01A3B_REPORT.md`](V2_E01A3B_REPORT.md)（其 §13.1/§13.2 已按本轮结论更正）。
> 结论：**PASS** —— 安全守卫以 Case B（收窄投影）方式关闭，前序唯一失败已消除，
> 合并回归 633 passed / 0 failed。本轮不新增 Agent 功能，未进入 E-01B。

---

## 1. 安全守卫失败的准确根因

失败断言有**两条**，都在 `tests/test_agent_control_plane.py::test_policy_context_has_no_route_to_trusted_state`
（该文件已提交、未改动）：

```python
assert set(vars(context)) == allowed          # :183  字段集白名单
...
assert not hasattr(context, "preference_snapshot")   # :214  显式禁止该字段
```

改动前基线（工作树未动）即 `1 failed, 164 passed`，失败项就是这条。根因不是"白名单忘了同步"，
而是 **M-02A 把一个不该跨界的对象挂到了策略上下文上**：

* `recommendation/control/loop.py:386`（改动前）把**整份 M9 snapshot** 直接交给策略：
  `preference_snapshot=snapshot`；
* 该 snapshot（`recommendation/memory/schemas.py:397`）带 `user_key` 与 `read_at`，其每条
  entry（`:348`）还带 `user_key`、**`source_text`（用户原话的逐字片段）**、`source_turn_id`、
  `extractor`、`created_at`、`supersedes`/`superseded_by`；
* 而 `recommendation/control/context.py` 的模块文档（提交版）恰恰写着：策略只能看到
  "how many explicit preferences are active - **a count, never the entries**"，且
  "What it must never expose ... ``user_key``, the memory store, **the preference entries
  themselves**"。守卫测试的 `:214` 就是把这句话钉成了断言。

即：不是守卫落后于设计，而是这次改动**没有遵守它所修改的那个模块自己声明的边界**（M-02A 也没有
更新那段文档）——因此属于任务书里的 **情况 B（暴露过宽）**，而不是情况 A。

## 2. `preference_snapshot` 的真实数据边界（逐字段核对，不只看字段名）

唯一消费者是 M-02A 的查询适配器 `recommendation/memory/query_adapter.py:88-110`，它通过
`getattr` 读取：

| 读到的 | 是否被使用 | 是否属于"控制决策所需" |
|---|---|---|
| `snapshot.entries` | 是（容器） | 部分（只需要 ACTIVE 项） |
| `entry.status` | 是（`== ACTIVE` 复核） | 是（用值本身就要求是活着的偏好） |
| `entry.polarity` | 是（只收 `PREFER`） | 是 |
| `entry.kind` | 是（`in TEXT_KINDS`） | 是 |
| `entry.value` | 是（并进查询词） | 是（本功能的全部意义） |
| `entry.logical_seq` | 是（稳定排序键） | 是（可复现顺序） |
| `entry.memory_id` | 是（排序 tie-break + `QueryAdaptation.memory_ids` 上报） | 可接受（不透明内部 id，非用户标识） |
| `snapshot.user_key` / `entry.user_key` | **否** | **否** |
| `entry.source_text`（用户原话） | **否** | **否** |
| `entry.source_turn_id` / `created_at` / `read_at` / `extractor` | **否** | **否** |
| `entry.supersedes` / `superseded_by` / 非 ACTIVE 条目 | **否** | **否**（用户已撤回的偏好不应再被用于检索） |

另外两处事实（用于评估暴露面）：

* **模型侧策略并不读它**：`recommendation/control/model_policy.py:262-322` 用白名单构造发给
  provider 的 payload，只取 `active_preference_count` 等字段，从不取 `preference_snapshot`。
  也就是说这是一条**潜在**通道（任何策略对象都能读到 `context.preference_snapshot.user_key`），
  而不是当前已被利用的通道——但 `context.py` 的边界原则正是"by construction rather than by
  instructing the policy not to do it"。
* **轨迹侧没有泄漏**：`PolicyContext.summary()`（`:291`）是白名单，只含
  `active_preference_count`，其文档明写"never a preference value"。

## 3. 修改文件与 file:line

| 文件:行 | 改动 |
|---|---|
| `recommendation/memory/query_adapter.py:75-142` | 新增 `ActivePreference`（六个字段）与 `ActivePreferenceProjection`（`.entries` + `count`），新增 `project_active_preferences(snapshot)`：只读容器 `entries`，只拷贝 ACTIVE 条目，只拷六个值；`__all__` 同步；模块文档新增"What crosses into a policy, and what does not" |
| `recommendation/control/context.py:25-36` | 模块边界文档改写：明确策略拿到的是**值级投影**，并逐项列出永不跨界的东西（snapshot、`user_key`、`source_text`、turn id、时间戳、已撤销条目） |
| `recommendation/control/context.py:166-175` | `preference_snapshot: Any \| None` → `active_preferences: ActivePreferenceProjection \| None`（注解用 `TYPE_CHECKING` 导入，运行期 control 不依赖 memory 层） |
| `recommendation/control/context.py:46-50` | `TYPE_CHECKING` 导入块 |
| `recommendation/control/loop.py:373-396` | `build_context()` 用 `project_active_preferences(snapshot)` 构造投影，并把"为何在建投影处收窄"写进 docstring；`import`：`loop.py:46` |
| `recommendation/demo/agent_policy.py:256-260` | 唯一消费者改读 `getattr(context, "active_preferences", None)`（保持 duck-typed，`use_memory=False` 分支不变） |
| `recommendation/memory/query_adapter.py` | `adapt_terms` **一字未改**（它本来就按 `.entries` + 六个属性读取） |
| `tests/test_agent_control_plane.py:182-189` | 白名单加入 `active_preferences`，并写明这是"deliberate widening and a bounded one"以及具体边界 |
| `tests/test_agent_control_plane.py:220-233` | 保留 `not hasattr(context, "preference_snapshot")`；新增投影为空时的断言 |
| `tests/test_agent_control_plane.py:279-361` | 新增 `test_the_preference_projection_carries_values_and_nothing_else`：用真实 M9 服务写入一条 ACTIVE、再写入一条 REMOVED（distinctive 值），跑真实 loop，逐项断言投影内容、被撤销条目不跨界、`user_key`/`source_text`/`source_turn_id`/`memory_id` 在 `repr(context)` 与 `context.summary()` 中均不可达，且 store/service/snapshot 均不可达 |
| `tests/test_memory_query_adapter.py:162-227` | 源代码级接线钉更新为新字段名，并**加强**为"策略源码中不得再出现 `preference_snapshot`"；新增 `test_the_projection_keeps_the_feature_and_carries_nothing_else`（功能保真 + 边界） |
| `tests/test_llm_selected_controller.py:36-51, 90-102` | 两处 `PolicyContext(...)` 改为传投影（与生产一致），并补注释说明"必须用生产实际交付的对象测" |

未改：`recommendation/memory/query_adapter.py:adapt_terms` 的算法与常量、M-02A 的任何测试期望、
`control/model_policy.py`、任何 store/service 实现、任何鉴权系统（本轮未引入）。

## 4. 为什么这个修复不破坏原有的可信状态隔离

1. **收窄而非放宽**：策略能看到的字段从"整份条目（12 字段）"变成"6 个值"；被移除的正是
   `user_key`、`source_text`、`source_turn_id`、`created_at`、`read_at`、`extractor`、
   `supersedes`/`superseded_by`，以及所有非 ACTIVE 条目。没有任何新字段被加进策略视野。
2. **边界仍由构造强制**：投影在**控制器**里构造（`build_context`，每一步一份），策略拿到的是
   冻结 dataclass，无法回指 snapshot、store 或 service；守卫测试 `:214` 的
   `not hasattr(context, "preference_snapshot")` 重新成立并且现在被保留为断言。
3. **越权路径没有被顺带重开**：守卫测试原有的全部断言（`trusted_user_history`、`user_key`、
   `tool`、`engine`、候选身份只在唯一通道、`grounded_candidates` 字段集、轨迹摘要不含身份、
   证据只给计数等）逐条保留并通过。
4. **功能等价**：`adapt_terms` 与其排序键 `(logical_seq, memory_id)` 未改，投影逐值携带同样的
   六个字段，因此查询词、`QueryAdaptation.memory_ids`、`changed` 与改动前逐值相同。归档回放套件
   `tests/test_provider_record_replay.py`（钉住 action 序列）在本轮合并回归中全绿，是这一点的外部证据。
5. **数据仍留在可信侧**：`state["preference_snapshot"]`（完整 snapshot）照旧保留给 capability 的
   匹配/重排与 M9 持久化读取（`loop.py:341`、`loop.py:859`），只有策略视野被收窄。

## 5. Catalog 冷/热差异：分类与证据（含对上一份报告的更正）

**上一份报告的描述有误**（归因于 service 共享的 `CatalogSearchSource`/索引首次使用）。本轮的定向
判定推翻了它：`CatalogSearchSource.__init__` 即时 `_build()` 索引（`recommendation/control/catalog_search.py:172`），
不存在懒加载；且同一 service 下**新建** controller 依旧复现"1 卡"行为。

**真实机制**（同一 controller 的第 1 次 vs 第 2+ 次运行）：

```text
run 1: actions=[recommend_from_history, search_catalog, finish]  presented=[cand-red]                         (1 卡)
run 2: actions=[recommend_from_history, finish]                  presented=[cand-red, cand-blue, cand-black]  (3 卡)
run 3: 同 run 2                                   policy._served 始终为 ['catalog_search', 'history']
```

* 泄漏点：`recommendation/demo/agent_policy.py:173` 的 `self._served` 只在实例化时为空；`choose()`
  仅重置 `_finished_plan`（`:211-213`），**从未清空 `_served`**——而类注释 `:171-174` 与
  `choose` 文档 `:200-204` 都明确声明"一个 controller 服务的每个 run 都应重置"、"可变状态会泄漏
  并悄悄改变第二个请求的行为"。
* 触发条件：`_proposal_for` 的守卫 `if context.candidate_state.grounded and source in self._served:
  return None`（`:247`）。第 2 次运行时，history 步已经 grounded，而 `catalog_search` 因上一轮
  残留仍在 `_served` 中 → 直接跳过 → `_finished_plan=True` → FINISH。
* 影响面：凡 `(k, user_key, plan)` 命中同一 controller 的重复请求。`/v1/demo/agent/recommend`
  的调用 user_key=None；新 `llm_agent` 模式同样以 user_key=None 运行，因此**所有会话共享同一
  controller**，第 2 个会话起即可能少跑一个来源（实测：session A/B/C 的
  `action_counts` 均为 `{recommend_from_history:1, finish:1}`，`catalog_search` 未被咨询）。
* 分类（按任务书四项）：**不是**首次调用延迟（步骤与耗时形状不变）；**是**候选数量/顺序问题
  （1 卡 vs 3 卡，plane 采纳集 vs capability 集）；**不是**工具 ready/初始化状态（同 service 新建
  controller 行为一致）；**是** controller 的"合法提前终止"——策略因泄漏的已服务状态提前收尾。
* 最小复现：`/v1/demo/agent/recommend` 同一 payload 连发两次即得 1 卡 / 3 卡，
  `steps=[history,catalog,finish]` vs `steps=[history,finish]`；白盒版见本节代码块。
* 建议修复（**当时按范围纪律未做，已在 E-01A-STATE-FIX 完成**）：
  在 `choose()` 的 `step_index == 0` 分支里同时 `self._served.clear()`，并把 controller 组合改为
  请求级（policy + plane + ledger 都不再跨请求），随后重跑
  `tests/test_provider_record_replay.py` 等归档回放套件 —— 回放全绿，说明变更只影响"同一 controller
  的第 2 次及以后运行"这一原先错误的行为。见
  [`docs/V2_E01A_STATE_FIX_REPORT.md`](V2_E01A_STATE_FIX_REPORT.md)。

## 6. 最终测试命令、通过/失败数量

单次 pytest 调用（21 个文件，含任务书点名的全部回归 + 受影响的内存/图/回放套件）：

```bash
.venv/bin/python -m pytest tests/test_agent_control_plane.py tests/test_memory_query_adapter.py \
  tests/test_memory_bm25_retrieval.py tests/test_llm_selected_controller.py \
  tests/test_llm_tool_selection.py tests/test_agent_chat_adapter.py \
  tests/test_demo_llm_agent_mode.py tests/test_demo_agent_api.py tests/test_demo_api.py \
  tests/test_demo_llm_mode.py tests/test_demo_recommendation_trace.py \
  tests/test_demo_control_plane.py tests/test_demo_multiturn.py \
  tests/test_demo_preference_pipeline_fixes.py tests/test_demo_sessions.py \
  tests/test_demo_web.py tests/test_demo_fusion_view.py tests/test_llm_agent_availability_guard.py \
  tests/test_agent_memory.py tests/test_agent_reranking.py tests/test_provider_record_replay.py -q
```

```text
633 passed in 625.44s
```

* **0 failed**（前序唯一失败已消除；守卫测试 `test_policy_context_has_no_route_to_trusted_state`
  与新增的 `test_the_preference_projection_carries_values_and_nothing_else` 都通过）；
* 未调整任何断言以掩盖问题：本轮只改了两处测试期望，且都是**契约随边界变更**的显式更新——
  `test_agent_control_plane.py` 的白名单（新增字段名，并补齐数据级断言）与
  `test_memory_query_adapter.py` 的源代码级接线钉（旧字段名 → 新字段名，并加强为"不得出现旧字段"）。
  两者的**原有语义断言全部保留**，原有失败不是靠放宽期望通过的。

## 7. E-01A 实际完成的功能 / 仍需真实 Provider 验证的部分

**已由自动化证据证明**：

* Chat API `decision_mode="llm_agent"`：真实路由 → `ToolSelectionPlanner`（可信校验）→
  `DemoAgentService.run_selection` → `LoopController` → `ActionValidator` → `CandidatePlane` →
  `GroundingVerifier` → `CandidateLedger` → `build_agent_chat_response` → `ChatResponse`；
* 单源/多源/direct/零候选/completion_refused/工具失败/未提供来源 七类场景的实测轨迹与
  `agent.outcome`、`action_counts`、`terminal`、`grounded` 证据；
* 菜单来源于 `available_sources ∩ SOURCE_PLANS`（TIGER 未 ready 时不进 Prompt）、预算取自控制器
  自身 `max_tool_calls`；
* Memory 边界：新路径 `use_memory=False`（历史偏好不进本轮查询，ledger `source_query` 可证），
  且策略只拿到值级投影；
* 旧 deterministic / llm / Agent Recommend / `/v1/recommend` 行为不变。

**仍需真实环境验证（本轮未执行，无授权）**：

* 真实付费 Provider 的端到端一次调用（步骤已写在 `docs/USAGE.md` 的 "Real-provider smoke test"）；
* 真实 SASRec checkpoint 与真实 TIGER worker（本轮只用合成 engine/catalogue）；
* 生产 launcher（`scripts/start_demo.sh`）上的第三种模式端到端（浏览器未加该模式的选择控件）。

## 8. 结束时实际 HEAD / git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a        （与开始时一致：未提交、未 reset）

$ git status --short -uall          # 逐文件清单
本轮开始：24 modified + 26 untracked
本轮结束：25 modified + 27 untracked
```

逐文件核对（任务书要求不凭计数判断）：

* **状态发生变化的只有 2 个**：
  * `tests/test_agent_control_plane.py`：此前已跟踪且干净 → 因守卫白名单与数据级断言更新而变为
    **modified**；
  * `docs/V2_E01A_CLOSE_REPORT.md`：本文件，**新增 untracked**。
* **原地编辑、状态不变的有 5 个**（本就 modified 或 untracked）：
  `recommendation/control/context.py`、`recommendation/control/loop.py` 在既有未提交改动之上追加；
  `recommendation/memory/query_adapter.py`、`tests/test_memory_query_adapter.py`、
  `tests/test_llm_selected_controller.py` 在既有未跟踪文件上原地修改。
* 前序 24 modified + 26 untracked 的**每一个文件逐条确认仍在**（含 TIGER / Demo / M-02A /
  E-01A-1/2/3A 的全部未提交成果与 3 份文档），无删除、无回退；
* 未执行任何 `commit / add / reset / checkout / restore / clean / stash`；未删除任何未跟踪文件；
  未安装依赖；未调用在线 LLM。

## 9. 结论

**PASS。** 安全守卫的失败被定位为真实的边界过宽（Case B）并以最小改动关闭：交给策略的对象从整份
M9 snapshot 收窄为值级投影，M-02A 的检索行为逐值不变，归档回放套件全绿；合并回归 633 passed /
0 failed。Catalog 冷/热差异被重新定位为 `DemoSourcePlanPolicy._served` 跨 run 泄漏导致的 controller
提前终止（**与 catalog 无关**），已给出最小复现、影响范围与一行建议修复，按范围纪律本轮不改。
本轮到此停止，不进入 E-01B。
