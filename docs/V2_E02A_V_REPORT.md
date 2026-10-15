# AgentRec-X V2 / E-02A-V — 多来源 Ledger → RRF → 最终推荐呈现正确性核验

> 前序：[`docs/V2_E02A_REPORT.md`](V2_E02A_REPORT.md)（Memory-aware Ranking；其 §11 边界 2 记录的正是本轮缺陷）。
> 结论：**PASS** —— 缺陷已复现（不是猜测）、已按最小范围修复、并以真实执行链证明：
> 多来源 run 的最终候选是本轮**全部合法执行来源**的 grounded 候选经**既有 RRF** 融合后的集合，
> 商品身份、来源、`score_kind`、来源分数与 provenance 保真，Memory Ranking 在该融合池上重排。

---

## 0. 一句话结论

```text
两来源、互不相交（A: item-a,item-b ／ B: item-c,item-d），k=4：

修复前  A→B: ChatResponse = [item-c, item-d]        ← 只剩最后一个来源
        B→A: ChatResponse = [item-a, item-b]        ← 只剩最后一个来源
修复后  A→B = B→A = [item-a, item-c, item-b, item-d]  ← 两来源融合，且与执行顺序无关
        （RRF 手算：item-a 与 item-c 各 1/61 并列 → 按身份稳定 tie-break；item-b 与 item-d 各 1/62）
```

## 1. HEAD 与初始工作树

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （与上一轮报告一致）
$ git status --short -uall   →  26 modified + 36 untracked
```

修改前已阅读 `recommendation/control/{candidate_plane,loop,candidate_ledger,constraint_eligibility}.py`、
`recommendation/demo/{agent_trajectory,agent_service,memory_ranking,agent_chat_adapter}.py` 的现有 diff。
未执行任何禁用 git 操作；未删除未跟踪文件；未安装依赖；未训练/加载模型；未调用在线 LLM；
未修改 Fast Serving `/v1/recommend`。

## 2. 原 Ledger → 最终 ChatResponse 的准确执行链（数据字段确认，非函数名推断）

| 环节 | 事实（源码 + 运行确认） |
|---|---|
| 1. 每个工具执行后 `GroundingVerifier` 记录什么？ | `CandidatePlane.execute`（`candidate_plane.py:356`）先 `tool.propose(...)`，再 `GroundingVerifier.record(ledger, source=…, candidates=…, score_kind=tool.score_kind, source_query=…, step_index=…)`；**只有映射/目录都确认的 identity 记为 `grounded`**，其它以 `ungrounded` 保留供审计（`LedgerEntry.verification_status`）。逐 identity 写 `CandidateProvenance(source, source_rank, source_score, score_kind, source_query, item_id, step_index)`。 |
| 2. 多来源时 Ledger 保留全集/并集/快照？ | **并集**：`record_from_source` 逐 identity 追加 provenance（同一商品多来源 = 多条记录，`is_multi_source` = `len(provenance)>1`），**不合并、不覆盖**。 |
| 3. 去重、provenance、RRF？ | 去重以 `parent_asin` 为键（`LedgerEntry` 一商品一条）；RRF 用 `reciprocal_rank_fusion`（`candidate_ledger.py:171`，`score = Σ 1/(K+rank)`，`K=60`，并列按 identity 字符串升序 → 全序稳定）；`ranked_entries()` / `ranked_parent_asins()` / `fusion_scores()` 是 Ledger 的融合视图。 |
| 4. `state["tool_result"]` 是累计还是最后一次？ | **最后一次**。`_adopt_plane_candidates`（`loop.py:939`，本轮修复前）与 capability 路径的 `_adopt_candidates`（`loop.py:1491`）都用**替换**语义写 `state["tool_result"]`；capability 路径（`RECOMMEND_FROM_HISTORY`）**完全不写 Ledger**（`loop.py:846` 的调度注释与运行确认）。 |
| 5. `presentable_candidates` 修复前从哪里取、如何决定顺序？ | 取 `state["tool_result"]`（= 最后一次 adoption），顺序即该来源自身顺序；再用硬约束投影白名单过滤。 |
| 6. `AgentRecommendationView` / ChatResponse 是否改变集合或顺序？ | 不改变：`build_recommendations_view` 按传入顺序 1..n 编号，卡片按同序输出；分数/种类由 `_score_for`（先 Ledger provenance，再退回 item）决定。 |
| 7. Memory Ranking 实际收到哪个集合？ | 修复前收到 5 的集合（最后一个来源）；修复后收到融合池。 |
| 8. `k`、展示上限、预算、终止状态在哪里生效？ | 工具/平面各自按 action 的 `limit`（= 策略 `default_k` = 请求 `k`）返回；预算由 `LoopController` 的 `max_tool_calls`/`max_steps` 计数控制；终止由 `CompletionGuard` + `TerminalView` 记录（`finished` / `completion_refused` / budget 等）。修复前**没有任何环节**在融合集合上再施加 `k`（因为集合恒等于单个来源的结果，天然 ≤k）。 |

**结论**：问题处在"Controller 终态 → 最终推荐视图"边界——呈现集合被定义为**最后一次 adoption**，
而 Ledger 与 RRF 明明持有本轮全部 grounded 候选。

## 3. 问题是否复现：完整来源与候选轨迹

真实组件（`LoopController` / `ActionValidator` / `CandidatePlane` / `GroundingVerifier` /
`CandidateLedger` / 既有 RRF / AgentRecommendationView / Chat Adapter），仅 Provider 与两个来源
payload 为确定性替身。来源 A = `history`（item-a, item-b），来源 B = `catalog_search`
（item-c, item-d；交集查询再返回 item-b）。

```text
场景                      已执行来源              Ledger 记录                     最终 ChatResponse（修复前）
A only  [history]         history              （capability 不写 ledger）        [item-a, item-b]        ✔
B only  [catalog]         catalog_search        item-c#1, item-d#2                [item-c, item-d]        ✔
A→B     [history,catalog] history, catalog      item-c#1, item-d#2                [item-c, item-d]        ✘ 丢 item-a/b
B→A     [catalog,history] catalog, history      item-c#1, item-d#2                [item-a, item-b]        ✘ 丢 item-c/d
A+B 交集 [history,catalog] history, catalog      item-b#1, item-c#2, item-d#3      [item-b, item-c, item-d] ✘ 丢 item-a
A+B 交集 [catalog,history] catalog, history      item-b#1, item-c#2, item-d#3      [item-a, item-b]        ✘ 丢 item-c/d
```

`state["tool_result"]`、`presentable_candidates()` 与 `AgentRecommendationView` 三者与最后一列完全一致，
即缺陷贯穿"终态 → 视图 → 响应"整段，但**没有**出现在 Ledger 内部（Ledger 始终持有正确并集与 provenance）。

## 4. 修复前后对照（item_id / source_rank / source_score / RRF / 最终 rank）

以"两来源互不相交、k=4、A→B"为例（B→A 修复后逐项相同）：

| item_id | parent_asin | 来源与来源内 rank | source_score（kind） | RRF 权重（手算） | 修复前最终 rank | 修复后最终 rank |
|---|---|---|---|---|---|---|
| 101 | item-a | history #1 | 0.91（sasrec_logit，capability 不写 ledger） | 1/61 = 0.016393 | — （未呈现） | 1 |
| 103 | item-c | catalog_search #1 | 1.203973（bm25） | 1/61 = 0.016393 | 1 | 2 |
| 102 | item-b | history #2 | 0.82（sasrec_logit） | 1/62 = 0.016129 | — （未呈现） | 3 |
| 104 | item-d | catalog_search #2 | 1.203973（bm25） | 1/62 = 0.016129 | 2 | 4 |

并列按 identity 升序稳定 tie-break（`item-a` < `item-c`，`item-b` < `item-d`），与
`reciprocal_rank_fusion` 的输出逐项一致（`tests/test_multi_source_fusion_presentation.py` 中另有
独立手算实现与之比较）。交集场景（B 返回 item-b, item-c, item-d，来源内 rank 1,2,3）：
item-b 权重 = 1/61 + 1/62 = 0.032522（两来源共同提出 ⇒ 居首），item-a = 1/61，item-c = 1/62，
item-d = 1/63 ⇒ 最终 `[item-b, item-a, item-c, item-d]`，两种执行顺序结果相同。

## 5. 具体修改文件与 file:line

| 文件:行 | 改动 | 理由 |
|---|---|---|
| `recommendation/control/loop.py:974-976` | capability adoption 记录本次来源的 ranking | 顺序来源的候选此前**只存在于 `state["tool_result"]`**，被后续 plane 步骤覆盖后无法恢复；Ledger 是 plane 的记录，不包含它 |
| `recommendation/control/loop.py:979-997` | 新增 `_record_candidate_ranking(*, source, tool_result)`：保留"每来源一个 ranking"，并重算累计呈现集合 | 需要一个"本轮全部已执行来源"的记录；只新增一个 state 键（`candidate_rankings`）与一个派生键（`presentable_tool_result`），不改动任何既有键的语义 |
| `recommendation/control/loop.py:999-1047` | 新增 `_fuse_presentable(...)`：单来源**原样返回**（逐字节不变），多来源调用**既有** `reciprocal_rank_fusion` | 复用既有 RRF（公式/K/tie-break 未改）；不做拼接列表；身份按 `parent_asin` 去重，首次提出的来源拥有其渲染记录（rank/item_id/score） |
| `recommendation/control/loop.py:1049-1057` | 新增只读 `presentable_tool_result()` | 单一真源：一次写入、多个消费者读取，避免各自重算融合 |
| `recommendation/control/loop.py:1591-1594` | plane adoption 同样记录 ranking | 与 capability 路径统一 |
| `recommendation/control/loop.py:1635-1644` | `refresh_feasibility` 的**评估**改用累计集合 | 否则硬约束下的可行性判定只覆盖最后一个来源，投影会比答案更窄 |
| `recommendation/control/loop.py:1660-1662` | 投影生成改用累计集合 | 同一真源：`constraint_projection` 的 `presented_identities` 描述"可呈现候选"，必须覆盖可呈现集合 |
| `recommendation/demo/agent_trajectory.py:243-267` | `presentable_candidates` 优先读累计集合（无则退回 `state["tool_result"]`） | 保持"被排序的集合 == 被发布的集合"；无控制器的 state（夹具/回放）行为不变 |
| `recommendation/demo/agent_trajectory.py:276,310-311` | `build_recommendations_view(..., limit=None)`：**排序之后**再截断 | `k` 应作用在融合集合上，而不是某一个来源的切片 |
| `recommendation/demo/agent_trajectory.py:140-200` | `build_grounded_view`：把"已采纳但无 plane provenance"的候选纳入 grounded 宇宙，并按运行自身的呈现顺序排列；来源列表用运行真实执行顺序 | 响应的 `grounded` 区块必须描述它**可能展示**的宇宙；plane-only run 逐字节不变 |
| `recommendation/demo/agent_trajectory.py:201-216` | 新增 `_run_candidates` / `_run_sources` 两个只读读取器 | 读累计状态，无副作用 |
| `recommendation/demo/agent_service.py:654,670-676` | 用累计集合生成标题，并向视图传 `limit=payload.k` | 标题与 `k` 都作用在同一个（融合）集合上 |
| `tests/test_multi_source_fusion_presentation.py`（**新增，13 项**） | 本轮执行型测试 | §6 A–G 全覆盖 |
| `tests/test_demo_memory_ranking.py`（更新 5 处断言 + 常量） | 契约变更 | 旧断言固化了"只呈现最后来源"的错误行为（`sources_present == ["catalog_search"]`、`multi_source_count == 0`、融合基线 = 词法来源顺序），已替换为更强的来源/融合/身份断言 |
| `docs/USAGE.md`、`docs/V2_E02A_REPORT.md` §11 | 文档 | 说明新的呈现契约，并标注 E-02A 报告当时的边界已修复 |

**为什么必须动 LoopController（而不是只改视图层）**：顺序来源的候选在后续 plane 步骤覆盖
`state["tool_result"]` 之后**在进程中已不存在**——Ledger 只记录 plane 来源，capability 不写 Ledger，
`state` 里没有第二处保留点。若不新增这一个累计记录，"全部已执行来源"在物理上无法恢复。
改动是**纯增量**：新增 2 个 state 键、1 个只读访问器、1 个融合函数，既有键、策略可见计数
（`candidate_state()`）、预算与终止判定均未触碰。

## 6. 单来源、多来源、交集、工具顺序、`k` 限制的执行结果

| 场景 | 期望 | 实测 |
|---|---|---|
| 单来源 A（`[history]`） | 该来源集合与顺序 | `[item-a, item-b]`，分数 0.91/0.82，`score_kind=sasrec_logit`，`fusion_sources=[]` |
| 单来源 B（`[catalog_search]`） | 该来源集合与顺序 | `[item-c, item-d]`，`score_kind=bm25`，`fusion_sources=["catalog_search"]`，`grounded_count=2` |
| 多来源无交集 A→B / B→A | 4 件商品、融合顺序 | 两者均 `[item-a, item-c, item-b, item-d]`（= 手算 RRF） |
| 多来源有交集（两顺序） | item-b 只出现一次、两来源权重合并、居首 | 均 `[item-b, item-a, item-c, item-d]`；`item-b` 一张卡片、一条 catalog provenance（rank 1）；权重 1/61+1/62 |
| 工具执行顺序 | 融合结果与顺序无关 | 两顺序的 `chat ids`、每来源 provenance 完全一致；轨迹分别为 `[recommend_from_history, search_catalog, finish]` 与 `[search_catalog, recommend_from_history, finish]` |
| `k` 限制 | 在融合集合上截断 | `k=2` → 卡片 2 张 = 融合顺序前 2（`[item-a, item-c]`），而 `grounded_count` 仍为 4（`k` 限制展示，不限制检索） |

Ledger 边界的交集验证（真实 `GroundingVerifier` + `CandidateLedger`，两个 **plane** 来源）：
`item-b` 同时被 `catalog_search#1(bm25=1.203973)` 与 `similar_item#1(item_item_similarity=0.44)` 提出，
条目保留**两条 provenance**、`is_multi_source=True`、`multi_source_parent_asins()==("item-b",)`；
手算权重 `2/61` 与 `ledger.fusion_scores()` 逐项一致；只被一个来源提出的 `item-c` 保留单条记录；
未 grounded 的 `item-e` 保留为审计项且**永不进入候选**（`ledger.ungrounded_entries`）。

## 7. Memory-aware Ranking 接收的真实候选集合

`rank_with_memory(candidates=presentable_candidates(state), ...)`：
输入即 6 中的融合集合（同一次运行同一个工件，测试同时断言 `result.state["presentable_tool_result"]`
的身份顺序与响应卡片顺序一致）。在交集场景上：

```text
无 Memory            : [item-b, item-a, item-c, item-d]（= RRF 基线，reranking_applied=false）
prefer blue          : [item-c, item-b, item-a, item-d]（item-c 由第 3 升到第 1；M10A match: details.Color=blue）
avoid red            : item-d 得到真实 violation（details.Color=red）；它本已末位，故顺序不变——
                       策略不会为"制造变化"而臆造位移（测试如实断言"证据变化、顺序不变"）
REMOVE               : 回到 RRF 基线，reranked_rank 全为 None
缺失 metadata        : 仍为 unknown（不匹配也不违反）——由 E-02A 的适配层测试覆盖，本轮未改匹配规则
```

三者被明确区分：**来源候选集合**（每来源自身顺序）→ **RRF 融合顺序**（`original_rank` /
`audit.original_order`）→ **Memory-aware 最终顺序**（`reranked_rank` / `audit.reranked_order`）。
重排前后 `(identity, score, score_kind, fusion_sources)` 集合完全一致（测试逐项比较 `spine`）。

## 8. 原始分数、`score_kind` 与 provenance 保真证据

| 卡片 | `score_kind` | `sasrec_score` | `fusion_sources` | 说明 |
|---|---|---|---|---|
| item-a | `sasrec_logit` | 0.91 | `[]` | 只有顺序来源提出；无 plane 记录，故无 provenance（未编造） |
| item-c | `bm25` | 1.203973 | `["catalog_search"]` | 词法来源自己的分数与类型 |
| item-b（交集） | `bm25` | 1.203973 | `["catalog_search"]` | 一商品一卡；发布分数取 Ledger 的真实记录并**标注其类型**，不是 RRF 值 |
| item-d | `bm25` | 1.203973 | `["catalog_search"]` | 同上 |

* 断言"没有任何卡片的 `score_kind` 是 `rrf_rank_fusion`"：融合只决定**顺序**，从不冒充某个来源的分数；
* 累计信封 `presentable_tool_result.score_kind == "rrf_rank_fusion"` 是**该工件自身**的说明，
  它不进入任何 per-card 字段（测试同时断言这两点）；
* `grounded.sources_present == ["history", "catalog_search"]`，`grounded_count == 4`，
  `verification_status` 全为 `grounded`；未被任何来源提出的商品不可能出现（未执行来源不参与）。

## 9. 失败、空候选、预算、请求隔离回归

| 情形 | 实测 |
|---|---|
| 某来源零候选（词法查询无匹配） | 顺序来源候选**不受影响**：`[item-a, item-b]`，`grounded_count=2`，`search_catalog` 仍记录在 `action_counts` |
| 全部来源零候选（仅词法计划） | 保持既有契约：`recommendations=[]`、`outcome=refused`、`terminal.termination_reason=completion_refused`、`message` 非空 |
| 某来源失败 / 未执行来源 | 未执行来源的商品永不进入结果（候选只来自 adoption 与 Ledger 的 grounded 条目） |
| 两个请求不混用 Ledger | 同一 harness 的两个 session：run1 = 4 件、run2（仅词法计划）= 2 件且 `sources_present==["catalog_search"]`；同 session 的后续单来源回合也不会继承上一回合的候选 |
| 预算与终止语义 | 未改动：`max_tool_calls`/`max_steps` 计数、`CompletionGuard`、`completion_refused` 全部沿用；终止状态在测试中显式断言（`finished` / `completion_refused`） |
| 请求级隔离与并发 | `tests/test_agent_state_isolation.py`（含 8 线程并发用例）全绿 |
| 硬约束投影 | 评估与投影改用累计集合：单来源 run 逐字节不变；多来源 run 的投影覆盖可呈现集合（既有约束测试全绿） |

## 10. 全部测试命令及通过/失败数

```bash
.venv/bin/python -m pytest tests/test_multi_source_fusion_presentation.py \
  tests/test_demo_memory_ranking.py tests/test_llm_selected_controller.py \
  tests/test_agent_state_isolation.py tests/test_agent_control_plane.py \
  tests/test_agent_chat_adapter.py tests/test_demo_llm_agent_mode.py \
  tests/test_demo_llm_agent_memory_acceptance.py tests/test_demo_llm_agent_memory.py \
  tests/test_demo_agent_api.py tests/test_provider_record_replay.py tests/test_demo_api.py \
  tests/test_demo_multiturn.py tests/test_demo_llm_mode.py tests/test_preference_matching.py \
  tests/test_preference_reranking.py tests/test_agent_reranking.py tests/test_agent_memory.py \
  tests/test_memory_query_adapter.py tests/test_memory_bm25_retrieval.py \
  tests/test_llm_tool_selection.py tests/test_demo_control_plane.py \
  tests/test_demo_recommendation_trace.py tests/test_demo_sessions.py tests/test_demo_web.py \
  tests/test_demo_fusion_view.py tests/test_llm_agent_availability_guard.py \
  tests/test_demo_preference_pipeline_fixes.py tests/test_control_stage2_candidates.py \
  tests/test_constraint_eligibility.py tests/test_candidate_reference_boundary.py \
  tests/test_llm_agent_policy.py tests/test_serving_fusion.py -q
```

```text
942 passed in 71.51s        （0 failed；本轮新增 13 项，E-02A 的 16 项在契约更新后全部通过）
```

回归确认：Legacy `deterministic` / `llm` Chat 不回归；新 `llm_agent` 不回归；Agent Recommend
不回归；E-02A Memory Ranking 不回归（断言按新契约加强）；Serving `/v1/recommend` 未修改且
`test_serving_fusion.py` / `test_demo_api.py` 全绿；控制器预算与终止语义不回归（§9）。
**契约变更说明**：`tests/test_demo_memory_ranking.py` 中 5 处断言此前把"只显示最后一个来源"
固化为期望值（`sources_present == ["catalog_search"]`、`multi_source_count == 0`、融合基线 =
词法顺序），本轮按要求替换为更强的断言（两来源都出现在 `sources_present`、每来源 provenance、
融合顺序 = 手算 RRF、身份/分数/spine 不变），并在注释中写明契约变更原因；**没有**弱化或删除
任何与身份、来源、融合、终态相关的断言。

## 11. 结束时 HEAD 与 git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （未提交、未 reset）
$ git status --short -uall
开始：26 modified + 36 untracked
结束：26 modified + 38 untracked      （本轮新增 2 个未跟踪文件：
                                       tests/test_multi_source_fusion_presentation.py、
                                       docs/V2_E02A_V_REPORT.md）
```

modified 集合未新增文件（改动全部落在更早轮次已 modified 的 `recommendation/control/loop.py`、
`recommendation/demo/{agent_trajectory,agent_service}.py` 与 `docs/USAGE.md`、
`docs/V2_E02A_REPORT.md`、`tests/test_demo_memory_ranking.py` 上）。前序 TIGER / Demo / Memory /
E-01A / E-01B / E-02A 成果与 9 份文档逐条确认仍在。未执行任何禁用 git 操作，未安装依赖，
未训练或加载模型，未调用在线 LLM。

## 12. 最终结论

**PASS。** 完成标准逐项成立（真实执行链，非 Ledger 单测替代）：

```text
Agent 合法执行多个来源（history + catalog_search，两顺序都验证）
  → Ledger 保留全部本轮 grounded 候选与 provenance（并集；交集商品两条记录）
  → 既有 RRF 正确融合（公式/K/tie-break 未改；手算逐项一致）
  → 最终 ChatResponse 展示正确的融合候选（4 件商品、顺序与手算一致、k 在融合后生效）
  → Memory Ranking 在这个融合池上可信重排（身份/分数/来源不变，仅顺序变化）
```

同时：单来源行为逐字节不变；执行顺序不影响融合结果；`grounded` 区块描述它可能展示的宇宙；
失败/零候选/预算/终止/请求隔离均不回归。

**仍然存在、且本轮明确不解决的边界**（如实记录）：

1. **顺序来源不是 plane 来源**：`RECOMMEND_FROM_HISTORY` 走 accepted capability，不写 Ledger
   provenance，因此 `grounded.multi_source_count`（Ledger 视角的重复计数）在 history+plane run
   中仍为 0，且该商品的 `fusion_sources` 为空。让 capability 成为 plane 来源属于控制平面改造，
   超出本轮范围（§8）；本轮改为在 **来源列表**（`sources_present`）与 **呈现融合** 中如实体现，
   并以 Ledger 边界测试证明"两个 plane 来源重叠时两条 provenance 都被保留"。
2. **呈现层融合的实现位置**：融合发生在 loop 的 adoption 记账（新增 2 个 state 键），而非 Ledger
   内部 API；理由见 §5（capability 候选在 Ledger 之外，且 Ledger 是 plane 的审计记录）。
   若未来要让 Ledger 成为唯一融合真源，需要把 capability 也纳入 plane（控制平面级变更）。
3. 真实 Provider、真实模型资产与大规范质量实验仍未纳入（沿用前几轮边界）。

本轮到此停止，不进入 E-02B。
