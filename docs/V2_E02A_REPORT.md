# AgentRec-X V2 / E-02A — Memory-aware Fusion Reranking

> 前序：[`docs/V2_E01B2_REPORT.md`](V2_E01B2_REPORT.md)（Memory 驱动工具决策的端到端验收）、
> [`docs/V2_E01B1_REPORT.md`](V2_E01B1_REPORT.md)（Session↔M9 接线与安全投影）。
> 结论：**PASS** —— 同一合法召回候选池、不同 ACTIVE Memory ⇒ M10A 产生不同真实偏好证据 ⇒
> M10B 产生可解释的最终排序变化，且**商品身份、原始分数、score_kind 与来源 provenance 全部不变**；
> 无 Memory 时顺序与融合顺序逐字节一致；撤销/更正可回到基线；旧路径不回归。

---

## 0. 一句话结论与允许的边界

```text
同一请求 "RedWidget BlueWidget BlackWidget GreenWidget"，同一 plan [history, catalog_search]，
同一真实候选池（4 个商品、来自真实 Controller/Ledger，bm25 provenance 完整）：

无 ACTIVE Memory        → 顺序 [blue, green, red, black]   audit: applied=false moved=0
prefer black + avoid red → 顺序 [black, blue, green, red]   audit: applied=true  moved=4
                            black: M10A match(details.Color=black)
                            red:   M10A violation(details.Color=red)
                            blue/green: unknown（颜色不同≠违反，不臆造证据）
每张卡片的 sasrec_score / score_kind / fusion_sources / original_rank 与无 Memory 时**完全相同**。
```

**边界（如实声明）**：本轮的 Memory 影响发生在**呈现候选的排序**（同一候选池内部的重排），
不改变工具选择、不改变候选池、不改变 RRF/来源分数、不重新打分；融合级 Memory-aware 重排之外的
"完整商品 RAG"（E-02B）与 Observe→Replan 未做。旧 `deterministic` / `llm` / Agent Recommend
路径完全不受影响（§4、§10）。

## 1. 开始时工作树状态

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a
$ git status --short | grep -c '^ M' ; grep -c '^??'
25 modified + 33 untracked      （与 E-01B-2 结束时一致；未执行任何禁用 git 操作）
```

## 2. 原 Ledger → 最终 ChatResponse 的真实排序链（定向追踪结论）

```text
ActionValidator 校验过的 action（每步一个来源）
  └─ CandidatePlane.execute(action)                        recommendation/control/candidate_plane.py:356
       ├─ tool.propose(...)                                来源自己的顺序与分数
       ├─ GroundingVerifier.record(self._ledger, ...)      逐来源写 provenance（source_rank/score/score_kind/query）
       └─ RecommendationDomainResult(grounded_parent_asins, item_ids, provenance)
  └─ LoopController._adopt_plane_candidates(result)         recommendation/control/loop.py:939
       └─ state["tool_result"] = ToolRecommendation(rank=位置, item_id, parent_asin, score=0.0)
          ⚠ 平面候选的 score 被**有意置 0**（"lexical 分数不是 SASRec 分数"），真实来源分数留在 ledger
  └─ AgentRecommendResponse 投影                            recommendation/demo/agent_trajectory.py:184
       ├─ presentable_candidates(state)                    硬约束投影后的**呈现集合**（唯一真源）
       ├─ _score_for(identity, item, ledger)               先读 ledger 的 source_score/score_kind，再退回 item.score
       └─ AgentRecommendationView(rank, item_id, parent_asin, score, score_kind, title, sources)
  └─ build_agent_chat_response → RecommendationCard        recommendation/demo/agent_chat_adapter.py:161
       └─ ChatResponse.recommendations / audit / trace / agent
```

**五个定向问题的答案**：

| 问题 | 结论 |
|---|---|
| 1. 最终候选是否确实按 Ledger/RRF 输出？ | 呈现集合是**最后一个候选产出步骤**的 grounded 集合（`_adopt_plane_candidates`），其顺序是该来源自己的顺序；Ledger 累积全部来源的 provenance 并提供 RRF（`ranked_entries()` / `fusion_scores()` / `reciprocal_rank_fusion`）。响应卡片按 `tool_result` 顺序输出，分数/种类按 `_score_for` 取自 ledger。 |
| 2. 可信目录元数据在哪里仍可访问？ | 只在 `DemoAgentService` 内部（`self._enricher` + 共享 `MetadataIndex`）。响应里只有 title；`state` 在服务返回后即被丢弃。**因此适配必须在服务内、在投影之前完成**，不能靠标题猜属性。 |
| 3. 最适合插入的环节？ | `_execute_plan` 内、`build_recommendations_view` 之前 —— 此时"呈现集合"（`presentable_candidates`）、真实 enricher、M10A/M10B 与 M9 快照同时可得，且尚未丢失任何可信事实。 |
| 4. M10B 哪些字段明确绑定 SASRec 单源语义？ | `CandidatePreferenceEvidence.original_rank`（"SASRec rank, copied unchanged"）、`sasrec_score`（"Raw SASRec score, copied exactly"）与 `RerankedCandidate.sasrec_score`。因此**绝不能**把 BM25/RRF 值放进该槽位，也**不能**把 M10A 报告原样对外发布。 |
| 5. 如何保持实际来源与原融合分数不变？ | 只做**重排**：身份、`item_id`、原（融合）位置、`score`、`score_kind`、`sources` 全部从 ledger/状态原样复制；M10B 的键只用 violations / matches / 原位置 / item_id，**根本不用分数**；适配层对"进去的集合 == 出来的集合"做前后校验，并显式回传 `original_rank` 与 `reranked_rank` 两个位置。 |

## 3. M10A/M10B 原输入语义与新增适配契约

| 项 | 原语义（未改） | E-02A 适配（`recommendation/demo/memory_ranking.py`） |
|---|---|---|
| M10A 输入 | `EnrichedRecommendation`（M8 产物）：`recommendation.rank/item_id/parent_asin/score` + `metadata`（`metadata_status` found/missing） | 由**呈现候选**构造同构信封：`rank=呈现位置`、`item_id/parent_asin` 真实、`score=`**候选自己的分数**（history 来源=真实 SASRec logit；平面来源=accepted 的 0.0 "no model score" 占位，**绝不**写入 BM25/RRF 值）、`metadata`=真实 enricher 结果 |
| M10A 语义 | 逐候选 × 逐 ACTIVE 偏好 → 三态证据（match / violation / unknown）+ `reason_code` + 读到的 `metadata_field`/`metadata_value` | 直接调用 `PreferenceCandidateMatcher.match(...)`，不新增任何匹配规则；缺失 metadata ⇒ 全 `unknown`，既不算 match 也不算 violation |
| M10B 输入 | `PreferenceEvidenceReport` | 直接调用 `PreferenceReranker.rerank(report)`；`original_rank` 取**呈现（融合）位置**，因此同分候选按融合顺序打破平局 |
| M10B 输出 | `RerankingReport`（同集合、`reranked_rank` 连续唯一、`moved_count`、`sort_key`、`diagnostics`、逐候选 `rerank_reason`） | 投影为 `MemoryRanking`（`RankedCandidate`：`rank` / `reranked_rank` / 三类计数 / evidence / reason）；**M10A 报告的 `sasrec_score` 槽位永不对外发布**，响应继续用 ledger 的 `score`+`score_kind` |
| 失败语义 | M10B 内部已用 multiset 校验候选宇宙不变 | 适配层再校验一次：集合或连续位置不成立 ⇒ `MemoryRankingError`（整轮失败，绝不静默丢候选） |
| 无 Memory | —— | `rank_with_memory` 返回 `None`：不 enrichment、不匹配、不重排（连开销都不付），顺序与之前逐字节一致 |
| 开关 | —— | `AGENTRECX_AGENT_MEMORY_RANKING`（默认开；`0/false/no/off` 关闭）→ 固定 E-01B 基线；非法值保持默认 |

## 4. 修改文件与 file:line

| 文件:行 | 内容 |
|---|---|
| `recommendation/demo/memory_ranking.py`（**新增**） | `MemoryRankingError:61`、`RankedCandidate:72`、`MemoryRanking:91`、`_narrowed_result:125`（候选信封：分数原样、绝不代入融合分数）、`_active_preferences:163`、`rank_with_memory:172`（enrich → M10A → M10B → 校验 → 投影） |
| `recommendation/demo/agent_schemas.py:53,266-291` | 导入 `PreferenceEvidence`；`AgentRecommendationView` 新增 6 个**可选**字段：`reranked_rank` / `match_count` / `violation_count` / `unknown_count` / `evidence` / `rerank_reason`（旧消费者不受影响） |
| `recommendation/demo/agent_trajectory.py:184-196` | 新增公开 `presentable_candidates(state)`：呈现集合的唯一真源（硬约束投影在此生效），排序与投影共用它，保证"被排序的集合 == 被发布的集合" |
| `recommendation/demo/agent_trajectory.py:199-258` | `build_recommendations_view(..., ranking=None)`：有 ranking 时列表按 M10B 顺序、`rank` 保留基线位置、并填充证据/计数；无 ranking 时行为与之前完全一致 |
| `recommendation/demo/agent_service.py:116,727` | `ENV_MEMORY_RANKING` + `memory_ranking_enabled()`（默认开，显式关闭） |
| `recommendation/demo/agent_service.py:205,247` | 构造函数新增 `memory_ranking` 参数与服务级开关 |
| `recommendation/demo/agent_service.py:563-616` | `run_selection(..., *, preference_snapshot=None)`（调用方可信读取；LLM 无法提供）+ `_execute_plan(..., preference_snapshot=None)`；**`run()`（旧请求声明路径）不传快照**，因此旧端点不受影响 |
| `recommendation/demo/agent_service.py:678-720` | `_memory_ranking(...)`：开关/快照/组件/候选四项任一缺失即返回 `None`；否则在**呈现集合**上调 `rank_with_memory`，query 为本轮用户消息 |
| `recommendation/demo/agent_chat_adapter.py:161-200` | 卡片：`reranked_rank` / 三类计数 / `evidence`（复用 `evidence_views`）/ `movement_summary`；分数与 provenance 不变 |
| `recommendation/demo/agent_chat_adapter.py:202-243` | `_audit_for(cards, ranked_with=...)`：`reranking_applied` = 是否有卡片带位置；`original_order` 按基线位置排序、`reranked_order` 按记忆位置排序、`moved_count` 为位置改变的卡片数；`ranked_with_preferences` 仅在真的重排时给出 |
| `recommendation/demo/agent_chat_adapter.py:295` | `build_agent_chat_response(..., ranked_with=...)` |
| `recommendation/demo/serialization.py:175` | `_evidence_views` → 公开 `evidence_views`（同一条转换逻辑，避免第二份实现） |
| `recommendation/demo/llm_agent.py:293-315` | `LLMAgentTurn.run(*, preference_snapshot=None)`：记忆在**执行时**由可信调用方传入，与"模型看到的上下文"分离 |
| `recommendation/demo/runtime.py:524-576` | 把**写前**快照传给执行（`turn.run(preference_snapshot=snapshot)`）、`ranked_with=active_preference_views(snapshot)`（面板=写后、审计=写前，两者可就区分）、并把真实重排写进执行证据 |
| `tests/test_demo_memory_ranking.py`（**新增，16 项**） | 适配层 / M10A-M10B / 真实 Controller+Ledger / Chat API / 开关 / 旧端点 |
| `tests/test_llm_selected_controller.py:126-133,141-146` | stub 的 `_execute_plan` 接受新关键字参数（并新增一条断言：selection 自带快照必为 `None`）——仅同步测试替身签名，不改变任何断言含义 |
| `docs/USAGE.md`（llm_agent 记忆段） | 描述新的排序行为、响应字段、开关与边界 |

未改：`CandidatePlane` / `GroundingVerifier` / `CandidateLedger` / `LoopController` / `DemoSourcePlanPolicy`（`use_memory=False` 隔离不变）/ M10A / M10B / M8 / M9 / `rrf` 算法 / `/v1/recommend` / SASRec 等模型与 checkpoint。

## 5. 无 Memory 的基线一致性证据

```text
同一 plan[history,catalog_search] + 同一请求，无 ACTIVE Memory：
  cards        : [blue, green, red, black]        （= 平面 bm25 顺序 = 融合顺序）
  reranked_rank: [None, None, None, None]
  evidence     : []（每张卡）
  audit        : {applied: false, candidate_count: 4, moved_count: 0,
                  original_order: [blue, green, red, black],
                  reranked_order: [blue, green, red, black],
                  ranked_with_preference_count: 0, ranked_with_preferences: []}
纯 history 运行（无 Memory）：[red(0.9), blue(0.8), black(0.7), green(0.6)]，score_kind=sasrec_logit
```

* 断言：`test_without_memory_the_order_is_the_fused_order_byte_for_byte`（含 audit 全字段相等）、
  `test_no_active_preference_means_no_ranking_and_no_enrichment`（连 enrichment 都不发生）。

## 6. 正向、负向、更正、撤销的实际商品排序对照

同一候选池 `{red, blue, black, green}`（真实 metadata：`details.Color`）：

| ACTIVE Memory | M10A 证据（真实） | 最终顺序 | moved | ranked_with |
|---|---|---|---|---|
| （无） | —— | `blue, green, red, black`（融合/历史基线） | 0 | 0 |
| `color=prefer:black` | black: **match**(details.Color=black)；red/blue/green: unknown | `black, blue, green, red`（history 场景）/ `black, blue, green, red`（融合场景） | 3 / 4 | 1 |
| `color=prefer:black` + `color=avoid:red` | black: match + unknown；red: **violation**(details.Color=red)；blue/green: unknown×2 | `black, blue, green, red` | 2 / 4 | 2 |
| 更正为 `prefer:green`（supersede black） | green: **match**；其余 unknown；**旧值 black 不出现在任何 evidence 里** | `green, blue, red, black` | —— | 1 |
| 撤销（REMOVE） | 无 ACTIVE ⇒ 不重排 | 回到基线 `blue, green, red, black`，`reranked_rank=None`，`applied=false` | 0 | 0 |

* 关键点：**"颜色不同"永远不是 violation**（unknown），只有 `avoid red` 命中红色商品才是 violation；
  没有任何由生成式模型补写的属性，也没有 LLM 主观判断参与证据。
* 对应测试：`test_a_preference_promotes_the_candidate_whose_metadata_matches`、
  `test_an_avoidance_demotes_the_candidate_that_violates_it`、
  `test_two_preferences_order_the_pool_by_the_canonical_key`、
  `test_add_supersede_and_remove_change_only_the_ranking`。

## 7. 多来源候选的 item_id、融合分数、来源与最终 rank 对照

真实 plan `["history", "catalog_search"]`，真实请求串（4 个标题词），真实 plane + ledger + RRF：

| item_id | parent_asin | 来源（provenance） | source_rank | source_score（bm25） | score_kind | 融合顺序（=原 rank） | 记忆 rank（prefer black + avoid red） | M10A 证据 |
|---|---|---|---|---|---|---|---|---|
| 902 | cand-blue | catalog_search（`source_query` = 本轮请求） | 1 | 1.311258 | bm25 | 1 | 2 | unknown×2 |
| 903 | cand-green | catalog_search | 2 | 1.311258 | bm25 | 2 | 3 | unknown×2 |
| 901 | cand-red | catalog_search | 3 | 1.311258 | bm25 | 3 | 4 | violation + unknown |
| 904 | cand-black | catalog_search | 4 | 0.966693 | bm25 | 4 | 1 | match + unknown |

* **不删除/不添加商品**：两边的 identity 集合完全相同（测试同时断言 4 ↔ 4、`reranked_rank` 为 1..4 的排列）；
* **不修改 RRF 分数**：两轮 run 的 provenance 记录逐字段相同，且用 ledger 自己的公开函数
  `reciprocal_rank_fusion` 对两轮 provenance 重算，结果完全一致（`test_memory_never_changes_identity_score_kind_or_provenance`）；
* **不修改 source provenance**：`source` / `source_rank` / `source_score` / `score_kind` / `source_query` 全部不变；
* **不伪造 SASRec score**：平面候选的 `score` 仍是 accepted 的 0.0（承载在 `ToolRecommendation`），
  响应里发布的 `sasrec_score` 是 ledger 的 bm25 值 + `score_kind="bm25"`（与该能力之前完全一致）；
  适配层另有一条测试断言"送进 M10A 的分数槽位是 0.0，绝不是 BM25 值"（`test_the_model_score_slot_never_carries_a_fusion_score`）；
* **顺序变化可追溯到真实匹配证据**：见 §6 的 M10A 三态记录（含 `metadata_field=details.Color` 与读到的值）。

## 8. 缺失商品 metadata 的处理

```text
只给 cand-black 建档、其余三个 identity 无目录记录：
  cand-black : match=1  violation=0  unknown=0   （details.Color=black）
  cand-red   : match=0  violation=0  unknown=1   ← 缺失 ⇒ unknown，既不是 match 也不是 violation
  cand-blue  : 同上        cand-green : 同上
  最终顺序   : [cand-black, cand-red, cand-blue, cand-green]（仅证据决定的位移）
```

* enrichment 返回 `metadata_status="missing"`、`metadata=None`；M10A 对缺失属性一律 `unknown`；
  M10B 的键对 unknown 既不加分也不减分（`test_missing_metadata_is_unknown_and_moves_nothing`）。
* 没有任何测试或代码通过生成式模型补写目录属性；缺失就是缺失。

## 9. Chat API 集成测试结果

`tests/test_demo_llm_agent_memory_ranking` 系列的 16 项（新文件）全部通过，覆盖：

1. 纯适配层：无 ACTIVE ⇒ `None` 且不 enrichment；重排器改集合 ⇒ `MemoryRankingError`；
   M10A 输入分数槽位不含融合分数；缺失 metadata ⇒ unknown；
2. M10A/M10B 行为：prefer ⇒ 提升并给出 `metadata_field=details.Color` 的真实证据；avoid ⇒ 降级为
   violation；两条偏好按 canonical key 排序且集合不变；
3. 真实 Controller/Ledger 多来源：两来源都执行、4 个候选带 bm25 provenance、记忆重排且身份/分数/
   provenance/RRF 不变；
4. Chat `llm_agent` API：ADD → 下一轮重排（卡片带证据、audit 如实）；
   **本轮"读了再写"语义**：同一轮既陈述偏好又请求推荐时，排序用的是**写前**记忆
   （`audit.ranked_with_preferences=[black]`，而面板已是 `{black, green}`）；
5. 无 Memory 基线、supersede、REMOVE、`AGENTRECX_AGENT_MEMORY_RANKING=0` 回归；
6. 旧 `/v1/demo/agent/recommend`（带 `user_key`、库里确实有偏好）**不被本能力重排**：顺序与基线一致、
   `reranked_rank` 全为 `None`。

## 10. 旧路径回归结果

单次 pytest 调用（27 个文件：E-02A 新增 + E-01B/E-01A 全部相关套件 + M8/M9/M10A/M10B + Agent API +
Demo API + 控制平面 + 归档回放）：

```bash
.venv/bin/python -m pytest tests/test_demo_memory_ranking.py \
  tests/test_demo_llm_agent_memory_acceptance.py tests/test_demo_llm_agent_memory.py \
  tests/test_demo_llm_agent_mode.py tests/test_demo_agent_api.py tests/test_llm_tool_selection.py \
  tests/test_llm_selected_controller.py tests/test_agent_chat_adapter.py \
  tests/test_agent_state_isolation.py tests/test_agent_control_plane.py tests/test_demo_api.py \
  tests/test_demo_multiturn.py tests/test_demo_llm_mode.py tests/test_demo_preference_pipeline_fixes.py \
  tests/test_memory_query_adapter.py tests/test_memory_bm25_retrieval.py tests/test_preference_matching.py \
  tests/test_preference_reranking.py tests/test_agent_reranking.py tests/test_agent_memory.py \
  tests/test_provider_record_replay.py tests/test_demo_control_plane.py \
  tests/test_demo_recommendation_trace.py tests/test_demo_sessions.py tests/test_demo_web.py \
  tests/test_demo_fusion_view.py tests/test_llm_agent_availability_guard.py -q
```

```text
797 passed in 72.79s      （0 failed；本轮新增 16 项）
```

* Legacy `deterministic` / `llm` Chat：不回归（`test_demo_multiturn.py`、`test_demo_llm_mode.py`、
  `test_demo_preference_pipeline_fixes.py` 全绿，M9 ADD/REMOVE/supersede 语义逐条通过）；
* Agent Recommend 不回归（`test_demo_agent_api.py`），且新增一条"旧端点不被重排"的显式断言；
* 每请求 Controller/Policy/Ledger 隔离不回归（`test_agent_state_isolation.py`，含并发用例）；
* Serving `/v1/recommend` 未改动（`test_demo_api.py` 的 M6 兼容用例通过，且其响应 schema 与本轮无关）；
* 归档 provider 回放（`test_provider_record_replay.py`）全绿：action 序列未变；
* **唯一需要同步的既有测试**是 `tests/test_llm_selected_controller.py` 的 `_StubService`：它复刻了私有
  `_execute_plan` 的签名，`run_selection` 新增关键字参数后必须接受它。该处**只同步签名并新增一条更强断言**
  （selection 自带快照必须为 `None`），没有放宽或删除任何既有断言。

## 11. 仍未解决的偏好相关性与真实模型质量问题

1. **相关性未解决**：本轮把**全部 ACTIVE 偏好**用于重排，没有做"与当前请求相关"的筛选。当历史偏好与
   当前请求无关时，M10A 会给出 `unknown`（不匹配也不违反），因此**不会**强行改变顺序——这是当前机制
   的天然保护，但它是"证据不足 ⇒ 不动"，不是"识别出无关并忽略"。领域相关性分类器（§7 明确不做）留待后续。
2. **多来源 presented pool 的边界（已由 E-02A-V 修复）**：本报告写作时 `_adopt_plane_candidates` 让
   **最后一个候选产出步骤**的集合成为呈现集合，因此一次两来源 run 只能看到最后一个来源的候选；
   本轮测试当时如实地把 `multi_source_count == 0` 与"呈现集合只含 catalog_search"记录为**当时的真实行为**。
   E-02A-V 已复现该缺陷并把呈现集合改为"本轮全部已执行来源的 grounded 候选按既有 RRF 融合"
   （见 [`docs/V2_E02A_V_REPORT.md`](V2_E02A_V_REPORT.md)），本报告中 E-02A 的自己测试已随契约变更更新
   （更强的来源/融合/身份断言）。仍存的边界：`RECOMMEND_FROM_HISTORY` 走 capability 因而不写 ledger
   provenance，所以 `multi_source_count`（plane 视角的重复计数）在 history+plane run 中仍为 0，
   而 `sources_present` 会如实列出两个来源。
3. **真实 Provider 与真实模型资产未联调**：模型决策仍来自受控 `ScriptedModelClient`；真实 provider 下
   "模型是否会更倾向使用历史偏好"未经验证，也不构成本轮结论。真实 SASRec checkpoint / TIGER worker 未加载
   （夹具为确定性轻量 engine + 合成目录）。可复现步骤见 `docs/USAGE.md` 的 "Real-provider smoke test"。
4. **没有质量指标**：未做 NDCG/Recall 实验，本轮只证明"排序可被真实偏好证据解释地改变"，不主张质量提升。
5. **M10A 的 `sasrec_score` 槽位语义**：平面候选携带 accepted 的 0.0 占位（既有设计），适配层**不发布**该槽位，
   响应的分数一律来自 ledger + `score_kind`。若未来要让 M10A 的命名与多来源语义完全对齐，应作为一次
   显式的 schema 迁移处理（本轮不做，避免大规模契约变更）。

## 12. 结束时 HEAD、git status

```text
$ git rev-parse HEAD
faeff0b6f790d8c9d077d7d07ee26998aeecad2a          （未提交、未 reset）

$ git status --short
开始：25 modified + 33 untracked
结束：26 modified + 36 untracked
```

* 新增 3 个未跟踪文件：`recommendation/demo/memory_ranking.py`（本轮新代码）、
  `tests/test_demo_memory_ranking.py`（16 项测试）、`docs/V2_E02A_REPORT.md`（本文件）；
* modified 计数 25 → 26：本轮唯一**新**进入 modified 集合的已跟踪文件是
  `recommendation/demo/agent_trajectory.py`（其余被编辑的 `agent_schemas.py`、`agent_service.py`、
  `serialization.py`、`runtime.py`、`docs/USAGE.md` 在更早轮次就已是 modified；`agent_chat_adapter.py`、
  `llm_agent.py`、`tests/test_llm_selected_controller.py` 是更早轮次新增的未跟踪文件）；
* 前序 TIGER / Demo / M-02A / Memory / E-01A / E-01B 成果与 8 份文档逐条确认仍在，无删除、无回退；
* 未执行 `commit` / `add` / `reset` / `checkout` / `restore` / `clean` / `stash`；未安装依赖；
  未训练模型；未调用在线 LLM；未修改任何模型推理算法或 checkpoint。

## 13. 最终结论

**PASS。** 完成标准要求的闭环得到执行型证据：

```text
相同合法召回候选（同一 plan、同一请求、同一 plane/ledger）
  + 不同 ACTIVE Memory（无 / prefer black / prefer black+avoid red / 更正 / 撤销）
  → M10A 产生不同**真实**偏好证据（match / violation / unknown，含 details.Color 读数）
  → M10B 产生可解释的最终排序变化（含 moved_count、original_order 与 reranked_order）
  → 商品身份、item_id、原始分数、score_kind 与来源 provenance 全部保持不变
```

同时：无 Memory 时顺序与融合顺序逐字节一致；撤销/更正可回到基线；跨 Session 隔离与身份绑定沿用
E-01B-1（未改）；旧 `deterministic` / `llm` / Agent Recommend / Serving 路径与 M9 语义不回归；
开关 `AGENTRECX_AGENT_MEMORY_RANKING=0` 可固定 E-01B 基线。§11 列出的真实 Provider 联调、
相关性筛选与 RRF 呈现顺序边界均为本轮明确保留的范围。本轮到此停止，不进入 E-02B。
