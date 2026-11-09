# P14-0 接口边界与最小契约（Shopping Companion / Search / Preference-Aware Recommendation / Agent Benchmark）

> 性质：**P14-0 只做架构准备**。本文档不实现 Shopping Companion、不训练模型、不改行为，只把现有代码的边界、可复用组件、最小契约和缺口（GAP）写成 P14 可直接对照实现的说明。
> 证据口径：直接阅读仓库源码（分支 `feat/agent-source-limit-ablation`）；下列每个英文 symbol 均已 grep 核对存在；未运行测试/服务/训练（P14-0 约束）。质量口径沿用 P13-R：P13-R = **LIMITED**（`P13R_REPORT.md`），工程质量 PASS 与质量结论是两件事，本文不产生质量结论。

## 0. 一句话结论

P14 不需要第二套 Agent：M5 事件状态、P3D Planner Action 契约、M9 偏好记忆、`CandidateLedger`、`CandidateEligibilityEvaluator`、Evidence、Trajectory 均已存在。P14 真正缺的是 **catalogue namespace、Search/Recommend 两个可替换后端的契约 + ProductInfo/Memory 两个工具契约、P14 benchmark 数据与 evaluator 版本**，以及 Planner 契约到运行时 `LoopController` 的桥接。

## 1. 三个代码边界

### 1.1 Legacy Recommendation（冻结，只读复用）

定义：Amazon Reviews 2023 **Sports_and_Outdoors 5-core** 的推荐底座与评测协议（SASRec / Two-Tower / Catalogue BM25 Serving V1 + checkpoint + mapping + `agentrecx.eval_protocol.v1`）；P14 只经服务级接口只读调用。

- 身份：`recommendation/preprocess.py::build_id_mapping`（`parent_asin -> int item_id`，从 `config.FIRST_REAL_ID=1` 起，`config.PAD_ID=0` 保留）；`recommendation/inference/sasrec.py::load_item_mapping` 加载 mapping，`SASRecInferenceEngine.has_parent_asin/parent_asin_to_item_id/item_id_to_parent_asin` 为 identity 三方法。
- Serving：`recommendation/inference/{sasrec,twotower,lexical,fusion}.py`；`fusion.py` 固定 `FUSION_SOURCES=("sequential","two_tower","catalog_lexical")`、`FUSION_SCORE_KIND="rrf_rank_fusion"`、`RRF_K=60`、`DEFAULT_SOURCE_HEAD=100`、profile `serving_v1`。默认入口 `recommendation/api/app.py` 的 `POST /v1/recommend`，wire 契约在 `recommendation/api/schemas.py`（`RecommendRequest.history: list[parent_asin]`、`RecommendationItem.item_id + parent_asin`）。
- 资产/口径：`data/processed/Sports_and_Outdoors_{mappings,sequences}.json`；`runs/sasrec_canonical_2026/best.pt`、`runs/twotower_public_2026/checkpoint.pt`；`data/p6_benchmark_v0/`。`P6_CLASSICAL_EVAL_REPORT.md`：实现 PASS、质量 UNVERIFIED、Fusion 严格 full-ranking BLOCKED。
- 硬规则：**旧 Sports 整数 `item_id` 绝不能被新 companion 悄悄复用**，它只在其 mapping（`mappings_sha256`）下有含义；`recommendation/contracts/namespaces.py` 已定 `LEGACY_INTEGER_NAMESPACE="amazon_sports_5core"` 与 `PAD_ITEM_ID="0"`。

### 1.2 Shared Agent Core（P14 增强，不重写）

定义：与目录/数据集无关的 Agent 决策与证据内核；P14 只允许加适配层，不允许复制或分叉。当前 owner：

| 组件 | 当前实现（文件 + 关键 symbol） |
| --- | --- |
| Memory state | `recommendation/state/`（M5 事件状态）；`recommendation/memory/`（M9 显式偏好写入） |
| Planner decision | `recommendation/demo/planner_contract.py`（P3D 契约）+ `planner_service_executor.py` + `planner_http_proposer.py`；运行时另有 `recommendation/control/schemas.py::ActionKind` |
| Tool contract | `recommendation/tools/{schemas,recommendation}.py`；后端开口 `recommendation/control/candidate_plane.py::CandidateSourceTool` |
| CandidateLedger | `recommendation/control/candidate_ledger.py::CandidateLedger` |
| Hard Constraint / Eligibility | `recommendation/control/grounded_reasoning.py::GroundedReasoner` + `recommendation/control/constraint_eligibility.py::CandidateEligibilityEvaluator` |
| Evidence Provenance | `recommendation/control/evidence.py::EvidenceItem/EvidenceStore` + `evidence_executor.py` + `recommendation/rag/schemas.py::ProductEvidence` |
| Agent Trajectory | `recommendation/control/trajectory.py::TrajectoryRecorder` + `recommendation/demo/agent_trajectory.py` |
| Evaluator | `recommendation/evaluation/`（ranking）与 `benchmarks/agent/`（task-level），见 §6 |

必须遵守：该内核**不得硬编码 benchmark 专用字段或商品类目**。当前类目特化只允许留在 legacy demo 层：`recommendation/demo/product_role.py`（`water_filter` 等类型词）与 `recommendation/demo/shopping_intent.py`（`_water_filter_query`、`_is_bottle_accessory_without_filter_device`）是 Sports/water-filter 专用 Role Gate；`recommendation/control/` 与 `recommendation/state/` 均不 import 它们（已 grep 确认），P14 不得把这些规则提升进内核或套用到新目录。

### 1.3 Dataset / Catalog / Backend Adapter（P14-0 只定义责任）

定义：把未来数据集、商品目录、检索后端、推荐打分器、商品信息提供方、评测器接入 Shared Agent Core 的薄适配层；P14-0 只声明职责与最小接口，不实现具体后端。

| 边界 | 允许 P14 做什么 | 禁止什么 |
| --- | --- | --- |
| Legacy Recommendation | 只经 `RecommendationEngine` / `FullCatalogueScorer` 只读调用 | 改 serving 默认、checkpoint、item_id 空间、复用旧类目快照 |
| Shared Agent Core | 增加与目录无关的参数/适配器（如 namespace），保持语义 | 第二套 Memory/Planner/Ledger/Eligibility；把 benchmark 字段写进内核 |
| Adapter 层 | 每个后端一个适配模块 + 契约测试 | 通用插件框架/工作流引擎；跨目录 id 静默归一 |

## 2. Shared Agent Core 复用表（symbol 均经 grep 核对）

| 能力 | 真实实现（文件::symbol） | 现有测试 |
| --- | --- | --- |
| Memory state（M5 事件溯源） | `recommendation/state/decision_state.py::UnifiedDecisionState`；`event_store.py::SQLiteEventStore`；`belief_updater.py::BeliefUpdater`；`state_projector.py::for_agent/for_ranker/for_constraints`；`state_manager.py::StateManager`；`shopping_adapter.py::build_turn_events/resolve_turn_identity`；`legacy_adapter.py::project_legacy_memory` | `tests/test_state_schemas.py`、`test_state_event_store.py`、`test_state_beliefs.py`、`test_state_projector.py`、`test_state_manager.py`、`test_state_legacy_adapter.py`、`test_m5_state_acceptance.py` |
| Memory（M9 显式偏好） | `recommendation/memory/service.py::PreferenceMemoryService.get_active_preferences/get_memory_history/process_turn`；`schemas.py::PreferenceMemoryEntry/PreferenceMemorySnapshot/PreferenceStatus`；`query_adapter.py::project_active_preferences/ActivePreference`；`store.py::InMemoryPreferenceStore/SQLitePreferenceStore` | `tests/test_memory_service.py`、`test_memory_store.py`、`test_memory_query_adapter.py`、`test_memory_bm25_retrieval.py`、`test_agent_memory.py` |
| Planner decision（ASK/RETRIEVE/RERANK/FINALIZE） | `recommendation/demo/planner_contract.py::PlannerAction`（ASK/RETRIEVE/RERANK/FINALIZE）、`PlannerObservation`、`ActionProposal`、`allowed_planner_actions`、`validate_action`、`run_planner`；`planner_service_executor.py::ServicePlannerExecutor`；`planner_http_proposer.py::HttpPlannerProposer`；prompt `training/agent_sft/planner_prompt.py` | `tests/test_planner_contract.py`、`test_planner_sft_v0.py` |
| Tool contract | `recommendation/tools/schemas.py::RecommendationToolRequest/RecommendationContext/RecommendationToolResult/ToolRecommendation`；`tools/recommendation.py::RecommendationTool`（`TOOL_NAME="recommend_products"`、`TOOL_VERSION=1`） | `tests/test_recommendation_tool.py`、`test_recommendation_tool_integration.py`、`test_agent_tool_e2e.py` |
| CandidateLedger + Source Policy | `recommendation/control/candidate_ledger.py::CandidateLedger/LedgerEntry/CandidateProvenance/CandidateVerificationStatus`、`record_from_source`、`ranked_parent_asins`、`reciprocal_rank_fusion`、`weighted_reciprocal_rank_fusion`、`SOURCE_POLICIES/SOURCE_POLICY_WEIGHTS/normalize_source_policy`；`candidate_plane.py::CandidatePlane/CandidateSourceTool`；`demo/source_limit.py::resolve_source_limit` | `tests/test_control_stage2_candidates.py`、`test_multi_source_fusion_presentation.py`、`test_candidate_reference_boundary.py`、`test_agent_state_isolation.py`、`test_p8c_source_policy.py` |
| Hard Constraint / Eligibility | `control/grounded_reasoning.py::ConstraintKind/ConstraintVerdict/ConstraintReport/GroundedReasoner`；`control/constraint_eligibility.py::CandidateEligibilityEvaluator/CandidateEligibility/FeasibleCandidateView` | `tests/test_constraint_eligibility.py`、`test_m6_1_contract.py`、`test_p3_constraint_empty_message.py` |
| Evidence Provenance | `control/evidence.py::EvidenceStatus/EvidenceItem/EvidenceStore/LocalEvidenceArtifact`；`evidence_executor.py::EvidenceExecutor`；`rag/schemas.py::ProductEvidence.provenance`；`catalog/schemas.py::ProductMetadata.source/details` | `tests/test_phase4_evidence_similar_item.py`、`test_p12_evidence.py`、`test_control_stage4_reasoning.py` |
| Agent Trajectory | `control/trajectory.py::TrajectoryRecorder`（`record/as_dicts/actions/refusals`）；`control/schemas.py::TrajectoryStep/VerificationResult/StateChange`；`demo/agent_trajectory.py::build_trajectory_summary/build_grounded_view/build_eligibility_view` | `tests/test_control_stage3_interaction.py`、`test_control_stage4_wiring.py`、`test_demo_recommendation_trace.py` |
| Evaluator | 见 §6（`recommendation/evaluation/` + `benchmarks/agent/`） | 见 §6 |

## 3. 四个最小工具契约（in/out shape + 当前映射 / GAP）

### 3.1 Search（current query → relevant products）

| 方向 | 字段（类型）：含义 |
| --- | --- |
| in | `query: str` 当前查询（用户请求或 Planner/R1 改写后的检索 query，untrusted）；`catalog_namespace: str` 目标目录命名空间（P14 新增） |
| in | `top_k: int` / `limit: int` 返回条数；`budget: int` 本次工具预算；`retrieval_options: dict`（`mode`/`fields`/`source_policy` 等，后端自定义且必须可记录） |
| out | `hits: list[{item_id, parent_asin?, rank, score, score_kind, matched_fields}]` 按该源自身顺序；`source: str` 后端名（必须落进 ledger 的 `CandidateSource`） |
| out | `relevance_score: float` + `score_kind: str` 原始分数及含义（`bm25`/`bge_dense`/`hybrid_bm25_bge`，禁止跨源比较）；`retrieval_metadata: dict`（`query/matched/catalogue_size/index_version/provenance`） |

当前映射：`recommendation/control/catalog_search.py::CatalogSearchSource.search/candidates` → `CatalogSearchResult`（`hits: tuple[CatalogSearchHit]`，`CatalogSearchHit.parent_asin/score/matched_fields/title`），`CATALOG_SEARCH_SCORE_KINDS` 固定 score kind 词表；策略层入参是 `control/arguments.py::SearchCatalogArguments.terms`（结构化 terms，非自由 query），`candidate_plane.py::_CatalogSearchSourceTool` 装配 query。
GAP：① 无 `catalog_namespace`；② 无只读后端注册 seam（`CandidatePlane.__init__` 是具名 slot，新源要改 `CandidateSource` 枚举 + `CANDIDATE_PRODUCING_SOURCES`，该表当前属冻结行为）；③ P13-R 的 BM25/BGE fixed-pool 检索只在 `experiments/p13r/retrieval.py`，未接入 serving 源。

### 3.2 Recommend（long-term preference + current need → rank/match FIXED candidates）

| 方向 | 字段（类型）：含义 |
| --- | --- |
| in | `user_preferences: list[ActivePreference]`（`memory_id/kind/polarity/value/status/logical_seq`）；`current_query: str` 当前 need（untrusted） |
| in | `candidates: fixed list[{item_id, parent_asin?, namespace}]` **固定候选集**，非空；工具只能在其内排序/匹配，不得重搜全目录 |
| out | `ranked: list[{item_id, rank, score?, score_kind}]` 只重排不增删；`matched_preferences: list[PreferenceEvidence]` 命中项（含元数据字段/值 provenance） |
| out | `conflicting_preferences: list[PreferenceEvidence]` 明确冲突项；`unknown_fields: list[{preference_id, support, reason_code}]` UNKNOWN/证据不足；`evidence_source: str` 证据来源 |

当前映射：固定集内的“匹配 + 重排”已存在但拆在两个模块 —— `recommendation/preference_matching/matcher.py::PreferenceCandidateMatcher/match_candidates` → `schemas.py::PreferenceEvidenceReport/PreferenceEvidence`（`EvidenceStatus` = MATCH/VIOLATION/UNKNOWN；`ReasonCode`；`MatchingSupport` = SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED_FOR_MATCHING；`metadata_field`/`metadata_value`）；`recommendation/reranking/reranker.py::PreferenceReranker.rerank` → `RerankingReport`，排序键冻结为 `RERANK_SORT_KEY_DOC="violation_count ASC, match_count DESC, original_rank ASC, item_id ASC"`。
GAP：① 无单一“Recommend Backend”工具接收 `(preferences, query, FIXED candidates)`；`recommendation/tools/recommendation.py::RecommendationTool.run` 是从 history 经 engine 生成候选（非固定集重排），其 `RecommendationToolRequest` 明确不接受 query/preferences；② 偏好证据未进 `CandidateLedger`，P14 若要“证据即 provenance”须新增字段而非复用现有 `CandidateProvenance`；③ 硬约束仍必须由 Eligibility 执行（§5），Recommend 后端不得旁路。

### 3.3 Memory（session/user context + requested scope → usable preferences）

| 方向 | 字段（类型）：含义 |
| --- | --- |
| in | `user_key: str` 记忆命名空间；`task_id: str \| None` 任务上下文；`requested_scope: str` = `task`/`session`/`long_term`（M5 `PreferenceScope`）；`filter: {kinds?: list[str], max_entries?: int}` |
| out | `preferences: list[{memory_id, kind, polarity, value, status, logical_seq}]` 可用偏好；`source_event_id`/`source_text`/`source_turn_id`/`evidence` 来源事件与证据 span |
| out | `effective_status`：ACTIVE/SUPERSEDED/REMOVED + M5 lifecycle（INFERRED_HYPOTHESIS/EXPLICIT/CONFIRMED/RETRACTED）；`timing/conflict`：`logical_seq/created_at/read_at`、`supersedes/superseded_by`、task 覆盖 long-term 的 suppressed ids |

当前映射：M9 读面 `recommendation/memory/service.py::get_active_preferences(user_key)` / `get_memory_history(user_key)` → `PreferenceMemorySnapshot`（`entries`/`active_entries`）；最小投影 `recommendation/memory/query_adapter.py::project_active_preferences` → `ActivePreferenceProjection`（只复制 6 字段，removed/superseded 不进入）；M5 读面 `recommendation/state/state_manager.py::StateManager.load_or_create/replay_state` + `state_projector.py::for_agent/for_ranker/for_constraints`；M9→M5 只读桥 `recommendation/state/legacy_adapter.py::project_legacy_memory`。
GAP：无统一 Memory 工具读契约（scope + provenance + effective status + conflict 一次返回）；M9 与 M5 是两个 store、两套读 API，P14 必须**组合**而非合并/迁移。

### 3.4 Product Info（product id + namespace + requested attributes）

| 方向 | 字段（类型）：含义 |
| --- | --- |
| in | `product_id: str` + `catalog_namespace: str` 商品身份；`requested_attributes: tuple[str, ...]` 请求属性（闭集，超出返回 `unsupported`） |
| out | `status: "ok" \| "empty"`；`attributes: dict[str, ProductFact]`，`ProductFact={status: value/missing/unreadable/unsupported, raw, value, value_type, unit, derived, source, verification}` |
| out | `attribute_provenance`：`source`（如 `catalog:details:Brand Name`、`catalog:price_text`）+ `catalog_source`；`missing_fields/unreadable_fields/unsupported_fields`（缺失与 unknown 状态，绝不填默认值） |

当前映射：`recommendation/control/product_research.py::ProductResearchTool.inspect_product/compare_products`、`inspect_record`、`_fact_for`、`SUPPORTED_ATTRIBUTES/DEFAULT_ATTRIBUTES` → `ProductInspection/ProductFact`（四态 status + `verification`）；控制面接线 `control/reasoning_executor.py::_inspect_product`（`ActionKind.INSPECT_PRODUCT`，先校验“必须是本 run 的 grounded 候选”再读目录）；只读目录协议 `recommendation/catalog/schemas.py::MetadataLookup.lookup/lookup_many` 返回 `ProductMetadata | MissingMetadata`；RAG 证据 `recommendation/rag/schemas.py::ProductEvidence`。
GAP：① 所有实现均无 `catalog_namespace` 参数，`_record(records, parent_asin)` 只在一个 records mapping 内解析；② `ProductResearchTool` 只接受已加载 records 映射，无 namespace→目录解析层；③ control-plane 路径只允许读本 run 的 grounded 候选，P14 若需“按 id 查任意目录商品”须新增只读入口并复用 `ProductFact` 形状。

## 4. Catalogue namespace 规则

### 4.1 当前隐式身份约定（已核对）

| 位置 | 当前约定 |
| --- | --- |
| `recommendation/preprocess.py` + `recommendation/inference/sasrec.py` | Sports 5-core：`parent_asin ↔ int item_id`，`PAD=0`，mapping 即身份权威 |
| `recommendation/catalog/schemas.py` / `metadata.py::MetadataIndex` | `parent_asin` 为 opaque key；`ProductMetadata.parent_asin` + `source` 记来源；accepted mapping 为权威 |
| `recommendation/control/grounding.py::GroundingVerifier` + `candidate_ledger.py::LedgerEntry` | `parent_asin` 是 entry key；`CandidateProvenance.item_id: int \| None` 仅作同一 mapping 的附注 |
| `recommendation/control/catalog_search.py` | `MetadataIndex.records` 的 key（即 `parent_asin`）就是检索结果身份 |
| `recommendation/backends/tiger_backend.py` | 后端只见 `item_id`、`PAD_SENTINEL=-1`，**禁止** `parent_asin` 越界；身份在 adapter 外重新挂接 |
| `recommendation/demo/agent_service.py` | 全链路父身份 `parent_asin`（history、ledger、response 均为 ASIN 字符串） |
| P13-R 数据/代码 | `data/p13r/electronics/tasks.jsonl` 用 `target_parent_asin`（Amazon ASIN，如 `B0C5PHZ4QS`）；`catalog_meta.jsonl` 的 `item_id` 多为 ASIN，但 76,918 条中 72 条非 ASIN（如 `007329506X`）；`experiments/p13r/retrieval.py::load_catalog` 把 C4 `item_id` 直接映射成 `SimpleNamespace(parent_asin=item)` |

**已发现的不一致**：`experiments/p13r/prepare_data.py` 写出的是 `catalog.jsonl`，但仓库数据目录中是 `catalog_meta.jsonl`（无对应 in-repo producer）；P13-R 报告写“身份：item_id / parent_asin”，实际代码把 C4 `item_id` 当作 `parent_asin` 使用。P14 复用 P13-R catalogue 前必须显式登记 namespace 与 producer/hash，不能默认其等价于 Amazon `parent_asin`。

### 4.2 P14 规则（contract seed 已落地）

Lead 已新增 `recommendation/contracts/namespaces.py`（+ `recommendation/contracts/__init__.py`、`tests/test_contracts_namespaces.py`），作为 P14-0 契约种子：

- 跨边界商品身份用 `ProductRef{namespace: CatalogNamespace, item_id: str, parent_asin?}`；`namespaced_id` 规范序列化为 `"<namespace>:<item_id>"`；`ensure_same_namespace(*refs)` 混用时报 `NamespaceMismatchError`。
- `CatalogNamespace.name` 规则 `^[a-z][a-z0-9_]{1,63}$`；`KNOWN_CATALOG_NAMESPACES` 记录现存命名空间：`amazon_sports_5core`（整数 id，`PAD_ITEM_ID="0"` 禁止作真实商品）、`amazon_electronics_c4`（P13-R 76,918 items）、`demo_catalog`（本地 demo/CI 合成目录）。
- P14 引入 Shopping Companion 目录时新增命名空间（如 `p14_shopping_companion`），并在**同一次变更**里登记 `KNOWN_CATALOG_NAMESPACES` 且提供该目录 adapter；legacy PAD 拒绝逻辑保护旧 Sports id 空间。
- 不允许静默跨目录融合：ledger 的 RRF 只在同一 namespace 的候选间有意义；P14 若要多目录，必须显式分目录 ledger/评测后再决定是否合并。

### 4.3 落地位置（不回溯改冻结路径）

新契约用于 P14 adapter 的输入输出（Search/Recommend/ProductInfo/benchmark/evaluator），不强行改造 Serving V1 `/v1/recommend` 的 wire 形状（`item_id + parent_asin` 保持冻结）。顺序：`recommendation/contracts/namespaces.py` → P14 adapter 使用 `ProductRef` → ledger/provenance 增加 namespace 字段（新增字段，不改现有语义）→ evaluator 按 namespace 分区。

## 5. CandidateLedger / Eligibility 复用

当前链路（`candidate_plane.py::CandidatePlane` 注释即权威描述）：

```text
source tool --(parent_asin, rank, raw score, score_kind)--> GroundingVerifier.ground/record
  -> CandidateLedger.record_from_source(source, candidates, score_kind, item_ids, ...)
  -> ranked_parent_asins()/fusion_scores()（weighted RRF）
  -> CandidateEligibilityEvaluator.assess(...) -> FeasibleCandidateView.narrow(...)
```

P14 必须原样复用：① 身份去重、证据不合并（一个 `LedgerEntry` 一个商品，多来源=多条 `CandidateProvenance`，raw score 永不相加/平均，只做 rank-based RRF）；② `CANDIDATE_PRODUCING_SOURCES` 白名单 + `GroundingVerifier` 先 grounding 后记录（`record_from_source` 是唯一记录入口，无 public `add`；grounding 失败记 UNGROUNDED 审计项、不展示）；③ `CandidateVerificationStatus`（GROUNDED/UNGROUNDED）与 `SOURCE_POLICIES`（P8-C 冻结）不变；④ Eligibility 是独立视图（`CandidateLedger != FeasibleCandidateView`）：三态 `VERIFIED_ELIGIBLE/INELIGIBLE/UNRESOLVED`，proved violation 优先于 unresolved，UNKNOWN 不算通过也不算违反，`FeasibleCandidateView.narrow` 只保持调用方顺序并删除，不排序不缓存。

新增 Search/Recommend 后端的唯一扩展点：实现 `candidate_plane.py::CandidateSourceTool`（`source`、`score_kind`、`propose(arguments, limit)`）并在 `control/arguments.py::CandidateSource` + `CANDIDATE_PRODUCING_SOURCES` 登记。**这属于冻结行为**：P14 加源必须走显式冻结例外（改枚举 + 改测试 + 记录），不能绕过 plane 直接写 ledger。

## 6. Evaluator 复用（协议不可互换）

| 评测代码 | 权威负责什么 | 协议/口径 | 不得混用 |
| --- | --- | --- | --- |
| `recommendation/evaluation/{split,metrics,evaluator,batched}.py` | next-item ranking 规范评测：`agentrecx.eval_protocol.v1` / `temporal_leave_two_out`，PAD 排除、seen mask、target 保留、tie=低 item id 优先、HR/Recall/NDCG@5/10/20；`batched.py::evaluate_batched` 同语义（oracle 仍是 `metrics.py`） | full-catalogue、单正例 | 与 P6/P13/P14 任何固定 pool 结果不可并列 |
| `recommendation/evaluation/experiment.py` | 模型对比清单/bootstrap（`ExperimentManifest`、`paired_bootstrap_delta`） | 同上 | 只提供不确定性，不定义新指标 |
| `recommendation/evaluation/decision.py` | M8 decision-level 指标（`DecisionTrajectoryRecorder`、`compute_decision_metrics`） | 交互决策过程 | 不是 ranking 指标，不能当 NDCG/Recall |
| `benchmarks/agent/{evaluator,evaluator_v2}.py` + `FROZEN_PROTOCOL.md` | shopping task 确定性断言（final identity/role/forbidden/retrieval_action）与 V2 独立标签（`evaluate_run`/`evaluate_run_v2`） | task-level | 不能当用户满意度；human labels 缺失必须保持 missing |
| `experiments/p6_benchmark/metrics.py` | P6 v0：Recall@10/50、NDCG@10、MRR@10、CatalogueCoverage@10 + dynamic task 指标 | full-catalogue，但 history 有两个版本（`experiments/p6_benchmark/full_ranking.py::VARIANTS`：`p6_shared_history`=`items[:-2]` vs `eval_protocol_v1_test_history`=`items[:-2]+validation_target`） | 两个 history 口径分开报告，不得把 P6 v0 数字冒充 canonical |
| `experiments/p13/retrieval_metrics.py`、`experiments/p13r/` | observed-positive 的 hit/rank proxy（`hit@10/50/100` 为计数、`mrr@10`、`ndcg@10`），固定 pool quota（P13-R：BM25 top50+BGE top50） | 单 observed positive，其余 unjudged；固定 pool≠全目录 | 不能与 full-ranking/canonical 混排，也不能当 sampled-negative 结果 |
| `experiments/p11_agent_exec.py` + `data/p11_benchmark/schema.json` | 构造式 shopping task 的 Agent 执行：strict bounded-pool success / catalogue-valid / InvalidFinalize / hard violation | Sports 任务 + bounded pool | 不是 ranking；不可改名成 frozen test |

**硬警告**（与 `AGENTS.md` §6、`docs/EXPERIMENTS.md` §4 一致）：① 本仓库不存在 sampled-negative 评测结果（`recommendation/evaluation/` 只实现 full-ranking candidate set），任何未来 sampled-negative 结果都不得与 full-catalogue 并列；② 单正例下 `HR@K == Recall@K` 是定义使然，协议一旦多正例 Recall@K 定义就变了 —— “Recall@k”在不同协议里不是同一个量，不能只按指标名合并；③ P6 `protocol.json` 虽名 temporal_leave_two_out，但 test history 与 `split.py` test 模式不同（`P6_CLASSICAL_EVAL_REPORT.md` 已记录），P13-P13R fixed-pool 指标定义也与 P6 不同 —— P14 若要统一必须新开协议版本 + 重冻结 + 预注册，不得直接合并现有 evaluator 或改写旧指标；④ 已冻结的 holdout/dev（P6/P8/P9 Holdout、P11/P12 Dev）不得在 P14-0 运行，也不得改名成新 Test，P13-R reserved test 未运行、未用于选型。

## 7. P14 实际需要新增的 adapter（≤8 项，每项一个模块）

| # | 要建的东西 | 插进哪个现有接口 |
| --- | --- | --- |
| 1 | Planner→runtime 桥（`planner_contract.py` 的 ASK/RETRIEVE/RERANK/FINALIZE 映射到 `control/schemas.py::ActionKind` 并接入 `/agent/shopping`） | `planner_service_executor.py::ServicePlannerExecutor`、`control/loop.py::LoopController`；当前只在 `experiments/p6_benchmark/run_dynamic_dev_p64.py` 使用 |
| 2 | P14 dataset loader（`recommendation/datasets/` 新模块） | `recommendation/state/shopping_adapter.py::build_turn_events`（产出 `StateEvent`）；benchmark 侧对齐 `benchmarks/agent/tasks.py::task_from_mapping/BenchTask` |
| 3 | catalogue namespace 注册（新目录条目） | `recommendation/contracts/namespaces.py::KNOWN_CATALOG_NAMESPACES` + `recommendation/catalog/metadata.py::MetadataIndex/build_metadata_artifact` |
| 4 | Search backend adapter | `control/candidate_plane.py::CandidateSourceTool`（`source/score_kind/propose`）+ `control/arguments.py::CandidateSource`/`CANDIDATE_PRODUCING_SOURCES` |
| 5 | Recommendation/preference scorer adapter（固定候选集 + 偏好重排） | `preference_matching/matcher.py::PreferenceCandidateMatcher` + `reranking/reranker.py::PreferenceReranker`；评测侧接 `experiments/p6_benchmark/full_ranking.py::FullCatalogueScorer` |
| 6 | Product info provider（namespace→catalogue 解析） | `recommendation/catalog/schemas.py::MetadataLookup`（`lookup/lookup_many/__contains__`）+ `control/product_research.py::ProductResearchTool` |
| 7 | P14 benchmark task builder | `benchmarks/agent/tasks.py::load_tasks/validate_tasks` 的 schema 模式 + 新版本 schema（参考 `data/p11_benchmark/schema.json`，不复用 `agentrecx.p11.shopping_task.v1`） |
| 8 | P14 evaluator 版本（不改旧文件） | 复用 `benchmarks/agent/evaluator.py::evaluate_run` 断言模式与 `evaluator_v2.py` 的“独立版本+独立标签”模式；ranking 侧复用 `recommendation/evaluation/` 规范 metric |

## 8. 明确“不要建”（P14 禁止项）

1. **不要第二个 Agent runtime**：不重新实现 `LoopController`/`AgentGraph`/planner loop/Agent 请求响应 schema；只在现有 runtime 上加 adapter/参数。
2. **不要第二条 serving path**：`POST /v1/recommend` → `FusionInferenceEngine` 是默认路径，不新增平行推荐服务；P14 只经 `RecommendationEngine`/`FullCatalogueScorer` 只读接入。
3. **不要插件框架 / 通用 workflow 引擎**：没有 `CandidateSource` 动态注册中心、没有“任意后端自动发现”；每个后端一个薄模块 + 契约测试。
4. **不要复制 Memory/Planner/Ledger/Eligibility**：不能出现第二个 `CandidateLedger`、第二套 M5 event schema、第二套 Eligibility 三态；只能扩展字段/枚举并保留原语义。
5. **不要过早做完整 adapter 框架**：先接一个目录 + 一个搜索后端 + 一个偏好来源，跑通再抽象；P14-0 只列 §7 的 8 项。
6. 不新增 `CandidateSource` 成员或改 `SOURCE_POLICY_WEIGHTS`/RRF 常量而不走冻结例外；不把 P6/P13 指标并入 `recommendation/evaluation/`；不跨 namespace 合并 id、不做隐式 `parent_asin`/`item_id` 互转、不把旧 Sports id 当新 companion id。

## 9. 冻结行为清单（P14 不得改变）

1. **Planner action semantics**：`demo/planner_contract.py::PlannerAction` 的 ASK/RETRIEVE/RERANK/FINALIZE 及 `allowed_planner_actions`/`validate_action` 约束（RETRIEVE 需 query、RERANK/FINALIZE 需候选、终态后无 action、max_steps）。
2. **M5 事件溯源规则**：`recommendation/state/` 的 event schema、append-only store、`BeliefUpdater` 转移规则、`for_agent/for_ranker/for_constraints` 投影语义、“只有 USER_CONSTRAINT_STATED 生成 hard constraint”“inference 不能变 filter”。
3. **Source Policy**：`SOURCE_POLICY_WEIGHTS` 三档权重（balanced 1.0/1.0、intent_focus 1.0/2.0、history_focus 2.0/1.0）、`normalize_source_policy`、`weighted_reciprocal_rank_fusion` 语义；默认 balanced 与旧 unweighted RRF 数值一致。
4. **CandidateLedger source 约束**：`CANDIDATE_PRODUCING_SOURCES` 白名单、grounding 先于记录、无 public add、UNGROUNDED 保留审计、raw score 不融合。
5. **Role Gate**：`demo/product_role.py` 的 MATCH/MISMATCH/UNKNOWN 与 `infer_target`（legacy Sports 场景）；P14 新目录需新版本，不得改旧判定。
6. **Eligibility**：三态与 precedence（violated > unresolved > verified）、不缓存、只给 membership 不给排序。
7. **默认 Serving V1 `/v1/recommend`**：`FUSION_SOURCES`、`RRF_K=60`、source head=100、request/response schema 与 `score_kind` 语义。
8. **旧 Sports item-id 空间**：`PAD=0`、`parent_asin ↔ item_id` mapping 及 checkpoint 一致性（`mappings_sha256`）。
9. **P11/P12 task + evidence 数据**：`data/p11_benchmark/tasks.jsonl`、`tasks_p12_evidence.jsonl`、`data/p12/review_keys.jsonl` 及 schema 不变，不重算不覆盖。
10. **P13/P13-R reports + negative results**：`P13_REPORT.md`、`P13R_REPORT.md` 与 `data/p13/`、`data/p13r/` 结论（P13-R = LIMITED，个性化/新 Ranker NO-GO）保留。
11. **协议版本**：`PROTOCOL_VERSION="agentrecx.eval_protocol.v1"`、`PROTOCOL_NAME="temporal_leave_two_out"`、`benchmarks/agent/FROZEN_PROTOCOL.md` 冻结集；P14-0 不新增训练、不启 GPU、不改 dependency/requirements。

## 10. P14 接入建议（compact）

| 现有模块 | 建议 | 说明 |
| --- | --- | --- |
| `recommendation/state/`（M5） | **原样复用** | 事件/投影直接给 P14 companion 用；只加 dataset→event adapter |
| `recommendation/memory/`（M9） | **原样复用（长期写入方）** | P14 用 `get_active_preferences` 读；不迁移、不改 store schema |
| `recommendation/demo/planner_contract.py` | **原样复用契约 + 新增桥接** | 契约不改；补 action→`ActionKind` executor 接入 runtime |
| `recommendation/control/{candidate_ledger,grounding,constraint_eligibility,trajectory}.py` | **原样复用** | 只允许新增字段/枚举（走冻结例外） |
| `recommendation/control/{catalog_search,product_research}.py` | **适配** | 加 namespace/固定候选集输入；核心算法不动 |
| `recommendation/preference_matching/` + `reranking/` | **适配为 Recommend 后端** | 复用三态证据与冻结排序键，不重写匹配逻辑 |
| `recommendation/tools/recommendation.py` | **保留（legacy engine 路径）** | P14 不把它改成偏好重排工具；新工具另立 |
| `recommendation/catalog/` + `contracts/namespaces.py` | **适配 + 冻结 seed** | 接 `MetadataLookup`；新目录走 namespace 注册；seed 只加目录条目不改变校验/序列化 |
| `recommendation/evaluation/` + `benchmarks/agent/` + P6/P13 experiments | **冻结（只读参考）** | 新协议/新 benchmark 另立版本，不改旧 evaluator |
| `recommendation/demo/`（agent_service、shopping_intent、product_role） | **不动** | legacy demo 运行时；P14 不在此层扩展，类目规则不上升 |

## 11. GAP 摘要（P14 实现必须补齐的接口）

| ID | GAP | 最小建议 |
| --- | --- | --- |
| G1 | Search/Recommend/ProductInfo 均无 namespace 字段 | 用 `ProductRef` 作为三者共同身份类型 |
| G2 | 无后端注册 seam（`CandidatePlane` 具名 slot + 冻结枚举） | P14 第一个新源走显式冻结例外；不做动态注册框架 |
| G3 | 无“固定候选集 + 偏好”的 Recommend 工具 | 新适配器组合 `PreferenceCandidateMatcher` + `PreferenceReranker`，输出走 ledger/eligibility |
| G4 | Memory 读面无统一 scope/provenance/conflict 契约 | 新只读投影器组合 M9 snapshot + M5 `for_*`，不合并 store |
| G5 | ProductInfo 无 namespace/目录解析；control-plane 限本 run 候选 | 新增只读 provider，复用 `ProductFact` 形状 |
| G6 | Planner ASK/RETRIEVE/RERANK/FINALIZE 未接入 HTTP runtime | §7 第 1 项桥接；不得另建第二 planner |
| G7 | 无 P14 dataset loader/task builder/evaluator 版本 | §7 第 2、7、8 项；新 schema version |
| G8 | `FullCatalogueScorer` 只支持 Sports int item_id；Fusion 无 full-catalogue score vector | 新 scorer adapter 以 namespace+opaque id 为准；Fusion 严格 full-ranking 仍记 BLOCKED |
| G9 | P13-R `catalog_meta.jsonl` 身份/生产者漂移，`experiments/p13r/retrieval.py` 把 C4 `item_id` 当 `parent_asin` | 复用前补 producer/namespace 声明；不得默认等价 Amazon ASIN |
| G10 | Role Gate 类目规则（water_filter 等）在 legacy demo 层 | P14 新目录需新版本规则，不放内核、不改旧版本 |

**给 Lead 的关键不一致（代码 vs 描述）**：① Planner 契约（ASK/RETRIEVE/RERANK/FINALIZE）只在 P3D 实验 harness + tests 中运行，未接入 `/agent/shopping` 运行时（运行时是 `ActionKind`：`recommend_from_history`/`finish`/`select_source`/`search_catalog`/…）；② P13-R 数据身份命名与代码实际使用不一致（§4.1）；③ P6 v0 `protocol.json` 的 temporal_leave_two_out 命名与 `split.py` test 模式不同（`P6_CLASSICAL_EVAL_REPORT.md` 已记录，仍未统一）；④ Fusion 的严格 full-ranking 尚无 scorer（P6 报告 BLOCKED）。
