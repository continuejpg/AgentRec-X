# P5 事实报告：个性化推荐链路与 P4D 候选反馈

## 1. 实际调用链

真实路径：

```text
ServicePlannerExecutor.execute(RETRIEVE)
  -> DemoAgentService.run_shopping_turn
     (recommendation/demo/agent_service.py:1502 self.run)
  -> DemoAgentService._execute_plan
  -> LoopController / CandidatePlane
     -> history 源: RecommendationTool.run
        -> FusionInferenceEngine.recommend
           -> sequential: SASRecInferenceEngine
           -> two_tower: TwoTowerInferenceEngine
           -> catalog_lexical: CatalogueLexicalIndex (BM25)
           -> RRF(K=60)
     -> catalog_search 源: CatalogSearchSource (BM25)
     -> CandidateLedger RRF
  -> presentable candidates -> eligibility/role -> final recommendations
```

关键源码位置：

- `recommendation/demo/agent_service.py:916`：`history_tool=self._tool`
- `recommendation/control/candidate_plane.py:301`：HISTORY 源注册
- `recommendation/control/capability.py:241-248`：读取 trusted history 并调用
  `RecommendationTool.run`
- `recommendation/tools/recommendation.py:174`：`self._engine.recommend(list(history), k=k)`
- `recommendation/inference/fusion.py:82`：
  `FUSION_SOURCES = ("sequential", "two_tower", "catalog_lexical")`
- `recommendation/inference/fusion.py:468` 起：逐源调用并执行 RRF
- `recommendation/demo/agent_service.py:3580-3648`：ledger fused candidates → final view
- `recommendation/control/candidate_ledger.py:171`：Agent 层第二层 RRF

## 2. 个性化信号结论

实际运行配置（demo-user-1，真实 history 长度 5）：

```text
engine = FusionInferenceEngine
serving_fusion_profile = serving_v1
fusion_sources = sequential, two_tower, catalog_lexical
source_score_kinds = sasrec_logit, two_tower_cosine, bm25
score_kind = rrf_rank_fusion
```

实际调用证据：

```text
FusionInferenceEngine.recommend(history=[B019YN5ZSA, B001U9YK8I, ... 5 items], k=5)
returned fused ids = [B0BX5QFWQN, B0BJH866HJ, B09LW2KHPM, B016HM7QRE, B091VYKC96]
```

Agent trajectory：

```text
recommend_from_history -> search_catalog -> finish
grounded sources = history, catalog_search
grounded candidates = 15
final recommendations = [B08QYL2SD1, B0BX5QFWQN, B001HBUW1I, B09LW2KHPM, B016HM7QRE]
history-derived IDs in final = [B0BX5QFWQN, B09LW2KHPM, B016HM7QRE]
```

结论：

- **SASRec：是。** `history` 源的实际引擎是 `FusionInferenceEngine`，其
  `sequential` 组件是 `SASRecInferenceEngine`；它不是仅被初始化，而是参与 fused
  history 结果。
- **Two-Tower：是（在 Fusion 内部）。** `FusionInferenceEngine.two_tower` 存在且
  `two_tower` 是 serving_v1 的固定 source；它不是 Agent 的独立 `two_tower` source
  （`service._two_tower_tool is None`）。
- **真实用户历史：是。** `RecommendationTool` 通过 `RecommendationContext` 接收
  trusted chronological history；实际调用收到 5 个历史 ID。
- **进入最终候选：是。** 3/5 最终推荐 ID 来自 history 源输出。
- **BM25：是。** `catalog_lexical` 在 Fusion 内；`catalog_search` 又作为 Agent 独立源。
- **RRF：是。** 两层：Fusion 内部三源 RRF；Agent CandidateLedger 对
  history/catalog_search 再做 RRF。
- **未验证：** 本次没有单独记录每个内部 source 的 raw candidate count；只能确认
  serving_v1 三源配置和实际 fused 调用。`sasrec_candidate_count`、
  `two_tower_candidate_count`、`lexical_candidate_count` 未单独暴露，不能编造数值。
- **无历史：** `AgentRecommendRequest.history` 有最少 1 项校验；空 history 请求在
  API schema 层被拒绝，不存在可观察的“无历史 fallback”路径。

## 3. P4D 候选反馈结论

### `final_candidate_count=3` 的含义

它是 `_execute_plan` 中 `build_recommendations_view` 刚生成的列表长度
（`agent_service.py:3632`），并在 `candidate_filter_diagnostics` 中记录为
`final_candidate_count`（`agent_service.py:3691` / `3716`）。

该值处于：

```text
role + eligibility 之后
但 final explicit-type promotion 之前
```

不是最终 API 展示数。

### `candidate_ids=[]` 的原因

`run_shopping_turn` 在调用 `self.run` 之后执行：

```python
promoted = promote_explicit_type_matches(
    ..., response.recommendations, records
)
...
response = response.model_copy(update={"recommendations": promoted})
```

位置：`recommendation/demo/agent_service.py:1529-1540`。

对本案例：

```text
query = "water filter replacement cartridge under $20"
pre-promotion candidates = 3
  B0BX5QFWQN = LifeStraw 完整净水器
  B09LW2KHPM = Vont 露营灯
  B0B4VB4VTD = Brita 水瓶替换滤芯（非目标净水器 cartridge）
promotion = ()
```

`shopping_intent.py:496-509` 的 accessory gate 要求 replacement 词必须由
water-filter accessory 标题证据支持；这三项都不是该目标的 replacement cartridge，
因此 promotion 为空。它**没有**把“完整净水器”错误地强制成 replacement 角色，而是
正确地排除了不匹配的候选。

### 是否存在数据丢失

- CandidateLedger：15 个 grounded candidates，未丢失；
- role/eligibility：`eligible=3, ineligible=9, unresolved=3`，未丢失；
- `build_recommendations_view`：返回 3；
- **第一个变化点**：`promote_explicit_type_matches`（`agent_service.py:1529`）
  把 3 个候选过滤为 0；这是显式类型 scope/accessory 证据过滤，不是 CandidateLedger
  或 Executor 的候选丢失。
- `candidate_ids=[]` 是 `response.recommendations` 的真实值；`final_candidate_count=3`
  是 promotion 前诊断值，二者口径不同。

### 最小修复

只修复 Planner Observation 的误导性命名，不修改推荐逻辑：

`recommendation/demo/planner_service_executor.py`：

```text
final_candidate_count
  -> post_role_eligibility_candidate_count
presented_candidate_count = len(response.recommendations)
```

修复后实测：

```json
{
  "candidate_ids": [],
  "post_role_eligibility_candidate_count": 3,
  "presented_candidate_count": 0,
  "hard_constraint_eligible_count": 3,
  "hard_constraint_ineligible_count": 9,
  "hard_constraint_unresolved_count": 3
}
```

这明确区分了“role/eligibility 后候选”与“最终展示候选”，Planner 不再把 3 误认为
最终可 FINALIZE 的候选数。

源码 diff：`P5_CANDIDATE_FEEDBACK_SOURCE_DIFF.txt`

## 4. 实际测试

```text
python -m pytest tests/test_planner_contract.py -q
11 passed
```

候选反馈修复探针实际执行通过；P4D 原始轨迹保留：
`P4D_ADAPTIVE_RETRIEVAL_TRACES.jsonl`。

## 5. 最终决策

**PARTIAL。**

理由：

- 个性化信号真实进入 Agent 最终候选：SASRec / Two-Tower 在
  `FusionInferenceEngine` 内被调用，history 真正传入，3/5 最终 ID 来自 history；
- Agent 层 RRF 和 CandidateLedger 工作；
- Planner 最终消费的是 Agent 的 `response.recommendations`；
- P4D 的候选丢失已澄清为 promotion 前后统计口径不同，不是 Ledger 数据丢失；
- 最小命名修复已做。

尚不足以判定 READY：

- 未单独量化每个 source 的候选数和贡献，无法做 SASRec/Two-Tower 归因；
- 还没有 frozen personalized benchmark，不能宣称最终质量公平可比；
- 无历史输入的 fallback 未实现/未验证（schema 直接拒绝空 history）；
- Planner 的观察现在更清楚，但真实模型是否因此改善重检索仍需下一轮验证。

## 交付

- `P5_PERSONALIZATION_AUDIT_REPORT.md`
- `P5_CANDIDATE_FEEDBACK_SOURCE_DIFF.txt`
- 关键证据脚本运行结果见本报告
