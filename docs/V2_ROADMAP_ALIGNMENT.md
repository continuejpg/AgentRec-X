# V2 路线映射与最小推荐链路确认（任务 04）

## 1. V2 闭环必需项 ↔ 当前服务能力
| V2 闭环必需项 | 当前能力 | 证据 | 状态 |
|---|---|---|---|
| 用户行为 → 推荐响应（单一入口） | `POST /v1/recommend`（`RecommendRequest{history,k}` → `RecommendResponse{recommendations[{rank,item_id,parent_asin,score}],requested_k,returned_k,history_length}` ✓） | `recommendation/api/app.py:643` ✓、Schema `api/schemas.py:22/43/87` ✓ | **具备** ✓ |
| 真实推荐模型参与 | SASRec（`api/app.py:76/159` ✓）、Two-Tower（`:83/:219` ✓）真实权重加载 ✓ | Gate C-1/C-2 两次运行 R200 ✓ | **具备** ✓ |
| 两条独立后端链路 | `serving_v1_sequential_only` / `serving_v1_two_tower_only`（单 source ✓，无融合/补位/降级 ✓） | `fusion.py:FUSION_PROFILES` ✓、`tests/test_v2_gate_c12_profiles.py` 4 passed ✓ | **具备** ✓ |
| 缺省融合链路不回归 | `serving_v1` = 三源 RRF（缺省未变 ✓） | 36 passed ✓ | **具备** ✓ |
| 商品身份/排名/去重/掩码契约 | item_id ↔ parent_asin 同命名空间 ✓、rank 1-based 连续 ✓、响应内无重复 ✓、历史不回声 ✓ | Gate C-1/C-2  legality 全 0 违规 ✓ | **具备** ✓ |
| 非法 Profile 被拒 | `AGENTRECX_FUSION_PROFILE` 未知名 ⇒ 启动即 `RuntimeError` ✓；`for_profile` 抛 `FusionError` ✓ | `tests/test_v2_gate_c12_profiles.py::test_unknown_profile_is_refused` ✓ | **具备** ✓ |
| 响应可追溯 | 逐例 JSONL（两轮）+ 离线参考 + parity/repeatability/metrics/resources ✓ | `runs/gate_c12_service_validation_r2/**` ✓ | **具备** ✓ |
| 错误处理明确 | 空/短/超长历史、非法类型、非法 K → **4xx**（422）✓；**未知身份 → 500** ✗（**已知缺陷**，见任务 05） | `tests/test_v2_task05_service_boundaries.py` 17 passed ✓ | **部分** ◐ |
| Agent/工具编排环节（V2 范围） | 已实现且**有独立离线验收**：E-01A/B、E-02A/B/C、E-03A/B/C（1070 项离线回归 0 失败 ✓）；真实 LLM 决策的在线额度 **4/4 已用尽** ✗ | 各阶段报告 ✓ | **存在但未纳入本轮服务闭环** ◐（**不排除出 V2 定义** ✓，其质量证据等级见报告 ✓） |

## 2. 最小链路（确认，**不新建平行 API** ✓）
```text
调用方 → POST /v1/recommend（唯一入口 ✓）
        → FusionInferenceEngine（profile 决定参与 source 集合 ✓）
        → [serving_v1] sequential + two_tower + catalog_lexical（RRF ✓，缺省 ✓）
        → [serving_v1_sequential_only] 仅 sequential ✓
        → [serving_v1_two_tower_only] 仅 two_tower ✓
        → 身份：item2id（1-based / PAD=0 ✓）→ parent_asin ✓
        → 响应：rank 1-based、无重复、无 PAD、无越界 ✓
```
**结论**：既有服务**已满足**最小链路定义 ✓（无需重构 ✓）；本轮**未**新增任何 API ✓。

## 3. 未纳入本轮的范围（如实 ✓）
* 质量侧：更大固定队列的统一质量评测（P3 ✗）、融合对照（P4 ✗，需单独批准 ✗）。
* Agent 侧：其离线验收已存在 ✓，但**真实 LLM** 部分受额度限制 ✗，故不纳入本轮服务闭环结论 ✓。
