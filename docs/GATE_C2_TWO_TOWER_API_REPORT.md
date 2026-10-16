# Gate C-2 · Two-Tower 真实 API 端到端验收报告
> 路由：`POST /v1/recommend`（官方 Schema；服务的是 `serving_v1` 融合 profile）。

> 结论：**PARTIAL** ✗（与 C-1 同因 ✗）。`/v1/recommend` 的真实请求对 Two-Tower 同样**共享同一路由** ✓，
> 两轮 100/100 合法且确定 ✓；但**单后端** API 证据与 **API↔离线一致性** 均 **UNAVAILABLE** ✗。
> **未**加载 TIGER ✓、**未**融合 ✓、**未**改生产代码/权重/映射/Schema ✓、**未**使用 SASRec 评分代码冒充 Two-Tower 离线基线 ✗
> （离线参考用 `TwoTowerInferenceEngine.score_catalog` + `encode_inference_history`（GRU 历史编码路径 ✓）✓）。

## 归档
```text
runs/gate_c12_service_validation_r3/two_tower/{engine_top20_run1.jsonl, engine_top20_run2.jsonl, offline_reference.jsonl,
             parity.json, repeatability.json, metrics.json}   ✓（独立两轮，未复制 ✗）
Two-Tower : runs/twotower_public_2026_full/checkpoint.pt（sha 0e56be01… ✓，前后一致 ✓）
            GRU / emb 128 / maxlen 50 / params 40.55M ✓；离线 is_ready ✓
```

## 结果
| 项 | 值 |
|---|---|
| 路由请求/成功/Schema | 100/100 ✓ / 100/100 ✓ / 100/100 ✓（与 C-1 同一路由 ✓） |
| 非法 / PAD / 重复 / Seen 违规 | **0 / 0 / 0 / 0** ✓ |
| 路由两轮一致 | **100/100** ✓ |
| 单源 accessor（`_two_tower_ranking`）可用性 | **false** ✗ ⇒ 单源 Top-20 与 API parity **UNAVAILABLE** ✗（空比较 ✗，非不一致证据 ✗） |
| 引擎重复调用一致（若能取到句柄时） | **100/100** ✓（`repeatability.json`；⚠️ 同样受句柄缺失影响，为空比较 ✗ —— **不作结论** ✗） |
| 离线参考（冻结协议 ✓） | hits@5/10/20 = **1 / 1 / 2**，HR@5/10/20 = **0.01 / 0.01 / 0.02** ✓、HR@20 = 0.02 ✓ |
| 与 Gate B-2 归档对照 | B-2：HR@5/10/20 = 0.02/0.02/0.03 ✗（**不完全相同** ✗；差异原因同 C-1 §3 的参考实现差异 ✗，非模型变化 ✓；历史归档未改 ✓） |

## 工程判定
**PARTIAL** ✗：无非法/PAD/重复/Seen ✓、Schema 合法 ✓、两轮确定 ✓，但**单后端 API 证据缺失** ✗ 与**一致性不可比** ✗。
**注意**：本轮任何数字**都不构成** Two-Tower 与 SASRec 的优劣结论 ✗，也**不**触发融合 ✗。
