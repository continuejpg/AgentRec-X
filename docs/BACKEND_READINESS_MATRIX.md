# AgentRec-X · 后端质量准入矩阵（只读汇总，2026-10-15）

> 全部数字来自**已归档**结果（Gate A / B-1 / B-2 / B-3 及其结项文档）✓；本轮**未**加载任何模型、**未**重新推理 ✓。
> **口径纪律**：不同队列规模、不同目标定义、生成式 vs 全目录评分的数字**不得直接比较** ✗；
> 下表每一行都标注了「队列 / 目标 / 机制」三要素 ✓。

## 1. 矩阵

| 后端 | 模型与数据版本 | 队列与 Target 定义 | 质量（标注口径） | 候选合法性 | 延迟 / 资源 / 失败 | ID 命名空间 | 判定 |
|---|---|---|---|---|---|---|---|
| **SASRec** | `runs/sasrec_canonical_2026/best.pt` 121,675,470 B，sha `352bd3ae…` 与 `run.json` **一致** ✓；SASRec hidden 64 / 2 blocks / maxlen 50 / params 10,135,104 ✓；数据 Sports_and_Outdoors（412,445 用户 / 156,746 商品 ✓） | **100 例探索性**：`temporal_leave_two_out`，target = `items[-1]` ✓；另存 sealed 全队列参考（412,445 例 ✓） | 探索性 100 例：HR@5/10/20 = **0.01 / 0.01 / 0.02**、NDCG = **0.006309 / 0.006309 / 0.008756** ✓；sealed 全队列：HR@20 = **0.021489**、NDCG@20 = **0.009009** ✓（**统计相容**，非复现 ✗） | `invalid=0 / pad=0 / seen=0` ✓；masks = `excluded_seen` ✓ | 加载 1.42 s；100 例 9.40 s；RSS 峰值 **636.1 MB**；`model_failures=0` ✓ | `mappings.item2id`，1-based、PAD=0、156,746 ✓ | **准入通过（接口+行为）** ✓；质量为**探索性** ◐ |
| **Two-Tower** | `runs/twotower_public_2026_full/checkpoint.pt` 162,221,222 B，sha `0e56be01…`（前后一致 ✓）；GRU pooling / emb 128 / maxlen 50 / params 40,551,297 ✓ | **同一 100 例**（与 SASRec **逐案例同序** ✓，`same_ids_in_same_order=true` ✓） | 探索性 100 例：HR@5/10/20 = **0.02 / 0.02 / 0.03**、NDCG = **0.016309 / 0.016309 / 0.018663** ✓；**无 sealed 参考** ✗（`run.json` 无 formal metrics） | 同上 ✓（`invalid=0 / pad=0 / seen=0` ✓） | 加载 1.57 s；100 例 22.35 s；RSS 峰值 **872.4 MB**；`model_failures=0` ✓ | 同上 ✓ | **准入通过（接口+行为）** ✓；质量为**探索性** ◐ |
| **BM25 / catalog_search** | 无权重；生产实现 `recommendation/control/catalog_search.py` ✓ | 真实目录 156,746 条建索引 ✓；查询由**历史标题**派生（合法但**非协议原生** ✗） | 探索性 100 例：HR@20 = **0.02** ✓ —— **NOT_COMPARABLE 作为模型质量** ✗（机制=词法检索） | 20 例 smoke 全 `matched` ✓；`invalid=0 / pad=0` ✓ | 建索引 13.79 s；100 例 175.6 s；RSS 峰值 2,642 MB ✓ | 同上 ✓ | **准入（作为词法来源）** ✓；**不作**下一物品模型 ✗ |
| **TIGER** | `runs/tiger_public_2026/checkpoint/tiger.pt` 45,181,927 B ✓ + SID/特征齐备 ✓ | **同一 100 例** ✓ | **冻结值 HR@20 = 0/100** ✓（保留 ✓）；复测辅助观测 **1/100**（唯一命中 = case 29 合法历史重复 ✓） | SID 解析/映射 **0 失败** ✓、`invalid=0 / pad=0` ✓、`seen_items_filtered=48` ✓ | 冷启动 177.73 s（另一次调用 58.83 s ✓）；热推理 0.22–0.67 s；树 RSS 峰值 798.8 MB ✓ | 同上 ✓（SID 去重后与商品**一一对应** ✓） | **不准入** ✗（**PARTIAL 结项**：质量与跨运行一致性未建立 ✗；训练侧协议 **UNKNOWN** ✗） |
| **Similar-item（M4）** | 索引 392 MB + heads 93 MB ✓ | **未评测** ✗ | **UNKNOWN** ✗ | 未评测 ✗ | 未评测 ✗ | 同上 ✓ | **UNKNOWN** ✗ |

## 2. 接口与服务证据（代码级 ✓，非本轮实测 ✗）

| 项 | 证据 |
|---|---|
| 生产服务使用真实权重 | `recommendation/api/app.py:76`（SASRec 默认权重 ✓）、`:159`（构建顺序引擎 ✓）、`:83/:219`（Two-Tower 目录与 checkpoint ✓）✓ |
| **真实服务响应归档** | **缺失** ✗（尚无「用真实权重调用 `/v1/recommend` 并归档响应」的证据 ✗）——列为阻塞项 B2 ✓ |
| 离线接口契约 | `tests/test_api.py`、`tests/test_serving_fusion.py` 等在既有离线全量运行中通过 ✓（此前 1070 项 0 失败 ✓） |
| 统一身份契约 | 三后端共用 `mappings.item2id`（1-based / PAD=0 / 156,746 ✓）；GroundingVerifier 的 `identity_map` 协议 ✓ |

## 3. 已通过 / 未通过 / UNKNOWN

* **已通过** ✓：真实加载（SASRec、Two-Tower）✓；真实前向/生成 ✓；身份命名空间一致 ✓；候选合法性与掩码 ✓；
  同案例对齐 ✓；资源记录 ✓；TIGER 生成链路与 SID 映射 ✓。
* **未通过/不准入** ✗：TIGER 质量与跨运行一致性 ✗（结项 PARTIAL ✓）。
* **UNKNOWN** ✗：训练历史窗口/目标偏移/截断方向（TIGER）✗；Two-Tower 的 sealed 对照 ✗；Similar-item 评测 ✗；
  **质量证据规模**（三模型均仅 100 例探索性，无置信区间）✗。

## 4. 两项既有失败（持续单独披露 ✓，未修复、未放宽）

① `tests/test_backend_boundaries.py::test_t7a_adapter_never_names_canonical_identity` ✗
（`recommendation/backends/tiger_engine.py` 4 个身份标识符违反 AGENTS.md §19.2 字面契约 = `known_architecture_contract_failure` ✓）；
② `tests/test_tiger_engine.py::test_result_satisfies_tiger_source_tool` ✗（**A 类**：FakeTransport 替身/预期漂移 ✓，既有 ✓）。
**二者均不阻塞 SASRec / Two-Tower 链路** ✓，但**必须随准入记录保留** ✓。
