# Learned Ranker 独立评估 —— 最小可行方案（Phase 2B.6 交付）

> 状态：**方案，未执行**。本文件不运行任何实验、不修改模型/评测器/预注册。
> 目的：在 Phase 2B.5/2B.6 结束 catalog 查询策略主线后，给出进入 Learned Ranker 评估的
> **最小、可预注册、单变量**方案。

## 1. 为什么需要一次独立评估

现有 LR V1 证据互相矛盾，且基线有已知缺陷：

| 产物 | 口径 | RRF 基线 | LightGBM | 结论 |
|---|---|---|---|---|
| `runs/lr_v1_formal/metrics.json` | 训练管线的 RRF（**受限 universe**） | HR@5 0；HR@20 3/766 | HR@5 0.0104；HR@20 0.0274 | 看似 LR 大幅领先 |
| `corrected_test_comparison.json` | 全精度 RRF（post-hoc 修正） | HR@20 30/766；NDCG@20 0.013636 | HR@20 21/766；NDCG@20 0.01239 | LR 反而略低，配对 Δ NDCG@20 = −0.001246，95% CI [−0.006189, 0.00407] |
| `A6_authoritative.json` | 另一版 post-hoc oracle | HR@5 6/766；NDCG@20 0.013627 | HR@5 9/766；NDCG@20 0.01204 | Δ NDCG@20 = −0.001587，CI [−0.006543, 0.003704] |

`docs/LEARNED_RANKER_DEPLOYMENT.md` 已把质量标为 **NOT ESTABLISHED**。两个问题必须在新评估里修掉：

1. **基线 bug**：训练管线把 RRF 限制在候选池子集上算，把基线 HR@20 压到 3/766；全精度 RRF
   在同一批 766 组上是 30/766。新评估必须用**全精度 RRF over 完整 fused universe**。
2. **数据复用**：766 个测试组来自 5,000 组训练 cohort（#1001–6000），且模型选择、A6 修正都在
   同一批数据上看过。新评估必须使用**从未参与训练/选择的 cohort**。

## 2. 目标与假设

- **H1（主）**：在完全相同的候选池、深度、目录和检查点下，冻结的 LR V1 重排相对全精度 RRF
  提升 **NDCG@20**。
- **H0**：无差异。
- 次要：HR@5、NDCG@5（产品 k=5）；HR@10/20、NDCG@10/20 全部报告。

选择 NDCG@20 作为主终点，因为它与模型训练目标（`ndcg_eval_at=[20]`）和既有 A6 口径一致；
不在看到结果后改主终点。

## 3. 固定因素（单变量：only the ranker changes）

| 因素 | 固定值 |
|---|---|
| 任务 / 切分 | `agentrecx.eval_protocol.v1` / `temporal_leave_two_out`；`history = case.test_history = items[:-1]`，`target = items[-1]` |
| 候选池 | 三源（SASRec / Two-Tower / catalogue lexical）各自 **top-100** 的并集，去掉历史商品、去重、只保留目录内身份——与 `experiments/r_lr01_materialize.py` 完全一致 |
| 深度 | `head_depth = 100`（不加深、不降低） |
| RRF | `reciprocal_rank_fusion`，`sum 1/(60 + source_rank)`，全精度；顺序 `(-rrf, item_id)` |
| LR | 冻结 `runs/lr_v1_formal/model.txt`（22 棵树）+ `feature_schema.json`（15 维），`build_feature_rows`（生产同一实现），排序 `ranker_score DESC → rrf_score(6dp) DESC → item_id ASC` |
| 目录 / 映射 / 检查点 | 与训练时相同的 `data/processed/*`、SASRec、Two-Tower；不启用 TIGER |
| k | 5 / 10 / 20（全部报告） |

**不做的**：不重训、不调参、不改特征、不改 RRF、不改深度、不换 cohort 选择规则。

## 4. 评估 cohort（新，未被使用）

| 项 | 值 |
|---|---|
| 起点 | artifact index **9000**（`valid #9000+`） |
| 规模 | **2,000** 个有效案例（若不足，报告实际可用数，不降标准） |
| 与 LR 训练重叠 | 0（训练用 valid #1001–6000） |
| 与 P3 / Gate 重叠 | 0（P3 前 1,000） |
| 与 2B.5 / 2B.6 重叠 | 0（2B.5 #6500–6709，2B.6 #6710–~7760） |
| 冻结 20k cohort | 按 `user_int_id` 排除（精确） |
| R-LR 划分 | 不适用（该 cohort 从未参与 LR 训练/选择） |

预期池内正例率约 **14.5%**（A6 的 111/766），2,000 例约 **290** 个正例；约为既有 766/111 的
2.6 倍样本、2.6 倍正例。

## 5. 评估流程（下一阶段实现，本轮不执行）

`experiments/r_lr05_independent_eval.py`（新文件，独立目录 `runs/lr_v2_independent_eval/`）：

1. 逐案例跑三源 recall（top-100），**只跑一次**，缓存 per-source `(rank, score)`；
2. 构建候选池：三源并集、去历史、去重、只保留已知身份；
3. 计算全精度 RRF（完整 fused universe），得到 **baseline order**；
4. 用 `build_feature_rows(item_ids, pool, ranks, scores, rrf_by_asin, rrf_universe=fused_order)`
   构造 15 维特征（生产同一实现）；
5. 冻结 booster 预测，`order_by_ranker` 得到 **treatment order**；
6. 逐案例记录：目标在 baseline / treatment 中的池内 rank；HR@5/10/20、NDCG@5/10/20；
7. 配对 bootstrap（10,000 次，按 case 重采样，seed 预注册）报告 Δ 与 95% CI；
8. 写 `config/cases/samples/summary/SHA256SUMS`，拒绝覆盖已存在目录。

**指标实现**：池内 rank 用冻结评测器的 `hit_at_k` / `ndcg_at_k`（`recommendation/evaluation/metrics.py`），
不另写公式；池外目标记为 miss（与部署行为一致：引擎只返回池内排序）。
**断言**：baseline 与 treatment 的候选池**逐案例完全相同**；目标从不被插入候选池；特征行数
= 池大小；模型/schema SHA 与部署文档一致；`fallback_count = 0`；HR == Recall（单正例）。

## 6. 预注册的验收门槛（先写死）

| 编号 | 条件 | 不满足时的动作 |
|---|---|---|
| **G1（主）** | 配对 ΔNDCG@20 的 95% CI **下界 > 0** | LR 质量保持 NOT ESTABLISHED；不切默认；归档 |
| **G2（回归）** | ΔHR@5 ≥ 0（或 CI 下界 > −0.002） | 即使 G1 成立也不切换：先查 Top-5 伤害 |
| **G3（完整性）** | 池身份 100%、特征 15 维、模型/schema 哈希一致、fallback=0 | 结果作废重跑 |
| **G4（独立性）** | cohort 与训练/P3/2B.5/2B.6 重叠 = 0 | 结果作废重跑 |

**判定**：

- G1 不成立 → 结束 LR 主线或转为“记录负结果”，不进入生产验证。
- G1 成立但 G2 不成立 → 只说明 NDCG@20 有提升，产品 k=5 有回归风险；仍不切换。
- G1+G2 都成立 → **只代表值得进入独立生产验证**（灰度和回滚方案另立），不等于切换默认值。

## 7. 前置条件（环境，非代码）

现有 `.venv` **没有** LightGBM，`.venv-ranker` 不存在。按部署文档建立独立环境：

```bash
python3.10 -m venv .venv-ranker
.venv-ranker/bin/pip install -r requirements-serving-ranker.txt   # lightgbm==4.7.0, numpy==1.26.4
.venv-ranker/bin/python -c "import lightgbm; print(lightgbm.__version__)"
```

默认 `serving_v1` 不导入 LightGBM；该环境只用于本评估。

## 8. 运行命令（下一阶段）

```bash
# 1) 先校验冻结模型与 schema
sha256sum runs/lr_v1_formal/model.txt runs/lr_v1_formal/feature_schema.json
#    model  c1b37d4a2afe9187000c73bc7cb7bb22cc811dda3a2fe9bfe2118fa513db6eb9
#    schema ad1d0be168dc9f183c45683c26e24a0086e882632570e2feffa8f9971598a422

# 2) 独立评估（2,000 例；预计分钟级，三源 recall 一次/例）
.venv-ranker/bin/python -m experiments.r_lr05_independent_eval \
    --cases 2000 --skip 9000 --out runs/lr_v2_independent_eval

# 3) 只跑相关测试（不跑全量）
.venv/bin/python -m pytest -q tests/test_v2_learned_ranker.py
```

## 9. 已知限制 / 风险

1. **统计功效**：2,000 例约 290 个池内正例；按 1/√n 估算，ΔNDCG@20 的 CI 半宽从 766 例的
   ~0.005 收窄到 ~0.003。小于 ~0.003 的真实效应仍可能判 NOT ESTABLISHED，这是可接受的。
2. **模型未重训**：本方案只回答“冻结 LR V1 是否优于 RRF”，不回答“LR 能否做得更好”。
3. **候选池口径**：池是深度 100 的三源并集；LR 只在池内重排，不扩召回。任何“召回提升”都不属于
   本评估的结论。
4. **与 Agent 动态平面的边界**：`DemoAgentService._execute_plan` 明确拒绝
   `LearnedRankerEngine`（动态多源融合会覆盖 LR 排序）。本评估对应固定三源 `/v1/recommend`
   路径，**不是** Agent 多源平面的评估。
5. **A6 是 post-hoc**：本方案是第一次在独立 cohort 上、用正确基线、单变量、预注册门槛的评估；
   不得用本方案的结果反过来重算 A6 的数字。
6. **不修改冻结资产**：评测器、预注册、历史报告、检查点、RRF、深度、来源计划均不动。
