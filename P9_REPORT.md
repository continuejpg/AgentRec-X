# P9：真实个性化推荐候选质量提升报告

## 0. 结论

**扩大 History 候选预算能提升候选可达性，但没有提升最终 Top-10 排序质量。**

| 项 | 判断 |
|---|---|
| 候选可达性提升 | **LIMITED：History Top-50 → Top-100 从 14/300 到 21/300** |
| 最终 Top-10 排序 | **NO-GO：没有改善，balanced 甚至下降** |
| 成本 | 可接受，但仍受 CPU 模型推理限制 |
| 是否需要继续调 Top-K | **否；不应继续以加预算为主** |
| 下一步 | **改进模型候选生成/排序能力，而不是回到 Agent 策略学习** |

## 1. 实验基线

- P8-D 固定 300 个 Dev case：
  - 157 canonical Dev + 143 extended Dev；
  - validation target（`items[-2]`）；frozen Holdout 未使用；
  - target-in-history = 0；
- 当前 Serving V1：
  - sources = `sequential`（SASRec）、`two_tower`、`catalog_lexical`；
  - source head depth = 100；
  - History Tool / Tool `MAX_K` = 100；
- Catalog 使用 P8-D 缓存的 Hybrid Top-50，不重新调用 BGE；
- Source Policy / M5 / Role Gate / Eligibility / Gold / Evaluator 未修改；
- 未训练模型，未使用新 Embedding/Reranker。

## 2. 当前候选可达率

基线（History Top-50 + Catalog Top-50）：

| 来源 | target 可达数 / 300 |
|---|---:|
| History Top-50（实际 Serving V1） | 14 |
| Catalog Hybrid Top-50 | 1 |
| Union | **15** |

**主瓶颈不是 catalog，而是 History source 的 raw candidate scoring：285/300 的 target 不在两路 Top-50。**

## 3. SASRec 与 Two-Tower 的独立贡献

使用真实 checkpoint 的 full-catalog score，按与 Serving 相同的 deterministic top-k 规则做离线 probe：

| Source | Top-50 | Top-100 | Top-200 probe |
|---|---:|---:|---:|
| History source（实际 fusion） | 14 | 21 | — |
| SASRec | 10 | 14 | 24 |
| Two-Tower | 14 | 17 | 25 |
| Catalog Hybrid | 1 | — | — |

Top-100 的 unique 贡献：

| Source | 可达 | unique-only |
|---|---:|---:|
| SASRec | 14 | 5 |
| Two-Tower | 17 | 8 |
| SASRec ∪ Two-Tower | 22 | — |
| 两者 overlap | 9 | — |

Top-50：

- SASRec 10，Two-Tower 14；
- union 18，overlap 6；
- SASRec-only 4，Two-Tower-only 8。

Top-200 offline probe：

- SASRec 24，Two-Tower 25，union 36，overlap 13；
- 但 200 不是 Serving 合法预算，只是 score-matrix headroom probe。

## 4. 扩大候选预算的效果

实际 History source 从 Top-50 扩到合法上限 Top-100：

| 集合 | target 可达数 / 300 |
|---|---:|
| History Top-50 + Catalog Top-50 | **15** |
| History Top-100 + Catalog Top-50 | **21** |
| SASRec ∪ Two-Tower Top-50 | 18 |
| SASRec ∪ Two-Tower Top-100 | 22 |
| SASRec ∪ Two-Tower Top-200（offline） | 36 |
| History100 + Catalog + component100（offline union） | 28 |

History 50→100 新增可达 **7 个** target，例如：

```text
classical-dev-00057 -> B004J4B2IW
classical-dev-00069 -> B01KZSCVQW
classical-dev-00092 -> B01N967JUG
classical-dev-00125 -> B07392XS6V
p8d-dev-ext-00198 -> B000FICCQC
p8d-dev-ext-00228 -> B01FV03URI
p8d-dev-ext-00265 -> B08Y656WJ9
```

SASRec 50→100 新增 4 个，Two-Tower 50→100 新增 3 个；200-deep probe 分别再新增 10/8 个（offline）。

**可达率确实上升，尤其是组件 200 headroom；但这还不等于最终排序提升。**

## 5. 候选融合/过滤损失

有少量 target 在组件 source 中可达，但没有进入实际 History source：

| 情况 | 数量 |
|---|---:|
| SASRec Top-100 可达但 History source Top-100 不可达 | 4 |
| Two-Tower Top-100 可达但 History source Top-100 不可达 | 3 |
| SASRec Top-50 可达但 History source Top-50 不可达 | 3 |
| Two-Tower Top-50 可达但 History source Top-50 不可达 | 4 |

说明 fusion/dedup/source head 有少量损失，但不是主要瓶颈；主体仍是 raw model score 没有把 target 排进 head。

## 6. 最终 Top-10 质量

用同一份 P8-D Catalog Top-50，把 History source 换成 Top-50/Top-100，并使用同一条 weighted RRF：

| 配置 | Recall@10 | NDCG@10 | MRR@10 | 有 target 在 Top-10 |
|---|---:|---:|---:|---:|
| History50 + Catalog50, balanced | **0.0200** | **0.01105** | **0.00831** | 6 |
| History50 + Catalog50, history_focus | **0.0200** | **0.01497** | **0.01333** | 6 |
| History100 + Catalog50, balanced | 0.01667 | 0.00836 | 0.00569 | 5 |
| History100 + Catalog50, history_focus | 0.0200 | 0.01497 | 0.01333 | 6 |

结论：

- History Top-50→100 把 union 可达从 15 提到 21；
- **但 Top-10 Recall 没有提高**；
- balanced 的 NDCG/MRR 反而下降；
- history_focus 与 baseline 完全持平；
- 新增的 6 个可达 target 没有转化成 Top-10 命中。

**候选可达性提升 ≠ 最终排序质量提升。**

## 7. 成本

本次 P9 评测：

- 模型加载（两次断点续跑的总和）：173.24s；
- 300 case 的 per-case measured source/rank time 合计：569.72s；
- 平均 History source：1.52s/case；
- 平均 SASRec full-score + 200 rank：0.082s/case；
- 平均 Two-Tower full-score + 200 rank：0.297s/case；
- Top-50→Top-100 不增加模型 scoring 成本：Serving source head 本来就是 100，只是多返回 50 个已计算候选；
- 候选列表 +50 IDs/case 的内存开销可忽略；
- 没有加载 BGE，也没有新建索引；
- 未测量精确进程 RSS，但本轮没有新增模型/索引，主要成本仍是原有 CPU 模型推理。

因此“更大候选池”的计算成本本身不高；真正的限制是收益没有传递到 Top-10。

## 8. 主要瓶颈判断

1. **SASRec / Two-Tower 的原始候选排序是主要瓶颈**：target 通常不在合法 Top-50/100。
2. **Top-K 截断确实有贡献，但不是唯一原因**：
   - History Top-50→100 只新增 7/300 可达；
   - 200-deep offline probe 才把 union 提到 36，且不是合法 serving 预算。
3. **Catalog 不是本轮个性化 next-item 的主要来源**：仅 1/300 独立可达。
4. **fusion/dedup 损失是次要的**：最多 4 个 component-reachable targets 未进入 History source。
5. **最终融合排序没有改善**：达到 union 的目标没有进入 Top-10。

## 9. 下一步建议

**不建议继续扩大 Top-K 作为主要优化方向。**

- 合法 Top-100 仍只有 21/300 可达；Top-200 offline 也只有 36/300；
- 可达性增加没有带来 Top-10 Recall/NDCG 提升；
- 因此应先改进现有 SASRec / Two-Tower 的候选评分/表示/训练目标，再考虑更大 head 或新模型；
- 本轮不重训、不换架构；下一步应做有明确目标的模型优化实验，而不是继续调预算。

**不建议此时回到 Planner SFT-v1 或 Source Policy 学习**：当前候选能力不足以稳定支撑策略选择标签。

## 10. 交付物

- `P9_REPORT.md`
- `P9_SOURCE_DIFF.txt`
- `experiments/p9_candidate_depth_eval.py`
- `experiments/p9_combine.py`
- `runs/p9/candidate_depth/summary.json`
- `runs/p9/candidate_depth/per_case.jsonl`
- `runs/p9/candidate_depth/per_case.first210.jsonl`
- `runs/p9/candidate_depth/per_case.last80.jsonl`

配置：

- P8-D 300 Dev cases，validation target；
- History source budget 50/100；Catalog 50；
- component probe limit 200（offline label）；
- Serving source head 100、MAX_K=100 未改动；
- BGE / Holdout / Gold / Evaluator / M5 / Role Gate / Eligibility 均未使用或修改。

未训练 SFT-v1，未运行 Holdout，未 commit/push/deploy；GPU 实例未关闭。
