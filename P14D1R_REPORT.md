# P14-D1-R — Real History Metadata Recovery & Retrieval Diagnosis

> 范围：P14-D1-R。针对 P14-D1 的真实 MemRerank Electronics Dev 任务，补齐历史商品 Metadata、诊断 BM25/BGE 候选覆盖、重新做真实历史 vs Query Only vs Shuffled History 消融。
> 不训练/微调，不改 V0/Gate/Planner/Memory/模拟标签，不运行公开 Test 或项目 Reserved Test，不重建全目录索引，不 commit。
> **Dev 仍是 P13-R 已使用过的 exposed split；所有结论仍为 provisional，未做人工核验。**

## 0. 结论

- 历史 metadata：唯一 ASIN 覆盖 **126 → 5,576（38.0%）**；有历史 metadata 的任务 **50 → 135**；history items 覆盖 **126 → 5,783**；弱历史属性代理可用任务 **35 → 126**。
- 自然检索：original BM25 Top-20 目标可达 **10/135**，Top-100 **20/135**；rewritten-query BM25 将可达提升到 **Top-20 18/135、Top-100 30/135**。
- **BGE 现有 index 与 P13-R Electronics catalogue 不对齐**（156,746 个 identity，与 P13-R item ID 交集为 0；样本标题为 Books），本轮不重建全目录 BGE，故 BGE 对照不可用，已如实报告。
- 历史消融（同一 Qwen3-Reranker-4B、同一 original query、同一候选池）：
  * natural：Query Only 6/135，History 7/135，Shuffled History 8/135；
  * target-included 1-in-5：Query Only **108/134**，Recovered History 100/134，Shuffled **102/134**；
  * rewritten-retrieval 池（target reach 18/135）：Query Only **9/135**，History 8/135，Shuffled 9/135。
- **真实历史在 metadata 补齐后仍然没有超过 Query Only 或 shuffled 负对照；shuffled 甚至经常略高。** 这不是“模型看不到历史”造成的假象；更可能是该公开数据的历史—目标关系太弱、历史 text 仍只有 38% 覆盖，以及目标自然不可达。

**最终：`HISTORY_ABLATION = FAIL`（无增益），`HISTORY_METADATA_RECOVERY = LIMITED`，`RETRIEVAL_DIAGNOSIS = LIMITED`，`V0_REAL_HISTORY_COMPATIBILITY = LIMITED`，`P14-D1-R = LIMITED`。**

## 1. 历史 Metadata 恢复

`data/p14d1r/`、`history_metadata_manifest.json`：

| 项 | 恢复前 | 恢复后 |
| --- | ---: | ---: |
| 历史条目总数 | 14,945 | 14,945 |
| 唯一 history ASIN | 14,655 | 14,655 |
| 唯一 ASIN 有 metadata | 126 (0.86%) | **5,576 (38.0%)** |
| history items 有 metadata | 126 | **5,783** |
| 至少一条历史 metadata 的任务 | 50 | **135 / 135** |
| 弱属性代理可行任务 | 35 | **126** |

来源：

- 既有 Amazon-C4 sampled metadata cache：**803** 个唯一 history ASIN；
- **官方 Amazon Reviews 2023** `raw/meta_categories/meta_<Category>.jsonl` 按 history ASIN 集合流式过滤：**5,155** 个唯一 ASIN；
  - Grocery_and_Gourmet_Food: 1,493
  - Health_and_Household: 923
  - Pet_Supplies: 469
  - Beauty_and_Personal_Care: 659
  - Electronics: 1,611
- 资源策略：只 stream 这 5 个类别，**网络上限 15GB / 时间上限 1200s**；没有把全部 33 类下载到磁盘，不重建全目录；
- 每条恢复行保存 item_id、title/text、metadata_category、source；未知属性保留 UNKNOWN；target 从未作为查询输入或 metadata 请求；
- 仍有 62% 的 history ASIN 未恢复（Home_and_Kitchen、Clothing 等大类别未下载）；这是后续最大数据缺口。

## 2. 真实历史构造

- 只使用 target 前、严格时间截断的真实 history；target 自身事件删除；无 ≥target 事件；
- history summary 使用固定最近 12 条，只包含类别、恢复标题/ID、verified 状态；
- 明确区分：
  * 真实交互事实（timestamp / verified）；
  * 商品 metadata（title/text/category）；
  * 模型未生成的弱兴趣代理（observed positive attributes only，无负偏好、无 target 属性）；
- 没有把历史商品直接当作用户明确喜欢的商品；没有伪造负偏好。
- Shuffled History：从其他用户取 history，按长度与类别分布匹配，并移除 recipient target；作为合法负对照。

## 3. 召回诊断（无 target 注入）

`runs/p14d1r/retrieval_diagnosis.json`：

| BM25 query | Target in Top-20 | Target in Top-100 |
| --- | ---: | ---: |
| Original Amazon-C4 query | **10/135** | **20/135** |
| Released rewritten query | **18/135** | **30/135** |
| Original + rewritten | 10/135 | 22/135 |
| History-expanded (recovered titles) | 8/135 | 17/135 |

- **rewritten query 是唯一显著改善自然 target 覆盖的检索变体**（Top-20 10→18）；但 rewritten 会丢失 brand/spec/个人细节，只作为检索诊断，未与 original 主结果混合；
- history-expanded BM25 反而更差，说明 recovered history 不能直接当 query expansion；
- **BGE**：`runs/p8_bge/bge_base_dense.npz` 存在，但 item_count=156,746、样本标题为 Books，且与 P13-R item_id **交集 0**；使用它需要重建当前 catalogue 全量 BGE index，F 明令禁止，因此 BGE Top-20/100 未运行，不能报告 BGE 数字。
- 在 target-included Protocol B 中，target 本身可能完全不在 BM25 Top-100；134 个 target5 任务中 **114 个 target 不在 BM25 Top-100**，distractors 的 BM25 rank 在 1–100；因此 target-included 只是条件重排诊断，不是自然检索能力。

## 4. 历史消融（同一模型/Query/候选池）

### 4.1 Natural BM25 Top-20（n=135，target 可达 10）

| 方法 | Hit@1 | Hit@5 | MRR@10 | 条件 n=10 Hit@1 |
| --- | ---: | ---: | ---: | ---: |
| Query Only | 6/135 | 10/135 | 0.055 | 6/10 |
| Real History (旧 metadata) | 8/135 | 9/135 | 0.063 | 8/10 |
| **Real History (恢复 metadata)** | **7/135** | 9/135 | 0.060 | **7/10** |
| Shuffled History | **8/135** | 9/135 | 0.064 | **8/10** |

Natural 条件集只有 10 条，不能做稳定结论；可见 **真实历史没有超过 shuffled，且恢复 metadata 后反而从 8 降到 7**。

### 4.2 Target-included 1-in-5（n=134）

| 方法 | Hit@1 | MRR@10 |
| --- | ---: | ---: |
| Query Only | **108/134** | **0.891** |
| Real History (旧) | 101/134 | 0.861 |
| Real History (恢复 metadata) | 100/134 | 0.861 |
| Shuffled History | 102/134 | 0.869 |
| V0 weak proxy | 18/134 | 0.366 |

- 真实历史相对 Query Only：improved 6 / worsened 13；相对 shuffled：improved 5 / worsened 9；
- 恢复 metadata 没有修复真实历史增益；shuffled 略高于真实历史；
- 按 history metadata 覆盖分组：1–2 条 16 vs QueryOnly17；3–7 条 25 vs28；8+ 条 59 vs63，均无分组正增益。

### 4.3 Rewritten-query retrieval pool（target reach 18/135）

为排除“target 完全不可达”，单独用 rewritten BM25 构造固定候选池（target 仍不注入），Qwen 仍接收 original query：

| 方法 | Hit@1 | Hit@5 | MRR@10 | 条件 n=18 Hit@1 |
| --- | ---: | ---: | ---: | ---: |
| Query Only | **9/135** | 15/135 | 0.086 | 9/18 |
| Real History (recovered) | 8/135 | 15/135 | 0.082 | 8/18 |
| Shuffled History | **9/135** | 16/135 | 0.085 | 9/18 |

即使 target 可达从 10 提升到 18，真实历史仍然没有增益；shuffled 与 Query Only 持平。

## 5. V0 与真实历史

- V0 proxy 使用恢复后的历史商品 observable attributes（positive only，无 negatives、无 target）；
- target-included：V0 proxy 从恢复前的 7/134 提升到 **18/134**，说明 metadata 让 weak profile 更可用；
- 但仍远低于 Query Only 108/134，natural proxy 1/135；**V0 不能从真实历史合法地得到显式正负偏好**；
- `V0_REAL_HISTORY_COMPATIBILITY = LIMITED`，不能把 proxy 当作 V0 的真实历史能力。

## 6. 成本

| 项 | 成本 |
| --- | --- |
| 历史 metadata 恢复 | 5 个类别 streamed；网络 cap 15GB；不落盘全量；官方 5,155 + C4 803 unique ASIN |
| Qwen history natural | 5,400 pairs；load 21.03 s；350.6 s；peak 6,386 MiB |
| Qwen history target5 | 1,340 pairs；load 17.6 s；87.96 s；peak 4,894.8 MiB |
| Qwen improved retrieval | 8,100 pairs；load 17.92 s；444.25 s；peak 5,836.5 MiB |
| 训练/微调 | 无 |

## 7. 回答核心问题

**排除“模型看不到历史商品”和“目标根本检索不到”两个明显干扰后，真实历史能否改善排序？**

- metadata 覆盖率从 0.86% unique ASIN / 50 tasks 提升到 **38.0% / 135 tasks**；
- 自然 target 覆盖通过 rewritten retrieval 从 10/135 提升到 18/135；
- **在相同模型、query、候选池下，真实历史仍不优于 Query Only，也不优于 shuffled 负对照；target-included 甚至从 108 降到 100。**
- 因此：**当前数据下没有可测量的真实历史增益**；但仍有 62% history ASIN 未恢复、自然 target 覆盖仍只有 13.3%–22.2%，**不能据此断言真实历史永远无价值**。这是 **LIMITED / data-limited negative result**。

## 8. 最终判定

| 判定 | 结果 | 说明 |
| --- | --- | --- |
| `HISTORY_METADATA_RECOVERY` | **LIMITED** | 目标五类完整恢复 5,155；总体唯一 ASIN 覆盖 38.0%；Home/Clothing 等大类别未恢复 |
| `RETRIEVAL_DIAGNOSIS` | **LIMITED** | rewritten BM25 Top-20 10→18；BGE index 不对齐（0 交集）且禁止重建，BGE 不可用 |
| `HISTORY_ABLATION` | **FAIL** | 恢复历史后仍 7/135 natural、100/134 target5；不优于 Query Only，也不优于 shuffled |
| `V0_REAL_HISTORY_COMPATIBILITY` | **LIMITED** | 弱代理 target5 18 但远低于 Qwen；无显式偏好来源 |
| **P14-D1-R** | **LIMITED** | 已排除历史文本缺失与目标注入两类干扰到当前可行上限，仍未发现历史增益；数据缺口与公开 Dev exposure 仍限制结论 |

**是否值得进一步开发学习式真实历史推荐模型？** **当前不值得。** 没有历史增量证据；优先缺口是：
1. 补齐剩余 62% history metadata（Home/Clothing 等大类别）；
2. 对齐/重建一个当前 Electronics catalogue 的 dense index（本轮禁止，需单独里程碑）；
3. 提升自然 target 可达率（135 条中 **115 条** 连 Top-100 都不可达，114 条 target-included 任务 target 不在自然 Top-100）；
4. 在这些修复前不做真实历史 LTR/微调。

## 9. 交付与限制

- `P14D1R_REPORT.md`、`P14D1R_SOURCE_DIFF.txt`
- `data/p14d1r/`：历史 metadata 映射、官方/C4 来源与 coverage manifest、recovered pools
- `runs/p14d1r/`：retrieval diagnosis、真实/消融/负对照结果、逐任务、盲评 25 条 pending、decision
- 测试：`tests/test_p14d1r_history.py` 7 passed（相关回归另行记录）
- 未训练/SFT、未改 V0/Gate/Planner/Memory/旧标签、未运行公开 Test/Reserved Test、未启停实例、未 commit；Dev 为 exposed；target 时间未独立重算。
