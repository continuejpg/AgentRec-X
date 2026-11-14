# Ranker Refresh — Archive Legacy Models, ShopRanker-0.6B, DCN-V2 Preparation

> 状态：**Ranking Refresh 已完成（真实实验，非仅可行性分析）**
> 生产默认 Ranker 仍为 `PreferenceAwareScorer`；本轮没有修改 Planner / Scorer / Gate / P18，没有训练 Qwen，没有重建 BGE。
> 旧 LightGBM / Neural Memory Ranker 已退出维护；ShopRanker-0.6B 已完成真实 zero-shot P20 Shadow；DCN-V2 已完成真实数据 Builder、Feature Schema 和 GPU forward/backward smoke，但因独立 Event-Sourced Memory 监督不足，**没有伪造正式训练**。

---

## 0. 结论摘要

1. **旧模型归档**：LightGBM 与 Neural Memory Ranker 的权重、日志和模型专用可重建输出已清理。
   - 本地释放 **7,708,600 bytes**（7.35 MiB）。
   - AutoDL 释放 **5,931,509 bytes**（5.66 MiB）。
   - 合计 **13,640,109 bytes**（约 13.01 MiB）。
2. **ShopRanker-0.6B 已完成真实 GPU zero-shot**（frozen，官方 prompt + yes/no logit）：
   - Query Only：Hit@1 **18/19**，MRR@10 **0.956140**
   - Query + Active Memory：Hit@1 **19/19**，MRR@10 **1.000000**
   - Existing Scorer Memory ON：Hit@1 **18/19**，MRR@10 **0.973684**
   - Qwen3-Reranker-0.6B Base + Memory：Hit@1 **18/19**，MRR@10 **0.973684**
   - P20 标签是 provisional，**不能据此宣布 ShopRanker 优于生产 Scorer**。
3. **Event-Sourced Memory Counterfactual**：16 cases，ShopRanker 排序变化 15/16，top1 变化 12/16；但方向一致性弱，尤其 negative preference 不可靠；2 个 conflict case 标记 `requires_clarification`。
4. **DCN-V2 数据**：真实 Build 出 **51 groups / 896 candidate rows**：
   - P20：19 groups / 380 rows / 210 provisional acceptable labels。
   - Counterfactual mechanistic：32 groups / 516 rows（无监督标签）。
   - 93 numeric features，其中 57 memory-related。
   - `training_ready=false`：只有 19 个 P20 groups，且 acceptable 以外的候选是 unjudged，不是 negatives；counterfactual 是规则构造，不是用户偏好 gold。
5. **DCN-V2 模型**：实现并实际执行 GPU forward/backward smoke：
   - Memory OFF 29,889 params；Memory ON 37,185 params。
   - `backward_ok=true`；GPU peak 17.88 MB。
   - **没有正式训练**，因为没有足够独立的 Query + Event-Sourced Memory + Candidate 监督。
6. **最终选择**：ShopRanker-0.6B 作为当前 zero-shot shadow 候选继续验证；DCN-V2 等待真实 memory supervision；旧 LightGBM / Neural 不再继续优化。

---

## 1. 旧模型归档摘要

完整归档：[docs/ARCHIVED_RANKING_EXPERIMENTS.md](docs/ARCHIVED_RANKING_EXPERIMENTS.md)

### 1.1 LightGBM OFF / ON

| 项目 | Memory OFF | Memory ON |
|---|---:|---:|
| Features | 95 | 213（95 + 118 memory） |
| Rating MAE | 0.855586 | 0.756594 |
| Rating RMSE | 1.135553 | 1.065706 |
| High/Low ROC-AUC | 0.571768 | 0.749143 |
| High/Low PR-AUC | 0.917892 | 0.957258 |
| Observed Pairwise | 0.569103 | 0.586711 |
| High-vs-Low Pairwise | 0.594005 | 0.613079 |

数据：`data/memory_ranking_v1/`；22,427 train / 14,157 val / 14,157 test；catalog-only history。
旧输出目录 `runs/memory_ranking_v1/` 已删除；指标保留在归档文档和 Git 历史中。

### 1.2 Neural Memory OFF / ON

| 项目 | Memory OFF | Memory ON |
|---|---:|---:|
| Params | 163,203 | 325,251 |
| GPU train time | 6.867 s | 15.043 s |
| Peak GPU | 201.8 MB | 458.8 MB |
| Full test Hit@10 | 0.916278 | 0.918407 |
| Full test nDCG@10 | 0.770639 | 0.775925 |
| Full test MRR@10 | 0.727636 | 0.733822 |
| Observed Pairwise | 0.509692 | 0.515964 |
| High-vs-Low Pairwise | 0.511196 | 0.521433 |

数据：`data/neural_memory_v1/`；43,435 train / 14,157 val / 77,029 test。
旧 `.pt` 权重已删除；数据保留为可复用真实序列/评测资产。

### 1.3 P20 Shadow

| Arm | Hit@1 | Hit@3 | Hit@5 |
|---|---:|---:|---:|
| Existing Scorer Memory ON | **18/19** | 19/19 | 19/19 |
| Neural OFF | 15/19 | 18/19 | 19/19 |
| Neural ON | 13/19 | 17/19 | 19/19 |
| Neural ON + BGE blend | 14/19 | 18/19 | 19/19 |

### 1.4 为什么迁移失败

1. 训练输入是 **inferred rating history**，不是 Event-Sourced explicit preferences。
2. 没有 query-relevance 训练特征；shadow 的 BGE blend 是手工后处理。
3. P20 Shadow 13/19 明显低于 Existing Scorer ON 18/19。
4. Observed Pairwise 接近随机（0.51-0.52）。
5. Phase 3 sampled candidate 中 popularity 远强于学习模型。
6. 非 Catalog history 缺 metadata；70% 历史事件为 UNKNOWN item。
7. 标签是 observed ratings，不是 human pairwise preference。

**决定：旧 LightGBM / Neural Ranker 退出维护，不再继续优化。**

---

## 2. 清理：准确文件与释放空间

### 2.1 本地删除

`runs/memory_ranking_v1/` 全部 9 个旧 LightGBM 输出：

```text
comparison.json
feature_importance.json
memory_off_classifier.txt
memory_off_model.txt
memory_on_classifier.txt
memory_on_model.txt
ranking_group_metrics.json
test_predictions.parquet
training_summary.json
```

**1,722,157 bytes**。

`runs/neural_memory_v1/` 中 8 个旧模型权重/可重建输出：

```text
memory_off.pt
memory_on.pt
memory_off_hard.pt
memory_on_hard.pt
memory_off_hard_noattr.pt
memory_on_hard_noattr.pt
expanded_lightgbm_off.txt
expanded_lightgbm_on.txt
```

**5,931,509 bytes**。

旧训练日志 `runs/neural_memory_v1/remote_logs/`（7 files）：**54,934 bytes**。

**本地合计释放：7,708,600 bytes（7.35 MiB）。**

### 2.2 AutoDL 远端删除

同一组 8 个旧模型权重文件：

```text
/root/autodl-tmp/AgentRec-X-phase2/runs/neural_memory_v1/memory_off.pt
... memory_on.pt
... memory_off_hard.pt
... memory_on_hard.pt
... memory_off_hard_noattr.pt
... memory_on_hard_noattr.pt
... expanded_lightgbm_off.txt
... expanded_lightgbm_on.txt
```

**远端释放：5,931,509 bytes（5.66 MiB）。**

**总释放：13,640,109 bytes（约 13.01 MiB）。**

### 2.3 明确保留

- Amazon Reviews / ESCI raw & gold
- Electronics Catalog / Metadata Sidecar / enriched catalog
- Original BGE Index / vectors
- Event-Sourced Memory 数据
- P14 / P18 / P20
- 生产 `PreferenceAwareScorer` / `DecisionGate` / Planner / CandidateLedger
- Qwen3.5 Base / Qwen3-Reranker-4B / BGE model
- `data/memory_ranking_v1/`、`data/neural_memory_v1/` 可复用数据
- 小型 compact JSON 汇总（comparison / shadow / subsets）
- 新 ShopRanker 模型与 DCN 数据

没有删除任何原始数据、生产组件或未来实验资产。

---

## 3. ShopRanker-0.6B 实际运行结果

### 3.1 模型与加载

- Model: `srpone/zoowork-shopranker-0.6b`
- 0.6B base + LoRA adapter；Frozen，无 Fine-tuning。
- 官方 scoring prompt、Tokenizer、原生 `yes/no` logit。
- 评分：`softmax(logits[:, [NO, YES]])[:, 1]`。
- LoRA 用 `PeftModel.from_pretrained(base, model_id)`，保持 adapter fp32（模型卡方法）。
- Qwen3-Reranker-0.6B Base 通过 `disable_adapter()` 使用同一 base，不额外下载。
- GPU: RTX 4090 D；Peak **3,838.2 MB**。

### 3.2 固定池协议

- P20 19 个 recommend tasks。
- Original BGE Top-20 候选池。
- Existing Scorer 的共享 eligible 集合（hard_tier=0 + role MATCH/NOT_REQUIRED）。
- Product text：title + product type + verified attributes + evidence + `unknown_attributes`（不把 UNKNOWN 写成缺失）。
- Memory text：只使用 active positive/negative preferences，作为 Query 后缀；removed/superseded 不进入输入。
- P20 标签只用于评测，不参与 Prompt 设计、权重或阈值调参。

### 3.3 结果

| Arm | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---:|---:|---:|---:|
| A. Existing Scorer Memory ON | 18/19 | 19/19 | 19/19 | 0.973684 |
| B. ShopRanker Query Only | 18/19 | 18/19 | 18/19 | 0.956140 |
| C. ShopRanker Query + Active Memory | **19/19** | **19/19** | **19/19** | **1.000000** |
| D. Qwen3-Reranker-0.6B Base + Memory | 18/19 | 19/19 | 19/19 | 0.973684 |

**B vs C（Memory 独立影响）：**

```text
Hit@1: 18/19 -> 19/19  (+1 task)
Hit@3: 18/19 -> 19/19
Hit@5: 18/19 -> 19/19
MRR@10: 0.956140 -> 1.000000
Top-1 changed: 7/19
Top-3 set changed: 11/19
Full ranking changed: 11/19
```

**C vs A（与现有 Scorer）：**

```text
Hit@1: 18/19 -> 19/19
MRR@10: 0.973684 -> 1.000000
Top-1 changed: 16/19
Top-3 set changed: 15/19
Full ranking changed: 18/19
```

注意：这是 **19 个 provisional P20 tasks** 上的差异；一个 task 的变化不足以证明统计或生产优势。**不宣布 ShopRanker 优于生产 Scorer。**

### 3.4 0051

```text
ShopRanker Query Only ranking:  B08D32JM5F > B004K6R43G > B07BKPKXTT
ShopRanker Query+Memory ranking: B08D32JM5F > B004K6R43G > B07BKPKXTT
acceptable = B08D32JM5F
scores Query Only: B08=0.955319, B004=0.826712, B07=0.798187
scores Query+Memory: B08=0.970688, B004=0.904651, B07=0.835484
```

ShopRanker 在两个输入下都把 acceptable `B08D32JM5F` 排第一；Memory 提高了所有分数，但未改变 0051 排序。
这是 zero-shot 模型的可用信号，不等同于显式 Memory 语义验证。

### 3.5 延迟与显存

```text
ShopRanker Query+Memory, 20 candidates, batch call, GPU:
mean 135.437 ms, p50 133.971 ms, max 180.016 ms
Peak GPU: 3,838.2 MB
```

### 3.6 Event-Sourced Memory Counterfactual

复用 Phase 1 的 16 个既有案例；ShopRanker Memory ON 在 baseline/mutated 两个状态上分别评分：

```text
cases: 16
top1 changed: 12/16
top3 set changed: 14/16
full ranking changed: 15/16
requires_clarification: 2 (cf010, cf011)
```

行为解读：

- **Positive preference**：
  - `cf001 rechargeable`：3 个 rechargeable 候选全部 rank 改善；
  - `cf002 on_ear`：2 个 on-ear 候选中 1 改善、1 变差；
  - `cf003 microphone`：12 个 microphone 候选中 6 改善、4 变差。
- **Negative preference**：
  - `cf004 over_ear`：10 个候选 3 改善、4 变差；
  - `cf005 microphone`：12 个候选 4 改善、6 变差。
  - **方向不一致**：negative preference 并不稳定压低对应候选，不能声称模型理解了负向记忆。
- **Update / Withdrawal**：全部 full ranking 变化；withdrawal 0051-like 2 cases top1 均变化；但无法证明变化符合用户意图。
- **UNKNOWN**：所有 4 个 case 排序变化；说明 Query 文本变化会带来非语义分数漂移，而不是严格 UNKNOWN 证据推理。
- **Conflict**：2 个 case 标记 `requires_clarification`；**没有强行输出正常 Top-K Gold**。

结论：ShopRanker 对 Memory 文本敏感，但**不是可靠的 Event-Sourced Memory 推理器**；只能作为 zero-shot shadow 候选。

---

## 4. 独立 Memory Ranking Evaluation 扩充

新增清单：[data/memory_eval_inventory/manifest.json](data/memory_eval_inventory/manifest.json)

按来源分为四类，**不混算**：

### A. Query Relevance Gold

| Source | Scale | Label |
|---|---|---|
| Amazon ESCI test | 181,701 rows / 8,956 queries | Human E/S/C/I |
| TREC 2025 Recommendation | 50 queries / 47 judged / 7,633 qrels | Human relationship labels |

ESCI/TREC 只证明 query relevance / product relation，不是长期 Memory Gold。

### B. Product Relation Gold

- ESCI E/S/C/I。
- TREC S2/S1/S0/C2/C1/C0/NR/UA。
- 与 Catalog 精确交集：ESCI 626 items；TREC 115 items（test-only）。

### C. Explicit Memory State Consistency

- Phase 1 **16 constructed cases / 126 checks**：
  - positive / negative / update / withdrawal / conflict / UNKNOWN。
- P20 19 recommend tasks（provisional acceptable）。
- 这些不是用户满意度 Gold；counterfactual 明确标记 mechanistic。

### D. User Preference / Choice Supervision

- Amazon Reviews 2023 Electronics：1,104,764 catalog interactions；8,029,455 full interactions。
- Observed ratings / next interactions；**没有 human pairwise preference**。
- Phase 3 expanded sequences：43,435 train / 14,157 val / 77,029 test。
- 不能把 unobserved 商品当作负反馈。

### 独立评测缺口

- 没有足够独立的 **Query + Event-Sourced Memory + Candidate** 人类偏好标签。
- ESCI/TREC 不能冒充长期 Memory。
- 16 个 counterfactual 是规则/状态构造，不是用户选择。
- 因此 DCN-V2 暂不能进行正式 supervised training。

---

## 5. DCN-V2 数据集（真实规模与标签来源）

路径：[data/dcn_v2_ranking_v1/](data/dcn_v2_ranking_v1/)

### 5.1 规模

| Source | Groups | Candidate Rows | Labels |
|---|---:|---:|---|
| P20 fixed pool | 19 | 380 | 210 provisional acceptable；170 unjudged |
| Counterfactual mechanistic | 32 | 516 | 无监督标签（`not_usable_for_supervision`） |
| **Total** | **51** | **896** | — |

Split：

```text
P20: train 300 rows / 15 groups; val 20 rows / 1 group; test 60 rows / 3 groups
Counterfactual: mechanistic 516 rows / 32 groups (not train/val/test)
```

Group-level split；同一 Query/Memory snapshot/candidate pool 在同一 group；counterfactual 不进入训练 Split。

### 5.2 Feature Schema

```text
numeric features: 93
  query features: 53
  candidate features: 40
  memory-related features: 57
  memory-off numeric: 36
```

字段包括：

- `recommendation_group_id`
- `query_id` / query len / hard constraint count / query soft preference count
- candidate item id
- BGE rank / BGE score / query relevance score
- active positive / negative memory counts and attribute flags
- preference match / conflict features
- UNKNOWN / attribute coverage features
- product metadata attributes
- supervised label + label provenance + split
- `memory_snapshot_id`

完整 schema：[data/dcn_v2_ranking_v1/feature_schema.json](data/dcn_v2_ranking_v1/feature_schema.json)

### 5.3 是否具备合理训练监督

**否，`training_ready=false`。**

原因：

1. 只有 19 个 P20 recommend groups / 380 rows；
2. P20 acceptable 之外是 **unjudged**，不是 negatives；
3. counterfactual 32 groups 是 rule-derived mechanistic，不是独立用户偏好；
4. ESCI/TREC 没有 Event-Sourced Memory；
5. Amazon ratings 没有 explicit memory event / candidate selection；
6. 没有真实曝光候选集。

因此本轮**没有训练 DCN-V2**，没有伪造训练指标。

---

## 6. DCN-V2 模型与真实 GPU Smoke

代码：

- [experiments/dcn_v2/build_dataset.py](experiments/dcn_v2/build_dataset.py)
- [experiments/dcn_v2/model.py](experiments/dcn_v2/model.py)
- [experiments/dcn_v2/smoke.py](experiments/dcn_v2/smoke.py)

模型：标准 DCN-V2 / DCN-Mix 小型实现：

- input projection；
- 3 个 element-wise cross layers；
- parallel MLP；
- scalar ranking head；
- memory OFF / ON 输入维度不同，结构同族。

真实 GPU smoke（`data/dcn_v2_ranking_v1/dcn_smoke.json`）：

| Model | Input dim | Params | Batch | Backward | GPU peak |
|---|---:|---:|---:|---|---:|
| Memory OFF | 36 | 29,889 | 64 | **true** | 17.88 MB |
| Memory ON | 93 | 37,185 | 64 | **true** | 17.88 MB |

```text
Memory OFF smoke loss (gradient-flow probe): 0.005381
Memory ON  smoke loss (gradient-flow probe): 0.008188
```

该 loss **只是无监督梯度流探针**，不是训练目标、不是模型质量指标。
`training_performed=false`，因为独立监督不足。

---

## 7. 公平比较与最终选择

### 7.1 P20 固定池

所有 P20 数字使用：

- 同一 Original BGE Top-20；
- 同一 eligible 集合；
- 同一 active memory；
- 同一 provisional acceptable 标签；
- 同一 Gate/eligibility 语义（ShopRanker 只做候选排序，没有替换 Gate）。

| Ranker | Memory OFF | Memory ON |
|---|---:|---:|
| Existing Scorer | 14/19 Hit@1（Phase 1 B） | **18/19**（Phase 1 C） |
| ShopRanker-0.6B | **18/19** Query Only | **19/19** Query + Memory |
| Qwen3-Reranker-0.6B Base | — | 18/19 |
| DCN-V2 | 未训练 | 未训练 |

**不能因为 ShopRanker 19/19 就宣布它优于生产 Scorer**：只差 1 个 provisional task，且同一标签集、同一 candidate pool、没有独立人工复核。

### 7.2 结论

1. **旧 LightGBM / Neural Memory Ranker 退出维护**，不再继续优化。
2. **ShopRanker-0.6B 是当前最有希望的 zero-shot shadow 候选**：Query+Memory 在 P20 上 19/19，但只有 provisional 证据；negative memory 方向不一致，不能称为 memory-understanding。
3. **DCN-V2 数据/模型链路已准备**，但监督不足，不进入正式训练。
4. **生产默认仍是 PreferenceAwareScorer**；所有新模型仅 Shadow，不自动替换。
5. 下一轮优先：
   - 独立人工审核的 Query+Memory+Candidate 偏好对；
   - 扩大 P20/真实场景（不只 19 tasks）；
   - ShopRanker 在更多独立 benchmark 上验证（ESCI/TREC 只作为 query relevance，不混 Memory）；
   - DCN-V2 在至少数百到数千独立 groups 上训练 Memory OFF/ON；
   - 真实曝光候选/点击/选择数据。

---

## 8. 工程、测试与 Git

### 8.1 新增代码

```text
experiments/shopranker/__init__.py
experiments/shopranker/benchmark.py
experiments/dcn_v2/__init__.py
experiments/dcn_v2/build_dataset.py
experiments/dcn_v2/model.py
experiments/dcn_v2/smoke.py
```

### 8.2 新增数据/证据

```text
runs/shopranker_v1/shopranker_results.json
runs/shopranker_v1/remote.log
data/dcn_v2_ranking_v1/groups.parquet
data/dcn_v2_ranking_v1/feature_schema.json
data/dcn_v2_ranking_v1/manifest.json
data/dcn_v2_ranking_v1/dcn_smoke.json
data/memory_eval_inventory/manifest.json
docs/ARCHIVED_RANKING_EXPERIMENTS.md
docs/RANKER_REFRESH_REPORT.md
```

### 8.3 实际运行

```text
ShopRanker:
  model download 1.2GB
  frozen Load + P20 19 tasks x (Query Only / Query+Memory / Qwen base+Memory)
  Counterfactual 16 cases x 2 states
  GPU RTX 4090 D, peak 3.84GB
  latency p50 133.971ms / 20 candidates, batch call

DCN-V2:
  dataset build actual 51 groups / 896 rows
  GPU forward/backward smoke for Memory OFF/ON
  no formal training
```

### 8.4 测试/验证

- ShopRanker actual GPU run completed; scores/probabilities finite.
- DCN-V2 smoke `backward_ok=true`.
- Legacy model archive/cleanup paths verified before deletion.
- ShopRanker P20 input uses original BGE Top-20 and shared eligible set.
- P20 labels not used for prompt/weight tuning.

### 8.5 Git

本轮只创建本地 commit，不 Push。最终 HEAD 以 `git log -1` 为准。

保护性检查：

```text
Production Scorer modified: no
Production Gate modified: no
Planner modified: no
Original BGE modified/rebuilt: no
P18 modified: no
Qwen training: no
Legacy model weights retained: no (archived)
ShopRanker fine-tuned: no (frozen)
DCN-V2 formally trained: no (insufficient supervision)
Git Push: no
```
