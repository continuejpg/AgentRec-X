# AgentRec-X — Memory-Aware Recommendation v1 报告

> 状态：**Phase 2 已完成（本地 CPU，真实训练 + 真实测试）**
> 范围：用真实 Amazon Reviews 2023 Electronics ratings 构建样本，训练 LightGBM Memory OFF / ON 两个 pointwise 模型，并报告真实测试指标。
> 本轮**没有**启动 AutoDL/GPU、没有调用 Qwen、没有重建 BGE、没有修改 Planner / Scorer / Gate / P18、没有重新下载大型数据。

---

## 0. 执行摘要

- 实际使用 **1,104,764** 条 Amazon Electronics catalog ratings（其中 1 条 rating=0 被丢弃），来自本地已有的 **Electronics.valid.csv + Electronics.test.csv** 派生文件；**没有找到本地 Electronics.train.csv**，也没有下载。
- 生成 **50,741** 个真实预测目标：
  - **train 22,427**
  - **val 14,157**
  - **test 14,157**
- 覆盖 **14,157 个真实用户**、**7,147 个真实商品**；历史长度均值 **1.589**，最大 **60**。
- LightGBM CPU 两个模型：
  - **Memory OFF**：95 个 item metadata + training-only popularity 特征。
  - **Memory ON**：同样 95 个特征 + **118 个 inferred-from-ratings memory 特征**。
- 真实 test 结果：
  - Rating Regression：MAE **0.855586 → 0.756594**（降低 **0.098992**，相对 **11.57%**）；RMSE **1.135553 → 1.065706**（降低 **0.069847**）。
  - High/Low Classification：ROC-AUC **0.571768 → 0.749143**（+**0.177375**）；PR-AUC **0.917892 → 0.957258**（+**0.039366**）；Log Loss **0.361290 → 0.290767**。
- Memory gain 在所有历史长度分组都存在，且随历史变长扩大：历史长度 2 的 MAE 改善 **0.093950**，历史 10+ 的改善 **0.164110**。
- 真实 held-out pair 排序组：存在 **4,218 个 size-2 observed pairs**，但排序指标弱且 ceiling 明显；pointwise 改善没有完全转化为 pairwise ordering 改善。
- **没有训练 LambdaMART**：当前只有 size-2 的 observed pair groups，没有可用的多候选/曝光组；建议 Phase 3 再构建 leave-two-out 或曝光数据后尝试。

---

## 1. 实际使用的原始数据

| 数据 | 路径 | SHA-256 / 说明 |
|---|---|---|
| Amazon Electronics catalog ratings | `data/gold/amazon_reviews_electronics_v1/electronics_catalog_ratings.csv` | `8cd8df42dd7e31082f678048d8c3ff082a1e65bae3456216f3ec61c9ef6d8ae3`；84,484,749 bytes；1,104,764 rows |
| 商品 metadata sidecar | `data/enrichment/electronics_metadata_sidecar.jsonl` | `2b76df5b650b5d017dcb9918d871e6584b793de6c05d7dac97e9df6aef1bda07`；149,403,094 bytes；76,263 items |
| 当前 catalog identities | `data/p13r/electronics/catalog_meta.jsonl` | 76,918 items；用于 provenance/身份背景，不直接作为模型 label 源 |
| 旧 gold 数据 manifest | `data/gold/amazon_reviews_electronics_v1/manifest.json` | 记录 raw valid/test 来源、catalog intersection、标签类型 |

### 1.1 本地 train 分区检查

本地可用的 ratings 文件是从 **Electronics.valid + Electronics.test** 过滤到当前 catalog 的
`electronics_catalog_ratings.csv`：

```text
source raw (recorded in the gold manifest, not accessed this round):
  /root/autodl-tmp/p13r/raw/Electronics.valid.csv
  /root/autodl-tmp/p13r/raw/Electronics.test.csv
local derived file:
  data/gold/amazon_reviews_electronics_v1/electronics_catalog_ratings.csv
source_split:
  Electronics.valid 522,258 rows
  Electronics.test  582,506 rows
```

本地没有发现 `Electronics.train.csv`；本轮**没有访问 AutoDL、没有下载任何新的评论数据**。
因此训练集不是 Amazon 官方 train split，而是从真实 valid/test ratings 重新构造的
**per-user temporal split**。

### 1.2 标签来源

- 监督标签是**真实 observed rating 1–5**。
- 没有人工 pairwise preference 标签。
- 没有把未交互/未评分商品当作负例。
- 没有生成虚构的用户满意度标签。

---

## 2. 实际生成的训练样本数量

| Split | Rows | Users | Items |
|---|---:|---:|---:|
| train | **22,427** | 14,157 | 4,890 |
| val | **14,157** | 14,157 | 3,642 |
| test | **14,157** | 14,157 | 3,565 |
| **total target rows** | **50,741** | 14,157 | 7,147 |

Train 组成：

```text
history_length = 0: 14,157 rows  (cold-start training targets)
history_length >= 1: 8,270 rows
```

说明：每个用户的事件都会作为 target 出现在且仅出现在一个 split；train 中包含第一件商品的
cold-start target（history 为空），因此 train 比之前的 8,270 更大。Val/test 每个用户各一条，
分别对应倒数第二条和最后一条真实评分。

每个 target 同时保存：

- `history_item_ids` / `history_ratings` / `history_timestamps`；
- `history_metadata_json`：历史商品 title、main/leaf category、canonical attributes、
  attribute coverage、price 的逐项 metadata；
- target 商品 metadata 字段：title、main/leaf category、attributes、attribute count/coverage、price；
- `item_features.parquet`：全部 7,147 个商品的可复用 metadata 特征表。

因此历史商品 metadata 既逐 row 保存在 `history_metadata_json`，也能通过
`item_features.parquet` join 还原。

Ranking pair 评测集（不是训练集）：

```text
ranking_pairs.parquet
  groups: 4,218
  rows:   8,436
  group size: 2
```

---

## 3. 真实用户、商品、历史长度统计

### 3.1 数据漏斗

```text
raw catalog ratings:                 1,104,764
rating in [1,5]:                     1,104,763  (1 row rating=0 dropped)
raw users:                           1,005,308
raw catalog items:                      27,403
users with >=3 catalog interactions:    14,157
events used from those users:            50,741
items appearing in used events:           7,147
users excluded (fewer than 3):          991,151
users with exactly 2 interactions:       62,872  (not used for target/split)
```

### 3.2 历史长度

| Split | mean | median | max | 0 | 1 | 2 | 3-4 | 5-9 | 10+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1.589 | 1.0 | 60 | 14,157 | 14,157 | 14,157 | 5,903 | 1,947 | 420 |
| train | 0.964 | 0.0 | 58 | 14,157 | 4,218 | 1,685 | 1,329 | 776 | 262 |
| val | 1.584 | 1.0 | 59 | 0 | 9,939 | 2,533 | 1,206 | 413 | 66 |
| test | 2.584 | 2.0 | 60 | 0 | 0 | 9,939 | 3,368 | 758 | 92 |

### 3.3 真实评分分布

| Split | 1★ | 2★ | 3★ | 4★ | 5★ | high(>=4) | low(<=2) | neutral(3) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1,403 | 688 | 1,442 | 3,236 | 15,658 | 18,894 | 2,091 | 1,442 |
| val | 854 | 481 | 900 | 1,946 | 9,976 | 11,922 | 1,335 | 900 |
| test | **883** | **454** | **867** | **1,925** | **10,028** | **11,953** | **1,337** | **867** |

分类测试集（去掉 3★）为 **13,290** rows，正类率 **0.899398**，属于明显类别不平衡。

### 3.4 Metadata 覆盖

- 7,147 个实际使用商品中，**42 个**没有 sidecar metadata（0.59%）。
- 这些商品仍保留真实 rating 和 popularity 特征；metadata 特征使用 `meta_missing=1`，不伪造属性。
- 76,263 个 sidecar items 中，只有 7,147 个进入本轮实际训练/评测。

---

## 4. Feature 列表与设计

### 4.1 Non-Memory Features（95）

| 组 | 数量 | 说明 |
|---|---:|---|
| item metadata base | 13 | title len、text len、feature count、description count、detail count、price log、missing flags 等 |
| main category one-hot | 11 | 来自 sidecar `main_category` |
| leaf category one-hot | 15 | 来自 sidecar categories 叶子类目 |
| detail key presence | 20 | Brand、Manufacturer、Color、Connectivity Technology、Form Factor、Special Feature、Material、Item Weight、Product Dimensions、Compatible Devices、Power Source、Batteries、Model Number、Item model number、Date First Available、Included Components、Cable Type、Hardware Platform、Operating System、Screen Size |
| canonical attribute presence | 23 | over_ear、on_ear、in_ear、wireless、wired、bluetooth、microphone、rechargeable、noise_cancelling、waterproof、foldable、portable、gaming、mechanical、silent、ergonomic、touch、rgb、led、usb、hdmi、4k、remote |
| popularity | 13 | pop_seen_train、count、count_log、user_count、user_count_log、mean_rating、std_rating、high_share、low_share、popularity_pct、age_days、main_cat_count_log、leaf_cat_count_log |

**没有使用**：`user_id`、`target_item_id`、target rating、history ratings、metadata sidecar 的
`average_rating` / `rating_number`（后者可能聚合到目标 review）。

### 4.2 Memory Features（118）

全部来自 **历史评分 + 历史商品 metadata**，标记为 `inferred_from_ratings`，**不是**
`explicit_event_sourced_preference`。

#### 历史评分/数量/分布（17）

```text
mem_hist_count
mem_hist_count_log
mem_hist_mean_rating
mem_hist_std_rating
mem_hist_min_rating
mem_hist_max_rating
mem_hist_high_count
mem_hist_low_count
mem_hist_neutral_count
mem_hist_high_share
mem_hist_low_share
mem_hist_neutral_share
mem_hist_rating_1_share
mem_hist_rating_2_share
mem_hist_rating_3_share
mem_hist_rating_4_share
mem_hist_rating_5_share
```

#### 最近趋势、时间间隔、时间衰减（14）

```text
mem_hist_last_rating
mem_hist_last2_mean
mem_hist_last3_mean
mem_hist_last5_mean
mem_hist_last_minus_prev_mean
mem_hist_trend_slope
mem_hist_days_since_last
mem_hist_tenure_days
mem_hist_avg_gap_days
mem_hist_last_gap_days
mem_hist_decay_weight_sum
mem_hist_decay_mean_rating
mem_hist_decay_high_share
mem_hist_decay_low_share
```

时间衰减半衰期固定为 **180 天**，`decay = 0.5 ** (days / 180)`。

#### 历史属性证据、价格与 coverage（6）

```text
mem_hist_attr_coverage_mean
mem_hist_high_attr_coverage_mean
mem_hist_low_attr_coverage_mean
mem_hist_price_mean_log
mem_hist_high_price_mean_log
mem_hist_low_price_mean_log
```

#### 候选商品与历史偏好的匹配/冲突（12）

```text
mem_candidate_high_match_count
mem_candidate_low_conflict_count
mem_candidate_high_match_rate
mem_candidate_low_conflict_rate
mem_candidate_high_attr_weight
mem_candidate_low_attr_weight
mem_candidate_attr_seen_ratio
mem_candidate_attr_unknown_count
mem_candidate_high_main_cat_share
mem_candidate_low_main_cat_share
mem_candidate_high_leaf_cat_share
mem_candidate_low_leaf_cat_share
```

#### 历史属性出现次数（23 attrs × 3 = 69）

对以下每个 attribute：

```text
over_ear, on_ear, in_ear, wireless, wired, bluetooth, microphone,
rechargeable, noise_cancelling, waterproof, foldable, portable,
gaming, mechanical, silent, ergonomic, touch, rgb, led, usb, hdmi, 4k, remote
```

生成：

```text
mem_all_attr_count_<attr>
mem_high_attr_count_<attr>   # history rating >=4
mem_low_attr_count_<attr>    # history rating <=2
```

总计：17 + 14 + 6 + 12 + 69 = **118** memory features。

### 4.3 特征兼容性

- Non-Memory 特征只依赖商品 metadata / training-split popularity，可在预测时获得。
- Memory 特征只依赖目标时间之前的评分历史，**不能读取目标 rating 或未来事件**。
- 本轮明确区分：
  - `inferred_from_ratings`（实际使用）
  - `explicit_event_sourced_preference`（未使用，留待生产接入）
- 兼容 adapter 见 §10。

---

## 5. 数据切分与避免未来泄漏的具体做法

### 5.1 Split 规则

```text
for each user:
  sort catalog ratings by timestamp ASC
  last rating        -> test target
  second-last rating -> val target
  all earlier ratings -> train targets (first has empty history)
  target item never in its own history
  history timestamps strictly < target timestamp
```

- 每个 rating event 只作为 target 出现一次；
- Val/test 每个用户各一条，没有用户重复；
- 历史里允许出现更早的 train/val event（这是正常的顺序历史），但不能出现 target 本身或未来事件。

### 5.2 Item popularity 防泄漏

- `pop_*` 只从 **training split** 计算。
- 对 training row，item 的 count / rating sum / high count / low count / user count 使用
  **leave-one-out** 去掉该 row 自己的 evaluation，防止模型读取自己的 label。
- Val/test 只读取 training-split popularity，不读取 val/test ratings。
- `average_rating` 和 `rating_number` 来自 metadata sidecar，可能聚合了包含 target review 的
  全部 review，因此**完全排除**。

### 5.3 其他防泄漏约束

- 不使用 `user_id`、`target_item_id`、`history_item_ids`、`history_ratings`、
  `history_timestamps`、`source_file_split` 作为 feature。
- 不把未评分商品作为负例；模型只在已评分 target 上训练/测试。
- 目标商品 metadata 只使用商品属性，不使用评论/评分聚合。
- `ranking_pairs.parquet` 只用于独立评测：同一用户最后两个已评分商品共享同一个严格
  历史截断，不进入训练/early stopping。

### 5.4 自动验证

`tests/test_memory_ranking_data.py` 验证：

- target 从不出现在自己的 history；
- 所有 history timestamp 严格早于 target timestamp；
- 50,741 个 target 事件唯一且只属于一个 split；
- val/test 每个用户只有一条；
- ranking pair 两个候选共享同一 history；
- cold-start row 的 history 为空；
- feature schema 与列一致；
- label 是 observed rating，未评分商品没有被当作负例。

---

## 6. Memory OFF 与 Memory ON 模型实际测试指标

### 6.1 训练配置（两个模型完全相同）

```text
LightGBM 4.7.0, CPU
objective: regression / binary
n_estimators: 600
learning_rate: 0.05
num_leaves: 31
min_child_samples: 40
subsample: 0.9, subsample_freq: 1
colsample_bytree: 0.9
reg_lambda: 1.0
early stopping: 50 on validation
random_state: 42
n_jobs: 4
```

特征数：

```text
Memory OFF:  95 features
Memory ON:  213 features (95 non-memory + 118 memory)
```

Best iteration：

```text
regression  OFF: 8
regression  ON: 71
classification OFF: 124
classification ON:  56
```

OFF 的 regression 在验证集很早停止，说明纯 item/popularity 特征在本 test 上解释力有限；
ON 使用更长的训练路径并取得更好的 test 指标。

### 6.2 Rating Regression

| Model | MAE | RMSE | mean actual | mean predicted |
|---|---:|---:|---:|---:|
| Constant train mean | 0.860477 | 1.141330 | 4.395847 | 4.384849 |
| Memory OFF | **0.855586** | **1.135553** | 4.395847 | 4.376661 |
| Memory ON | **0.756594** | **1.065706** | 4.395847 | 4.422025 |

相对 Memory OFF：

```text
MAE  reduction: 0.098992  (11.5701%)
RMSE reduction: 0.069847  ( 6.1509%)
```

Paired row bootstrap（500 次）：

```text
MAE  delta ON-OFF: -0.099063, 95% CI [-0.104229, -0.094166]
RMSE delta ON-OFF: -0.070097, 95% CI [-0.077139, -0.063073]
```

负 delta 表示 Error 降低；区间不跨 0。

### 6.3 High/Low Classification

测试集去掉 3★ 后：**13,290 rows**，high(>=4) **11,953**，low(<=2) **1,337**，
正类率 **0.899398**。Majority-class accuracy 为 **0.899398**。

| Model | ROC-AUC | PR-AUC | Log Loss | Accuracy |
|---|---:|---:|---:|---:|
| Memory OFF | 0.571768 | 0.917892 | 0.361290 | 0.888864 |
| Memory ON | **0.749143** | **0.957258** | **0.290767** | **0.899699** |
| Delta (ON-OFF) | **+0.177375** | **+0.039366** | **-0.070523** | +0.010835 |

Paired bootstrap AUC delta（300 次）：**+0.177212, 95% CI [0.161542, 0.193278]**。

注意：

- Accuracy 接近 majority baseline，因为正类率约 90%；**accuracy 不能作为主要结论**。
- Memory OFF 的 accuracy 甚至低于 majority baseline；AUC/PR-AUC 更能反映排序能力。
- Confusion matrix（threshold 0.5）：
  - OFF：`[[37, 1300], [177, 11776]]`
  - ON：`[[45, 1292], [41, 11912]]`
- 模型对 high 类的召回很高，但 low 类样本极少；PR-AUC 的绝对提升 +0.039 要在类别不平衡背景下理解。

---

## 7. 不同历史长度下 Memory 的作用

### 7.1 Regression MAE by history length

| History length | n | OFF MAE | ON MAE | MAE reduction (OFF-ON) |
|---|---:|---:|---:|---:|
| 2 | 9,939 | 0.884635 | 0.790685 | **0.093950** |
| 3 | 2,533 | 0.820589 | 0.713535 | **0.107054** |
| 4 | 835 | 0.743589 | 0.633058 | **0.110531** |
| 5-9 | 758 | 0.737089 | 0.619544 | **0.117545** |
| 10+ | 92 | 0.673665 | 0.509555 | **0.164110** |

趋势：历史越长，Memory 的 MAE 改善越大；但 10+ 只有 92 个 test rows，不能过度解读。

RMSE 同样在所有分组下降：

```text
2:   1.177582 -> 1.108194
3:   1.088698 -> 1.012080
4:   0.943809 -> 0.876589
5-9: 0.937286 -> 0.886425
10+: 0.812908 -> 0.665275
```

### 7.2 Classification AUC by history length

| History length | binary n | OFF AUC | ON AUC | delta |
|---|---:|---:|---:|---:|
| 2 | 9,298 | 0.574103 | 0.738499 | +0.164396 |
| 3 | 2,404 | 0.553462 | 0.751194 | +0.197732 |
| 4 | 782 | 0.551255 | 0.773295 | +0.222040 |
| 5-9 | 719 | 0.608209 | 0.805562 | +0.197353 |
| 10+ | 87 | 0.876471 | 0.752941 | **-0.123530** |

10+ 分组只有 87 rows，AUC 下降属于小样本不稳定；同组 log loss 从 0.142287 降到 **0.079396**，
accuracy 从 0.977011 升到 0.988506，不能只看 AUC 单项。

### 7.3 按 target attribute evidence coverage

Test 商品属性覆盖只落在三个桶（没有 0.75–1.00）：

| Coverage | n | OFF MAE | ON MAE | reduction |
|---|---:|---:|---:|---:|
| 0.00-0.25 | 11,559 | 0.847542 | 0.740437 | 0.107105 |
| 0.25-0.50 | 2,553 | 0.893749 | 0.831788 | 0.061961 |
| 0.50-0.75 | 45 | 0.756463 | 0.640867 | 0.115596 |
| 0.75-1.00 | 0 | — | — | — |

Memory gain 在低 coverage 商品上仍然存在，说明它主要来自用户历史评分模式，而不是依赖
完整商品属性。

---

## 8. 是否存在可用于 LambdaMART 的真实 Ranking Groups

### 8.1 结论

**存在真实 observed pair groups，但只有 size-2，不是可用于可靠 LambdaMART 的多候选组。**

本轮额外构建了 `ranking_pairs.parquet`：

```text
users with >=4 interactions: 4,218
groups: 4,218
group size: 2
candidates: 8,436
```

构造方式：

```text
对于每个有 >=4 条 catalog ratings 的用户：
  history = 最后两条之前的所有 ratings
  candidates = 最后两条真实已评分商品
  两个 candidate 共用同一个 history cutoff
```

没有使用任何 unobserved 商品作为负例；两个 candidate 都是真实 observed ratings。

### 8.2 真实 pair 排序指标

| Metric | OFF | ON | Delta (ON-OFF) | n groups |
|---|---:|---:|---:|---:|
| Pairwise ordering accuracy（所有 rating 不同的 pair） | 0.569103 | **0.586711** | **+0.017608** | 1,505 |
| Pairwise ordering accuracy（high vs low） | 0.594005 | **0.613079** | **+0.019074** | 367 |
| Pairwise ordering accuracy（high vs not-high） | 0.587379 | 0.576976 | **-0.010402** | 721 |
| MRR（high relevance） | 0.962243 | 0.962243 | 0.000000 | 4,039 |
| nDCG@2（graded rating gain） | 0.983933 | 0.984613 | +0.000680 | 4,218 |
| nDCG@2（binary high relevance） | 0.930876 | 0.930876 | 0.000000 | 4,218 |

解释：

- **不存在有意义的 nDCG@5 / nDCG@10**：group size 只有 2，报告这些数字会误导。
- 大多数 pair 是 5|5（2,396 组）、4|5（699 组），排序空间很小；MRR 和 binary nDCG 因此接近 ceiling。
- 有差异的 pair 只有 1,505 组；high-vs-low 只有 367 组。
- Memory ON 在“所有不同 rating”和“high vs low”上升，但在“high vs not-high”上略降，
  说明 pointwise MAE 改善**没有完全转化**为稳定 pairwise ordering 改善。

### 8.3 可训练 group 的数量

```text
users >= 5 interactions: 1,685
users >= 6 interactions:   850
```

即使为这些用户构造 size-2/3 的训练 group，规模也很小，且仍是“同一用户的两个已评分商品”，
不是推荐曝光候选集。因此本轮没有把 pair 数据当作 LambdaMART 训练集。

---

## 9. 下一步是否值得进行 LambdaMART

**当前不建议直接进入 LambdaMART。** 理由：

1. 当前只有 **4,218 个 size-2 held-out pairs**；可训练 group 更少（只有 1,685 个用户有 >=5 条交互）。
2. 没有真实推荐曝光/候选集合，只有用户已评分商品；listwise/pointwise 排序目标会非常窄。
3. Pointwise Memory ON 的 MAE/AUC 改善明显，但 pair ordering 只在部分子集改善，说明
   “rating prediction 提升”与“候选排序提升”不是同一件事。
4. 现在训练 LambdaMART 很容易在一个很小的 observed-pair benchmark 上过拟合，且无法证明
   full candidate ranking 收益。

**建议**：

- Phase 3 先构造 **leave-two-out / leave-K-out** 的真实组，并确保组内候选共享同一 history cutoff；
- 或收集带曝光/日志的 candidate sets（同一请求下的多个候选及其真实反馈）；
- 在这些数据上再训练 LambdaMART / listwise ranker，并与 pointwise ON 对比；
- 在此之前，把 pointwise Memory ON 作为 signal/feature 来源，而不是生产 ranker。

---

## 10. Learned Ranker 与 AgentRec-X 生产接口的差异

### 10.1 当前生产的输入/输出

AgentRec-X 当前生产链路：

```text
Original BGE Top-k
 -> PreferenceAwareScorer（固定权重 + Event-Sourced active memory + hard constraints）
 -> structural role/hard filter
 -> DecisionGate
 -> presentable[0]
```

Scorer 使用的是 **explicit Event-Sourced preferences**（用户主动声明的偏好）和 query hard
requirements，不是 Amazon rating history。

### 10.2 本轮模型的输入

本轮模型：

```text
candidate item metadata
+ training-split popularity
+ inferred_from_ratings user history features
-> LightGBM pointwise rating
```

两者不是同一个输入空间。具体差异：

| 维度 | 生产 Scorer | 本轮 Memory Ranker |
|---|---|---|
| 用户偏好来源 | Event-Sourced active preferences | inferred from prior ratings |
| Query 信号 | BGE rank / query hard requirements | **没有 query 特征** |
| 候选范围 | 当前 BGE Top-k 固定池 | 已评分 target 的历史候选 |
| 标签 | 无监督/规则权重 | observed rating 1–5 |
| 目标 | 排序/选择 | pointwise rating / high-low |
| 负例 | unobserved 不参与 | unobserved 不参与 |
| 输出 | presentable[0] | 预测评分/概率 |

### 10.3 Feature Adapter（本轮新增）

`experiments/memory_ranking/feature_adapter.py` 提供：

- item metadata → `meta_*` features；
- training-split popularity context → `pop_*` features（需要授权历史 popularity table）；
- Event-Sourced active preferences → **attribute-level proxy** for positive/negative counts；
- query relevance 输入槽位（为未来 v2 保留）；
- 显式兼容性报告：`model_ready`、missing features、unknown values。

**它明确声明不能直接接入生产模型**：

- Event-Sourced active memory 无法重建 rating-history 的均值/趋势/衰减/价格/类目分布；
- 当前 v1 模型没有 query relevance 特征，不能直接接收 production BGE rank；
- popularity 需要服务时的历史统计表；
- 没有任何 production Scorer/Gate 行为修改。

`model_ready` 在本轮始终为 **False**，缺的特征会被明确列出，不伪装成已接入。

### 10.4 推荐接入路径

1. **Retrain v2**：加入 query relevance（BGE rank、score、semantic score）和候选池上下文。
2. **Memory projection**：将 Event-Sourced active preferences 投影成与训练一致的 memory
   features；或单独训练 event-memory 特征并与 rating-history 特征对齐。
3. **Candidate policy**：保持 BGE Top-k 固定池，先做 shadow rerank，不改变 Gate 的
   presentable/finalize 语义。
4. **Offline gate**：在 P20 19 个 recommend tasks + provisional acceptable labels 上对比
   Learned Ranker vs 现有 Scorer，报告 Hit@1/3/5、MRR、Gate finalization。
5. **Online/shadow**：收集真实曝光和用户反馈后，才考虑生产替换。

---

## 11. 工程与验证

### 11.1 新增文件

```text
experiments/memory_ranking/__init__.py
experiments/memory_ranking/features.py
experiments/memory_ranking/build_dataset.py
experiments/memory_ranking/train_baseline.py
experiments/memory_ranking/evaluate.py
experiments/memory_ranking/feature_adapter.py
tests/test_memory_ranking_data.py
tests/test_memory_ranking_adapter.py
docs/MEMORY_RANKING_V1_REPORT.md
```

### 11.2 生成产物（`runs/`/`data/` 不提交 Git）

```text
data/memory_ranking_v1/
  train.parquet
  val.parquet
  test.parquet
  ranking_pairs.parquet
  item_features.parquet
  feature_schema.json
  manifest.json

runs/memory_ranking_v1/
  memory_off_model.txt
  memory_on_model.txt
  memory_off_classifier.txt
  memory_on_classifier.txt
  training_summary.json
  test_predictions.parquet
  comparison.json
  feature_importance.json
  ranking_group_metrics.json
```

### 11.3 执行命令

```bash
python -m experiments.memory_ranking.build_dataset
python -m experiments.memory_ranking.train_baseline
python -m experiments.memory_ranking.evaluate
python -m pytest tests/test_memory_ranking_data.py tests/test_memory_ranking_adapter.py -q
```

验证结果：

```text
11 passed
```

### 11.4 数据/模型完整性

- train/val/test 50,741 个 target 事件唯一且 split 互斥。
- target 从不出现在自己 history；history timestamp 严格早于 target。
- 模型文件已保存为 LightGBM text format。
- `comparison.json` 嵌入了 dataset manifest 摘要。
- `feature_importance.json` 记录 gain importance。
- `manifest.json` 记录 raw/data hashes、rows、features、leakage controls、protection flags。

### 11.5 Git

本轮只创建本地 commit，不 Push。代码、测试和本报告在同一个本地提交中；最终 HEAD 以
`git log -1` 为准。

保护性检查：

```text
Production Scorer modified: no
Production Gate modified: no
Planner modified: no
Retrieval/BGE modified: no
P18 modified: no
GPU / AutoDL / Qwen used: no
Large raw data re-downloaded: no
Git Push: no
```

---

## 12. 局限性与未决问题

1. **用户规模小**：1,005,308 个原始用户中只有 14,157 个有 >=3 个 catalog interactions；
   991,151 个用户被排除，62,872 个只有 2 条交互的用户未用于 target/split。
2. **商品覆盖小**：7,147 个实际使用商品（源 catalog ratings 有 27,403 个）。
3. **没有本地 raw train 分区**：只使用 valid/test 派生的 catalog ratings；没有访问 AutoDL，
   没有重新下载。
4. **Train/test 用户重叠**：这是 per-user temporal split，不是用户级 cold-start split；
   test 用户都有至少 2 条历史。
5. **没有真实推荐曝光数据**：只有用户已经评分的商品；没有完整候选集/未选择日志。
6. **没有 human pairwise 标签**：标签是 observed rating 1–5，不是 pairwise preference。
7. **分类类别不平衡**：正类率约 90%，accuracy 接近 majority baseline；AUC/PR-AUC 更可信，
   但仍不能代表真实推荐满意度。
8. **Ranking pair 只有 size-2**：nDCG@5/@10 不可用；pairwise 改善不覆盖所有子集。
9. **42 个商品缺失 metadata**（0.59%），使用 missing indicator，不伪造属性。
10. **Event-Sourced Memory 与 inferred ratings 不同空间**：Adapter 目前只是 partial，
    `model_ready=False`；没有生产接入。
11. **LightGBM OFF regression 早停于 8 iterations**：非记忆 baseline 很弱；Memory gain
    包含用户评分历史/用户评分尺度信号，不能等同于“商品偏好理解”。
12. **没有线上/真实用户满意度证据**：`real user satisfaction = not established`。

---

## 13. 结论

- 已交付**真实训练数据**、**两个已训练完成 CPU 模型**、**真实 test 指标对照**。
- 在本数据集上，Memory ON 对 observed rating prediction 和 high/low classification 都有
  明显、bootstrap 区间不跨 0 的改善。
- 改善随用户历史长度增加而增大，符合“长期记忆有贡献”的预期。
- 但 pointwise 改善没有完全转化为稳定的 size-2 pairwise ordering 改善；当前数据不足以
  训练/验证 LambdaMART。
- 下一步应优先补 **leave-two-out / 曝光候选组**、**query relevance 特征** 和
  **Event-Sourced memory 投影**，再进行 v2 retrain / shadow rerank。
