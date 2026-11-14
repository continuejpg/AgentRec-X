# AgentRec-X — Memory-Conditioned GPU Ranking v1 报告

> 状态：**Phase 3 已完成（AutoDL RTX 4090 D，真实 GPU 训练 + 真实离线评测）**
> 本轮使用完整 Electronics observed interactions 作为历史，目标商品限制在当前 76,918 商品 Catalog；训练 Neural Memory OFF / ON、LightGBM expanded baseline，并完成 P20 Learned Ranker Shadow。
> 未训练 Qwen，未微调 Qwen3-Reranker-4B，未重建 Original BGE，未修改生产 Planner/Scorer/Gate，未修改 P18。

---

## 0. 执行摘要

- 使用 **8,029,455** 条真实 Electronics 交互（5,139,179 users / 552,872 items），历史不再限制在 Catalog 内；**target 仍限制在 Catalog**。
- 有 **77,029 个用户**至少 2 次 Catalog 交互；生成 **134,621** 个真实 next-observed-catalog-item 样本：
  - **train 43,435**
  - **val 14,157**
  - **test 77,029**
- 相比 Phase 2：
  - test targets **14,157 → 77,029**（5.44×）
  - train targets **22,427 → 43,435**（1.94×）
- 训练两个神经模型：
  - **Neural Memory OFF**：163,203 params
  - **Neural Memory ON**：325,251 params（history encoder + candidate-conditioned attention + attribute interaction）
- 主评测采用 **category-matched sampled candidates（1 positive + 99 sampled unobserved catalog items）**；另保留 random sampled protocol 作为对照，不混合协议。
- 真实 test（77,029 targets）Neural OFF → ON：
  - Hit@10：**0.916278 → 0.918407**（+0.002129）
  - nDCG@10：**0.770639 → 0.775925**（+0.005286）
  - MRR@10：**0.727636 → 0.733822**（+0.006186）
- Common 14,157 用户子集：Hit@10 **0.905347 → 0.910221**，nDCG@10 **0.737407 → 0.743299**，MRR **0.688000 → 0.694111**。
- History length 10+ 子集增益更大：nDCG +0.009150，MRR +0.009543。
- **但 popularity baseline 在这些 sampled protocols 上更强**：
  - category-matched common：Popularity Hit@10 **0.950060**，nDCG **0.753286**，MRR **0.694438**
  - neural ON common：Hit@10 **0.910221**，nDCG **0.743299**，MRR **0.694111**
- **Pairwise 结果接近随机**：Neural ON pairwise **0.515964**，high-vs-low **0.521433**；LightGBM expanded pairwise **0.5927**，高于 neural。
- P20 Shadow：Existing Scorer Memory ON **18/19**，Neural ON **13/19**，Neural ON + BGE **14/19**；**不可迁移**。
- 结论：**Memory-conditioned neural ranker 已真实训练并产生小幅、方向一致的 nDCG/MRR 增益；但不足以超越 popularity baseline，也不能忠实消费 Event-Sourced Memory，当前不适合接入 AgentRec-X 生产路径。**

---

## 1. 扩大后的真实训练数据规模

### 1.1 原始数据

| 数据 | 内容 |
|---|---|
| Electronics.valid/test | 8,029,455 interactions；5,139,179 users；552,872 items |
| 当前 Catalog | **76,918 items** |
| Catalog 内 observed interactions | **1,104,764** |
| Sidecar metadata | 76,263 items；catalog 内 655 items 无 sidecar |
| Catalog 用户 | 1,005,308（至少 1 次 Catalog 交互）；**77,029 有 ≥2 次** |

没有重新下载原始数据；使用远程已存在的 `/root/autodl-tmp/p13r/raw/Electronics.valid.csv` 与 `Electronics.test.csv`。

### 1.2 Target 构造

```text
用户按时间排序全部 Electronics interactions
只把 Catalog 商品作为预测 target
last catalog event      -> test
second-last catalog     -> val（仅 ncat>=3 用户）
earlier catalog events  -> train
train 中 history 为空的 target 被丢弃（不是有效 next-item 样本）
history 使用 target 之前的所有 Electronics interactions
```

结果：

| Split | Rows | Users | Items | Full history mean | Max |
|---|---:|---:|---:|---:|---:|
| train | **43,435** | 36,345 | 6,957 | 8.116 | 757 |
| val | **14,157** | 14,157 | 3,642 | 10.572 | 758 |
| test | **77,029** | 77,029 | 8,360 | 5.295 | 779 |

原始 Catalog target 事件共 **176,485**；删除 41,864 条 empty-history train 后为 **134,621**。

### 1.3 扩大幅度

| 指标 | Phase 2 | Phase 3 | 倍数 |
|---|---:|---:|---:|
| Train targets | 22,427 | 43,435 | 1.94× |
| Test targets | 14,157 | 77,029 | 5.44× |
| Total targets | 50,741 | 134,621 | 2.65× |
| Test users | 14,157 | 77,029 | 5.44× |
| Candidate universe | 7,147 used items | 76,918 Catalog items | 10.8× |

完整交互中只有 **27,403** 个 Catalog 内商品有 observed ratings；全交互商品 **552,872**，因此全量商品中只有 **4.96%** 有当前 sidecar metadata。

---

## 2. 完整 Electronics 用户历史覆盖率

### 2.1 历史长度

| Split | Full history mean | Catalog history mean | Catalog coverage |
|---|---:|---:|---:|
| train | 8.116 | 0.498 | **6.13%** |
| val | 10.572 | 1.584 | **14.99%** |
| test | 5.295 | 1.291 | **24.38%** |

- 77,029 个用户的完整历史共 **594,477** 条 interactions，其中 Catalog 内 **176,485** 条（**29.7%**），Catalog 外 **417,992** 条（**70.3%**）。
- 也就是说：**历史显著扩大，但历史中的商品 metadata 没有同步扩大。**

### 2.2 Metadata 覆盖

```text
full interaction items:        552,872
items with sidecar metadata:    27,403  (only Catalog items with observed ratings)
metadata item coverage:          4.96%
Catalog items:                  76,918
Catalog items missing sidecar:      655
```

- Catalog 外 history item **没有 metadata**。
- 本轮没有伪造 metadata，也没有用 title/ASIN 模糊匹配。
- 非 Catalog history item 被编码为 **UNKNOWN item + rating/time only**；没有使用 item-id embedding，以避免模型只能依赖 ID、破坏 Event-Sourced 兼容性。
- 这是本轮 Memory ON 增益有限的最重要数据原因之一。

---

## 3. 模型结构与参数

### 3.1 Neural Memory OFF

```text
item encoder: 90-dim metadata/popularity features -> 128 -> 128
score head: 128 -> 256 -> 128 -> 1
rating auxiliary head: 128 -> 1
parameters: 163,203
```

不使用 user_id、target item ID、历史 items。

### 3.2 Neural Memory ON

```text
item encoder: 90-dim -> 128
history item encoder: (item 90 + rating/recency/known/mask) -> 128
history pooling:
  - masked additive attention
  - candidate-conditioned attention: candidate embedding · history item embeddings
memory encoder: 92-dim full-history stats / attribute counts -> 128
memory-candidate interaction:
  - candidate attribute vector (23)
  - historical high/low attribute frequencies (23+23)
  - attribute × high/low products (46)
  - 6 scalar match/conflict features
score head: 128*4 + memory + interaction -> 256 -> 128 -> 1
rating auxiliary head: 128 -> 1
parameters: 325,251
```

- 无 user_id / raw item-id embedding。
- 历史 sequence 只取最近 **64** 条；aggregate memory 使用完整历史。
- Memory 来源是 **inferred_from_ratings**，不是 explicit Event-Sourced preference。

### 3.3 训练协议

```text
train rows: 43,435
negative sample: 32 sampled unobserved catalog items / positive
batch size: 256
epochs: 8
optimizer: AdamW, LR=1e-3, weight_decay=1e-4
loss: sampled softmax ranking + 0.1 * rating MSE
GPU: RTX 4090 D 24GB, PyTorch 2.11+cu128
```

Memory OFF/ON 使用同一 train/val/test、同一 candidate arrays、同一训练预算、同一 early/best-epoch 规则。

### 3.4 真实 GPU Smoke

训练开始前执行了一个 batch=256 的真实 forward/backward：

```text
Memory OFF smoke: loss=3.519783, ranking=3.447151, rating=0.726326, backward_ok=True
Memory ON  smoke: loss=3.772890, ranking=3.500808, rating=2.720824, backward_ok=True
```

---

## 4. OFF/ON GPU 训练结果

### 4.1 训练成本

| Model | Params | Best epoch | Train time | Peak GPU |
|---|---:|---:|---:|---:|
| Memory OFF | 163,203 | 6 | **6.867 s** | **201.8 MB** |
| Memory ON | 325,251 | 7 | **15.043 s** | **458.8 MB** |
| ON attribute ablation | 294,275 | 8 | **14.321 s** | **421.8 MB** |

- GPU：RTX 4090 D，24,564 MiB，训练时无其他模型占用。
- 总 GPU 训练时间约 **36 秒**（两主模型 + ablation）；data build / evaluation 在 CPU。
- 只保存最终模型 6 个 `.pt`，没有保存大量中间 checkpoints。
- 数据盘启动时约 **11GB available**，训练/数据产物约 **167MB data + 5.8MB runs**，结束约 **10GB available**；没有做破坏性清理。

### 4.2 主评测协议

**Category-matched sampled candidates**：

```text
1 real observed positive target (next catalog item)
+ 99 sampled unobserved catalog items
negative sampling: same leaf category, then same main category, then random
not confirmed negative feedback
```

同时保留 **random 99 negatives** 作为附录协议；两个协议的数字不混用。

### 4.3 Neural test 指标

#### Category-matched，full test 77,029

| Metric | Memory OFF | Memory ON | Delta |
|---|---:|---:|---:|
| Hit@10 | 0.916278 | **0.918407** | +0.002129 |
| nDCG@10 | 0.770639 | **0.775925** | +0.005286 |
| MRR@10 | 0.727636 | **0.733822** | +0.006186 |

#### Category-matched，common 14,157

| Metric | Memory OFF | Memory ON | Delta |
|---|---:|---:|---:|
| Hit@10 | 0.905347 | **0.910221** | +0.004874 |
| nDCG@10 | 0.737407 | **0.743299** | +0.005892 |
| MRR@10 | 0.688000 | **0.694111** | +0.006111 |

#### Random candidates，full test（附录协议）

| Metric | Memory OFF | Memory ON | Delta |
|---|---:|---:|---:|
| Hit@10 | 0.924301 | 0.924483 | +0.000182 |
| nDCG@10 | 0.774119 | 0.776893 | +0.002774 |
| MRR@10 | 0.728904 | 0.732400 | +0.003496 |

**结论：** Memory ON 有一致但很小的 nDCG/MRR 增益；Hit@10 只有 +0.2~0.5pp，不能称为显著排序提升。

---

## 5. Top-K 排序效果与对比

### 5.1 Common 14,157，category-matched candidates

| Model | Hit@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Popularity | **0.950060** | **0.753286** | **0.694438** |
| LightGBM expanded OFF | 0.521368 | 0.266235 | 0.199477 |
| LightGBM expanded ON | 0.594335 | 0.295687 | 0.216107 |
| Neural OFF | 0.905347 | 0.737407 | 0.688000 |
| Neural ON | 0.910221 | 0.743299 | 0.694111 |

### 5.2 Common 14,157，random candidates（附录）

| Model | Hit@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Popularity | **0.964682** | **0.767024** | **0.706860** |
| LightGBM expanded OFF | 0.542841 | 0.275288 | 0.202583 |
| LightGBM expanded ON | 0.634386 | 0.310115 | 0.219466 |
| Neural OFF | 0.916790 | 0.742482 | 0.690183 |
| Neural ON | 0.917073 | 0.747319 | 0.696323 |

### 5.3 Full test 77,029（neural / popularity）

| Model | Hit@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Popularity (category-matched) | **0.956640** | **0.784271** | 0.732604 |
| Neural OFF | 0.916278 | 0.770639 | 0.727636 |
| Neural ON | 0.918407 | 0.775925 | **0.733822** |
| Popularity (random) | **0.968155** | **0.795024** | **0.742284** |

**关键负结果：Popularity 在 sampled candidate 协议上明显强于所有学习模型。**
这不是说 popularity 是完整推荐器；它反映 sampled negatives 中大量长尾商品 + positive target 本身偏热门，导致 global popularity 成为极强 prior。**不能把这里的 popularity Hit@10 当作 full-catalog recommender 指标。**

### 5.4 与现有 LightGBM v1 的关系

Phase 2 LightGBM v1 的原始协议是 catalog-only history + pointwise rating，不能直接塞进本轮的 1+99 candidate Top-K 表；其 Phase 2 参考数字：

```text
LightGBM v1 Memory OFF: MAE 0.855586, AUC 0.571768, pair 0.569103
LightGBM v1 Memory ON:  MAE 0.756594, AUC 0.749143, pair 0.586711
```

本轮额外训练的 **LightGBM expanded** 使用完整历史 + catalog metadata + rating pointwise，再在相同 candidate arrays 上按预测 rating 排序；结果如 5.1/5.2。它 pairwise 高于 neural，但 Top-K 明显差于 neural/popularity。因此不能混合两组数字。

**ItemCF / SASRec：本轮未运行。** 仓库中已有的 SASRec/Two-Tower 实现与当前 full-history + 76,918 catalog sampled-candidate 协议没有现成对齐，ItemCF 在 552,872 个全交互商品上的 item-item 矩阵也无法在剩余预算内做一个可信的、同协议的同口径基线。本轮不把未对齐的旧 SASRec/ItemCF 数字塞进表中；这是明确的未覆盖项，不是 PASS。

### 5.5 Pairwise Task

Common held-out user pairs：14,157 groups，两个真实已评分商品，共享同一 history cutoff。

| Metric | Neural OFF | Neural ON | LightGBM expanded OFF | LightGBM expanded ON |
|---|---:|---:|---:|---:|
| Pairwise ordering（所有 rating 不同 pair） | 0.509692 | **0.515964** | 0.592930 | 0.592740 |
| High-vs-Low pairwise | 0.511196 | **0.521433** | **0.625080** | 0.622521 |
| nDCG@2 graded | 0.941755 | **0.942734** | 0.952717 | 0.952692 |
| MRR high | 0.859500 | **0.861754** | **0.887067** | 0.886760 |

**Neural pairwise 接近随机（~0.51-0.52）**；Memory ON 比 OFF 高约 +0.6pp，但远低于 LightGBM expanded 的 ~0.59/0.62。
说明当前 neural ranker 主要学到的是全局 item popularity/popularity 特征，而不是稳定的用户偏好排序。

---

## 6. 不同历史长度 / Attribute Coverage 下的 Memory 作用

### 6.1 History length（category-matched，full test）

| History | n | OFF Hit@10 | ON Hit@10 | ΔHit | OFF nDCG | ON nDCG | ΔnDCG | ΔMRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24,439 | 0.932690 | 0.933753 | +0.001063 | 0.806362 | 0.814343 | +0.007981 | +0.009956 |
| 2 | 15,100 | 0.925298 | 0.925828 | +0.000530 | 0.793021 | 0.798158 | +0.005137 | +0.006530 |
| 3-4 | 15,529 | 0.916414 | 0.917767 | +0.001353 | 0.782734 | 0.784934 | +0.002200 | +0.002504 |
| 5-9 | 12,807 | 0.905755 | 0.908956 | +0.003201 | 0.750250 | 0.751547 | +0.001297 | +0.000653 |
| 10+ | 9,154 | 0.872078 | 0.879506 | **+0.007428** | 0.646359 | 0.655509 | **+0.009150** | **+0.009543** |

历史越长，Memory 增益总体越大；10+ 子集 Hit@10 +0.74pp、nDCG +0.92pp。但 10+ 的绝对 Hit 仍最低，说明长历史用户的 next-item 更难。

### 6.2 Target attribute coverage（category-matched，full test）

| Coverage | n | OFF nDCG | ON nDCG | ΔnDCG | ΔMRR |
|---|---:|---:|---:|---:|---:|
| 0.00-0.25 | 61,094 | 0.761882 | 0.763326 | +0.001444 | +0.001087 |
| 0.25-0.50 | 15,684 | 0.807593 | 0.825919 | **+0.018326** | **+0.023575** |
| 0.50-0.75 | 251 | 0.593264 | 0.718753 | **+0.125489** | **+0.160748** |
| 0.75-1.00 | 0 | — | — | — | — |

Attribute coverage 较高的目标上 Memory ON 收益更大，支持“属性匹配有独立贡献”，但高 coverage 样本非常少。

### 6.3 Attribute ablation

Remove candidate-attribute interaction block from Neural ON（保留 rating/time history）：

| Condition | Hit@10 | nDCG@10 | MRR@10 | Pairwise | High-vs-Low |
|---|---:|---:|---:|---:|---:|
| ON without attribute interaction | 0.917239 | 0.773006 | 0.730565 | 0.507412 | 0.504798 |
| ON with attribute interaction | **0.918407** | **0.775925** | **0.733822** | **0.515964** | **0.521433** |
| Attribute contribution | +0.001168 | +0.002919 | +0.003257 | +0.008552 | **+0.016635** |

结论：**属性匹配有独立、可测量的贡献，但幅度很小；主要仍来自历史 rating/time 和 item popularity。**

---

## 7. Event-Sourced Memory Shadow

### 7.1 P20 固定池 Shadow

使用 P20 19 个 recommend tasks，固定 Original BGE Top-20，provisional acceptable labels 不参与训练。

| Arm | Hit@1 | Hit@3 | Hit@5 |
|---|---:|---:|---:|
| Relevance only（Phase 1 A） | 14/19 | 19/19 | 19/19 |
| Existing Scorer Memory OFF（Phase 1 B） | 14/19 | 19/19 | 19/19 |
| Existing Scorer Memory ON（Phase 1 C） | **18/19** | 19/19 | 19/19 |
| Neural OFF | 15/19 | 18/19 | 19/19 |
| Neural ON | 13/19 | 17/19 | 19/19 |
| Neural ON + BGE blend（λ=1.0） | 14/19 | 18/19 | 19/19 |

- Neural ON 比 Neural OFF 更差（13 vs 15），说明 pseudo Event-Sourced history 对模型是 OOD 输入。
- Neural ON + BGE 只达到 relevance-only 的 14/19，**没有超过 Existing Scorer ON 的 18/19**。
- Top-1 变化：neural ON vs Existing Scorer ON 有 **15/19** 改变；blend 也有 **10/19** 改变，排序不稳定。
- Hard constraint violations：**0**（neural 只在共享 eligible 集合内排序；本轮没有替换 Gate，也没有执行 finalize）。
- 推理延迟：p50 **1.54 ms**，mean **18.06 ms**（包含第一次 CUDA cold-start，max 309.8ms），batch=1 / 20 candidates / GPU。

### 7.2 Event-Sourced Memory 适配问题

模型训练输入是 **rating history**；P20 使用的是 **explicit Event-Sourced preferences**（无 rating、无 item sequence）。Shadow 采用：

```text
positive preference -> synthetic item, rating=5
negative preference -> synthetic item, rating=1
```

这是一个 **伪历史兼容性探针**，不是真实用户 rating，也不是 Event-Sourced 训练。
因此不能声称模型“理解了 explicit memory”。实际结果也证明它没有：在 0051 上，Neural ON 的 eligible 排序为：

```text
B004K6R43G > B07BKPKXTT > B08D32JM5F
```

而 observed acceptable 是 `B08D32JM5F`，Triple 中排最后。
Counterfactual 修改（撤回 microphone、加入 negative microphone、加入 positive in_ear）会改变分数，但没有形成正确、稳定的语义排序。
结论：**迁移失败。**

---

## 8. 与 AgentRec-X 生产接口的差异

| 维度 | 生产链路 | 本轮 Neural Ranker |
|---|---|---|
| 用户信号 | Event-Sourced active preferences | inferred from ratings |
| Query | Original BGE Top-20 + query hard constraints | 没有 query 特征 |
| 历史 | 当前 Memory Reader | full rating history / 伪历史 |
| 目标 | candidate ranking / Gate finalize | next-item sampled ranking |
| 标签 | 无监督规则 / provisional labels | observed ratings |
| 负例 | 未观察商品不参与 | sampled unobserved candidates（训练/评测协议） |
| 输出 | `presentable[0]` | score / rating |
| 生产修改 | — | **无** |

- 本轮**没有修改** Planner、Scorer、Gate、CandidateLedger、Eligibility 或 P18。
- Shadow 只是在固定候选池上打分，没有接入 DecisionGate finalize。
- 由于缺少 query relevance feature，shadow 用固定 `z(neural) + z(BGE relevance)` 融合；这不是训练时存在的输入，不能称为完整模型。
- 由于 Event-Sourced memory 与 rating history 不可忠实互转，**不应把缺失特征填零后声称兼容**；本报告明确标记失败。

---

## 9. 是否适合接入 AgentRec-X

**结论：不适合接入当前生产 AgentRec-X。**

原因：

1. **P20 Shadow 低于现有 Scorer Memory ON（13/19 vs 18/19）**，且排序不稳定。
2. **无 query relevance 训练特征**；shadow 的 BGE blend 是手工后处理，不是模型能力。
3. **Event-Sourced Memory 与 rating history 输入不兼容**；伪历史分数响应不等于理解 explicit memory。
4. **Popularity baseline 在 sampled protocol 上更强**；当前模型没有证明个性化/记忆排序价值。
5. **Pairwise 接近随机**；pointwise/next-item gain 没有转化为用户偏好排序。
6. **非 Catalog history 无 metadata**；70% 历史事件是 UNKNOWN item，属性信号受限。
7. 模型规模/成本低不是问题，**质量与接口不满足生产门槛**。

---

## 10. 失败案例与下一轮明确改进方向

### 失败案例

1. **P20 `0051`**：Neural ON eligible 排序 `B004 > B07 > B08`；acceptable `B08` 最后。Event-Sourced memory 的 withdrawal/addition 只改变分数，没有产生正确排序。
2. **Pairwise**：Neural ON 0.515964，high-vs-low 0.521433；LightGBM expanded 分别 0.5927/0.6225，说明 neural 没学到稳定的 user-specific pairwise preference。
3. **Popularity dominance**：category-matched/random protocols 上 popularity Hit@10 0.950-0.968，高于所有 neural 模型；sampled negatives 太偏长尾。
4. **Non-catalog metadata missing**：test history 只有 24.4% 的 events 有 metadata；train 仅 6.1%。属性匹配信号严重不足。
5. **Attribute ablation** 只带来 +0.3pp nDCG/MRR、+1.7pp high-low pairwise；独立但很小。
6. **Rating auxiliary** ON 的 MAE 1.02，OFF 0.97；rating head 也没有成为可靠辅助任务。

### 下一轮明确改进

1. **Population-controlled / hard-negative protocol**：按 popularity 分桶、同 category、同 price 区间采样；避免 popularity 统治评测。
2. **Full-candidate / exposure evaluation**：收集真实曝光候选集；当前 sampled 1+99 不能代表完整 Top-K。
3. **Pairwise/listwise loss**：在真实同用户 observed pairs 上训练 LambdaRank / pairwise margin；当前 neural 只做 sampled softmax。
4. **Full metadata coverage**：获取非 Catalog Electronics metadata 或可靠 item content embedding；当前 UNKNOWN 比例过高。
5. **Event-Sourced memory encoder**：单独训练 explicit preference encoder，或构建 event-sourced preference → rating-history projection；不能继续用伪历史。
6. **Query relevance feature**：将 BGE rank/score 加入训练，而不是 shadow 手工 blend。
7. **更大用户/目标规模**：当前 77k users/134k targets 仍偏小；需要更多 ≥3 catalog 用户和真实曝光。
8. **不接生产**：在 P20 Shadow 至少达到 Existing Scorer ON 之前，Learned Ranker 只能作为研究模型。

---

## 11. 工程产物、测试与 Commit

### 11.1 代码

```text
experiments/neural_memory_ranking/__init__.py
experiments/neural_memory_ranking/build_sequences.py
experiments/neural_memory_ranking/build_hard_candidates.py
experiments/neural_memory_ranking/model.py
experiments/neural_memory_ranking/train_neural.py
experiments/neural_memory_ranking/eval_subsets.py
experiments/neural_memory_ranking/train_lightgbm_baseline.py
experiments/neural_memory_ranking/shadow_p20.py
tests/test_neural_memory_ranking_model.py
tests/test_neural_memory_ranking_data.py
docs/MEMORY_CONDITIONED_GPU_V1_REPORT.md
```

### 11.2 数据与模型产物（本地镜像；`data/`/`runs/` 不入 Git）

```text
data/neural_memory_v1/
  train.parquet / val.parquet / test.parquet
  item_features.npy, item_ids.json, item_feature_names.json
  memory_{train,val,test}.npy
  train_negatives[_hard].npy
  {val,test}_candidates[_hard].npy
  {val,test}_positive_position[_hard].npy
  pairs.parquet
  manifest.json
  hard_candidates_manifest.json

runs/neural_memory_v1/
  memory_off.pt / memory_on.pt
  memory_off_hard.pt / memory_on_hard.pt
  memory_on_hard_noattr.pt
  expanded_lightgbm_off.txt / expanded_lightgbm_on.txt
  comparison.json / comparison_hard.json
  comparison_hard_noattr.json
  lightgbm_comparison.json / lightgbm_comparison_hard.json
  neural_subsets.json
  shadow_p20.json
  remote_logs/
```

主实验模型：`memory_off_hard.pt` / `memory_on_hard.pt`（category-matched 主协议）。
Random protocol 模型保留在 `memory_off.pt` / `memory_on.pt`，避免协议混淆。

### 11.3 远端执行命令

```bash
cd /root/autodl-tmp/AgentRec-X-phase2
PY=/root/autodl-tmp/agentrecx-sft/bin/python

$PY -m experiments.neural_memory_ranking.build_sequences
$PY -m experiments.neural_memory_ranking.build_hard_candidates
$PY -m experiments.neural_memory_ranking.train_neural
CANDIDATE_SUFFIX=_hard RUN_SUFFIX=_hard $PY -m experiments.neural_memory_ranking.train_neural
CANDIDATE_SUFFIX=_hard RUN_SUFFIX=_hard ATTR_ABLATION=1 $PY -m experiments.neural_memory_ranking.train_neural
$PY -m experiments.neural_memory_ranking.train_lightgbm_baseline
CANDIDATE_SUFFIX=_hard RUN_SUFFIX=_hard $PY -m experiments.neural_memory_ranking.train_lightgbm_baseline
$PY -m experiments.neural_memory_ranking.eval_subsets
$PY -m experiments.neural_memory_ranking.shadow_p20
```

### 11.4 测试

```text
WSL local CPU:
tests/test_neural_memory_ranking_model.py
  3 passed

Windows local CPU:
tests/test_neural_memory_ranking_data.py
  5 passed
```

模型 forward/backward shape、memory OFF 不依赖 history、attribute ablation block、data split/history/candidate arrays 均已验证。

### 11.5 Git

本轮只创建本地 commit，不 Push。代码、测试与本报告在同一个本地提交中；最终 HEAD 以 `git log -1` 为准。

保护性检查：

```text
Production Scorer modified: no
Production Gate modified: no
Planner modified: no
Original BGE modified/rebuilt: no
P18 modified: no
Qwen / Qwen3-Reranker training: no
GPU used: yes (RTX 4090 D, allowed)
Large raw data re-downloaded: no
Git Push: no
```
