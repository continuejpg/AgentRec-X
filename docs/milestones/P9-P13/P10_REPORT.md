# P10：LightGCN 与 BGE-Enhanced SASRec 实验报告

## 0. 结论

**两个模型都真实训练成功，但没有在 P8-D 300 用户 Dev 上带来可靠的质量提升。**

| 项 | 判断 |
|---|---|
| LightGCN 实现与训练 | **VERIFIED** |
| BGE-Enhanced SASRec 实现与训练 | **VERIFIED** |
| Enhanced SASRec 优于原 SASRec | **否**（Top-10 更低） |
| LightGCN 优于原 Two-Tower | **否**（全部主要指标更低） |
| 固定预算融合提升 | **NO** |
| 接入默认 Serving V1 | **不接入** |
| 进入下一阶段 Agent 策略学习 | **条件不充分** |

最终建议：**保留原 SASRec + Two-Tower，不启用本次两个实验模型。**

## 1. 新增文件及原因

| 文件 | 原因 |
|---|---|
| `recommendation/models/lightgcn.py` | LightGCN 核心模型：嵌入、归一化图传播、层平均、BPR |
| `recommendation/datasets/lightgcn.py` | 共用图归一化，training/eval 不各写一套 |
| `recommendation/models/bge_sasrec.py` | SASRec + 冻结 BGE 投影的增强表示 |
| `recommendation/datasets/bge_features.py` | 从现有 BGE 索引构建与 item ID 严格对齐的冻结特征矩阵 |
| `experiments/p10_train_lightgcn.py` | LightGCN 数据构图、训练、checkpoint |
| `experiments/p10_train_bge_sasrec.py` | Enhanced SASRec 训练入口 |
| `experiments/p10_eval_four.py` | 四模型同协议评测与固定预算融合 |
| `tests/test_p10_models.py` | 图泄漏、BGE 对齐、模型输出检查 |

修改的现有文件：

- `recommendation/models/sasrec.py`
  - 新增 `item_representations()` 单一商品表示入口；
  - 输入 embedding、training logits、full-catalog scores 统一使用该入口；
  - 原 base 模型仍返回原 item embedding，行为不变；84 个模型相关测试通过。
- `recommendation/models/__init__.py`
  - 导出两个实验模型。
- `recommendation/datasets/lightgcn.py` / `bge_features.py`
  - 共用图/特征构建，避免训练与评测偏移。

## 2. 复用情况

已复用：

- `recommendation.datasets.sasrec` 的 sequence/padding/negative sampling 数据契约；
- `recommendation.models.sasrec` 的 block、attention、全目录 scoring；
- `recommendation.training.sasrec` 的 loss/trainer；
- `recommendation.evaluation.metrics.evaluate_case` 与 P8-D 300 case；
- 现有 BGE-base P8-B index（不重新编码、不重新下载）；
- 现有原 SASRec / Two-Tower checkpoints；
- 原 item/user mapping 和 catalogue。

没有新增：

- 第二套 data loader；
- 第二套 item mapping；
- 第二套 evaluator；
- CandidateLedger / Source Fusion / Serving API；
- 模型注册中心、插件系统、复杂继承体系或兼容层。

唯一的 base-SASRec 修改是一个单一表示入口，且原有测试全部通过。

## 3. 是否重复功能/多余抽象

没有新增独立训练框架或第二套数据处理。

- LightGCN 图归一化统一在 `datasets/lightgcn.py`；
- BGE 特征对齐统一在 `datasets/bge_features.py`；
- Enhanced SASRec 直接继承 SASRec，仅覆盖 `item_representations()`；
- 四模型评测使用同一 batch/masking/metric 路径。

## 4. 真实训练结果

### LightGCN

GPU：RTX 4090D 24GB；torch 2.11 / CUDA 12.8。

| 项 | 值 |
|---|---|
| 用户 | 412,445 |
| 商品 | 156,746 |
| 训练图边 | 2,663,742（仅 `train_history`） |
| 配置 | 64 dim, 3 layers, BPR, batch 8192, lr 1e-3, wd 0, 10 epochs, seed 2026 |
| 训练时间 | 216.5s |
| BPR loss | 0.6696 → 0.1697 |
| checkpoint | `runs/p10/lightgcn/checkpoint.pt` |
| 训练日志 | `runs/p10/lightgcn/session.log` |

图构建：

- 只用训练历史构造 user-item 边；
- 验证/测试 target 不作为边；
- 负样本排除该用户已知训练正样本；
- 采用对称归一化邻接矩阵、层间平均；
- 无 attention、非线性或额外变体。

LightGCN 是 transductive 模型：用户必须存在于训练图中。评测 300 用户都来自该图，但没有为 unseen user 伪造 embedding。

### BGE-Enhanced SASRec

| 项 | 值 |
|---|---|
| 基础模型 | 原 SASRec checkpoint（不重训原模型） |
| 可训练内容 | ID embedding + BGE 投影 + scale |
| BGE | `BAAI/bge-base-en-v1.5`，768 维，冻结 |
| 训练 | 3 epochs, batch 256, lr 5e-4, seed 2026 |
| 数据 | 412,445 users, 2,263,252 train transitions |
| 训练时间 | 253.5s |
| loss | 0.6157 → 0.5695 |
| ranking accuracy | 0.9392 → 0.9483 |
| checkpoint | `runs/p10/bge_sasrec/checkpoint.pt` |

BGE vector 与 `item2id` 严格对齐；PAD row 为零；训练和全目录推理使用同一个
`item_representations()`。BGE 不参与梯度更新。

## 5. 同协议四模型评测

数据：P8-D 300 Dev cases；相同 history、validation target、mapping、seen masking；
指标由 `evaluate_case` 计算。

| Model | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | MRR@10 | Top-50 / Top-100 可达 |
|---|---:|---:|---:|---:|---:|---:|
| SASRec（原） | **0.0167** | 0.0333 | 0.0467 | **0.00884** | **0.00642** | 10 / 14 |
| BGE-Enhanced SASRec | 0.0067 | **0.0367** | **0.0533** | 0.00278 | 0.00159 | 11 / 16 |
| Two-Tower（原） | **0.0233** | **0.0467** | **0.0567** | **0.01285** | **0.00977** | 14 / 17 |
| LightGCN | 0.0133 | 0.0267 | 0.0400 | 0.00515 | 0.00270 | 8 / 12 |

### 解读

- **Enhanced SASRec 没有优于原 SASRec**：R@10、NDCG@10、MRR@10 均下降；只有 R@50/R@100 小幅增加。
- **LightGCN 没有优于原 Two-Tower**：所有主要指标更低；Top-50/100 可达也更低。
- 两者实现成功不等于质量提升。

### 独有 Top-10 命中

| Model | 独有的 Top-10 命中 |
|---|---:|
| SASRec | 0 |
| BGE-Enhanced SASRec | 0 |
| Two-Tower | 3 |
| LightGCN | 2 |

LightGCN 有 2 个独特命中，但整体指标下降；不能据此选择 LightGCN。

### Top-50 候选重叠

| Pair | mean Jaccard |
|---|---:|
| SASRec / Two-Tower | 0.122 |
| BGE-Enhanced / SASRec | 0.296 |
| BGE-Enhanced / Two-Tower | 0.126 |
| LightGCN / SASRec | 0.231 |
| LightGCN / Two-Tower | 0.137 |

新模型确实产生不同候选，但差异没有转成更好的 Top-10。

## 6. 固定总候选预算融合

使用同一 RRF、同一 seen masking，固定总候选预算：

| 配置 | 总预算 | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | Top-50/100 可达 |
|---|---:|---:|---:|---:|---:|---:|
| SASRec 50 + Two-Tower 50 | 100 | **0.0200** | 0.0433 | 0.0600 | **0.01497** | 13 / 18 |
| 四模型各 25 | 100 | 0.0167 | 0.0433 | 0.0600 | 0.00815 | 13 / 18 |
| SASRec 100 + Two-Tower 100 | 200 | **0.0200** | **0.0467** | 0.0667 | **0.01497** | 14 / 20 |
| 四模型各 50 | 200 | **0.0200** | 0.0400 | 0.0700 | 0.00916 | 12 / 21 |

结论：

- 100 预算下四模型融合不提升 Recall/NDCG，Top-10 反而下降；
- 200 预算下四模型 R@100 略高 0.0033，但 NDCG/R@10 明显更差，Top-50 可达更少；
- **没有证据表明固定预算融合带来质量提升**。

## 7. 代码质量与测试

- `py_compile` PASS；
- `tests/test_p10_models.py`、`tests/test_sasrec_model.py`、`tests/test_twotower.py`：**84 passed**；
- 覆盖：
  - LightGCN 图只含 train history、item row offset 正确；
  - BGE feature 与 item ID 对齐、PAD 为零、缺失映射拒绝；
  - Enhanced SASRec 表示/full-catalog score 形状；
  - 原 SASRec / Two-Tower 回归。
- 已删除失败尝试的远程输出目录和临时上传包；
- 保留 successful checkpoint、run record、train log 和评测结果。

## 8. 最终决策

### 8.1 是否采用新模型

**不采用。**

- Enhanced SASRec 未超过原 SASRec；
- LightGCN 未超过原 Two-Tower；
- 固定预算融合未改善 Top-10/NDCG；
- 没有真实质量增量，不能为了“训练成功”而替换来源。

### 8.2 是否做 History v2 opt-in

**不做。** 保持默认 Serving V1 与原模型不变。本轮没有修改 Serving、M5、Planner、
Source Policy、Role Gate、Eligibility、Gold/Evaluator。

### 8.3 是否具备进入下一阶段 Agent 策略学习条件

**不具备。**

当前候选质量仍未改善；新模型没有稳定提升 Recall/NDCG。下一步不应把这两个模型接入
Agent 或开始 SFT-v1，而应先解决内容融合/协同过滤的真实质量瓶颈，或回到更基础的模型
优化实验。

## 9. 交付物

- `P10_REPORT.md`
- `P10_SOURCE_DIFF.txt`
- 代码：
  - `recommendation/models/lightgcn.py`
  - `recommendation/models/bge_sasrec.py`
  - `recommendation/datasets/lightgcn.py`
  - `recommendation/datasets/bge_features.py`
  - `recommendation/models/sasrec.py`（单一表示入口）
  - `experiments/p10_train_lightgcn.py`
  - `experiments/p10_train_bge_sasrec.py`
  - `experiments/p10_eval_four.py`
  - `tests/test_p10_models.py`
- 训练配置/日志：
  - `runs/p10/lightgcn/{checkpoint.pt,run.json,session.log}`
  - `runs/p10/bge_sasrec/{checkpoint.pt,run.json,train.log}`
- 评测：
  - `runs/p10/eval/summary.json`
  - `runs/p10/eval/per_case.jsonl`

## 10. 未解决/限制

- Enhanced SASRec 是从原 checkpoint 固定 3 epoch fine-tune，不是从头训练；内容融合只验证了一个简洁方案。
- LightGCN 是 transductive，对训练图外的新用户没有可用 embedding。
- 300 Dev 命中数很小；本结论不等价于全量 412k 用户上的最终模型优劣。
- 没有运行 Holdout、没有运行 Agent 动态任务对照、没有训练 Planner SFT-v1。
- GPU 实例未关闭；为释放显存停止的是远程 Qwen Planner server，不是 GPU 实例。

本轮未 commit/push/deploy。
