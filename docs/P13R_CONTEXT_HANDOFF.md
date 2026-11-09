# AgentRec-X — P13-R 会话压缩 Handoff

> 用途：在进入下一轮架构/技术路线讨论前，用这一份文件替代冗长的历史对话。
> 生成日期：2026-11-08。所有结论来自真实运行；这是压缩状态，不是新结论。

## 0. 当前一句话状态

P5–P13R 已完成。最终可用的新数据底座是 **Electronics + C4 sampled metadata + BGE retrieval**；但个性化排序监督不足，**P13-R = LIMITED**，不训练新 Ranker、不进入 Agent 适配、不启动 SFT-v1。

## 1. 阶段结论

| 阶段 | 结论 |
|---|---|
| P11 Shopping Benchmark | **LIMITED**：164 tasks / 86 users；Level A=0，Level B=100，UNKNOWN=64 |
| P12 Review Evidence | **LIMITED**：801/801 train-history reviews 精确恢复；56 个 Level A heuristic candidate，0 confirmed |
| P13 Sports Retrieval | **LIMITED / 排序 NO-GO**：Sports 有效样本仅 159；固定 pool dev 7/18；history 无增量 |
| P13-R Electronics Rebase | **LIMITED**：Electronics 有效 1,095；检索可用；个性化/新 Ranker 仍 NO-GO |

## 2. P13-R 最终数据

选择 **Electronics**（Beauty 为候补）。

| Split | 原始 | 严格有效 | 清理后 |
|---|---:|---:|---:|
| train | 818 | 817 | **817** |
| dev | 135 | 135 | **123** |
| reserved test | 143 | 143 | **129** |
| 合计 | 1,096 | 1,095 | **1,069** |

- 清理规则：删除 dev/test 中 user 或 target 与更早 split 重叠的行；train 不变。
- 最终 user/qid/target 跨 split overlap = 0。
- **reserved test 未运行、未用于选型。**
- Query 是 Amazon-C4 从 target review 构造的 **proxy query**；原始搜索 Query 不可得。
- 1,069 个最终 Query 无 exact duplicate；只有 21 个 target ASIN 被多个用户共享，真实“同 Query 不同用户”对照不存在。
- target timestamp 来源：官方 timestamp benchmark CSV；严格有效 1,095/1,096。
- 所有 history `timestamp < target_timestamp`；target 商品若在 history 中则丢弃；target review 不作为历史特征。
- History 跨类别保留，但文本 metadata 只覆盖约 **6%**；其余只能 identity/category。

## 3. 新的 Electronics catalogue

- 来源：Amazon-C4 `sampled_item_metadata_1M` 的 Electronics subset。
- 大小：**76,918 items**。
- 身份：`item_id` / `parent_asin`。
- **不使用旧 Sports 5-core item2id。**
- target 覆盖：**1,095/1,096（99.9%）**。
- 诚实限制：这是 **C4 sampled Electronics catalogue**，不是完整 Electronics 全目录；正式训练前仍需评估是否补全。
- 未把跨类别 history 塞进旧 Sports mapping。

## 4. P13-R 真实检索结果

同一 Query、catalogue、observed positive、Top-100。

| Split | Source | Hit@10 | Hit@50 | Hit@100 | MRR@10 |
|---|---|---:|---:|---:|---:|
| train 817 | BM25 | 44 | 88 | 114 | 0.027 |
| train 817 | **BGE** | **147** | **266** | **352** | **0.086** |
| dev 123 | BM25 | 6 | 14 | 17 | 0.018 |
| dev 123 | **BGE** | **15** | **36** | **53** | **0.062** |

- BGE 独有 dev 命中 40；BM25 独有 4 → BGE 明显更好。
- 未跑 BLaIR / Qwen3-Embedding；先确认 BM25+BGE 数据质量。
- 固定 pool = BM25 Top50 + BGE Top50，dev target 可达 **40/123（32.5%）**。
- BGE 单路 Top100 可达 53/123；固定融合仍损失 13 个 BGE 可达目标。

## 5. 历史个性化诊断

固定 pool 上，固定规则（无训练）：

| Split | 配置 | Recall@10 | MRR@10 |
|---|---|---:|---:|
| dev | Query-only | 0.350 | **0.153** |
| dev | Query+History | 0.150 | 0.029 |
| train | Query-only | 0.368 | 0.158 |
| train | Query+History | 0.140 | 0.053 |

结论：固定历史规则使排序变差；只能说明排序变化，不能说明质量提升。没有可信历史监督。

## 6. 关键资产位置

### 数据

- `data/p11_benchmark/` — P11 tasks/schema/manifest/query_pools
- `data/p11_benchmark/tasks_p12_evidence.jsonl` — P12 review-enriched copy（原 tasks 未覆盖）
- `data/p12/review_keys.jsonl` — 801 train-history reviews keys
- `data/p13/sports/` — P13 Sports 数据（保留负结果）
- `data/p13r/electronics/` — **P13-R 主数据**
  - `tasks.jsonl`（1,069 clean）
  - `tasks_raw.jsonl`
  - `manifest.json`
  - `eval_tasks.jsonl`（train/dev，940）
  - `catalog_meta.jsonl`（76,918 catalogue）
- `runs/p12/reviews/reviews_matched.jsonl` — 801 matched reviews
- `runs/p12/evidence/` — Level A candidate audit
- `runs/p13r/retrieval/queries.jsonl`、`summary.json`
- `runs/p13r/diagnosis/summary.json`

### 代码

- P11：`experiments/p11_build_benchmark.py`、`experiments/p11_evaluate.py`、`experiments/p11_agent_exec.py`
- P12：`experiments/p12_match_reviews.py`、`p12_audit_evidence.py`、`p12_diagnose.py`、`p12_hybrid_diagnosis.py`
- P13：`experiments/p13/{prepare_data,target_timestamps,retrieval,retrieval_targets,retrieval_metrics,ranking}.py`
- P13-R：`experiments/p13r/{prepare_data,clean_split,retrieval,diagnose}.py`
- 测试：`tests/test_p11_benchmark.py`、`tests/test_p12_evidence.py`、`tests/test_p13_pipeline.py`、`tests/test_p13r_pipeline.py`

### 报告

- `P11_REPORT.md` / `P11_SOURCE_DIFF.txt`
- `P12_REPORT.md` / `P12_SOURCE_DIFF.txt`
- `P13_REPORT.md` / `P13_SOURCE_DIFF.txt`
- `P13R_REPORT.md` / `P13R_SOURCE_DIFF.txt`
- `P11_BENCHMARK_SCHEMA.md`

## 7. 关键 P11/P12/P13 指标（仅作历史上下文）

### P11

- 164 tasks / 86 users；Level A=0，B=100，UNKNOWN=64
- verified pool 156；pool cap 100
- BM25 156-task pool reach@50 = 140/156
- BGE = 102/156；Hybrid = 86/156
- History source = 17/164
- Rule Agent 10 tasks：strict success 2，catalog-valid 4，invalid finalize 5，hard violation 3

### P12

- 官方 Sports raw reviews 19,595,170 条；P11 train-history keys 801/801 精确匹配；0 ambiguous；rating mismatch 0
- 100/100 Level B claims 有 review；56 个 heuristic explicit-positive candidate；11 个 negative candidate；**0 confirmed Level A**
- 69 same-query pairs；44 对至少一侧有 candidate；仍无人工确认
- Hybrid 根因：fixed Top-K RRF 中 dense-only 候选挤占 BM25 候选槽位；bounded BM25 head 实验 86→87，无实质修复
- History cache 与 P8-D 输入 163/163 一致，但 History 是 query-independent；不能作为 query-aware 比较

### P13 Sports

- 462 raw rows；有效 159（121/18/20）
- BGE 最好；固定 fusion dev 7/18；LambdaMART 58 train/7 dev，弱监督，无增量
- P13 测试 4 passed

## 8. 不得再声称/不得再做的事

- 不得声称 Level A 偏好已被确认（P12 只有 unreviewed candidate）。
- 不得把 C4 proxy Query 当作真实搜索日志。
- 不得把单 observed positive + 大量 unjudged 候选称为真实点击/负反馈学习。
- 不得把 P13-R 的 catalogue 当作完整 Electronics 全目录。
- 不得把历史排序变化当作个性化收益。
- 不得把 P11 bounded pool 与 C4 observed-positive 混成同一分数。
- 不训练新 Ranker、不跑 Planner SFT-v1、不进入 Agent 适配（P13-R 停止条件）。
- 不修改默认 Serving V1、SASRec、Two-Tower、M5、Planner Contract、CandidateLedger、Role Gate、Eligibility、正式 Source Policy。
- 不运行 P6/P8/P9 冻结 Holdout。
- 不把已暴露的 P11/P12 Dev 或 P13/P13-R Dev 改名成冻结 Test。
- 不 commit/push/deploy。
- 不自动关闭/暂停/释放 GPU 实例。

## 9. 当前基础设施

- GPU：RTX 4090D 24GB，实例未关闭。
- 本机：Windows，仓库 `D:\IT\CODE\PROJECTION\agentrec\X`。
- 远端：AutoDL `/root/autodl-tmp/p10`、`/root/autodl-tmp/p13`、`/root/autodl-tmp/p13r`；BGE model/index、BLaIR model、C4 metadata、timestamp CSV 在远端。
- 本地 Python 实验环境：`C:\Users\zxt\AppData\Local\Temp\agentrecx_p3c_env`；LightGBM 4.7.0 已安装（P13 用过，P13-R 未训练 ranker）。
- Qwen Planner server 未启动；P13/P13-R 不需要。
- 未 commit；工作区有大量未提交实验产物。

## 10. 下一轮架构讨论前最重要的开放问题

1. 是否接受 **C4 sampled Electronics catalogue**，还是先补全 Electronics full metadata？
2. 是否转向 **JDsearch 等真实搜索日志**，因为 C4 proxy 查询没有 judged negatives / 多正例？
3. 人工相关性标签如何构造：多正例、judged negatives、query-level relevance？
4. 历史信号是否值得保留：当前文本覆盖 ~6%，固定历史规则无收益。
5. 下一阶段只做 Retrieval Backend，还是先做 label/eval 数据工程？
6. 是否保留 BGE retrieval 作为唯一可靠底座，暂不引入 BLaIR/Qwen？
7. P13-R reserved test（129）如何冻结与维护。

## 11. 推荐给下一轮架构/技术路线输入

一句话：

> 我们已把数据从 Sports/C4 的 159 个错配样本迁移到 Electronics/C4 的 1,095 个严格样本；BGE 检索明显优于 BM25，但固定 Top-100 只覆盖 40/123 dev 正例，历史无有效增量。下一步应先解决“真实相关性监督/多正例/judged negatives/完整 catalogue”的数据问题，而不是继续训练 Ranker。
