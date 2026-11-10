# P14-D1 — Real-History Personalized Reranking Evaluation

> 范围：P14-D1。使用公开 **MemRerank / Amazon-C4 user-purchase-history** Electronics Dev 真实交互数据，独立观察正例，比较 Query Only 与 Real History；不训练、不微调、不改 V0/Gate/Planner/旧标签，不运行原 Reserved Test，不 commit。
> **数据是已暴露的 P13-R Electronics Dev（不是未见测试）**；所有人工核验为 pending。
> **结论：REAL_DATA_ALIGNMENT PASS，TEMPORAL_ISOLATION LIMITED，RERANKING_EVALUATION LIMITED，V0_REAL_HISTORY_COMPATIBILITY LIMITED，P14-D1 = LIMITED。真实历史没有提供一致可测的排序增益；模拟 benchmark 上的 V0 高分不能直接外推为真实个性化能力。**

## 0. 数据来源与下载

已实际下载（`data/p14d1/raw/`，见 `download_manifest.json`）：

- `zhiyuanpeng/amazon-c4-user-purchase-history` → `Electronics/dev.jsonl`（135 行）、`Electronics/train.jsonl`（818 行，下载但本报告未使用）、`amazon_c4_user_query_dict_o3-mini_rewritten_v2.json`、README；
- 复用 `McAuley-Lab/Amazon-C4` 的独立采样 catalogue（P13-R `catalog_meta.jsonl`，76,918 items）；
- 目标时间戳复用 P13-R 从 Amazon-C4 `valid/test` CSV 派生的 `target_timestamp_ms`；
- P13-R dev 是此前 P13/P14 已使用过的 Amazon-C4 查询：**exposed split**，不得称为未见测试。

## 1. 对齐统计（真实数据，不做强制拼接）

| 层 | 数量 |
| --- | ---: |
| 下载的 MemRerank Electronics dev 行 | **135** |
| Query dict 匹配 original / rewritten | **135 / 135** |
| P13-R dev 行匹配（user/qid/pos） | **135 / 135** |
| 目标 parent_asin 在独立 catalogue | **134 / 135** |
| 历史条目总数 | **14,945** |
| 历史条目在 catalogue（可取得商品文本） | **126 / 14,945 (0.84%)** |
| 历史中有任何 catalogue metadata 的任务 | **50 / 135** |
| 历史 verified_purchase | **5,847 / 14,945** |
| 目标出现在 BM25 top-20 | **10 / 135** |
| 目标出现在 BM25 top-100 | **20 / 135** |
| 目标出现在自身 history | **0** |
| history 时间 ≥ target 时间 | **0** |
| history 时间顺序违规 | **0** |

- original Amazon-C4 query 为主评测 query；rewritten query 单独记录、单独消融，不混合；
- 目标 ID 只存在于 `dev_targets.jsonl`，所有模型输入与候选池不含目标标记；
- 商品身份使用 parent_asin；target 1 条不在独立 C4 catalogue（无法自然/目标注入评分）；
- 历史商品事实极其稀疏：14,945 个历史条目只有 126 个能在独立 catalogue 找到文本；**不能把历史商品自动包装成显式偏好**。

## 2. 时间隔离

- target 时间来自 P13-R 对 Amazon-C4 valid/test 的派生时间戳；
- 删除 target 自身历史事件；删除时间 ≥ target 的事件；history 保持时间排序；
- 目标时间未在 released HF row 中独立重算，因此该判定为 **LIMITED**：数据构造与 P13-R manifest 一致，但不能声称独立验证了目标事件时间。

## 3. 协议 A：自然检索（BM25 top-20，target 不注入）

n=135（含全部不可达）；括号内为 target 在 Top-20 的 10 条条件结果：

| 方法 | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 2/135 (2/10) | 4/135 (4/10) | 5/135 (5/10) | 0.020 (0.269) |
| V0 weak proxy | 1/135 (1/10) | 4/135 (4/10) | 4/135 (4/10) | 0.014 (0.192) |
| Qwen3-Reranker Query Only | 6/135 (6/10) | 10/135 (10/10) | 10/135 (10/10) | 0.055 (0.745) |
| **Qwen3 + Real History** | **8/135 (8/10)** | 9/135 (9/10) | 10/135 (10/10) | **0.063 (0.844)** |
| Qwen3 Query Only rewritten | 2/135 (2/10) | 7/135 (7/10) | 8/135 (8/10) | 0.027 (0.368) |

**关键限制：自然检索中只有 10/135 的 target 在 Top-20，20/135 在 Top-100；** 大多数指标由不可达主导。条件子集 n=10 极小，Qwen+History 的 8/10 vs Query Only 6/10 不能当稳定结论；只是“在十个自然可达目标上历史和模型有少量信号”。

## 4. 协议 B：target-included 1-in-5 诊断（条件重排，不是端到端检索）

134 个 target 在 catalogue 的任务；target 作为 5 个候选之一，另外 4 个来自固定 BM25 top-100 的确定性随机 distractors（不使用 Scorer 选择，不把未观察商品称为真实负例）：

| 方法 | Hit@1 | Hit@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| BM25 | 9/134 (6.7%) | 134/134 | 0.269 |
| V0 weak proxy | 7/134 (5.2%) | 134/134 | 0.275 |
| **Qwen3 Query Only** | **108/134 (80.6%)** | 134/134 | **0.891** |
| Qwen3 + Real History | 101/134 (75.4%) | 134/134 | 0.861 |
| Qwen3 Query Only rewritten | 94/134 (70.1%) | 134/134 | 0.835 |

**同一模型、同一原始 Query、同一候选池：加入真实历史后 Hit@1 从 108 降到 101，MRR 从 0.891 降到 0.861。真实历史在本数据上没有带来增益，反而有小幅损伤。**
Qwen Query Only 的 80.6% 只说明“在目标已被包含的 5 选 1 中，通用相关性很强”，不能解释为端到端检索或个性化能力。

## 5. 历史增量消融

| 协议 | Query Only Hit@1 | +Real History Hit@1 | Query Only MRR | +History MRR |
| --- | ---: | ---: | ---: | ---: |
| Natural（n=135，含不可达） | 6/135 | 8/135 | 0.055 | 0.063 |
| Target-included（n=134） | 108/134 | 101/134 | 0.891 | 0.861 |

按历史长度（target-included）：

- 1–9 条：25→23（QO→History）
- 10–49 条：42→41
- 50+ 条：41→37
- 有 catalogue metadata：43→41
- 无 catalogue metadata：65→60

所有分组都没有正增益；历史越长、metadata 越多也未改善，说明当前“真实历史”文本大部分是类别/ID 且与 query 目标关系弱。**没有证据支持真实历史有稳定增量。**

## 6. Query 改写消融

- original query 为主：target-included Qwen Query Only 108/134；
- rewritten Query Only 94/134，明显更差；改写删除了规格/品牌/细节，不能与原始 Query 结果合并；
- 本报告所有主表都使用 original Query；rewritten 仅作为独立诊断。

## 7. V0 与真实历史的兼容性

V0 原始输入是显式正负偏好；真实数据只提供历史商品与交互，不提供“喜欢/讨厌某属性”。本轮构造的是**弱代理**：

- 只使用 pre-target 且能在 catalogue 找到的商品；
- 只取观察到的属性/形态作为 positive weak interests，不做 negative，不引用 target，不冒充用户明确表达；
- 仅 35/135 任务有可用弱画像，原始历史条目中 99.16% 没有 catalogue 文本。

结果：natural V0 proxy 1/135（BM25 2/135）；target-included V0 proxy 7/134（BM25 9/134，Qwen Query Only 108/134）。**V0 对真实历史没有可信适配；V0_REAL_HISTORY_COMPATIBILITY = LIMITED，不能把 V0 proxy 的数字当作 V0 的真实历史能力。**

## 8. 逐任务结果、盲评与人工核验

- 逐任务：`runs/p14d1/per_task.jsonl`（natural + target5，每方法 target rank / hit / mrr）；
- 盲评材料：`runs/p14d1/blind_review.json`，**25 条 pending_human_review**；包含 query（original/rewritten）、真实历史片段、匿名候选（A–E，无 target ID、无模型排名、无标准答案）；未有人类审核，因此没有人工评价数字；
- same-query repeated users：dev 中 **0** 组，无法做“同 Query 不同用户”区分度分析；
- target category 全部 Electronics（Dev 定义），无跨类目性能比较；
- verified_purchase 只存在于历史条目（5,847/14,945），target 本身无独立 verified 标记；因此不能按 target verified 分层。

## 9. 成本

- Qwen3-Reranker-4B 冻结：NF4 4-bit，revision `22e6836…`；
- Natural：8,100 pairs；加载 22.62 s；三组打分 336.21 s；峰值额外显存 5,326.2 MiB；
- Target5：2,010 pairs；加载 21.03 s；三组打分 87.87 s；峰值 4,256.3 MiB；
- V0 weak proxy 为 CPU 轻量打分；BM25 为现有索引；
- 未启动/停止/释放 AutoDL 实例，未训练/微调。

## 10. 报告问题回答

**原先 V0 的高分，到底有多少是模拟规则任务的特点？换成真实历史与独立观察目标后，历史偏好是否仍然具有可测量价值？**

- V0 在 P14-E/E2/Eval100 上的高分**主要来自与规则标签器同源的模拟偏好结构**（规则标签定义合格商品，规则 Scorer 更容易吻合）；这一点在真实独立监督下**没有得到验证**。
- 在真实 MemRerank Dev 上，历史输入在没有目标泄漏、严格时间截断的前提下**没有一致可测的增益**：
  * natural 全 135：History 8 vs Query Only 6（但仅 10 个 target 可达到 Top-20，条件样本极小）；
  * target-included 1-in-5：History 101 vs Query Only 108，MRR 也下降；
  * V0 weak proxy 明显低于 BM25/Qwen，且只有 35 个任务有可用弱画像。
- **因此不能声称“历史偏好已经提供可测量价值”**；更准确的结论是：**当前公开数据与现有独立 catalogue 的历史文本覆盖太低（0.84% 条目有 metadata），无法充分支持个性化历史增量评测**。这既不是模型胜负的强结论，也不是历史无用的证明，而是 **LIMITED / 数据条件不足**。

## 11. 最终判定

| 判定 | 结果 | 依据 |
| --- | --- | --- |
| `REAL_DATA_ALIGNMENT` | **PASS**（exposed） | 135/135 user/query/history/target 对齐；original/rewritten query 完整；target 134/135 在 catalogue；无目标历史/未来事件；但 dev 是此前 P13-R 已用过的 exposed split |
| `TEMPORAL_ISOLATION` | **LIMITED** | 目标自身从 history 删除、无 ≥target 事件、history 有序；target 时间来自 P13-R/Amazon-C4 派生值，未在 released HF row 独立重算 |
| `RERANKING_EVALUATION` | **LIMITED** | 自然 Top-20 可达仅 10/135、Top-100 仅 20/135；target-included 1-in-5 为条件诊断；历史 metadata 稀疏；结果只能作为受控诊断 |
| `V0_REAL_HISTORY_COMPATIBILITY` | **LIMITED** | V0 需要显式偏好；真实历史只能提供弱属性代理，覆盖 35/135 任务；proxy 明显低于 BM25/Qwen，不能制造 V0 真实历史成绩 |
| **P14-D1** | **LIMITED** | 真实数据下载/对齐/时间校验/双协议评测全部完成，但自然可达性与历史 metadata 不足，且无历史增量证据；不能升级为 GO |

## 12. 与模拟 Benchmark 的隔离

- P14-E/E2/Eval100：保留用于工具/记忆撤回/ASK/Product Type/Gate/规则偏好诊断，**不**与本报告成绩合并；
- P14-D1：只使用真实 MemRerank Dev + 独立 catalogue + observed positive，**不**再生成模拟 persona 或新模拟标签；
- 本报告 **不**把 target 当负例，不声称未观察商品是负例，不运行公开 Test 或原 Reserved Test。

## 13. 产物

- `P14D1_REAL_HISTORY_REPORT.md`、`P14D1_REAL_HISTORY_SOURCE_DIFF.txt`
- `data/p14d1/`：raw Dev/Train/query dict/README、`dev_tasks.jsonl`、`dev_targets.jsonl`、`alignment_manifest.json`、`download_manifest.json`
- `runs/p14d1/`：候选池、真实 Qwen3 分数、`comparison.json`、`per_task.jsonl`、`candidate_manifest.json`、`blind_review.json`（25 pending）、`model_manifest.json`
- `experiments/p14d1/{load_align,build_pools,evaluate}.py`；`tests/test_p14d1_real_history.py`：8 passed；与 P14-B3-Eval100/P14-B3/P14-C0-R 定向回归合计 **34 passed**（29.4 s）
