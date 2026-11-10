# P14-B3-Eval100 — 100 条新任务上的 V0 vs Qwen3-Reranker-4B 对照

> 范围：P14-B3-Eval100。新增 100 条此前未见任务（80 recommend + 20 non-recommend），用冻结 V0 与冻结 Qwen3-Reranker-4B + Preference Instruction 在同一固定候选池上真实对照。
> 未训练/微调，未改 V0/Gate/Memory/Planner/标签，未运行原 Reserved Test；所有人工核验为 pending；未 commit。
> **结论：`KEEP_V0`（provisional）。Qwen3-Reranker-4B 在这批更大规模的新任务上没有稳定超过 V0，且负偏好冲突更多。**

## 0. 数据规模与隔离

`data/p14b3_eval100/`：

| 类型 | 数量 |
| --- | ---: |
| 总任务 | **100** |
| recommend | **80** |
| 其中：no-history | 20 |
| 其中：history influenced | 24 |
| 其中：same-query persona pair | 20（10 对） |
| 其中：preference update | 8 |
| 其中：preference withdrawal | 8 |
| clarification required | 12（8 insufficient + 4 conflict） |
| no_verified_match | 4 |
| unverifiable_constraint | 4 |

- 新 ID：`p14b3_eval100_0001..0100`；新 persona 100 个；与 P14-E/P14-E2 任务、persona、query **零重叠**；
- 构建由真实 catalogue + 独立 evidence labeler 完成；没有用 BM25/V0/Qwen 排序反向选择正确答案；
- agent 文件不含 labels/effective/withdrawn/expected 等 evaluator 字段；
- 100 条均 `review_status=pending_human_review`，**无 human_verified**；
- 原 P14-E/E2 标签与 Reserved Test 未改/未运行；新数据不用于训练。

> 数据修正说明（透明记录）：首轮内存读取检查发现 3 条 `preference_withdrawal` 历史中“ergonomic shape/rechargeable/microphone”等描述无法被现有 Memory extractor 识别，导致对应撤回值仍可能出现在合法输入中。仅修正这些撤回事件的文本/取值（不改变任务数量、任务 ID 和标签口径），重建候选池并重新运行全部 Qwen3 打分；**没有根据任何模型成绩修改任务或标签**。最终结果均来自修正后的 100 条任务。

另输出：
- `runs/p14b3_eval100/validation.json`（schema/泄漏/隔离/计数检查全部通过）；
- `runs/p14b3_eval100/review_checklist_divergence.json`（27 条代表性待审案例，含负偏好、更新/撤回、模型分歧、不可达）；
- 初始全量 100 条审核清单 `data/p14b3_eval100/review_checklist_initial.json`。

## 1. 固定候选池

- 复用 P14-B2/P14-B3 的 BM25 Top-20 + P14-C0-R Product Type/Role 过滤；
- 100 任务共 **591 个合法候选**；2 个 recommend 任务角色过滤后为空；
- 所有已完成排序方法共享完全相同的候选集合、真实 title/text、tie-break 与 evaluator 标签（标签仅评估）；
- **7 个 recommend 任务的 verified 候选不在 Top-20**：0037、0045、0049、0050、0061、0062、0069。这 7 条是 **candidate retrieval unreachable**，单独报告，不归因于 Reranker；主指标分母为 **73 条 reachable recommend 任务**。
- 类型违规 0、Hard violation 0（所有方法）。

## 2. 真实模型与运行

Qwen3-Reranker-4B 冻结设置（与 P14-B3 一致）：

- `Qwen/Qwen3-Reranker-4B`，revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`，Apache-2.0；
- NF4 4-bit + fp16 compute，官方 yes/no token logit 打分，max_length 2048，batch 4；
- preference summary 来自合法 M9/M5 读取；撤回偏好不进入输入（新任务的撤回事件已用 Memory 可识别的描述重建）；
- 本次 100 任务、591 候选、三组共 **1,773 对**真实打分：加载 5.27 s，总打分 44.66 s，峰值额外显存 **3,606.2 MiB**；Qwen 服务持续运行，未停止/重载；
- V0 为 P14-B0 冻结确定性 Scorer（CPU，成本可忽略）。

## 3. 主结果（reachable recommend n=73）

| 方法 | Top-1 | Top-5 | MRR@10 | 偏好匹配/冲突/未知 Top-1 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 46/73 (63.0%) | 72/73 | 0.792 | 40 / 5 / 72 |
| **V0 Scorer** | **67/73 (91.8%)** | **73/73** | **0.953** | **61 / 2 / 53** |
| Qwen3 Query Only | 50/73 (68.5%) | 72/73 | 0.812 | 41 / 7 / 68 |
| Qwen3 + Preferences | 56/73 (76.7%) | 72/73 | 0.861 | 52 / 8 / 56 |
| Qwen3 + Preference Instruction | 58/73 (79.5%) | 72/73 | 0.879 | 55 / 5 / 56 |

全 80 条 recommend（含 7 条不可达）作为参考：V0 67/80，Qwen3 instruction 58/80；BM25 46/80。

**模型分歧**：
- Qwen3 instruction 相对 V0 **gained 3**（0024、0034、0067），**lost 12**（0023、0033、0035、0040、0043、0048、0052、0058、0063、0066、0068、0071）；
- Qwen3 的负偏好冲突 Top-1 为 **5**，V0 为 **2**；正向偏好匹配 55 vs 61；
- 结论：在这批新任务上 V0 明显更稳定，Qwen3 的指令增益不足以抵消更多损失。

### 分类表现（reachable）

| 类别 | 任务 | V0 Top-1 | Qwen3 Instruction Top-1 |
| --- | ---: | ---: | ---: |
| no-history | 20 | 20 | 20 |
| history influenced | 23 | 21 (91.3%) | 18 (78.3%) |
| same-query pair | 15 | 14 (93.3%) | 10 (66.7%) |
| preference update | 7 | 4 (57.1%) | 2 (28.6%) |
| preference withdrawal | 8 | 8 | 8 |

- 无历史任务两者相同（20/20），没有伪造偏好；
- 历史影响、same-query persona、偏好更新三类 Qwen3 均更差；
- 撤回任务两者都 8/8（Qwen3 未错误使用撤回值，但 V0 同样正确）。

### Same-query cross-persona

10 对同 Query 不同 Persona：V0 在 6 对中改变 Top-1，Qwen3 在 4 对中改变；V0 正确数 14/15，Qwen3 10/15。Qwen3 在少数 pair 上未能根据 persona 偏好区分。详细见 `comparison.json` 的 `same_query_pairs`。

## 4. 非推荐任务：Decision Gate 单独评价（不进入 Top-1 分母）

20 条非推荐（8 insufficient + 4 conflict + 4 no_verified + 4 unverifiable）由冻结 Rule executor + Product Type/Role + Decision Gate 运行（未使用 Qwen Planner）：

- **19/20 行为正确**；
- 失败 1 条：`p14b3_eval100_0087`（insufficient_information，预期 clarification，实际 `finalized`）；这是 Gate/Planner 行为问题，未修改 Gate；
- 其余 12 clarification 正确 ASK，4 no_verified 正确有界终止，4 unverifiable 正确保留 UNKNOWN/有界终止；
- 这些任务**不并入**排序 Top-1 指标。

## 5. 失败案例与人工审核

- 7 条候选不可达（0069/0037/0045/0049/0050/0061/0062）所有排序方法均无法命中；应走检索/候选覆盖修复，而不是 Reranker 调参。
- Qwen3 损失的 12 条中包含偏好更新、same-query pair 和 history 任务；典型是负偏好未被完全遵守、或把相似但偏好不符的商品排前。
- V0 仍有 2 条负偏好冲突和 6 条 Top-1 错；不是完美基线。
- `review_checklist_divergence.json` 共 **27 条** pending 案例，覆盖明确负偏好（如 in_ear/wired/gaming 负偏好）、更新/撤回、属性歧义、V0/Qwen 分歧；**没有真人结论，未宣称 human_verified**。

## 6. 成本

| | V0 | Qwen3-Reranker-4B |
| --- | --- | --- |
| 设备 | CPU | GPU（与 Qwen 服务共存） |
| 本批 591 候选 × 3 组 | ~毫秒级 CPU | 1,773 对，44.66 s + 5.27 s 加载 |
| 峰值额外显存 | 0 | 3,606.2 MiB |
| 质量（73 reachable） | 67/73 | 58/73 |

Qwen3 没有带来质量收益，且需要 GPU、量化、加载和推理成本；不采纳。

## 7. 明确回答

**Qwen3-Reranker-4B 在更大批全新模拟购物任务上，是否稳定优于 V0？**

**否。** 在 73 条候选可达的 recommend 任务上，V0 Top-1 67/73（91.8%），Qwen3 instruction 58/73（79.5%）；Qwen3 相对 V0 损失 12 条、仅增益 3 条；Top-5 两者相差 1 条，但 MRR V0 更高（0.953 vs 0.879）；Qwen3 的负偏好冲突更多（5 vs 2）。方向在多个类别上一致，不支持替换 V0。该结论为 **provisional**：标签由独立 labeler 生成，尚无人工核验。

**决策：`KEEP_V0`。**（`ADOPT_QWEN3_RERANKER`: 否；`INCONCLUSIVE`: 否——样本方向和模型分歧足够明确，但保留人工核验限定。）

## 8. 判定

| 判定 | 结果 | 说明 |
| --- | --- | --- |
| 数据规模/隔离 | **PASS** | 100 条新任务，80 recommend；schema/ID/query/persona 隔离校验通过；Evaluator 不泄漏 |
| 模型真实推理 | **PASS** | Qwen3-Reranker-4B 冻结设置真实完成 1,773 对打分；无伪造 |
| 公平对照 | **PASS** | BM25/V0/Qwen3 同 Query、同 role-filtered 候选、同标签、同 tie-break |
| V0 vs Qwen3 | **KEEP_V0** | 73 reachable：67 vs 58；losses 12 vs gains 3；负偏好冲突 2 vs 5 |
| 候选不可达 | 单独报告 | 7 条；不计入主指标分母 |
| 非推荐 Gate | 19/20 | 单独评价，不并入 Top-1；1 条 false finalize |
| 人工核验 | **pending** | 27 条代表性案例；无 human_verified |

## 9. 产物

- `P14B3_EVAL100_REPORT.md`、`P14B3_EVAL100_SOURCE_DIFF.txt`
- `data/p14b3_eval100/`（100 agent/evaluator、personas、history events、schema、manifest、review checklist）
- `runs/p14b3_eval100/`（pools、真实 Qwen3 scores、comparison、per_task、validation、nonrecommend_gate、review checklist divergence、decision、model manifest）
- `experiments/p14b3_eval100/{build,validate,build_pools,evaluate}.py`
- `tests/test_p14b3_eval100.py`：8 passed；与 P14-B3/P14-B2/P14-C0-R 定向回归合计 **36 passed**（65.7 s）
