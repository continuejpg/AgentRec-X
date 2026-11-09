# P14-B2 — Preference-Conditioned Neural Reranker（真实预训练推理与公平对照）

> 范围：仅 P14-B2。真实运行一个预训练 Cross-Encoder Reranker，在固定、已通过 Product Type/Role 过滤的候选池内与 BM25、V0 Scorer 对照，并做 Query Only / Query+Active Preferences 消融。
> 未训练/SFT/LTR，未改 Memory、Scorer V0、Gate、Planner 或 Qwen；未运行 Reserved Test；P14-C0-R 结果未覆盖；未 commit。

## 0. 结论

真实预训练 **BAAI/bge-reranker-base**（XLM-RoBERTa base cross-encoder，278M 参数，MIT）已在现有 AutoDL RTX 4090D 上完成真实推理（与运行中的 Qwen 服务共存，未停止 Qwen 服务）。

**预训练神经 Reranker 没有超过 V0 Scorer；加入历史偏好有用但不足以改变采纳决策。** 新 Dev 20 的 recommend 任务（n=15，固定 role-filtered 池）：

| 方法 | Top-1 | Top-5 | MRR@10 | 偏好匹配/冲突/未知（Top-1） | Hard 违规 | 类型违规 | 不可达 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 6/15 | 14/15 | 0.611 | 5 / 2 / 17 | 0 | 0 | 1 |
| **V0 Scorer** | **11/15** | **14/15** | **0.833** | 9 / 1 / 14 | 0 | 0 | 1 |
| Neural Query Only | 8/15 | 13/15 | 0.678 | 5 / 4 / 16 | 0 | 0 | 1 |
| Neural Query + Prefs | 10/15 | 14/15 | 0.744 | 9 / 3 / 13 | 0 | 0 | 1 |

**V0 仍是当前默认推荐底座；不采纳神经 Reranker，也不在本次启动 LTR/微调。**

## 1. 模型与运行 Manifest

- 模型：`BAAI/bge-reranker-base`（`XLMRobertaForSequenceClassification`，278,044,417 参数）；
- 来源：Hugging Face，经 `hf-mirror.com` 下载到 `/root/autodl-tmp/models/bge-reranker-base`；License MIT；
- revision：`2cfc18c9415c912f9d8155881c133215df768a70`；`model.safetensors` sha256 `ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd`；
- 推理：fp16、CUDA、max_length=320、batch=8、Query–商品文本 pair；**一次性进程**，未建立长期服务；
- 资源：加载 3.76 s；Dev 20 共 111 候选 / 222 对，打分 0.842 s（均值 0.042 s/task）；峰值 GPU allocated **580.2 MiB**；运行前 GPU 已被 Qwen 占用 18,033 MiB、剩余 5,828 MiB，运行后剩 5,154 MiB；未影响 Qwen；
- 模型 Manifest、权重身份、来源、资源与输入配置：`runs/p14b2/model_manifest.json`。

## 2. 固定候选池与公平性

所有四种方法使用同一批、预先冻结的候选：

1. 相同 Current Query（agent-visible）；
2. 相同 BM25 Top-20；
3. 相同 **P14-C0-R Product Type/Role** 检查（只保留 MATCH/NOT_REQUIRED；类型不匹配不能因神经高分回流）；
4. 相同真实 catalogue title/text（截断至 450/320 token 内），无 evaluator-only 字段；
5. 相同最终 Top-K=20 和稳定 tie-break（BM25 rank → item_id）；
6. 相同评价标签（只在打分后用于评估）。

Dev 20：BM25 400 个命中中，role 过滤后保留 **111 个合法候选**（平均 5.55/任务，范围 1–10）；类型不匹配候选全部被拒绝，未出现类型违规（0/15 所有方法）。BM25/V0/Neural 的候选集合完全一致（per-task 校验通过）。

偏好输入来自 M9/M5 合法读取，固定格式：

```text
User preferences: positive preferences: wireless, ergonomic; negative preferences: gaming
```

- 只含当前有效偏好；**已撤回偏好不进入输入**（例：Train Smoke 0031 撤回 noise_cancelling，pool active prefs 仅剩 wireless）；
- 无历史任务不生成偏好文本，Query Only 与 Query+Prefs 的输入/分数完全一致（4/4 无历史任务排序未变）；
- evaluator-only persona、gold 商品、verified labels 均未进入模型输入。

## 3. Query Only vs Query+Active Preferences 消融

| | Query Only | Query + Prefs |
| --- | ---: | ---: |
| Top-1（全 15） | 8/15 | **10/15** |
| Top-5 | 13/15 | **14/15** |
| MRR@10 | 0.678 | **0.744** |
| 偏好匹配 Top-1 | 5 | 9 |
| 偏好冲突 Top-1 | 4 | 3 |
| 偏好未知 Top-1 | 16 | 13 |

- 11 个有历史 recommend 任务中，**8 个排序发生改变，4 个 Top-1 改变**；
- 有历史的 Top-1 正确数从 4（Query Only）升到 6（Query+Prefs）；
- 无历史 4 个任务完全不变（无伪造偏好）；
- 但加入偏好后仍低于 V0（10 vs 11 Top-1；MRR 0.744 vs 0.833），仅有助于 2 个原本 V0 错的任务（0047、0054）。

## 4. 哪些变好、哪些变差

**相对 V0 改善（Top-1）**：

- `0047`（history, mouse）：V0 选到 wired/gaming 鼠标；Neural Query+Prefs 选到 wireless/ergonomic 鼠标，Top-1 正确；
- `0054`（preference_update）：V0 选到与当前有效偏好不符的耳机；Neural（含 Preference 消融）Top-1 正确；
- `0041`、`0044` 无历史任务 Neural 与 V0 同为正例；`0050`、`0052` 等保持正确。

**相对 V0 变差（Top-1）**：

- `0045`（history, headphones, 正：over_ear/noise_cancelling；负：in_ear）：V0 正确；Neural Query Only 降到第 6，Query+Prefs 选到入耳式/骨传导，**忽略 in_ear 负偏好**；
- `0048`（history, speaker）：V0 正确；Neural 选到与 microphone/wireless 冲突的 speaker；
- `0053`（same-query persona pair, keyboard，负：wired）：V0 正确；Neural 选到**wired gaming keyboard**，负偏好被通用相关性覆盖；
- `0049`（history, earbuds）：Neural Query Only 变差；加入偏好后恢复正确。

**负偏好问题**：Neural Query Only Top-1 有 4 个任务违反负偏好，Query+Prefs 仍有 3 个；V0 仅 1 个。V0 的结构化 hard/soft 约束仍比通用 cross-encoder 稳定。

## 5. 失败案例与不可达分离

- **0046**（history, keyboard）：role 过滤后的池只有 **1 个候选且无 verified 候选**；四种方法都错，属于 **候选不可达/检索覆盖不足**，不是 Ranker 排序错误。应与 Ranker 错误分开报告（`candidate_unreachable=1`）。
- **0053**：V0 正确但神经模型错；这是“通用 query-title 相关性压过已恢复的 wired 负偏好”。
- **0045 / 0048**：偏好已合法恢复，但神经模型未将偏好文本用于稳定约束；属于“历史偏好利用不充分”，不是类型/Gate 问题。
- **0051**：所有方法都未命中（池内 verified 排第 2/4；候选存在但排序失败），是偏好排序失败而非类型错误。
- 所有方法：Hard constraint violation 0；Product Type violation 0；未产生 UNKNOWN 商品的“无依据推荐”（候选均来自真实 catalogue type/pool）。

## 6. Train 监督是否足以做轻量 LTR

`runs/p14b2/training_supervision.json`：

- 32 个 recommend Train 任务：positive `acceptable_id_count` **32,820**；
- 每任务 `hard_violated_pool_count` 均为 **0**，`near_miss_examples` 仅 **8/任务 = 288**；
- `hard_unknown_pool_count` **21,881**（未判断，不可当负例）；
- 40 个 Train 任务合计 hard-violated 1,714（来自非 recommend/约束冲突任务），但 recommend 的独立负监督仍很薄。

结论：**目前不足以支持有统计意义的神经 Reranker 微调/LTR**；若后续要做，应以 positive + hard-constraint/negative near-miss + 未判断中性池做分组 LTR，并在独立 Dev 上预注册；本轮不训练。

## 7. 运行成本与相对 V0

| | V0 | Neural |
| --- | --- | --- |
| 设备 | CPU | GPU（与 Qwen 共享） |
| 111 候选耗时 | **0.0013 s** | load 3.76 s + scoring 0.842 s |
| 每任务 | 可忽略 | ~0.042 s 打分 + 模型驻留 |
| 峰值额外显存 | 0 | **580.2 MiB** |
| 模型/服务 | 确定性代码 | 278M 参数、一次性进程 |

神经 Reranker 的额外成本不小，而质量没有超过 V0，**当前不值得替换默认 Scorer**。

## 8. 报告问题回答

1. **神经 Reranker 是否优于 V0？** 否。Top-1 10/15 vs 11/15，MRR 0.744 vs 0.833；Top-5 相同 14/15。结果可信但为负向。
2. **加入历史偏好是否有独立价值？** 有：Query+Prefs 相对 Query Only +2 Top-1、MRR +0.066、8/11 历史任务排序改变；但幅度不足以超过 V0，且负偏好仍会被忽略。
3. **哪类商品匹配问题得到改善？** 有偏好的鼠标（0047）和偏好更新（0054）Top-1 改善；无历史任务与 V0 同为正确。
4. **哪类问题反而变差？** 耳机 over-ear/负 in_ear（0045）、speaker（0048）、键盘负 wired（0053）；通用 cross-encoder 的 query-title 相关性压过结构化偏好。
5. **性能收益是否值得增加模型成本？** 否；V0 更好且几乎零成本。
6. **是否有必要进一步训练轻量 LTR 或微调 Reranker？** 当前不值得；独立负监督不足（near-miss 288、recommend hard-violated 0），先修检索不可达与偏好注入方式，若仍需要再预注册 LTR。
7. **当前推荐底座是否已经足够支持 P14-C1 Planner SFT？** 是（在不更换 V0 的前提下）：V0 + Product Type/Role Gate 仍是当前最优稳定底座；神经 Reranker 不采纳；0046 的候选不可达是已知检索缺口，不应阻塞 Planner SFT。P14-C1 可继续以 V0 为默认，Reranker 保持可选/禁用。

## 9. 验收与判定

| 判定 | 结果 | 依据 |
| --- | --- | --- |
| `NEURAL_RERANKER` | **LIMITED** | 真实预训练模型、真实推理、公平对照完成；但无质量收益、成本更高，不采纳 |
| `PREFERENCE_ABLATION` | **PASS** | Query Only 与 Query+Prefs 输入/分数真实不同；8/11 历史排序改变；无历史不变；撤回偏好排除 |
| `FAIR_COMPARISON` | **PASS** | 四种方法同 Query/池/类型检查/证据/标签/TopK/tie-break；标签仅评估；失败案例保留 |
| `RUNTIME_INTEGRATION` | **PASS** | 真实 one-shot GPU 推理；未建长期服务；与 Qwen 共存；资源记录完整 |
| `LEGACY_REGRESSION` | **PASS** | 定向测试 27 passed；Memory/Scorer V0/Gate/Planner/Serving/标签未改；P14-C0-R 保留 |
| **P14-B2** | **GO** | 实验可信、可复现、已决定不采纳 Neural；保留 V0 作为 P14-C1 默认推荐底座（GO 不等于模型超过 V0） |

> 说明：Dev 20 已参与 P14-C0-R 开发，本结果只是开发集诊断；未运行 Reserved Test；标签仍 provisional。结论不能外推为独立测试或生产收益。

## 10. 产物

- `P14B2_REPORT.md`、`P14B2_SOURCE_DIFF.txt`
- `runs/p14b2/pools/{train_smoke,dev}.jsonl`（agent-visible 固定池）
- `runs/p14b2/scores_{train_smoke,dev}.json`（真实 BGE 推理分数）
- `runs/p14b2/model_manifest.json`（模型/版本/哈希/许可/资源）
- `runs/p14b2/comparison.json`、`per_task.jsonl`、`training_supervision.json`
- `experiments/p14b2/{build_pools,evaluate}.py`；`tests/test_p14b2_reranker.py`
- 测试：P14-B2 + P14-C0-R + P14-C0 contract 定向回归 **27 passed**（32.9 s）
