# P14-C1 — BGE Retrieval Integration & Qwen Planner SFT

> 范围：P14-C1。复用 P14-D2 的 76,263 件身份对齐 BGE 索引接入现有 Agent Search Backend；用合法 Train 轨迹做一次固定配置 QLoRA；在 Dev20 上比较 Rule / Qwen Base / Qwen SFT。
> 未扩展 Benchmark、未训练 Reranker、未改 V0/Gate/Memory/旧标签、未运行 Reserved Test；Dev 为反复使用的 exploratory set。

## 0. 结论

- **BGE 已真实接入现有 Search Backend**：76,263 个身份与 P13-R Electronics catalogue 完全一致；Rule/Base/SFT 全部使用同一 `mode='bge'` search source；BM25 默认和 fallback 保留。BGE 检索可达性来自 P14-D2：Original Query Recall@20 10→27、Recall@100 20→66。
- **Train-only Planner SFT 数据可用但有限**：40 Train 任务、134 个合法多步样本；动作 RETRIEVE 40 / RERANK 44 / FINALIZE 44 / ASK 6；10 个 Gate 恢复样本；无 evaluator 字段、无 target ID。
- **QLoRA 训练成功**：Qwen3.5-9B Base + NF4 4-bit，LoRA r16/alpha32，2 epochs，1536 max_seq，134 samples，训练 250.12 s，loss 0.0658（末段近 0），可训练参数 43.3M；Adapter 已保存并真实服务推理。
- **Dev20 BGE 对照（探索性）**：

| Planner | Top-1 | FINALIZE recommend | false ASK | false refusal | errors | avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule | 12/15 | 14/15 | 0 | 1 | 0 | — |
| **Qwen Base** | **15/15** | **15/15** | 0 | 0 | **1**（非推荐 0060 非法 ASK） | **3.1 s** |
| Qwen SFT | 12/15 | 14/15 | 0 | 1 | 0 | 8.5 s |

- **SFT 未超过 Base**：Top-1 12 vs 15；SFT 更慢（8.5 s vs 3.1 s，输出 tokens 10,891 vs 5,734）；失败任务 0042/0049/0051 与 Rule 行为高度相似，说明策略更像“模仿 Rule”，而不是学到更好的决策。
- **不采用 SFT；保留 Qwen Base Planner。** BGE 检索接入值得保留。

## 1. BGE 接入现有 Search Backend

实现位于 `experiments/p14c1/search_backend.py`，**不复制 Search Runtime**：

- 使用 `recommendation.control.catalog_search.CatalogSearchSource` 的既有 `bge` 模式；
- Dense index 直接复用 `runs/p14d2/bge_index.npz`（76,263 × 768，identity 与 catalogue exact equal；sha256 见 `runs/p14c1/bge_integration.json`）；
- 本地 BGE encoder：`models/bge-base-en-v1.5`，query instruction 与 P14-D2 一致；
- 命中 ID 返回 namespaced `amazon_electronics_c4:<parent_asin>`，与现有 `CatalogSearchSource` 的 records 过滤和 CandidateLedger 对齐；executor 继续使用既有 `Product Type/Role`、`Eligibility`、V0、Gate；
- `ElectronicsCatalogue` 的新进程默认仍为 **bm25**（Serving V1 不被破坏）；只有实验调用 `configure_catalogue` 切到 bge；
- BGE 不可用时 **显式 fallback 到 BM25**，不伪造 dense 命中；所有 Rule/Base/SFT run 的 summary 都记录 `mode=bge,status=active`；
- 集成测试：BGE 命中全部可在 catalogue 解析；fallback 测试返回 bm25 并记录原因。

## 2. SFT 数据

`runs/p14c1/sft_train.jsonl` / `sft_data_manifest.json`：

| 项 | 值 |
| --- | ---: |
| Train tasks | 40 |
| 有样本的任务 | 40 |
| 合法样本 | **134** |
| RETRIEVE / RERANK / FINALIZE / ASK | 40 / 44 / 44 / 6 |
| Gate 恢复样本 | 10 |
| Dev/Test 样本 | 0 |
| evaluator 字段在输入 | **否** |

- 输入只包含 PlannerObservation：current query、Memory/m5 状态、candidate IDs/evidence、allowed actions、last tool observation、remaining budget 等；
- 监督为 **实际被 executor 接受并通过 contract 校验的 PlannerAction**（Rule 轨迹）；不将 Rule 所有动作宣称为最优，只作为合法动作监督；
- 排除非法/未被接受动作，不复制轨迹制造样本量，不虚构 ASK 回复，不使用 target/evaluator 标签；
- 数据质量判定：**LIMITED**——134 条/40 Train 任务足够跑一次固定 SFT，但规模小且动作策略主要来自 Rule，SFT 容易模仿 Rule 的偏差。

## 3. QLoRA 训练

`runs/p14c1/training_manifest.json`、`runs/p14c1/sft_adapter/`：

- Base：`/root/autodl-tmp/models/Qwen3.5-9B-ms`；NF4 4-bit QLoRA；
- LoRA r=16, alpha=32, dropout=0.05, all-linear；2 epochs，max_seq 1536，batch1 × grad_accum8，lr 1e-4，warmup 8 steps，seed 20261109；
- 134 samples；训练 250.12 s；train loss 0.0658（末段接近 0）；trainable 43,278,336 / 8,997,081,600；峰值显存 18,785.6 MiB；
- Adapter 实际部署到 model_server 的 `--adapter` 服务，Dev 中 69 次生成 **0 非法动作**；
- 没有超参数搜索，未训练 Reranker/Search。

## 4. Dev20 公平对照

同一 BGE search source、同一 V0 Scorer、Product Type/Role、Eligibility、Gate、工具预算；每个任务独立 fresh Memory。

| 指标 | Rule | Qwen Base | Qwen SFT |
| --- | ---: | ---: | ---: |
| recommend Top-1 | 12/15 | **15/15** | 12/15 |
| recommend FINALIZE | 14/15 | **15/15** | 14/15 |
| false ASK | 0 | 0 | 0 |
| false refusal | 1 | 0 | 1 |
| false FINALIZE（非推荐） | 0/5 | 0/5 | 0/5 |
| hard / type violations | 0 / 0 | 0 / 0 | 0 / 0 |
| errors | 0 | 1（0060 非法 ASK） | 0 |
| model calls | 0 | 67 | 69 |
| tokens in/out | — | 46,439 / 5,734 | 50,231 / 10,891 |
| avg model latency | — | **3.1 s** | 8.5 s |
| tool calls | 88 | 83 | 88 |

- Base 在 BGE 候选上 15/15，全 FINALIZE；SFT 12/15，**0042 false refusal、0049/0051 与 Rule 相同的错误选择**；
- SFT 在非推荐的 0060 上没有 Base 的非法 ASK 错误，但这是**以 3 个推荐任务失败换来的**，整体无收益；
- SFT 输出明显更长（10,891 vs 5,734 tokens），平均延迟近 3 倍；对低成本部署不利；
- 结论：**BASE_VS_SFT = FAIL；保留 Base**。

## 5. 归因（不把 BGE 改善归功于 SFT）

- **BGE 召回改善**：P14-D2 已测得 Original Query target Recall@20 10→27 / Recall@100 20→66；本轮三个 planner 共享同一 BGE，候选池改善对三者都存在；
- **推荐评分错误**：Base 15/15 说明在 BGE 候选下 V0/Gate/推荐排序未成为瓶颈；SFT/Rule 的失败选择集中在同一类偏好排序案例，属于 Planner 选错候选或过早终止，而非 BGE 召回丢失；
- **Gate 阻断**：未出现错误 FINALIZE；SFT/Rule 各有 1 个 false refusal，Base 0；
- **Planner 动作错误**：Base 1 个非推荐非法 ASK（0060），SFT 0 个；SFT 的 3 个推荐任务失败是更重要的错误类别；
- **SFT 学习收益**：没有可归因的正增益；SFT 更像 Rule 轨迹模仿，且训练损失极低（可能过拟合），未改善 Base；
- 因此 **BGE 改善不能归功于 SFT**；SFT 也不应因 BGE 候选更好而采用。

## 6. 失败案例

- Base：`p14e2_lite_0060`（no_verified_match）发出非法 ASK，被 contract 拒绝；没有错误 FINALIZE。
- SFT：`0042` 有界拒绝（未 FINALIZE，Base 正确）、`0049`/`0051` 选错候选（与 Rule 相同，Base 正确）。
- SFT 生成动作全部合法（0 contract errors），说明 adapter 学会了输出格式；失败在决策质量，不是格式。
- Rule：`0042` 未 FINALIZE；其余与 SFT 相似。

## 7. 判定

| 判定 | 结果 | 说明 |
| --- | --- | --- |
| `BGE_INTEGRATION` | **PASS** | 76,263 identity exact；现有 CatalogSearchSource mode=bge；BM25 默认/fallback；Serving V1 未改；CandidateLedger/Role/Eligibility 正常 |
| `SFT_DATA_QUALITY` | **LIMITED** | 134 合法样本/40 Train；无标签泄漏；但规模小且主要是 Rule 行为模仿 |
| `SFT_TRAINING` | **PASS** | 真实 QLoRA 完成，Adapter 保存并可生成合法 action（Dev 0 errors） |
| `BASE_VS_SFT` | **FAIL** | Base 15/15 vs SFT 12/15；SFT 更慢、更长、无质量收益 |
| `LEGACY_REGRESSION` | **PASS** | P14-C1 + D2/D1R/D1/Eval100/B3/C0-R 定向回归 53 passed；旧 Serving 默认 BM25 |
| **P14-C1** | **LIMITED** | BGE 集成成功、SFT 流程成功，但 SFT 不优于 Base；保留 Base Planner |

**工程质量状态：** 工程实现与定向回归为 PASS；**质量状态仍为 NOT ESTABLISHED**（Dev 为反复使用的 exploratory set，无人工核验，不能把本次结果当作质量通过）。

**最终决定：** 保留 **BGE retrieval** 与 **Qwen Base Planner**；不部署 SFT adapter。若要做 Planner SFT，下一轮需要更大/更高质量的动作监督（超出 40 Train 任务）和独立验证，而不是在当前 134 样本上继续调参。

## 8. 产物与资源

- `P14C1_REPORT.md`、`P14C1_SOURCE_DIFF.txt`
- `runs/p14c1/`：BGE integration record、SFT data/manifest、training manifest/adapter、Rule/Base/SFT Dev trajectories/summaries、comparison/per_task、decision、model manifest
- `experiments/p14c1/{search_backend,build_sft_data,run_dev,compare}.py`
- `tests/test_p14c1.py`：6 passed；与 D2/D1R/D1/Eval100/B3/C0-R 定向回归合计 **53 passed**（96.7 s）
- 训练/评价服务：Base 服务在本轮启动后停止以加载 SFT adapter；SFT adapter 服务用于 Dev 评价；未对 AutoDL 实例做启动/停止/释放；未 commit。
- 未运行公开 Test 或项目 Reserved Test；Dev 是 exploratory；无人工核验。
