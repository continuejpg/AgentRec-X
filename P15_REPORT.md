# P15 — Evidence-Grounded Shopping RAG

> 范围：P15 only。复用现有 Agent Runtime、Search Backend、Memory、CandidateLedger、Product Type/Role、Eligibility、Decision Gate 与 Planner Executor；新增最小 Evidence Bundle / Claim Validation，并在 FINALIZE 路径接入真实 Qwen Base 解释。
> 不训练模型、不重建索引、不扩大 Benchmark、不运行 Reserved Test、不修改 V0/Gate/Memory 核心规则。

## 0. 结论

- **证据链路真实可用**：9 个 FINALIZE 任务全部生成 Evidence Bundle 与真实 Qwen Base 最终解释；4 个 ASK/无验证任务保持不推荐。
- **商品事实可追溯**：15/15 个有效 `product_fact` claims 都引用真实产品 Evidence ID；`missing_evidence_refs = 0`；每个 span 与原始字段精确一致。
- **Memory provenance 可追溯**：12/12 个有效 `preference_link` 引用 active preference 的 `memory_id/source_turn_id/source_text`；0 个 inactive/撤回偏好被当作 active；0 个跨用户引用。
- **UNKNOWN 未被冒充事实**：price/brand/stock/material/size/warranty 全部保持 UNKNOWN；9/9 条最终回答都有 catalogue limitation；0 个 UNKNOWN promoted-to-known。
- **Claim 验证是 automated structural/lexical validation，不是语义真理证明**；Qwen 原输出有 4 个 claims 被自动拒绝（2 个 UNKNOWN 命名错误、2 个 recommendation 带 fact refs），最终回答只使用通过验证的 claims。
- **P15 整体 LIMITED**：工程链路 GO 级可用，但语义真实性仍需人工审核；不能宣称完全没有语义幻觉。

## 1. 实现与复用

| 能力 | 复用/新增 |
| --- | --- |
| 商品检索 | 复用 P14-D2 BGE index 与 `CatalogSearchSource(mode='bge')` |
| Agent Runtime | 复用现有 `ElectronicsPlannerExecutor` + Rule/Planner Executor；未建第二套 Runtime |
| Memory | 复用 M5/M9 `HistoryStateReader` 的 active/inactive audit trail |
| FINALIZE | 在现有 executor 的 FINALIZE 分支增加**可选** RAG hook；默认 `None` 时行为不变 |
| 最终生成 | 真实 Qwen3.5-9B Base（无 Adapter）`/explain`；不是 SFT Adapter |
| 证据/Claim | 新增 `recommendation/control/rag_evidence.py`（最小结构，不新建 RAG framework） |
| Gate | Decision Gate 仍然是唯一 FINALIZE 权威；RAG 只在 `ALLOW_FINALIZE` 之后运行 |

E2E 实际检索模式为 **`bge_dense`**（所有 13 条轨迹一致）；不是“以为 BGE、实际 BM25”。检索 backend 记录与 BGE index identity 均由 P14-D2 复用。

## 2. Evidence Bundle

Schema：`runs/p15/schema/evidence_bundle.schema.json`。

每个最终候选包含：

- `namespace` / `product_id` / `product title` / `selected_product`；
- 原始 catalogue 字段：title、main_category、categories、features、description（以及可用的原始字段）；
- `product_evidence`：Evidence ID、source record、raw field、精确 span、verification status；
- `unknown_fields`：price、brand、stock、material、size、warranty（及缺失的 product）；
- CandidateLedger source、search mode、Product Type/Role、Eligibility、Gate decision。

用户偏好单独包含：

- Evidence ID、preference/memory ID、source_turn_id、source_text；
- active/inactive 状态、positive/negative/update/retraction 语义；
- 当前授权用户 key；inactive/removed/superseded 不能用于有效 active preference claim。

商品证据和偏好证据在 Bundle 中**物理分开**；商品字段只证明 catalogue 写了什么，不证明质量、库存、真实体验或用户满意度。

## 3. Claim-Evidence Validation

实现：`recommendation/control/rag_evidence.py::validate_claims`。

自动检查：

1. 引用商品存在，且属于 selected product；
2. 商品 namespace/ID 正确；
3. `product_fact` 引用字段属于该商品，span 与原始字段精确一致；
4. Memory 引用属于当前授权用户；
5. inactive/撤回偏好不得作为 active preference 证明；
6. UNKNOWN 不得被表达为已知事实；
7. 商品事实不得引用 memory evidence，偏好不得引用 product evidence；
8. 不存在的 Evidence ID 被拒绝；
9. 推荐判断不得用 fact refs 伪装商品事实（本实现选择拒绝这类 claim）。

明确标记：

```json
{"automated_validation": true, "semantic_truth_verified": false}
```

**这不是语义 entailment 证明**；自然语言 claim 是否真的被 span 支持，仍需要人工语义审核。

## 4. E2E 运行

13 条 Dev 任务（4 条非 FINALIZE，9 条 FINALIZE），覆盖：无历史、正/负偏好、preference update、withdrawal、ASK、无验证商品、硬约束、商品类型/角色。

| 指标 | 值 |
| --- | ---: |
| E2E tasks | 13 |
| FINALIZE tasks | 9 |
| 正常完成率 | **9/9 = 100%** |
| 非 FINALIZE | 4（2 ASK + 2 no verified） |
| 非 FINALIZE 被错误推荐 | **0** |
| BGE mode | `bge_dense`（全部） |
| 商品 Evidence / task | 5.33 |
| 偏好 Evidence / task | 2.22 |
| 有效 claims | 30 |
| 被自动拒绝 claims | 4 |
| valid product_fact | 15/15 |
| valid preference_link | 12/12 |
| valid unknown | 3 |
| missing evidence refs | **0** |
| inactive 偏好被当 active | **0** |
| product/memory confusion | **0** |
| UNKNOWN promoted-to-known | **0** |
| 最终回答含 UNKNOWN 说明 | **9/9** |
| Qwen `/explain` 有效输出 | 9/9 |
| Qwen 延迟 mean / min / max | **11.6 s / 8.1 s / 13.6 s** |
| Qwen input tokens mean | 3,348 |
| Qwen output tokens mean / max | 364 / 428 |

注：Qwen 输出上限从 512 调到 1024 是为完整 JSON；未反复调 Prompt/权重。最终回答中的 catalogue limitation 和 “Why I selected it” 由确定性 renderer 在 Qwen claims 基础上补齐，避免模型省略 UNKNOWN/推荐判断。

## 5. 具体 Evidence 示例

### 5.1 无历史正常推荐 — `p14e2_lite_0043`

Query：`I need a keyboard for working at my desk. It needs to be wireless...`
Selected：`B09828X5WY`（Gate `ALLOW_FINALIZE`，BGE mode）

Product evidence：

- `pev_01` / title：`Wireless Number Pads...`
- `pev_04` / attribute `wireless`：title span `Wireless`

Valid claims：

- product_fact：*“The product is a wireless numeric keyboard designed for Excel and accounting.”* → `pev_04`
- product_fact：*“The keyboard features 22 keys and is described as a numeric keyboard.”* → `pev_01`
- unknown：*“The price of this keyboard is unknown.”*

最终回答明确区分 catalogue facts、UNKNOWN、catalogue limitation；“Why I selected it”只引用 catalogue facts（无历史偏好），没有虚构用户偏好。

### 5.2 历史偏好 + 商品事实 — `p14e2_lite_0045`

Query：`Which headphones would work well for my commute? ...`
Selected：`B0BLSQ5T75`

Active memory evidence：

- `mev_050bd...`：*“I also prefer over-ear.”*（active positive）
- `mev_ffcae...`：*“I prefer noise cancelling.”*（active positive）

Valid claims：

- product_fact：*“The selected headphones are over-ear wireless models with built-in noise cancelling technology.”* → `pev_04`, `pev_05`
- preference_link：*“This product aligns with your active preference for over-ear headphones.”* → `mev_050bd...`
- preference_link：*“This product aligns with your active preference for noise cancelling features.”* → `mev_ffcae...`

最终回答把商品事实与偏好关联分节展示，并保留 price/brand/stock/material/size/warranty UNKNOWN。

### 5.3 偏好更新/撤回 — `p14e2_lite_0054`、`p14e2_lite_0055`

- 0054：inactive `miv_bbd...`（removed over-ear）存在，但最终回答只引用 active in-ear/wireless；没有把 removed over-ear 当作当前偏好。
- 0055：inactive `miv_24f...`（removed noise cancelling）存在，最终回答只引用 active wireless；0 个 inactive 被用作 active。

### 5.4 ASK 与无验证 — `p14e2_lite_0056`、`0057`、`0060`

- 0056（冲突偏好）：Gate `NEED_USER_CLARIFICATION`，最终回答询问 wired vs wireless 谁优先；**没有强行推荐**。
- 0057（信息不足）：Gate 要求澄清；最终回答询问用途/预算/功能，不推荐。
- 0060（无验证商品）：`no_verified_candidate_in_scope`；最终回答明确无法验证，没有编造商品或卖点。
- 0042 同样无验证，保持不推荐。

## 6. 失败/限制案例

- Qwen 原输出有 **4 个 claims 被自动拒绝**：2 个 UNKNOWN 命名错误（把已知属性标成 UNKNOWN），2 个 recommendation 携带 fact refs。它们被删除，未进入最终回答。
- 没有 valid `recommendation` claim（2/2 被拒绝）；最终回答使用确定性 “Why I selected it” 句，说明现有 Gate 选择与证据基础，不把它冒充商品事实。
- 0060 Gate reason 是 “search wider before concluding”，但本轮按停止条件未继续扩大检索；记录为 bounded no-verified，而不是伪造最终商品。
- **未做人工语义审核**；自动检查只能说明引用/span/用户范围结构正确，不能证明自然语言结论在语义上完全被证据支持。因此 `CLAIM_GROUNDING = LIMITED`，不能宣称“零语义幻觉”。

## 7. 判定

| 判定 | 结果 | 说明 |
| --- | --- | --- |
| `EVIDENCE_RETRIEVAL` | **PASS** | 真实商品字段 + 精确 span + UNKNOWN；所有 FINALIZE 均有 bundle；BGE mode 明确为 `bge_dense` |
| `CLAIM_GROUNDING` | **LIMITED** | 15/15 product fact 引用存在；0 missing refs；但仅有 automated structural/lexical validation，无人工语义 entailment |
| `MEMORY_PROVENANCE` | **PASS** | active/inactive 分离；12 个有效偏好引用；0 inactive 误用、0 商品/记忆混淆、0 跨用户 |
| `FINAL_ANSWER_INTEGRATION` | **PASS** | 9/9 FINALIZE 有真实 Qwen 解释；Gate 不被绕过；4 个 ASK/no-verified 不推荐 |
| `LEGACY_REGRESSION` | **PASS** | P15 + P14-C1/D2/C0-R/Eval100 定向回归 36 passed；旧默认行为未改 |
| **`P15`** | **LIMITED** | 工程链路可用；语义质量仍未建立，需人工审核后再考虑把解释质量当质量结论 |

**是否具备进入 P16 Skills 的条件？** **工程上可以（conditional GO）**：现有 Agent Runtime 已有稳定 Evidence Bundle、Claim-Validation、FINALIZE 集成和机器可读轨迹，可作为 Skills 的输入。**但不能把 P15 的 automated validation 当作语义质量通过**；P16 若需要面向用户承诺，必须先做人工/独立语义审核。

## 8. 成本与边界

- Qwen Base `/explain`：真实推理，9 次，mean 11.6 s，峰值输出 428 tokens；输入 mean 3,348 tokens（含 evidence bundle）。
- BGE retrieval 复用 P14-D2 index；未重建；未新增商品库/Search Backend/Memory Store/Planner。
- 未训练/微调；未运行 Reserved Test；未修改 V0、Gate、Memory 核心规则、旧标签；未 commit/push/deploy。
- 商品标题/描述均按不可信数据处理；system prompt 明确禁止执行嵌入指令；但这不构成对恶意 prompt injection 的完整语义安全证明。

## 9. 交付

- `P15_REPORT.md`、`P15_SOURCE_DIFF.txt`
- `runs/p15/schema/evidence_bundle.schema.json`、`runs/p15/schema/claim_validation.schema.json`
- `runs/p15/e2e/`：13 条完整轨迹（query、候选、Evidence Bundle、Qwen 原始输出、Claim 验证、最终回答）
- `runs/p15/e2e_trajectories.json`、`metrics.json`、`per_task.jsonl`
- `recommendation/control/rag_evidence.py`、`experiments/p15/{explainer,run_e2e,metrics}.py`
- `tests/test_p15_rag.py`：7 passed；定向回归合计 36 passed
- 修改文件：`experiments/p14c0/electronics_executor.py`（可选 RAG hook + 实际 score_kind 记录）、`training/agent_sft/model_server.py`（新增 `/explain` 端点）
