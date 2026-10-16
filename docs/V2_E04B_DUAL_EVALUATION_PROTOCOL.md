# AgentRec-X V2 / E-04B-P — 双层评价协议（冻结）与可执行性审计

> 状态：**冻结（按 E-04B-P 交付）**。本轮**未**加载任何神经网络权重、**未**启动 TIGER Worker、
> **未**联网/调用在线 LLM、**未**安装或下载、**未**修改 Serving 与生产推荐逻辑、**未**执行任何 Git 写操作。
> 机读版本：[`experiments/e04b_protocol_v2.py`](../experiments/e04b_protocol_v2.py)（本文档与它必须一致）；
> 校验测试：[`tests/test_v2_e04b_protocol.py`](../tests/test_v2_e04b_protocol.py)（12 项，全绿 ✓）。
> **本协议不包含任何未实际运行的性能数值** ✓。

---

## 1. 双层评价目标

| 层 | 对象 | 数据 | 标签 | 本轮能否给出真实指标 |
|---|---|---|---|---|
| **Layer A — Recommendation Quality** | 单源 / 固定多源 RRF 的下一物品预测 | 真实 Amazon Reviews 2023 Sports_and_Outdoors（412,445 用户 / 156,746 商品） | 真实（`temporal_leave_two_out` 的留出正例） | **能**（需 Gate B 复跑真实权重；本轮不加载） |
| **Layer B — Agentic Recommendation Quality** | 对话式请求下的工具选择、记忆遵循、追加检索、反馈身份与解释grounding | 受控合成任务（`evaluation/agent/cases.py` 登记册）+ 行为日志 + 回放录制 | **真实相关性标签缺失** ✗ → 只能做**行为正确性**判定 | **只能给行为指标**；relevance 指标一律 `NOT_COMPARABLE` ✗ |

**目标不变**（方向 C）：既要准确率，也要自主能力，并用**固定策略 vs Agent 策略的受控比较**证明增量价值 ——
但**增量价值必须在同预算、同 case、同身份空间下**度量（§5、§6）。

## 2. 四类测试场景与输入合同

机读定义：`SCENARIOS`（`experiments/e04b_protocol_v2.py`）。

| 场景 | 合法输入 | 需求文本来源 | 真实相关性标签 | 可用来源 | 任务成功判据 | 约束违反判据 |
|---|---|---|---|---|---|---|
| **A. 历史充分、无文本需求** | `train_history` / `validation_target` / `test_history` | **无**（协议原生就是序列） | ✓ 真实（留出 target） | SASRec / Two-Tower / TIGER / Similar-item | 目标进入 top-K（HR/NDCG） | 展示集合中出现 PAD / 越界 id / 已见未屏蔽项 |
| **B. 明确自然语言需求、历史较少** | `request_text` | **缺失**：仓内**没有**带相关性标签的真实用户请求 ✗ | ✗ | 词法（BM25，探索性）+ 规则策略（B2） | 只能判**行为**（是否澄清、是否越权、是否调用合法工具） | 非法工具 / 不在允许来源 / 无澄清的臆断 |
| **C. 历史 + 当前文本需求** | `test_history` + `request_text` | 同上，**缺失** ✗ | ✗ | 同上 | 行为 + **偏好遵循契约**（非增益） | 违反已声明硬约束 |
| **D. 多轮变化 / 候选不足 / 用户反馈** | `test_history` + 多轮 `request_text` + `presented_cards` + `user_feedback` | 同上，**缺失** ✗ | ✗ | 全部 + E-03B 追加 + E-03C 反馈 | 行为（追加合法且 ≤1 次；反馈身份绑定正确；撤销后状态正确） | 追加越权 / 预算越界 / 反馈指向未展示商品 |

**四类数据必须区分**（不得混称）：**真实行为数据**（A：真实交互序列）✓、
**受控合成任务**（D：`cases.py` 登记册；合成，已声明 ✓）、**人工标注任务**（B/C：**尚不存在**，见 §10）✗、
**真实用户反馈**（D 的反馈语句：目前仅有测试与受控语料，**不是**真实用户意图 ✗）。

🚫 **禁止**：从隐藏 `test_target` 的标题/描述/分类反向生成该 case 的用户请求，再用同一 target
作为独立的相关性证据（`LEAK_RULES` 第 2 条；`assert_query_is_label_free` 已把它变成**断言** ✓）。

## 3. 五个对照组及实现状态

机读定义：`ARMS`。**状态只允许四个取值**：`EXECUTABLE_OFFLINE` / `BLOCKED_NO_PROVIDER_AUTHORIZATION` /
`PARTIAL_REPLAY_ONLY` / `NOT_IMPLEMENTED_FOR_EVAL`（测试断言了取值集合 ✓）。

| 组 | 工具与能力 | 入口（file:line） | 预算 | 当前状态 |
|---|---|---|---|---|
| **B0** 固定适用单源 | 一次单源召回 | `recommendation/inference/sasrec.py:232`、`inference/twotower.py:70`、`backends/tiger_engine.py:90` | tool 1 / LLM 0 | **EXECUTABLE_OFFLINE** ✓（权重已在本地；本轮未加载） |
| **B1** 固定多源 RRF | 固定来源集合 + RRF | `recommendation/inference/fusion.py:226`、`experiments/benchmark_twotower.py:92`、`benchmark_genrec.py:104` | tool = \|集合\| / LLM 0 | **EXECUTABLE_OFFLINE** ✓ |
| **B2** 规则工具选择 | 确定性策略、约束过滤、空结果恢复 | `recommendation/evaluation/agent/runner.py:216-513`、`demo/agent_policy.py` | tool ≤2 / LLM 0 | **EXECUTABLE_OFFLINE** ✓ |
| **B3** LLM 自主工具选择 | 模型选择来源计划 | `recommendation/demo/tool_selection.py:214` | tool ≤2 / LLM 1 | **BLOCKED_NO_PROVIDER_AUTHORIZATION** ✗（E-04A 4/4 额度已用尽） |
| **B4** 完整 AgentRec-X | Memory Ranking + 有界 Replanning + Product RAG + 反馈 | `demo/runtime.py`、`demo/observation.py`、`demo/replan_policy.py`、`demo/explanation.py` | tool ≤2 / LLM ≤3 | **PARTIAL_REPLAY_ONLY** ◐（可回放 `runs/m3_execution/*.jsonl`，属**过去证据**，不是新测量 ✗） |

**七条硬要求**（逐条落到机制上）：① 每组工具/能力已列表 ✓；② 同场景同 case（`assert_same_case_ids` ✓）；
③ 模型选择 vs 预算增加分开报告（每组 `budget.llm_calls` 明确 ✓，且 §5 要求分列）；
④ 统一记录真实工具调用次数与成本 ✓（`evaluation/agent/metrics.py:64 tool_calls/unnecessary_tool_calls`）；
⑤ Replanning 能力边界**至多一次**（E-03A/E-03B 契约 ✓，报告中必须写出）；⑥ **RAG 解释质量 ≠ HR/NDCG** ✓（分属不同指标表）；
⑦ B0 的单源规则**评测前冻结**、且以**每源一组**报告，禁止"逐 case best-of" ✓（`selection_rule` 冻结文本 + 测试检查 `frozen` 字样）；
⑧ 不得用测试目标反推最优工具计划 ✓（`LEAK_RULES` 第 1/2/4 条）。

## 4. 主要指标及计算定义

机读定义：`METRICS`（含 `layer`/`source`/`labels`/`definition`/`code`）。`source = log` 表示可由执行日志确定 ✓；
`source = human` 表示**必须人工判断** ✗。

| 指标 | 层 | 来源 | 分母 / 定义 | 需要的标签 | 代码 |
|---|---|---|---|---|---|
| HR@5/10/20 | A | log | `1[rank<=k]` 的均值 | `test_target` | `evaluation/metrics.py:311` |
| NDCG@5/10/20 | A | log | `1/log2(rank+1)`（k 内）的均值 | `test_target` | `evaluation/metrics.py:317` |
| Candidate Coverage | A | log | 去重返回 id / 目录规模 | — | `evaluation/evaluator.py` |
| 平均工具调用数 / 多余调用 | B | log | 每 case 真实调用数 / 超出声明下限的次数 | — | `evaluation/agent/metrics.py:64` |
| 合法工具选择率 | B | log | 通过 `ActionValidator` 的提议 / 提议总数 | — | `evaluation/agent/metrics.py:160` |
| 约束违反率 / 强制检查率 | B | log | 展示集合中出现已验证违反的 case 占比 / 展示集合**确实被检查**的占比 | — | `evaluation/agent/metrics.py:79-100` |
| 偏好遵循率 | B | **human** | **今日不可计算** ✗（无真实"请求→偏好"标注） | 人工标注 | `demo/memory_ranking.py` |
| 任务成功率 | B | **human** | 部分可判（约束类可判 ✓；开放式请求 ✗） | 每 case 成功判据 | `evaluation/agent/cases.py:46` |
| RAG 可验证 claim 比率 / 弃答率 | B | log | 通过校验的 claim / 提议 claim；`insufficient_evidence` 占比 | — | `demo/explanation.py:228` |
| 反馈身份指代正确率 / 生效率 / 撤销正确率 | B | log | 解析后与预期卡片一致的比例等 | — | `demo/feedback.py:253` |
| Replanning 合法追加率 / 候选不足恢复率 / 候选增量 / 额外成本 | B | log | 分别为 Gate 允许后的合法追加占比、零候选→有候选的恢复占比、追加来源新增 grounded 数、附加 tool 数与耗时 | — | `demo/observation.py`、`demo/replan_policy.py` |

**标签门槛是可执行的**：`metric_is_computable(name, available_labels)` —— 标签缺失时**返回 False** ✓
（测试断言 `preference_follow_rate` 与 `task_success_rate` 在今天的标签集合下**不可计算** ✓）。

## 5. 防泄漏与标签隔离协议

`LEAK_RULES`（6 条，测试逐条检查关键词 ✓）：

1. 推荐请求**永不**携带 `test_target` / `validation_target`（Gate A 已在真实数据上运行时验证 ✓）；
2. 查询**永不**由隐藏 target 的标题/描述/分类派生 ✓（`assert_query_is_label_free` 断言 ✓；`QUERY_INPUT_FIELDS`
   只允许 `test_history` 与 `request_text` ✓）；
3. Memory **只能**来自用户自述文本与**已展示卡片** ✓（E-01B/E-03C 契约）——**禁止**用测试期未来交互构造 Memory
   再宣称 Memory 增益 ✗（E-04B-1 §5 已确认代码中不存在该路径 ✓）；
4. 标签缺失的任务**不得**产出相关性指标 ✓；
5. 回放录制是**过去证据**，必须标注为 replay ✓；
6. **不同工具预算**的结果不得标为同预算公平对比 ✓（`assert_same_budget` 断言 ✓）。

附加：`excluded_seen(history, target)` 会把正例自身移出掩码（重复交互 case 仍可评分 ✓，Gate A 已实测并断言 ✓）；
PAD=0 永不进入候选 ✓；非法 id 与越界在真实数据上为 0 ✓。

## 6. 工具预算与成本归一规则

* **同预算对比**：仅当每个 arm 的 `budget`（tool/LLM）完全一致时才可标 `same-budget` ✓（校验函数会拒绝 ✗）。
* **成本必须分列**：`llm_calls`（模型决策次数，含 Tool Selection / Replanning / RAG）与 `tool_calls`
  （真实检索调用次数）分开记录 ✓；另记录墙钟与峰值内存（Gate A 已给出测量方法 ✓）。
* **预算差异必须显式命名**：若被迫在不等预算下比较（如 B0=1 source，B4≤2 tools + ≤3 LLM），
  报告标题必须写明预算差异，并给出**同预算子集**（例如把 B1/B4 的 tool 预算都压到 1）✓。
* **Replanning 成本**：至多一次追加（E-03B ✓），其 tool 与 LLM 成本计入该 case 总数 ✓。
* **不得**把 RAG 的解释质量算作召回收益 ✓；也不得把"多调用了来源"带来的收益归因于"更聪明" ✓。

## 7. 当前真实数据可支持的结论

1. **Layer A**：真实数据 + 真实权重（本地齐备 ✓）可支持 `next-item` 的 HR/NDCG/Coverage，且
   各 arm 在**同一 case 集合、同一身份空间、同一 K、同一掩码**下可比 ✓（代码入口已存在 ✓）。
2. **Layer B（行为层）**：受控合成任务 + 真实执行链可支持**合法工具选择率、工具调用数/多余调用、
   约束违反率与强制检查率、RAG 校验/弃答、反馈身份正确率、Replanning 合法追加率与恢复率** ✓
   （全部来自日志 ✓，`evaluation/agent/metrics.py` 已有实现 ✓）。
3. **Gate A 的 BM25 结果**：**仅**为"真实词法检索功能可用"的证据（HR@20=0.02 为 exploratory ✗），
   **不得**当作 Agent 或任一模型的推荐准确率 ✗。

## 8. 暂时无法成立的质量结论（必须保留的限制）

1. **Layer B 的相关性/准确率** ✗ —— 无真实请求与相关性标注（B/C/D 三场景）；
2. **Memory 的真实增益** ✗ —— 离线队列无真实用户自述偏好；只能验证**排序契约**（同一候选池、
   不同偏好 ⇒ 顺序变化 ✓，E-02A/E-03C 已离线验证），不能给 HR/NDCG 提升 ✗；
3. **Agent 自主性的增量价值** ✗ —— B3/B4 需要真实 LLM（额度已用尽 ✗）；回放不是新测量 ✗；
4. **任务适配率 / 平均工具调用数的"合理性"** ◐ —— 可测（日志）但"适配"需要独立可审计标准；
   当前只有**下限声明**（cases.py）可判定"多余调用" ✓，开放式适配需标注 ✗；
5. **线上效果与生产稳定性** ✗（无 A/B、无真实流量）；
6. **跨口径比较** ✗（sampled-negative 与 full-catalogue 不可并列；单正例下 HR≡Recall 必须写明 ✓）。

## 9. 现有评测代码可复用清单

| 能力 | 位置 |
|---|---|
| 切分协议 / 候选 / 身份校验 | `recommendation/evaluation/split.py:46,150,168,202` |
| 指标（HR/NDCG/排序/掩码） | `recommendation/evaluation/metrics.py:215,243,270,311,317,179` |
| 批量评测驱动（小内存友好） | `recommendation/evaluation/evaluator.py`、`batched.py` |
| Agent 任务登记册与受控场景 | `recommendation/evaluation/agent/cases.py:46,551` |
| Agent 行为指标 | `recommendation/evaluation/agent/metrics.py:38,64,79-100,160` |
| 确定性策略库（B2 各变体） | `recommendation/evaluation/agent/runner.py:216-513,554` |
| 失败归因 | `recommendation/evaluation/agent/attribution.py` |
| 融合与基准入口（B0/B1） | `recommendation/inference/fusion.py:226`、`experiments/benchmark_twotower.py:92`、`benchmark_genrec.py:104` |
| 既有配对统计模板 | `runs/b1_evidence/b1_paired_statistics.json`、`runs/m2-paired-uncertainty.json`、`runs/m3_execution/m3_ndcg_statistics.json` |
| 已录制轨迹（B4 回放） | `runs/m3_execution/agent_behavior.jsonl`、`agent_rankings.jsonl` |
| 真实资产与身份映射 | E-04B-1 审计（`docs/V2_E04B_REAL_ASSET_AUDIT.md`） |

## 10. 需要人工标注或新增数据的最小需求

`ANNOTATION_MINIMUM`（机读 ✓）：单元 = `(request, user, judged relevant item ids)` 三元组；
覆盖场景 B/C/D；探索性规模 **100**、配对结论规模 **500**；**2 名标注者**并报告 Cohen's κ；
请求必须**人工撰写且不得看过留出 target**；**禁止**由 target 标题/描述/分类生成请求 ✗。
在获得该标注之前，B/C/D 的 relevance 指标保持 **NOT_COMPARABLE** ✓（不得为了让计划完整而伪造条件 ✗）。

## 11. Gate B 真实模型复跑应使用的固定配置

```text
协议        : agentrecx.eval_protocol.v1 / temporal_leave_two_out（split.py:46-47）
seed        : 20261015（冻结；case 选择规则：artifact 顺序，无需抽样）
case 集合   : 由 Gate A 的流式读取器产出（experiments/e04b2_gate_a.py 的 stream_sequences）
K           : 5 / 10 / 20
掩码        : excluded_seen(test_history, test_target)（正例可评分；重复交互已处理 ✓）
候选口径    : full-catalogue（PAD 排除、越界拒绝）
身份        : mappings.item2id ↔ 目录记录（三方一致 156,746 ✓）
arm         : B0 每源一组（SASRec / Two-Tower / TIGER；相似物品可选）、B1 固定集合 + RRF（常数不变）
资源        : CPU-only、单进程、分批（可用内存 <1.2 GB 中止）；TIGER 首启约 12 分钟（放最后）
输出        : runs/e04b2_gate_b/<arm>__<source>.json（新目录，禁止覆盖既有归档）
成本        : 每 case 记录 tool_calls / llm_calls(=0/1) / 秒 / 峰值 RSS
```

## 12. Agent 受控评价的后续执行顺序

1. **P0（可立即执行，无需授权）**：B2 在 `cases.py` 登记册上跑一遍，产出 §4 的**行为指标**（tool_calls、
   多余调用、约束违反/强制检查、RAG 校验与弃答、反馈身份正确率 ✓）；
2. **P1（需 Gate B 授权）**：B0/B1 真实权重复跑 → Layer A 的 HR/NDCG/Coverage 与成本 ✓；
3. **P2（需标注）**：完成 §10 的 100 条标注 → B/C 场景的**探索性** relevance 指标 ✓；
4. **P3（需新的 LLM 授权）**：B3/B4 真实 LLM 决策 → 与 B2/B1 的**同预算**对照 ✓（并给出 llm_calls 成本列 ✓）；
5. **P4**：配对统计（复用 `b1_paired_statistics` 模板）+ 质量—成本联合呈现 ✓；不足 500 条标注时**只报探索性** ✓。

## 13. 最终质量—成本比较方案

* **主轴**：x = 每 case 的 `tool_calls` 与 `llm_calls`（分开两条），y = HR@10 / NDCG@10（Layer A）或
  行为指标（Layer B）；同预算点用同一 case 集合配对 ✓；
* **三张必出的表**：① 同预算下质量对照（`assert_same_budget` 通过才可命名 ✓）；② 实际调用成本对照
  （tool/LLM/耗时/内存分列 ✓）；③ 质量—成本联合表（每 arm 一行，含预算差异标注 ✓）；
* **结论纪律**：任何"Agent 更好"的表述必须来自**同预算、同 case、同身份空间**的配对结果 ✓；
  预算不同者只能报告"以更多调用换取更多召回"，并同时给出额外成本 ✓；
* **不得**出现的图表：把 RAG 质量画成召回收益 ✗、把回放当新测量 ✗、把不同预算画成同一预算 ✗。

---

## 附 A：执行可行性矩阵（只读审计结论；证据见 E-04B-1 与本轮 `file:line`）

| 场景 | 真实数据可用 | 相关性标签 | 后端准备情况 | 评测入口 | 阻碍 |
|---|---|---|---|---|---|
| A 历史充分 | ✓ 412,445 用户 / 3,500,587 交互 | ✓ 真实留出正例 | SASRec ✓ / Two-Tower ✓ / TIGER ✓（有条件）/ Similar-item ✓ | `evaluation/evaluator.py`、`benchmark_twotower.py:92`、`benchmark_genrec.py:104` | 需 Gate B 授权加载权重；仅 7 GB 内存（串联执行） |
| B 文本需求 | ✗ 无合法请求数据 | ✗ | 词法 `control/catalog_search.py` ✓；B2 策略 ✓ | 行为：`evaluation/agent/*` | **标签缺口**（§10）→ relevance 指标 NOT_COMPARABLE |
| C 历史 + 文本 | ✗ 同上 | ✗ | 同上 + Memory 投影 ✓（E-01B） | 同上 | 同上 |
| D 多轮/反馈 | ◐ 行为可由受控任务与录制产生 | ✗（relevance） | E-03B ✓ / E-03C ✓（`tests/test_v2_product_feedback.py` 23 项 API 级验收 ✓） | `demo/runtime.py`（`llm_agent`）、回放 `runs/m3_execution/*` | B3/B4 需 LLM 授权；relevance 需标注 |
| 真实 LLM Provider | — | — | `provider_adapter.py`（`deepseek` profile ✓，凭据在 `.env` ✓ 未读取值） | `demo/tool_selection.py:214` 等 | **E-04A 4/4 额度已用尽** ⇒ B3/B4 标记 BLOCKED，需新授权 ✗ |

**不能用于构造 Memory 的数据** ✗：任何 `test_target`/`validation_target`/`future frontier` 信息、
隐藏标签派生的文本、其他 session 的展示记录 ✓（LEAK_RULES 1–3）。
**已有实现（不得推断未验收）**：E-03C 反馈 = 已有 API 级验收 ✓（`tests/test_v2_product_feedback.py`）；
E-03A/B 有离线执行型验收 ✓；Layer B 的 relevance 部分**从未**被独立验收 ✗（本协议据实标注）。

## 附 B：协议校验测试结果

```bash
.venv/bin/python -m pytest tests/test_v2_e04b_protocol.py -q      → 12 passed ✓
.venv/bin/python -m pytest tests/test_v2_gate_a_protocol.py -q   → 6 passed ✓（E-04B-2 既有，本轮未重跑 Gate A 质量部分）
```

覆盖：协议身份冻结 ✓、四场景输入/标签声明 ✓、五 arm 声明与状态取值 ✓、B0/B1/B2 的冻结选择规则 ✓、
同 case id 强制 ✓、同预算命名强制 ✓、标签门槛（人工标签缺失时不可计算）✓、指标须声明层/来源/定义/代码 ✓、
查询输入不得含隐藏标签 ✓、六条防泄漏规则关键词 ✓、标注最小需求完备 ✓、既有 case 登记册可驱动 B2 ✓。

## 附 C：本轮修改文件

| 文件 | 类型 | 说明 |
|---|---|---|
| `docs/V2_E04B_DUAL_EVALUATION_PROTOCOL.md`（新增） | 文档 | 本冻结协议（13 节 + 三附录） |
| `experiments/e04b_protocol_v2.py`（新增） | 协议配置 | 机读定义（场景/arm/指标/规则/校验函数/标注需求），无模型、无网络 |
| `tests/test_v2_e04b_protocol.py`（新增，12 项） | 轻量协议检查 | §7 要求的不变量断言 |
| 生产推荐逻辑 / Serving / 训练代码 | — | **未修改** ✓ |

## 附 D：HEAD、git status 与结论

```text
HEAD   : b3af9da79a87aada73cf389ee4c22df5f2169d54（v2/e04a-pass-checkpoint，未变 ✓）
status : 新增未跟踪文件（本轮 3 个 + E-04B-2 的 4 个），无 modified、无删除；
         未执行 add/commit/push/reset/restore/clean/stash/checkout ✓
```

**E-04B-P = PASS。** 依据 §九：双层对象/输入/标签/指标/基线/预算/统计方式均已明确且可审计 ✓；
真实数据"能支持 A 层真实指标与 B 层行为指标、不能支持 B/C/D 层相关性指标"的界限清楚 ✓；
Gate B 与 Agent 评测可按 §11/§12 的冻结配置继续执行 ✓。
**明确标注为不可执行/不可比较者**：B3/B4（`BLOCKED_NO_PROVIDER_AUTHORIZATION`）、
B/C/D 的 relevance 指标（`NOT_COMPARABLE`，见 §10 标注需求）、Memory 增益与线上效果（✗）。
**停止点**：不进入 Gate B、不加载任何模型权重 ✓。
