# P14-C0 — Real Qwen Base Planner & Rule Baseline（已补齐真实 Qwen 实验）

> 初始 P14-C0 因使用了两个过期的 SSH endpoint 得到 Connection refused，错误地判断实例未运行；**用户纠正后重新排查，确认实例一直在运行**。
> 实际可用的连接参数：`connect.cqa1.seetacloud.com:39793`（Windows key `agentrecx_autodl`；WSL/旧 config 里的 nmb1:32237、nmb1:22199、bjb3:25658 等均已失效）。此前 blocked 记录与报告保存在 `runs/p14c0_pre_qwen/`，本次真实结果与旧记录不混用。
> 本报告只覆盖 P14-C0；未训练/SFT、未修改 Scorer/Gate/Memory/标签、未运行 Reserved Test、未 commit。

## 0. 结论

真实 Qwen3.5-9B Base（本地 checkpoint `/root/autodl-tmp/models/Qwen3.5-9B-ms`，无 adapter）已通过既有 `/plan` HTTP 服务与 SSH tunnel 接入共享 Executor，并在新 Dev 20 上完成与 Rule Planner 的**同一冻结配置**对照：

| | Rule | Qwen3.5-9B Base |
| --- | ---: | ---: |
| 可推荐任务 FINALIZE | 15/15 | 15/15 |
| Top-1 命中已标注合格商品 | **6/15** | **8/15** |
| 非推荐任务错误 FINALIZE | 0/5 | 0/5 |
| 正确 ASK（clarification 任务） | 4/5 | 4/5（另 1 个 no_verified 任务误发 ASK 被 contract 拒绝） |
| 已知硬约束违规 | 0 | 0 |
| 解析失败 / 非法动作 | 0 | 0 解析失败；**1 个动作被 run_planner 拒绝**（不伪造成功） |
| 真实模型调用 | 0 | **67 次**（47,872 input tokens / 5,770 output tokens） |
| 平均/最大模型延迟 | — | **3,905.6 ms / 7,308.1 ms** |

**P14-C0 = GO**（真实、可复现对照已完成）；20 条 Dev 规模有限，Qwen +2/15 的差异不构成统计显著或泛化优势，仍需 P14-C1/后续数据验证。

## 1. 实例/服务恢复过程（根因纠正）

| 检查 | 结果 |
| --- | --- |
| 最初失败原因 | 使用了 `nmb1:22199`、`bjb3:25658`（config）及 WSL `nmb1:32237`（旧 config）→ TCP/SSH banner refused；这些端口是**过期连接参数**，不等于实例停止 |
| 实际可用参数 | 通过历史记录/known_hosts 扫描发现 `connect.cqa1.seetacloud.com:39793` TCP 可达且 SSH 登录成功（Windows key `agentrecx_autodl`） |
| 远程实例 | `hostname=autodl-container-5525458f7b-e2f18254`；GPU `NVIDIA GeForce RTX 4090 D 24564 MiB`（启动前仅 1 MiB 占用） |
| 模型 | `/root/autodl-tmp/models/Qwen3.5-9B-ms`（19G，config `model_type=qwen3_5`；server 启动日志 `adapter=None` ⇒ **Base，无 LoRA/SFT adapter**） |
| 服务 | 既有 `training/agent_sft/model_server.py`；启动命令：`python -m training.agent_sft.model_server --model /root/autodl-tmp/models/Qwen3.5-9B-ms --port 8009 --max-new-tokens 256`；日志 `proposal server on 127.0.0.1:8009 adapter=None` |
| 本地接入 | SSH tunnel `18009 -> 127.0.0.1:8009`；真实 health probe：`POST /plan` 200，结构化 `RETRIEVE`，latency 2693.7 ms，input 359 tokens |
| 资源 | GPU 显存 ~17.5 GB 用于模型；单请求生成 `do_sample=False, num_beams=1, max_new_tokens=256`；未停止/重启/释放实例 |

> 未自动启动/停止/暂停/释放 AutoDL 实例；仅在已运行实例内恢复了一个既有推理服务进程。

## 2. 共享 Agent Core 与 Qwen 接线

`experiments/p14c0/electronics_executor.py`：Rule 与 Qwen **只替换 planner 对象**，共享同一个 `ElectronicsPlannerExecutor`：M9/M5 legal history read、真实 BM25（P13-R）、Product Info、Preference Scorer V0、CandidateLedger、Eligibility、Decision Gate、预算全部一致。

Qwen 路径：既有 `HttpPlannerProposer`（`POST /plan`）+ `PlannerObservation` → `run_planner`：schema 验证 → action/参数校验 → Executor → Ledger → Eligibility → Gate → FINALIZE。非法动作被 `PlannerContractError` 拒绝并记录，不会静默变成成功。

冻结配置（`runs/p14c0/config.json`）：模型 checkpoint/服务命令/prompt sha256/预算（max_steps=8、top20、inspect5、gate evidence 1/retrieval 1）/解码参数；Rule 与 Qwen 相同。

## 3. Train Smoke（真实 Rule + 真实 Qwen）

同一 5 类代表任务、相同工具链：

| 任务 | 类型 | Rule | Qwen Base |
| --- | --- | --- | --- |
| p14e2_lite_0001 | no_history_recommend | finalized（Top-1 正确） | finalized（Top-1 错误） |
| p14e2_lite_0009 | history_influenced | finalized（正确） | finalized（正确） |
| p14e2_lite_0031 | preference_withdrawal | finalized（Top-1 错误） | finalized（Top-1 错误） |
| p14e2_lite_0035 | insufficient_information | awaiting_clarification | awaiting_clarification |
| p14e2_lite_0040 | no_verified_match | need_more_candidates | **error**：`PlannerContractError: ASK requires an ambiguity/need signal`（Qwen 在有候选时发 ASK，被 contract 拒绝） |

Train Smoke 证明：真实 Qwen 输出可以被解析并执行共享工具；真实 ASK/未解决 ASK 不被伪造；非法动作不会成为成功轨迹。

## 4. 新 Dev 20：逐任务对比（真实）

完整逐任务见 `runs/p14c0/comparison.json`、`rule_dev/trajectories/`、`qwen_dev/trajectories/`、`error_analysis.jsonl`。

分类（15 条 recommend）：

| 类别 | 数量 |
| --- | ---: |
| both right | 4 |
| **Qwen only right** | **4**（0043 no-history、0046 history、0053 pair、0054 update） |
| Rule only right | 2（0042 no-history、0049 history） |
| both wrong | 5（0041、0047、0050、0051、0055） |

同一任务上 Qwen 与 Rule 的选中商品大多不同；Qwen 的 4 条增量均可在轨迹中追溯到对应模型决策，非 evaluator 泄漏（Qwen 只看 query/历史/工具/Gate）。

非推荐 5 条：
- Rule：4 条正确 `awaiting_clarification`（conflict+3 insufficient），1 条 no_verified → `need_more_candidates`；
- Qwen：4 条正确 `awaiting_clarification`，1 条 no_verified → **`error`**（非法 ASK 被 contract 拒绝）。

## 5. 错误分析（逐条定位）

`error_analysis.jsonl` 分类：

| 错误类别 | Rule | Qwen |
| --- | ---: | ---: |
| correct | 6 | 8 |
| product_type_mismatch | **9** | **7** |
| no_finalize（非推荐正常终止/ASK） | 5 | 4 |
| planner_or_tool_error | 0 | 1 |

- **两家共同的绝对主因仍是商品类型/角色 grounding**：Rule 9 条错误全部是选中商品类型与任务池不符（keyboard→earbuds、headphones→earbuds、keyboard→lap desk/backdrop 等）；Qwen 7 条同类错误。Gate 只验证 query 显式硬约束属性，不约束 product type/role（P14-B1 已记录，本轮未修）。
- Qwen 独有的 1 条错误：在 `no_verified_match` 任务上模型选择 ASK（要求用户回答），但 observation 已有候选且未请求澄清 → `validate_action` 拒绝（记录 `planner_or_tool_error`）。这是真实 Qwen 的行为错误，未被静默接受。
- Qwen 相比 Rule 的 +2 条正确来自 4 条 qwen-only 对 2 条 rule-only；样本 15 条，不显著。
- 硬约束违规：两家均为 0；Gate 未误拦合法 FINALIZE（recommend 任务 false ask/refusal 均为 0）。
- 解析失败：0/67 次调用；模型输出均为合法 JSON action；1 次动作语义非法（如上）。
- 记忆：no-history 任务按设计无偏好；其他任务均有 1–3 条合法历史读取；未发现 evaluator 泄漏。

## 6. ASK 行为

- 4/5 非推荐任务（1 conflict + 3 insufficient）两家都正确 ASK，且 ASK 文本为真实可回答问题，未伪造回复；
- Qwen 在 no_verified 任务误 ASK 被 contract 拒绝（不是“伪造完成”）；Rule 在该任务走 `need_more_candidates`，暴露出 **Gate retry 终态映射**需要修正（Qwen 与 Rule 各有 1 条非推荐终止不理想）；
- 没有“未解决 ASK 后 FINALIZE”。

## 7. Gate 泛化观察

- 在 15 条 recommend 任务上：Rule/Qwen 的 false ask、false refusal 均为 0；Gate 没有阻止任何合法推荐；
- Gate 允许了 7–9 条 type-mismatch 的 FINALIZE，原因是 Gate 证据面仅覆盖请求的硬约束属性；这属于 **`gate_evidence_scope` / product-type role 缺口**，不是 `gate_blocked_valid_action`；
- 1 条 no_verified 的 Gate retry 后终态不理想（Rule: need_more_candidates；Qwen: 模型 ASK 错误），记录为 `gate_retry_terminal_mapping`，下一阶段修。

## 8. 资源与可复现性

| 项 | 值 |
| --- | --- |
| Dev Qwen 调用 | **67** 次（20 任务） |
| Tokens | 47,872 input / 5,770 output |
| 单次延迟 | avg **3,905.6 ms**，max **7,308.1 ms** |
| GPU | RTX 4090D 24GB；模型 ~17.5GB 显存；未训练 |
| 轨迹 | `runs/p14c0/qwen_dev/trajectories/`、`rule_dev/trajectories/`、`train_smoke/` |
| Manifest | `runs/p14c0/qwen_manifest.json`（status=EXECUTED，含模型/服务/端口/tunnel/统计） |
| 旧 blocked 结果 | `runs/p14c0_pre_qwen/`（preserved，不和本次混用） |
| 测试 | P14-C0 contract tests + P14-B1/E2 相关回归 **32 passed**（32.4 s） |
| Reserved Test | **未运行**；未修改原始标签/Scorer/Gate |

复现命令（服务已运行、tunnel 保持时）：

```bash
AGENTRECX_QWEN_PLANNER=1 AGENTRECX_QWEN_PLANNER_URL=http://127.0.0.1:18009 \
  .venv/bin/python -m experiments.p14c0.run_dev --planner both
# 或仅聚合已产生的 Qwen 轨迹并重跑 Rule：
.venv/bin/python -m experiments.p14c0.finalize
```

## 9. 报告问题回答

1. **Qwen Base 是否真正接入 Runtime？** **是**：真实 Qwen3.5-9B Base 通过既有 `/plan` 服务、共享 Executor/工具/Gate 完成 20 条 Dev 推理（67 次真实调用），非伪造。
2. **Rule 和 Qwen 哪个完成得更好？** 在 20 条新 Dev 上：Qwen Top-1 **8/15**，Rule **6/15**；两者都 15/15 可 FINALIZE、0 硬违规、0 非推荐错误 FINALIZE。Qwen 略好但样本小，不能宣称显著。
3. **Qwen 在哪些场景优于 Rule？** 0043（无历史）、0046（历史影响）、0053（同 Query 对照）、0054（偏好更新）这 4 条；均可在轨迹中追溯模型决策。
4. **Qwen 在哪些场景犯错？** 7 条商品类型/角色错误（与 Rule 同源）；1 条 no_verified 任务误发 ASK 被 contract 拒绝；Train Smoke 中 0001/0031 也为类型错误。
5. **ASK 是否合理？** Qwen 对 4/5 非推荐任务正确 ASK；no_verified 一次 ASK 不合法被拒；ASK 问题真实、无伪造回复。
6. **是否存在 Gate 误拦截？** recommend 任务 0 次误拦/误 ask；Gate 的主要问题是证据范围未覆盖 product type/role（导致错误 FINALIZE 被放行）；另有 1 条 retry 终态映射问题。
7. **主要瓶颈是检索、偏好评分、记忆还是 Planner？** 首要仍是**商品类型/角色 grounding + 检索候选质量**（Qwen 7/15、Rule 9/15 类型错误）；其次是 Gate retry 终态与 Qwen 的非法 ASK 行为；Memory 与偏好评分不是主要错误源。
8. **现有 Qwen Base 是否值得进入 SFT？** 初步值得作为 P14-C1 候选（Base 已略优于 Rule 且真实可运行），但在修复 product-type grounding、扩大 Dev 样本、完成至少一轮稳定性复跑前，不能下“SFT 会提升”的结论。
9. **下一阶段更应该优化推荐模型还是 Planner？** 优先 **推荐模型/角色 grounding（product-type 约束、候选质量）**；Planner 侧只需修复非法 ASK/retry 终态；P14-B2 推荐升级优先于 P14-C1 SFT。

## 10. 最终判定

| 判定 | 结果 | 说明 |
| --- | --- | --- |
| `QWEN_RUNTIME` | **PASS** | 真实 Qwen3.5-9B Base `/plan` 服务经 SSH tunnel 接入；67 次真实调用；无伪造 |
| `RULE_COMPARISON` | **PASS** | Rule 与 Qwen 同一冻结配置、同一工具链、同一 Dev 20；逐任务轨迹完整 |
| `EVALUATION_ISOLATION` | **PASS** | Qwen 只看 query/历史/工具/Gate；无 evaluator 泄漏；Reserved 未运行 |
| `LEGACY_REGRESSION` | **PASS** | 32 项定向测试通过；共享组件/Scorer/Gate/Memory/旧 Serving 未改 |
| **P14-C0** | **GO** | 真实、可复现对照完成；Qwen Base 已可作为 P14-C1 起点 |

> 结论仍是 pilot-dev / provisional：20 条 Dev、23 条 E2 pending、无人工核验；不宣称统计显著或泛化优势。下一步先做 **P14-C0-R 商品类型/角色修复**，再决定 P14-C1 SFT；本次不停止/重启/释放 AutoDL 实例，不 commit。
