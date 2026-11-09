# P2-Bridge 实现报告

## 0. 实现状态

**执行桥已实现并真实运行。**

链路：

```text
Qwen3.5-9B (Base/D1 adapter)
  → operation JSON
  → schema + evidence validator
  → ProposalActions
  → ExplicitPreference/Constraint/Retraction
  → M5 StateIntegration
  → effective state / M5 DB
  → R1 / M6.1
  → final Top-K
```

C/D0/D1 均已真实运行 20 个 dev 任务。没有使用 Gold Proposal、没有用规则输出冒充模型输出。

---

## 1. 代码修改

| 文件 | 修改 |
|---|---|
| `recommendation/demo/proposal_bridge.py` | 新增 operation→M5 映射、evidence/schema 验证、role/goal 映射、诊断 |
| `recommendation/demo/agent_service.py` | `run_shopping_turn(..., proposal_operations=...)`；proposal 作为本轮唯一 M5 输入；goal/role target 更新；诊断读取 |
| `benchmarks/agent/adapters.py` | `LLMPreferenceAdapter` 将真实 proposal payload 传给 service；不再只写日志；注入器支持 proposer 对象 |
| `benchmarks/agent/runner.py` | 支持外部真实模型 proposer；C/D 使用相同 runner 配置 |
| `training/agent_sft/model_server.py` | 真实 Qwen3.5 Base/D1 推理 HTTP server |
| `training/agent_sft/http_proposer.py` | 本地 proposer 客户端；无效 JSON 保留为显式 invalid proposal |
| `training/agent_sft/bridge_metrics.py` | 真实逐任务 V2/工具/token/延迟统计 |
| `tests/test_proposal_bridge.py` | proposal→M5 单元测试 |

关键行为：

- D0/D1 有 proposal 时**关闭规则提取器的同轮 M5 写入**，避免双写；
- invalid proposal 不写 M5，记录 schema/evidence 错误；
- evidence 必须是当前用户原话子串；
- 同 turn identity 由 M5 event identity 保持幂等；
- M5 事件通过既有 `StateIntegration`，未修改数据库内部表。

---

## 2. 单元/回归测试

```text
pytest -q tests/test_proposal_bridge.py tests/test_p2sft_contract.py
14 passed
```

覆盖：

- 有效 budget proposal 改变 M5 effective state；
- 相同事件重复 apply 不重复；
- invalid evidence 拒绝；
- 30→10→50 只保留最新；
- retract 后旧 superseded 约束不复活；
- noop/clarify 不产生状态更新。

---

## 3. Smoke Test（1 个多轮任务）

任务：`dev_06_cookware_material`（3 turns：set_goal → add_preference titanium → retract_preference）。

D1 真实轨迹：

```text
turn 0: set_goal(product_type=cookware), valid
turn 1: add_preference(material=titanium), valid
turn 2: retract_preference(material), valid
```

每条 turn 的 `memory_events.proposal_bridge` 均包含：

```text
valid=true
accepted_operations=[...]
source="llm_proposal"
```

M5 DB 在 smoke 目录生成，R1/M6.1 使用更新后的状态返回最终商品。

轨迹：

`training/agent_sft/eval/P2_BRIDGE_SMOKE_TRAJECTORY.jsonl`

---

## 4. 正式 20 dev 任务结果

同 C/D0/D1 使用：

- 相同 dev tasks、Prompt/schema、R1/M6.1、catalog、Top-K；
- D0/D1 相同 decode：greedy、max_new_tokens=256、非 thinking；
- C 使用同一 `apply_constraints` 配置重新运行；
- 未修改 Evaluator V2 或 Gold。

| 指标 | C | D0 Base | D1 SFT |
|---|---:|---:|---:|
| Evaluator V2 PASS | **12** | 12 | 12 |
| FAIL | 5 | 5 | 5 |
| UNKNOWN | 3 | 3 | 3 |
| legacy pass | 8 | 8 | 7 |
| V2 task-state PASS | 17/20 | 17/20 | 17/20 |
| Preference lifecycle PASS | 17/20 | 17/20 | 17/20 |
| Hard constraint PASS | 18/20 | 18/20 | 18/20 |
| Product role PASS | 17/20 | 17/20 | 17/20 |
| Retrieval ADEQUATE | 17/20 | 17/20 | 17/20 |
| Proposal valid turns | — | **1 / 61** | **52 / 64** |
| Accepted operations | — | 1 | 52 |
| RETRIEVE / RERANK | 32 / 18 | 32 / 17 | 34 / 18 |
| p50 / p95 wall ms | 9173 / 20167 | 12139 / 19755 | 17823 / 33086 |
| LLM tokens | 0 | 25,973 | 25,760 |

逐任务：

`training/agent_sft/eval/P2_BRIDGE_AGENT_COMPARISON.csv`
`training/agent_sft/eval/P2_BRIDGE_AGENT_SUMMARY.json`

---

## 5. 关键发现

1. **D0 Base 的 proposal 绝大多数是无效 JSON**：61 个 proposal 轮次只有 1 个通过 schema，60 个被拒绝；因此 D0 的 M5 基本没有模型写入。没有用规则 fallback 掩盖。
2. **D1 SFT 能真实写入 M5**：64 轮中 52 轮有效，52 个 operation 被接受；invalid 12 轮被拒绝，未写状态。
3. **D1 没有提高 V2 成功率**：D1 12/5/3，与 D0、C 完全相同；legacy C 8、D0 8、D1 7；D1 没有超过 C/D0。
4. **M5 生命周期仍为 17/20**，失败集中在 dev_06/dev_07/dev_14 等偏好生命周期任务，以及 dev_05/dev_17 budget 场景的 UNKNOWN。
5. **D1 延迟/Token 更高**：p50 从 12.1s 到 17.8s，p95 从 19.8s 到 33.1s；Token 基本相同（25.97k vs 25.76k）。
6. **瓶颈已不是“proposal 只写日志”**：桥已打通；当前瓶颈在 D0 的 JSON/解析可靠性、以及 goal/role 语义值、M5 lifecycle、R1/候选覆盖。

---

## 6. 最终回答

1. **LLM Proposal 是否真正写入 M5？**
   - 是。D1 的 52 个有效 operation 经 validator/bridge 写入 M5；smoke 轨迹和 `memory_events` 可见；D0 只有 1 个有效写入。
2. **D0/D1 是否真实调用对应模型？**
   - 是。D0 调用 `Qwen3.5-9B` 原始权重；D1 调用同一模型 + 现有 LoRA adapter；真实 HTTP 推理，无 Gold/规则 proposal。
3. **M5 状态是否影响下游 R1/M6.1？**
   - 是，proposal 触发 M5 写入并改变 effective state；但本次 20 任务中该影响没有转化为 V2 成功率提升，部分任务因 goal/role 语义值和候选覆盖导致空列表。
4. **D1 相比 D0 端到端是否改善？**
   - **没有**。V2 均为 12/5/3；legacy C 8、D0 8、D1 7；V2 没有提升，D1 更慢。
5. **当前失败主要来自 Parser、Validator、Memory、Planner 还是召回？**
   - D0：**Parser/JSON 有效性**（60/61 无效）；
   - D1：Parser 已大幅改善（52/64 valid），剩余在 **goal/role 语义字段映射、M5 lifecycle、R1/候选召回**；
   - 不是单一 Memory 问题：M5 桥已真实工作。

---

## 7. 约束

未重新训练、未扩数据、未修改 Parser/Gold/Evaluator、未运行 frozen benchmark、未修改 Serving V1/推荐模型、未开始 RL、未 commit/push/deploy。

真实 C/D0/D1 轨迹：

- `runs/p2bridge_c/trajectories.jsonl`
- `runs/p2bridge_d0/trajectories.jsonl`
- `runs/p2bridge_d1/trajectories.jsonl`
