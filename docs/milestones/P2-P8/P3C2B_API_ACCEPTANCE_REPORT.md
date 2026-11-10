# P3C-2B Qwen API 验收与候选过滤定位报告

## 1. 结论分级

| 项目 | 状态 |
|---|---|
| Qwen 可选 API 代码路径 | 已实现 |
| Stub HTTP Server → 真实 Shopping API 路由集成 | **PASS（测试替身，非 Qwen）** |
| Stub 两轮 M5/R1/M6.1 链路 | PASS |
| 模型不可用失败路径 | PASS（503，M5 事件增量为 0） |
| 真实 Qwen SFT-v2 API 验证 | **因无 GPU 未验证** |
| 候选过滤诊断 | 已实现；Stub 运行有真实分层计数 |

---

## 2. 配置关系

Stub 验收显式设置：

```text
AGENTRECX_QWEN_PROPOSAL=on
AGENTRECX_QWEN_PROPOSAL_URL=<stub>
AGENTRECX_STATE_MODE=apply_constraints
AGENTRECX_R1_DYNAMIC_RETRIEVAL=on
AGENTRECX_M5_SOFT_RERANK=on
AGENTRECX_M7_GROUNDED_ACTIONS=on
AGENTRECX_M7_BUDGET=2
AGENTRECX_M8_TRAJECTORY=on
```

因此不是只开 Qwen 而没开状态消费；M5/R1/soft 全部实际启用。

默认入口未设置这些变量：

- `/v1/recommend` 不调用 Qwen/M5/R1；
- 默认 Agent API 不调用 Qwen；
- 只有显式 Profile 才走 Qwen。

---

## 3. Stub 验收结果

轨迹：`P3C2B_AGENT_API_TRAJECTORY.jsonl`  
必须视为 **STUB_TEST_DOUBLE_NOT_REAL_QWEN**，不是模型结果。

### 第一轮

请求：设置 water filter、预算 $30、lighter preference。

结果：

- HTTP 200，route=`recommend`；
- proposal `valid=true`，accepted 3 个 operation；
- M5：`events=6`，`hard_constraint_count=1`，`preference_count=2`；
- R1：`RETRIEVE`；
- soft M5：`rerank_applied=true`，`consumed_soft_preference_count=1`；
- 最终商品来自真实目录。

### 第二轮

请求：预算改 $50、撤回 lighter preference。

结果：

- proposal valid；retraction_count=1；
- M5 `retraction_match.resolved=1`，`hard_constraint_count=1`（预算更新后保留）；
- `preference_count=0`，soft `retraction_effective=true`；
- R1：`RETRIEVE`；
- 最终商品变化。

### 模型不可用

URL 指向不可达端口：

- HTTP **503** `qwen_proposal_unavailable`；
- M5 事件数 before=11, after=11, **delta=0**；
- 没有规则 Proposal fallback。

---

## 4. 候选过滤真实计数（Stub 两轮）

`P3C2B_CANDIDATE_FILTER_TRACE.csv` 记录的是 Stub 两轮，不是 accessory_to_main 场景：

第一轮：

```text
raw_recall_count=10（grounded ledger 身份数，非源原始 recall）
rrf_candidate_count=15
grounded_candidate_count=15
role_main: match=4, mismatch=8, unknown=3
final_candidate_count=5
```

第二轮：

```text
raw_recall_count=10
rrf_candidate_count=10
grounded=10
role_main: match=3, mismatch=5, unknown=2
final=5
```

这证明了当前代码已经能在真实 API 路径记录 role 层剔除原因。

### accessory_to_main

P3C1 受控记录仍是：

```text
grounded=10 → final=0
```

由于本阶段未用新 instrumentation 重跑该确切场景，精确的 role/hard 归零计数仍不能声称；
可确认的是代码已具备计数能力，且 Stub 证据显示 **role mismatch 是实际会剔出大量候选的阶段**。
因此报告标记为：

**UNKNOWN_BETWEEN_GROUNDED_AND_FINAL（需 accessory 场景重跑后确定）**

没有放宽角色过滤、没有用 Gold 补候选。

---

## 5. 真实 Qwen 验收

未执行：

- 当前 AutoDL 实例无 GPU（无 `/dev/nvidia*`，`nvidia-smi` 不可用）；
- 无法加载 Qwen3.5-9B SFT-v2模型。

因此：

- 没有真实 Qwen proposal；
- 没有真实 Qwen M5 事件；
- 没有真实 Qwen 两轮最终商品；
- 不把 Stub 轨迹冒充 Qwen。

模型/Adapter 文件在断电前已检查存在；本地也有 SFT-v2 adapter tar 备份。

---

## 6. 测试

```text
tests/test_p3c2_qwen_api.py
3 passed
```

覆盖：

- 默认关闭；
- 启用缺 URL 拒绝；
- 模型不可达抛 `qwen_proposal_unavailable`，无 fallback。

---

## 7. 回答

1. **Qwen SFT-v2 是否真正进入可选 Agent API？**
   - 代码路径已接线；Stub 证明 API route 会调用 proposer 并注入 proposal。
   - 真实 Qwen 因无 GPU **未验证**。

2. **是否通过 API 实际更新 M5？**
   - Stub 证明：是；真实 Qwen 未验证。

3. **配件→主商品候选在哪一步归零？**
   - P3C1 已知 grounded=10→final=0；新 instrumentation 已可记录 role/hard 层。
   - 该确切场景尚未重跑，精确阶段标记 UNKNOWN；Stub 显示 role mismatch 是真实剔除阶段。

4. **默认 Serving V1 是否保持不变？**
   - 是；未修改 `/v1/recommend`，默认 Agent 也不调用 Qwen。

5. **是否可进一步引入 LLM Planner？**
   - 暂不建议；真实 Qwen 未验收，候选 role/hard 归零与 M6.1 消费仍未完全定位，不应先做 Planner/RL。

---

## 8. 交付

- `P3C2B_API_ACCEPTANCE_REPORT.md`
- `P3C2B_AGENT_API_TRAJECTORY.jsonl`（STUB，明确标记）
- `P3C2B_CANDIDATE_FILTER_TRACE.csv`
- `P3C2B_SOURCE_DIFF.txt`
- `P3C2B_STUB_API_TRAJECTORY.jsonl`（Stub 原始输出）

未修改 Serving V1，未训练，未运行 frozen Benchmark，未 commit/push/deploy。  
本阶段完成 CPU/Stub 验收与失败路径验证；真实 Qwen 验收需有 GPU 后执行。
