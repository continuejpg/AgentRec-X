# P2-Bridge 任务失败分析

真实 C/D0/D1 20 dev 任务结果见 `P2_BRIDGE_IMPLEMENTATION_REPORT.md` 与
`training/agent_sft/eval/P2_BRIDGE_AGENT_COMPARISON.csv`。

## 1. 总体

| arm | V2 PASS | FAIL | UNKNOWN | legacy |
|---|---:|---:|---:|---:|
| C | 12 | 5 | 3 | 8 |
| D0 | 12 | 5 | 3 | 8 |
| D1 | 12 | 5 | 3 | 7 |

V2 成功任务集合变化不大；D1 没有超过 C/D0。

## 2. D0 失败

D0 共 61 个 proposal 轮次：

- 60 轮无效 JSON，被 Parser/Validator 拒绝；
- 1 轮有效；
- 没有规则 fallback 冒充模型 proposal。

因此 D0 的多数轮次没有模型 M5 更新；其 V2 结果基本等同 C，不能把 D0 的通过算作 LLM
proposal 生效。

## 3. D1 失败

D1 共 64 个 proposal 轮次：

- 52 轮 valid、12 轮 invalid；
- 52 个 accepted operations；
- M5 bridge 确实写入。

D1 的主要失败：

1. **goal/role 语义值映射不足**：例如 `filter_cartridge`、`mouthpiece`、`straw`、
   `complete_water_filter` 等值不是标准 category；桥已能按 accessory/main 归一化，
   但候选召回仍可能为空。
2. **偏好生命周期**：dev_06、dev_07、dev_14 的 lifecycle FAIL；C/D0/D1 都有，不是桥
   独有。
3. **budget 多轮**：dev_05、dev_17 为 UNKNOWN；预算更新本身 PASS，但最终 identity
   证据不足。
4. **角色/候选覆盖**：dev_13、dev_18 role FAIL，dev_12 UNKNOWN，属于产品角色与候选
   召回问题，不是 JSON 解析问题。
5. **D1 延迟/Token**：p50 17.8s、p95 33.1s，均高于 C/D0；Token 约 25.8k，与 D0
   持平。

## 4. 归因

| 失败类别 | C | D0 | D1 | 说明 |
|---|---:|---:|---:|---|
| Parser/JSON 无效 | 0 | 60/61 | 12/64 | D0 的首要问题；D1 明显改善 |
| M5 lifecycle FAIL | 3 | 3 | 3 | 偏好撤回/更新语义 |
| 产品角色 FAIL/UNKNOWN | 3 | 3 | 3 | goal/role 与候选 |
| 预算多轮 UNKNOWN | 2 | 2 | 2 | 最终证据/identity 不足 |
| 检索动作缺失 | 3 | 3 | 3 | 非桥接问题 |

结论：**桥接本身已工作**；D1 剩余瓶颈是 goal/role 语义字段、M5 lifecycle 与 R1/
候选覆盖，不再是“proposal 只写日志”。
