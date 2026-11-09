# P1C_POLICY_FAIRNESS — A/B/C/D 实验公平性审计

## 0. 结论先行

当前 **只有 C 组可运行**：

| policy | runnable | reason |
|---|---:|---|
| A | 0/1 | `llm_not_configured: LLMPolicyError` |
| B | 0/1 | `llm_not_configured: LLMPolicyError` |
| C | 20/20 | deterministic R1 |
| D | 0/1 | `llm_not_configured: LLMPolicyError` |

观察命令：

```text
python -m benchmarks.agent.run --split dev --policies A,B,D --state-mode shadow --limit 1 \
  --out runs/agent_benchmark/dev_c_p1c_policies
```

结果为 `evaluations: 3; runnable: 0`。没有生成模拟 LLM 输出，也没有把 unavailable
当失败样本。

因此当前不能声称 A/B/C/D 的通过率进行了有效对照；只能审计接口是否具备公平比较条件。

---

## 1. 共享事实

四个 adapter 均通过同一个 runtime 的同一个 `agent_service` 访问：

- 同一真实 catalogue / `MetadataIndex`；
- 同一 history、catalogue search、Two-Tower（若启用）与 RRF 基础；
- 同一 `k`/Top-K 请求接口；
- 同一 hard constraint / 商品资格基础服务（C/D 的 task stateful 路径）；
- 同一 frozen task history 与 user turns；
- 独立 session/task 目录与 SQLite state（每次 run 的 `--out` 独立）。

A/B/D 不可运行不是能力被故意削弱，而是未配置 LLM client；benchmark 不生成
fallback 或模拟回答。

---

## 2. 能力矩阵与实际入口

代码/运行审计：

| policy | execution entry | stateful | role eligibility | M5 soft rerank | requires LLM |
|---|---|---|---|---|---|
| A | `service.run` | 否 | adapter 的 `promote_explicit_type_matches` | 否 | 是 |
| B | `service.run` | 否 | 同上 | 否 | 是 |
| C | `run_shopping_turn` | 是 | task-scoped grounded role filter | soft_on 时可用 | 否 |
| D | `run_shopping_turn` | 是 | task-scoped grounded role filter | soft_on 时可用 | 是 |

A/B 与 C/D 的关键差异：

1. **stateful shopping**：C/D 有 M5/M7/R1/task state；A/B 只有单次 request，
   不能使用 M7 clarification、M5 soft state 或跨轮 role target；
2. **postprocessing**：C/D 的最终回答经过 `product_role` grounded role filter；
   A/B 仅有 legacy `promote_explicit_type_matches`，没有同一最终资格层；
3. **planner**：A/B 使用 LLM query rewrite；C 使用 deterministic R1；D 使用 LLM
   preference proposal。这是**策略变量**，不是接口差异，但当前 C 唯一可运行；
4. **clarification**：C/D 可进入 M7 问题；A/B 无跨轮澄清状态。

因此，即使未来接入 LLM，A/B 与 C/D 也不能直接比较通过率，除非：

- 两者的最终推荐都经过同一 postprocessor（role/hard/soft eligibility）；
- 或明确把 role/soft eligibility 设为同一变量并同时关闭；
- A/B 不得被剥夺完整 history 或有效约束能力来“削弱强基线”；
- 所有 arm 使用相同 tool budget、最大 clarification 轮次和 k。

---

## 3. 公平性现状评级

| 维度 | 状态 |
|---|---|
| 同一 catalogue / models / fusion | PASS（共享 runtime） |
| 同一 Top-K 与请求接口 | PASS |
| 同一候选来源集合 | PASS（由 runtime 注册） |
| 相同 tool budget | 代码配置相同；A/B unavailable，无法实测 |
| 相同 role/soft 最终资格 | **FAIL/未对齐**：A/B 缺 C/D 的 postprocessor |
| 相同 clarification 轮次/预算 | 未对齐：A/B 无 M7 state |
| 独立 state / session | PASS（独立 out dir；见 reproducibility） |
| 可比较延迟/完成状态 | A/B/D 无运行，无法比较 |
| 无模拟 LLM | PASS（unavailable 明确记录） |

结论：**公平性尚未建立，不是 C 优于 A/B/D 的证据**。P1C 的 C 8/20 只能作为 C arm
在当前 deterministic 配置下的观测。

---

## 4. 优先修复顺序

1. 配置真实 LLM client，使 A/B/D 可运行；仍不得生成模拟结果。
2. 建立共享的 `PolicyRecommendationPostprocessor`：A/B/C/D 都在最终列表上应用同一
   role/hard/soft eligibility；A/B 的 stateless 限制需要显式建模，而不是隐式绕过。
3. 统一 `k`、tool budget、maximum clarification rounds、candidate source 配置。
4. 将 M7 clarification 作为实验变量：A/B 不提问、C/D 可提问，需同时报告轮次和
   retrieval/tool 使用量，而不是只比较通过率。
5. 在 A/B 可运行前不报告四组对照结论。

---

## 5. 直接回答

- **A/B/C/D 是否使用同一 catalogue/model/source？** 代码上共享同一 runtime；
  实测仅 C 可运行。
- **A/B 是否被剥夺历史/约束能力？** 当前没有运行，无法支持该指控；但 A/B 的
  stateless 入口天然不使用 M5 task state，这是显式能力差异，不是 benchmark 的
  人为削弱。
- **是否有额外工具/状态能力差异？** 有：C/D stateful role/soft/M7；A/B 仅 stateless
  `service.run` + legacy type promotion。需要共享 postprocessor 后才可公平比较。
- **是否具备 LLM 对照条件？** 当前不具备：A/B/D unavailable；不得用模拟效果替代。
