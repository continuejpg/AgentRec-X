# P1C_EVALUATOR_VALIDITY — Benchmark Evaluator 语义修复与分层评分

## 0. 范围

P1C 只处理评测可信度，不修改推荐模型、Serving V1、M5/R1/M6.1/M7 策略、frozen tasks、
Gold 或哈希。

保留：

- `legacy_deterministic_pass` 原实现与原 8/20；
- 历史报告不重写；
- frozen split 未运行。

新增：

- `benchmarks/agent/evaluator_v2.py`
- `benchmarks/agent/v2_labels_dev.json`
- `P1C_LEGACY_V2_COMPARISON.csv`
- `tests/test_evaluator_v2.py`

V2 版本：

```text
Evaluator: agentrecx.eval_v2.1
Labels:    agentrecx.eval_v2.labels.1
```

本开发集 V2 标签是 **provisional_dev rubric**，不是 human relevance 标签，也不是
frozen Gold 的替代；reviewer 记录为 `P1C_audit`，证据是 task 语义 + 可信 catalogue
category/title。

---

## 1. 分层指标

`evaluator_v2.py` 独立输出：

| 指标 | 含义 |
|---|---|
| `task_state_correct` | 最终有效目标/角色是否与独立任务标签一致 |
| `preference_lifecycle_correct` | 期望 active 属性是否存在、inactive 是否被错误保留 |
| `product_role_satisfaction` | 最终商品类型/角色是否满足独立 rubric |
| `hard_constraint_satisfaction` | 可验证硬约束；证据不足为 UNKNOWN |
| `exact_identity_hit` | 是否命中 frozen expected identity；仅诊断，不用于 broad-role 成功 |
| `retrieval_adequacy` | 候选池是否足够；区分 `ADEQUATE`、`SELECTION_OMITTED`、`CANDIDATE_POOL_INSUFFICIENT` |
| `tool_action_validity` | `VALID` / `NOT_REQUIRED` / `NOT_EXECUTED` / `REQUIRED_MISSING` / `INVALID` |
| `clarification_correctness` | `NOT_REQUIRED` / `CORRECT_CONTINUED` / `STILL_WAITING` |
| `task_success_v2` | `PASS` / `FAIL` / `UNKNOWN` 综合判断 |
| `failure_class` | 语义失败、必要检索缺失、候选不足、证据 UNKNOWN、完成问题 |

关键区别：

- **legacy**：最终 `final_any` 必须命中 Gold，且 forbidden title term 直接按字符串匹配；
- **V2**：独立 category/title rubric 判断角色；broad-role 任务不强制唯一 expected ASIN；
  没有可验证价格/属性证据时报告 UNKNOWN，不判成功。

V2 **不 import** `recommendation/demo/product_role.py`，也不读取 runtime 的 MATCH 输出
作为标签。

---

## 2. Legacy 8/20 的失败归因

Legacy 8 pass、12 fail。按 V2 证据逐项分类：

| 分类 | 任务 | 数量 | 说明 |
|---|---|---:|---|
| 真实系统/状态问题 | dev_06, dev_07, dev_12, dev_13, dev_14, dev_18 | 6 | active preference 缺失；目标后无合适候选；tablets 当 filter；adapter 当主 filter |
| 评分定义/action/identity/证据问题 | dev_02, dev_09, dev_11, dev_15, dev_16, dev_17 | 6 | forbidden title term 误报；M7 answer 无 R1 action 被误判；broad-role 被唯一 ASIN 约束；价格证据不足 |

### 典型误报

- **dev_02/dev_11**：`B09CJN4XGM` 是 category `Water Filters` 的完整 filter pump，
  标题包含 “2 Collapsible Water Bottles”，legacy `must_exclude_title_terms=bottle`
  将其误判为 forbidden。V2 用独立 category/主商品角色 rubric，但仍拒绝真正的
  `Sports Water Bottles` 和 replacement/mouthpiece/adapter。
- **dev_09/dev_17**：M7 clarification answer 直接复用已验证候选，没有执行 R1；
  `actions=[]` 是真实动作缺失，不是漏记。只要候选快照满足当前目标，V2 记为
  `NOT_REQUIRED`，不再强制每个 FINALIZE 都有 R1 action。
- **dev_15/dev_16**：最终角色正确但不是唯一 expected ASIN；V2 对 broad-role 任务将
  `exact_identity_hit` 作为独立诊断，不把唯一 ASIN 当作唯一正确答案。
- **dev_16** 仍保留 caveat：最终商品是 “cookware/stove canister stand/mess kit” 类
  title/category 混合，V2 rubric 按独立 category+role 判定，exact identity 不命中。

---

## 3. Legacy vs V2 结果

基于 `dev_c_p1b3_soft_off_v2`（与 P1B-2 相同 C 配置）：

| 指标 | legacy | V2 |
|---|---:|---:|
| PASS | 8 | **12** |
| FAIL | 12 | **5** |
| UNKNOWN | 0 | **3** |

V2 变化：

**Legacy fail → V2 PASS（5）**

- dev_02, dev_09, dev_11, dev_15, dev_16

**Legacy fail → V2 FAIL（5）**

- dev_06, dev_07, dev_13, dev_14, dev_18

**Legacy fail → V2 UNKNOWN（2）**

- dev_12（最终为空，角色证据 UNKNOWN）
- dev_17（价格硬约束缺少可信价格证据）

**Legacy pass → V2 UNKNOWN（1）**

- dev_05：legacy 不检查价格证据；V2 对 price_max=50 缺可信价格时报告 UNKNOWN，
  不因为 legacy pass 就判成功。

> V2 不是单纯放宽：它把 dev_05 从 PASS 降为 UNKNOWN，把 dev_13/18 判为 FAIL；
> 12 个 V2 PASS 中 5 个来自明确的评分语义修复，而不是放宽硬约束。

逐任务证据：

```text
benchmarks/agent/reports/P1C_LEGACY_V2_COMPARISON.csv
```

字段包括 legacy/V2 每项指标、final_ids、candidate_pool、legacy_failure_class、V2 证据 JSON。

---

## 4. V2 是否仍会把错误商品判成成功？

V2 的通过条件是：

```text
product_role_satisfaction = PASS
preference_lifecycle_correct = PASS
hard_constraint_satisfaction = PASS
tool_action_validity != REQUIRED_MISSING/INVALID
clarification_correctness != STILL_WAITING
```

- dev_13 的 tablets（category Water Purifiers，标题 purification tablets）被
  `reject_title_terms` 明确拒绝，V2 FAIL；
- dev_18 的 inline adapter 被 `adapter` role term 拒绝，V2 FAIL；
- dev_05/dev_17 缺可信价格证据，V2 UNKNOWN，不 PASS；
- dev_06/dev_07/dev_14 的 active preference 缺失，V2 FAIL；
- UNKNOWN 不会被算作 MATCH 或成功。

尚未完成的独立人工相关性验证仍由 `human_labels_v2_schema.md` 定义；V2 分数不等于
human relevance，也不等于线上质量。

---

## 5. R1 action 语义

新增 `tool_action_validity`：

| 场景 | V2 |
|---|---|
| 没有 R1 action，但 final candidate snapshot 独立满足目标 | `NOT_REQUIRED`，可 PASS |
| 没有 R1 action，候选池也没有可接受候选 | `REQUIRED_MISSING`，FAIL |
| 没有 R1 action，final 证据不足 | `NOT_EXECUTED`，不虚构成动作 |
| 有 action 但名称/来源不合法 | `INVALID`，FAIL |
| 有可信 RETRIEVE/RERANK | `VALID` |

dev_09/dev_17 属于 **M7 已合法使用候选快照**，不是隐藏 RERANK；不补造动作。
原有 `actions_any` 断言保留为 legacy，不删除。

---

## 6. 直接回答

1. **原有 8/20 中多少失败由真实系统问题造成，多少受评分定义影响？**
   - 12 个 legacy fail 中：约 6 个是真实系统/状态/候选问题；
   - 约 6 个涉及 forbidden title term、唯一 ASIN 过窄、M7 action 断言或证据 UNKNOWN；
   - 其中 5 个 legacy-fail 在 V2 PASS，1 个 dev_17 仍 UNKNOWN。

2. **哪些任务 V2 判定变化，证据是什么？**
   - PASS 新增：dev_02/09/11/15/16；
   - dev_02/11 因 `Water Filters` category 与主商品角色，而非 blanket “bottle”；
   - dev_09 因 M7 合法快照无 R1 action；
   - dev_15/16 因 broad role，而非唯一 expected ASIN；
   - dev_13/18 因 tablets/adapter 证据被判 FAIL；dev_05 因价格证据 UNKNOWN。

3. **V2 是否仍会把错误商品判成成功？**
   - 未观察到错误主角色通过：tablets、adapter、无 active preference 的任务均被拒绝；
   - UNKNOWN 不成功；但 V2 是 provisional rubric，不是 human ground truth。

4. **M7 无 R1 action 何时合法、何时失败？**
   - 合法：已验证候选 snapshot 满足当前目标，无新目标/硬约束变化；
   - 失败：候选不满足目标，或目标变化要求新的检索而没有执行；
   - 动作不存在时记录 `not_executed`/`REQUIRED_MISSING`，不伪造。

5. **A/B/C/D 能力是否公平？**
   - 见 `P1C_POLICY_FAIRNESS.md`；当前仅 C runnable，A/B/D 因无 LLM unavailable；
   - 即使接入 LLM，A/B 仍缺少 C/D 的 task-scoped role/soft postprocessor，不能直接
     与 C/D 比较；需共享执行接口后再比较。

6. **共享 SQLite 与任务顺序是否影响结果？**
   - 见 `P1C_REPRODUCIBILITY.md`；forward/repeat/reverse 的 final、actions、候选池、
     R1 query/constraints、语义 state 全部 20/20 一致；
   - 原始 `state_digest` 因 session/turn/event identity 不同而不同，不能作为语义
     不一致证据；各运行使用独立 state 目录。

7. **是否具备接入真实 LLM 后做有效对照的条件？**
   - **尚不具备**：A/B/D 需 LLM client；需共享 postprocessor、相同 role/soft/
     hard-constraint 输出接口、相同 tool budget；当前只能明确 unavailable，
     不能生成模拟效果。

---

## 7. 测试

`tests/test_evaluator_v2.py` 覆盖：

- forbidden title term 误报；
- role 与 exact identity 分离；
- M7 无 R1 action 的合法性；
- 候选不足时必要检索缺失；
- UNKNOWN 证据不升格；
- V2 labels 版本/数量；
- A/B/C/D capability manifest。

V2 不修改 legacy evaluator；legacy 测试保持原语义。

最终相关回归：

```text
235 passed
```

