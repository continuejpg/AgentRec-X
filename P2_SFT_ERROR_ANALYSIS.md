# P2-SFT 分类别错误分析

## 0. 口径

- 数据：冻结的 200 条 Parser Test；
- 复用已有 Base/SFT/Rule 预测，**未重新运行模型**；
- Gold 和原始 metrics 未修改；
- 原始 macro 指标保留；新增 supported-macro 复算，原因是原宏平均把没有 Gold support 的
  操作类型也计入分母；
- 原始文件：`training/agent_sft/eval/{base,sft,rule_local}/`
- 分类别结果：
  - `training/agent_sft/reports/P2_SFT_TEST_OPERATION_METRICS.csv`
  - `training/agent_sft/reports/P2_SFT_TEST_ERROR_SUMMARY.json`
  - `training/agent_sft/reports/P2_SFT_TEST_ERROR_CASES_{base,sft,rule}.json`

---

## 1. 总体指标与分母

| provider | valid JSON / 200 | Exact Match | Macro（原始，12 op） | Macro（有 support 的 9 op） |
|---|---:|---:|---:|---:|
| Rule | 200 | 0.820 | 0.750 | 1.000 |
| Base | 66 | 0.095 | 0.254 | 0.339 |
| SFT | 194 | 0.945 | 0.703 | **0.938** |

关键点：

- **Exact Match 的样本分母固定是 200**；无效 JSON 直接算不匹配；
- Rule 的 82% Exact 中，36 个错误是 `noop/abstain` 的 `scope` 字段不一致，不是操作类型错；
- Base 有 134/200 个无效 JSON；SFT 有 6/200 个无效 JSON；
- 原始 macro 把 **12 个操作标签全部等权**，其中 `update_preference`、
  `retract_preference`、`set_constraint` 在该测试集 Gold support=0，因此这些类型
  的 F1=0 也被算进分母，压低了 SFT 的 macro；
- 仅按有 support 的 9 个操作计算，SFT macro 为 **0.938**，Rule 为 **1.000**。

---

## 2. SFT 分类别结果

| Operation | Gold support | Pred | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| set_goal | 18 | 18 | 1.000 | 1.000 | 1.000 |
| update_goal | 36 | 36 | 1.000 | 1.000 | 1.000 |
| add_preference | 36 | 36 | 1.000 | 1.000 | 1.000 |
| update_constraint | 19 | 19 | 1.000 | 1.000 | 1.000 |
| retract_constraint | 19 | 19 | 1.000 | 1.000 | 1.000 |
| resolve_reference | 18 | 18 | 1.000 | 1.000 | 1.000 |
| clarify | 18 | 7 | 1.000 | 0.389 | 0.560 |
| noop | 18 | 23 | 0.783 | 1.000 | 0.878 |
| abstain | 18 | 18 | 1.000 | 1.000 | 1.000 |
| update_preference | 0 | 0 | 0 | 0 | 0 |
| retract_preference | 0 | 0 | 0 | 0 | 0 |
| set_constraint | 0 | 0 | 0 | 0 | 0 |

SFT 的错误主要集中在 **clarify ↔ noop**：

- 5 个 ambiguity 样本被输出成 `noop`；
- clarify 预测 7 个，Gold 18 个，Recall 0.389；
- noop 预测 23 个，Gold 18 个，因此 precision 0.783；
- 其他 8 个有 support 的操作类型全部 F1=1.0。

### 主要错误案例

1. `p2sft_test_00073`（ambiguity）
   - Gold：`clarify(product_type)`
   - SFT：`noop(declined_preference)`
   - 原因：模型把“帮我选类别”误读为无需操作；不是 JSON 格式错误。

2. Base 的主要错误是格式/slot 错误：
   - invalid JSON 134 条；
   - 有效样本中 wrong_value 19、wrong_direction 16、wrong_attribute 10；
   - 因此 Base Exact 0.095 与 macro 0.254。

3. Rule 的 36 个 Exact 错误：
   - 全部是 `noop/abstain` 的 `scope` 字段（Rule 写 `task`，Gold 为 `null`）；
   - 操作类型 F1 仍为 1.0，所以 Rule 有较高 macro。

---

## 3. 混淆矩阵（SFT，首操作）

```text
set_goal          -> set_goal          18
update_goal       -> update_goal       36
add_preference    -> add_preference    36
update_constraint -> update_constraint 19
retract_constraint-> retract_constraint 19
resolve_reference -> resolve_reference 18
clarify           -> clarify           13
clarify           -> noop               5   <-- 主要错误
noop              -> noop               18
abstain           -> abstain            18
```

完整 confusion rows 在：

`training/agent_sft/reports/P2_SFT_TEST_OPERATION_METRICS.csv`
（`operation=__confusion__` 行）。

---

## 4. 为什么 Exact 94.5%，Macro 只有 70.3%？

不是指标自相矛盾，而是两者分母不同：

- **Exact Match**：每条样本整体 operation 序列完全匹配；SFT 194/200 通过；
- **Macro-F1**：按 12 个操作标签等权平均；其中
  - 3 个标签在本测试集没有 Gold support，F1=0；
  - clarify F1=0.56、noop F1=0.878；
  - 这些都会进入 macro 分母；
- 因此 11 个样本级错误会同时造成 `clarify` recall 和 `noop` precision 下降，
  在 12 类 macro 中被放大为约 0.23 的落差。

**修正后的版本化指标**为 `macro_f1_supported=0.938`；原始 `0.703` 保留，不覆盖。

---

## 5. 无效 JSON / NOOP / ABSTAIN / 撤回分母

| 类别 | 分母 | Base | SFT | Rule |
|---|---:|---:|---:|---:|
| 总样本 | 200 | 200 | 200 | 200 |
| 有效 JSON | 200 | 66 | 194 | 200 |
| NOOP Gold support | 18 | 18 | 18 | 18 |
| ABSTAIN Gold support | 18 | 18 | 18 | 18 |
| 撤回（retract_constraint） | 19 | 19 | 19 | 19 |
| 撤回（retract_preference） | 0 | 0 | 0 | 0 |

- NOOP/ABSTAIN 的 support 和预测都在表中明确显示；
- 无效 JSON 没有被算作“正确 abstain”，也没有被排除出总样本；
- retraction accuracy 的样本分母是 19（`retract_constraint`），SFT 为 1.0；
- 本测试集没有 `retract_preference` Gold 样本，因此不能据此评价 soft-preference 撤回，
  需要独立困难测试集补足。

---

## 6. 结论

- SFT 在 **set_goal / update_goal / add_preference / update_constraint /
  retract_constraint / resolve_reference / abstain** 上表现很好；
- 明显弱点是 **clarify vs noop** 的语义区分，造成 5 个样本级错误；
- Macro 低是“12 类等权 + 3 个零 support 类别”与 clarify/noop 少数错误共同造成的，
  不是 Exact 指标错误；
- 原始指标未修改；`macro_f1_supported` 作为版本化复算指标单独报告。
