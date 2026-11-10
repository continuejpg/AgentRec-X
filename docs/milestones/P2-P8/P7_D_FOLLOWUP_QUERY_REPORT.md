# P7-D Follow-up 检索目标保留修复报告

## 0. 结论

已定位并最小修复普通 follow-up 丢失显式商品类型的问题：

- 修改位置：`recommendation/demo/agent_service.py` 的 Query 构造处；
- 修复前：turn1 Query 只有 `Those look fine...`，cookware 目标丢失；
- 修复后：turn1 Query 为 `cookware + Those look fine...`；
- `normal_termination` 的召回从 **role MATCH=0 / final=0** 改善为 **role MATCH=2 / final=5**；
- 但任务结果仍为 **UNKNOWN**（5 个最终候选中 3 个 role-UNKNOWN），不是 PASS；
- `goal_switch` 仍 PASS：role_target 为 `filtered_bottle/main`，最终商品为新目标，没有被旧目标替换。

本轮只验证两个 Dev 场景，未运行全部 Benchmark、未启动 Qwen/GPU。

## 1. 修复位置与源码 diff

修改文件：

- `recommendation/demo/agent_service.py`
- `tests/test_p7_d_followup_target_anchor.py`（1 个针对性回归）

修复逻辑：

```python
if (
    r1_plan_action == "RETRIEVE"
    and not role_target_changed
    and not r1_goal_correction
):
    r1_run_terms = _with_effective_target_type(
        r1_run_terms, self._role_target_for(payload.user_key)
    )
```

`_with_effective_target_type()` 只把**当前 `role_target.type`** 补入 Query；不重用旧 Query terms、不重用旧偏好、不重用旧预算文本。

因此：

- 普通 follow-up 省略商品名时，保留当前有效类型；
- 显式目标切换设置 `role_target_changed=True`，不追加旧类型，以新消息为准；
- 预算更新/偏好撤回时不会因为本修复重新引入旧偏好；
- 没有修改 M5/M9、Role gate、Eligibility、Validator、BM25/SASRec/Two-Tower/RRF、Serving V1。

源码 diff：`P7_D_FOLLOWUP_QUERY_SOURCE_DIFF.txt`

## 2. Follow-up 前后的实际 Query

### normal_termination

**turn0**

```text
message: Show me camping cookware under $50.
query_terms: Show, me, camping, cookware, under, $50.
role_target: {type: cookware, role: main}
```

**turn1 修复前**

```text
message: Those look fine if you have eligible options; show me the best ones.
query_terms: Those, look, fine, if, you, have, eligible, options;
```

没有 `cookware`。

**turn1 修复后**

```text
message: Those look fine if you have eligible options; show me the best ones.
query_terms: cookware, Those, look, fine, if, you, have, eligible, options;
role_target: {type: cookware, role: main}
```

已有候选窗口为空（`dynamic_retrieval_enabled()` 在该路径为 false，`_r1_last_responses` 未保存），因此 turn1 只做 Query 类型保留，没有依赖旧候选复用。

### goal_switch

**turn1**

```text
message: Actually, I would like a water bottle instead.
query_terms: water filter for hiking, water bottle, Actually, I would, like, ...
role_target: {type: filtered_bottle, role: main}
final target: filtered_bottle/main
```

`role_target_changed=True`，本修复不追加旧 `water_filter` 类型；新目标 `filtered_bottle` 由新消息和 role target 生效。  
轨迹中仍出现旧的 `water filter for hiking` 文本，这是此前 legacy M9 query 内容，不是本修复追加的旧 anchor；最终目标/展示商品以新 target 为准。

## 3. 两个 Dev 场景的真实结果

| 任务 | 修复前 Rule | 修复后 Rule | 变化 |
|---|---|---|---|
| normal_termination | FAIL，final=0 | **UNKNOWN**，final=5 | 召回改善，但未 PASS |
| goal_switch | PASS，final=`B0BFR4VJTW, B0BCJ6LBJ4` | **PASS**，同 final | 保持新目标，不受影响 |

### normal_termination 候选变化

**修复前 turn1**

| 阶段 | 值 |
|---|---:|
| query | 无 cookware |
| RRF candidates | 7 |
| role MATCH | 0 |
| final candidates | 0 |

**修复后 turn1**

| 阶段 | 值 |
|---|---:|
| query | `cookware, Those...` |
| RRF candidates | 9 |
| role MATCH | 2 |
| role mismatch / unknown | 3 / 4 |
| final candidates | 5：`B07S8GNB4T, B09FXNJZTV, B08557VLNT, B09YCYNCSK, B091VYKC96` |
| final role | MATCH=2, unknown=3 |

修复前 turn1 的 `role MATCH=0 / final=0` 是 P7-C 定位的召回丢失；修复后该丢失消失，至少两个 cookware 候选进入最终展示。  
但 `normal_termination` 仍为 **UNKNOWN**：三个最终商品 role-UNKNOWN，不是 role MATCH；这是商品证据限制，不是 Query 目标丢失。

### goal_switch

- 修复前后均 PASS；
- 最终 ID 均为 `B0BFR4VJTW, B0BCJ6LBJ4`；
- final role target 为 `filtered_bottle/main`，不是 `water_filter/main`；
- 本修复没有把旧类型追加到新目标查询。

## 4. 是否修复了已知召回失败

**是部分修复，但不是任务成功修复。**

- 已修复：follow-up Query 不再丢失 `cookware`；
- 已改善：turn1 role MATCH 0→2，final candidates 0→5；
- 未修复：最终 5 个候选中仍有 3 个 role-UNKNOWN，任务结果为 UNKNOWN；
- 未修复：258 个合格商品的整体召回覆盖（本轮不做扩建）；
- 不能宣称 `normal_termination` 已 PASS。

## 5. Rule / Qwen 后续公平对照的影响

**会影响。**

- 修复位于共享的 `_execute_plan` Query 构造路径，Rule 和 Qwen Planner RETRIEVE 都经过这里；
- P6.4 的旧 Rule 结果和 P7-B 的旧 Qwen Base/SFT 结果是在修复前产生的，**不能与修复后结果直接混合比较**；
- 下一轮公平对照必须让 Rule / Base / SFT-v0 都在同一修复后代码上重跑受影响任务；
- 本轮的 Rule 结果只用于验证两个 Dev 场景，不是新的 6-task 基线；
- P7-B 的 Soft-State 隔离未改变：Query 只增加当前有效类型，不写 M5/M9、不改变 role target。

## 6. 验证

```text
tests/test_p7_d_followup_target_anchor.py    1 passed
```

验证内容：

- follow-up terms 不含类型时，补入当前有效 `cookware`；
- 已含类型时保持原 terms；
- 显式目标切换在代码中由 `role_target_changed` 跳过，不追加旧类型。

## 7. 残余问题

1. `goal_switch` 的旧 `water filter for hiking` 文本仍存在于 legacy M9 query 内容中；本轮未删除，因为它不是本修复追加的旧 anchor，且新 target/final 正确。若后续要求 Query 完全去除旧目标词，需要另做最小 query-cleanup。
2. `state_constraints_changed` 仍会触发 turn1 重新 RETRIEVE；本轮只修复了该 RETRIEVE 的 Query 目标保留，没有改变 M5 触发逻辑。
3. 258 个合格 cookware 的整体召回覆盖未优化；修复只是让当前目标不再因 follow-up 丢失。

## 8. 交付物

- 本报告：`P7_D_FOLLOWUP_QUERY_REPORT.md`
- 源码 diff：`P7_D_FOLLOWUP_QUERY_SOURCE_DIFF.txt`
- 修复后 Rule 两场景：`runs/p6_4_dynamic/rule_20261107_192756/`
- 修复前对照：`runs/p6_4_dynamic/dev_20261107_160922/`（P6.4 Rule）
- 定向回归：`tests/test_p7_d_followup_target_anchor.py`

本轮未修改 Serving V1/Gold/Evaluator，未训练，未运行 Holdout，未扩充 Benchmark，未 commit/push/deploy，未操作 GPU 实例。
