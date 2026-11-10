# P3C 最终验收报告

## 0. 结论

| 项目 | 状态 |
|---|---|
| Stub Qwen API → M5/R1/M6.1 验收 | **PASS**（测试替身，不是 Qwen） |
| 模型不可用失败路径 | **PASS（503，无 M5 写入）** |
| 第一轮 3 Accepted → 6 M5 Events 语义 | **已解释，非 Bug** |
| MATCH=4/3 与 Final=5 差异 | **已解释：UNKNOWN 保留 + 后续显式类型收敛** |
| accessory_to_main 空候选 | **已定位到 Role MATCH 阶段** |
| M5 hard constraint 是否被 eligibility 消费 | **未消费：`hard_constraint_evaluated=false`，未修复** |
| 真实 Qwen SFT-v2 API | **未验证：无 GPU** |
| 默认 Serving V1 | **未修改** |

---

## 1. 最终商品 item_id / Role / 硬约束 / 入选原因

Stub 轨迹：`P3C2B_AGENT_API_TRAJECTORY.jsonl`。Stub 不是真实 Qwen。

### Turn 1

API 返回商品：

| parent_asin | item_id | Role | 入选原因 |
|---|---:|---|---|
| B00CBR1UI6 | 38577 | MATCH | trusted title/category 为 water filter，main role |
| B08XJQ9HPX | 123918 | MATCH | trusted title/category 为 water filter，main role |
| B09Y53NYDK | 136697 | MATCH | trusted title/category 为 water filter，main role |

候选层 `final_candidate_count=5` 实际包含：

- 4 个 MATCH；
- 1 个 UNKNOWN：`B00LCWB5UY`（item_id 48441，money belt；标题/类目无法证明 water_filter）。

最终 API 列表只保留 3 个 MATCH。差异原因：

- `select_grounded_recommendations`（`recommendation/demo/product_role.py:383-392`）在存在 MATCH 时
  会 **MATCH + UNKNOWN 一起保留**，因此候选层出现 5；
- 后续显式类型收敛/最终身份校验移除了 UNKNOWN（money belt），API 返回 3 个 MATCH。

这不是资格校验 Bug；它是“UNKNOWN 不作违规、但显式类型收敛后不展示”的预期路径。

### Turn 2

API 返回 3 个商品，全部 MATCH：

| parent_asin | item_id | Role |
|---|---:|---|
| B08XJQ9HPX | 123918 | MATCH |
| B09Y53NYDK | 136697 | MATCH |
| B00CBR1UI6 | 38577 | MATCH |

候选层是 3 MATCH + 2 UNKNOWN（`B001HBUW1I`、`B00LCWB5UY`），最终只返回 MATCH。

### 硬约束

Turn 1/2 的 M5 状态显示：

```text
hard_constraint_count = 1 (price_max)
```

但候选 trace 和 R1 诊断显示：

```text
hard_constraint_evaluated = false
hard_constraint_eligible_count = 0
hard_constraint_ineligible_count = 0
hard_constraint_unresolved_count = 0
R1 constraints = []
```

因此当前执行路径**没有把 M5 `price_max` 交给硬约束 eligibility filter**；最终商品没有经过
hard filter 验证。这是一个仍未解决的执行缺陷；未用 Gold 或放宽规则掩盖。

---

## 2. 3 Accepted Operations → 6 M5 Events

Turn 1 的 3 个 accepted operations 对应 6 个 M5 events，逐条为：

| seq | event_type | 来源 |
|---:|---|---|
| 1 | `task_context_updated` | 每轮上下文 |
| 2 | `user_preference_stated` `goal:water_filter` | `set_goal` |
| 3 | `user_preference_stated` `weight/minimize` | `add_preference` |
| 4 | `user_constraint_stated` `price_max=30` | `update_constraint` |
| 5 | `candidate_pool_referenced` | 候选快照 |
| 6 | `ranking_snapshot_referenced` | 排序快照 |

因此：3 个 **状态事实** + 1 个上下文 + 2 个受托快照 = 6 events。

这不是一对多 Bug；也没有规则 extractor 双写：当 proposal 传入时，同轮规则 M5 写入被关闭
（`agent_service.py` 的 proposal 分支）。

Turn 2：3 个事件（上下文、预算更新、偏好撤回）+ 2 个快照 = 5 events，累计 11。

**幂等性**：本 Stub 未重复发送同一请求，因此 API 级幂等性仍为未验证；M5 event identity
的重复 apply 行为已由 P3C-1 单元测试覆盖。

---

## 3. accessory_to_main 空候选定位

使用 P3C1 受控数据中的 accessory_to_main grounded pool，按当前 catalogue 和
`ProductTarget("water_filter", TargetRole.MAIN)` 逐项评估：

```text
grounded pool = 10
Role MATCH   = 0
Role MISMATCH= 2   (B0BPXGTP31, B0BCJZ6S8D: accessory/consumable)
Role UNKNOWN = 8   (无受信 water_filter main 证据)
```

`select_grounded_recommendations` 的规则是：

- 有 MATCH：保留 MATCH + UNKNOWN；
- **无 MATCH：返回空**（`product_role.py:392`）。

因此 accessory_to_main 的 Final=0 发生在 **Role MATCH 阶段**：

```text
Grounded (10)
  -> Role filter: 0 MATCH, 2 MISMATCH, 8 UNKNOWN
  -> selected = []   ← 这里归零
  -> Hard constraint / M6.1 / Final 不再有候选
```

不是 Query 构造、RRF、硬约束或最终排序把它们删掉；是候选池里没有任何可证明为
water_filter main 的商品。不能通过放宽 role 或 Gold 补候选。

---

## 4. 修复/未修复

### 已完成

- 真实 Shopping API Stub 验收：Qwen opt-in → proposal → M5 → R1/M6.1 → final；
- 模型不可用 503 且 M5 event delta=0；
- 候选过滤诊断已可记录 grounded/RRF/role/hard/final；
- 发现并记录 M5 hard constraint 未进入 eligibility 的缺陷；
- 已识别 `MATCH=4/3 vs Final=5` 是 UNKNOWN 保留与后续显式类型收敛，不是 role bug。

### 未修复

- M5 `price_max` 在 stub API 路径中未进入 hard filter；R1 constraints 为空；
- 真实 Qwen SFT-v2 API 未运行（无 GPU）；
- API 级同轮幂等性未重复请求验证；
- `accessory_to_main` 只能确认 role 阶段归零；若要进一步确认 hard/M6.1 为空，可在 role
  修复为“有 MATCH”时用另一场景验证，但当前没有合格 MATCH，不应强行制造。

---

## 5. 需要 GPU 验证

- Qwen3.5-9B SFT-v2 model_server 真实加载；
- 第一轮 set_goal + budget + soft preference；
- 第二轮 budget update + soft preference retraction；
- 真实模型 proposal、validator、M5、R1/M6.1、final 与 latency；
- 真实模型 failure/非法 JSON 的 503/拒绝行为。

当前 AutoDL 无 GPU，故标记 **BLOCKED_NO_GPU**；Stub 结果不冒充真实模型。

---

## 6. 回答

1. **为什么 MATCH=4/3 而 Final=5？**
   - 候选层保留 MATCH+UNKNOWN：4 MATCH+1 UNKNOWN/3 MATCH+2 UNKNOWN，所以候选层 final=5；
   - API 最终列表随后只保留 MATCH（3/3），UNKNOWN 被显式类型收敛移除。不是资格 Bug。
2. **3 Accepted → 6 Events 是 Bug 吗？**
   - 不是；3 个状态事实 + 1 个上下文 + 2 个快照 = 6，逐条可追溯。
3. **accessory_to_main 为什么 Final=0？**
   - grounded=10 中 Role MATCH=0（2 MISMATCH、8 UNKNOWN），`select_grounded_recommendations`
     在无 MATCH 时返回空；归零在 Role MATCH 阶段。
4. **默认 Serving V1 是否改变？**
   - 否。
5. **真实 Qwen 是否验证？**
   - 未验证，无 GPU；只有 Stub。

未重新训练、未运行完整 Agent Benchmark、未修改默认 Serving V1、未开始 RL、未 commit/push/deploy。
