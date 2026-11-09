# P3C-1 执行链路修复报告

## 1. 实际修复

### Fix 1：规则目标字符省略时的角色切换

`recommendation/demo/agent_service.py::_update_role_target`

原逻辑只在 `target.type is not None` 时更新；用户说 “show me a replacement mouthpiece instead.”
时 `infer_target` 得到 `type=None, role=accessory`，旧主商品目标没有被标记变化，R1 错误走
RERANK。

修复：当 `target.type is None` 且 `target.role == ACCESSORY`，保留 previous type，仅把
current target 改为 `ProductTarget(previous.type, ACCESSORY)`。这是通用目标/角色语义修复，
不是 dev 任务特判。

效果（受控）：

- RULE_APPLIED `main_to_accessory`：修复前 R1=RERANK，最终仍主商品；
- 修复后 R1=**RETRIEVE**（`grounded_role_target_changed`），最终商品变化。

### Fix 2：规则撤回的通用匹配

`recommendation/demo/agent_service.py::_retractions_from_extraction`

- 规则 extractor 对 “clear it.” 生成 `value="it"` 的无 kind removal；
- 修复：当 selector 是指代代词且当前消息含 budget/price/spend/cost/maximum/ceiling
  词汇时，映射为 `PreferenceKind.PRICE_MAX` 的 kind retraction；
- 不恢复 superseded 状态，由既有 M5 reducer 负责。

`recommendation/state/shopping_adapter.py::resolve_retraction`

- 增加 `weight/lighter/lighter option/lightest/lightweight` 的唯一 feature belief 解析；
- 仅当唯一匹配时解析，仍拒绝歧义，不猜测多个 kind。

效果（受控）：

- RULE_APPLIED `retract_budget`：修复前 `retraction_match resolved=0`、`hard_constraint_count=1`；
- 修复后 `resolved` 有效，`hard_constraint_count=0`；
- 旧 budget 不会复活，final 变化。

### Fix 3：M6.1 soft preference 消费

`recommendation/demo/soft_preference_ranking.py::_feature_to_soft`

- 规则 extractor 产生 `feature="lighter option"`，原逻辑只识别 `feature="weight"`；
- 修复：只对明确的轻量语义集合映射为 `weight/lower`，其余 feature 仍不猜测。

`recommendation/demo/agent_service.py` 新鲜检索返回路径

- 在 `final = _attach_intent_and_reasons(...)` 后增加 `_m5_soft_rerank` 调用；
- 之前只有候选复用/澄清分支调用 M6.1 soft rerank，fresh retrieval 不会消费 M5 soft。

效果（受控）：
- RULE_APPLIED/LLM_APPLIED `confirm_weight`：修复后 `rerank_applied=True`、
  `consumed_soft_preference_count=1`，不再是 `soft diagnostics=None`。

---

## 2. 修复前后受控结果

完整表：`P3C1_BEFORE_AFTER.csv`（18 行，修复前后均在干净状态 DB 上复跑）。

关键对比：

| profile / scenario | before | after |
|---|---|---|
| RULE retract_budget | hard=1, resolved=0 | **hard=0**, final changed |
| RULE main→accessory | R1=RERANK, 仍主商品 | **R1=RETRIEVE**, final changed |
| RULE confirm_weight | soft diagnostics=None | **rerank_applied=True, consumed=1** |
| LLM confirm_weight | soft diagnostics=None | **rerank_applied=True, consumed=1** |
| LLM/规则 budget | R1 变化 | 保持 RETRIEVE，状态正确 |

`DEFAULT` 保持 M5 off、R1 off、soft off，不改变默认行为。

---

## 3. 配件 → 主商品空候选

`P3C1_CANDIDATE_DROP_TRACE.csv`：修复后该场景仍是：

```text
RULE_APPLIED: grounded_candidate_pool=10, final_ids=[], R1=RETRIEVE
LLM_APPLIED: grounded_candidate_pool=10, final_ids=[], R1=RETRIEVE
```

已确认的层级只是：

```text
Query -> ... -> grounded candidate pool (10) -> final (0)
```

`raw_recall`、`RRF`、`role eligibility`、`hard filter` 的逐级计数当前服务没有单独持久化，
因此精确归零步骤标记为 **UNKNOWN_BETWEEN_GROUNDED_AND_FINAL**，不猜测；也没有用
Gold ASIN 或放宽角色验证来补候选。

---

## 4. 测试

新增 `tests/test_p3c1_state_consumption.py`：

```text
3 passed
```

覆盖：

- accessory request 省略 type 时保留 parent type、切换 role；
- budget pronoun removal 映射为 price retraction；
- “lighter option” 映射为 weight-lower soft preference。

---

## 5. 仍未解决

1. **配件→主商品仍为空**：状态与 R1 已变化，但 grounded pool 10 到 final 0，具体是角色
   资格、证据还是最终过滤无法由现有 diagnostics 区分；需要新增逐层计数。
2. **soft rerank 只修复了已识别路径**：fresh retrieval 分支已调用，但仍需在完整 20 dev
   任务上验证是否转化为任务成功（本阶段未重跑大 Benchmark）。
3. **默认 API 仍不启用 M5/R1/soft/LLM**：修复只在实验 profile 生效；默认 Serving V1 不变。
4. **Agent RL 未启动**。

---

## 6. 文件

- `P3C1_BEFORE_AFTER.csv`
- `P3C1_CANDIDATE_DROP_TRACE.csv`
- `P3C1_SOURCE_DIFF.txt`
- `tests/test_p3c1_state_consumption.py`

未修改 Serving V1、Qwen 权重、SFT 数据、Gold/Evaluator V2/frozen benchmark；未 commit/push/deploy。
