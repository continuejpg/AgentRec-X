# Gate B-3 最终定点复核 —— 决策

## 1. 本轮实测（同一 100 例、同一冻结配置，第二次真实执行）

```text
cases_executed = 100 ；modes = {CERTIFIED: 100} ✓
raw_contains_target = 1 ；after_mask_contains_target = 1
generative_topk_hits = {5: 1, 10: 1, 20: 1} ⇒ HR@5/10/20 = 0.01 / 0.01 / 0.01（分母 100 ✓）
总耗时 72.62 s ；逐例 0.232 / 0.296 / 39.822 s（min/中位/max ✓）
engine 构造 0.0 s（Worker 懒启动 ✓）；worker_closed = true ✓
prefix_budget_exhausted = UNAVAILABLE ✗（引擎未暴露该字段 ⇒ 不虚构 ✓）
```

## 2. case 29 的结构性检查（§3 重点）

```text
test_target = 145997 ；history_items = […, 145997]（长度 6 ✓，目标同时是较早的历史项 ✓）
target_in_history = true ✓ ；target_in_seen_mask = true ⚠️ ；target_in_mask_used_for_ranking = false ✓
candidates_before_mask[0] = 145997（**rank 1** ✓）；candidates_after_mask[0] = 145997 ✓
⇒ **未被 Seen Mask 结构性排除** ✓（`excluded_seen` 把正例移出掩码 ✓，与冻结协议一致 ✓）
```

## 3. 与 Gate B-3 的差异（**关键发现** ✗）

| 运行 | 命中 | 说明 |
|---|---|---|
| Gate B-3（首次，归档 `runs/e04b_gate_b3_tiger_stages/` ✓） | **0 / 100** ✗ | 原始实测值，**保留** ✓ |
| 本轮（第二次，归档 `runs/e04b_gate_b3_final_diag/` ✓） | **1 / 100**（case 29，rank 1 ✓） | 同配置、同案例、同引擎路径 ✓ |

⇒ 同一冻结配置下，**候选生成在不同 Worker 实例/运行之间对该案例不完备复现** ✗
（case 29 本轮在**原始候选列表第 1 位**即出现 ✓，首次运行未出现 ✗）。
**已可排除**：身份映射 ✗（0 解析/映射失败 ✓）、Seen Mask 结构性排除 ✗（§2 ✓）、rank 语义 ✗（1-based/唯一/排序 ✓）、
目标不可映射 ✗（全目录单射覆盖 ✓）、未来泄漏 ✗（位置 [5,6] ✓ 严格提前 ✓）、APPROXIMATE 混入 ✗（100% CERTIFIED ✓）。
**未能归因**：训练窗口/预测偏移（**UNKNOWN** ✗，见 `GATE_B3_FINAL_PROTOCOL_CHECK.md` ✓）与
"候选生成跨运行不完全可复现"的机制（**未定位** ✗，`prefix_budget_exhausted` 不可得 ✗）。

## 4. 决策（按 §6 词表）

* 发现"同配置跨运行候选差异"这一**具体现象** ✓，但**不足以**证明 0/100 的因果根因 ✗
  ⇒ **PARTIAL**（与上一轮同级，但证据更强：目标确实可被生成 ✓）。
* **发现目标仅因 Seen Mask 被过滤**：**不成立** ✗（§2 ✓）。量化影响 = 0 例 ✓。
* **排序/截断 Bug**：**未发现** ✓（无最小复现可提供 ✗）。
* **协议一致 + 两入口一致仍 0/100**：**不适用** ✗（本轮为 1/100 ✓；CLI parity 不可执行 ✓）。
* **关键训练元信息缺失**：**成立** ✗ ⇒ **结论上限**：只能说"该模型在**本次**受限队列上实测命中极低（0–1/100）✓，
  且在两次同配置运行间存在候选差异 ✗"，**不得**声称已定位质量根因 ✗，**不得**据 100 例宣称模型总体优劣 ✗。

## 5. 指标处理（不覆盖、不美化 ✓）

* **原始 Gate B-3 指标保留为 0/100** ✓（冻结归档未改 ✓），本轮 1/100 记为**重复运行的辅助观测** ✗
  ⇒ **替换冻结指标须经你明确决定** ✓（本轮不替换 ✓）。
* 其余口径不变：全部预选案例分母 100 ✓、无填充/无借候选/无删失败 ✓。

## 6. 对 CERTIFIED 的措辞约束（§5 ✓）

CERTIFIED = `branch_and_bound`，其完备性条件是"最优剩余前缀 `S(p) < T`（第 k′ 名已完成项）" ✓；
**在预算受限时是否仍构成数学意义上的精确 top-K′，本轮未获证据** ✗（`prefix_budget_exhausted` 不可得 ✗）
⇒ 报告中**不得**无条件宣称所有返回结果都是可认证的精确 Top-K ✗。

## 7. 时间口径（§5 ✓）

| 数字 | 区间与生命周期 |
|---|---|
| 177.73 s | Gate B-3 **首例**：Worker spawn + checkpoint/481 MB 特征加载 + 首次认证搜索 + 1 例 ✓（冷启动 ✓） |
| 59.9 s | Gate B-3 **阶段 5 总耗时**，**含**上述冷启动 ✓（非 5×单例 ✗） |
| 6.53 / 30.23 s | 阶段 20 / 100：**热推理**（含前缀重复计算 ✓） |
| 72.62 s | 本轮 100 例：热推理总和（max 单例 39.82 s = 首个搜索的预热 ✓） |

## 8. 状态

工作树：1 modified（`docs/ARCHITECTURE.md`）+ 新增本轮 4 份交付物与 1 个新结果目录 ✓；
HEAD `b3af9da79a87aada73cf389ee4c22df5f2169d54` 未变 ✓；资产字节数/哈希未变 ✓；
两个既有失败测试（身份边界契约 ✗、`test_result_satisfies_tiger_source_tool` 替身漂移 ✗）状态**未变、未隐藏** ✓。
**停止**：不自动进入 E-04C、不做参数搜索、不改冻结协议、不重训 ✓。

---

# 9. 结项决定（Gate B-3 CLOSURE，2026-10-15）

技术负责人接受 PARTIAL 并结项。以下为**决定记录**（本轮**仅做文档修订** ✓：无新一轮同案例推理 ✗、
无 beam / prefix_budget / 历史长度搜索 ✗、未调整目标定义 ✗、未覆盖旧结果 ✗、未改生产后端 ✗、
未进入融合或 E-04C ✗）。

| # | 决定 | 落地位置与状态 |
|---|---|---|
| 1 | **保留 Gate B-3 原始冻结指标 `HR@20 = 0/100`**，不覆盖、不重命名、不修改 | `docs/V2_E04B_GATE_B3_TIGER_REPORT.md` §8 与 `runs/e04b_gate_b3_tiger_stages/stage_100.json` **原样保留** ✓（本轮未改一字 ✓） |
| 2 | Final DIAG 复测 **1/100** 作为**独立辅助观测**归档；唯一命中 = **case 29**（合法历史重复商品） | `runs/e04b_gate_b3_final_diag/{GATE_B3_FULL_CANDIDATES.jsonl,final_diag_summary.json}` ✓；case 29：`target_in_history=true`、`candidates_before_mask[0]=145997`、`target_rank_before_mask=1`、`target_rank_after_mask=1` ✓ |
| 3 | Gate B-3 最终状态 = **PARTIAL**：真实生成链路通过 ✓；推荐质量与**跨运行一致性未建立** ✗ | 本节 + `GATE_B3_FINAL_PROTOCOL_CHECK.md` §结论上限 ✓ |
| 4 | `required_frontier` **仅作 k′ 数量预算**，**不作**评价目标 | `contracts.py:356`（`K_max + \|seen(case)\|` ✓）、`cli.py:838-849` ✓；`docs/V2_E04B_GATE_B3_ZERO_HIT_DIAG.md` §3.1/§7 ✓ |
| 5 | **训练历史窗口、目标偏移、截断方向保持 UNKNOWN**；不得声称训练与推理协议完全一致 | `GATE_B3_FINAL_PROTOCOL_CHECK.md` 三项 UNKNOWN ✗ ✓ |
| 6 | CERTIFIED 运行状态**已核查** ✓；**预算耗尽信息不可观测**时**不作**无条件精确 Top-K 保证 ✗ | `prefix_budget_exhausted = UNAVAILABLE` ✗（引擎未暴露 ✓）；§6 措辞约束 ✓ |
| 7 | **两项已有失败测试继续单独披露** | ① `tests/test_backend_boundaries.py::test_t7a_adapter_never_names_canonical_identity`（`known_architecture_contract_failure` ✗）② `tests/test_tiger_engine.py::test_result_satisfies_tiger_source_tool`（**A 类**替身/预期漂移 ✗）—— 均**未修复、未放宽、未隐藏** ✓ |
| 8 | 依据**现有日志**核对 177.73 s 与 59.9 s 的计时口径 | **已核实、无矛盾** ✓（见 §9.1 ✓）—— 未启动任何新推理 ✓ |

## 9.1 §8 计时口径核对结果（只用既有日志 ✓）

```text
smoke 调用（runs/e04b_gate_b3_tiger/，目录创建 20:52:02 ✓）：
  engine 构造 = 0.0 s（Worker 懒启动 ✓）；case 1 = 177.73 s；stage_1 总耗时 = 177.73 s（同一区间 ✓）
stages 调用（runs/e04b_gate_b3_tiger_stages/，目录创建 20:53:53 ✓——**另一次独立调用** ✗）：
  stage_5 总耗时 = 59.9 s，其中**首个案例 = 58.83 s**、第 2 例 = 0.23 s ⇒ 59.9 s ≈ 该次调用自身的冷启动 + 5 例 ✓
  stage_100 总耗时 = 30.23 s，逐例中位 = 0.27 s ⇒ 纯热推理 ✓
```

**核实结论** ✓：两个数字**不是同一区间** ✗ —— 177.73 s 与 58.83 s 是**两次不同调用各自的首次调用冷启动** ✓
（相差约 3 倍，属**已观测事实** ✓，本轮**不**对其成因下结论 ✗）；59.9 s 是**第二次调用**的阶段总耗时且**含其自身冷启动** ✓。
⇒ 三者**不可相加**、也不可当作同一 Worker 生命周期 ✗；此前报告中的表述与此一致 ✓（无矛盾需标注 ✓）。

## 9.2 结项状态

```text
最终判定：Gate B-3 结项；**TIGER 暂不满足推荐质量准入条件** ✓
可准入的部分：真实加载 ✓ / 原生 CERTIFIED 生成 ✓ / SID→真实商品可靠映射 ✓ / 目标隔离与同案例对齐 ✓
不可准入的部分：推荐质量未建立（0–1/100）✗ / 跨运行候选一致性未建立 ✗ / 训练侧协议 UNKNOWN ✗
本轮之后：不再建立新的诊断 Gate ✗；不进行同案例重推理 ✗；不进入 TIGER 融合或 E-04C ✗
```
