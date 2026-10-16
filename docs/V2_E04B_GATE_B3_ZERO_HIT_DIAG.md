# AgentRec-X V2 / E-04B-2 Gate B-3-DIAG — TIGER 零命中与目标/排名契约诊断

> 结论：**PARTIAL** —— 已**排除**身份、映射、时序、排名与认证状态导致的零命中 ✓；
> 但**未能**在现有归档下证明零命中的因果根因 ✗（缺逐例候选 ID 清单；训练窗口契约无法由现有工件证明 ✗）。
> 因此：**Gate B-3 的 0/100 仍为当前实测值** ✓，**不得**据此宣称"模型质量低下已获证明" ✗。
> 本轮**未**重启 Worker（无需重算已存指标 ✓，且归档不含候选清单 ⇒ 双目标复核不可执行 ✓）；
> **未**加载任何模型权重、**未**联网/下载、**未**安装依赖、**未**训练、**未**改 checkpoint/生产后端/评测协议、
> **未**执行 Git 写操作 ✓。

---

## 1. 开始时工作树（复核）

```text
branch : v2/e04a-pass-checkpoint   commit : b3af9da79a87aada73cf389ee4c22df5f2169d54（未变 ✓）
status : 1 个已跟踪修改（docs/ARCHITECTURE.md）+ 19 个未跟踪文件 ✓（与上轮报告一致 ✓）
```

## 2. 归档是否足够（§2 四问）

| 问 | 答 |
|---|---|
| 1. 是否保存 100 例的完整合法 Top-K 商品 ID 与 rank？ | **否** ✗ —— `per_case` 仅存 `{case, mode, generated, valid_candidates, rank, seconds, timeout_risk}`（`runs/e04b_gate_b3_tiger_stages/stage_100.json` ✓）；**rank 仅在被命中时非空** ✓，**候选 ID 列表未落盘** ✗。B-1 同理仅存 `{case, rank, returned}` ✗。 |
| 2. 是否保存候选数 / SID 过滤数 / 目标与输入历史摘要？ | **计数：是** ✓（§5 报告的表 ✓）；**可核对的输入/候选摘要：否** ✗。 |
| 3. 能否只读归档重算 HR/NDCG？ | **能重算正式指标** ✓（`rank` + 命中数 ✓ 已在归档 ✓），**不能**做任何按候选集合展开的分析 ✗。 |
| 4. 哪些诊断必须重载模型？ | 只有"对**同一份预测**换目标复核"这类需要候选 ID 的分析 ✗ —— 而它已被 §3.1 判定为**不适用** ✓（见下），故本轮**无需**重载 ✓。 |

## 3. 第一项诊断：训练目标与测试目标（§3）

| 概念 | 定义 / 构造 | 位置（file:line） | 时间边界 |
|---|---|---|---|
| `test_history` | `items[:-1]`（训练前缀 + 验证项） | `recommendation/evaluation/split.py:97,112`；本测试断言 ✓ | 严格早于 test 交互 ✓ |
| `test_target` | `items[-1]`（最后一次交互） | `split.py`（`EvaluationCase.test_target` ✓） | 单正例标签 ✓ |
| `required_frontier` | **AgentRec-X 自算的 k′ 预算 ＝ `K_max + |seen(case)|`** ✓ | `backends/.../contracts.py:356` ✓；消费 `cli.py:838-849`（`enough = all(count >= want …)` ✓、`k_prime_field: "required_frontier"` ✓） | 与标签无关 ✓ |
| TIGER 训练输入 / 监督 | 后端侧构造；本仓可核对的是**生成器布局**（`generator_layout.json`：code_space 1024、codebook 256、3 levels、`level_offsets [0,256,512,768]`、special pad 1024 / bos 1025 / eos 1026 / **sep 1027**、`distinct_sid_tokens 850` ✓） | `runs/tiger_public_2026/checkpoint/generator_layout.json` ✓ | — |
| 推理截断/窗口 | L2 `max_hist_items=20`（原生默认 ✓）+ BOS/history/SEP 前缀约定 ✓ | `tiger_engine.py:130-142` ✓、`generator_layout.json:audit` ✓ | — |
| 生成的 SID 代表什么 | 目录受限的下一位置生成结果 ✓（trie 约束 ✓，分数＝SID 序列 log-prob ✓ `tiger_source.py` `score_kind` ✓） | `retrieve.py` ✓、`tiger_source.py` ✓ | — |

### 3.1 `required_frontier` 的真实含义（**不是**推荐标签 ✓）

**硬证据**：`contracts.py:356` 原文 —— "``required_frontier`` is AgentRec-X's own computation
(**`K_max + |seen(case)|`**) and is …" ✓；`cli.py:838-849` 把它当作**数量下界**校验 ✓
（`enough = all(count >= want …)` ✓，字段名 `k_prime_field` ✓）。实测值吻合公式 ✓：
`case_id 0 → 50 = 20 + 30`（history len 30 ✓）、`case_id 1 → 36 = 20 + 16` ✓✓。
⇒ 它是**交接边界/k′ 预算**，用于"让被 seen 掩码削减后的 top-K 仍能存活" ✓（worker 自述 ✓ `serve_worker.py:41` ✓），
**既不是**训练监督目标 ✗，**也不是**可评价的推荐目标 ✗。⇒ 按 §7：**NOT_APPLICABLE** ✓ ——
**不计算**"required_frontier HR" ✓（且交接队列 `case_id` 为 0/1… 的**抽样**队列 ✓，与本协议 100 例的
`user_int_id 1..100` **不构成同一清单** ✗，故亦无法据其构造合法对照 ✓）。

### 3.2 原生训练协议（可核对与不可核对，均如实登记）

* **可核对** ✓：生成器词表与特殊符号（pad/bos/eos/**sep** ✓）、3 级 × codebook 256 ✓、level offsets ✓、
  SID 去重方案与全目录覆盖 ✓、`score_rule.json`（softmax / child renormalisation / `eos_in_score` ✓）。
* **不可由现有工件证明** ✗：训练时的**历史窗口长度**与**预测偏移量**（`checkpoint/tiger.json` 中未含
  序列长度类键 ✓；本轮未做参数搜索 ✓）。⇒ `max_hist_items=20` 是**原生默认值** ✓，
  它与训练窗口是否逐字一致**尚无证据** ✗ —— 登记为**未闭合风险** ✓，但**不足以解释 0/100** ✗
  （截断只会降低命中率，不会把 2,720 个候选全部排除在目标之外 ✓）。

### 3.3 目标映射（§3.3）

* 100 个 `test_target` 均落在 `mappings.item2id` 的 1..156,746 命名空间内 ✓；
* SID 工件**全目录覆盖** ✓：`assignment` 列表 156,747 项（含索引 0 哨兵 ✓）、
  去重后 `distinct_sids = 156746 = num_items` ✓、`collision_groups = 0` ✓（本仓测试已锁定 ✓）；
  ⇒ **每个真实商品（含 100 个目标）都存在且仅存在一个合法 SID** ✓；
* 运行时一致性 ✓：100 例中 `sid_parse_failures = 0`、`sid_mapping_failures = 0`、
  `invalid_ids = 0`、`pad_ids = 0` ✓（2,720 个生成项全部可解析、可映射、合法 ✓）。
* ⇒ **不存在**身份或数据版本错位导致的"目标不可映射" ✗（因此零命中**不能**归因于此 ✓）。

## 4. 第二项诊断：未来目标泄漏与重复交互（§4）

* 唯一重复案例 = **case 29**：目标商品在 `item_ids` 中的位置为 **[5, 6]**（共 7 项，test 位置 = 6 ✓）
  ⇒ 较早一次交互**严格早于**测试交互 ✓，其出现在合法历史中属**真实历史可见** ✓。
* 分类：**`target_item_previously_in_legal_history`** ✓ ；**不是** `future_test_interaction_injected` ✗。
* 强断言（已加入并通过 ✓，`tests/test_v2_e04b_gate_b3.py`）：`test_history == items[:-1]` ✓、
  `test_target == items[-1]` ✓、且**非法构造**（把 target 追加进输入）与合法输入**可区分** ✓
  ⇒ 任何"非法追加未来交互"都会使断言失败 ✓（不再是弱断言 `0 ≤ count ≤ cases` ✗）。
* 三模型历史源一致性 ✓：三者使用**同一** artifact、**同一** `split_sequence` 与**同一**案例顺序 ✓
  （case_id 前缀比对 ✓）。
* ⚠️ **统计定义不可横向比较** ✗：B-1 的 `target_ids_fed_to_model`（0）与 B-3 的（1）**定义不同** ✗
  （B-1 计入"target 同时出现在**掩码**中"的重复项 ✓；B-3 计入"target 出现在 `test_history`" ✓）⇒
  报告中**不得**把两个数值直接对比 ✓。

## 5. 第三项诊断：Top-K 排序与 rank 语义（§5）

* 引擎**校验** worker 返回的 rank：必须为 1-based 整数 ✓、同一响应内不得重复 ✓
  （`tiger_engine.py:811-816` ✓），随后**按 rank 排序**输出 ✓（`:848` ✓）。
* 本轮 B-3 指标**不依赖** worker 的 rank ✗，而用**协议掩码后的候选顺序**（`excluded_seen` ✓）自行赋 rank ✓
  —— 与冻结协议一致 ✓，且对 HR@K 只会**更严格** ✓（过滤只移除非目标项 ✓）。
* 既有失败 `test_result_satisfies_tiger_source_tool`：期望 `[("B0TRUSTED03",1,-1.25), ("B0TRUSTED01",2,-2.5), …]`
  而实际得到 `[('B0TRUSTED02', 1, -3.75)]` ✗ —— 该处使用的是 **`FakeTransport` 测试替身** ✓（非真实 worker ✗），
  且引擎对真实响应的 rank 校验/排序在本轮 100 例上**未出现任何契约错误** ✓（0 例 rank 异常 ✓）。
  ⇒ 分类：**A（测试预期与替身数据漂移）** ✓ —— 属**既有**问题 ✗，**不是**生产候选排序违约 ✗；
  本轮**未**修复、**未**放宽 ✓。
* **rank 语义无法解释 0/100** ✓：2,720 个候选中目标**从未出现** ✗ —— 排名偏置只会改变命中名次，
  不会把命中数变成 0 ✓。

## 6. 第四项诊断：CERTIFIED 的实际保证（§6）

* 代码事实 ✓（`retrieve.py:11-28,58-88`）：`CERTIFIED = "branch_and_bound"` ✓；剪枝条件是
  "best remaining prefix has `S(p) < T`，其中 `T` 是当前第 k′-th 个**已完成**项的分数；
  此后**任何未访问项都无法进入认证前沿**" ✓；并且其正确性被声明为**可用小规模暴搜 oracle 机械验证** ✓。
* ⇒ 属于**在该模型自身评分规则下的精确 top-k′** ✓（不是"全局最优推荐质量"的证明 ✗）。
* **未核实** ✗：搜索预算耗尽时的返回路径/是否存在"截断但仍标 CERTIFIED"的分支 —— 本轮**未**在源码中
  定位到该分支证据 ✗（登记为未闭合风险 ✓）；运行日志层面 100 例**全部** `CERTIFIED` ✓ 且候选恒 ≥20 ✓。

## 7. 第五项诊断：双目标探索性核对 —— **NOT_APPLICABLE**（§7）

`required_frontier` 已被证明为 **k′ 预算**而非标签 ✓（§3.1 ✓）⇒ 按 §7 规定**不构造**该指标 ✓，
**不重跑模型** ✓，**不做**任何参数搜索 ✓。因此本报告**没有**"required_frontier HR"数字 ✓（不强行生成 ✓）。

## 8. 时间成本口径核对（§8，无需重跑 ✓）

| 数字 | 实际计时区间 | Worker 生命周期 |
|---|---|---|
| `load.seconds = 0.0` | 仅**引擎对象构造**（Worker 懒启动 ✓） | 尚未 spawn ✓ |
| 案例 1 = **177.73 s** | **冷启动**：spawn + checkpoint/481 MB 特征加载 + 首次认证搜索 + 1 例 ✓ | 启动并常驻 ✓ |
| 阶段 5 = **59.9 s** | 阶段总耗时（**含**冷启动 ✓，故非 5×单例 ✗） | 同一 Worker 复用 ✓ |
| 阶段 20 = **6.53 s** | 纯热推理（含案例 1–5 重复计算 ✓） | 复用 ✓ |
| 阶段 100 = **30.23 s** | 纯热推理（含前 20 例重复 ✓） | 复用 ✓ |
| 逐例热推理 = **0.22–0.67 s** | 单例（`recommend` 调用 ✓） | 复用 ✓ |

⇒ 5/20/100 为**嵌套前缀** ✓，各阶段耗时含重复计算 ✓；**不得**相加为 125 个独立用户或"总时长" ✗。

## 9. 十二问结论（§9）

| # | 问题 | 结论 |
|---|---|---|
| 1 | 0/100 是否已可解释？ | **未获因果根因** ✗（身份/时序/排名/认证均已排除 ✓，训练窗口契约无法由工件证明 ✗） |
| 2 | `test_target` 与 TIGER 训练目标是否一致？ | **未能证明** ✗（训练窗口/偏移不可由现有工件核对 ✓；登记为未闭合风险） |
| 3 | `required_frontier` 是什么？ | **k′ 预算 = `K_max + |seen|`** ✓（非标签、非监督目标 ✓） |
| 4 | 100 个目标是否都可合法映射为 SID？ | **是** ✓（全目录单射覆盖 + 运行时 0 解析/映射失败 ✓） |
| 5 | 历史截断是否与训练一致？ | **未证明** ✗（`max_hist_items=20` 为原生默认 ✓，非按命中率选择 ✓） |
| 6 | 是否存在真实未来交互泄漏？ | **否** ✓（case 29 位置 [5,6]，较早交互严格提前 ✓；负向测试已加入 ✓） |
| 7 | Top-K/rank 契约是否正确？ | **生产路径正确** ✓（1-based ✓、唯一 ✓、按 rank 排序 ✓、0 例违约 ✓）；既有失败属**替身数据漂移** ✗ |
| 8 | CERTIFIED 的真实保证？ | 该模型评分规则下的**精确 top-k′** ✓（≤ 该 k′ 内无未访问项可插入 ✓），**非**质量保证 ✗ |
| 9 | 两种目标差异可否合法计算？ | **NOT_APPLICABLE** ✓（无合法第二标签 ✓） |
| 10 | 是否需要重跑正式评测？ | **不需要** ✓（正式指标分子分母齐全 ✓；重跑只会复现同一归档缺口 ✗） |
| 11 | 0/100 是否仍为当前实测值？ | **是** ✓（保留 ✓，未替换、未美化 ✓） |
| 12 | 工作树/资产/归档是否保持不变？ | **是** ✓（HEAD `b3af9da` ✓；tiger.pt 45,181,927 B、semantic_ids.json 5,775,237 B 未变 ✓；未覆盖任何归档 ✓） |

## 10. 本轮新增与测试状态

* 新增诊断报告 `docs/V2_E04B_GATE_B3_ZERO_HIT_DIAG.md`（本文件 ✓）。
* 更新 `tests/test_v2_e04b_gate_b3.py`：新增**负向**未来注入测试 ✓ → 该文件 **6 passed** ✓
  （`test_the_model_input_is_the_frozen_history_and_never_a_future_interaction` ✓）。
* **未修复、未放宽**的既有失败（继续单独披露 ✗）：
  ① `tests/test_backend_boundaries.py::test_t7a_adapter_never_names_canonical_identity`
  （`known_architecture_contract_failure` ✓）；
  ② `tests/test_tiger_engine.py::test_result_satisfies_tiger_source_tool`（**A 类**替身/预期漂移 ✓，既有 ✗）。
* 本轮**未**新增结果目录、**未**改动生产代码、**未**执行 Git 写操作 ✓。

## 11. 结论（§9 词表）

**PARTIAL** —— 已发现并**验证**了关键契约事实（`required_frontier` 的 k′ 语义 ✓、身份/时序/排名/认证的排除 ✓），
但**不足以**证明 0/100 的因果根因 ✗。**明确声明**：诊断未发现错误 **≠** 模型质量低下已获证明 ✗；
正式正式指标仍为 **0/100（分母 100，全部预选案例口径 ✓）** ✓。
若需闭合根因，下一步只能是**新的、预先冻结**的最小实验（例如在同一 100 例上、以**同一**模型与解码配置，
记录逐例候选 ID 与 worker 原始 rank ✓，并用后端自身的 `cli.py` 评测路径在同一队列上做一致性对照 ✓）——
**不得**做 beam/prefix_budget/max_hist_items 参数搜索 ✗。**停止点**：未改模型、未重训、未调参、
未改冻结协议、未进入融合或 E-04C ✓。
