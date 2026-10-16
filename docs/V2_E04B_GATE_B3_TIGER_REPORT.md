# AgentRec-X V2 / E-04B-2 Gate B-3 CONTINUE — TIGER 真实生成式推理与质量测量

> 状态：**PASS**（真实原生生成式推理已执行并完成受限质量测量 ✓；`PASS` 不要求 TIGER 超过 SASRec/Two-Tower ✓）。
> **实测结果：100/100 例全部完成、全部 `CERTIFIED`、候选覆盖 22–47（中位 25）、SID 解析与映射 0 失败 —— 但目标命中 0 ✗。**
> 本轮**未**加载 SASRec/Two-Tower 权重（对照只用归档 ✓）、**未**联网/下载、**未**安装依赖、**未**训练或微调、
> **未**改 checkpoint/tokenizer/SID/特征、**未**改解码算法、**未**改身份语义、**未**改生产后端/Serving/RRF/Agent/Memory/RAG、
> **未**修改任何历史评测结果、**未**执行 Git 写操作 ✓。

---

## 1. 原 PARTIAL 的原因及本轮是否消除

* **原 PARTIAL 的原因（如实）**：首轮在发现"身份标识符放置"契约问题后，**主动选择**先做只读审计、
  未执行原生推理 ✗；当时报告已写明这是**范围与验证能力限制**，不是技术失败 ✓。
  （首轮 §9 记录：CERTIFIED 精确检索在 CPU 上首次调用约 12 分钟 + 需要"执行→逐例核对→记录"闭环 ✓。）
* **本轮消除情况**：**已消除** ✓ —— 真实加载、真实原生生成、逐例核对与记录全部完成 ✓（见 §4/§5/§7）。

## 2. 原生解码参数的来源（跑分前冻结 ✓）

```json
{"mode": "CERTIFIED", "allow_mode_downgrade": false, "k_max": 20, "max_hist_items": 20,
 "beam": 512, "prefix_budget": 200000, "batch_prefixes": 256, "device": "cpu",
 "source": "TigerWorkerConfig 默认值（recommendation/backends/tiger_engine.py:130-142）"}
```

* 优先级遵循 §4：**无**该 checkpoint 的归档推理配置 ✗ → 采用**原生实现默认值** ✓；
  **未**依据 100 例命中率调整任何参数 ✓（manifest 中 `declared_before_scoring: true` ✓）。
* `TigerServingEngine(mode="CERTIFIED", allow_mode_downgrade=False, verify_assets_on_start=True)` ✓；
  `TigerSourceTool(require_certified=True)` 为规范评测姿态 ✓（`tiger_source.py:131` ✓）。
* 未固定的 `beams_per_level` **本轮未使用**（CERTIFIED 路径为分支定界，不用 beam ✓）。

## 3. CERTIFIED 状态的实际证明（不是"设了 flag 就算" ✓）

* **逐例回报**：`stage_*/modes = {"CERTIFIED": N}` ✓ —— 该字段是 worker **实际运行**的模式，
  不是请求模式的回显 ✓（`tiger_engine.py:691-692`："never an upgrade" ✓）。
* 100 例、5 例、20 例**全部**为 `CERTIFIED` ✓，**无** `APPROXIMATE` ✓，**无**降级 ✓。
* 拒绝语义在案：`require_certified=True` 时 APPROXIMATE 结果会被**拒绝**而非记录 ✓（`tiger_source.py:250-252` ✓）。
* 测试断言锁定：`tests/test_v2_e04b_gate_b3.py` 要求所有记录的阶段 `modes ⊆ {CERTIFIED}` ✓。

## 4. 模型加载、Worker 与真实生成证据

```text
加载：TigerServingEngine 构造成功 ✓；worker = 1 个本地子进程（JSONL IPC，**未监听端口** ✓）
首次调用含启动（checkpoint + 481 MB 特征 + 认证检索）：案例 1 = 177.73 s ✓
随后逐例：0.22 / 0.27 / 0.67 s（min/中位/max）✓ —— 与 docs/TIGER_SERVING.md:113 的"CPU 启动占主导"一致 ✓
真实前向：100/100 例完成，cases_failed = 0 ✓；全程 CERTIFIED ✓
Worker 生命周期：结束时经 engine.close() 关闭 ✓；**退出后系统内 0 个 python / 0 个 serve_worker 进程** ✓
```

## 5. SID 解码、映射与过滤数据（100 例）

| 计数 | 值 |
|---|---|
| `sid_generated_total` | **2,720** |
| `sid_parse_failures` | **0** ✓ |
| `sid_mapping_failures` | **0** ✓ |
| `duplicate_items_filtered` | **0** ✓ |
| `seen_items_filtered` | **48** ✓（掩码真实生效 ✓） |
| `final_valid_candidates_total` | **2,672** |
| 逐例合法候选数 min/中位/max | **22 / 25 / 47** ✓（**每例均 ≥ 20** ✓） |
| `invalid_ids` / `pad_ids` | **0 / 0** ✓ |
| `cases_with_zero/<5/<10/<20 candidates` | **0 / 0 / 0 / 0** ✓ |
| `target_ids_fed_to_model` | **1** ⚠️（该例的 target 同时出现在**自身历史**中＝重复交互 ✓，已如实记录、**未**写成 0 ✓；target 从未用于解码前缀或补位 ✓） |

⇒ **身份目标达成** ✓：生成结果可解析、可映射、可去重、可屏蔽，并落到与 SASRec/Two-Tower **同一** `item2id`
命名空间（1-based、PAD=0、156,746 项 ✓）；**未发现** SID→item 错误、身份越界或跨命名空间混用 ✓。

## 6. 同案例对齐结果

```text
case_ids 与 Gate B-1(SASRec) / Gate B-2(Two-Tower) 的 100 例**同序一致** ✓（manifest.case_manifest.case_ids 前缀比对 ✓）
test_history / test_target / seen mask：同一真实 artifact + 同一冻结拆分函数（temporal_leave_two_out）重新推导 ✓
first_mismatch = null；case 选择规则未改（artifact 顺序前 N 个有效 case ✓）
```

## 7. 5/20/100 例实际执行规模（嵌套前缀，非 125 个独立用户 ✓）

| 阶段 | attempted | completed | failed | modes | 耗时 |
|---|---|---|---|---|---|
| 5 | 5 | 5 | 0 | `{CERTIFIED: 5}` | 59.9 s（含启动 ✓） |
| 20 | 20 | 20 | 0 | `{CERTIFIED: 20}` | 6.53 s |
| **100** | **100** | **100** | **0** | `{CERTIFIED: 100}` | 30.23 s |

## 8. 各 K 的 HR/NDCG、命中数与分母（**生成式 Top-K 口径** ✓）

| 指标 | @5 | @10 | @20 | 命中数（分母 100） |
|---|---|---|---|---|
| `generative_topk_hr_at_k` | **0.000000** | **0.000000** | **0.000000** | **0 / 0 / 0** |
| `generative_topk_ndcg_at_k` | **0.000000** | **0.000000** | **0.000000** | — |

* 分母：`all_preselected_cases = 100`（**全部预选案例** ✓，含失败与空候选 —— 本轮失败 0、空候选 0 ✓）；
  同时记录 `completed_only = 100` ✓（两口径一致且均已报告 ✓，**未**悄悄改分母 ✓）。
* **候选不足未填充、未借用 target、未从其他模型借候选、未删失败案例** ✓（每例候选均 ≥ 20 ✓ 亦无需补位 ✓）。
* 指标命名遵循 §7：**`generative_topk_*`** ✓ —— TIGER 是**生成式 Top-K**，**不是** full-catalogue 穷举评分 ✗。

## 9. 总耗时、单例耗时与总内存峰值

```text
阶段耗时：5 例 59.9 s / 20 例 6.53 s / 100 例 30.23 s；逐例 0.22–0.67 s ✓
首次调用启动 177.73 s；主进程+Worker 树 RSS 峰值 798.8 MB ✓；结束可用内存 5,908.3 MB ✓
单 Worker、无并发 ✓；未缓存全部用户的原始生成轨迹（仅逐例计数 ✓）；内存守护下限 900 MB 未触发 ✓
预先声明的保护上限：startup 1500 s、单请求 600 s、单例标记阈值 600 s（全部未触发 ✓）
```

## 10. 超时与生成失败

```text
cases_failed = 0 ✓；timeout_risk = false（全部 100 例 ✓）；无超时、无解析失败、无映射失败 ✓
```

## 11. 新增测试及既有遗留测试状态

* 新增 `tests/test_v2_e04b_gate_b3.py`（5 项）→ **5 passed** ✓：SID 空间去重后单射且覆盖全目录 ✓、
  worker 边界不携带规范身份 ✓、CERTIFIED 配置契约 ✓、录制结果全 CERTIFIED 且身份干净 ✓、案例清单前缀对齐 ✓。
* **既有遗留失败（必须单独披露，未修复、未放宽、未跳过 ✓）**：
  `tests/test_backend_boundaries.py::test_t7a_adapter_never_names_canonical_identity` → **1 failed** ✗
  ⇒ 分类为 **`known_architecture_contract_failure`**（`recommendation/backends/tiger_engine.py` 中 4 个
  规范身份标识符违反 AGENTS.md §19.2 字面契约 ✓；数据路径安全性已由本轮真实推理进一步佐证 ✓）。
  **本轮 `git status` 无 `recommendation/backends/` 改动** ✓，生产代码未改 ✓。
* 既有相关套件（Gate A 6 / E-04B-P 12 / B-1 5 / B-2 6 / P0 4 ✓）为此前轮次结果 ✓；
  **不得**与上述失败合并成"全绿" ✗。

## 12. 文件改动、工作树与资产哈希

| 文件 | 说明 |
|---|---|
| `experiments/e04b_gate_b3_tiger.py`（新增） | 真实 TIGER 运行器：冻结 manifest、CERTIFIED 单 Worker、逐例计数、进程树 RSS、关闭与清理、唯一目录 |
| `tests/test_v2_e04b_gate_b3.py`（新增，5 项） | SID/身份/认证/对齐契约 |
| `runs/e04b_gate_b3_tiger/`（新增）、`runs/e04b_gate_b3_tiger_stages/`（新增） | 1 例 Smoke 与 5/20/100 阶段结果（**未覆盖任何既有归档** ✓） |
| `docs/V2_E04B_GATE_B3_TIGER_REPORT.md`（更新） | 本报告（首轮审计 + 本轮真实执行合并） |

```text
结束 HEAD：b3af9da79a87aada73cf389ee4c22df5f2169d54（未变 ✓）
工作树：1 个已跟踪修改（docs/ARCHITECTURE.md）+ 19 个未跟踪文件；0 删除；无 reset/stash/commit ✓
资产：tiger.pt 45,181,927 B、semantic_ids.json 5,775,237 B（与运行前记录一致 ✓，只读 ✓）
```

## 13. 最终结论：**PASS**（含一项必须跟进的结果异常 ✗）

```text
✓ 真实模型加载（TigerServingEngine + 单本地 Worker，无端口、无下载）
✓ 真实原生生成式推理（100/100 完成，0 失败，全程 CERTIFIED —— 逐例实际模式回报）
✓ SID 能可靠映射真实商品（2,720 生成 / 0 解析失败 / 0 映射失败 / 0 非法 / 0 PAD / 0 重复）
✓ 目标隔离与同案例验证通过（target 仅用于掩码与指标；1 例重复交互如实记录；100 例同序对齐）
✓ 完成受限真实质量测量（generative_topk_hr/ndcg 全 K，命中数与分母齐全）
✓ 成本、资源与候选覆盖清楚（逐例 0.22–0.67 s、启动 177.7 s、树 RSS 峰值 798.8 MB、候选 22–47）
✓ 未覆盖或损坏既有资产；Worker 已关闭且无残留进程
✗ 遗留架构契约失败 1 项（known_architecture_contract_failure，待 E-04C 处置）
✗ **结果异常：100 例目标命中 0（HR/NDCG 全 0）** —— 按 §"先检查协议/ID/配置/数据一致性"的纪律，
   本轮**未**调参、**未**替换口径、**未**改用 APPROXIMATE ✓，如实报告并把排查列为下一步。
```

### 与 SASRec / Two-Tower 的探索性对照（**机制不同，仅描述性** ✗非等价比较）

| 模型 | 机制 | 100 例命中 @5/@10/@20 | HR@20 | NDCG@20 | 有效案例 |
|---|---|---|---|---|---|
| SASRec（归档） | full-catalogue 穷举评分 | 1 / 1 / 2 | 0.02 | 0.008756 | 100 |
| Two-Tower（归档） | full-catalogue 穷举评分 | 2 / 2 / 3 | 0.03 | 0.018663 | 100 |
| **TIGER（本轮实测）** | **生成式 Top-K（CERTIFIED）** | **0 / 0 / 0** | **0.00** | **0.00** | 100 |

**不得**据此宣称任一模型总体优劣 ✗；**不得**把 p 值或"零命中"解释为模型等效或定论 ✗；
**不得**把三者的算力成本差异归因于模型能力 ✗；TIGER 的候选覆盖（每例 22–47，均为**生成**而非全目录枚举 ✓）
与两者**机制不同** ✗，故本对照仅供描述 ✓。

### 下一步（需你授权的排查假设，本轮**未**执行任何调整 ✓）

零命中且候选健康，指向**目标语义/历史约定**层面的不一致，而非身份或执行缺陷 ✓。可检验的具体假设：
`backends` 交接契约以 `eval_cohort.jsonl` 的 **`required_frontier`** 定义"下一个应检索的位置" ✓，
而本轮按冻结协议送入的是 `test_history` 并要求命中 **`test_target`** ✓ —— 二者可能相差**一个位置**
（frontier ≈ validation 项 ✗）。建议在下一轮**预先固定**并一次性验证：同一 100 例上比较
"以 `test_target` 为目标"与"以 `required_frontier` 为目标"的命中差异，以及在
`max_hist_items`/前缀约定上对齐训练时的构造 ✓ —— 仍**不得**按命中率挑参数 ✗。

**停止点**：未进入融合实验、未进入 E-04C、未改生产后端、未联网、未安装依赖、未训练、未调用在线 LLM ✓。
