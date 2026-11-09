# P14-A — Shopping Companion 数据与 Agent 运行链路接入 报告

> 性质：P14 子阶段 A，只做**数据 Adapter + catalogue namespace + 真实商品工具 + Memory 接入 + Planner 执行桥 + 一条真实端到端轨迹**。
> 不做新推荐模型、不训练 Planner、不做 P14-B/C Benchmark、不改旧 Serving V1、不 commit/push、不使用 GPU。
>
> 环境：Windows 11 工作区 + WSL2 Ubuntu-22.04 + 仓库 `.venv`（Python 3.10.12，CPU-only）。
> 本轮**未使用 GPU、未连接/占用 AutoDL**；所有测试为小批次运行（未跑全仓库 pytest）。

## 0. 结论摘要

**P14-A 总判断：LIMITED。**

- **已经真实成立**：官方 Shopping Companion **train** 数据接入（800 个真实任务、stage_1/stage_2 正确配对、oracle 隔离）→ `shopping_companion` 独立 namespace → **81,637 个真实商品**的 limited catalogue + 真实 BM25 `product_search` → 真实 `product_view`/Product Info（UNKNOWN 语义、跨 namespace 拒绝）→ 既有 CandidateLedger/GroundedReasoner/Eligibility → Rule Planner 通过**既有 Planner Action 契约**驱动真实工具 → 真实轨迹 + 独立 evaluator。**一条真实 train 单商品任务完整跑通**，无答案泄漏。
- **没有成立**：官方**对话历史语料未发布**，因此正常的 stage_1 无法恢复真实偏好（Memory 只能诚实报告 `history_unavailable`）；**完整 1,298,797 商品目录未索引**（只做了 6.29% 的真实前缀样本）；Qwen 真实推理未运行（本机无服务，且本轮不启 GPU）；add-on/bundle 只做了结构性 Smoke，未实现组合/优惠优化。

**判定依据（详见 §E5）**：真实历史 + 真实偏好恢复 + 完整目录是 GO 的必要条件；本轮缺少前两项（客观数据阻塞）且目录为 limited，因此 **LIMITED**，不能宣称完整 Agent 能力已验证。

---

## 1. 实际使用的数据版本

| 项 | 值 |
| --- | --- |
| Dataset | `yuzhan2205/Shopping-companion`（revision `main`，经 hf-mirror 访问） |
| 数据卡 | https://huggingface.co/datasets/yuzhan2205/Shopping-companion |
| 官方仓库 | https://github.com/Jarvis33yu/Shopping-Companion |
| 使用的 split | **train only**（`data/train.parquet`） |
| `train.parquet` | 5,943,374 B，sha256 `bc87aad3685da0176df4777d05942c8aa05d4dd4ac3230d6dc96a78f33214377`，1,600 行 |
| **test split** | **未下载、未读取、未运行、未用于选型**（`p14a_prepare_dataset.py` 对 test.parquet 直接拒绝） |
| 商品目录文件 | `data/products.jsonl`（官方 20,030,340,587 B，1,298,797 商品；本轮**只取连续前缀**，见 §3） |
| 本地数据位置 | `data/p14a/`（git-ignored，manifest 记录路径/哈希） |

## 2. Train 数据结构与映射

官方 train 每行是一个 **stage-level instance**；1,600 行 = **800 个购物任务 × 2 个 stage**（不是 1,600 个任务）。

| 事实（已核对） | 值 |
| --- | --- |
| tasks | **800**（`task_id = official extra_info.question_id`，唯一） |
| stage 实例 | stage_1 = 800，stage_2 = 800；每个任务恰好 1+1 |
| question_type | single_product **400** / add_on_deals **400** |
| conversation_id | 400 个；**一个 conversation 携带两个不同任务**（`_1`=single_product，`_2`=add_on_deals），不是 task key |
| stage_1 prompt | 2 条消息：官方 system + 用户当前需求（含 `Current Date: …` 行） |
| stage_2 prompt | 官方 4 条消息：system + 同一用户需求 + **oracle memory block** + "I will not provide you with any more information." |
| **oracle 隔离** | **800/800 个 oracle memory block 从 agent 可见输入中移除**（合计 3,098,889 字符；min 1,264 / max 7,862 / mean 3,873.6），仅保存在 evaluator-only 文件 |
| agent 可见文件 | `data/p14a/dataset/train_tasks.jsonl`（800 行，4,332,995 B，sha256 `8caeee24…11ed074`） |
| evaluator-only 文件 | `data/p14a/dataset/train_eval.jsonl`（800 行，11,702,561 B，sha256 `ee4a9e0b…815f6`，含解析后的 ground_truth + 原始 oracle block + prompt 哈希） |
| 官方 prompt 保真 | stage_1/stage_2 system prompt **逐字节保留**（sha256 已记录在 manifest）；stage_2 system prompt 中的 `@REC::product_id@` 是答案格式说明，不是答案 |

映射实现：`experiments/p14a_prepare_dataset.py`（pyarrow 只在转换工具里使用）→ `recommendation/shopping_companion/dataset.py`（运行时仅 stdlib）。转换后重新打开 agent 可见文件并**硬失败**于任何 oracle/GT/target-id 泄漏。

## 3. 商品目录和索引情况

**这是 LIMITED catalogue，不是完整目录；报告与 trajectory 均带 `limited_catalogue=true`。**

| 项 | 值 |
| --- | --- |
| 官方完整目录 | 1,298,797 商品 / 20,030,340,587 B |
| 本轮实际处理 | **81,637 商品**（= 连续字节前缀，1,288,490,188 B = 源文件 **6.43%**） |
| 选择方式 | 连续前缀（prepared before seeing any task target；**未用 target product 构造候选池**） |
| docs 投影 | `data/p14a/catalogue/docs.jsonl`，120,676,740 B（115.09 MiB），81,637 行，sha256 `29dc078e…bed0b56` |
| raw 前缀 | `data/p14a/catalogue/products_prefix.jsonl`，sha256 `86804471…b526d`（保留原始样本可复核） |
| 索引类型 | **BM25**，复用 `recommendation.control.catalog_search.CatalogSearchSource`（与仓库其余检索同一 tokenizer/常量/IDF）；进程内索引，无落盘 index 文件 |
| 索引字段 | title / store / main_category / categories / features / description（由真实商品记录归一化，无 reviews） |
| 覆盖率 | 1,298,797 → 81,637 = **6.29%** |
| 字段覆盖（kept 商品，builder 统计） | category/price/seller_name **100%**；attributes 83.18%；options 55.23%；description 61.60% |
| 下载 | 154 × 8 MiB ordered Range chunks，10 workers，86.6 s @ **14.19 MiB/s**；无 early-stop |
| 解析/投影 | 76.45 s；0 重复 ID / 0 malformed JSON / 0 UTF-8 错误 |
| 数据完整性 | builder 独立复算 + Lead 侧复核：docs sha256 与 81,637 唯一 id 一致（`runs/p14a/logs/isolation_check.json`） |

## 4. Search 实现及资源成本

实现：`recommendation/shopping_companion/search.py::ShoppingCompanionSearchTool`，薄封装既有 `CatalogSearchSource`，不重排、不注入候选、不返回空结果以外的任何东西。

| 指标（真实 81,637 商品目录，WSL CPU） | 值 |
| --- | --- |
| catalogue load（81,637 docs → `ProductMetadata`） | **8.20 s** |
| BM25 index build（进程内一次） | **3.98 s** |
| `product_search` latency（10 个真实 train query，limit=10） | **median 32.6 ms / max 115.7 ms** |
| 命中 | mean 10.0 hits/query（limit=10；limited catalogue 内） |
| 进程峰值 RSS（probe） | **1,523 MB**（E2E 单任务进程 1,583 MB） |
| 过滤语义 | 官方 `shop_id` / `price="min-max"` 过滤保留；有过滤时先取较宽 head 再过滤，保持 catalogue 自身顺序 |
| 诚实语义 | 0 命中返回空 + `matched=false`；不生成 fallback 列表 |

## 5. Product Info 实现

`recommendation/shopping_companion/product_info.py::ShoppingCompanionProductInfoTool`（对应官方 `product_view`）：

- 输入：`product_id`（raw 或 namespaced）+ **namespace**（默认 `shopping_companion`）+ 可选 attributes；
- 输出：name / category / price（数值 + `currency="UNKNOWN"`）/ seller（id+name）/ attributes / options / description 摘要 / 每个请求属性的 `ProductFact` 四态 / missing / unknown / unsupported / provenance / 原始 product_id；
- **UNKNOWN 语义**：目录快照缺 currency、stock、实时可得性 → 明确报告缺失，不猜测、不把缺失当满足；
- **跨 namespace 拒绝**：Sports 整数 id、P13-R Electronics ASIN、demo catalogue 一律 `namespace_mismatch` 拒绝；
- 同一 product_id 多次查询结果**逐字节一致**（测试覆盖）；
- 真实 E2E 中 10 次 `product_view` 全部来自 catalogue 记录（`catalog_verified`）。

## 6. Memory 历史数据来源（客观阻塞）

- 官方 release **只有** `train.parquet` / `test.parquet` / `products.jsonl`；官方 memory 服务要求的 `data/shopping_companion_s_cleaned.jsonl`（haystack sessions，长程对话历史）**不在发布内容中**，`extra_info` 也只有 `conversation_id` 等索引参数。
- 因此正常 P14-A 运行**没有合法历史**：`recommendation/shopping_companion/memory.py::ShoppingCompanionMemory` 的新实例诚实返回 `available=false, status="history_unavailable"`，`preferences=[]`，reason 明确说明历史语料未发布；**不生成合成历史、不读 ground truth、无 target id 通道**。
- 已实现的真实接入（复用既有 M5/M9，非复制）：
  - M9：`PreferenceMemoryService`（唯一写入者）+ `InMemoryPreferenceStore` + `RuleBasedPreferenceExtractor` + 既有 schema；
  - M5：`build_turn_events` → `BeliefUpdater`（写前 dry-run）→ `StateManager(EphemeralSQLiteEventStore(":memory:"))` → `state_projector.for_agent`；另有既有 M9→M5 `legacy_adapter` 只读桥；
  - 新写的只有：session 归一化、逐轮 evidence ledger（日期/session/turn/source_kind）、确定性 lexical preference search 读模型、oracle-diagnostic 显式门。
- 唯一真实对话文本是 stage_2 prompt 中的 **oracle memory block**（evaluator-only 文件保存）。它只允许以 `source_kind="released_oracle_diagnostic"` + `oracle_diagnostic=True` 的实例消费；runtime bridge 独立拒绝非 diagnostic 运行里的 oracle evidence（有测试）。**P14-A 未运行 oracle-diagnostic 轨迹**（CLI 未接线；仅 API+测试就绪，不计成功）。

## 7. Planner Runtime Bridge（P14-0 G6）

- **契约完全复用**：`recommendation/demo/planner_contract.py` 的 `PlannerAction`（ASK/RETRIEVE/RERANK/FINALIZE）、`PlannerObservation`、`ActionProposal`、`validate_action`、`run_planner` 均未修改。
- 新增薄桥：`recommendation/shopping_companion/planner_bridge.py`
  - `P14ARulePlanner`：**确定性 Rule 策略**（不是 Qwen、不是训练模型），按 executor 记录的 phase 在既有动作词汇内决策；
  - `P14AToolExecutor`：实现既有 `PlannerExecutor` 协议，真实调用 Memory / `product_search` / `product_view` / CandidateLedger / GroundedReasoner+Eligibility；
  - Tool error **记录进轨迹并终止为显式错误**，不静默 fallback；oracle evidence 在非诊断模式触发 tool_error。
- **实际动作序列（真实 E2E）**：stage_1 `RETRIEVE`（memory，返回不可用）→ `ASK`（记录未解决澄清）；stage_2 `RETRIEVE`（真实 BM25）→ `RERANK`（真实 product_view + 约束/eligibility）→ `FINALIZE`（来自 ledger 的候选）。
- Qwen：本机无 Planner server，且本轮不启 GPU ⇒ **Qwen 真实推理未执行**；只交付 Rule 真实链路 + Qwen executor contract 兼容（既有 HTTP proposer 未改）。不得宣称 Qwen Agent 已通过。

## 8. 真实端到端轨迹（D3 必需项）

**任务**：官方 train split 的 single-product 任务 `0014b39c4159d54fd5eafcd6c2bd6699_1`（question_date 2025/08/30，query "collar"）。

> **语义修正说明（P14-A 补充核实）**：初版轨迹把整条 feasible head（10 件）当作 FINALIZE 结果，并在 stage_1
> `ASK` 未解决时继续 stage_2 且报告任务成功；这两点已修复（详见 [`P14A_BLOCKER_VERIFICATION.md`](P14A_BLOCKER_VERIFICATION.md) §5/§6）。
> 下面是修复后的真实结果。

| 步骤 | 真实执行（默认路径 `…_1.jsonl`） |
| --- | --- |
| 加载任务 | agent-visible `train_tasks.jsonl`（oracle 已移除） |
| stage_1 memory | `ShoppingCompanionMemory` → `history_unavailable`（无伪造偏好）→ Planner `ASK` 记录未解决澄清（不做假用户回复） |
| task terminal | **`awaiting_clarification` / `succeeded=false`**；官方两阶段协议没有 stage_1 偏好答案可交给 stage_2，默认不执行 stage_2（`stage_2_executed=false`） |
| 轨迹文件 | `runs/p14a/trajectories/real_limited_catalogue_0014b39c…_1.jsonl`（0 recommendation，2 tool calls） |
| evaluator | `…_1.eval.json`：`target_ids=['4076792817']`、`task_completed=false`、`awaiting_clarification=true` |
| 运行资源 | 进程 wall 73.6 s（含 catalogue load）；峰值 RSS **962 MB** |

**显式 diagnostic 路径**（`--diagnostic-stage2-without-preferences`，文件
`real_limited_catalogue_…_1_diagnostic_stage2.jsonl`）：真实 BM25 search（10 hits，全部进 CandidateLedger，
source=`catalog_search`、score_kind=`bm25`、rank 保留）→ 真实 `product_view` → Eligibility → FINALIZE；
由于单商品任务语义，**只呈现 1 件商品**（`presented_count=1`、`candidate_count=10`）；但 terminal 仍为
**`awaiting_clarification` / `succeeded=false`**（诊断续跑，不计任务成功）。evaluator 记录
`single_product_presentation_ok=true`、`target_in_search_hits=false`（limited catalogue 的诚实结果）。
运行资源：单任务内部 4.14 s；进程 wall 73.5 s；峰值 RSS **1,583 MB**。

**add-on 结构性 Smoke（默认路径）**：`0014b39c…_2` → `awaiting_clarification`、`succeeded=false`、0 推荐、
`bundle_structural_only=true`、`constraints=[]`（不把 bundle budget 静默降级为单品价格上限）；
P14-A 不实现 bundle 组合/优惠优化，不计入成功案例。

模式标签：本轮只产生 `real_limited_catalogue` 轨迹；`contract_only` 仅用于测试 fixture；`oracle_diagnostic` 未运行；`real_full_catalogue` 未达成。四类互不混用。

## 9. Oracle / ground-truth 泄漏检查（独立复核）

`runs/p14a/logs/isolation_check.json`（Lead 侧独立脚本，不依赖任何 teammate 结论）：

| 检查 | 结果 |
| --- | --- |
| agent 可见 800 tasks 中 target product id 出现次数 | **0**（eval target ids 共 1,449 个） |
| agent 可见文件中 oracle marker 出现次数 | **0** |
| agent 可见文件中 `ground_truth`/`reward_model` 键 | **0** |
| oracle 对话指纹（每个 oracle 用户轮前 40 字符）在 agent 文件出现 | **0** |
| evaluator-only 文件中 oracle block | 800/800（仅此文件） |
| 三条修复后真实 trajectory 中 oracle marker / GT 键 / target id | **0 / 0 / 0** |
| catalogue docs 重复 id / reviews 泄漏 | 0 / 0 |
| **总判定** | **ISOLATION_OK** |

此外：`train_eval.jsonl` 只在 trajectory 完成后由 `evaluator.py` 读取（CLI 顺序固定），runtime/planner/memory/search/info 均无读取路径；`memory.py` 无 target-id 参数。

## 10. 复用的核心代码（没有第二套）

| 能力 | 复用（未修改） |
| --- | --- |
| Planner Action 契约 + step loop | `recommendation/demo/planner_contract.py`（`PlannerAction/Observation/ActionProposal/validate_action/run_planner`） |
| 候选账本 | `recommendation/control/candidate_ledger.py`（`CandidateLedger.record_from_source`、`ranked_parent_asins`、provenance/score_kind） |
| 约束与 Eligibility | `recommendation/control/grounded_reasoning.py`（`GroundedReasoner`）+ `constraint_eligibility.py`（`CandidateEligibilityEvaluator/ConstraintRequirement`） |
| 商品目录合同 | `recommendation/catalog/schemas.py::ProductMetadata`（真实商品归一化后直接喂给 `GroundedReasoner`） |
| BM25 | `recommendation/control/catalog_search.py::CatalogSearchSource` |
| Product fact 形状 | `recommendation/control/product_research.py::ProductFact` |
| M9 偏好记忆 | `recommendation/memory/`（service/store/extractor/schemas） |
| M5 决策状态 | `recommendation/state/`（`build_turn_events`、`StateManager`、`BeliefUpdater`、`state_projector`、`legacy_adapter`） |
| namespace 契约 | `recommendation/contracts/namespaces.py`（P14-0 seed；仅**新增** `shopping_companion` 条目） |

## 11. 新增 / 修改代码清单

完整列表（含每个文件大小与 sha256）见 [`P14A_SOURCE_DIFF.txt`](P14A_SOURCE_DIFF.txt)。摘要：

- 新增 `recommendation/shopping_companion/`：`namespace.py`、`catalogue.py`、`search.py`、`product_info.py`、`planner_bridge.py`、`runtime.py`、`evaluator.py`、`dataset.py`、`memory.py`、`__init__.py`
- 新增 experiments：`p14a_build_catalogue.py`、`p14a_prepare_dataset.py`、`p14a_run_task.py`、`p14a_probe_search.py`
- 新增测试：`tests/test_p14a_{namespace,catalogue_search,product_info,planner_runtime,dataset,memory}.py` + `tests/data/p14a/` 4 个 fixture
- 修改（仅 2 个 tracked 文件）：`recommendation/contracts/namespaces.py`（注册 namespace，additive）、`.gitignore`（让 `tests/data/p14a/*.jsonl` 可提交）
- 未修改：Serving V1/SASRec/Two-Tower/M5/M9/Planner/CandidateLedger/GroundedReasoner/Eligibility 代码；28 个既有未提交修改原样保留。

## 12. 测试结果

| 批次 | 命令 | 结果 |
| --- | --- | --- |
| P14-A 全部测试 | `pytest tests/test_p14a_*.py`（6 文件） | **74 passed in 63.2 s** |
| 旧核心默认行为 | `pytest tests/test_api.py tests/test_serving_fusion.py tests/test_planner_contract.py tests/test_docs.py` | **142 passed in 25.6 s** |
| 编译 | `python -m py_compile`（新模块） | PASS |
| 真实 E2E（默认） | `p14a_run_task.py --task-id …_1` / `…_2` | exit 0；`awaiting_clarification`、succeeded=false、0 推荐；trajectory+eval 落盘 |
| 真实 E2E（显式诊断） | `p14a_run_task.py --task-id …_1 --diagnostic-stage2-without-preferences` | exit 0；工具链跑通、`recommendation=1`、terminal=`awaiting_clarification`、succeeded=false（诚实） |
| 资源 probe | `p14a_probe_search.py` | exit 0；见 §4 |
| 隔离复核 | `verify_p14a_isolation.py` | **ISOLATION_OK** |

未运行（按约束与本轮范围）：全仓库历史 pytest、P13-R reserved test、Shopping Companion 官方 test split、冻结 Holdout、GPU/模型服务。

## 13. 遗留问题

1. **真实历史数据未发布**（阻塞 Memory 正常路径）：stage_1 只能诚实报告 `history_unavailable`；真实偏好恢复不可验证。需要官方后续 release 的 conversation/haystack JSONL，或另行取得授权历史语料。
2. **完整目录未索引**：只覆盖 6.29%（81,637/1,298,797）；target 可检索率因此被人为压低（本例 target 不在样本内）。P14-B 前需决定：远程/更大样本/完整 index（BGE 可作为后续增强，本轮未做）。
3. **已修复（P14-A 补充核实）**：① 单商品 FINALIZE 曾呈现 10 件 → 现只呈现 1 件；② stage_1 `ASK` 未解决时曾继续 stage_2 并报告成功 → 现默认不执行 stage_2，记录 `awaiting_clarification`；诊断续跑需显式 opt-in 且仍不计成功。详见 `P14A_BLOCKER_VERIFICATION.md`。
4. **oracle-diagnostic 未接线**：API + 15 项 memory 测试就绪、bridge 可接受显式 tagged oracle evidence，但 CLI 尚未从 `train_eval.oracle_memory_block` 解析 session 并注入；未运行、未计入成功。
5. **Qwen Planner 未真实运行**：本机无服务、本轮不启 GPU；仅 Rule 链路真实验证。需在具备 Qwen server 时单独验收（不得用预写响应冒充）。
6. **BGE/dense 检索未做**（非本轮必需）；Serving V1 的 BGE 工具未被使用。
7. **add-on/bundle**：只验证 schema 与工具边界；voucher/预算/多商品组合优化留给 P14-B/C。
8. **Resource 观测**：单进程峰值 ~1.5–1.6 GB（limited 81k 目录）；完整 1.3M 目录需专门容量评估，不能在本机 7.7 GB WSL 内直接照搬。
9. **P13-R 身份漂移（P14-0 G9）仍存在**：本报告未改动 P13-R 数据；若 P14 复用 Electronics 目录需先修 namespace/producer 记录。

## 14. 是否具备进入 P14-B 的条件

**部分具备（LIMITED）**：
- 具备：真实数据 Adapter、namespace、真实商品工具、CandidateLedger/Eligibility 接线、Planner 执行桥、轨迹/evaluator、隔离与测试基础——这些都是 P14-B Preference-Aware Recommendation 可以直接使用的扩展点；`recommendation/shopping_companion/` 为 P14-B 保留干净接口（search 返回真实候选集、Product Info 给出已验证属性、evaluator 独立读 GT）。
- 不具备：**真实偏好监督/历史数据**与**完整目录**。P14-B 若要做 preference-aware scoring，需要先解决 (a) 历史语料来源/或改用可发布的历史替代，(b) 目录覆盖与候选可达性（否则 ranker 评估会被 limited pool 主导）。

**建议下一步（不在本轮执行）**：P14-B 先做**评估协议与候选可达性**（完整或更大目录、judged negative / preference 标签来源），再决定是否训练 preference-aware scorer；Qwen/LLM 规划层另开验收，不混入 P14-B 质量结论。

---

## E5. 分项判定

| 领域 | 判定 | 依据 |
| --- | --- | --- |
| Dataset Adapter | **PASS** | 真实 train 800 任务、1+1 配对、400/400 类型、question 一致；oracle 800/800 移除且双向隔离检查通过 |
| Catalogue / Search | **LIMITED** | 真实 81,637 商品 BM25 可用（median 32.6 ms，真实命中）；但只是完整目录 6.29%，非全目录索引 |
| Product Info | **PASS** | 真实 catalogue facts、UNKNOWN 语义、namespace 拒绝、重复查询一致 |
| Memory | **LIMITED** | M5/M9 adapter 真实集成并通过测试；**官方历史语料未发布** ⇒ 正常 run 只能 `history_unavailable`，真实偏好恢复未验证 |
| Planner Runtime Bridge | **PASS** | 既有 Action 契约 + 真实 executor；Rule 真实动作序列；tool error 显式；Qwen 未运行（单独声明） |
| E2E Train Task | **LIMITED** | 一条真实 single-product 任务全链路 + 轨迹 + evaluator 成立；但 stage_1 因数据阻塞无真实偏好，target 不在 limited catalogue；add-on 仅结构性 smoke |
| Legacy Regression | **PASS** | 142 项 API/Serving/Planner/docs 测试通过；旧代码零修改；默认 Serving V1 行为不变 |

**P14-A 总判断：LIMITED。**

> 若后续获得官方历史语料并完成更大/完整目录索引，P14-A 的 Dataset/Search/E2E 可升级为 PASS 并重新评估 GO；在此之前不得声称“完整 Shopping Companion Agent 已跑通”，也不得声称推荐质量或 Planner 优越性（属 P14-B/C）。

---

## 附：关键产物路径

- 轨迹与评测：`runs/p14a/trajectories/`
- 运行日志 / 资源：`runs/p14a/logs/`（`e2e_*.time.log`、`search_probe.json`、`isolation_check.json`）
- 数据 manifest：[`P14A_DATA_MANIFEST.json`](P14A_DATA_MANIFEST.json)
- 源码差异：[`P14A_SOURCE_DIFF.txt`](P14A_SOURCE_DIFF.txt)
- 数据（git-ignored）：`data/p14a/dataset/`、`data/p14a/catalogue/`

## 15. 补充核实（P14-A BLOCKER VERIFICATION）

P14-A 交付后按“只核实三个阻塞问题”的要求做了一轮补充核实，结论与最小修复记录在
[`P14A_BLOCKER_VERIFICATION.md`](P14A_BLOCKER_VERIFICATION.md)：

- **MEMORY_DATA = NOT_PUBLICLY_FOUND**：官方精确历史语料未公开；公开的 LongMemEval-S cleaned（MIT）只是底料，
  且已发布的 400 个 `conversation_id` 无法由公开 session ids 重建（0 匹配）。
- **FULL_CATALOGUE = FEASIBLE**：官方 pyserini/Lucene 方案 + 资源估算（≈25–28 GB 磁盘、≥16 GB 内存更合适）；
  本机 WSL 7.7 GB 不建议作为构建机；未构建、未装依赖。
- **AGENT_TRAJECTORY = FIXED**：单商品 FINALIZE 由 10 件修正为 1 件；stage_1 ASK 未解决时默认
  `awaiting_clarification`、不执行 stage_2；诊断续跑显式 opt-in 且不计成功。修复后 P14-A 测试 74 passed、
  旧核心回归 142 passed，轨迹已重新生成并复核隔离。

因此 §8 的旧“terminal=succeeded / 10 件推荐”描述已被本节与 §8 修正版替代；P14-A 总判断仍为 **LIMITED**。
