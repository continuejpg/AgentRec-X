# AgentRec-X V2 / E-04B-2 Gate A — 真实评测协议自检与 BM25 基线报告

> 状态：**Gate A PARTIAL** —— 协议自检 **PASS** ✓、真实 BM25 Smoke **PASS** ✓、
> 真实小样本指标**已算出但不可作为质量结论**（exploratory ✓，且 BM25 该口径本身不适合作为质量基线 ✓）。
> 本轮**未**加载任何神经网络权重、**未**联网、**未**安装依赖、**未**修改生产代码、**未**执行任何 Git 写操作。

---

## 1. HEAD、分支与初始工作树

```text
branch : v2/e04a-pass-checkpoint
commit : b3af9da79a87aada73cf389ee4c22df5f2169d54   （E-04A 检查点；本轮未改、未重写 ✓）
初始工作树：仅 ?? docs/V2_E04B_REAL_ASSET_AUDIT.md（E-04B-1 产物，未跟踪）
结束时工作树：?? docs/V2_E04B_REAL_ASSET_AUDIT.md、?? experiments/e04b2_gate_a.py、
             ?? tests/test_v2_gate_a_protocol.py、?? docs/V2_E04B2_GATE_A_REPORT.md
             （结果文件写在 `runs/e04b2_gate_a/`，该目录被 .gitignore 忽略 ✓）
```

## 2. 实际授权边界

允许并已执行：读取评测代码/协议/数据 ✓；不加载权重的协议自检 ✓；用**真实** Sports_and_Outdoors 商品目录
运行真实 BM25 ✓；在协议成立的前提下做小样本指标 ✓；新建 Gate A 脚本、测试、报告与结果文件 ✓。
未执行：加载 SASRec/Two-Tower/TIGER 权重 ✗、启动 TIGER Worker ✗、安装依赖或下载 ✗、联网/在线 LLM ✗、
训练 ✗、修改 Serving ✗、四源融合/Gate B ✗、E-04C ✗。

## 3. 真实数据与索引路径

| 用途 | 路径 | 规模 |
|---|---|---|
| 用户序列（真实交互） | `data/processed/Sports_and_Outdoors_sequences.json` | 333 MB，`agentrecx.sequences.v1`，412,445 用户 / 3,500,587 交互 |
| 商品目录（真实） | `data/processed/Sports_and_Outdoors_products.jsonl` | 294 MB，156,746 条（含表头记录） |
| 身份映射 | `data/processed/Sports_and_Outdoors_mappings.json` | 38 MB：`item2id` 156,746 / `user2id` 412,445 / `pad_id=0` / `first_real_id=1` |
| BM25 实现（生产路径） | `recommendation/control/catalog_search.py`（`CatalogSearchSource.search` `:238`） | 索引 **156,746** 条真实记录，构建 13.79 s |
| 结果输出（独立目录） | `runs/e04b2_gate_a/gate_a_summary.json` | 56 KB |

## 4. 时间划分与测试输入防泄漏证据（运行时，非 grep）

```text
协议        : agentrecx.eval_protocol.v1 / temporal_leave_two_out（split.py:46-47）
样本        : 100 个真实用户（artifact 顺序，前 100 个通过协议者；无需随机抽样，见 §5 声明）
users_checked=100, valid=100, skipped=0, skip_reasons={}          ← 真实运行时统计
pad_hits=0, out_of_range_ids=0                                    ← 序列内从无 PAD、从不越界
```

* **输入与标签分离（运行时证据）**：`EvaluationCase` 把 `train_history`、`validation_target`、
  `test_target` 作为**不同字段**；测试时刻可见输入为 `test_history = train_history + validation_target`
  （`len(test_history) == sequence_length - 1`，已断言 ✓），标签是 `test_target`（`items[-1]`）✓。
* **目标文本隔离（运行时证据）**：脚本记录"哪些 item 的目录文本被读取过"（`text_used`）。
  100 例扫描中仅有 1 个 id（`117042`）同时出现在**某个 case 的 history** 与该 case 的 target 上 ✓——
  复核确认它正是**该用户自己的历史项**（重复交互）✓，即它是作为**历史输入**被读取的 ✓，
  而不是从隐藏标签反向构造查询 ✓。结论：**未发生标签泄漏** ✓。
* **重复交互与掩码（协议细节，已断言）**：`excluded_seen(history, target)` 会把 target 自身从掩码中移除
  （否则重复交互的 case 会把正例删掉而无法评分 ✓）；原始 `test_seen` 仍是 history ∪ {validation_target} 的超集 ✓。

## 5. 商品身份映射与 PAD / seen 检查

```text
num_items=156746 ↔ mappings.item2id=156746 ↔ 目录可检索记录=156746   ← 三方一致 ✓
返回结果中：invalid_ids=0、pad_returned=0、masked-out 命中=75（已按协议过滤 ✓）
```

事先声明（§五要求，执行前固定）：`selection_rule` = artifact 顺序取前 N 个有效 case（无抽样）；
`seed=20261015`（仅记录，本规则不需要随机性）；`K={5,10,20}`；`query_rule` = **仅用 test-time 可见的
`test_history` 最近 ≤5 项的**目录文本 token；`mask_rule` = `excluded_seen(test_history, test_target)`；
`invalid_case_rule` = 协议拒绝或目标被掩码者跳过；指标口径 = full-catalogue 单正例（HR 与 Recall 等价 ✓）。

## 6. BM25 是否真正加载真实目录

**是。** 由**生产**词法来源构建：`CatalogSearchSource(MetadataIndex.from_records(目录记录))` ——
`indexed_records = 156,746`（与真实 `num_items` 完全一致 ✓），构建耗时 **13.79 s**，
构建后 RSS **2,571 MB**（可用内存 3,949 MB ✓）。**未**使用 Fake Catalog 或合成商品 ✓。

## 7. 真实检索案例（脱敏示例，来自 20 例 smoke 的第 1 例）

```text
case user_int_id=1 | query tokens=60（来自 history 目录文本） | matched=true | 返回 20 条 | 耗时 961.75 ms
top-1: item_id=13547, parent_asin=B001U9YK8I, score=80.609005, score_kind="bm25", masked=true（该 id 属该用户历史 ✓ 已在排序前过滤）
masked-out 命中=5（本 case） | invalid_ids=0 | pad_returned=0 | 查询未读取 target 文本 ✓
```

## 8. 查询合法性及离线指标是否可比较

* **查询合法性**：查询只由 `test_history` 的目录文本构成 ✓（测试时刻可见 ✓）；**未**使用
  `test_target` 的名称/描述/分类反构造 ✓（§4 的运行时证据 ✓）。
* **可比较性**：**不具备模型质量比较资格** ✗。原因有二：
  (1) 协议本身按**历史交互 id** 定义输入，而词法来源需要**自然语言查询**；由历史标题派生的查询是
      一种**合法但非协议原生**的输入构造 ✓，因此它只能作为"真实 BM25 功能基线"，
      不能与其他模型的协议原生指标并列 ✓；
  (2) 样本量仅 100 例（exploratory ✓）。
  → 按 §五，本报告对质量部分标注为 **exploratory / NOT_COMPARABLE（作为模型质量结论）** ✓。

## 9. 实际计算的 HR/NDCG（exploratory）

| 指标 | @5 | @10 | @20 |
|---|---|---|---|
| HR（= Recall，单正例等价 ✓） | 0.000000 | 0.010000 | 0.020000 |
| NDCG | 0.000000 | 0.003333 | 0.006036 |

```text
cases_scored=100（全部有效 ✓）  命中数（rank ≤ max K）=2   invalid_ids=0   pad_returned=0
```

**解释（诚实边界）**：这是**词法来源 + 历史标题查询**在 100 例上的 exploratory 结果 ✓；
低频命中符合预期（BM25 无法编码序列偏好 ✓）。**不得**据此声称任何模型"质量低/高" ✗，
也**不得**与 sampled-negative 或别的口径比较 ✗。

## 10. 内存、耗时与样本规模

```text
开始：RSS 12.3 MB，可用内存 6,451.7 MB
A-1 扫描 100 用户后：RSS 424.4 MB
A-2 建索引（156,746 条）：13.79 s，RSS 2,571.1 MB（可用 3,949.1 MB）
A-3 100 例评分：总墙钟 175.61 s，结束 RSS 2,642.4 MB，结束可用 3,888.4 MB
smoke 单例检索：961.75 ms（20 例全部 matched ✓）
内存守护：可用内存 <1,200 MB 时中止（本轮未触发 ✓）；无 GPU（CPU-only ✓）
```

## 11. 测试命令与通过/失败数量

```bash
# 新增 Gate A 协议测试（运行时断言，不加载权重）
.venv/bin/python -m pytest tests/test_v2_gate_a_protocol.py -q
#   → 6 passed

# 相关既有回归（划分 / 掩码 / BM25 / 指标 / 目录 / 融合）
.venv/bin/python -m pytest tests/ -q -k "split or mask or bm25 or metric or catalog or fusion"
#   → 255 passed, 3 skipped, 2714 deselected in 408.41s

# Gate A 主脚本（真实数据 + 真实 BM25 + 小样本指标）
.venv/bin/python -m experiments.e04b2_gate_a --cases 100 --smoke 20
#   → 100 有效 case、索引 156,746 条、20 例 smoke 全部命中、HR/NDCG 见 §9
```

## 12. 新增或修改的文件

| 文件 | 说明 |
|---|---|
| `experiments/e04b2_gate_a.py`（新增） | Gate A 主脚本：流式读真实序列（一次一个用户 ✓）、真实协议切分、运行时隔离断言、真实 BM25 索引与检索、掩码后 full-catalogue 排序、HR/NDCG、资源守护 |
| `tests/test_v2_gate_a_protocol.py`（新增，6 项） | 运行时协议断言（PAD/越界、掩码与正例可评分性、输入-标签分离、HR==Recall/NDCG 公式、目录身份空间一致） |
| `runs/e04b2_gate_a/gate_a_summary.json`（新增，被忽略目录） | 完整结果与逐例明细（含声明的选择规则/seed/K/查询规则/掩码规则 ✓） |
| 生产代码 | **未修改** ✓（GB 级资产未移动/未覆盖 ✓） |

## 13. Gate A 已验证与未验证范围

**已验证** ✓：真实数据的协议自检（100 例）、真实 BM25 在**全量真实目录**上的加载与检索、
返回身份/分数/score_kind/rank、掩码过滤、无 PAD/无越界、目标文本隔离、full-catalogue 单正例指标路径可运行、
资源与耗时记录完整、可复现（规则与 seed 已声明 ✓）。
**未验证** ✗：SASRec / Two-Tower / TIGER 真实推理（未加载权重 ✓ 按授权 ✗）、四源 RRF 与公平性对照（Gate B）、
Memory 增益（离线无真实用户自述偏好 ⇒ 不可量化 ✓）、线上效果与生产稳定性 ✗、
以及任何"模型质量优劣"的结论 ✗。

## 14. 结束时 HEAD、git status

```text
HEAD   : b3af9da79a87aada73cf389ee4c22df5f2169d54（与开始时一致，检查点未受影响 ✓）
branch : v2/e04a-pass-checkpoint
status : ?? docs/V2_E04B_REAL_ASSET_AUDIT.md
         ?? experiments/e04b2_gate_a.py
         ?? tests/test_v2_gate_a_protocol.py
         ?? docs/V2_E04B2_GATE_A_REPORT.md
         无 modified、无删除；未执行 add/commit/push/reset/restore/clean/stash/checkout ✓
```

## 15. 最终结论

**Gate A = PARTIAL。**

```text
真实测试协议有效                ✓（100 例运行时检查，PAD/越界/掩码/输入-标签分离全部通过）
运行时目标隔离成立              ✓（只读 test_history 的目录文本；唯一重复交互 id 已复核为历史输入）
真实 BM25 目录检索成功          ✓（156,746 条真实记录，生产来源，返回合法身份与 bm25 分数）
返回合法商品身份与分数          ✓（invalid=0、PAD=0、score_kind 标注正确）
资源及结果可复现                ✓（规则/seed/K/掩码/查询规则执行前固定；RSS/耗时已记录）
小样本指标                      ◐ exploratory 且 NOT_COMPARABLE（查询为合法但非协议原生的历史标题构造）
```

→ 按 §九通过标准：**检索 Smoke 部分 PASS，质量基线部分标注 NOT_COMPARABLE，Gate A 以 PARTIAL 结束** ✓。
**未**虚构 HR/NDCG，也**未**把 100 例 exploratory 结果当作模型质量结论 ✓。

**明确停止点**：Gate A 结束，**不进入 Gate B** ✓；未加载任何 SASRec/Two-Tower/TIGER 权重 ✓；
未安装依赖、未下载资产、未联网 ✓。等待人工审阅与下一次授权。
