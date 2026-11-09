# P14-A 补充核实报告 — 三个阻塞问题

> 范围：只核实 ①历史对话可得性 ②全量目录可索引性 ③单商品 Agent 轨迹语义，并对发现的明确错误做最小修复。
> 未重跑 P14-A、未做全仓库审计、未启动 P14-B、未构建全量索引、未训练、未使用 GPU、未操作 AutoDL、未 commit/push。
> 环境：Windows 11 + WSL2 Ubuntu-22.04；仓库 `.venv`（Python 3.10.12，CPU-only）。
>
> 本报告新增证据文件：`runs/p14a/logs/longmemeval_provenance_check.json`、`runs/p14a/logs/catalogue_coverage.json`、
> 重新生成的 `runs/p14a/trajectories/*`；修复后的行为以本报告为准，旧 P14-A 报告中的对应段落已同步修订。

## 0. 三个判定

| 项 | 判定 |
| --- | --- |
| **MEMORY_DATA** | **NOT_PUBLICLY_FOUND**（官方发布的**精确历史语料**未公开；其公开组成部分 LongMemEval 可合法取得，但不足以重建官方语料；不能据此断言“绝对不存在”） |
| **FULL_CATALOGUE** | **FEASIBLE**（有可执行的官方/备用索引方案；需要 Java 11 + pyserini + 约 26–29 GB 磁盘和 ≥16 GB 内存的更宽裕环境；本机 WSL 7.7 GB 不适合作为首选构建机） |
| **AGENT_TRAJECTORY** | **FIXED**（发现并修复两处明确任务语义问题：单商品 FINALIZE 呈现 10 件 → 1 件；stage_1 ASK 未解决却继续 stage_2 并报告成功 → 默认 `awaiting_clarification`、不再当作完成） |

---

## 1. 历史对话数据来源追踪与最终可用性

### 1.1 已核实的发布内容（没有历史文件）

| 位置 | 实际内容 | 证据 |
| --- | --- | --- |
| HF 数据集 `yuzhan2205/Shopping-companion` | 仅 **6 个文件**：`data/products.jsonl`、`data/train.parquet`、`data/test.parquet`、`README.md`、`.gitattributes`，以及 `data/` 目录项 | HF tree API（mirror）逐文件列出；`refs` 只有 `main` 与 `refs/convert/parquet` |
| 作者 HF 账号 `yuzhan2205` | 只有这一个 dataset | HF datasets API `author=yuzhan2205` |
| 官方 GitHub `Jarvis33yu/Shopping-Companion` | 只有 `main` 分支；**0 个 release**（无附件）；issues=0 | GitHub API |
| `train.parquet` 的 agent 可见字段 | `prompt`（stage_1 只有 system+用户当前需求）、`extra_info.tools_kwargs`、`reward_model` | 逐行解析（见 1.3） |

### 1.2 官方工具服务从哪里读取历史（代码链路）

官方流程（`run_gen.sh`、`run_mem_preprocess.sh`、`src/build_mem_indexes.py`、`src/serve_mem.py`）：

```text
data/longmemeval_s_cleaned.json         ← 公开基础语料（LongMemEval）
        + data/product_indexes（20GB 商品目录的 Lucene 索引）
        + data/products.jsonl
   └─ src/dialogue_gen.py   → data/preferences/*.jsonl   （LLM 生成购物偏好对话，模型 gpt-5-2025-08-07，OpenAI 兼容 API）
   └─ src/question_gen.py   → data/shopping_companion_s_cleaned.jsonl（把偏好对话 + LongMemEval 会话混合、随机插入日期）
   └─ src/build_mem_indexes.py（SentenceTransformer 编码会话）→ data/mem_indexes/<conversation_id>/
   └─ src/serve_mem.py（本地 Flask 服务）← mem_search/mem_view 工具 HTTP 调用
```

关键代码事实：

- `run_mem_preprocess.sh:3` 写死 `conversation_file="data/shopping_companion_s_cleaned.jsonl"`；
  **没有下载步骤**，README 只说 “after placing the conversation JSONL file at the path expected by the script”。
- `src/build_mem_indexes.py` 直接 `open(conversation_file)`；`src/serve_mem.py` 只从本地 `data/mem_indexes` 加载 index。
- `src/dialogue_gen.py` 使用 `gpt-5-2025-08-07-GlobalStandard`（OpenAI 兼容 endpoint，需 `OPENAI_API_KEY`），
  生成过程依赖 LLM（行 119/244/357/407/475 附近），**没有 random.seed**（`grep` 无命中 ⇒ 生成不可确定性复现）。
- `src/question_gen.py:490` 有 `random.seed(42)`，但输入是上面生成好的 `preferences_dir`；
  `src/question_gen.py:325` 定义 `base_question_id = md5(",".join(conversation.haystack_session_ids))`。
- `src/question_gen.py:204/222/239/257/279`：session id 由生成器构造
  （`single_product_<product_id>`、`add_on_deals_<product_id>`、`repeat_purchase_<product_id>_<i>`、`complement_<product_id>`）
  与 LongMemEval 的原始 session id 混合后一起参与 md5 ⇒ **官方 conversation_id 无法从公开文件重建**。
- `src/serve_mem.py` 的数据源是本地文件/索引，`extra_info.tools_kwargs` 不含服务 URL 或凭据（见 1.3）。

### 1.3 `extra_info.tools_kwargs` 与 agent 可见输入（逐行扫描 800 行）

| 检查 | 结果 |
| --- | --- |
| `tools_kwargs` 出现的工具名 | 仅 `mem_search` / `mem_view` / `mem_summarize_by_date` / `product_search` / `product_view` |
| `create_kwargs` 内容 | 仅 `conversation_id` + `question_id`（product 工具仅 `question_id`） |
| URL / `://` / 文件路径 / index / token / key / password / credential 引用 | **0 处** |
| stage_1 `user` 消息 | 全部以 `Current Date: …` 开头；最大 307 字符、平均 115.8 字符；**0 条**包含 `user:`/`assistant:` 对话轮标记 |
| stage_1 是否包含 haystack/history 文本 | **没有**（表格/历史会话不在 agent 可见数据中） |

### 1.4 公开基础语料核实（LongMemEval-S cleaned）

- 来源：`https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned`（**MIT**）；
  文件 `longmemeval_s_cleaned.json`（277,383,467 B，sha256 `d6f21ea9…c3a442`）正是官方脚本期望的
  `data/longmemeval_s_cleaned.json` 文件名与格式。
- 实测：500 条记录；字段与官方 `ConversationDTO` 一致
  （`answer/answer_session_ids/haystack_dates/haystack_session_ids/haystack_sessions/question/question_date/question_id/question_type`）；
  每题 haystack 会话数 38–62（中位 48），对话轮 396–616（中位 491）；session id 形如 `sharegpt_yywfIrx_0`，日期形如 `2023/05/20 (Sat) 02:21`。
- **公开基础可合法使用**，但它只是官方历史语料的“底料”：
  官方 benchmark 的历史还叠加了 LLM 生成的购物偏好对话与随机插入/日期（1.2），因此不能只靠 LongMemEval 恢复每题的历史。

### 1.5 关键证明：官方 conversation_id 不能由公开文件重建

- 将公开 LongMemEval-S 每题 `haystack_session_ids` 按官方公式做 md5（有序与排序两种），
  与 400 个已发布 `conversation_id` 对比：**匹配数 = 0 / 0**
  （`runs/p14a/logs/longmemeval_provenance_check.json`）。
- 原因即 1.2：已发布 `conversation_id` 是“LongMemEval 原始 session ids + LLM 生成的购物偏好 session ids”合并后的哈希；
  生成部分的对话文本/ids 未发布。可确定：**仅凭公开数据无法定位/恢复任一 train 任务的真实历史**。
- `reward_model.ground_truth.dialogue` 里确实包含“相关偏好对话”（oracle answer sessions），
  但它是 **oracle/ground truth**，使用它即构成泄漏，不能作为 stage_1 历史（本报告未使用）。

### 1.6 尝试与最终可用性

| 尝试 | 结果 |
| --- | --- |
| 从 HF 数据文件找历史 | 只有 3 个数据文件；无 conversation JSONL / mem index |
| 从 GitHub 仓库/Release/分支找 | 无数据文件、0 release、仅 main 分支；README 明示“additional dataset artifacts will be documented … as they are released” |
| 从 `extra_info.tools_kwargs` 找路径/URL/凭据 | 只有 conversation_id/question_id（0 处路径/URL/凭据） |
| 通过官方 `mem_search` 机制检索任一 train 任务 | **无法执行**：官方服务代码需要 `data/shopping_companion_s_cleaned.jsonl`（未发布），该文件不在发布包/仓库/README 下载链接中；无公开 mem 服务 endpoint |
| 用公开 LongMemEval 重建 | 公开底料可得，但生成式偏好对话（LLM 生成、无 seed、需全量商品索引+GPT-5 API）未发布且不可确定性复现；conversation_id 无法匹配 |
| 用 ground truth 反推 | 禁止（oracle），未执行 |

**结论：`MEMORY_DATA = NOT_PUBLICLY_FOUND`**（指官方发布的精确历史语料与其构造产物）。
这不是“官方历史绝对不存在”：它存在于作者侧（由公开 LongMemEval + LLM 生成管线构造），但不在公开渠道；
按任务要求给出联系渠道与询问信（§1.7），**不擅自发送**。

### 1.7 作者联系渠道与询问信（草稿，不发送）

渠道（均可在浏览器核实）：

1. GitHub 仓库：`https://github.com/Jarvis33yu/Shopping-Companion`（公开 issue；作者 Zijian Yu，profile 未公开 email）
2. HF 数据集讨论：`https://huggingface.co/datasets/yuzhan2205/Shopping-companion/discussions`
3. 论文作者（README 引用）：Zijian Yu、Kejun Xiao、Huaipeng Zhao、Tao Luo、Xiaoyi Zeng —— 可通过论文/单位邮箱联系（README 未给 email）

询问信草稿：

> 主题：Request for the conversation-history artifact used by Shopping Companion (benchmark reproduction)
>
> 您好，
>
> 我们正在复现 Shopping Companion benchmark 的 stage_1（long-term memory retrieval）。我们已从 HF 取得
> `data/train.parquet`、`data/test.parquet` 与 `data/products.jsonl`，并按官方流程使用公开的
> LongMemEval-S cleaned 作为基础语料；但 `run_mem_preprocess.sh` 所需的
> `data/shopping_companion_s_cleaned.jsonl`（以及/或 `data/preferences/` 生成偏好对话、`data/mem_indexes/`）
> 未在发布包中找到，无法运行官方 `mem_search`/`mem_view`。
>
> 请问这些历史对话/索引产物是否可以按研究用途提供，或其在公开渠道的下载位置/生成脚本补丁（含模型版本与随机种子）？
> 若不可公开提供，能否说明可用范围与获取条件？
>
> 我们仅用于学术复现，不会重新分发原始数据；如需要可提供使用说明与引用方式。
>
> 感谢！
> [姓名/单位/用途]

---

## 2. 能否合法运行官方 `mem_search`？

**当前发布物下：NO。** 阻断点在数据而不是代码：

- 官方 `mem_search` 是本地 HTTP 工具（`src/tools/mem_search.py` → `http://127.0.0.1:5632/search`）；
  服务 `src/serve_mem.py` 只加载 `data/mem_indexes/`；
  `data/mem_indexes` 只能由 `src/build_mem_indexes.py --conversation_file data/shopping_companion_s_cleaned.jsonl` 构建，
  而该 conversation 文件/索引均未发布（§1.1/1.2）。官方脚本中也没有下载步骤或 URL。
- 已发布的 train parquet 不携带 haystack sessions；`tools_kwargs` 只有 `conversation_id`（§1.3）。
- 公开可得的 LongMemEval-S 只是基础语料，不能替代混合后的 benchmark 历史（§1.5 的 0 匹配）。
- 因此对任一 train stage_1 任务，**无法在合法访问条件下取得真实历史证据片段/ID**；
  本报告选择任务 `0014b39c4159d54fd5eafcd6c2bd6699_1` 的官方机制检索结果为“输入文件不存在”，
  而不是合成历史（未使用 ground truth/oracle）。

> 合法替代只能是：① 向作者申请官方 artifact；② 用公开 LongMemEval（MIT）+ 公开商品目录自行生成一份
> **明确标注为 AgentRec-X 变体**的历史语料（不可再称官方 benchmark），并记录生成模型/种子/许可；③ 改用有公开
> 历史或相关性标签的数据集。P14-A 运行时对缺失历史继续返回 `history_unavailable`，不伪造。

---

## 3. 全量商品目录的资源需求与可执行索引方案

### 3.1 事实（官方与实测）

| 项 | 值 |
| --- | --- |
| 完整商品数 / 体积 | **1,298,797** / **20,030,340,587 B（≈18.66 GiB）**（HF 数据卡 + HEAD 实测；etag `8d597f78…c596`） |
| 官方索引实现 | **pyserini Lucene**：`src/build_product_docs.py`（流式转 docs.jsonl）→ `python -m pyserini.index.lucene --collection JsonCollection --generator DefaultLuceneDocumentGenerator --threads 8 --storePositions --storeDocvectors --storeRaw` → `src/serve_product.py`（LuceneSearcher，Flask/waitress，`/product_search`、`/product_view`） |
| 官方依赖 | `requirements.txt` 含 `pyserini`（需 **Java 11** + Python 3.11）；`install.sh` 用 `uv venv --python 3.11` |
| 本机环境 | WSL：无 java（apt 候选 `openjdk-11-jdk 11.0.32.1` 可装）；未装 pyserini；Windows 也无 java；**未安装**（本轮不构建全量） |
| 预构建索引 | **没有**：HF 数据集只有 6 个文件；GitHub 0 release；install/run 脚本无索引下载步骤 ⇒ 无法合法复用 |
| 当前本地样本 | 81,637 商品；docs.jsonl 120,676,740 B（≈115 MiB）；raw 前缀 1,288,490,188 B（6.43%） |

### 3.2 全量方案与资源估算（仅方案，不在本轮构建）

```text
磁盘（D: 当前空闲 409.6 GB，足够）：
  raw products.jsonl                       20.0 GB
  docs.jsonl（流式投影，去 reviews）        ≈1.9 GB（按本地样本 1,478 B/商品外推）
  Lucene index（positions+docvectors+raw）  ≈3–6 GB（经验 1.5–3× docs；需实测）
  临时/日志                                ≈0.5 GB
  合计                                     ≈25–28 GB

时间（按本机实测速率）：
  下载（10 workers Range，实测 14.19 MiB/s）    20 GB ≈ 24–45 min
  docs 流式投影（实测 17 MB/s）                 20 GB ≈ 20 min
  Lucene 索引（8 threads，1.3M docs）           ≈30–60 min（需实测）
  合计                                          ≈1.5–2.5 h

内存：
  docs 投影：单行流式，常数级（实测样本进程 <1.6 GB）
  Lucene 建索引/服务：JVM 可配 -Xmx4g–8g；建议 ≥16 GB 机器
  本机 WSL 7.7 GB：构建偏紧（JVM+页面缓存），不建议作为首选；Windows 原生（15.6 GB）或用户自行开启的
  AutoDL 更合适（本轮不自动操作实例）
```

可执行步骤（官方路线，供用户批准后执行）：

1. 下载 `products.jsonl` 到 D:（可用我们的 range 并行脚本；实测 14+ MiB/s）。
2. 安装 Java 11（`apt install openjdk-11-jdk`，或 Windows JDK）+ Python 3.11 + `pip install pyserini`（**不改项目 .venv/requirements**，放独立环境）。
3. `python src/build_product_docs.py --products_file data/products.jsonl --docs_file data/product_resources/docs.jsonl`（流式，不整文件载入内存）。
4. `python -m pyserini.index.lucene --collection JsonCollection --input data/product_resources --index data/product_indexes --generator DefaultLuceneDocumentGenerator --threads 8 --storePositions --storeDocvectors --storeRaw`。
5. `python src/serve_product.py --index_dir data/product_indexes --port 5631`；
   `product_search` 通过 Lucene 查询，`product_view` 用 `searcher.doc(product_id)` 读取 `--storeRaw` 保存的完整商品 JSON ⇒ **无需把 20GB JSONL 载入内存**。

备用路线（无 Java/无 pyserini 时；本轮已验证可用性，不实施）：

- 本机 WSL Python 自带 **SQLite FTS5 可用**（`sqlite_fts5_ok`）；可设计“流式写 docs → FTS5 全文表 + product 表”的磁盘索引，
  查询/详情按 rowid 读取，内存有界；缺点是需新写组件、排序/BM25 权重与官方 pyserini 不完全一致。
- 现有仓库 BM25（`CatalogSearchSource`）是**全内存**索引，1.3M 商品预计需远大于 7.7 GB，不能直接用于全量。

### 3.3 结论

**`FULL_CATALOGUE = FEASIBLE`**：官方方案成熟、磁盘/时间可接受；关键前置是 Java 11+pyserini 依赖与 ≥16GB 内存环境；
本机 WSL 不适合作为构建机（可做 streaming/serving 验证，不宜做 1.3M Lucene 索引）；需要用户批准后再执行（本轮禁止构建）。

---

## 4. 当前 81,637 商品目录的实际覆盖限制

当前 catalogue 为 **products.jsonl 连续字节前缀**：1,288,490,188 B = 源文件 **6.43%**；81,637 商品 = 完整目录 **6.29%**。
（`runs/p14a/logs/catalogue_coverage.json`）

| 统计（train split，未用 test） | 数值 |
| --- | --- |
| single_product 任务：target 在 catalogue 中 | **23 / 400（5.75%）** |
| add_on_deals 任务：**全部** target 在 catalogue 中 | **0 / 400（0%）** |
| add_on_deals 任务：至少一个 target 在 catalogue 中 | 58 / 400（14.5%） |
| 正例 target id 总数：在 catalogue 中 | **84 / 1,449（5.80%）** |
| 前缀 sample 的顶层类目分布（前 5） | Beauty 7,802 / Automotive 7,550 / Electronics Parts & Accessories 6,551 / Tools & Home Improvement 5,671 / Groceries 3,947 |

含义：在 limited catalogue 下，即使检索完全正确，最多也只有约 5.75% 的 single-product 任务能命中参考商品；
add-on 任务在结构上几乎不可能被完整覆盖（需多个目标同时命中）。这解释了 P14-A E2E 的 target 不命中（不是链路错误），
也说明 **P14-B 若做 preference-aware ranking，必须先换更大/完整目录**，否则评估被 candidate reachability 主导。

---

## 5. 单商品输出数量是否正确（修复）

**官方定义**：single-product = “find one product”；evaluate.py 解析 `@REC::([0-9,]+?)@`
（`src/evaluate.py:228`），`metric["n"] = len(products)`，并按产品逐一判断；add_on 则要求 `metric["n"] == metric["n_gt"]`
（`src/evaluate.py:388`）。因此单商品 FINALIZE 应呈现 **1 件**商品。

**发现的错误（P14-A 旧轨迹 `…_1`）**：FINALIZE 把 `candidate_ids` 全部当最终推荐 → `recommendation` 列表 **10 件**。
位置：`recommendation/shopping_companion/planner_bridge.py::_finalize`（旧实现 `result.final_ids = ids`，`recommendations` 遍历全部 ids）。

**最小修复**：`BridgeConfig.final_count = 1`；`_finalize` 只呈现 `ids[:final_count]`；
轨迹同时保留 `candidate_count` 与 `presented_count`（ledger 仍保存全部候选用于审计，未被删除）。
`recommendation/shopping_companion/runtime.py` 的 `BridgeConfig` 默认值即 1。

**修复后证据**：`runs/p14a/trajectories/real_limited_catalogue_…_1_diagnostic_stage2.jsonl`
→ `recommendation` 长度 **1**、`presented_count=1`、`candidate_count=10`；单商品任务默认路径（无历史）为 `awaiting_clarification`、0 推荐（不伪造）。

---

## 6. ASK / FINALIZE 是否存在状态错误（修复）

**旧行为（P14-A 旧轨迹 `…_1`）**：

1. stage_1 planner 循环：`RETRIEVE`（memory）→ `history_unavailable` → 计划 `ASK`，stage_1 终止于 ASK（`needs_user_input=True`）；
2. runtime **仍继续** 执行 stage_2 planner 循环（独立的一次 `run_planner`），用原 question 做真实检索并 `FINALIZE`；
3. 最终 trajectory `terminal.status=succeeded`、`succeeded=true` —— 即 **ASK 尚未解决，却被同一任务当作已完成**；
   且 ASK 的问题（“历史语料未发布，无法恢复偏好”）实际不是用户可以回答的澄清，而是数据阻塞。

**最小修复**（不改 Planner 动作契约，ASK/RETRIEVE/RERANK/FINALIZE 语义不变）：

- `runtime.py` 计算 `stage_1_awaiting`（stage_1 终止于 ASK，或状态为 `history_unavailable`/`clarification_pending`）；
- 默认：**不执行 stage_2**，`stage_2_executed=false`，`terminal.status="awaiting_clarification"`、`succeeded=false`，
  并记录 `stage_2` `not_run` 的原因（官方两阶段协议没有 stage_1 偏好答案可交给 stage_2）；
- 仅当显式 `--diagnostic-stage2-without-preferences` 时，stage_2 作为 **明确标注的诊断续跑**（`stage_2_diagnostic_only=true`），
  仍保持 `terminal.status="awaiting_clarification"`、`succeeded=false`，不计入任务成功；
- `awaiting_clarification`、`stage_2_executed`、`stage_2_diagnostic_only` 全部写入 trajectory/eval，便于复核；
- 原有 “ASK 与 FINALIZE 来自两个不同 `run_planner` 分支”的事实被显式记录，但不再让 FINALIZE 冒充 ASK 的答案。

**修复后真实轨迹**：

| 轨迹 | terminal | stage_2 | recommendation | 说明 |
| --- | --- | --- | --- | --- |
| `…_1.jsonl`（默认） | `awaiting_clarification` / succeeded=false | 未执行 | 0 | 无历史 → 等用户，不伪造 |
| `…_2.jsonl`（add-on，默认） | `awaiting_clarification` / false | 未执行 | 0 | 同上；bundle 仅结构标记 |
| `…_1_diagnostic_stage2.jsonl`（显式诊断） | `awaiting_clarification` / false | 已执行（diagnostic） | **1** | 工具链诊断证据，不当作任务成功 |

评测文件（`.eval.json`）新增 `task_completed`、`awaiting_clarification`、`stage_2_executed`、`stage_2_diagnostic_only`、
`presented_recommendation_count`、`single_product_presentation_ok` 字段；诊断轨迹的 `single_product_presentation_ok=true`。

---

## 7. 最小修复与定向测试

**改动的文件（仅 P14-A 语义相关，未改旧 Planner 动作/旧 Serving）**：

| 文件 | 改动 |
| --- | --- |
| `recommendation/shopping_companion/planner_bridge.py` | `BridgeConfig.final_count=1`；`_finalize` 只呈现 1 件（ledger 仍保留全部候选） |
| `recommendation/shopping_companion/runtime.py` | `stage_1_awaiting` 判定；默认不跑 stage_2；`awaiting_clarification` 终态；诊断 opt-in；轨迹/eval 字段 |
| `experiments/p14a_run_task.py` | 新增 `--diagnostic-stage2-without-preferences`；诊断输出加 `_diagnostic_stage2` 后缀 |
| `recommendation/shopping_companion/evaluator.py` | 输出任务完成/等待澄清/单商品数量检查字段 |
| `tests/test_p14a_planner_runtime.py` | 更新为：默认 ASK→`awaiting_clarification` 且不跑 stage_2；诊断路径只呈现 1 件且不算成功；合成偏好路径成功且 1 件 |
| `runs/p14a/trajectories/*` | 用修复后代码重新生成 3 条真实轨迹（默认 2 + 诊断 1） |

**定向测试结果（WSL，小批次）**：

| 测试 | 结果 |
| --- | --- |
| `pytest tests/test_p14a_namespace.py tests/test_p14a_catalogue_search.py tests/test_p14a_product_info.py tests/test_p14a_planner_runtime.py tests/test_p14a_dataset.py tests/test_p14a_memory.py` | **74 passed in 63 s** |
| 旧核心默认行为回归：`pytest tests/test_api.py tests/test_serving_fusion.py tests/test_planner_contract.py tests/test_docs.py` | **142 passed in 26 s** |
| 独立泄漏复核（`verify_p14a_isolation.py`） | **ISOLATION_OK**（3 条轨迹中 oracle marker/ground_truth 键/target id 命中均为 0） |
| 未运行 | 全量 pytest、reserved test、官方 test split、全量索引构建、GPU 任务 |

---

## 8. P14-B 是否具备启动条件

**不能直接启动“preference-aware 训练”**，但可以启动不依赖真实偏好监督的准备工作。核心判定：

| 条件 | 状态 |
| --- | --- |
| 真实历史/偏好监督数据 | **不具备**（MEMORY_DATA = NOT_PUBLICLY_FOUND） |
| 完整或足够大的候选目录 | 方案可行但未构建（FULL_CATALOGUE = FEASIBLE，非已具备） |
| Agent 轨迹/工具链语义 | 已修复（AGENT_TRAJECTORY = FIXED），可作为 P14-B 的接入底座 |
| 旧系统不污染 | 保持（默认 Serving/旧核心未改；142 项旧回归通过） |

**建议的替代数据策略（按推荐顺序）**：

1. **向作者申请**官方 `shopping_companion_s_cleaned.jsonl` / `data/mem_indexes`（§1.7 询问信）；获批后可直接恢复官方 stage_1/E2E。
2. **构建 AgentRec-X 自有可复现历史变体**：以公开 MIT 的 LongMemEval-S cleaned 为真实长程对话底座 + 公开商品目录，
   用**确定性、带 seed、记录模型/许可**的偏好对话合成器（可复用本仓库既有 preference construction 思路）生成
   “AgentRec-X Shopping Companion variant”；明确标注不是官方 benchmark，用于 P14-B 的协议/管线验证与偏好回归测试。
3. **先做目录与可达性工程**：按 §3 方案构建更大/完整 catalogue（或至少按目标覆盖分层采样），解决 §4 的 5.75% 可达率问题。
4. **P13-R 路线仅作 retrieval-label 对照**：Electronics + BGE 已有观测正例与 explicit fixed-pool 协议，可作为
   “不用个性化偏好”的检索/排序回归，但不能替代 preference supervision。

**P14-B 建议的启动顺序**：先解决历史数据（方案 1 或 2）与候选覆盖（方案 3 的最小可行版本），再训练/评测任何
preference-aware scorer；在此之前只能做协议、adapter、可复现性和负结果工作。

---

## 附：本次核实产生的证据文件

- `runs/p14a/logs/longmemeval_provenance_check.json`（公开 LongMemEval 核验 + 0 匹配证明）
- `runs/p14a/logs/catalogue_coverage.json`（train 正例覆盖统计）
- `runs/p14a/logs/isolation_check.json`（修复后轨迹隔离复核）
- `runs/p14a/logs/e2e_*.time.log` / `*.stdout.log`（修复后 3 条真实轨迹运行日志与资源）
- `runs/p14a/trajectories/`（默认 2 条 + 诊断 1 条轨迹 + 对应 eval）
- 代码改动见 `P14A_SOURCE_DIFF.txt`（已重新生成）
