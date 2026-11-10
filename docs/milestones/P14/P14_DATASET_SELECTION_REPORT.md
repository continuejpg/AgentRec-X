# P14-D — Public Benchmark Feasibility & Selection 报告

> 性质：一次性、有终点的公开数据集选型核实。只做**真实数据读取与结构/监督/隔离验证**，不修改 Agent 核心、不训练、不建索引、不运行官方评测、不启动 P14-B。
> 环境：Windows 11 + WSL2（CPU-only）；未使用 GPU、未连接/操作 AutoDL；本地缓存与证据在 `data/p14d/`、`runs/p14d/`（git-ignored）。
>
> 本次实际下载并解析的数据（sha256 见 `data/p14d/manifest.json`）：
> - PAHF `data/shopping/*`：4 个 phase JSON + 4 个 persona JSON，合计 ~2.6 MB（MIT）；
> - VitaBench 2.0 `tasks.json`：131,234,835 B（HF `meituan-longcat/VitaBench-2.0`，sha256 `3a05f7e2…`）。

## 0. 结论摘要

| 判定 | 结果 |
| --- | --- |
| **PAHF** | **LIMITED**（可作购物偏好/ASK/反馈辅助 Benchmark；不能作为检索式推荐主 Benchmark） |
| **VITABENCH** | **LIMITED**（可作长期记忆/主动性辅助 Memory Benchmark；不是电商推荐、无训练划分、依赖其 runtime） |
| **SELF_BUILT_BENCHMARK** | **REQUIRED**（AgentRec-X 的购物检索 + 偏好感知推荐主路线必须自建可追溯 Benchmark） |
| **P14_NEXT_STEP** | 冻结自建 Benchmark 规格并用现成真实商品底座构建最小可行集；PAHF 作为 ASK/feedback 辅助、VitaBench 子集作为 memory 辅助（不建第二套 Runtime） |

**主路线**：自建 AgentRec-X Shopping Preference Benchmark（真实商品目录 + 可追溯偏好监督/协议），**不是** PAHF 或 VitaBench 二选一。
两者各有明确辅助职责：PAHF 验证“偏好如何通过 ASK/反馈逐轮形成”，VitaBench 验证“跨会话长期记忆与偏好更新”；它们都不具备“真实商品检索 + 个性化排序 + 可训练评测”的组合，且都不是无偏的真实购买日志。

---

## 1. PAHF 实际文件与样本 Schema

来源：`facebookresearch/PAHF`（MIT；Meta Superintelligence Labs 等）。`data/shopping/` **确实可读取**，共 8 个 JSON：

| 文件 | 大小 | sha256（前 16） | 内容 |
| --- | ---: | --- | --- |
| `phase1.json` | 529,287 B | `f299c2ae5a9b0ec5` | 900 个任务（训练阶段） |
| `phase2.json` | 529,287 B | `3b4dc65c31d1b340` | 900 个任务（评测阶段） |
| `phase3.json` | 529,606 B | `5856d7d68a4d6464` | 900 个任务（更新偏好后训练） |
| `phase4.json` | 529,597 B | `8078969d28deb893` | 900 个任务（更新偏好后评测） |
| `original_persona.json` | 150,862 B | `d81e73d02b211613` | 20 个 persona 文本（初始） |
| `updated_persona.json` | 211,422 B | `4d45718e3f487430` | 20 个 persona 文本（更新后，含“过去偏好”叙述） |
| `persona_info.json` | 153,403 B | `15c8297cf0dff4b8` | 20 用户 × 10 商品类目的结构化偏好 |
| `updated_persona_info.json` | 153,403 B | `8678fc55f0e5a1b1` | 更新后的结构化偏好 |

**单任务样本**（phase1 真实第 0 行）：
```json
{"product": "microwave oven",
 "Option A": ["side-sliding door", "convection fan heating", "membrane touch keypad"],
 "Option B": ["top-hinged pull-down door", "inverter microwave control", "membrane touch keypad"],
 "Option C": ["right-hinged swing door", "infrared radiant heating", "membrane touch keypad"],
 "User": "Emma", "Task": "Looking to buy a microwave oven that matches what I want", "gt": null}
```
- 每个任务 = 1 个商品类目 + **3 个手工构建的选项包（每包 3 条属性值）** + 隐式 `D) Do not buy anything`；共 **900 行/phase、20 用户、10 个商品类目**；每用户每 phase 45 行。
- `gt` 是**评分标签**：`A/B/C` 或 `null`（`null` ⇒ 正确答案为 D）。phase1 `gt` 分布：`D=300, A=205, B=216, C=179`；phase2：`D=300, A=209, B=184, C=207`；phase3：`D=619, A=109, B=96, C=76`；phase4：`D=610, A=97, B=106, C=87`。
- `persona_info` 结构：`persona_info[user].persona_info[product][attribute] = {like_most, like_second[], dislike[]}`（每类目 3 个属性维度，每维度 6 个候选值）。
- 商品空间**没有真实 catalogue/价格/店铺/检索**：只有类目名与 3 个选项包的属性值文本；这不是“从商品目录中检索并排序”的推荐任务。

## 2. VitaBench 2.0 实际文件与样本 Schema

来源：`meituan-longcat/VitaBench-2.0`（repo MIT；HF 数据集单文件 `tasks.json` 131,234,835 B，sha256 `3a05f7e2…`；HF 无 dataset card/README，**数据集许可未单独声明**）。代码 repo ~3.4 MB（`src/vita` 包，`pip install -e .`）。

**数据规模（逐条解析 tasks.json 核实，与 README 一致）**：

| 项 | 值 |
| --- | --- |
| 用户（用户级 task） | **56** |
| 子任务（subtasks） | **771** |
| 域分布 | delivery **404** / instore **223** / ota **144** |
| 每用户子任务数 | 10–20（中位 13，均值 ~13.8） |
| 语言 | 中文为主（instruction CJK 比例均值 0.911；rubric 0.791）；README 称 English version coming soon |
| 划分 | **数据中没有 train/val/test split 字段**（仅 56 用户评测集） |

**单子任务样本**（user 0 / subtask 0，delivery）字段：
`subtask_id, domain, instruction, current_time, start_date, end_date, environment, user_scenario, evaluation_criteria, rubric, target_product_ids, interactions, historical_chat, historical_behavior, skill_tested, user_intention, n_changed, n_chat, n_noise, noise, task_turn_num, message_history`。

- `environment`：候选池（该子任务 `stores` 43 个；instore 样本 `shops` 43 个；OTA 样本 `hotels` 45 个），每个实体含 `name/score/location/tags/products`，**原始 JSON 还含 `store_type`/`product_type`（target/distraction）与 `distraction_reason`（明确写出“违反 rubric3/5”）**。
- `user_scenario`：`user_profile` + `personalized_preference_memory`（**ground-truth 偏好**，含 `current` 与 `preference_tag_change_history`），模型注释写明“All the information that will be sent to the user simulator”。
- `evaluation_criteria`：`expected_states`（`expected_orders`、`state_rubrics`）+ `overall_rubrics`，模型注释写明“will be sent to the evaluator”。
- `interactions`：该子任务之前可用的历史交互（样例 76 条；带 `date/behavior/dialogue` 等），用于更新 memory；`rubric`、`target_product_ids`、`user_intention`、`noise` 也在子任务行内。
- 评测框架：`vita run --domain personalization --memory-type {null,groundtruth,full_context,rewrite,rag,rag_cache}`，agent/user simulator/evaluator 三个 LLM 均通过 OpenAI-compatible endpoint；RAG memory 需要 embedding endpoint；评分是 LLM judge（rubric/outcome），支持 `--num-tasks/--task-ids` 抽样、`--num-trials` Avg@N。

## 3. 两者真实数据规模与监督类型

| | PAHF（shopping） | VitaBench 2.0 |
| --- | --- | --- |
| 数据量 | 4×900 任务；20 users；10 类目；总 JSON ~2.6 MB | 56 users / 771 subtasks；131 MB |
| 任务单位 | 单任务 = 3 个选项包 + D（多选一） | 用户 = 10–20 个连续子任务（跨域生活服务） |
| “历史”来源 | **无预置对话历史**；memory 从空开始，在训练 phase 的 ASK 回答与 post-action feedback 中**在线学习**（默认 3 个 learning iteration），phase3/4 载入此前 memory 后适配更新偏好 | 每子任务提供 `interactions` 历史交互序列；子任务之间 memory 由框架更新并注入，支持 6 种 memory 后端；有目标/干扰实体候选池与逐条 rubric |
| 监督类型 | **`gt`（A/B/C/D）** + 规则化 post-action feedback（用 gt/persona）+ LLM 模拟用户（persona）对 ASK 的回答；无人工/真实购买标签 | **LLM judge 的 rubric/expected_order 评分** + 隐藏 ground-truth preference memory（仅 user simulator / groundtruth memory 使用）+ 环境 target 实体（评测用） |
| 是否真实用户/购买 | 否：**persona 是模拟用户设定**，feedback 是模拟/规则生成，不是真实购买日志 | 否：生活服务任务 + 模拟用户 + LLM 评测；场景/交互是构造数据 |
| 训练用途 | 可做偏好学习/反馈学习的训练与评测（phase1/3 train，phase2/4 eval） | 官方定位 **evaluation benchmark**（Personalization leaderboard）；**无训练 split** |

## 4. 历史缺失与 Oracle 泄漏风险

### PAHF
- **不存在“历史语料缺失”问题**（与 Shopping Companion 相反）：PAHF 不提供长对话历史，而是把历史/偏好学习放在运行过程中（ask→answer→memory；wrong answer→feedback→memory）。
- **隔离实测**：agent prompt 只由 `scenario_text + options_text` 构造（`load_shopping_data()` 的 `mc_gen_raw`），`gt`/`persona_info` **不进入 agent prompt**；`gt` 只用于评分与训练阶段反馈；persona 只进入 simulated human。`prompts/shopping_prompts.py` 中无 `{gt}`/`persona_info` 插值（grep 计数 0）。
- **评测隔离**：phase2/phase4 调用时 `enable_pre_feedback=False, enable_post_feedback=False`（代码注释“Evaluation only (NO feedback)”），不会在评测集上继续学习/更新 memory；phase1/3 的 feedback 才是训练信号。
- **残余风险**：① 任何自建 harness 若把 `gt`/persona 注入 agent 即泄漏；② 我们的简单“persona→正确选项”确定性规则只能重建部分 `gt`（900 行中 229 行不一致；persona 的 like_most/like_second/dislike 与 gt 生成规则并非同一简单规则），因此**不要自创标签规则**，应直接使用官方 `gt` 或官方 feedback 代码；③ persona 是模拟人设，不能当真实购买偏好写进对外结论。

### VitaBench 2.0
- **有完整历史**（`interactions`）与偏好变化记录（`preference_tag_change_history`），子任务按时间顺序执行，框架在子任务间更新 memory。
- **原始数据内确实含泄漏标记**：environment 里 `store_type/product_type = target|distraction`、`distraction_reason` 直接写出“违反 rubric 哪一条”；`target_product_ids`、`rubric`、`personalized_preference_memory` 也在数据行内。
- **官方框架的隔离机制已验证**：领域数据模型（`Store/StoreProduct`、`Shop/ShopProduct`、OTA models）只声明用户可见字段，**不包含** `store_type/product_type/distraction_reason`，`DB.load` 用 pydantic 模型装载时丢弃这些额外字段 ⇒ 工具返回给 agent 的内容不含泄漏标记；`evaluator/outcome_evaluator.py` 还显式定义 `FIELDS_TO_STRIP={shop_type,store_type,product_type,distraction_reason,quantity}` 与 `LEAKY_META_FIELDS` 用于 judge prompt。`user_scenario`（偏好真值）与 `evaluation_criteria`（rubric）分别被模型注释限定为“给 user simulator”与“给 evaluator”。
- **残余风险**：任何**自建 adapter/自建 runtime** 若直接读原始 `tasks.json` 字段、或把 environment 原始 dict 透传给工具/模型，会泄漏 target/distraction 标签和 rubric。我们构造的 minimal agent-visible sample 已排除这些字段，隔离检查通过（`runs/p14d/samples/isolation_check.json`）。
- 另：`user_intention`（隐藏意图）只在 agent 主动问且 skills=`proactive` 时披露；这属于框架设计，但要求按官方 orchestrator 使用。

## 5. 商品搜索与推荐底座如何接入

| 能力 | PAHF | VitaBench 2.0 |
| --- | --- | --- |
| 真实商品目录/搜索 | **无**（只有 10 个类目 + 3 个手写选项包；无价格/店铺/属性检索） | **无电商商品目录**（生活服务实体：餐厅/门店/酒店/车次/景点；候选池由环境提供） |
| 与现有 BM25/BGE/Search/Ledger 的兼容 | 只能把“选项包”当固定候选做选择；无法使用 Search/Ledger；不能训产品 Ranker | 必须先实现其 domain tools/environment 才能搜索候选；与我们的 Search/Ledger 不通用；可复用其候选池但需要适配层 |
| 兼容现有 M5/Planner | Planner 的 ASK/RETRIEVE/FINALIZE 语义可映射到 PAHF 的 ask/choose；M5 需要把其 SQLite/FAISS memory 当外部 memory bank 适配（非直接复用） | 它的 `BaseMemory.read()/update()` 接口很干净，可用 M5/M9 实现（但需要跑其 orchestrator/评测器） |
| 是否需要第二套 Runtime | 若要跑官方协议需要（PAHF 自带 agent 循环；或把其任务输入移植到我们的 Planner，但需要自写 human simulator/evaluator） | 是：官方 run 需要其 `vita` CLI、domain tools、user simulator、evaluator；不应复制进本项目 |

## 6. Agent 能力能否训练与评价

- **PAHF**：能训练/评价“偏好恢复 + ASK + 反馈更新 + 是否购买(D)”，指标 acc/ff（feedback frequency）/ACPE（跨 learning iteration 的个人化误差）；不能训练/评价检索、排序、真实商品推荐的 Recall/NDCG/HR。它的 900×4 规模小，适合作为**辅助偏好/交互协议验证**，不适合作为主推荐 Benchmark。
- **VitaBench 2.0**：能评价“长程记忆 + 个性化/主动性 + 生活服务任务完成”，有 6 种 memory 后端、rubric/judge、leaderboard；**没有训练 split**，不能拿来直接训练偏好 Ranker；官方定位 evaluation-only；全量跑需要三个 LLM（agent/user/evaluator）+ 可选 embedding，成本高，可抽样子集。
- 共同点：两者都是**模拟用户/构造数据**；不能把模拟 feedback 或 LLM judge 当成真实购买偏好/点击；对外报告必须区分“真实商品数据”“构造任务”“模拟反馈”。

## 7. 与 Shopping Companion 的区别

| | Shopping Companion | PAHF | VitaBench 2.0 |
| --- | --- | --- | --- |
| 商品/服务底座 | 真实 Amazon-C4/Lazada 商品目录（1,298,797；20 GB；形态：商品） | 手写 10 类目 × 3 选项包（无目录） | 生活服务候选池（餐厅/门店/酒店等；非商品目录） |
| 历史对话 | **官方未发布**（P14-A 判定 NOT_PUBLICLY_FOUND） | 无预置历史，在线学习 | 有逐子任务 `interactions` 历史 |
| 查询 | C4 proxy query（合成） | 手写购物意图 | 中文生活服务指令 |
| 监督 | 单 observed positive + oracle memory（不可用） | `gt` A/B/C/D + 规则/模拟 feedback（phase2/4 无反馈） | rubric/LLM judge + ground-truth preference memory |
| 泄露状态 | 未发布历史是主要阻塞；P14-A 已隔离 oracle | agent prompt 无 gt/persona；训练用 feedback、评测无反馈 | 原数据含泄漏标记，官方模型/评测器会剥离；自建须防泄漏 |
| 适配 AgentRec-X | P14-A 已接入商品工具/轨迹，但无真实历史 | 可作 ASK/feedback 辅助；无商品检索 | 可作 memory/主动性辅助；非电商、eval-only |

**Shopping Companion 的核心缺口（真实历史 + 完整目录）在 PAHF/VitaBench 上并不能“直接补上”**：PAHF 没有真实商品目录；VitaBench 没有电商商品目录/训练划分。两者解决的是“偏好交互/记忆协议”，不是“商品检索与排序监督”。

## 8. 本地及 AutoDL 资源需求

| | PAHF | VitaBench 2.0 |
| --- | --- | --- |
| 数据读取 | 2.6 MB JSON，本机 CPU 秒级可读 | 131 MB JSON，本机 CPU 可解析（实测 json.load 正常） |
| 运行依赖 | `openai>=1.0` SDK 调 GPT 模型（`OPENAI_API_KEY`；SDK 支持 `OPENAI_BASE_URL`，可指向本地 Qwen/vLLM OpenAI-compatible 服务）；可选 FAISS（CPU）与 embedding | `pip install -e .` + OpenAI-compatible agent/user/evaluator LLM + RAG embedding endpoint；官方 CLI |
| 全量运行成本 | 4 phases × 900 × learning_iter：LLM 调用量中等；可只跑 1–2 用户/少量任务子集 | 771 subtasks × trials(默认 3–4) × 多轮 × (agent+user+evaluator) 调用；全量昂贵，可 `--num-tasks/--task-ids` 抽样 |
| GPU | 不需要（数据/驱动推理都走 API；若本地 Qwen 服务才需要 GPU） | 不需要（走 API）；若本地部署 Qwen/embedding 才需要 GPU |
| AutoDL | 可用但**不必须**；本轮未启动/未操作 | 同上；不建议在全量评测前占用 GPU，子集 API 评测即可 |
| 本地存储 | 已缓存 `data/p14d/raw/pahf/` | 已缓存 `data/p14d/raw/vitabench/tasks.json`（131 MB）；不构建任何索引 |

> 注：本轮没有运行任何官方 agent/eval（需要 API key），也没有安装 PAHF/VitaBench 的运行依赖；只做真实数据读取、schema/隔离验证与最小输入构造。

## 9. 下一阶段明确的唯一主路线

**唯一主路线：自建 AgentRec-X Shopping Preference Benchmark（真实商品候选 + 可追溯偏好监督/协议），P14-B/P14-C 均围绕它。**

理由（不是“两个都不好”，而是职责不匹配）：
1. PAHF 只能验证**偏好/ASK/反馈**，无法覆盖 Search、真实商品候选、推荐排序质量（C1 明确要求不能只做固定候选选择）→ 只能辅助。
2. VitaBench 只能验证**长期记忆/主动性**，非电商、无训练 split、依赖其领域 runtime 与 LLM judge → 只能辅助。
3. Shopping Companion 的**真实历史未发布 + 目录仅 6.29%**，因此 P14-B 需要自己的、可追溯到真实商品与偏好来源的数据与协议。
4. 现有项目已有可复用底座：P13-R Electronics **76,918 个真实商品** + BGE/BM25 检索、P11/P12 任务/证据结构、P14-A adapter/轨迹/namespace/工具链。自建 Benchmark 不需要重新发明基础设施。

## 10. 是否必须自建补充 Benchmark

**是，必须自建（REQUIRED）**，且它是 P14 的**主 Benchmark**，不是补充：
- 需要一组**候选商品来自真实目录**（Electronics/P13-R 或未来 Shopping Companion 完整目录），有 namespace、可复核；
- 需要**偏好监督的可追溯来源**：要么真实历史（向作者申请/长期），要么公开可复现协议（LongMemEval-S MIT 底料 + 确定性偏好注入，标注为 AgentRec-X 变体），要么人工/半自动构造的偏好证据并记录来源与生成器版本；
- 需要有**独立的 evaluator-only 文件**（目标/偏好/rubric）与**agent-visible** 文件分离，沿用 P14-A 的 oracle 隔离方式；
- 需要 train/dev/reserved-test 时间/用户切分（不能用同一批任务调参再报告），并记录“真实商品 / 构造任务 / 模拟偏好”三层来源；
- PAHF 与 VitaBench 作为**辅助**：
  * PAHF：验证 ASK/pre-action clarification、post-action feedback、偏好逐轮更新的协议正确性（可作为 Planner ASK 与 M5 更新的外部压力测试；只跑子集，不并入主评测分数）。
  * VitaBench（子集 2–3 users）：验证 M5/M9 memory 更新/检索/冲突处理的外部指标；**不复制其 runtime**，要么在其官方 harness 里实现 `BaseMemory` adapter 指向 M5，要么把它当纯外部对照，不参与 P14-B 训练。

### 不建议使用（明确排除）

- 不建议把 **PAHF 当作主推荐 Benchmark**：没有真实商品检索/排序，只有 3 个手写选项包 + D；会重演“固定候选/小规模选择题”的失配。
- 不建议把 **VitaBench 当作商品推荐主 Benchmark**：生活服务域（外卖/到店/OTA），不是商品目录；无训练 split，评测是 LLM judge rubric；适配成本高且不提供商品排序监督。
- 不建议使用 **PAHF 的 persona 作为真实购买偏好**、**VitaBench 的 LLM judge 分数作为商品相关性标签**、或把二者的模拟用户反馈当作真实用户反馈对外宣称。
- 不建议为这两个 benchmark 各建一套 Agent Runtime（D 要求：不要两套 Runtime）。

### 最小非 Oracle 输入构造样例（真实数据）

- PAHF：`runs/p14d/samples/pahf_agent_input_sample.json` — 由 phase1 第 0 行按官方 `load_shopping_data()` 规则构造：scenario + A/B/C/D + 可用动作 + 空 memory；**不含 gt/persona**（隔离检查 0 命中）。
- VitaBench：`runs/p14d/samples/vitabench_agent_input_sample.json` — 取 user 0/subtask 0：instruction、domain、候选池统计、公开实体样例、历史交互条数/样例、memory 接口；**不含** target/distraction 标记、target ids、GT preference memory、rubric（隔离检查 0 命中）。
  这两个样例证明“agent 可见输入”可以和“evaluator/simulator 数据”严格分离，不需要为了跑 benchmark 而把答案送进上下文。

---

## 11. 最终判定

| 判定 | 结果 |
| --- | --- |
| `PAHF` | **LIMITED** — 数据和隔离真实可核；可做购物偏好/ASK/feedback 辅助；无真实商品检索/排序，不能作主 Benchmark |
| `VITABENCH` | **LIMITED** — 数据和历史真实可核；可做长期记忆/主动性辅助；非电商、eval-only、无训练 split、依赖其 runtime，不能作主 Benchmark |
| `SELF_BUILT_BENCHMARK` | **REQUIRED** — 主路线必须自建，基于真实商品目录 + 可追溯偏好来源与协议 |
| `P14_NEXT_STEP` | **P14-E：冻结自建 Benchmark 规格并构建最小可行集**，具体：① 定义 task schema（agent-visible vs evaluator-only）、split（train/dev/reserved-test）与来源标注；② 用 P13-R Electronics 76,918 真实商品 + 现有 BGE/BM25 + P14-A adapter 构建 200–500 个带候选 provenance 的任务；③ 偏好监督用“可复现生成器（版本/seed/许可记录）”或人工构造并记录来源；④ 同时跑 PAHF 子集（2 users × 5 tasks）验证 ASK/feedback 协议、VitaBench 子集（2–3 users）验证 memory 更新；⑤ 不训练 Ranker、不建全量索引、不复制第三方 runtime，直到该 Benchmark 与协议冻结。 |

> 本轮不运行任何官方评测（无 API key、未安装依赖），也不把 README 的说法当数据证据；上面的数量、字段、隔离与泄漏结论均来自实际下载/解析的真实文件。数据缓存与样例在 `data/p14d/`、`runs/p14d/`（git-ignored），未 commit/push、未使用 GPU、未操作 AutoDL。
