# AgentRec-X 全项目实验与问题总结

> 截至 2026-11-10。本文只整理已有报告与结果，不重跑实验；不同实验的数据集、指标和测试阶段不同，不能横向相加或比较。

## 1. 项目简介

AgentRec-X 是一个面向电子产品购物的 LLM Shopping Agent。用户可以用自然语言找商品、比较商品、澄清偏好；系统会调用检索、记忆、证据和推荐工具，最后通过 Decision Gate 检查后再给出推荐。项目最终完成了可本地运行的 Web Demo、真实 Qwen Base 规划、MCP 工具链和多轮记忆闭环原型。

## 2. 核心指标总览

| 模块 | 关键指标 | 结果/状态 | 口径与来源 |
|---|---|---|---|
| 商品目录 | 可用 Electronics 商品 | **76,263** 件 | P14-E / P14-D2；catalogue 接受量 |
| BGE 索引 | 向量维度 / 文件 | **768 维 / 约 213 MB** | P14-D2；76,263 个身份对齐 |
| C4 Proxy Test | BM25 / BGE Hit@100 | **18.60% / 50.39%** | 129 条冻结 proxy test；不是真实搜索 |
| C4 Dev 135 | BM25 / BGE Hit@100 | **20/135 / 66/135** | 旧 Dev；与 129 条 test 不同 |
| ESCI 外部评测 | 查询 / 标注候选 | **8,956 / 181,701** | 官方 English-US Task 1 test |
| ESCI semantic gain | BM25 / BGE nDCG@10 | **0.6907 / 0.7366** | 主结果；paired 差 +0.0459 |
| ESCI official-script gain | BM25 / BGE nDCG@10 | **0.6691 / 0.7144** | 官方脚本 S/C 映射；敏感性口径 |
| 推荐排序（P14-B3-Eval100） | BM25 / V0 / Qwen3 Reranker | **46/73 / 67/73 / 58/73** | 73 条候选可达 recommend；标签 provisional |
| 真实历史消融（C4 Dev 135） | Query / Real / Shuffled Hit@1 | **13 / 15 / 16** | C4 proxy；真实历史未超过 shuffled |
| Planner 选择（P14-C1） | Rule / Qwen Base / SFT | **12/15 / 15/15 / 12/15** | Dev20，小样本；最终保留 Base |
| Planner 延迟（P14-C1） | Base / SFT 平均 | **3.1 s / 8.5 s** | 不同服务与输出长度，只作该实验对照 |
| MCP / Skills | 工具 / Skills | **默认 7 个；Qwen 时 8 个 / 2 个** | P17 / P16 |
| Qwen Live 历史验收 | 真实 GPU 浏览器 / 全量 E2E | **2/2 / 9/9** | BGE 优化前的 Qwen Live 版本 |
| 当前 Web 回归 | CPU Live / Qwen Replay | **7/7** | BGE 优化后的最终提交 |
| ASK → Memory → Resume | 最终 Gate 商品 | **B00M0KUF44** | 修复后真实闭环；14 个 eligible |
| BGE 首次查询 | 优化前 → 优化后 | **16.28 s → 0.552 s** | BFF 三次冷启动平均；不是模型推理提速 |

### 三个先读口径

第一，**C4 与 ESCI 不是同一种数据**。C4 的 query 由商品评论构造，只能证明“在 proxy 上 BGE 比 BM25 好”；ESCI 是真实购物 query，但仍只是在官方 judged candidates 内排序，不是全库召回。

第二，**ESCI 的 semantic gain 与 official-script gain 是两套不同口径**。主结果是 E=1、S=0.1、C=0.01、I=0；官方仓库评价脚本使用 E=1、S=0.01、C=0.1，因此两个数字不能混用。

第三，**合同测试、历史 GPU 验收和当前提交回归不是一回事**。Qwen Live 的 9/9 是早期冻结版本的真实 GPU 结果；BGE 优化后的当前提交只确认了 CPU/Replay 7/7 与 Qwen 合同测试，不能把历史 9/9 说成最新提交复跑。

## 3. 重要实验与结果

### 实验一：BM25 与 BGE 检索

**目的：** 判断语义检索是否比关键词检索更能找到目标商品。

C4 是评论衍生的 proxy query；C4 Dev 135 是旧的历史消融集，后来的 129 条是独立冻结的 proxy test，两者不能混为同一个测试。C4 proxy test 上，BM25 Hit@100 为 18.60%，BGE 为 50.39%。C4 Dev 135 上，BGE Hit@100 为 66/135，BM25 为 20/135。

ESCI 使用真实购物查询和人工相关性标签，但只对官方给出的已标注候选排序。8,956 条查询上，BM25/BGE 的 semantic gain nDCG@10 为 **0.6907/0.7366**，BGE 高 **0.0459**，95% CI **[0.0420, 0.0497]**；全候选 nDCG 为 **0.8184/0.8449**。

**结论：** 在 proxy 和 ESCI 两个检索任务中，BGE 都优于 BM25。

**限制：** C4 不是真实搜索日志；ESCI 是 judged-candidate reranking，不是全商品库召回。

### 实验二：个性化记忆与推荐排序

**目的：** 判断 V0 偏好评分、BM25 和 Qwen Reranker 是否能改善推荐。

在 P14-B3-Eval100 的 **73 条候选可达推荐任务**上，BM25 Top-1 为 **46/73**，V0 Scorer 为 **67/73**，Qwen3 Reranker 为 **58/73**。V0 明显更好，但标签由规则 labeler 生成、没有人工核验，和 V0 的评分逻辑存在同源风险，不能代表真实用户效果。

在另一组 C4 Dev 135 历史消融中，只给 Query 的 Hit@1 是 **13**，真实 History 是 **15**，打乱 History 是 **16**。真实历史略高于 Query，但没有超过 shuffled 负对照。

**结论：** 记忆和偏好机制已实现；真实个性化带来的稳定收益**没有被证明**。

### 实验三：Qwen Planner 与 SFT

**目的：** 判断 QLoRA SFT 能否比 Rule / Qwen Base 更好地完成工具规划。

P14-C1 用 40 个 Train 任务、134 条合法动作样本训练 QLoRA，在 Dev20 的 15 条推荐任务上比较：

| Planner | Top-1 正确 | 平均延迟 |
|---|---:|---:|
| Rule | 12/15 | 本地 CPU，不可直接与远端 Qwen 比较 |
| **Qwen Base** | **15/15** | **3.1 s** |
| Qwen SFT | 12/15 | 8.5 s |

SFT 的失败模式与 Rule 相似，输出更长、推理更慢。**最终采用 Qwen Base，不部署 SFT Adapter。** 该结论只基于小样本 Dev20，不能证明 SFT 在所有 Planner 任务上都更差。

### 实验四：Agent 系统能力

P15 的 Evidence RAG 能把商品事实与偏好来源绑定起来，缺失字段保持 UNKNOWN，不把未知值编成已知。P16 通过了商品比较和偏好澄清两个 Shopping Skills。P17 默认提供 7 个 MCP 工具；配置 Qwen Live 后增加 `run_agent_turn`，共 8 个。

P18 的 20 条 Reserved Test 全部完成，12 条推荐任务中 Top-1 正确 11 条，false ASK / false refusal / hard violation 为 0。但这些是**模拟规则标签**，只用于检查工作流泛化，不是真实用户准确率。Web 的 CPU/Replay 浏览器回归最终为 7/7。

### 实验五：真实 Qwen 多轮闭环

早期真实 Qwen Live 用 RTX 4090 完成了真实 `/plan`、MCP、Gate 和浏览器测试：Qwen 专项 2/2，完整 E2E 9/9（7 条 CPU/Replay + 2 条真实 Qwen）。当时 Qwen 并未对测试 prompt 主动 ASK，因此“真实 Qwen ASK”仍未被触发。

后续专用 ASK 测试首次达到 **LIMITED**：真实 Qwen 会 ASK，M9 会写入，但“一次回复”没有撤回旧 wired 偏好，最终 Gate 选到了与当前 wireless 要求冲突的商品。修复后：

- 新增“当前任务约束”，与长期 M9 历史分开；
- 用户回复后重新评估同一候选池；
- Gate 在最终推荐前重新检查当前约束。

真实闭环结果：输入 15 个候选，排除 `B091SF1545` 后 **14 个 eligible**；真实候选重新评估耗时约 **1.05 ms**；最终 Gate 选择 **B00M0KUF44**，有明确 wireless 证据。此前看到的 `B00NFL0E2A` 只是初始商品卡，不是最终 Gate 选中项。该阶段最终判定为 **GO**。

### 实验六：性能优化

BGE 首次用户查询原来要等编码器加载。把编码器预热、单次初始化（single-flight）和 Search Ready 检查放到服务启动阶段后，BFF 的首个用户查询从 **16.28 s** 降到 **0.552 s**，热查询仍约 **0.288 s**。

ESCI 的 BGE 批量编码原本在本地 CPU 上运行过慢；后来迁移到 AutoDL RTX 4090：164,900 个唯一商品编码 **80.46 s**，8,956 个查询编码 **1.67 s**，总计约 **82.1 s**。

### 补充：早期探索（P5–P12）

早期阶段主要完成了推荐底座、经典模型对照、Qwen 规划可行性和 BGE 复用验证。它们留下了几条重要教训：离线推荐指标容易受规则标签和候选池影响；Qwen 的合同测试、本地 Rule 和真实 GPU 推理必须分开；冷启动成本会显著影响 Web 体验。由于这些阶段大多是探索性 Dev 或小样本，本文只把它们作为后续设计依据，不把早期数字与 P14–P18 的最终指标混算。

## 4. 主要困难与解决方法

| 问题 | 简单原因 | 解决方法 | 最终结果 |
|---|---|---|---|
| BM25 召回不足 | 关键词难覆盖同义表达 | 接入 BGE 语义检索 | 两个检索评测中 BGE 均更高 |
| C4 数据不够真实 | Query 由评论构造 | 保留为 proxy；另做 ESCI | 真实搜索证据与 proxy 分开 |
| 合成任务高估 V0 | 标签规则与 V0 评分相近 | 明确标签 provisional，另做 ESCI/真实历史 | 真实收益仍未证明 |
| 真实历史无稳定增益 | 历史与目标关系弱，shuffled 也接近 | 保留 Memory 功能，不夸大质量 | History 未超过 shuffled |
| SFT 没超过 Base | 134 条轨迹主要模仿 Rule | 保留 Qwen Base，不部署 SFT | Base 15/15，SFT 12/15 |
| LLM 可能编造属性 | 商品字段不完整 | Evidence Bundle、UNKNOWN、最终检查 | 未知字段不被冒充为事实 |
| Qwen Live 服务连不上 | GPU/模型服务或 tunnel 未就绪 | 启动既有 4090 模型服务与 tunnel | 真实推理、MCP、浏览器通过 |
| Qwen 不主动 ASK | 模型倾向先 RETRIEVE | 增加通用 Planner 合同提示 | 真实 ASK 可触发 |
| 困难任务重复调用工具 | Planner 未及时收敛 | 有界预算终止 | 可结束，但仍有无效循环 |
| 当前偏好与长期 Memory 冲突 | 一次任务限制不等于永久删除 | 区分当前任务约束与长期历史 | 旧历史保留，当前要求优先 |
| 澄清后候选过期 | 偏好更新了，候选分数没更新 | 重新评估候选 | 15→14 eligible，约 1.05 ms |
| Gate 放行冲突商品 | 最终检查仍用旧任务状态 | Gate 增加当前任务过滤和最终复查 | 最终选中有 wireless 证据的商品 |
| Web 初始卡片被当成最终商品 | 测试读取了第一张商品卡 | 改用 Gate-selected 标记 | Web 与 MCP 最终都选 B00M0KUF44 |
| 首次 BGE 查询慢 | 编码器在用户请求时才加载 | 预热、single-flight、Search Ready | 首次查询约 0.552 s |
| 并发初始化/取消竞态 | 多请求同时初始化 Encoder | 单次初始化、等待、失败重试 | 并发时只初始化一次；可重试 |
| 本地 CPU 编码 ESCI 过慢 | 批量推理在 CPU 上执行 | 迁移到 RTX 4090 | 约 82.1 s 完成编码 |
| Git/报告分散、复现困难 | 多阶段代码和实验未统一归档 | 分阶段 commit、Manifest、source diff | 归档、哈希与复现说明已建立 |

这些问题里，有些是模型能力限制，有些是评测口径错误，有些是状态一致性问题。它们不能都归因于“模型不够强”：C4 provenance、ESCI 候选边界、Qwen Live 服务状态和 Web 初始卡片误判，首先都是数据或测试问题。把这些问题分开，才能避免用错误指标证明能力，也避免把已修复的历史故障继续写成当前缺陷。例如，ASK 最初 LIMITED 是当时 M9 语义的真实结果，后来 GO 是加入当前任务约束后的结果，两者都应保留，不能只引用最终数字。

## 5. 关键性能优化

| 项目 | 优化前 | 优化后 | 说明 |
|---|---:|---:|---|
| BFF Ready | 49.53 s | 50.14 s | 启动阶段多做了编码器预热 |
| **首次用户查询** | **16.28 s** | **0.552 s** | 约减少 96.6% |
| Ready → 首个结果 | 65.81 s | 50.69 s | 平均冷启动体验改善 |
| 热查询 | 0.288 s | 0.288 s | 基本不变 |
| 两个 MCP Profile 常驻内存 | 约 2.25 GB | 约 3.10 GB | 增加约 0.85 GB |
| ESCI GPU 编码 | 本地 CPU 很慢 | 4090 上约 82.1 s | 商品 164,900；查询 8,956 |

**必须说明：** 0.552 s 的主要原因是把模型预热从“用户请求阶段”移到了“服务准备阶段”，不是 BGE 模型计算本身快了 96.6%。用户仍然要等完整启动；这部分成本已计入 BFF Ready。

## 6. 已修复与仍未解决的问题

**已修复：**

- Qwen 真实 ASK 曾在一次回复后保留冲突偏好；现在当前任务约束优先，并会重新评估候选和 Gate。
- Web 测试曾把初始商品卡误认为最终选择；现在读取 Gate-selected 商品，Web/MCP 结果一致。
- BGE 首次查询的编码器加载、并发初始化、失败恢复和 Search Ready 已接入。
- ESCI CPU 编码已迁移到 GPU；GPU 实际参与并记录了耗时、显存和吞吐。

**仍未解决：**

- 真实用户长期个性化的稳定收益未证明；历史消融没有超过 shuffled。
- C4 仍是评论衍生 proxy，不是真实搜索日志。
- ESCI 是 judged-candidate reranking，不是全库检索 Recall。
- 自动生成的推荐解释缺独立人工语义审核。
- 商品价格、品牌、库存等字段部分缺失，必须保持 UNKNOWN。
- Planner 仍可能在困难任务上重复调用工具。
- Web 冷启动仍需约 50 秒，且没有认证、TLS 和生产部署能力。
- 真实 Qwen Live 9/9 是 BGE 优化前版本的结果；当前最终回归是 CPU/Replay 7/7，完整 Qwen Live 未在最新提交上重跑。

最后需要强调：本文所有 PASS 只表示预先约定的工程验收条件满足，不自动等于真实用户满意度，也不等于生产环境质量。

## 7. 最终结论

AgentRec-X 完成了可运行的 Shopping Agent 原型：真实商品目录、BM25/BGE 检索、Qwen Base 工具规划、MCP、Memory、澄清、证据检查、Gate 和 Web 多轮交互都已走通。最重要的工程成果，是发现并修复“记忆已更新、候选和 Gate 却仍用旧状态”的多轮决策问题，并把 BGE 初始化开销从首次用户请求中移走。

可信的结果包括：ESCI 上 BGE 的候选重排 nDCG@10 高于 BM25；Qwen Base 在小规模 Dev20 上优于 Rule 和当前 SFT；Agent 工具链和多轮闭环有真实运行证据。尚未证明的是：真实用户个性化收益、全库检索质量、解释语义正确性和生产级稳定性。项目适合作为 Agent 开发岗位的工程原型展示，不应描述为生产就绪或达到行业领先水平。

## 附录：实验来源索引

| 实验/主题 | 主要报告或证据 |
|---|---|
| 项目总览、目录与当前状态 | [README.md](README.md) |
| 早期探索（P5–P12） | [P5_PERSONALIZATION_AUDIT_REPORT.md](P5_PERSONALIZATION_AUDIT_REPORT.md)、[P6_CLASSICAL_EVAL_REPORT.md](P6_CLASSICAL_EVAL_REPORT.md)、[P7_B_QWEN_DEV_COMPARISON_REPORT.md](P7_B_QWEN_DEV_COMPARISON_REPORT.md)、[P8_B_REPORT.md](P8_B_REPORT.md) |
| C4 Dev 135、BGE 检索与历史消融 | [P14D2_REPORT.md](P14D2_REPORT.md) |
| C4 Proxy Test 129（内部冻结评测） | [REAL_RETRIEVAL_EVAL_REPORT.md](REAL_RETRIEVAL_EVAL_REPORT.md) |
| ESCI 真实查询外部评测、semantic/official-script gain | [ESCI_EXTERNAL_EVAL_REPORT.md](ESCI_EXTERNAL_EVAL_REPORT.md) |
| ESCI 数据、版本、GPU 编码与复现 | [ESCI_EXTERNAL_DATA_MANIFEST.md](ESCI_EXTERNAL_DATA_MANIFEST.md)、[experiments/esci_external/README.md](experiments/esci_external/README.md) |
| V0 / BM25 / Qwen Reranker 73 任务对照 | [P14B3_EVAL100_REPORT.md](P14B3_EVAL100_REPORT.md) |
| Rule / Qwen Base / Qwen SFT | [P14C1_REPORT.md](P14C1_REPORT.md) |
| Evidence RAG、UNKNOWN 与解释 | [P15_REPORT.md](P15_REPORT.md) |
| Shopping Skills | [P16_REPORT.md](P16_REPORT.md) |
| MCP 工具与协议 | [P17_REPORT.md](P17_REPORT.md) |
| P18 本地验收与模拟标签限制 | [P18_FINAL_REPORT.md](P18_FINAL_REPORT.md) |
| Web Demo 与 CPU/Replay 回归 | [WEB_DEMO_REPORT.md](WEB_DEMO_REPORT.md)、[FINAL_POLISH_REPORT.md](FINAL_POLISH_REPORT.md) |
| 真实 Qwen GPU / 浏览器验收 | [QWEN_LIVE_WEB_REPORT.md](QWEN_LIVE_WEB_REPORT.md) |
| 真实 ASK→M9→Resume（修复前） | [QWEN_ASK_RESUME_REPORT.md](QWEN_ASK_RESUME_REPORT.md) |
| 当前任务约束、候选重评与 Gate 修复（修复后 GO） | [QWEN_CLARIFICATION_FIX_REPORT.md](QWEN_CLARIFICATION_FIX_REPORT.md) |
| BGE 预热、single-flight 与首查性能 | [FINAL_POLISH_REPORT.md](FINAL_POLISH_REPORT.md) |
| Git 归档与来源 | [GIT_ARCHIVE_REPORT.md](GIT_ARCHIVE_REPORT.md) |

> `runs/` 下还有本地运行日志、截图和 JSON 证据；它们是 git-ignored 的实验资产，本总结引用的是对应的已提交报告与归档 Manifest。失败样本、GPU 日志和原始数据未全部入 Git，但报告内给出了产物路径或重生成方式。
