# AGENTRECX_HANDOFF — AgentRec-X 交接文档

**生成时间：** 2026-11-10 16:36（Asia/Shanghai）  
**用途：** 将当前工作从本会话交接给新的工作会话。本文本身不修改项目代码、不运行实验、不 Commit。

## 1. 一句话状态

AgentRec-X 已完成核心 Shopping Agent 原型、真实 Qwen Base Live 验收、BGE 首查优化和 ESCI 外部评测；当前没有运行中的服务或后台实验，**只剩若干未提交的文档**，项目处于“可交接、可归档”的状态。

## 2. 工作目录与 Git

```text
工作目录：D:\IT\CODE\PROJECTION\agentrec\X
分支：  archive/agentrecx-web-demo-v1
HEAD：  4c44f4c65351c062cc2247dd81431de779edac20
```

最近相关提交：

| Commit | 内容 |
|---|---|
| `4c44f4c` | ESCI 外部评测源码、Manifest、per-query 与失败样本归档 |
| `8a0aead` | Final polish consolidated source diff |
| `0b1312f` | Final polish report 与 BGE 启动文档 |
| `a40ec88` | BGE single-flight / warmup / Search Ready |
| `b7303e8` | Gate-selected E2E 断言与候选诊断修正 |
| `93360de` | Qwen 验收证据快照 |
| `3c89e7e` | Qwen Base Live / ASK→M9→Resume / 澄清一致性 |

**当前没有 Push、Release、Deploy。**

## 3. 当前 Git 状态

未跟踪文件：

```text
?? AGENTRECX_EXPERIMENT_SUMMARY.md
?? AGENTRECX_EXPERIMENT_SUMMARY_REVIEW.md
?? REAL_RETRIEVAL_DATA_MANIFEST.md
?? REAL_RETRIEVAL_EVAL_REPORT.md
?? AGENTRECX_HANDOFF.md
?? AGENTRECX_STATUS_SNAPSHOT.md
```

说明：

- 前两份是刚完成的全项目实验总结与审查说明；
- 中间两份是上一轮 C4 Proxy 审计的报告；
- 最后两份是本次交接材料；
- **没有 tracked 源码改动**；`runs/`、`data/`、模型、Embedding、GPU 日志仍在 git-ignored 目录。

**没有执行 Commit。** 交接后由新会话或用户明确决定是否归档这些文档。

## 4. 当前运行状态

- 本地 BFF：`127.0.0.1:8787` **closed**
- 本地前端：`127.0.0.1:3000` **closed**
- Qwen tunnel：`127.0.0.1:18009` **closed**
- 本地后台任务：没有仍在运行的实验任务
- Team 子代理：全部 inactive
- AutoDL：之前使用过的 RTX 4090 已完成 ESCI 编码；模型服务和 tunnel 已停止。**实例是否仍在租用/计费需在 AutoDL 控制台确认**；不要默认它已释放。

## 5. 项目已完成的能力

- 76,263 件真实 Electronics 商品目录；
- BM25 / BGE 检索；BGE 768 维索引；
- V0 Preference Scorer、Eligibility、CandidateLedger、Decision Gate；
- Qwen3.5-9B Base 真实 `/plan`、MCP 工具链、Shopping Skills；
- M5/M9 Memory、ASK→M9→Resume、真实多轮澄清；
- 当前任务约束与澄清后候选重评；
- Evidence RAG / UNKNOWN / Claim Validation；
- Web Demo、CPU/Replay E2E、历史 Qwen Live E2E；
- BGE 预热、single-flight 和 Search Ready；
- C4 Proxy 诊断与 ESCI 真实查询外部评测。

## 6. 关键实验结论（交接必须保留的口径）

| 主题 | 结论 | 边界 |
|---|---|---|
| C4 Proxy Test 129 | BGE Hit@100 50.39%，BM25 18.60% | C4 Query 是评论衍生 proxy，不是真实搜索 |
| C4 Dev 135 | BGE Hit@100 66/135，BM25 20/135 | 旧 Dev；和 129 条 test 不同 |
| ESCI semantic gain | BGE nDCG@10 0.7366，BM25 0.6907 | 8,956 真实 query、181,701 judged candidates |
| ESCI official-script gain | BGE 0.7144，BM25 0.6691 | S/C 映射与主结果不同，单独报告 |
| ESCI 任务类型 | judged-candidate reranking | 不是全商品库 Recall |
| V0 / BM25 / Qwen Reranker | 67/73、46/73、58/73 | 73 条 reachable；标签 provisional |
| Real vs Shuffled History | 15 vs 16（Query 13） | C4 proxy Dev 135；真实历史没有独特增益 |
| Rule / Qwen Base / SFT | 12/15、15/15、12/15 | Dev20 小样本；最终保留 Base |
| MCP / Skills | 7 默认工具、8 with Qwen；2 Skills | P17/P16 |
| Qwen Live | 真实 GPU 浏览器 2/2、全量 9/9 | 9/9 是 BGE 优化前历史版本 |
| 当前 Web 回归 | CPU/Replay 7/7 | BGE 优化后的最终回归 |
| ASK/Resume | 修复后 GO；最终 Gate B00M0KUF44 | 初始商品卡 B00NFL0E2A 不是最终推荐 |
| BGE 首查 | 16.28 s → 0.552 s | 主要是初始化提前，不是模型推理快 96.6% |
| ESCI GPU 编码 | 164,900 商品 80.46 s + 8,956 查询 1.67 s ≈ 82.1 s | RTX 4090；只用于外部评测 |

## 7. 新会话首先读什么

建议按顺序：

1. `AGENTRECX_HANDOFF.md`（本文）
2. `AGENTRECX_STATUS_SNAPSHOT.md`
3. `AGENTRECX_EXPERIMENT_SUMMARY.md`
4. `FINAL_POLISH_REPORT.md`
5. `ESCI_EXTERNAL_EVAL_REPORT.md` 与 `ESCI_EXTERNAL_DATA_MANIFEST.md`
6. `QWEN_CLARIFICATION_FIX_REPORT.md`、`QWEN_ASK_RESUME_REPORT.md`、`QWEN_LIVE_WEB_REPORT.md`
7. `REAL_RETRIEVAL_EVAL_REPORT.md` 与 `REAL_RETRIEVAL_DATA_MANIFEST.md`
8. `experiments/esci_external/README.md`

## 8. 新会话不要做什么

除非用户明确要求，否则不要：

- 重新训练、微调或切换 BGE / Qwen 权重；
- 重跑 Reserved Test 或重新冻结 Test；
- 用 ESCI Test 调参；
- 把 C4 Proxy 与 ESCI 数字混算；
- 把 ESCI 写成 full-catalog retrieval / full Recall；
- 把 Qwen 合同测试或历史 9/9 说成当前最新提交的真实 GPU 结果；
- 修改 Planner、V0、Gate、M9、MCP、Web 或主 BGE Index；
- Commit / Push / Deploy（需要用户明确授权）；
- 在未确认 AutoDL 计费状态时擅自创建、停止或释放实例。

## 9. 建议的下一步

1. 先确认 AutoDL 控制台是否仍在租用 GPU；如果不再使用，由用户决定停止实例。
2. 新会话先读本文和实验总结，确认要做的下一件事。
3. 如果用户同意，把 6 份未跟踪文档整理成一个本地 docs commit；**在用户确认前不要提交**。
4. 如果要继续 Qwen 现场 Demo，需要重新启动模型服务与 tunnel；当前没有服务在运行。
5. 如果只做简历/面试整理，优先使用 `AGENTRECX_EXPERIMENT_SUMMARY.md`，不要重新跑实验。

## 10. 可直接粘贴给新会话的启动提示

```text
请接手 AgentRec-X 项目。工作目录：
D:\IT\CODE\PROJECTION\agentrec\X
分支：archive/agentrecx-web-demo-v1
HEAD：4c44f4c65351c062cc2247dd81431de779edac20

先按顺序阅读：
AGENTRECX_HANDOFF.md
AGENTRECX_STATUS_SNAPSHOT.md
AGENTRECX_EXPERIMENT_SUMMARY.md
FINAL_POLISH_REPORT.md
ESCI_EXTERNAL_EVAL_REPORT.md

请先只做状态核对，不要重跑实验、不要训练、不要操作 AutoDL 实例、
不要 commit/push/deploy。确认当前 Git 状态、未跟踪文档、服务是否运行后，
再问用户下一步要做什么。
```
