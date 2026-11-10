# AGENTRECX_EXPERIMENT_SUMMARY.md — 文档审查说明

**审查日期：** 2026-11-10  
**交付文件：** `AGENTRECX_EXPERIMENT_SUMMARY.md`  
**审查性质：** 只做资料整理、数字核对和 Markdown 检查；没有重跑实验、没有修改旧报告、没有 Commit/Push/Deploy。

## 1. 阅读的主要来源

实际存在并检查过的报告包括：

- `README.md`、`P5_PERSONALIZATION_AUDIT_REPORT.md`、`P6_CLASSICAL_EVAL_REPORT.md`、`P6_BENCHMARK_V0_REPORT.md`、`P7_B_QWEN_DEV_COMPARISON_REPORT.md`、`P8_B_REPORT.md`、`P9_REPORT.md`、`P10_REPORT.md`–`P18_FINAL_REPORT.md`；
- `P13R_REPORT.md`；
- `P14B0_REPORT.md`、`P14B1_REPORT.md`、`P14B2_REPORT.md`、`P14B3_EVAL100_REPORT.md`、`P14C0_REPORT.md`、`P14C1_REPORT.md`、`P14D1_REAL_HISTORY_REPORT.md`、`P14D1R_REPORT.md`、`P14D2_REPORT.md`、`P14E_REPORT.md`、`P14E2_LITE_REPORT.md`；
- `P15_REPORT.md`、`P16_REPORT.md`、`P17_REPORT.md`、`P18_FINAL_REPORT.md`；
- `WEB_DEMO_REPORT.md`、`QWEN_LIVE_WEB_REPORT.md`、`QWEN_ASK_RESUME_REPORT.md`、`QWEN_CLARIFICATION_FIX_REPORT.md`；
- `FINAL_POLISH_REPORT.md`、`REAL_RETRIEVAL_EVAL_REPORT.md`、`REAL_RETRIEVAL_DATA_MANIFEST.md`、`ESCI_EXTERNAL_EVAL_REPORT.md`、`ESCI_EXTERNAL_DATA_MANIFEST.md`；
- `GIT_ARCHIVE_REPORT.md`、`experiments/esci_external/README.md`、`experiments/esci_external/ARCHIVE_MANIFEST.json`、`experiments/esci_external/results/esci_metrics.json`。

`runs/` 下的 JSON、日志和截图只作为本地运行资产参考；正文明确说明它们是 git-ignored，没有把所有证据都写成已入 Git。

## 2. 核对的指标数量

共核对 **30 项关键数字/口径**，包括：

- 商品目录 76,263；BGE 768 维/约 213 MB；
- C4 Dev 135 与 C4 Proxy Test 129 的样本、Hit@100；
- C4 Dev 135 的 Query/Real/Shuffled History 13/15/16；
- ESCI 8,956 queries、181,701 judgments、164,900 unique products；
- ESCI semantic gain 的 0.6907/0.7366、配对差 0.0459/CI 0.0420–0.0497、nDCG_all 0.8184/0.8449；
- 官方脚本 gain 0.6691/0.7144；
- V0/BM25/Qwen Reranker 67/46/58（73 条 reachable）；
- Rule/Qwen Base/SFT 12/15、15/15、12/15；Base 3.1 s、SFT 8.5 s、134 条 SFT 样本；
- MCP 7/8 工具、2 个 Skills；
- Qwen Live 2/2 真实浏览器、9/9 历史全量 E2E；
- 当前 CPU/Replay 7/7；
- ASK 修复后的 14 eligible、B00M0KUF44、约 1.05 ms 候选重评；
- BGE 首查 16.28 s→0.552 s、Ready 49.53/50.14 s、Ready→首个结果65.81/50.69 s、热查询0.288 s、内存增加约0.85 GB；
- ESCI GPU 商品164,900、商品编码80.46 s、查询1.67 s、总计约82.1 s。

这些数字均能对应到已存在报告或归档 JSON；没有使用无来源的中间值。

## 3. 发现的口径差异与处理

### 3.1 旧 C4 Dev 135 与 C4 Proxy Test 129 是不同评测

- **C4 Dev 135**：P14-D2 的旧 Dev，用于 BGE 检索修复和历史消融；Query 由评论构造。
- **C4 Proxy Test 129**：后来冻结的独立 proxy test；与 135 的样本、阶段和用途不同。

正文分别写了两者，没有把 135 的 13/15/16 与 129 的 Hit@100 混成一个结果。

### 3.2 C4 Proxy 与 ESCI 真实查询必须分开

- C4：评论衍生的 proxy query + observed positive，不是真实搜索日志。
- ESCI：官方真实购物 query + 人工相关性 judgments，但只是 **judged-candidate reranking**，不是全库召回。

正文在核心表、实验一、限制部分都标明了这一区别。

### 3.3 ESCI 两套 gain 必须分开

- **semantic gain 口径：** E=1, S=0.1, C=0.01, I=0；nDCG@10 为 BM25 0.6907、BGE 0.7366。
- **官方脚本口径：** E=1, S=0.01, C=0.1, I=0；nDCG@10 为 BM25 0.6691、BGE 0.7144。

正文主表只把 semantic gain 当主结果，官方脚本数字作为单独的敏感性口径，没有混合两者。

### 3.4 Qwen Live 9/9 是历史版本结果

`QWEN_LIVE_WEB_REPORT.md` 的 2/2 真实浏览器、9/9 全量 E2E 对应 BGE 首查优化之前的冻结版本。最终 BGE 优化后的当前提交验证是 CPU/Replay 7/7；完整真实 Qwen Live 9/9 **没有在最新提交上重跑**。正文明确标成“历史验收”，避免把合同测试或历史结果写成当前版本结论。

### 3.5 Qwen ASK 曾 LIMITED，修复后才 GO

`QWEN_ASK_RESUME_REPORT.md` 的初始真实 ASK→M9→Resume 是 **LIMITED**：一次回复没有撤回旧 wired，Gate 最终选到了冲突商品。`QWEN_CLARIFICATION_FIX_REPORT.md` 加入当前任务约束、候选重评和最终 Gate 检查后才 GO，最终选中 B00M0KUF44。正文按时间顺序保留了两个状态。

### 3.6 Qwen Planner 延迟没有统一的 2.6–3.6 s 来源

摘要写作时没有在归档报告里找到可直接支撑“2.6–3.6 s”的统一数字；实际可核对的报告是：

- P14-C1 Dev20 Qwen Base 平均 **3.1 s**；
- P14-C0 的平均 **3.9 s**、最大 **7.3 s**（不同运行）。

因此正文采用 P14-C1 的 3.1 s，并标明是特定实验，不把 2.6–3.6 s 当作全项目统一延迟。

### 3.7 P18 的 20/20 是模拟规则标签

P18 Reserved Test 的 20/20、Top-1 11/12 属于模拟规则标签工作流检查；正文明确“不是真实用户准确率”。Qwen Live 合同测试与真实 GPU 推理也在“三个先读口径”和实验中分开。

## 4. 证据不足之处

报告正文已显式保留以下限制：

- 真实长期个性化收益未证明；真实 History 没有超过 Shuffled。
- ESCI 只评 judged-candidate reranking，不是全商品库 Recall。
- 自动解释只有自动证据检查，没有独立人工语义审核；P18 20 条人工复核材料仍为 pending。
- 部分商品属性未知，不能把 UNKNOWN 当成已知。
- Planner 仍可能在困难任务上出现无效工具循环。
- 当前 Qwen Live 最新提交未重跑完整 9/9；GPU Demo 依赖模型服务和 Tunnel。
- 没有真实用户 A/B Test，也没有生产级认证、TLS 和公网部署证据。
- SFT 结论只来自 134 个训练样本/Dev20 小样本；合成 V0 对照标签为 provisional。

## 5. 字数与链接检查

- 正文 **3,025 个汉字**（不含代码、表格中的英文/数字，按 CJK 字符统计），处于要求的 3,000–4,500 范围。
- 总字符约 9,600，主要来自表格、数字和必要的英文术语。
- 文档内 **23 个 Markdown 链接**，本地路径检查 **missing = 0**。
- 不存在的报告没有虚构；`runs/` 中未入 Git 的运行资产在正文中作为本地证据说明。

## 6. 结论

`AGENTRECX_EXPERIMENT_SUMMARY.md` 已完成并通过轻量文档审查：

- 覆盖主要成功实验和失败实验；
- 覆盖关键问题、修复和仍未解决项；
- C4 Proxy 与 ESCI 真实查询分开；
- semantic gain 与官方脚本 gain 分开；
- Qwen Live 历史验收与当前回归分开；
- 已修复问题与当前缺陷分开；
- 普通开发者可在 5–10 分钟理解项目成果、边界和失败经验。

没有修改任何旧报告，没有重跑实验，没有 Commit/Push/Deploy。
