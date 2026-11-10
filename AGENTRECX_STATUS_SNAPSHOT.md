# AGENTRECX_STATUS_SNAPSHOT — 当前状态说明

**快照时间：** 2026-11-10 16:36（Asia/Shanghai）  
**性质：** 只读状态记录；不修改代码、不运行实验、不 Commit、不 Push、不 Deploy。

## 1. 结论

AgentRec-X 当前处于 **可交接、可归档、未运行** 的状态。

- 核心工作已本地提交；
- 没有 tracked 源码改动；
- 没有运行中的 BFF、前端或 Qwen tunnel；
- 没有运行中的本地实验任务；
- 剩余只是文档未提交。

## 2. Git 快照

```text
分支：archive/agentrecx-web-demo-v1
HEAD：4c44f4c65351c062cc2247dd81431de779edac20
```

`git status --porcelain`：

```text
?? AGENTRECX_EXPERIMENT_SUMMARY.md
?? AGENTRECX_EXPERIMENT_SUMMARY_REVIEW.md
?? REAL_RETRIEVAL_DATA_MANIFEST.md
?? REAL_RETRIEVAL_EVAL_REPORT.md
?? AGENTRECX_HANDOFF.md
?? AGENTRECX_STATUS_SNAPSHOT.md
```

说明：

- 前两份是本次全项目实验总结及审查；
- 中间两份是上一轮 C4 Proxy 审计报告；
- 最后两份是本次交接材料；
- 没有 `M`/`D`/`R` 等 tracked 代码改动；
- 没有执行 `git add` 或 `git commit`。

## 3. 文件规模

| 文件 | 约大小 |
|---|---:|
| `AGENTRECX_EXPERIMENT_SUMMARY.md` | 16 KB |
| `AGENTRECX_EXPERIMENT_SUMMARY_REVIEW.md` | 7 KB |
| `REAL_RETRIEVAL_DATA_MANIFEST.md` | 6.5 KB |
| `REAL_RETRIEVAL_EVAL_REPORT.md` | 24 KB |
| `AGENTRECX_HANDOFF.md` | 交接说明 |
| `AGENTRECX_STATUS_SNAPSHOT.md` | 本文件 |

`runs/`、`data/`、模型、Embedding 和 GPU 日志仍在 git-ignored 目录，没有进入 Git。

## 4. 服务与任务

| 项目 | 状态 |
|---|---|
| BFF `127.0.0.1:8787` | **closed** |
| 前端 `127.0.0.1:3000` | **closed** |
| Qwen tunnel `127.0.0.1:18009` | **closed** |
| 本地后台实验任务 | **无运行中任务** |
| Team 子代理 | 全部 inactive |
| AutoDL 实例 | 之前使用过 RTX 4090；模型服务/tunnel 已停止；**实例计费状态需在控制台确认** |

## 5. 最近验证过的状态

| 检查 | 状态 |
|---|---|
| ESCI 源码/Manifest/结果归档 commit | `4c44f4c` |
| ESCI 干净 Git archive 复现检查 | PASS |
| ESCI per-query / failure 归档 | 8,956 / 1,529 |
| BGE 首查优化后 CPU/Replay E2E | 7/7 |
| Qwen Live 历史真实 GPU E2E | 2/2；历史全量 9/9（BGE 优化前版本） |
| Qwen ASK 修复后 | GO；最终 Gate B00M0KUF44 |
| C4 Proxy Test 129 | 已完成冻结 proxy 评测 |
| ESCI external evaluation | 已完成（judged-candidate reranking） |

## 6. 尚未完成/需要用户决定

1. **是否 Commit 剩余文档**：当前按用户要求“不动 Git，只生成 Handoff 和状态说明”，因此未提交。
2. **是否继续新的工作**：当前没有进行中的研发任务；下一步由新会话/用户决定。
3. **AutoDL 计费**：请确认实例是否仍在租用；如果不再使用，由用户决定是否在控制台停止。
4. **真实 Qwen 现场 Demo**：如继续演示，需要重新启动模型服务与 tunnel；当前服务已停。
5. **真实个性化 / 全库检索 / 解释语义审核**：仍是已知未证明项，不应在当前文档中夸大。

## 7. 交接后的安全建议

- 先读 `AGENTRECX_HANDOFF.md`；
- 先核对 `git status` 与 HEAD，不要先跑实验；
- 未经用户明确同意，不运行 Reserved Test / 全量 GPU 编码 / 训练；
- 不修改 BGE、V0、Qwen、M9、Gate、MCP、Web；
- 不 Commit/Push/Deploy；
- 如需要归档文档，先让用户确认要提交的文件清单。

## 8. 明确声明

本次操作：

- **未修改任何旧报告**；
- **未重新运行实验**；
- **未修改 Git 历史**；
- **未 Commit、未 Push、未 Release、未 Deploy**；
- **未启动/停止/释放 AutoDL 实例**。

当前状态可以安全交接。
