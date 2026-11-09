# P18 — Final Integration, Demonstration & Evaluation

> 本项目收官轮。目标不是制造更漂亮的指标，而是冻结核心能力、完成可复现演示和评测、区分工程成功与未建立的质量证据。
> **GPU 结论：P18 不需要 GPU。** CPU MCP Mode 使用本地 CPU BGE；Qwen Agent Mode 的 GPU 服务当前未连接，P18 只用既有真实轨迹做 **Replay**，不冒充实时调用。

## 0. 最终总结

### 已确认的工程成功

- 真实 Electronics catalogue + BGE 索引身份对齐（76,263 件）；
- P15 Evidence Bundle、P16 Skills、P17 官方 MCP v2 stdio Server 均可运行；
- 独立 MCP Client 实际发现 7 个工具并完成真实调用；
- 三个 CPU MCP 演示场景有完整工具轨迹；
- Reserved Test 在冻结后只运行一次，20/20 完成，无 contract error；
- 冻结核心哈希、运行模式、索引和模型版本已记录。

### 已确认的算法改进

- BGE 检索相对 BM25 在冻结 Dev 上的候选可达性提升明确：Recall@20 10→27，Recall@100 20→66（P14-D2）；
- 这是候选检索层的改进，不等价于最终推荐质量或真实用户准确率。

### 未成功的实验

- 真实长期历史 Reranker：metadata 恢复后仍不优于 Query Only/Shuffled（P14-D1-R）；
- Planner SFT：Base 15/15 vs SFT 12/15，不采用（P14-C1）；
- BM25+BGE Hybrid：RRF 无增量（P14-D2）；
- 语义真实性/用户满意度：未建立。

### 尚未建立的质量证据

- 自然语言解释的语义真实性没有人工审核；
- 真实用户行为/满意度没有证据；
- 公网认证、授权、TLS、限流/会话安全未实现；
- Reserved Test 是模拟规则标签，只用于有限工作流泛化检查。

### 本地可用 vs 公网生产

- **本地可用：** CPU MCP、搜索/推荐、比较、澄清、Memory 更新、证据引用；
- **尚不具备：** 公网部署、真实用户认证、无认证 Memory 暴露、生产级 SLA、真实线上质量。

## 1. Freeze / 两种运行模式

- 冻结 Manifest：`runs/p18/freeze_manifest.json`，Git HEAD `4dec6d2...`，16 个核心文件哈希，BGE 索引/模型版本记录；
- **CPU MCP Mode**：本地 stdio + BGE/BM25 + Rule Planner/Gate + P15/P16；无 Qwen/GPU；
- **Qwen Agent Mode**：现有 Qwen3.5-9B Base `/plan` + P15 `/explain`，需要 GPU 服务；P18 未启动，只使用既有轨迹 Replay。

两者不混称。`runs/p18/qwen_replay.json` 明确 `live_call=false`。

## 2. 三个真实演示

| Demo | 场景 | 结果 |
|---|---|---|
| Demo 1 | 搜索 + 推荐 + UNKNOWN | BGE 检索；推荐 `B004K6R43G`（Audio Technica Earbuds）；Gate `ALLOW_FINALIZE`；price/brand/stock/material/size/warranty UNKNOWN |
| Demo 2 | 2 商品比较 | `B004K6R43G` vs `B00008VF63`；P16 comparison_ready；商品事实/证据引用/条件性结论；不宣称客观质量 |
| Demo 3 | 澄清 + M9 更新 + 继续推荐 | user 0057：ASK → 显式测试回复 “I prefer noise cancelling and wireless.” → active prefs 更新 → 继续推荐 finalized |

轨迹：`runs/p18/demos/`。用户回复均为 `explicit_test_user_reply`，不是真实线上用户。

## 3. 性能诊断

`runs/p18/performance.json`：

- MCP Server 冷启动：约 **34.2 s / 40.4 s**（两个独立 server）；
- Catalogue load：8.69 s；BGE index/search source：5.32 s；memory fixture：14 ms；
- BGE encoder load：16.87 s；首次 query：1.75 s；热查询：**0.15 s**；
- MCP tool discovery：2.33 ms；recommendation：148 ms；comparison：4.43 ms。

冷启动主要成本是 catalogue/index/model 加载；进程内已复用已加载模型和索引，没有重建索引/更换模型。P18 未做无依据的优化百分比声明。

## 4. 最终质量评价

### 工程正确性

- P17/P18 MCP、P16/P15、P14-C1/D2、C0-R、Eval100 定向回归 **55 passed**；
- Tools/协议/identity/Memory/Gate/Skill 边界已测试；
- 冻结后 Reserved Test 只运行一次；没有根据结果调参。

### 证据语义质量

- 生成 **20** 条人工审核材料：`runs/p18/human_review_materials.jsonl`；
- **0 条已审核，20 条 pending_human_review**；
- `semantic_truth_verified=false`，没有把自动检查称为人工结果；
- 因此不能宣称自然语言解释没有语义幻觉。

### 冻结后的 Reserved Test（一次性）

`runs/p18/reserved/`：

- 20/20 任务完成；12 个 recommend，Top-1 正确 **11/12**；
- 8 个 non-recommend：4 个正确 clarification，4 个正确 unknown termination；
- false ASK / false refusal / false finalize / hard violations / errors = **0**；
- 仍是模拟规则标签任务，只用于受限工作流泛化检查，不能证明真实用户推荐准确率；
- `reserved_run_manifest.json` 记录 `one_time=true`，脚本拒绝第二次运行。

## 5. 失败与边界验证

- 无效商品 ID / 错误 namespace / 参数超限 → 明确 MCP tool error；
- 缺失证据、`no_verified_candidate_in_scope` → 不伪造推荐；
- 未回复澄清 → `awaiting_clarification`，不 FINALIZE；
- 写入关闭 / 未确认 / 过期 / 跨用户 handle → 明确拒绝；
- MCP 未知 tool / 错误参数 → `is_error`，不静默成功；
- 商品文本不进入 Qwen 生成路径；P15/P18 只保留真实 catalogue 证据与 UNKNOWN。

## 6. 最终回答

1. **MCP Server 真正运行？** 是，官方 SDK 2.3.0，stdio，protocol 2026-07-28。
2. **独立客户端发现和调用 Tools？** 是，7/7 tools，26 calls（P17）。
3. **直接复用原 Core？** 是，无第二套 Runtime。
4. **商品与 Memory 可追溯？** 是，raw field/span/source，memory provenance，active/inactive 分离。
5. **两个 Skills 可通过 MCP 调用？** 是，P18 Demo2/Demo3 实测。
6. **身份与写入权限验证？** 本地绑定和写入确认验证通过；无生产认证，安全能力 **LIMITED**。
7. **哪些仅适合本地？** 全部本地 CPU MCP；远程 HTTP/认证/Memory 暴露均未实现。
8. **是否具备演示/部署条件？** 本地演示 GO；公网部署尚不具备安全条件。

## 7. 判定

| 判定 | 结果 |
|---|---|
| `MCP_PROTOCOL` | **PASS** |
| `TOOL_DISCOVERY` | **PASS** |
| `SEARCH_RECOMMEND_TOOLS` | **PASS** |
| `SKILLS_EXPOSURE` | **PASS** |
| `MEMORY_ACCESS_CONTROL` | **LIMITED** |
| `END_TO_END_CLIENT` | **PASS** |
| `LEGACY_REGRESSION` | **PASS** |
| **P18** | **GO（本地可复现）** |

P18 GO 只表示本地集成、演示和受限评测可复现；不表示真实用户质量或公网生产就绪。

## 8. 已知限制

- 无人工语义审核；解释质量未建立；
- 真实历史个性化增益未证实；
- Qwen Agent Mode 当前仅 Replay，未重新连接 GPU；
- 无公网认证/授权/HTTPS/限流；
- Reserved Test 为模拟规则标签，不是真实用户准确率；
- BGE CPU 冷启动/首次编码较慢；仅本地演示可用；
- 无网页前端。

## 9. 交付物

- `P18_FINAL_REPORT.md`、`P18_SOURCE_DIFF.txt`
- `PROJECT_ARCHITECTURE.md`、`DEMO_GUIDE.md`、`EXPERIMENT_SUMMARY.md`
- `runs/p18/freeze_manifest.json`
- `runs/p18/demos/`
- `runs/p18/performance.json`、`performance_components.json`
- `runs/p18/human_review_materials.jsonl`、`human_review_status.json`
- `runs/p18/qwen_replay.json`
- `runs/p18/reserved/`（one-time）
- `tests/test_p18_final.py`
- README P18 section

未训练模型、未重建 BGE、未新增 Skill/Reranker、未运行第二次 Reserved Test、未 commit/push、未操作 AutoDL 实例。
