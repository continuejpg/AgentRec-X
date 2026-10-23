# 工程实践主线（Engineering Practice）

> 目的：让**新会话/新同事**在 10 分钟内知道这个项目是怎么一步步建起来的、当时守了哪些规矩、
> 踩过哪些坑、以及哪些结论是"已验证"、哪些只是"未确立"。
>
> 三层证据，各管一件事，**不要互相替代**：
>
> | 层 | 位置 | 回答什么 | 权威性 |
> | --- | --- | --- | --- |
> | 变更 | `git log`（98 commits） | **什么时候、改了什么** | 不可变，最硬 |
> | 理由 | `docs/`（82 份） | **为什么这么做、怎么验收的** | 当时冻结，可追溯 |
> | 指令 | `docs/SESSION_MILESTONES.md`（120 条，脚本可再生） | **人当时怎么要求的**（约束、边界、止损） | 原始意图 |
>
> 本文件是**综合层**：把三层压成一条可读的路线 + 可复用的实践规则。

---

## 1. 阶段路线（按时间）

| 阶段 | 时间 | 主题 | 关键锚点 |
| --- | --- | --- | --- |
| S0 起点 | 09-13 ~ 09-14 | 环境、预处理、统一评测协议、SASRec 基线、最小 API | `859fd0b` → `4de3091` |
| S1 Agent 骨架 | 09-19 ~ 09-20 | 受控控制面 → bounded AgentPolicy → Agent 评测面 | `5040dda` · `6db2386` · `3a9a8be` |
| S2 约束与真实 LLM | 09-20 ~ 09-21 | 硬约束、可行候选视图、真实 Provider 评测与回放、Prompt v2 | `69f56ef` 等 |
| S3 模型扩展 | 09-21 ~ 09-22 | Two-Tower 召回基线、Semantic-ID 生成式检索 | 见 `docs/MODEL_EXPANSION_HANDOFF.md` |
| S4 TIGER 攻坚 | 09-23 ~ 09-27 | Gate B → Gate C → FP32 canonical remediation → Step 2.6/2.7/2.8 资格认证 | `62cb9b5` · `fa31309` · `6447d75` |
| S5 Agent 复评与 M 系列 | 09-26 ~ 10-02 | M3 预注册、Agent 评测臂、M4 封存、LLM Agent demo | `5214c8f` · `55f8071` · `dc50635` |
| S6 V2 交付 | 10-11 ~ 10-16 | 认知整合 → M4 诊断 → TIGER 实时引擎 → 四源 Demo → E-01…E-04 → Gate B/C → P3 → rc1 冻结 | `b3af9da` · `ac0d058` · `21007eb` · `c4dc0ef` · `4facd07` |
| S7 Learned Ranker | 10-19 ~ 10-20 | 只读审计 → 物化 5,000 组 → LambdaRank C0 → Serving 集成 → 预测路径 bugfix | `7f908a7` · `925d495` |
| S8 Agent 集成 | 10-20 | AgentGraph 真实接入、动态平面守卫（明确不支持） | `7821173` · `f4bb9fd` |

> 逐条指令与日期见 [`SESSION_MILESTONES.md`](SESSION_MILESTONES.md)；逐提交见
> [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md)。

---

## 2. 十条实践规则（都是被事故逼出来的）

### 1) 协议先行，预注册再执行
先冻结评测协议/切分/指标（`agentrecx.eval_protocol.v1`、`temporal_leave_two_out`、seed），再做实验。
**没有预注册就不产生"确认性结论"**，只能标探索性。
锚点：`docs/M3_PREREGISTRATION.md`、`docs/B1_PREREGISTRATION.md`、`docs/P3_EVALUATION_PROTOCOL.md`。

### 2) 证据分层，不许一个 PASS 冒充另一个
固定四层：**工程正确性** / **效果** / **系统完整性** / **交付与持久化**。
典型写法：`ENGINEERING FINAL PASS` + `QUALITY NOT ESTABLISHED` 同时成立。
反例：精排 HR@20 21/766 vs RRF 30/766，差值 CI 跨 0 ⇒ 只报工程通过，不报质量提升。

### 3) Fail-closed，且 UNKNOWN 必须显式
未知商品身份 → 422（不是 500、不是静默过滤）；缺证据 → `UNKNOWN`；未执行 → `NOT EXECUTED`。
锚点：`ac0d058`（专用异常 + 精确 422）、`docs/V2_KNOWN_LIMITATIONS.md`。

### 4) 一次只改一个变量，最小修复
先复现、再定位、再改；不为"顺手"重构。多轮里反复出现"最小 diff"要求，是刻意的。
反例：一次改动同时动 universe + 分数映射，导致定位成本翻倍。

### 5) 静默降级是头号敌人
回退必须**可观测**：`ranker_used` / `fallback_reason` / `fallback_count`，并进入响应或指标。
事故：`rank()` 引用未定义变量 → `NameError` 被 fallback 吞掉 → HTTP 仍 200，但排序一直是 RRF ✗。
修法：**反转 mock**（让模型故意给出与 RRF 相反的序，断言输出 ≠ RRF）+ 统计 predict 调用次数。
锚点：`925d495`。

### 6) 训练-推理特征必须是同一份实现
同一函数供离线物化与在线 Adapter；名次用全精度、分数统一量化；**排名全集**在过滤前确定。
事故链：池内位置当名次（1/64 vs 1/71）→ 全集成员丢分（名次整体上移 1）→ `universe` 未定义。
验收：逐列对齐（15/15 列、20 例 5,420 行、1e-6 容差），不是只看 shape。

### 7) 排序归属权唯一，且必须可声明
谁决定了最终顺序，必须能写进响应：`rrf_rank_fusion` / `lightgbm_lambdarank` / 回退原因。
不允许"模型排完再被融合一次，却宣称模型决定结果"。
锚点：`7821173`（动态平面**前置拒绝**不支持的组合）。

### 8) 授权边界就是工程边界
不擅自训练/调参、不擅自 Git 写、不擅自扩范围；每次执行前把"允许/禁止"写成清单。
反例也要记录：一次 `set -e` 被管道吞掉，导致带红测试的提交被推上去，随后用独立修复提交（不 amend）纠正。

### 9) 可复现优先于结论
每个结论都要能重跑：数据哈希、模型哈希、随机种子、依赖版本、运行命令。
`runs/` 不入 Git，模型必须**独立归档**并附 `SHA256SUMS`；源码交付 ≠ 权重交付。

### 10) 复盘与自我更正要留痕
结论被推翻时**公开撤销**（如"QUALITY IMPROVED"→"NOT ESTABLISHED"），保留旧报告并标注失效。
错误修复用**新提交**，不改写已推送历史。

---

## 3. 真实事故清单（新会话最该先读的部分）

| # | 现象 | 根因 | 修法与锚点 |
| --- | --- | --- | --- |
| 1 | 授权调用被浪费一次 | 把 SDK 的方法当属性用 | 先读契约再调用 |
| 2 | 未知商品 → HTTP 500 | 缺少专用异常与处理器 | 子类化 FusionError + 422（`ac0d058`） |
| 3 | 单源剖面在融合路径崩溃 | 多源无"单源参考"，KeyError | 用 `NOT_APPLICABLE` 显式标记 |
| 4 | 离线 RRF 与线上不一致 | 用过滤后位置代替来源真实名次 | 按 `(query,item)` 对齐；补齐已知案例（1/71） |
| 5 | `rrf_rank` 全量偏移 1 | 传给排序的分数映射只含合法池 | 传**完整** RRF 映射 |
| 6 | HTTP 200 但模型没生效 | `universe` 未定义，异常被回退吞掉 | 修变量 + 反转 mock 回归（`925d495`） |
| 7 | 带红测试的提交被推送 | `set -e` 不覆盖管道 | `pipefail` + 独立修复提交（`f4bb9fd`） |
| 8 | Memory 两轮 502 | 测试夹具二次 lifespan 复用已关闭 SQLite | 同一 lifespan 内跑完；非应用缺陷 |
| 9 | pydantic 校验失败 | `provenance` 必须是 dict-of-dicts | 按 schema 改组装 |
| 10 | 文档被 heredoc 破坏 | 反引号/大括号被 shell 替换 | 用引号 heredoc 或 Python 写文件 |

---

## 4. 维护方式

```bash
.venv/bin/python scripts/mine_session_milestones.py          # 重新生成里程碑时间线
.venv/bin/python scripts/mine_session_milestones.py --check  # 检查是否过期（可用于 CI）
```

- **每完成一个阶段**：在本文 §1 追加一行阶段，事故进 §3 表。
- **每出现一次"结论被推翻"**：进 §3，并在对应 doc 标注失效。
- 会话记录只读，脚本仅流式读取、不改写。

> 保持这份文档的价值在于**诚实**：PASS 与 UNKNOWN 必须能共存，且永远写清楚"证据在哪一层"。
