# 新会话入口（5 分钟）

> **Status: CURRENT.** 这是当前工作区的接手入口；历史架构和冻结报告只作为参考，不覆盖当前代码。
>
> 给接手这个仓库的新会话/新同事：**先读这一页，再动手**。它只回答三件事：
> 现在在哪、按什么顺序读、哪些事绝对不要假设。

## 1. 现在的坐标

| 项 | 值 |
| --- | --- |
| 工作分支 | `feat/agent-source-limit-ablation` |
| 当前 HEAD | 以 `git log -1` 为准；最近的功能提交是 `4e82943`（adaptive preference elicitation） |
| 默认服务剖面 | `serving_v1`（SASRec + Two-Tower + catalogue lexical + RRF）—— **不要静默修改** |
| 可选剖面 | `serving_v1_learned_ranker`（LightGBM，实验性，回退 RRF）；TIGER 需显式启用 |
| 精排模型 | `runs/lr_v1_formal/model.txt`，**不在 Git 里** |
| 默认决策模式 | deterministic / offline；LLM Agent 是 opt-in 提案层 |

## 2. 阅读顺序

1. [`AGENTS.md`](../AGENTS.md) —— 工作规则与边界（最先读，优先级最高）
2. [`README.md`](../README.md) —— 项目入口、快速启动、能力边界
3. [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) —— **当前**架构、`/v1/recommend` 与购物 Agent 的区别、目录地图
4. [`docs/ENGINEERING_PRACTICE.md`](ENGINEERING_PRACTICE.md) —— 项目怎么一步步建起来、十条实践、真实事故
5. [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) —— 正式评测协议、指标和威胁
6. [`docs/PROJECT_HISTORY.md`](PROJECT_HISTORY.md) —— 里程碑 ↔ 提交对照（历史）
7. [`docs/V2_FINAL_ACCEPTANCE.md`](V2_FINAL_ACCEPTANCE.md) + [`docs/V2_KNOWN_LIMITATIONS.md`](V2_KNOWN_LIMITATIONS.md) —— 哪些 PASS、哪些还没

历史架构在 [`docs/HISTORICAL_ARCHITECTURE.md`](HISTORICAL_ARCHITECTURE.md)；旧文字图在
[`docs/ARCHITECTURE_TEXT_DIAGRAM.md`](ARCHITECTURE_TEXT_DIAGRAM.md)。两者不描述当前实现。

## 3. 开场 prompt（复制即用）

见 [`NEW_SESSION_PROMPT.md`](NEW_SESSION_PROMPT.md)：A 只读接手 / B 带任务执行 /
C 只读审计 / D 极简。

## 4. 三条铁律

1. **不擅自训练/调参/重物化**；不擅自 Git 写；不扩大范围 —— 每轮先把“允许/禁止”写成清单。
2. **一个 PASS 不能冒充另一个**：工程正确性、效果、完整性、交付四层分开报；效果未证实就写
   `NOT ESTABLISHED`。
3. **静默降级是头号敌人**：回退必须带 `fallback_reason`/计数；模型是否真的参与，要有可断言的
   证据（反转 mock + predict 计数）。

## 5. 十分钟自检

```bash
git log --oneline -5
.venv/bin/python -m pytest -q tests/test_docs.py
.venv/bin/python -m pytest -q tests/test_preference_construction.py
.venv/bin/python -m pytest -q tests/test_v2_learned_ranker.py
python -m compileall -q recommendation tests experiments
node --check recommendation/web/app.js
```

## 6. 已知缺口（不要当成已完成）

- **效果未确立**：离线融合与精排的独立 test 配对 CI 跨 0；样本/统计功效有限。
- **Learned Ranker**：实验性，依赖外部模型资产；**不能声称优于 RRF**。
- **TIGER**：PARTIAL，未准入默认融合。
- **LLM Agent 在线闭环**：PARTIAL；真实 Provider 仅一次性验证，额度有限。
- **动态 CandidatePlane + 精排**：**不支持**，已在前置位置拒绝。
- **分阶段性能计时**、独立存储备份：未完成（见 `V2_KNOWN_LIMITATIONS.md`）。
- **真实用户效果**：没有线上 A/B、点击、转化或偏好条件相关性标签；构造场景不等于线上购物成功率。
