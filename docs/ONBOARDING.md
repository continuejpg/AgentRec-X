# 新会话入口（5 分钟）

> 给接手这个仓库的新会话/新同事：**先读这一页，再动手**。它只回答三件事：
> 现在在哪、按什么顺序读、哪些事绝对不要假设。

## 1. 现在的坐标

| 项 | 值 |
| --- | --- |
| 工作分支 | `feat/learned-ranker-v1`（HEAD `f4bb9fd`，已推送） |
| 默认服务剖面 | `serving_v1`（三源 RRF）—— **不要改** |
| 实验剖面 | `serving_v1_learned_ranker`（LightGBM 精排，可回退 RRF） |
| 版本标签 | `v2.0.0-rc1` **未创建**（证据只在同盘 `backups/`，非独立存储） |
| 精排模型 | `runs/lr_v1_formal/model.txt`（归档 SHA256 见部署文档），**不在 Git 里** |

## 2. 阅读顺序

1. [`AGENTS.md`](../AGENTS.md) —— 工作规则与边界（最先读，优先级最高）
2. [`README.md`](../README.md) —— 产品形态、能做什么、不能做什么
3. [`docs/ENGINEERING_PRACTICE.md`](ENGINEERING_PRACTICE.md) —— **项目怎么一步步建起来的 + 十条实践 + 真实事故清单**
4. [`docs/PROJECT_HISTORY.md`](PROJECT_HISTORY.md) —— 里程碑 ↔ 提交 对照
5. [`docs/SESSION_MILESTONES.md`](SESSION_MILESTONES.md) —— 120 条原始指令（当时的约束与止损）
6. [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) —— 当前结构
7. [`docs/V2_FINAL_ACCEPTANCE.md`](V2_FINAL_ACCEPTANCE.md) + [`docs/V2_KNOWN_LIMITATIONS.md`](V2_KNOWN_LIMITATIONS.md) —— 哪些 PASS、哪些还没

## 3. 开场 prompt（复制即用）

见 [`NEW_SESSION_PROMPT.md`](NEW_SESSION_PROMPT.md)：A 只读接手 / B 带任务执行 / C 只读审计 / D 极简。

## 4. 三条铁律

1. **不擅自训练/调参/重物化**；不擅自 Git 写；不扩大范围 —— 每轮先把"允许/禁止"写成清单。
2. **一个 PASS 不能冒充另一个**：工程正确性、效果、完整性、交付四层分开报；效果未证实就写 `NOT ESTABLISHED`。
3. **静默降级是头号敌人**：回退必须带 `fallback_reason`/计数；模型是否真的参与，要有可断言的证据（反转 mock + predict 计数）。

## 5. 十分钟自检

```bash
git log --oneline -5
.venv/bin/python -m pytest -q tests/test_v2_learned_ranker.py          # 13 passed
.venv/bin/python scripts/mine_session_milestones.py --check           # 里程碑是否过期
sha256sum runs/lr_v1_formal/model.txt && cat docs/LEARNED_RANKER_DEPLOYMENT.md | head -20
```

## 6. 已知缺口（不要当成已完成）

- **效果未确立**：融合 HR@20 ≈ 3.1%、精排 2.7%；独立 test 的配对 CI 跨 0。评测样本只有 111 个含正例请求，统计功效不足。
- **TIGER**：PARTIAL，未准入融合（两次运行 0/100 与 1/100，训练协议无法确认）。
- **LLM Agent 在线闭环**：PARTIAL（真实 Provider 额度已用尽，仅一次性验证）。
- **动态 CandidatePlane + 精排**：**不支持**，已在前置位置拒绝；不要把它接回去。
- **分阶段性能计时**、独立存储备份：未完成（见 `V2_KNOWN_LIMITATIONS.md`）。
