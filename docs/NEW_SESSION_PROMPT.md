# 新会话开场 Prompt（复制即用）

> 用法：新开一个会话，粘贴下面任一段。它会先建立**坐标**和**边界**，再谈任务，
> 避免新会话凭想象重做审计、重跑训练或改动默认服务。

## A. 只做"接手了解"（5–10 分钟，不改任何文件）

```text
你是 AgentRec-X 项目的新接手会话。先不要改任何文件，也不要运行训练或大规模评测。

1) 按顺序读：AGENTS.md 第 20 节 → docs/ONBOARDING.md → docs/ENGINEERING_PRACTICE.md。
2) 再读 docs/PROJECT_HISTORY.md 的阶段表与 docs/SESSION_MILESTONES.md 的最近 10 条。
3) 运行这三个只读检查并原样贴出结果：
   git branch --show-current && git log --oneline -5
   .venv/bin/python -m pytest -q tests/test_v2_learned_ranker.py
   .venv/bin/python scripts/mine_session_milestones.py --check

输出一份不超过 20 行的交接摘要：
- 当前坐标（分支/HEAD/默认 profile/实验 profile/未创建的 tag）
- 已验收 PASS 的东西（分别注明是哪一层：工程 / 效果 / 完整性 / 交付）
- 明确标为 NOT ESTABLISHED / UNSUPPORTED / PARTIAL 的东西
- 你认为最值得先做的 3 件事，各自的最小验证方式

不要提出新功能，不要重构，不要提交 Git。
```

## B. 带任务执行（最小改动、强约束）

```text
你是 AgentRec-X 的执行会话。任务见下，但先做三件事再动手：

1) 读 AGENTS.md（尤其第 19–20 节）、docs/ONBOARDING.md、docs/ENGINEERING_PRACTICE.md 的
   "十条实践"与"事故清单"。
2) 用不超过 5 行复述：这次任务允许改什么、禁止改什么、验收证据长什么样。
3) 先只读定位：给出真实文件路径与行号；不确定就写 UNKNOWN，不要猜。

任务：<在这里写你的任务>

硬约束：
- 不重训/不调参/不重物化/不切换默认 profile/不改 Agent 决策与 Memory/TIGER；
- 每轮只改一个变量；先复现再修；
- 任何"回退/降级"必须可观测（fallback_reason + 计数），HTTP 200 不能当作成功证据；
- 结论必须分层写（工程 / 效果 / 完整性 / 交付），效果未证实就写 NOT ESTABLISHED；
- 不执行 git commit / push，除非我明确授权。
```

## C. 只读审计（回答"某能力到底实现了没"）

```text
只读审计，不改任何文件。目标：判断 <能力 X> 在源码中真实实现到哪一层。

方法：
1) 找入口 → 找调用链 → 找测试证据，三层都要给 file:line。
2) 区分：有函数定义 / 有真实调用 / 有行为测试 / 在默认入口启用。
3) 每条结论必须落到等级之一：L0 仅规划 / L1 有代码 / L2 有调用 / L3 有测试 / L4 有受控行为证据。
4) 没有证据的写 UNKNOWN 或 EVIDENCE_MISSING，不要用设计文档代替实现。

输出：能力矩阵（能力 | 入口 | 调用链 | 测试 | 等级 | 能否写进简历）+ 三条真实调用链 + 缺口清单。
不要跑训练、不要跑大规模评测、不要改代码。
```

## D. 一句话极简版（你已熟悉项目时）

```text
接手 AgentRec-X：先读 AGENTS.md §20 + docs/ONBOARDING.md，跑 pytest tests/test_v2_learned_ranker.py
与 scripts/mine_session_milestones.py --check，给我 10 行现状与风险摘要，不改文件、不提交。
```
