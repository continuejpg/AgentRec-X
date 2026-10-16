# V2 原始路线覆盖映射（冻结 2026-10-16 07:15）

| 原始 V2 必备能力 | 已有证据（冻结） | 状态 |
|---|---|---|
| V1 稳定底座继承 | `POST /v1/recommend` 语义未改；`serving_v1` = sequential + two_tower + catalog_lexical 行为未变（`test_serving_fusion` + `test_api` 36 passed） | **PASS** |
| E-01A/B 控制面与多轮记忆 | `recommendation/control/*`、`recommendation/demo/*` + 离线套件（1070 项 0 失败）+ 阶段报告 | **PASS（离线）** |
| E-02A/AV/B/C 记忆排序 / 多源展示 / RAG / 叙述 | `demo/product_knowledge.py`、`demo/explanation.py`、`demo/agent_chat_adapter.py`、`web/*` + 4 份报告 + 离线套件 | **PASS（离线）** |
| E-03A/B/C 观测+门控 / 有界重规划 / 商品反馈 | `demo/observation.py`、`demo/replan_policy.py`、`demo/feedback.py` + 三个专项测试文件（63 项） | **PASS（离线）** |
| E-04A 真实 Provider 联调 | 4/4 授权调用用尽；Tool Selection、Memory、Replanning、Product RAG 均有真实调用证据 | **PASS（额度用尽，不可再调用）** |
| E-04B 真实资产与评测协议 | E-04B-1 审计、Gate A 协议+BM25、E-04B-P 双层协议冻结、P0 规则基线 | **PASS** |
| E-04 推荐后端（真实服务） | Gate C-1 SASRec PASS、Gate C-2 Two-Tower PASS（parity 100/100、repeat 100/100） | **PASS** |
| 端到端最小闭环（历史→推荐响应） | 真实 API 三路径 ×1,000 例两轮（成功率 1000/1000、两轮一致 1000/1000、违规 0） | **PASS** |
| Agent/工具工作流的**真实 LLM 在线**闭环 | E-04A 额度用尽后未再执行；B3/B4 类在线臂标记 `BLOCKED_NO_PROVIDER_AUTHORIZATION` | **PARTIAL / UNKNOWN** |
| TIGER 生成式后端 | Gate B-3 PARTIAL 结项：生成链路真实、质量 0–1/100、训练协议 UNKNOWN | **PARTIAL（不准入）** |
| 系统异常处理 | 未知身份 → 422（本轮修复）；空/短/超长历史、非法 K、非法 profile 按契约；内部 5xx 守卫缺实跑证据 | **PASS（含缺口）** |
| 资源限制与版本复现 | 哈希/协议/seed/归档齐全；RSS 6.1–6.4 GB（验收共驻形态）；**未**做并发压测 | **PARTIAL** |

**整体覆盖：PARTIAL** —— 不得据本轮 API 结果宣称“AgentRec-X 完整闭环 PASS”。
