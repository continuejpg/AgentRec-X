# P2B 交接（Phase 2B / 2B.1 / 2B.2 / 2B.3 已完成）

> 一页纸：做到哪、结论是什么、下一步做什么。**不要把它当架构文档**，架构见
> [`ARCHITECTURE.md`](ARCHITECTURE.md)，入口见 [`ONBOARDING.md`](ONBOARDING.md)。

## 1. 已有实现

| 文件 | 作用 |
|---|---|
| `recommendation/demo/source_limit.py` | `resolve_source_limit(requested_k)`：读 `AGENTRECX_AGENT_SOURCE_LIMIT`；未配置 ⇒ 等于 `requested_k`（历史行为）；非法值报错不 clamp |
| `recommendation/demo/agent_service.py:544-570` | `_build_controller` 把该预算传给三种 policy；**最终截断仍是 `limit=payload.k`** |
| `tests/test_agent_source_limit.py` | 6 项：默认兼容 / 空白 / 独立生效 / 非法拒绝 / 三 action 用同一预算 / proposal 携带预算 |
| `tests/test_p2b_metrics.py` | 指标单测（hr/ndcg/pool/配对 CI），从驱动模块直接导入被测函数 |
| `tests/test_p2b_proxy_query.py` | 2B.3 纯函数测试 14 项：代理查询构造/去重/缺标题回退、`history[-2:]` 种子选取与目标排除、四分支分类 |
| `experiments/p2b_agent_source_limit_ablation.py` | 真实 Agent 控制面四臂配对驱动（`DemoAgentService.run` → `LoopController` → `CandidatePlane`） |

## 2. 已有结果（可复算，勿夸大）

| 产物 | 规模 | 结论 |
|---|---|---|
| `runs/p2b_agent_source_limit/` | 12 例 × 4 臂 | 池 10→199；进池率 0→16.7%；HR/NDCG 全 0 |
| `runs/p2b_agent_source_limit_40/` | 40 例 × 4 臂 | 池 10/40/100/200；进池率 0/2.5/5/5%；HR/NDCG 全 0；p50≈1.4 s 四臂持平 |
| `runs/p2b_agent_source_limit_40b/` | 40 例 × 4 臂（代理查询 + 埋点） | 池 9.9/39.0/95.1/187.1；进池率 2.5/2.5/5/5%；HR@5 恒 1/40；`filtered=0`；38/40 在 depth=100 外。详见 [`P2B_AGENT_SOURCE_LIMIT_ABLATION.md`](P2B_AGENT_SOURCE_LIMIT_ABLATION.md) |

**已知混杂（已在 2B.3 消除）**：2B.2 的驱动给全部案例用了同一条消息
`"recommend something for me"`。而 `catalog_search` 的查询只从消息抽词
（`demo/agent_policy.py:339-349`），因此查询恒定、返回同一组商品，两个商品在 **40/40** 个
Top-5 里固定出现。**“5% 进池”是实验输入造成的，不是召回结论。** 2B.3 改为 `history[-2:]`
标题的案例级代理查询（40/40 不同），最高单商品 Top-5 频次降到 4/40，并补了源级埋点与四分支归因。

## 3. 2B.3 任务单（已执行，结果见 [`P2B_AGENT_SOURCE_LIMIT_ABLATION.md`](P2B_AGENT_SOURCE_LIMIT_ABLATION.md)）

- **A** 案例级代理查询：取 `history[-2:]` 的**商品标题**，确定性拼接；**禁用** `items[-1]`（目标）与
  `items[-2]`（验证）；在 `cases.json` 标记 `query_source: "history-title-proxy"`。
- **B** 最小埋点（复用 ledger provenance / 响应视图，不新建账本）：
  每源 `requested_limit / effective_limit / returned / rank_in_source / ms`；失败/截断
  （实现名 `failed` / `exhausted_short_of_limit` / `internal.history_truncated`）；
  去重前后数量与目标是否被过滤；`fused_rank`；最终排名。`history` 源标 `pre_fused=true`。
- **C** 同 40 案例、四臂（5/20/50/100）、`requested_k=5`，写入 `runs/p2b_agent_source_limit_40b/`。
- **D** 四分支归因：从未出现 / 被过滤 / 进池但靠后 / 进 Top-5。
- **E** `docs/P2B_AGENT_SOURCE_LIMIT_ABLATION.md` + 产物 SHA-256 清单。

## 4. 复现

```bash
.venv/bin/python -m experiments.p2b_agent_source_limit_ablation --cases 40 --out runs/p2b_agent_source_limit_40b
.venv/bin/python -m pytest -q tests/test_agent_source_limit.py tests/test_p2b_metrics.py tests/test_p2b_proxy_query.py
```

## 5. 约束（不得违反）

不改生产查询构造、RRF 权重、排序策略、`requested_k` 默认值；不动冻结评测器 / 预注册 / 历史报告 /
检查点；不跑全量测试、不训练、不起服务、不调用外部 LLM；不自动提交 Git。

## 6. 当前判定

工程闭环成立（解耦、默认不变、可配对执行、埋点可复核，相关测试 37 passed）；**推荐质量
NOT ESTABLISHED**。2B.3 的直接读数：池放大 19 倍而 Top-5 命中不变（1/40），`filtered=0`，
38/40 目标在 depth=100 内两个源都没返回——**瓶颈不在 Agent 源预算这一层**；但案例级代理查询
与目标标题中位 Jaccard 仅 0.029，因此**上游召回位置仍未验证**。
