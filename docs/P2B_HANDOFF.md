# P2B 交接（Phase 2B / 2B.1 / 2B.2 已完成，2B.3 待做）

> 一页纸：做到哪、结论是什么、下一步做什么。**不要把它当架构文档**，架构见
> [`ARCHITECTURE.md`](ARCHITECTURE.md)，入口见 [`ONBOARDING.md`](ONBOARDING.md)。

## 1. 已有实现

| 文件 | 作用 |
|---|---|
| `recommendation/demo/source_limit.py` | `resolve_source_limit(requested_k)`：读 `AGENTRECX_AGENT_SOURCE_LIMIT`；未配置 ⇒ 等于 `requested_k`（历史行为）；非法值报错不 clamp |
| `recommendation/demo/agent_service.py:544-570` | `_build_controller` 把该预算传给三种 policy；**最终截断仍是 `limit=payload.k`** |
| `tests/test_agent_source_limit.py` | 6 项：默认兼容 / 空白 / 独立生效 / 非法拒绝 / 三 action 用同一预算 / proposal 携带预算 |
| `tests/test_p2b_metrics.py` | 指标单测（hr/ndcg/pool/配对 CI），从驱动模块直接导入被测函数 |
| `experiments/p2b_agent_source_limit_ablation.py` | 真实 Agent 控制面四臂配对驱动（`DemoAgentService.run` → `LoopController` → `CandidatePlane`） |

## 2. 已有结果（可复算，勿夸大）

| 产物 | 规模 | 结论 |
|---|---|---|
| `runs/p2b_agent_source_limit/` | 12 例 × 4 臂 | 池 10→199；进池率 0→16.7%；HR/NDCG 全 0 |
| `runs/p2b_agent_source_limit_40/` | 40 例 × 4 臂 | 池 10/40/100/200；进池率 0/2.5/5/5%；HR/NDCG 全 0；p50≈1.4 s 四臂持平 |

**已知混杂（关键）**：驱动给全部案例用了同一条消息 `"recommend something for me"`。而
`catalog_search` 的查询只从消息抽词（`demo/agent_policy.py:383-390`），因此查询恒定、返回同一组商品，
Top-5 跨案例几乎相同（positions 1/3/5 完全一致）。**"5% 进池"是实验输入造成的，不是召回结论。**

## 3. 2B.3 要做的事

- **A** 案例级代理查询：取 `history[-2:]` 的**商品标题**，确定性拼接；**禁用** `items[-1]`（目标）与
  `items[-2]`（验证）；在 `cases.json` 标记 `query_source: "history-title-proxy"`。
- **B** 最小埋点（复用 `last_outcome` / `last_proposals` / ledger provenance，不新建账本）：
  每源 `requested_limit / effective_limit / returned / rank_in_source / ms / truncated_reason`；
  去重前后数量与目标是否被过滤；`fused_rank`；最终排名。`history` 源标 `pre_fused=true`。
- **C** 同 40 案例、四臂（5/20/50/100）、`requested_k=5`，写入 `runs/p2b_agent_source_limit_40b/`。
- **D** 四分支归因：从未出现 / 被过滤 / 进池但靠后 / 进 Top-5。
- **E** `docs/P2B_AGENT_SOURCE_LIMIT_ABLATION.md` + 产物 SHA-256 清单。

## 4. 复现

```bash
.venv/bin/python -m experiments.p2b_agent_source_limit_ablation --cases 40 --out runs/p2b_agent_source_limit_40b
.venv/bin/python -m pytest -q tests/test_agent_source_limit.py tests/test_p2b_metrics.py
```

## 5. 约束（不得违反）

不改生产查询构造、RRF 权重、排序策略、`requested_k` 默认值；不动冻结评测器 / 预注册 / 历史报告 /
检查点；不跑全量测试、不训练、不起服务、不调用外部 LLM；不自动提交 Git。

## 6. 当前判定

工程闭环成立（解耦、默认不变、可配对执行）；**推荐质量 NOT ESTABLISHED**；瓶颈位置**未验证**。
