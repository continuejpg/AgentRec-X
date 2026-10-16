# Gate B-3 后端 CLI 入口对照（parity）

**结论：在"同一 100 例"上不可执行 ⇒ 记 UNAVAILABLE** ✗（不虚构数字 ✓）。

| 对照维度 | 结果 |
|---|---|
| 原始输入是否一致 | **不可比** ✗：后端 CLI（`backends/tiger_public/src/tiger_public/cli.py`）消费的是**交接队列** `runs/tiger_backend_handoff_prod/eval_cohort.jsonl`，其 `case_id` 为 0/1…（抽样队列 ✓，实测 history len 30/16 ✓），与本协议 100 例的 `user_int_id 1..100` **不是同一身份空间** ✗ |
| 原始候选集合是否一致 | **不可比** ✗（同上；且本协议案例未导出为该 CLI 可读队列 ✓） |
| Rank / Mask / Top-K 是否一致 | **不可比** ✗ |
| HR / NDCG 是否一致 | **不可比** ✗ |
| **可做的等价检查（已做）** | **入口一致性** ✓：本轮与 Gate B-3 均经 **同一** `TigerServingEngine` + **同一** Worker 路径 + **同一** CERTIFIED 模式与配置 ✓（`raw_candidates` 由该入口直接产出 ✓） |

**说明（避免误读 ✓）**：入口一致性**不等于**独立的模型质量证据 ✗；两条入口共享同一模型与 Worker ✓，
因此即使完全一致也只能证明"入口一致"✓，不能证明质量 ✓。
若需真正的 CLI parity，需要**新的**一项工作：把本协议 100 例导出为后端 CLI 的 cohort 形态（含其 `history`/
`required_frontier = K_max + |seen|` ✓），在其自有评测路径上重跑并比对 ✓ —— 属**新增实验**，本轮不做 ✗。
