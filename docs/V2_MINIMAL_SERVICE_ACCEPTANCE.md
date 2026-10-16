# V2 最小推荐服务 · 系统端到端验收（任务 05）

> 复用冻结结果（**不重复** C-1/C-2 的 100 例评测 ✓），本轮只补少量**边界覆盖**测试 ✓。
> 新增 `tests/test_v2_task05_service_boundaries.py` → **17 passed** ✓（真实应用 lifespan + 真实路由 ✓）。

## 1. 边界矩阵（真实 `/v1/recommend`，profile = `serving_v1_sequential_only`）
| 场景 | 输入 | 实际结果 | 判定 |
|---|---|---|---|
| 正常历史 | 真实 6 项 parent_asin，k=10 | **200** ✓，返回 ≤10 张卡、rank 1..n 连续 ✓、无重复 ✓、`parent_asin` 非空 ✓ | PASS ✓ |
| 单条短历史 | 1 项，k=5 | **200** ✓（`history` min_length=1 ✓） | PASS ✓ |
| 超长历史 | 120 项（> max_seq_len 50 ✓） | **200** ✓，`history_length` 回显输入长度 ✓，返回 ≤20 ✓（**既有截断行为** ✓，未改规则 ✓） | PASS ✓ |
| 合法 K 变化 | k ∈ {1,5,20,50} | 全部 **200** ✓，`requested_k` 回显 ✓、`returned_k ≤ k` ✓ | PASS ✓ |
| 非法 K | k ∈ {0,−1,10000} | **422** ✓（Schema 边界 ✓） | PASS ✓ |
| 空历史 / 缺字段 / 类型错误 / 空串身份 | 5 组非法 payload | **4xx（422）** ✓，**无 500** ✓ | PASS ✓ |
| **未知商品身份** | `["NOT-A-REAL-ASIN-000"]` | **HTTP 500** ✗（`FusionError` 未被映射为 4xx ✗） | **FAIL（已知缺陷 ✗）** |
| 非法 Profile | `AGENTRECX_FUSION_PROFILE=bogus` | 启动即 **RuntimeError** ✓（不静默回退 ✓） | PASS ✓ |
| 应用启动 / 关闭 | `TestClient(app)` lifespan ✓ | 启动加载真实模型 ✓、退出无残留进程 ✓（本轮结束 python 进程数 = 0 ✓） | PASS ✓ |
| 单后端配置不回归 | 两个新 profile + 缺省 | 新 profile 各自单源 ✓；缺省 `serving_v1` 未变 ✓（36 + 4 passed ✓） | PASS ✓ |
| 无未授权 TIGER 调用/隐式降级 | 两个单源 profile 的 source 集合 | 均不含 `tiger` ✓、不含 `catalog_lexical` ✓；无 fallback ✓ | PASS ✓ |

## 2. 直接引用的冻结验收证据（不在本轮重跑 ✓）
* **API↔离线有序 Top20 parity**：SASRec **100/100** ✓、Two-Tower **100/100** ✓
  （`runs/gate_c12_service_validation_r2/{sasrec,two_tower}/parity.json` ✓）。
* **两轮重复调用一致**：各 **100/100** ✓（`repeatability.json` ✓）。
* **合法性**：成功 100/100 ✓、Schema 100/100 ✓、非法/PAD/重复/历史回显 = 0 ✓。
* **延迟/RSS**：SASRec p50 23.02 ms / p95 24.26 ms、Two-Tower p50 161.40 ms / p95 174.56 ms ✓；
  峰值 RSS 6,127 / 6,375 MB ⚠️（见 `V2_RESOURCE_RISKS.md` ✓）。
* 质量（exploratory, n=100 ✓）：SASRec HR@20 **0.01**；Two-Tower HR@20 **0.02** ✓（**不作总体质量结论** ✗）。

## 3. 已知缺陷（**如实披露，未擅自修复** ✗）
**未知商品身份 → HTTP 500** ✗：`FusionError("history contains N identity(ies) outside the served catalogue")`
（`recommendation/inference/fusion.py:456` ✓）未被 `api/app.py` 映射为客户端错误 ✗。
**影响**：调用方拿到 500 而非 4xx ✗，不利于区分"输入错误"与"服务故障" ✗。
**最小修复建议（需批准 ✗）**：在 `app.py` 增加 `@app.exception_handler(FusionError)`（与 `InferenceError` 一并）
→ 返回 **422/400 + `ErrorResponse{error,detail}`** ✓，并加一条针对性测试 ✓；
**不改**评分、掩码、Schema 结构 ✓（仅新增错误映射 ✓）。本轮**未实施** ✓。

## 4. 任务 05 判定：**PARTIAL** ◐
理由：核心链路（真实模型 + 稳定接口 + 身份/掩码合法 + 错误处理在多数分支明确 ✓）**已验收** ✓；
唯一未闭合项是上述 **500 缺陷** ✗，属**生产路由语义变更**，按纪律**暂停并申请批准** ✗，故整体判 **PARTIAL** ✓。
