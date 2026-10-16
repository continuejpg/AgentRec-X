# Gate C-1 · SASRec 真实 API 端到端验收报告
> 路由：`POST /v1/recommend`（官方 Schema；服务的是 `serving_v1` 融合 profile）。

> 结论：**PARTIAL** ✗ —— **真实 /v1/recommend 路由本身 100/100 通过合法性、Schema 与重复调用一致性** ✓，
> 但**无法通过该路由单独验收 SASRec** ✗（生产 serving 无单后端 profile ✓），且**API↔离线有序 Top20 一致性未能核验** ✗
> （应用未暴露已构建引擎句柄 ✗）。**未**修改任何生产代码/权重/队列/映射/Schema ✓；**未**加载 TIGER ✓。

## 归档与冻结
```text
archive      : runs/gate_c12_service_validation_r3 （新目录，未覆盖历史 ✓）
manifest     : manifest.json（协议 agentrecx.eval_protocol.v1 / temporal_leave_two_out / seed 20261015 ✓
               queue_sha256 / mappings_sha256 / sasrec_sha256=352bd3ae…105912 ✓ / two_tower_sha256=0e56be01… ✓
               K=20 / 队列=Gate B-1/B-2 同一 100 例 ✓ 不因重复交互或空结果排除 ✓）
SASRec       : runs/sasrec_canonical_2026/best.pt（sha 与 run.json 一致 ✓）；离线引擎 is_ready ✓；参数量 10.14M ✓
```

## 1) 真实路由执行（共享路由，服务的是 `serving_v1` **融合** profile ✗）
| 项 | 值 |
|---|---|
| 请求数（两轮） | **200**（100/轮 ✓） |
| 成功数 | **100/100** ✓（无 4xx/5xx ✓） |
| Schema 通过 | **100/100** ✓ |
| 非法商品 / PAD / 重复 / Seen 违规 | **0 / 0 / 0 / 0** ✓ |
| 非空响应 | **100/100** ✓ |
| 两轮有序 Item ID 一致 | **100/100** ✓（独立两轮文件 ✓，未挑选运行 ✓） |
| 单轮耗时 | 97.87 s / 96.89 s ✓ |
| 延迟 | p50 **997.48 ms**、p95 **1308.71 ms**、max **1480.04 ms** ✓ |
| 峰值 RSS（进程） | **6,313.5 MB** ⚠️（融合引擎 + 离线引擎同时驻留，接近 7 GB 上限 ⚠️） |
| 未解决异常 / 超时 | **0 / 0** ✓ |
| 路由级探索性指标 | HR@5/10/20 = **0.00 / 0.01 / 0.01**（hits 0/1/1，n=100 ✓）—— **属融合 profile** ✗，**不得**归因于 SASRec ✗ |

## 2) API ↔ 离线一致性：**UNAVAILABLE / NOT_COMPARABLE** ✗
* **不可比的原因（如实）**：① 路由输出是 **fusion（sequential+two-tower+lexical）** ✗，SASRec 单源输出 ≠ 路由输出 ✗；
  ② 应用**未暴露**已构建引擎（`app.state.engine` 读到 `None` ✗），故无法在不改生产代码的前提下取单源 Top-20 ✗。
  ⇒ 按任务规定判 **NOT_COMPARABLE** ✓，**不伪造 parity** ✓（归档中 `sasrec/parity.json` 的 `identical_top20=0/100`
  是**空比较产物** ✗，已在 `accessor_available=false` 标注 ✓，不作为不一致证据 ✗）。
* **离线参考（可用且有效 ✓）**：冻结协议下 100 例 Top-20 → hits@5/10/20 = **0 / 0 / 1**，HR@20 = **0.01** ✓
  —— 与 Gate B-1 归档（HR@5/10/20 = 0.01/0.01/0.02 ✗ 见下）**不完全相同**，差异来自**离线 top-k 掩码尾部实现** ✗（见 §3）✓。

## 3) 与 Gate B-1 的差异说明（不修改历史 ✓）
* 本轮离线参考的 hit@5/@10 为 0 ✗，而 Gate B-1 为 1 ✗：B-1 的 rank 来自 `rank_of_target`（**全目录名次** ✓），
  本轮离线参考取的是**掩码后列表的前 K** ✓，两者对"名次在 K 附近且被掩码项挤位"的案例**不等价** ✗ ——
  属**我本轮驱动的参考实现差异** ✗，**不是**模型或历史指标变化 ✓；**历史归档未改** ✓，正式指标仍以 B-1 为准 ✓。

## 4) 工程判定
**PARTIAL** ✗ —— 判定依据（逐条对照 PASS 条件 ✓）：
`100/100 路由执行 ✓`、`无异常 ✓`、`Schema 合法 ✓`、`无非法/PAD/重复/Seen ✓`、
`模型与映射符合冻结配置 ✓`，但 `API↔离线在可比协议下全部一致` **未达成** ✗（不可比 ✗）、
`单后端经路由验收` **未达成** ✗。**最小必要修复**：增加只读单后端 serving profile（配置层 ✓，不改评分/掩码/Schema ✓）
并在 `app.state` 暴露只读引擎引用 ✓ —— 二者均需**批准**后实施 ✗（本轮未改生产代码 ✓）。
