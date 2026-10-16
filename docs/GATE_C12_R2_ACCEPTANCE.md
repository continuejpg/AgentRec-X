# Gate C-1 / C-2 · 最小修复后连续复测（R2）最终验收

> 结论：**C-1 SASRec = PASS** ✓ ； **C-2 Two-Tower = PASS** ✓ ；**统一建议 = A（双后端工程验收完成）** ✓。
> 说明：这是**工程服务验收**（真实路由 / 合法性 / 离线一致性 / 重复调用稳定），**不是**推荐效果达到生产质量标准 ✗。
> 上一轮的 PARTIAL 报告与其归档 **原样保留** ✓（本轮写入新目录 `runs/gate_c12_service_validation_r2/` ✓）。

## 1. 批准的最小配置修复（diff 摘要，仅配置层 ✓）
```text
recommendation/inference/fusion.py | 5 +++++      # FUSION_PROFILES += serving_v1_sequential_only / _two_tower_only
recommendation/api/app.py          | 11 +++++++++-  # 启动时按 AGENTRECX_FUSION_PROFILE 选择 profile（缺省 serving_v1 ✓）
2 files changed, 15 insertions(+), 1 deletion(-)
```
* 新增 profile 只选 **一个** source ✓，不改评分、Seen Mask、Top-K、tie-breaking、Schema、RRF ✓。
* 缺省（未设置 `AGENTRECX_FUSION_PROFILE`）⇒ `serving_v1` ✓ **行为不变** ✓：`tests/test_serving_fusion.py` + `tests/test_api.py` → **36 passed** ✓。
* 新 profile 隔离性测试：`tests/test_v2_gate_c12_profiles.py` → **4 passed** ✓（默认 profile 未变 ✓、各自的 sources 精确为单一 source ✓、**均不含 tiger** ✓、未知 profile 被拒绝 ✓）。
* ⚠️ **未达成 §2.9 的部分（如实登记 ✗）**：应用仍会构建全部三个引擎（`sequential`/`two_tower`/`lexical`）✗，
  仅**参与排序的 source** 被 profile 限制 ✓ ⇒ 峰值 RSS 6.1–6.4 GB ⚠️（接近上限 ✗）。未静默宣称已隔离 ✓。

## 2. 统一验收表（真实 `/v1/recommend`，两轮 × 100 例，逐例记录可复算 ✓）

| 指标 | SASRec（`serving_v1_sequential_only`） | Two-Tower（`serving_v1_two_tower_only`） |
|---|---|---|
| 模型 SHA256 | `352bd3ae…105912` ✓（与 run.json 一致 ✓） | `0e56be01…` ✓ |
| 实际加载成功 | ✓（离线引擎 `is_ready` ✓ + 路由 200 ✓） | ✓ |
| 真实 API 请求数 | 200（两轮 ✓） | 200（两轮 ✓） |
| 请求成功数 | **100/100** ✓ | **100/100** ✓ |
| Schema 通过数 | **100/100** ✓ | **100/100** ✓ |
| 非法商品数 | **0** ✓ | **0** ✓ |
| PAD 数 | **0** ✓ | **0** ✓ |
| 重复商品数 | **0** ✓ | **0** ✓ |
| 历史回显（history echo） | **0** ✓ | **0** ✓ |
| 候选数分布 | 20 / 20（恒为 K ✓） | 20 / 20 ✓ |
| **API vs 离线有序 Top20 一致** | **100/100** ✓ | **100/100** ✓ |
| **同输入重复调用一致** | **100/100** ✓ | **100/100** ✓ |
| HR@5/10/20（路由回包 ✓） | 0.00 / 0.00 / **0.01**（hits 0/0/1 ✓） | 0.01 / 0.01 / **0.02**（hits 1/1/2 ✓） |
| NDCG@5/10/20（路由回包 ✓） | 0.00 / 0.00 / **0.002447** ✓ | 0.006309 / 0.006309 / **0.008663** ✓ |
| 参考口径 hits（离线 ✓） | 0/0/1 ✓（与路由一致 ✓） | 1/1/2 ✓（与路由一致 ✓） |
| 单请求延迟 p50 / p95 | **23.02 / 24.26 ms** ✓ | **161.40 / 174.56 ms** ✓ |
| 单轮耗时 | 2.35 s / 2.35 s ✓ | 16.28 s / 16.31 s ✓ |
| 峰值 RSS | 6,127.0 MB ⚠️ | 6,375.1 MB ⚠️ |
| 未解决异常 / 超时 | **0 / 0** ✓ | **0 / 0** ✓ |
| **工程 Gate 状态** | **PASS** ✓ | **PASS** ✓ |
| 推荐质量证据等级 | **exploratory service-level evaluation, n=100** ✓ | 同 ✓ |

## 3. 排序契约与旧口径的关系（未改任何掩码/排名 ✓）
* 本轮 parity 参考 = 已冻结的 **API 排序契约**：`sorted_ranking(score_catalog, history=test_history)[:K]` ✓
  —— 路由不知道 target ✓，故参考也**不施加** target 掩码 ✓（这正是"同一排序问题" ✓）。
* Gate B-1 的 `rank_of_target` 使用**评测掩码**（`excluded_seen` ✓），与本轮参考**口径不同** ✗ ⇒
  两个数字**不得混用** ✗：B-1 历史指标（HR@5/10/20 = 0.01/0.01/0.02 ✓）**原样保留** ✓，
  本轮路由/参考口径为 **0.00/0.00/0.01**（SASRec）✓ 与 **0.01/0.01/0.02**（Two-Tower）✓。
* 两轮 parity 与 repeatability 均为 **100/100** ✓ ⇒ 上一轮 `NOT_COMPARABLE` ✗ **已不再适用** ✓，
  且归档中不再以 `0/100` 形式呈现不可比 ✗（`parity.json` 现为真实逐例比较 ✓）。
* **未**为追平任何 HR 数字修改掩码或 tie-breaking ✓。

## 4. 资源与执行纪律
* 严格**进程隔离顺序** ✓：C-1 独立进程 → 结束并确认系统内 python 进程数 = 0 ✓ → 再启 C-2 独立进程 ✓ → 结束 ✓。
* 未训练、未调参、未加载 TIGER、未融合、未进入 E-04C ✓；未改模型权重/身份映射/队列/Schema ✓。
* 旧归档未覆盖 ✓：`runs/gate_c12_service_validation_r3/`（上一轮 PARTIAL ✓）与
  `runs/e04b_gate_b*/`、`runs/e04b_p0*/` 全部保留 ✓；本轮结果在 `runs/gate_c12_service_validation_r2/` ✓
  （含 `manifest.json`、每后端 `responses_run{1,2}.jsonl`、`offline_reference.jsonl`、`parity.json`、
  `repeatability.json`、`metrics.json`、`resources.json`、`{backend}_summary.json` ✓）。
* 既有 TIGER 两项失败**继续单独披露** ✓（身份边界契约 ✗、`test_result_satisfies_tiger_source_tool` A 类 ✗），
  **未**计入本轮新增回归 ✓。

## 5. 结论与下一步
* **C-1 = PASS** ✓、**C-2 = PASS** ✓ ⇒ **统一建议 A：双后端工程验收完成** ✓；
  推荐下一阶段开展**统一质量评测**（更大固定队列）与**最小推荐系统交付** ✓，
  但**不自动启动融合** ✗（融合需单独批准 ✓）。
* 遗留（非阻塞）⚠️：引擎加载未按 profile 裁剪（内存 6.1–6.4 GB ✗）、
  质量证据仍为 100 例探索性 ✓、Two-Tower 无 sealed 全队列对照 ✗。
