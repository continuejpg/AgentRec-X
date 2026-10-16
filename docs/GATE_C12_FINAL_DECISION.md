# Gate C-1 + C-2 最终决策

## 决策：**C —— SASRec 工程验收未完成**（PARTIAL）✗

```text
C-1 SASRec     : PARTIAL ✗（路由层合法性/Schema/重复性 100/100 通过 ✓；单后端 API 证据与 API↔离线一致性 UNAVAILABLE ✗）
C-2 Two-Tower  : PARTIAL ✗（同一共享路由 ✓，同因受阻 ✗；C-2 的路由层检查已执行并全部通过 ✓）
```
按任务规定：C-1 非 PASS ⇒ **不推进**后续阶段 ✗；**不启动融合** ✗；**不进入 E-04C** ✗。

## 三条阻塞（最小必要修复，均需批准后实施 ✗，本轮未改生产代码 ✓）
1. **无单后端 serving profile** ✗：`FUSION_PROFILES` 仅 `serving_v1`（融合）与 `serving_v2_tiger`（禁用 ✗）
   ⇒ 该路由无法通过 `POST /v1/recommend` 单独服务 SASRec 或 Two-Tower ✗。
   **建议**：新增**只读**的 `serving_v1_sequential_only` 与 `serving_v1_two_tower_only`（仅 profile/source 选择 ✓，
   不改评分、掩码、Schema、tie-breaking ✓），并加回归测试 ✓。
2. **应用未暴露已构建引擎** ✗：`app.state.engine` 读到 `None` ✗ ⇒ 无法在不改生产代码的前提下核验单后端 Top-20 与离线一致性 ✗。
   **建议**：在 `app.state` 暴露**只读**引擎引用（或 health 中列出已启用 source/profile ✓），不改变行为 ✓。
3. **内存余量不足** ⚠️：本轮峰值 **6,313.5 MB / ~7 GB** ✗（融合引擎 + 离线引擎同时驻留）。
   **建议**：验收驱动**串行**加载并显式释放（`del` + `gc.collect()` ✓），或将离线核对拆成独立进程 ✓。

## 已确证可用的部分（可继续复用 ✓）
* 真实应用生命周期 + 官方 Schema 路由 ✓；100×2 请求 **0 异常、0 非法/PAD/重复/Seen** ✓；**两轮逐位一致** ✓。
* 真实权重经离线路径可用 ✓：SASRec HR@20 **0.01**、Two-Tower HR@20 **0.02**（exploratory，n=100 ✓，
  与 B-1/B-2 同量级 ✓；**不构成**总体质量或优劣结论 ✗）。
* 归档：`runs/gate_c12_service_validation_r3/{manifest.json, route/*, sasrec/*, two_tower/*, final_summary.json}` ✓（可由逐例记录重算全部统计 ✓）。
* 两项既有失败测试与 TIGER 结项记录**未被触碰** ✓。

## 下一步（待批准，仅一项）
按阻塞 1+2 做**配置层**最小改动（只读 profile + 只读引擎引用）→ 重跑本驱动 → 目标：C-1/C-2 各自
**API↔离线 Top-20 100/100 一致** ✓ + 单后端路由 100/100 合法 ✓。预计成本：约 1 轮（CPU 分钟级 ✓，无下载/训练 ✓）。
