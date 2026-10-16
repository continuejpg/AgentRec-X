# V2 任务 03–05 最终决策

## 1. 状态
| 任务 | 内容 | 状态 |
|---|---|---|
| **03 后端准入冻结** | 冻结 SASRec / Two-Tower 的模型、映射、Profile 与哈希 ✓；记录工程验收 PASS 而总体质量待确认 ✓；记录 TIGER 不准入与禁止启用范围 ✓；核查 `serving_v1` 未变 ✓；保留历史 PARTIAL 与新增 PASS ✓；整理版本清单（未提交 ✓） | **PASS** ✓ |
| **04 最小推荐服务链路确认** | 复用既有 `/v1/recommend`（未建平行 API ✓）；两条独立后端链路 + 缺省融合链路 + 身份/排名/去重/掩码契约 + 非法 profile 拒绝 + 响应可追溯 全部确认 ✓；既有服务**已满足**最小链路定义（未重构 ✓）；Agent/工具编排**未被排除出 V2 定义** ✓（其离线验收已存在、真实 LLM 部分受限 ✗） | **PASS** ✓ |
| **05 系统端到端验收** | 17 项边界测试全绿 ✓（正常/短/超长历史、K 变化、非法输入 4xx、非法 profile、启动关闭、单源与缺省不回归、无 TIGER/无降级 ✓）；冻结 parity/repeatability 直接引用 ✓；**唯一未闭合：未知身份 → 500 缺陷** ✗（需批准才可改路由语义 ✗） | **PARTIAL** ◐ |

## 2. 交付物
`V2_BACKEND_ADMISSION.md` ✓、`V2_ROADMAP_ALIGNMENT.md` ✓、`V2_MINIMAL_SERVICE_ACCEPTANCE.md` ✓、
`V2_RESOURCE_RISKS.md` ✓、本文件 ✓；配套：`docs/GATE_C12_R2_ACCEPTANCE.md` ✓、
`experiments/gate_c12_backend_acceptance.py` ✓、`tests/test_v2_gate_c12_profiles.py`（4 passed ✓）、
`tests/test_v2_task05_service_boundaries.py`（17 passed ✓）。

## 3. 最小推荐服务是否成立
**成立（带明确限定）** ✓：
* 真正调用已有推荐模型 ✓（SASRec / Two-Tower 真实权重 ✓）；
* 接口行为稳定 ✓（两轮 100/100 一致 ✓、边界 4xx 明确 ✓）；
* 商品身份与掩码合法 ✓（非法/PAD/重复/历史回显 = 0 ✓，parity 100/100 ✓）；
* 能独立部署与复现 ✓（哈希/协议/队列/seed/归档齐全 ✓，缺省 profile 与单源 profile 均可启动 ✓）；
* **不依赖 TIGER** ✓（源集合不含 tiger ✓，无隐式降级 ✓）；
* 质量限定明确 ✓：**exploratory service-level, n=100** ✓，**不构成**生产质量结论 ✗。
**唯一保留项**：未知身份 → 500 ✗（阻塞 B-1，见 §5）。

## 4. 仍需完成的 V2 任务数量：**2 项（+1 项可选）**
1. **P3 统一质量评测**（✗ 未开始）：更大**预先固定**队列（建议 1,000 例起）同协议复跑 SASRec 与 Two-Tower，
   产出配对差异与不确定性 ✓ ⇒ 这是"总体推荐质量待确认"的收口任务 ✓。
2. **错误映射修复 + 小回归**（✗ 未开始，需批准 ✗）：`FusionError → 4xx` ✓ + 一条测试 ✓ ⇒ 闭合任务 05 的 PARTIAL ✓。
3. **（可选）P4 融合对照** ✗：仅当 P3 显示互补性且**单独批准**后才讨论 ✓（**不自动启动** ✗）。
（E-04C 属另一阶段 ✗，本轮不进入 ✓。）

## 5. 决策请求（各一条）
* **B-1（需批准）**：新增 `FusionError`/`InferenceError` → **4xx + ErrorResponse** 的错误映射（**不改**评分/掩码/Schema 结构 ✓）。
* **B-2（需批准）**：P3 统一质量评测的队列规模与预算（建议 1,000 例、CPU、分钟级 ✓）。
* **B-3（需批准，可选）**：`_build_fusion_engines` 按 profile **惰性构建**（降低常驻内存 ⚠️，属引擎生命周期变更 ✗）。

## 6. 状态
未训练、未调参、未加载 TIGER、未融合、未进入 E-04C ✓；未执行任何 Git 提交/清理/重置 ✓；
两个既有 TIGER 失败继续单独披露 ✓；本轮已跟踪改动仅**获批的两处配置层补丁** ✓（+ 早期授权的
`docs/ARCHITECTURE.md` 文档修正 ✓）。**暂停等待技术负责人决策。**
