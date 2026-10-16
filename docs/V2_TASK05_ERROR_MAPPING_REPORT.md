# V2 任务 05 · 未知身份 500 缺陷的最小修复报告

## 1. 错误发生位置的确认（A1）
`FusionInferenceEngine.recommend` 的身份校验（`recommendation/inference/fusion.py:456`）在
`is_known_identity` 判定失败时抛出 **`FusionError`** ✗ —— 而 `app.py` 只注册了
`EngineRequestValidationError`、`UnknownItemError`、FastAPI 校验错误与 `InferenceError` 的处理器 ✗，
**没有** `FusionError` 处理器 ⇒ 未处理异常 ⇒ **HTTP 500** ✗。

**逐类区分（源码证据 ✓）**：`fusion.py` 中 `raise FusionError` 共 **12 处** ✓ —— 其中
`:294/:296/:299/:302/:308` 为**配置期**校验（构造时即失败，路由不可达 ✓）、`:432` 为 `pragma: no cover` 的
内部一致性 ✓、`:449`（k 非法）与 `:452`（空历史）经 HTTP 时**已被 Schema 拦住**（422 ✓ 不可达 ✓）、
`:209/:263` 为构造期参数校验 ✓ ⇒ **只有 `:456`（未知目录身份）在路由上是"客户端输入错误"** ✓。
⇒ 因此**不能**把整个 `FusionError` 映射为 4xx ✗（其余站点并非输入错误 ✓）。

## 2. 实施的精确映射（A2，仅本次授权的分类 ✓）
| 改动 | 内容 |
|---|---|
| `recommendation/inference/fusion.py` | 新增专用子类 **`UnknownCatalogueIdentity(FusionError)`** ✓（文档明确其"由构造即为客户端输入错误" ✓），并**仅**把 `:456` 的 `raise` 改为该子类 ✓（其余 11 处 `FusionError` 原样保留 ✓） |
| `recommendation/api/app.py` | 新增 `@app.exception_handler(UnknownCatalogueIdentity)` → **HTTP 422** + 既有 `ErrorResponse(error="unknown_item", detail=…)` ✓（`app.py:561` ✓，模块级 import `:707` ✓） |

满足全部要求 ✓：① 4xx（422）✓ ② 响应符合既有 `ErrorResponse` ✓ ③ **真正的内部故障仍为 5xx** ✓
（映射仅覆盖新子类 ✓，`RuntimeError`/其他 `FusionError` 不受影响 ✓）④ 不泄露堆栈/路径/敏感信息 ✓
（测试断言响应体不含 `Traceback` 与 `/mnt/` ✓）⑤ 成功响应 Schema **未变** ✓ ⑥ 排序、Seen Mask、身份命名空间、
融合策略**全部未动** ✓。**未**改评分算法、未改默认 Profile、未改 API Schema 结构 ✓。

## 3. 最小回归（A3，实跑 ✓）
`tests/test_v2_task05_service_boundaries.py` + `tests/test_serving_fusion.py` + `tests/test_api.py`
+ `tests/test_v2_gate_c12_profiles.py` → **57 passed, 1 skipped** ✓（70.91 s ✓）
| 检查 | 结果 |
|---|---|
| 正常请求 | **200** ✓ 且返回合法卡片 ✓ |
| **未知商品身份** | **422** ✓ + `error="unknown_item"` ✓ + 无堆栈/路径泄露 ✓（回归测试 `test_unknown_identity_is_a_client_error` ✓） |
| 空历史 / 缺字段 / 类型错误 / 空串身份 | 仍为 **422** ✓（原契约不变 ✓） |
| 非法 K（0/−1/10000） | 仍为 **422** ✓ |
| 默认 `serving_v1` 与两个单源 Profile | **不回归** ✓（36 + 4 项通过 ✓） |
| 不触发 TIGER | 两个单源 profile 的 source 集合不含 `tiger` ✓ |
| ⚠️ 内部故障 5xx 守卫测试 | **skipped** ✗（`test_an_internal_failure_is_still_a_server_error`：本仓库无法从模块作用域取得引擎句柄 ⇒ 该测试**未执行** ✗）。**如实说明**：5xx 路径**未被我的改动覆盖** ✓（映射只针对新子类 ✓），但**"内部故障仍为 5xx"缺少一条实跑证据** ✗（需引擎句柄或注入点，属测试基础设施限制 ✗）。 |

## 4. 任务 05 判定：**PASS** ✓
原阻塞（未知身份 500 ✗）**已修复并回归验证** ✓；其余边界（200/422/截断/非法输入/非法 profile/启动关闭/
单源与缺省不回归/无 TIGER 与无降级）**全部通过** ✓。**保留限定** ✗：单条 5xx 守卫测试被跳过 ⚠️（见上 ✓）。
