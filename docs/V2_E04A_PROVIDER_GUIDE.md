# AgentRec-X V2 / E-04A — 真实 Provider 配置与执行指南（配置准备，未调用任何在线模型）

> 状态：**准备就绪，等待授权**。本文件不包含任何 API Key，也不要求把 Key 粘贴到聊天/报告/仓库。
> 本轮（E-04A 配置阶段）**没有发送任何在线请求**：`provider_configured() == False`，
> 全部 `AGENTRECX_LLM_*` 变量未设置，`experiments/e04a_provider_smoke.py` 仅执行了 dry run。

---

## 1. 项目实际兼容的 Provider 协议，以及 `BASE_URL` 是否需要 `/v1`

项目**不绑定某一家厂商**：`recommendation/control/provider_adapter.py` 实现的是
**OpenAI 兼容的 `/chat/completions`** 协议，并以「配置数据 + 两个 profile」区分厂商差异：

| profile | 适用 | 关键差异（`PROVIDER_PROFILES`，`provider_adapter.py:155`） |
|---|---|---|
| `deepseek`（默认匹配） | DeepSeek 官方 API | 关闭思维链需 `thinking={"type":"disabled"}`；该模式下忽略 `temperature`；支持 `response_format={"type":"json_object"}`；request id 头 `x-request-id`；`reasoning_content` **永不上报** |
| `openai_compatible` | 任何 OpenAI 兼容网关/自建推理服务（vLLM、SGLang、Ollama 的 OpenAI 端点、OneAPI 等） | 不做厂商特有字段；JSON mode 视部署而定（`AGENTRECX_LLM_JSON_MODE=0` 可关） |

**`BASE_URL` 是否必须包含 `/v1`：不必须。** 适配器的唯一规则是
（`provider_adapter.py:390`）：

```text
实际请求 URL = AGENTRECX_LLM_BASE_URL.rstrip("/") + "/chat/completions"
```

因此**判断标准是"拼接结果是否为官方文档的端点"**，而不是 base 里有没有 `/v1`：

| 配置的 BASE_URL | 实际请求 URL | 是否有效 |
|---|---|---|
| `https://api.deepseek.com` | `https://api.deepseek.com/chat/completions` | ✓ 官方端点（DeepSeek 文档给出的 base_url 之一） |
| `https://api.deepseek.com/v1` | `https://api.deepseek.com/v1/chat/completions` | ✓ 官方端点（`/v1` 与模型版本无关） |
| `http://127.0.0.1:8000/v1`（vLLM 等） | `.../v1/chat/completions` | ✓ 见部署文档 |

> 更正说明：本指南的早期版本写成"必须包含 `/v1`"，这是**过度限制** ✗。已按适配器真实拼接规则更正；
> 本项目 `.env` 里配置的是 `https://api.deepseek.com/v1`，拼接后为
> `https://api.deepseek.com/v1/chat/completions` ✓ 合法。

profile 的选择：设 `AGENTRECX_LLM_PROFILE`（`deepseek` / `openai_compatible`），不设时按 base_url
推断（`resolve_profile`）。

## 1b. 复用原启动器的 `.env` 加载（**推荐路径，无需重新配置**）

项目根目录**已有 `.env`**（git 已忽略 ✓），并且 `scripts/start_demo.sh` 自带一个**解析式**加载器
（`load_dotenv_into_environment`，`scripts/start_demo.sh:53-97`，在 `:157` 处调用）：
优先级 **进程环境 > `.env` > 应用默认值** ✓，`declare -p` 检测已存在的变量而**不覆盖** ✓，
**逐行解析而非 source** ✓（`.env` 里的 shell 语句不会执行 ✓），且**从不回显任何值** ✓。

该脚本被设计为**可安全 source**（`if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then return 0; fi`，
位于服务启动逻辑之前 ✓），因此直接复用它即可，**不必**新建 Provider、**不必**复制 Key、
**不必**改动任何业务代码：

```bash
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X
source scripts/start_demo.sh                       # 被 source 时定义加载器后立即返回，不启动服务
load_dotenv_into_environment "$PWD/.env"           # 进程环境优先；只解析，不回显
echo "loaded=${DOTENV_LOADED} preserved=${DOTENV_SKIPPED}"   # 只打印计数，不打印值
```

⚠️ 已知缺口（本轮只读，未修改代码 ✗）：独立脚本 `experiments/e04a_provider_smoke.py`
**不会自己加载 `.env`**（它只读 `os.environ` ✗），所以在它之前必须先跑上面的两行；
否则会得到 `provider_configured() == False` ✗（这正是 E-04A 首轮"未读到配置"的原因 ✓）。
若希望脚本自给自足，建议后续（**需你授权后再改**）在 `main()` 开头加一个只解析
`<repo>/.env`、且"已存在的环境变量优先"的 5 行加载器，行为与 `start_demo.sh` 保持一致 ✓。

## 2. 适用于当前 WSL/Linux 的环境变量配置命令模板

> 若 §1b 的 `.env` 已存在（本项目即是），**本节模板不必执行** ✓ —— 仅在你需要覆盖某个变量
> （例如临时换模型）时使用；进程环境的值优先于 `.env` ✓。

**不要把 Key 写进命令行**（会进 shell 历史 ✗）。推荐用 `read -s` 交互注入当前进程环境：

```bash
# ① 交互式注入（Key 不回显、不落盘、不进历史）
export AGENTRECX_LLM_BASE_URL="https://api.deepseek.com/v1"   # 必须含 /v1（见 §1）
export AGENTRECX_LLM_MODEL="deepseek-chat"
read -rsp "AGENTRECX_LLM_API_KEY: " AGENTRECX_LLM_API_KEY && echo
export AGENTRECX_LLM_API_KEY

# ② 可选：超时 / JSON 模式 / 思维链 / 单价（用于费用估算，见 §5）
export AGENTRECX_LLM_TIMEOUT=30
export AGENTRECX_LLM_JSON_MODE=1
export AGENTRECX_LLM_THINKING=0
export AGENTRECX_LLM_INPUT_PRICE=0.27      # 每百万 input token 的价格（按你的合同填写）
export AGENTRECX_LLM_OUTPUT_PRICE=1.10     # 每百万 output token 的价格

# ③ 本轮要覆盖的实验开关（默认全关）
export AGENTRECX_AGENT_REPLANNING=1        # E-03B 受限追加来源
export AGENTRECX_AGENT_EXPLANATIONS=1      # E-02B 商品 RAG

# ④ 脱敏自检（不打印任何 Key 内容，见 §7）
.venv/bin/python -m experiments.e04a_provider_smoke
```

前缀式（一次性、只作用于该命令）也可以，但**不要**把 Key 写在命令行里：

```bash
AGENTRECX_LLM_BASE_URL=... AGENTRECX_LLM_MODEL=... .venv/bin/python -m experiments.e04a_provider_smoke --run
# Key 仍需先通过 read -s 注入环境
```

## 3. Key 如何安全注入，避免落进 Git / 历史 / 日志 / 报告

| 风险面 | 措施（项目已具备的 + 操作要求） |
|---|---|
| Git | `.gitignore` 已忽略 `.env`、`.env.*`、`*.key`、`*.db`、`*.sqlite3`、`data/`、`artifacts/`、`*.pt` 等；**本轮与后续都不创建 `.env` 文件**（环境变量注入 ✓），因此不存在被 `git add` 的载体 |
| Shell 历史 | 只用 `read -rsp` 或由进程管理器注入；**禁止** `export KEY=sk-...` 字面量；如已误输，用 `history -d` 删除该行并轮换 Key |
| 日志 | `provider_adapter.py` / `model_client.py` / `tool_selection.py` / `replan_policy.py` / `explanation.py` 中**没有任何 logging 或 print**（已审计 ✓）；适配器对错误响应体只回状态码、不回传 body（`test_provider_record_replay.py` 断言 ✓），上游 `reasoning_content` 永不上报 ✓ |
| 报告 | 本指南只写变量名与模板；smoke 脚本只输出 `credential_present: bool` 与 `credential_length: int`（不含任何字符 ✓），base URL 只输出 scheme/host/port ✓ |
| 遥测文件 | `--telemetry PATH` 写出的是：用途、成功/失败、耗时、响应字符数、是否合法 JSON、token 数、费用估算——**不含 prompt 原文、不含响应原文、不含 Key** ✓ |
| 测试 | 全部 1070 项测试使用 `ScriptedModelClient`/fake（无网络 ✓），不需要也不需要读取真实 Key |

## 4. `experiments/e04a_provider_smoke.py` 行为审计（`--max-calls` 是否是全局硬上限）

**是，而且是全流程共享的硬上限。** 依据：

1. 脚本用 `runtime.use_llm_client(capped)` 替换 **运行时唯一的那一个** client
   （`runtime.py:364`）；而四条模型路径全部走同一个对象：
   Tool Selection（`runtime.py:358` → `ToolSelectionPlanner(self.llm_client())`）、
   Product RAG（`runtime.py:820` → `client=self.llm_client()`）、
   Bounded Replanning（E-03B 的 policy 由 service 用同一个 client 组装）、
   以及 `llm` 模式的 turn planner（`runtime.py:340`；`llm_agent` 模式不使用 ✓）。
2. `_CappedClient.complete` 在 `calls >= limit` 时**抛错而不发送** ✓，因此第 `N+1` 次调用不会产生任何网络请求 ✓。
3. **失败不重试**：脚本自身无重试循环 ✓，并调用 `use_llm_client(capped, max_attempts=1)`
   （`runtime.py:364`，默认是 2）⇒ 计划解析失败即该轮失败，不会二次请求 ✓。
   （若保留默认 2，则「一次决策最多 2 次调用」，同样被 cap 计数 ✓ —— 两种情形都不会突破 `--max-calls` ✓。）
4. **没有隐式调用**：`runtime.llm_client()` 只做构造与配置读取，不发请求 ✓；控制器、M9、M10A/M10B、
   RRF、GroundingVerifier 都不调用模型 ✓；`--run` 之外的一切（含 dry run）都不联网 ✓。

**调用次数上界**（默认参数、`--max-calls 6`）：Tool Selection 1 次 + Replanning ≤1 次 +
Product RAG 1 次 + Memory 轮 1 次（偏好语句仍走一次 Tool Selection）= **≤4 次**，
`--max-calls` 仍有 2 次余量；若把 `--replanning/--explanations` 全开并保留重试（`max_attempts=2`），
最坏情形也**不会超过 6** ✓（这正是 cap 的作用 ✓）。

## 5. 建议模型、调用次数、Token 上限与预估费用（**并明确说明本脚本不设费用硬上限**）

| 项目 | 建议 | 说明 |
|---|---|---|
| 模型 | ⚠️ 本项目 `.env` 当前是 **`deepseek-flash`**：它**不在** DeepSeek 官方文档的 API 模型标识中（官方为 `deepseek-chat` / `deepseek-reasoner`），倒与「DSH Harness 的模型别名」同名。**必须先按原样探测（1 次调用）再决定**，不得盲目更换 ✓。若 provider 回 `Model Not Exist`/400，再与你确认是否改为 `deepseek-chat`（本轮不改 ✗） | 只需结构化 JSON 输出；不需要长上下文模型 |
| 调用次数 | **硬上限 `--max-calls 6`**（建议先用 `--max-calls 4` 跑 A+B） | 由 `_CappedClient` 强制 ✓ |
| 分步调用预算 | A 1 次、B 1 次（偏好语句的 planning）、C ≤1 次、D 1 次 | 见 §6 |
| **Token 上限** | ⚠️ **脚本没有 token 硬上限**：只能**报告** provider 返回的 token 数与费用估算（§5 末），不能阻止一次超长响应 | 若要硬约束，需在 Provider 侧设置账户/密钥级配额（最可靠） |
| **费用硬上限** | ⚠️ **不存在**。`--max-calls` 只是**调用次数**上限，**不得**当作费用上限 | 诚实边界：单次调用费用取决于 prompt 长度与模型定价 |
| 费用**估算** | 设 `AGENTRECX_LLM_INPUT_PRICE` / `OUTPUT_PRICE`（每百万 token）后，脚本对每次调用输出 `estimated_cost`（`provider_adapter.py:402` 的 `estimate_cost` 同口径 ✓）；未设价格时输出 `null` ✓ | 估算依赖 provider 是否返回 usage |
| 粗估 | 本 smoke 的 4 次调用 prompt 都在 1–3 KB 量级（计划/观察/1 个商品），总 token 约 **3–8 K**；按 DeepSeek 公开价（约 ¥1/百万 input、¥2/百万 output 量级）总费用 **远低于 0.01 元** | 仅供量级参考，**以你的合同价与 provider 账单为准** |
| 超时 | `AGENTRECX_LLM_TIMEOUT=30`（默认 30s ✓） | 超时按 `ModelCallError` 处理，不重试 ✓ |

## 6. 各能力对应的测试与所需真实调用次数

| 能力 | 覆盖命令（离线） | 真实调用需求 |
|---|---|---|
| **Tool Selection** | `tests/test_llm_tool_selection.py`、`tests/test_demo_llm_agent_mode.py`、`tests/test_llm_selected_controller.py` | **1 次**：一次 `llm_agent` 推荐请求的结构化计划 |
| **Memory（多轮受限投影）** | `tests/test_demo_llm_agent_memory.py`、`tests/test_demo_llm_agent_memory_acceptance.py`、`tests/test_memory_query_adapter.py`、`tests/test_memory_bm25_retrieval.py` | **1 次**（偏好语句轮的 planning）；M9 写入/投影本身不调用模型 ✓ |
| **Bounded Replanning** | `tests/test_v2_bounded_replanning.py`、`tests/test_v2_observation_replan_gate.py` | **≤1 次**（仅当 Gate 允许且模型选择追加来源） |
| **Product RAG** | `tests/test_product_knowledge_rag.py`、`tests/test_candidate_reference_boundary.py` | **1 次**（claim 生成；检索与验证为确定性 ✓） |
| **商品反馈（E-03C）** | `tests/test_v2_product_feedback.py` | **0 次**（纯确定性身份绑定与 M9 写入 ✓） |
| **Schema 完整性** | `tests/test_v2_schema_integrity.py` | **0 次** |
| **错误/超时/非法结构** | `tests/test_provider_record_replay.py`（63 项）、`tests/test_v2_bounded_replanning.py` 的非法输出用例 | **0 次**（离线异常路径已覆盖；不为此人为制造线上错误请求 ✓） |

## 7. 脱敏配置自检方法

```bash
# 只输出「有没有配」与「Key 有多长」，绝不输出 Key 内容或完整 URL
.venv/bin/python -m experiments.e04a_provider_smoke          # dry run，不联网
```

输出字段：`configured` / `base_url`（scheme://host:port）/ `model` / `profile` /
`credential_present` / `credential_length` / `timeout_s` / `json_mode` / `thinking` /
`input_price_per_million` / `output_price_per_million` / `replanning` / `explanations` ✓。

补充交叉自检（不打印值）：

```bash
.venv/bin/python - <<'PY'
import os
print("base_url set:", bool(os.environ.get("AGENTRECX_LLM_BASE_URL")))
print("model set:", bool(os.environ.get("AGENTRECX_LLM_MODEL")))
print("key set:", bool(os.environ.get("AGENTRECX_LLM_API_KEY")),
      "length:", len(os.environ.get("AGENTRECX_LLM_API_KEY", "")))
from recommendation.control.provider_adapter import provider_configured, provider_settings
print("provider_configured:", provider_configured())
print("profile:", provider_settings().profile.name, "| timeout:", provider_settings().timeout)
PY
```

## 8. 执行授权后的运行命令（**在你第二次明确确认之前不会执行**）

```bash
# 第一步（最小）：Tool Selection + Memory，硬上限 4 次调用
.venv/bin/python -m experiments.e04a_provider_smoke --run --max-calls 4 \
    --telemetry /tmp/e04a_run1.json

# 第二步（可选）：加上 Replanning + Product RAG，硬上限 6 次调用
AGENTRECX_AGENT_REPLANNING=1 AGENTRECX_AGENT_EXPLANATIONS=1 \
  .venv/bin/python -m experiments.e04a_provider_smoke --run --max-calls 6 \
      --replanning --explanations --telemetry /tmp/e04a_run2.json
```

脚本会输出并可写出（`--telemetry`）：每步的 `planned_sources` / `action_counts` / `terminal`
（steps、tool_calls、limits）/ 最终商品 / `ranked_with_preferences` / `replan` / `rag` / 每条 reason 的
`status` 与引用的 `field_path`，以及每次调用的用途、成功与否、耗时、响应字符数、是否合法 JSON、
token 数与费用估算 ✓——**不含 Key、不含完整 prompt、不含 Memory 原文** ✓。

**授权前的当前状态**：无授权、无 Provider 配置 ⇒ 本轮按 **PARTIAL** 收尾（见
[`docs/V2_E04A_REPORT.md`](V2_E04A_REPORT.md)），未发送任何在线请求 ✓。
