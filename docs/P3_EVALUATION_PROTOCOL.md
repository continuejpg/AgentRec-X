# P3 统一推荐质量评测 · 冻结协议（**本轮未执行** ✗）

> **状态：NOT EXECUTED（本轮）** ✗ —— 本轮仅完成 Stage A（任务 05 修复 ✓）与协议冻结 ✓。
> P3 的 1,000 例三路径评测**未运行** ⇒ **不存在** `P3_UNIFIED_METRICS.json`、`P3_PAIRED_BOOTSTRAP.json`、
> `P3_BACKEND_COMPLEMENTARITY.md`、`P3_FINAL_DECISION.md` 的结果内容 ✗（**不产出无实测支撑的文件** ✗）。
> 原因：**会话剩余容量** ✗（三路径 × 1,000 例 + 两轮 + 逐例归档预计 25–40 分钟机器时间与相应验证/写作容量 ✓），
> **不是**系统/内存阻塞 ✗（三条路径各自独立进程运行 ✓，单后端在线约 0.6–0.9 GB ✓）。

## 1. 队列（A1：预冻结，不看命中率 ✓）
```text
数据与拆分 : data/processed/Sports_and_Outdoors_sequences.json + recommendation/evaluation/split.py
             （agentrecx.eval_protocol.v1 / temporal_leave_two_out ✓）
抽样算法   : artifact 顺序（已有 reader `experiments/e04b2_gate_a.stream_sequences` ✓），
             取"前 1,000 个通过协议的有效 case" ✓ —— 确定性、无随机性、无需 seed 抽样
             （seed 20261015 记录为协议常量 ✓；若改为随机抽样则必须固定 seed 并记录 ✓）
Case ID    : 非敏感整数 user_int_id ✓（队列 SHA256 与 case 列表写入 `P3_*` 归档 ✓）
重叠说明   : 前 100 例 = Gate B-1/B-2/C-1/C-2 已用队列 ✗ ⇒ **P3 不是独立确认性实验** ✓，
             报告中必须写明重叠 100/1,000 且前 100 例已参与开发与验收 ✓
用户唯一性 : 该 artifact 每个用户恰有一条序列 ⇒ 1 case = 1 用户 ✓（若出现同用户多条，
             必须按用户聚类做 Bootstrap ✓）
历史/截断  : test_history = items[:-1] ✓；引擎各自 maxlen（SASRec 50 / Two-Tower 50 ✓）
目标定义   : test_target = items[-1] ✓
重复目标   : 目标若在历史中出现，按 `excluded_seen` 约定（正例仍可评分 ✓）
命名空间   : mappings.item2id，1-based、PAD=0、156,746 ✓
K          : 5 / 10 / 20 ✓（统一）
失败口径   : 技术失败计入**全部预选案例**分母 ✓，同时报"仅完成案例"分母 ✓（两口径并列 ✓）
```

## 2. 三条路径（B2，同一队列同规则 ✓）
| 路径 | Profile | 机制 |
|---|---|---|
| P3-A | `serving_v1_sequential_only` | 单源（SASRec ✓） |
| P3-B | `serving_v1_two_tower_only` | 单源（Two-Tower ✓） |
| P3-C | `serving_v1`（缺省 ✓，**既有融合**，非新算法 ✓） | sequential + two_tower + **catalog_lexical** ✓ ⇒ 报告必须**区分"融合效果"与"双模型互补性"** ✓ |

**不加载 TIGER** ✓（缺省 profile 的 source 集合不含 tiger ✓；已核查 `FUSION_SOURCES` ✓）；
**无失败后跨后端补位** ✗（各路径独立 ✓）。

## 3. 进程与内存（B3 ✓）
```text
顺序执行：P3-A 独立进程 → 关闭/确认退出 → P3-B 独立进程 → P3-C 独立进程 ✓
离线参考评分与在线服务分进程 ✓（避免上一轮 6 GB 级共驻 ✗）
若内存不足：保存已完成产物并暂停 ✓，**不**减少案例、不改模型、不降低 K ✓
```

## 4. 指标与统计（B4 ✓）
* 每路径：HR@5/10/20 ✓、NDCG@5/10/20 ✓、Hits/1000 ✓、成功率 ✓、空响应率 ✓、
  非法/PAD/重复/Seen 违规 ✓、延迟 p50/p95 ✓、RSS ✓、逐例 target rank ✓。
* **同一 Case ID 配对** ✓；逐样本配对差异归档 ✓。
* 若 1 case = 1 用户 ⇒ 常规 Bootstrap；若同用户多条 ⇒ **按用户聚类 Bootstrap**，
  报告 **95% CI** ✓ 与样本量 ✓；**不得**仅凭聚合指标大小宣称显著优劣 ✗。

## 5. 互补性分析（B5 ✓，仅作参考 ✓）
SASRec 独有命中 / Two-Tower 独有命中 / 共同命中 / 共同失败 ✓；
两模型 Top-20 集合重叠率 ✓；**Top-20 并集命中覆盖** ✓ —— 并集可能 >20 商品，
**只能作为互补性参考** ✗，**不得**与固定长度 HR@20 当作公平算法对照 ✗；
**禁止**依据 test 命中现场设计融合权重 ✗。

## 6. 执行命令（冻结，可直接复跑 ✓）
```bash
# 每条约 15–25 分钟（1,000 例 × 2 轮 + 离线参考）；建议三条路径逐条后台执行
AGENTRECX_FUSION_PROFILE=serving_v1_sequential_only python -m experiments.gate_c12_backend_acceptance --backend sasrec --cases 1000 --out runs/p3_unified
AGENTRECX_FUSION_PROFILE=serving_v1_two_tower_only  python -m experiments.gate_c12_backend_acceptance --backend two_tower --cases 1000 --out runs/p3_unified
# P3-C（缺省融合）：同一驱动在无 profile 前提下运行（需在驱动中增加 fusion 分支，属测试驱动改动 ✓ 无需批准 ✓）
```
> 备注：现有驱动 `experiments/gate_c12_backend_acceptance.py` 已支持 A/B 两条路径与逐例归档 ✓；
> P3-C 需在驱动中加一个"缺省 profile"分支 ✓（**仅测试驱动** ✓，不改生产 ✓）。
