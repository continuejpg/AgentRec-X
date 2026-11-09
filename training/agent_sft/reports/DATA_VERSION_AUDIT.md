# P2-SFT 数据版本核验报告

## 0. 结论

**数据确实发生了变化，不能再宣称 “data unchanged”。**

- 用户给出的原始版本：`a031...` / `47d3...` / `d790...`
- 当前保留版本：`fa3b...` / `c831...` / `e85f...`
- 原始版本的 `.jsonl` 文件快照已不在本工作区；当前工作区只有后续版本。  
  因此不能对两版做逐样本/逐字节 diff，也不能声称两者只差换行。

从本次生成代码变更历史可以确定的差异类别：

1. `price_update` 的 Gold 数值改为与用户话语中的预算一致；
2. `catalogue_context` 改为按目标类型选择和均衡；
3. `material_pref` 的 previous state goal 从 water_filter 改为 cookware；
4. 文件写入从文本模式改为精确字节写入。

当前版本已锁定为 AutoDL 使用版本：`p2sft.data.v1.1`。

---

## 1. 文件与哈希算法

两版使用相同文件路径和相同算法 `SHA-256`；但**不能证明相同文件字节**：

- 原始哈希来自当时生成流程的 manifest；
- 当时写入使用 `Path.write_text`，manifest 哈希对 `content.encode("utf-8")` 计算；
  在 Windows 上 `write_text` 可能把 LF 写为 CRLF，因此 manifest 哈希不一定是磁盘
  raw bytes 的哈希；
- 当前生成器使用 `Path.write_bytes(LF)`，manifest 哈希与磁盘 raw bytes 哈希一致。

当前实测：

| split | raw bytes SHA-256 | manifest SHA-256 | canonical LF SHA-256 | CRLF |
|---|---|---|---|---|
| train | fa3b1c82... | fa3b1c82... | fa3b1c82... | 否 |
| val | c8317ad3... | c8317ad3... | c8317ad3... | 否 |
| test | e85f8576... | e85f8576... | e85f8576... | 否 |

所以：

- **相同哈希算法**：SHA-256；
- **相同文件角色**：train/val/test JSONL；
- **不保证相同字节表示**：原始 manifest 可能是 canonical LF hash，当前是 raw LF hash；
- **已确认内容变化**：原始内容快照不在，不能做字节级比对；但代码变更历史表明逻辑内容
  已变化，且无法用换行/排序解释全部哈希变化。

---

## 2. JSON 序列化、排序与换行

当前数据：

- 每行一个 JSON object；
- `json.dumps(..., ensure_ascii=False, sort_keys=True)`，紧凑分隔符；
- LF 换行，无 CRLF；
- sample ID 模式不变：`p2sft_{split}_{index:05d}`；
- 字段集合不变：
  `sample_id, split, scenario, template_id, state, catalogue_context, user_message,
  assistant, source`；
- 样本数不变：2400/240/200；
- 场景数 11。

原始 manifest 的“文本模式写入 + canonical hash”会同时造成：

- 原始磁盘文件可能为 CRLF；
- 原始 manifest hash 不是 raw-file hash。

这可以解释“哈希值不同”的一部分，但**不能解释全部**：因为代码变更已改动标签和上下文。

---

## 3. 已知的语义差异

### A. `price_update` Gold 数值

原始生成器示例中出现过：

```text
user_message: "Actually make the budget under $30."
assistant.operations[0].value: 80
```

当前生成器从 `evidence`/用户话语解析预算：

```text
user_message: "Actually make the budget under $30."
assistant.operations[0].value: 30
```

影响场景：`price_update`（当前计数 train 426 / val 22 / test 19）。  
原始版本的错误样本数未知，因为原始快照不存在；不能伪造具体条数。

### B. Catalogue context 对齐

原始上下文从 catalogue 随机选取，可出现：

```text
state.goal.product_type = water_filter
catalogue_context.category = Sleeping Bags
```

当前按目标类型均衡选取，water_filter 样本优先配 Water Filters/Water Purifiers。  
影响字段：`catalogue_context`；`new_goal` 场景还可能影响标签目标与 user message 的对应。

### C. `material_pref` 的 previous state

原始：

```text
state.goal = {"product_type": "water_filter", "role": "main"}
```

当前：

```text
state.goal = {"product_type": "cookware", "role": "main"}
```

影响场景：`material_pref`（当前 counts train 141 / val 22 / test 18）。

### D. 不应被误判为差异的项

以下维持不变：

- sample ID 格式；
- 样本数量；
- scenario 集合与每 split 的场景覆盖；
- assistant JSON Schema 版本；
- 观测字段集合；
- 没有 ASIN 监督目标；
- Parser Test 的样本来源仍是独立 test 模板。

---

## 4. 锁定版本

锁定文件：

`training/agent_sft/data/LOCKED_VERSION.json`

锁定值：

```text
version: p2sft.data.v1.1
train: fa3b1c828bdc9d8cbaca51e7eddd656fb495fbf15bc4cb3fe6751fa08bc8df97
val:   c8317ad3d9d8822096bf557559407d82f10af18fcaa3738751b413758f95dd05
test:  e85f85767508d0a2533134e031dd78e46553c93360b3866566532bd6b6807b02
```

从本次锁定起：

- 不重新生成训练/验证/test 数据；
- 不覆盖当前 Parser Test；
- AutoDL 使用上述 hash 作为实验记录；
- 任何后续数据修复必须新增版本号，不能覆盖当前锁定文件。

---

## 5. 人工核验准备

`training/agent_sft/data/manual_review_test_sample.csv` 已包含 20 条当前 test 样本，
但 `label_status=pending`，`human_verified=false`，**没有冒充人工审核**。

自动预检记录：

`training/agent_sft/reports/manual_review_20_check_record.json`

自动预检只检查：

- sample 存在于当前 test；
- schema/evidence 可验证；
- gold 与 test 一致；
- user message 与 test 一致；
- 20 个唯一 sample ID。

它不评定相关性、不替代人工核验。

---

## 6. 对 AutoDL 的影响

- 由于原始快照不可恢复，不能承诺“使用原始 a031 数据”；
- AutoDL 应使用锁定版本 `p2sft.data.v1.1`，并在训练报告中记录上述 SHA-256；
- 不要用旧报告中的 a031/47d/d790 作为训练数据 hash；
- 不要重新生成数据来“修复”版本差异；
- 原始 a031 版本若必须用于归档，只能从仓库外备份恢复；本工作区没有该备份。

## 7. 诚实性声明

- 本次没有重新生成数据；
- 没有覆盖原始 Test；
- 没有做人工相关性审核；
- 原版本文件缺失，无法提供逐样本 diff；
- 由于内容修复已经发生，后续实验只能按 `p2sft.data.v1.1` 记录。
