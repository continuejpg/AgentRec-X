# archive/legacy_experiments — 历史实验脚本归档

## 一、归档说明

本目录由 **P14-0（仓库清理）** 生成，用于把**不再被当前代码、测试或配置引用**的历史实验脚本、
一次性运行日志与调试脚本移出工作区，同时保留**逐字节可恢复**的副本。

- 归档日期：2026-11-08
- 归档计划：`archive_plan.json`（schema `agentrecx.p14.archive_plan.v1`，P14-0 冻结）
- 校验清单：仓库根目录 `P14_PREP_ARCHIVE_MANIFEST.json`
  （记录每个原始文件的 sha256/size，以及每个 tar.gz 的 sha256）
- 归档文件总数：60 个文件，6 个分组；归档前已逐组完成「tar 列表比对 + 解包 sha256 逐文件比对」，全部通过后才删除原件
- tar 包内成员名就是**仓库相对路径**（正斜杠、无 `./` 前缀），因此解包到仓库根目录即可原位恢复

## 二、分组清单

| 分组 | 阶段 | 文件数 | git 跟踪 | 归档文件 |
| --- | --- | --- | --- | --- |
| `v2_pipeline_gates` | S2/S4/S6 (V2 E-04 gates, Gate C-1/C-2, TIGER canonical pre-steps) | 14 | 14/14 | `archive/legacy_experiments/v2_pipeline_gates.tar.gz` |
| `tiger_canonical_step26_28` | S4 TIGER canonical remediation (Step 2.6 / 2.7 / 2.8) | 10 | 10/10 | `archive/legacy_experiments/tiger_canonical_step26_28.tar.gz` |
| `learned_ranker_v1` | S7 / R-LR (LightGBM LambdaRank V1) | 6 | 6/6 | `archive/legacy_experiments/learned_ranker_v1.tar.gz` |
| `p2b_agent_diagnostics` | P2B/P2C (agent source-limit ablation diagnostics) | 6 | 6/6 | `archive/legacy_experiments/p2b_agent_diagnostics.tar.gz` |
| `m4_frozen_materialization` | S5 M4 (frozen execution / materialization tooling) | 7 | 7/7 | `archive/legacy_experiments/m4_frozen_materialization.tar.gz` |
| `root_scratch_p7p8` | P7/P8 local runs (transient scratch artifacts) | 17 | 0/17 | `archive/legacy_experiments/root_scratch_p7p8.tar.gz` |

## 三、恢复命令

在**仓库根目录**执行（按分组恢复）：

```powershell
tar -xzf archive/legacy_experiments/<group>.tar.gz -C .
```

恢复全部 6 组：

```powershell
Get-ChildItem archive/legacy_experiments/*.tar.gz | ForEach-Object { tar -xzf $_.FullName -C . }
```

Git Bash / WSL 下的等价命令：

```bash
for f in archive/legacy_experiments/*.tar.gz; do tar -xzf "$f" -C .; done
```

> 注意：若工作区已存在同名文件，恢复会覆盖它们。恢复前建议先 `git status` 确认工作区状态。

## 四、校验命令

1）查看归档内容（应与 manifest 中该组 `files[].original_path` 列表一致）：

```powershell
tar -tzf archive/legacy_experiments/<group>.tar.gz
```

2）校验 tar 包本身（应与 manifest 中该组 `archive_sha256` 一致）：

```powershell
Get-FileHash archive/legacy_experiments/<group>.tar.gz -Algorithm SHA256
```

3）恢复后逐文件校验（应与 manifest 中每个文件的 `sha256` / `size` 一致）：

```powershell
$manifest = Get-Content P14_PREP_ARCHIVE_MANIFEST.json -Raw | ConvertFrom-Json
$manifest.groups.files | ForEach-Object {
  $p = $_.original_path -replace '/', '\'
  $h = (Get-FileHash $p -Algorithm SHA256).Hash.ToLower()
  if ($h -eq $_.sha256) { "OK   $p" } else { "FAIL $p" }
}
```

## 五、从 git 历史恢复

归档前曾被 git 跟踪的文件（上表「git 跟踪」列的分子），其内容也可从 HEAD 历史恢复：

```powershell
git show HEAD:<original_path>        # 查看内容
git checkout HEAD -- <original_path> # 恢复到工作区
```

以上表为准：**未被 git 跟踪**的文件（多为仓库根目录的临时运行日志，如 `p7*.log`、`p8a_run.log`
等）不存在于 git 历史中，只能从本目录对应的 tar.gz 恢复。本目录中的归档副本是这些文件的唯一留存。
