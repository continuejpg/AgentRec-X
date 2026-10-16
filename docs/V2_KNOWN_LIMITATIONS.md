# V2 已知限制（冻结 2026-10-16 07:15，全部如实披露）

1. **单源 Profile ≠ 模型加载隔离**：应用仍构建 sequential + two_tower + lexical 三个引擎（`_build_fusion_engines` 契约所致），
   profile 只决定**参与排序**的 source。**不得**宣称加载已按 Profile 隔离。

2. **验收环境 RSS 约 6.1–6.4 GB**（三路径实测 6,129 / 6,351 / 6,170 MB）：这是**独立生产服务 + 离线参考同进程共驻**的
   验收形态，**不得**与独立部署的资源数据混为一谈；单后端在线部分历史实测约 0.6–0.9 GB。

3. **未做正式并发压测**：本轮全部为单进程顺序请求；并发、多 worker 与 p99 均**无证据**。

4. **TIGER 两项既有失败未修复**：`test_backend_boundaries.py::test_t7a_adapter_never_names_canonical_identity`（架构契约放置）、
   `tests/test_tiger_engine.py::test_result_satisfies_tiger_source_tool`（A 类替身漂移）。继续单独披露，未计入任何 PASS。

5. **内部 5xx 守卫测试缺实跑证据**：`test_an_internal_failure_is_still_a_server_error` 被跳过（无法取得引擎句柄）；
   错误映射只覆盖新子类，5xx 路径按构造未被改动，但**缺少一条实跑证据**。

6. **真实 LLM 在线闭环证据边界**：E-04A **4/4 额度已用尽**；此后未再执行任何在线调用 ⇒ 在线闭环为 **PARTIAL / UNKNOWN**。

7. **质量证据等级**：n=1,000 **固定前缀**、与早期 100 例开发队列**重叠 100**、**多重比较未校正**；
   `serving_v1` **含 lexical** ⇒ 无法把其表现归因于双模型互补。**不得**宣称生产推荐质量。

8. **TIGER 训练侧协议 UNKNOWN**（窗口/目标偏移/截断方向不可由仓库工件证明）⇒ 不得声称训练-推理协议一致。

9. **版本化未完成**：HEAD 仍为 `b3af9da`，本轮全部源码/测试/文档改动**未提交**（未获授权）。

10. **未验证命令**：`uvicorn` 监听端口的真实启动命令未实跑（验收使用进程内 ASGI 客户端）；
    `runs/**`（495 个被忽略文件）为本地证据归档，**不在 Git 内**，需独立持久化保存。
