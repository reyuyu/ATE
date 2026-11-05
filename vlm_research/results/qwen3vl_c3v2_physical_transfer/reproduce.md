# 复现说明（停止状态，禁止自动续跑）

本结果是数值验收失败的交付，不是完成的质量/速度实验。冻结推理源码Commit为`dae5207d016acea8225f236ce3b81285d3d8bd90`；最终交付Commit仅增补CPU分析、日志和文档。

工作目录：`/data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer`。

模型环境：`/data/lk/深度学习/.envs/ate-qwen3vl/bin/python`。制图环境：`/data/lk/深度学习/.envs/ate-vlm-research/bin/python`。模型和图片使用Manifest里的服务器路径，Git不包含模型权重或原始图片。

CPU测试：

```bash
cd /data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python \
 -m vlm_research.qwen3vl_c3v2_physical.tests
```

CPU审计和作图脚本分别位于`reproduction/audit_cpu.py`、`reproduction/plot_cpu.py`。审计脚本会重建报告CSV和停止状态文件，复核交付时应先复制结果目录，避免改变已发布文件的时间记录和哈希索引。它们不加载模型、不运行GPU。可审查代码或在复制的独立目录修改其明确的W/O路径后执行。完整离线校验建议直接验证`artifact_index.json`中的SHA并复算已有CSV及形状数据。

历史GPU启动命令（仅记录，不会在交付时执行）：

```bash
cd /data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer
CUDA_VISIBLE_DEVICES=GPU-96bc853f-ceea-2404-c704-c48a3d1ca185 \
 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
 /data/lk/深度学习/.envs/ate-qwen3vl/bin/python \
 -m vlm_research.qwen3vl_c3v2_physical.worker 0 smoke
```

另一卡使用`GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6`、rank 1。每卡各加载一份模型。

现在已有STOP标记，原始绝对截止时间也已冻结。Worker会拒绝在STOP之后工作，命令不构成继续GPU任务的授权。不得直接移除STOP、延长截止或放宽容差来把失败变成通过。需要修正/定位物理误差时，应另行获得用户针对后续工作的授权、建立新分支并保留原失败现场。

`reproduction/historical_dispatch.py`仅记录当时的提交和资源预检实现。其freeze模式会修改Git，**不是交付校验命令**。

本轮没有正式20,000次Bootstrap、非劣效检验或重复测速结果。相应CSV明确标记NOT_RUN或仅保留表头；不得填入历史数值或把缺失值视作零。
