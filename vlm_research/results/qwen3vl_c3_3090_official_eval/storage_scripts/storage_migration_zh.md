# 本轮本地盘运行与持续盘备份

用户在评测运行期间明确授权：切换本地盘运行，持续盘保留备份。

本地根目录为 `/home/featurize/ate-local-runs/qwen3090-29535-20261110`。其 `worktree`、`data`、`model`、`control` 是 ext4 上的实际运行副本；原 `/home/featurize/work/` 下的四个专用路径通过 bind mount 指向这些副本，因此冻结的绝对路径和 Manifest 字节保持不变。实际映射、完整性检查及迁移时已完成预测数量见 `local_disk_migration.json`。

`persistent_work_disk` 是原 NFS 持续盘的独立挂载别名，其中保留迁移前原件。路径包含 `ate-local-runs` 不能用来判断它是否本地，应用 `findmnt -T <路径>` 查看真实文件系统。迁移后新增结果先写本地；最终备份状态以 `persistent_final_backup_receipt.json` 和 Windows SHA256 校验回执为准。

四个映射只对当前实例有效。恢复已有实例时，先检查挂载、目录和 GPU 进程；完整模型文件应符合冻结模型 Manifest。迁移脚本会拒绝覆盖既有迁移目录及叠加已有挂载。迁移前退出本实验 GPU 进程并确认显存释放，再复制、校验、切换并恢复单一进程。没有重新下载模型、修改冻结源码或重建 Benchmark 样本。

`resume_scripts/` 保留实际恢复入口及与冻结原入口的差异。它复用已完成的历史 Smoke/GQA1000、完整视图和逐题检查点，保留已停止的 OOM 视图，不重试该视图。未保存的在途生成可以重算；已完成预测保持原样。恢复入口没有改变模型执行或评分逻辑，仍检查冻结源码、模型、协议及 GPU UUID。

本目录中的控制脚本记录本次迁移及最终归档过程。`package_results.py` 和 `backup_local_results_to_persistent.py` 只在本实验 GPU 进程结束后使用，生成并校验结果压缩包；不包含模型权重、数据集图片、GitHub Push 或服务器关机操作。
