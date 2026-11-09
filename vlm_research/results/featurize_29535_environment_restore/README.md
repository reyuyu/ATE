# RTX 3090 29535：模型与优先数据恢复回执

本分支只同步已经完成的 CPU 环境恢复证据；未进行 GPU 推理、Benchmark 或融合实验，未修改 main 或冻结研究源码。
恢复范围为固定 Revision 的 LLaVA-1.5-7B、原锁定 Python 环境、GQA 12,578 题/398 图、TextVQA Paper-OCR 5,000 题/3,166 图。
其他五项图片数据按需暂缓；其既有研究结果保留。不能将本次恢复认作七项数据全部就绪或 GPU 推理验收。

[报告](report_zh.md) · [恢复步骤](restore_next_gpu_zh.md) · [CPU 完整性验收](resource_acceptance.json)

模型、数据和环境均在服务器本地 ext4；持续盘保留代码、历史结果和本次轻量证据。
权重按固定 LFS SHA256 校验，17 个模型文件齐全；两个真实 Windows→SFTP 数据包校验通过，原始输入与 Gold 哈希一致。
23 个历史文件直接复用，固定 200 题 CPU Token/像素绑定重建通过；CUDA 未初始化，未加载模型权重。

本机 SHA256 备份在 `D:\ATE-dataset-backups-3090\ATE-results\29535-environment-restore`。
`artifact_manifest.json` 描述完整离线恢复证据；GitHub 仅选择其中必要源码、报告和小型回执，省略安装日志。
不上传模型、数据集、图片或大型预测。恢复元数据中的未 Push 字段是资源恢复结束时的快照；
后续按用户既有同步要求发布独立证据分支，真实 Commit 与 HTTP 回读校验另保存于本机 `github_sync_receipt.json`。
