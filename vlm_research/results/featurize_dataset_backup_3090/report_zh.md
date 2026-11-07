# ATE 新 RTX 3090 数据快速验收与 Featurize 备份记录

**最终状态：PARTIAL。快速验收和可迁移本地备份已完成；Featurize 平台上传 BLOCKED_AUTH，0/6 成功，当前不具备平台跨实例恢复条件。** GitHub 保存的只是轻量证据，不能替代数据备份。

## 实际结果

| Benchmark | 题数 | 图片数 | 自包含包大小（未压缩） | Featurize 上传状态 | Dataset ID | 远端验证状态 |
|---|---:|---:|---:|---|---|---|
| MME Classic | 2,374 | 1,187 | 869,354,335 B（829.08 MiB） | 失败：401 / exit 101 | 无，未创建 | 未验证 |
| ScienceQA-IMG Test | 2,017 | 2,017 | 169,181,496 B（161.34 MiB） | 失败：401 / exit 101 | 无，未创建 | 未验证 |
| POPE 三策略 | 9,000 | 500 | 95,318,783 B（90.90 MiB） | 失败：401 / exit 101 | 无，未创建 | 未验证 |
| TextVQA + Paper OCR | 5,000 | 3,166 | 928,886,267 B（885.85 MiB） | 失败：401 / exit 101 | 无，未创建 | 未验证 |
| MMBench-EN Dev20230712 | 4,377 | 1,132 | 33,861,498 B（32.29 MiB） | 失败：401 / exit 101 | 无，未创建 | 未验证 |
| GQA testdev_balanced | 12,578 | 398 | 80,450,452 B（76.72 MiB） | 失败：401 / exit 101 | 无，未创建 | 未验证 |

合计 35,346 条基础题目、8,400 个图片路径，6 包共 **2,177,052,831 字节**。不同协议/策略视图不是新增评测题。包大小包含统一输入、独立 Gold、原样 Prompt、图片、必要清单、恢复脚本和部分原始标注；不含原始大归档/Parquet、MMBench 重复 base64 TSV、模型、环境、缓存或 Checkpoint。

六包已实际依次执行 `featurize dataset upload <dir>`。CLI 均显示打包完成，随后“创建数据集失败: Server http code 401 Unauthorized”，退出码均为 **101**。预置认证变量存在，但被平台拒绝；没有成功提示、真实 Dataset ID 或远端元数据确认。不对确定性 401 盲目重试。`upload_status.json` 保留逐包时间、退出码及错误摘要。

认证补充排查：预置 Token 与系统、工作区进程一致，无空白/引号或有效本地配置覆盖。当前官方 Linux CLI 已用独立用户目录中的兼容运行库成功启动（系统 libc/CLI 未替换）；只对最小的 MMBench 包补做 **1 次**真实上传，仍为 401 / exit 101。首次六次与这一次分开记录，共 7 次尝试，仍为 0/6 上传成功。未调用未文档化 API，也未找到公开认证刷新命令。需要平台修复实例认证或数据集创建权限；给客服的非敏感说明见 restore_guide.md。网页自动检查因工具无法可靠识别浏览器 URL 而停止，未操作网页。

远端统一标记 **REMOTE_CONTENT_NOT_FULLY_VERIFIED**。未重新下载平台数据，也没有远端逐文件 SHA 校验；这里更不能宣称平台已安全保存。认证修复并取得六个真实 ID 前，不能销毁当前实例来依赖这些“平台备份”。

## 本轮复核与历史证据

先读取并复用此前持久化报告 `../featurize_new3090_recovery/report_zh.md` 及对应 CPU/图片审计（它们仍在实例 work 目录，未随本次 5 个文件同步 GitHub）。历史 CPU 回归为 452 通过、5 跳过，辅助 4 通过；8,400 图片先前已全量解码并固定 SHA。指定 HF LLaVA Revision 先前已 CPU 加载；本轮没有加载模型。

本轮检查全部 14 视图 JSON/JSONL、28 个源 inputs/answers SHA、题数、ID 唯一性、Gold 双向映射、所有图片路径/长度及图像修改时间未晚于原全量审计。共 18 张代表图抽读；恢复 smoke 另抽读 3 张。新副本 67,024 行的非路径字段与原记录逐条一致，包括完整 Prompt。TextVQA 两协议已实际生成一个全新 runtime 目录并验证路径。没有重复全量回归、图像哈希或完整解码。

上传目录位于本地 ext4 `/home/featurize/data/ate-recovery/backup_staging_3090/ate3090-*`；图片用同文件系统硬链接，0 软链接，源图片未修改。包内使用相对图片路径；恢复脚本另行生成新绝对路径输入。源冻结文件和原 inputs/answers 保持不变。输入与 Gold、原始带 Gold 标注逻辑隔离。原始图像 SHA 全清单在平台待上传包内，不提交 GitHub。

持久化 work 为 NFS（30 GiB，检查时约 29 GiB 可用），数据所在本地盘检查时约 369 GiB 可用、CLI 打包后约 367 GiB 可用。没有把大图片复制进同步盘；CLI 归档缓存也未删除。

## 保留的版本差异

- MME `landmark/03d5e3bfc958be38.jpg` 及 `:0`/`:1` 两题：旧 SHA `61d010ea9fabdc0bf930362d3cc1e4dc554789036f219feba927551effecd510`；恢复图 SHA `ecd588bb65d6397126fec9dbe556b1759e47d72a735a7f8776f86598fbbf174a`。题目和 Gold 相同；其余 1,186 张图匹配。没有换图、删题或冒称完整重放。
- POPE 官方 9,000 题（3×3,000）与论文 8,910 行分别保存；不为对齐数量增删问题。
- TextVQA image-only 和 Paper OCR 各 5,000 题，共用 3,166 图，保留各自 Prompt。ScienceQA、MMBench、GQA 的旧图片匹配结论直接复用历史审计。

## 恢复与尚缺资源

先由平台恢复实例 CLI 认证，再按 `restore_guide.md` 对六个现有目录重试上传并记录真实 ID。成功后，另一台 **Featurize 实例**可按实际帮助 `featurize dataset extract <dataset_id> [destination]` 流式解压，再运行 `python restore_bundle.py BUNDLE --output NEW_RUNTIME_DIR`。当前 ID 为空，不能把这条通用语法当作已验证的跨实例恢复。

本地包已满足迁移的格式与依赖要求，但真正上传/跨实例远端内容仍未验证。模型权重与 Python 环境未纳入本次数据包；它们仍依赖当前实例本地文件或先前公开恢复流程。Author-Native 权重/CLIP/环境等旧缺件未在本轮补齐。没有启动新的 GPU 实验。

## GitHub 轻量同步

分支：`codex/featurize-dataset-backup-3090`，基于已存在 main `e89c8acce6e674ecfbd2c8acad57faf71bf4d547` 建立独立稀疏 worktree，只作为备份证据分支。研究源码实际基线仍是 alignment `e281754ef17b3accc09d6938b5b9f020048b2aec`，未假设 main 含最新实验。没有切换原研究目录或改动其他分支。

本次只同步 report_zh.md、dataset_registry.json、upload_status.json、restore_guide.md、restore_bundle.py 共 **5 文件、41826 字节**，最大单文件 16085 字节。仅明确路径暂存，提交前审查 staged stat/name/size；无图片、权重、大型哈希表、日志、LFS 或原始数据。

本报告对应提交 SHA 可在 GitHub 文件历史中查看；最终实际 Push 状态和 Commit SHA 另写实例 `local_git_sync.json`，并在交付消息列出，避免把未执行的 Push 写为成功。

官方参考：[CLI 上传/流式解压与续传](https://docs.featurize.cn/docs/manual/cli)、[数据集保存和长期不使用清理规则](https://docs.featurize.cn/docs/manual/dataset)。后者说明普通/付费用户分别 3/6 个月无使用可能被清理，成功上传也不表示永久保留。
