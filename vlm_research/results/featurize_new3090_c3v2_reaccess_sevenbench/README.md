# RTX3090 Physical C3-V2 + L11–L24 七 Benchmark

本目录仅保存冻结 B2 条件的研究证据。实际进度见 progress.json；只有最终记录和评分核验通过后才标为 COMPLETE。

- report_zh.md / score_summary.json：正式分数、MME任务、POPE三策略、MMBench Circular、OCRBench十子类。
- run_metadata.json：模型Revision、环境、输入和源码哈希、生成配置、物理语义及GPU资源。
- physical_smoke_summary.json：七类实际 Physical 执行检查与同算法短序列数值门禁；完整大跟踪不纳入GitHub或Windows必备归档。
- scoring_audit.json / manual_score_review.json：冻结评分复算、独立算术核验和少量逐项人工逻辑复核。
- paired_comparability.json：同协议前次C3与历史GQA对照的配对可比性；无完整Paper Native/OCR对照时不计算该差值。
- gqa_reuse_audit.json：历史12,578条B2的完整记录SHA、模型/源码、Gold和逐题实际输入张量哈希一致性；没有重新生成GQA正式全量。
- paper_protocol.json / gqa_protocol.json / ocr_protocol.json：三套原始协议分别保存，不能互换Prompt、图像预处理、生成上限或评分。
- ocr_SOURCE.json / ocr_statistics.json：官方OCRBench v1来源、固定Commit及SHA。原始id只在dataset内唯一；ATE键为官方行号。

实验计算基线：e281754ef17b3accc09d6938b5b9f020048b2aec。
历史GQA完整源码、原Manifest和结果：57e37ce024b1b2adf9fa24fc24aabec32a1fed08。
本轮入口是服务器已恢复环境的最小适配快照，依赖这两个已归档版本和恢复数据；没有修改冻结计算内核或五项评分器。

完整科研记录不纳入GitHub。Windows路径：

`D:\ATE-dataset-backups-3090\ATE-results\c3v2-reaccess-sevenbench`

其中 predictions.jsonl.gz 与 mapped_manifest.jsonl.gz 必须一起保存。前者包含原始输出、Gold和逐题分，后者包含完整Prompt、Options、Circular组/MME配对及图像关联；GQA历史对照压缩表也予以保留，以支持重评分、错误分析、配对与后续Bootstrap。恢复数据后按image_id解析原Linux image_path，无需SQLite。

七项数据位于同一Windows备份根目录，OCRBench-v1为新增完整解压目录，其余六项使用已核验的现有ZIP。
GitHub仅保存轻量证据及Windows实际核验凭据；最终 github_sync_receipt.json 和 shutdown_verification.json 在远端Push、文件HTTP核对与资源检查之后生成。
未取得所有最终门禁前，不得把本目录的生成完成状态解读为可以关闭服务器。本任务不会自动关机或删除服务器数据。
