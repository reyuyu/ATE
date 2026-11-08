# 八方法统一 E2E@32：GitHub 同步说明

240/240 正式请求、30个固定输入、单一RTX4090 UUID；所有8种方法通过Smoke和缓存审计。6轮Native漂移检查全部低于5%，最大绝对变化约3.00%。本次提交只同步已完成实验，不启动任何GPU任务。

- [中文报告](report_zh.md)
- [均值、中位数及分组统计](efficiency_summary.csv)
- [240个逐题计时](efficiency_per_input.csv)
- [7680行真实逐层KV布局](kv_layout_per_layer.csv)
- [完整逐请求Token、缓存和计时记录](formal_requests.jsonl.gz)
- [冻结协议及源码指纹](protocol.json)
- [验证结果](validation.json)
- [官方工作点和适配差异](algorithm_equivalence_audit.json)
- [规范路径下的必要代码与来源](../../external_baselines/README_e2e32.md)

GitHub清单`github_publication_manifest.json`核验本次发布的文件；原始`evidence_manifest.json`保留完整工作盘备份的历史指纹，包含未上传的SQLite、准备阶段记录和重复日志。不要把这些有意省略的文件误判为测量丢失：正式逐题记录已完整上传，完整可恢复Checkpoint仍在服务器和经过SHA256验证的Windows备份中。

原报告末尾“未修改远端GitHub”描述测量完成时的状态。本次独立分支提交是用户随后明确授权的同步，不改变任何已冻结算法、协议、预测或测量数值。

配置限制：SparseVLM实际层平均206.5；V2Drop沿用历史工作点；DUET使用C+all（平均191.75）；ApET平均192；FastV使用K3静态KV。效率行可用于注明这些工作点的同机效率表，不能直接与不匹配的公开质量分数拼接。

服务器完整结果：`/home/featurize/work/ATE/vlm_research/results/unified_e2e32_4090/`。
Windows完整结果：`D:\ATE-dataset-backups-3090\ATE-results\unified-e2e32-4090`。
Windows新增官方完整源码归档：同级`unified-e2e32-4090-source-archives`目录。
