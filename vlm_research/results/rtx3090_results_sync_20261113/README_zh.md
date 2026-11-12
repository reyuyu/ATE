# ATE RTX 3090 历史结果与完整证据索引

用户于2026-11-13明确授权“把之前结果都上传到github”。本次只归档既有实验与诊断文件，没有重跑实验、修改冻结源码、重新评分或改动原始统计。

## 当前四配置效率补测

[中文报告](../svh_component_ablation_e2e32_3090/report_zh.md) · [汇总表](../svh_component_ablation_e2e32_3090/summary.csv) · [逐请求数据](../svh_component_ablation_e2e32_3090/per_request.csv) · [两轮独立统计](../svh_component_ablation_e2e32_3090/session_summary.csv)

两个独立Session，每轮4配置×30题，240/240正式请求；输入为冻结GQA15+TextVQA Paper-OCR15，固定实际32 Token。执行正确性通过，保留性能警告，状态为COMPLETE_WITH_PERFORMANCE_WARNINGS。准确率沿用已验证SQA-IMG/POPE质量结果，本轮未重跑质量。

| 配置 | KV@32 MiB | TTFT ms | TPOT ms/token | E2E@32 ms | 同轮Native E2E降低 |
|---|---:|---:|---:|---:|---:|
| Native Full | 351.50 | 237.72 | 64.558 | 2239.02 | — |
| SVH w/o Selective Transfer | 55.56 | 239.97 | 59.342 | 2079.57 | 7.12% |
| SVH w/o Attention Bypass | 63.50 | 182.50 | 64.100 | 2169.61 | 3.10% |
| SVH Full | 55.56 | 184.81 | 59.165 | 2018.92 | 9.83% |

数值为两轮均值的平均；标准差、探索性配对区间与警告见原报告和CSV。24项Native校准均未超过2%，仍有3条正式请求内波动警告，未将结果宣称为无性能警告的稳定PASS。

## 本分支归档的十组结果

| 实验/诊断 | 报告 | 状态说明 |
|---|---|---|
| Native E2E@32初次复测 | [报告](../native_e2e32_3090_recheck/report_zh.md) | 原15条正式记录保留，稳定性未通过；不作为完整基线 |
| Native只读稳定性审计 | [报告](../native_e2e32_3090_stability_audit_20261110/report_zh.md) | 既有日志分析及修订协议建议 |
| Native受控根因诊断 | [报告](../native_e2e32_3090_rootcause_diagnostic/report_zh.md) | 84次诊断生成；物理根因尚未唯一确定；Profiler请求隔离 |
| 八方法首次PARTIAL | [报告](../unified_e2e32_eight_methods_3090/report_zh.md) | 原31条记录、失败Token与日志保留 |
| 八方法Session2准入失败 | [报告](../unified_e2e32_eight_methods_3090_session2/report_zh.md) | 无正式请求；原预热、准入与失败证据保留 |
| 八方法repeatability初版 | [报告](../unified_e2e32_eight_methods_3090_repeatability/comparison/report_zh.md) | 原151条记录及历史Token验收失败保留 |
| 八方法repeatability_v2 | [报告](../unified_e2e32_eight_methods_3090_repeatability_v2/comparison/report_zh.md) | 480/480正式请求；COMPLETE_WITH_PERFORMANCE_WARNINGS |
| 18条件统一效率 | [报告](../rtx3090_unified_efficiency_18conditions/final_report.md) | 三Backbone共1080/1080正式请求；保留6项性能告警 |
| Section5.5质量消融及生成长度 | [报告](../svh_ablation_efficiency_3090/report.md) | 质量逐题预测及长度实验720条正式记录；保留原性能警告 |
| 四配置组件消融E2E@32 | [报告](../svh_component_ablation_e2e32_3090/report_zh.md) | 两轮240/240正式请求；COMPLETE_WITH_PERFORMANCE_WARNINGS |

各目录对应不同时间、GPU UUID和协议；禁止把这些绝对延迟混合成单一同机实验，禁止跨GPU计算算法Speedup。旧PARTIAL和失败轮次不与完整轮次合并、不重标PASS。

## 原始证据与压缩恢复

本次同步1323个源文件的完整内容：报告、逐题CSV/预测JSONL、真实KV逐层布局、SQLite Checkpoint、预热/Smoke/校准记录、GPU/CPU遥测、原始日志、Profiler轨迹、源码快照、协议及SHA清单。较大的JSONL、SQLite和超过GitHub单文件限制的Profiler JSON采用无损gzip，文件名附加`.gz`；其他原文件逐字节保留。

[sync_manifest.json](sync_manifest.json)逐个记录源文件SHA256、GitHub文件SHA256、源路径与压缩映射。将gzip解压到新的恢复目录即可得到源字节，避免覆盖历史目录。SQLite的瞬态共享内存、空WAL，Python字节码，以及可重建的CUDA编译工具依赖未归档，逐项原因已记录于清单。模型权重和数据集图片不属于结果归档。

旧artifact/source清单继续保留原文；若条目指向未压缩原文件，请先按本次映射恢复，再核对原SHA。本次没有修改它们以伪装原路径全部直接存在。之前`GITHUB_SYNC.md`中“服务器独有”的叙述描述上一次轻量同步时点，本次已补齐其原始证据，以本索引和sync_manifest为准。原protocol/report中的不自动Push条款是实验阶段限制，后续用户已单独授权本次发布；原文不改。

## 已有GitHub历史参考

- [原4090八方法冻结实验](https://github.com/reyuyu/ATE/tree/5ee2a0dbe78df336229b55fed14a4a6d0d880bed/vlm_research/results/unified_e2e32_4090)
- [原4090 Retain128实验分支](https://github.com/reyuyu/ATE/tree/codex/unified-retain128-e2e32-4090)
- [八方法两轮首次发布](https://github.com/reyuyu/ATE/commit/bc3e34d393110981444e6d2ffc07d57d30c9ad74)
- [Section5.5首次轻量发布](https://github.com/reyuyu/ATE/commit/94234ab402091af965632903d1bafc8a09e7aa8f)

此次独立分支为`codex/rtx3090-results-complete-20261113`，基于Section5.5已发布提交94234ab。未修改main、未创建PR、未改写服务器历史结果，未启动GPU工作。
