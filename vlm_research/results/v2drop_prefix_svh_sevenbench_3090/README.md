# V²Drop-prefix + SVH：RTX 3090 结果归档

本分支同步已经完成的实验，不新增 GPU 运行、不修改 main。保留模型固定 Revision
`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`，FP16 / Eager，PyTorch 2.6.0+cu124，Transformers 4.49.0。

| Method | KV (MiB) ↓ | TPOT (ms) ↓ | GQA (%) ↑ | SQA (%) ↑ | TextVQA (%) ↑ | POPE (%) ↑ | MME ↑ | MMB (%) ↑ | OCRBench /1000 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V²Drop-prefix + SVH | 97.5625 | 62.8795 | 56.1775 | 67.7739 | 49.8100 | 85.3648 | 1780.3954 | 62.3299 | 87 |

质量评测 36,256/36,256 条，新增 33,682 条，复用 2,574 条；MMB 为 4,377 个问题对应
1,176 个完整 Circular 组。TextVQA 是十答案 VQA Soft Accuracy；MME 是 Accuracy+Accuracy+
按任务累加的 Total。OCRBench 官方 v1 为 87/1000，确实偏低，未删去 126 条空输出或 14 条截断，
也未为改善分数调参。GQA 使用官方对齐协议，不和旧 USER-only/crop 协议混算差值。

KV/TPOT 是冻结 GQA15+TextVQA15 输入、固定 128 个输出 Token 的测量均值；
主 TPOT 为墙钟 `(E2E−TTFT)/(L−1)`。32 Token 与 CUDA 次级计时同样保留。
不能把这一 30 输入测量当作七个完整数据集的全请求平均延迟。

## 内容与来源

- [七项完整报告](report_zh.md)、[正式分数](score_summary.json)、[效率原始计时](raw_timings.jsonl.gz)。
- [固定 200 题报告](../v2drop_prefix_svh_accuracy_3090/report_zh.md)：四方法预测与配对比较均保留。
- [MME 全量报告](../v2drop_prefix_svh_mme_full_3090/report_zh.md)：与同协议历史 SVH 的独立计分复核。
- 原始物理记录无损压缩：
  [四图严格门禁](../v2drop_prefix_svh_accuracy_3090/physical_smoke.json.gz)、
  [MME 双题 Smoke](../v2drop_prefix_svh_mme_full_3090/MME_integration_smoke.json.gz)、
  [九条新输入检查](integration_smoke.json.gz)。报告中的 `.json` 文件名指服务器/Windows 原始文件。
  本轮继承的四图记录与前述四图文件字节完全一致，不重复上传。

本融合只使用 V²Drop 的 L3/L17 选择（576→288→144），L20 前 SVH 退出视觉 Hidden，
没有 L22 剪枝；P−1/Decode 无原始视觉 KV，答案 Attention 在 L2–L5 物理跳算。
四图严格数值门禁 PASS，阈值没有放宽；新输入结构检查 PASS。

研究执行基线为 `e281754ef17b3accc09d6938b5b9f020048b2aec`，新融合文件使用 SHA256 冻结。
本证据分支以已经公开的 V²Drop 快速复现 `0c94eaa451a6000e2a7e0bec5570278a1de12292` 为父提交，
复用其官方源码、许可证和 HF 适配器。18 个核心/运行依赖逐字节一致；另外 8 个历史 GQA 辅助脚本
从 `57e37ce024b1b2adf9fa24fc24aabec32a1fed08` 原样恢复，26 个冻结 SHA256 全部匹配。
没有修改冻结算法核心。新增代码是运行时的实际版本；旧计时入口、受控断点及计时协议补充均保留。
重跑仍需要恢复离线完整输入清单、图片及历史来源文件，不能只凭轻量发布目录运行。

## 完整离线备份

完整预测、数据库、大型输入清单和原始未加同步说明的报告均保留于 Windows：

```text
D:\ATE-dataset-backups-3090\ATE-results\v2drop-prefix-svh-3090-20261109
D:\ATE-dataset-backups-3090\ATE-results\v2drop-prefix-svh-mme-full-3090-20261109
D:\ATE-dataset-backups-3090\ATE-results\v2drop-prefix-svh-sevenbench-3090-20261109
```

本次发布前，按三份 `artifact_manifest.json` 再次核对全部 24+15+39 个文件的大小与 SHA256。
各 `windows_backup_receipt.json` 是此前真实 SFTP 回读校验的回执。
`artifact_manifest.json` 描述完整离线归档，**不是所有条目均上传 GitHub**；本次实际发布范围、
变换（报告增加同步说明、Smoke 无损 gzip）及源文件 SHA256 见 `publication_manifest.json`。
完整七项逐题预测和数据库保留在上述本机目录；模型、数据集与图片不上传 GitHub。

实验原报告/元数据中的“未 Push”是实验结束时的历史状态，本次获用户明确授权后发布；
保留原历史记录，未伪造当时已同步。实际 Commit 与逐文件 GitHub HTTP SHA256 校验结果
保存在本机随后生成的 `github_sync_receipt.json` 中。

模型进程已结束。服务器能否关机还须以本次同步完成后的实时进程检查为准；本提交不执行关机。
