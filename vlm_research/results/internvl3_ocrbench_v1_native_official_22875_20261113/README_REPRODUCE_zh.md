# CPU复核与恢复

状态为BLOCKED，禁止将3题Smoke解释为完整Native得分。本目录不含权重或930原图。

1. 从ATE固定提交 `93ff60f839644fabf81f202d699bbc0638484f42` 取回目录 `vlm_research/results/internvl3_ocrbench_v1_protocol_audit_20261113/`。本次新证据目录作为独立目录。
2. 用标准库Python运行（无需GPU或Torch）：

```bash
python -B offline_audit.py --results ./results --audit-dir /path/to/internvl3_ocrbench_v1_protocol_audit_20261113
```

检查1000题顺序与Gold；仅对3个真实预测执行原官方OCRBench循环、原固定VLMEvalKit scorer与独立规范化逻辑。将重新生成 `independent_audit.json`、`independent_scoring.csv`、`coverage.csv`。如需保持证据只读，先复制结果目录再运行。

文件完整性：

```bash
sha256sum -c SHA256SUMS
```

模型恢复须使用 `OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9`，比对 `model_integrity.json`；图片使用原审计930张SHA表。不要重复下载已有完整权重或更换图片。

GPU入口为 `native_official.py`，默认根目录是执行器所在目录，旁边需有 `audit/`、`control/model_integrity.json`。路径常量明确记录于源码；迁移时按实际本地盘挂载准备同等路径。`prepare`仅CPU，生成新的、隔离的协议和输入清单；不要覆盖本次结果，程序会拒绝已有protocol。`run`是GPU入口，必须先具备足够显存并重新通过固定4题Smoke。本次3090在25块题0743失败，不应重复无谓GPU重试。

官方函数通过未改写AST执行；保存了函数AST哈希、模型源码SHA、实际环境冻结清单、模型完整性回执。`native_gpu.log` 约17KB，保留原OOM栈。未上传模型权重/图片。
