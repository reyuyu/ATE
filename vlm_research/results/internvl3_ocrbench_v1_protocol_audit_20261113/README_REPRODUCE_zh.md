# InternVL3-8B OCRBench v1 协议审计：归档与新服务器复核

本分支只归档既有 CPU 协议审计，没有新增模型加载、GPU 推理、OCRBench 生成或计分规则变更。原审计报告、原 SHA256SUMS、历史预测及失败证据均按字节保留。原文件中的 github_push=NOT_RUN、未推送等文字是**审计完成时**的状态；本次发布记录与其分开，不改写原证据。

审计结论：10 PASS / 4 FAIL / 2 UNVERIFIED。Native 751/1000 可独立复算，但单图块与自定义问题后缀不符合所核对的公开 VLMEvalKit OCRBench 默认配置，不能声称等同于作者 880 的原始运行。Late9 仅完成 3/1000，另 997 题当时因原图 SHA 不可匹配而阻塞；本审计确认原图已恢复，但没有补跑。

## 必需文件与用途

| 文件/目录 | 用途 |
|---|---|
| report_zh.md、audit_table.csv/json、minimal_fix_plan.md | 原报告、逐项 PASS/FAIL/UNVERIFIED 与最小修复方案 |
| official_protocol_reference.json、references.json | 公开配置摘要、官方源码固定 Commit 与来源链接 |
| source_manifest.json、model_reference/ | 模型 Revision、配置、原生对话模板与原模型文件校验回执；无权重 |
| question_image_sha256.csv | 1,000 题的 ID、原始相对图片路径、SHA256 |
| unique_images_sha256.csv | 930 张唯一原图的 SHA256、字节数和尺寸 |
| cpu/ | 原数据、评分、预处理 CPU Smoke、动态切块计划及来源完整性核验 |
| official/ | 审计使用的固定官方标注、Prompt/切块/Scorer 源码快照 |
| historical/ | 原协议、逐题预测/Token IDs、输入清单、历史源码与 Late9 的缺图/失败证据 |
| code/reproduce_cpu.py | 新服务器可直接执行的轻量 CPU 复核入口，仅 Python 标准库 |
| code/audit_cpu.py、audit_final_checks.py、publish_audit.py | 原审计执行/整理脚本，保留当时的绝对路径；不应在新服务器直接执行 |
| new_server_files.json、publication_manifest.json、publication_validation.json | 迁移清单、发布文件/原证据绑定及归档检查 |
| SHA256SUMS、evidence_manifest.json | 原始审计完整性清单，未经改写 |
| PUBLICATION_SHA256SUMS | 本次发布中全部原文件与新增复现材料的 SHA256 清单（不包含自身） |

## 新服务器 CPU 复核

将本结果目录从本分支取到新服务器。不需要下载整个仓库历史；已有 ATE 仓库可以只检出本目录。完整迁移清单见 new_server_files.json。

在仓库根目录：

~~~bash
AUDIT="$PWD/vlm_research/results/internvl3_ocrbench_v1_protocol_audit_20261113"
(cd "$AUDIT" && sha256sum -c PUBLICATION_SHA256SUMS)
python3 "$AUDIT/code/reproduce_cpu.py" \
  --audit-dir "$AUDIT" \
  --output-dir "/tmp/internvl3-ocrbench-audit-recheck-$(date +%Y%m%d-%H%M%S)"
~~~

脚本验证发布文件、原审计 89 项 SHA、1,000 题标注映射与 930 图 SHA 清单；实际执行固定官方评分循环重算 Native 751、历史 Physical-D 32，以及 Late9 三题结果，并验证 997 条阻塞记录恰好对应历史缺图集合。它不导入 Torch、不加载模型，不覆盖任何原文件。

**新服务器原图实物核验**需额外准备已有备份中的原始 OCRBench.json 与 OCRBench_Images/，然后使用新的输出目录：

~~~bash
python3 "$AUDIT/code/reproduce_cpu.py" \
  --audit-dir "$AUDIT" \
  --output-dir "/tmp/internvl3-ocrbench-audit-images-$(date +%Y%m%d-%H%M%S)" \
  --annotation "/absolute/path/to/OCRBench.json" \
  --image-root "/absolute/path/to/OCRBench_Images"
~~~

未提供图像路径时，脚本明确记为 fresh_original_images=NOT_RUN；不得把归档中旧 SHA 检查当成新服务器图片已验证。指定路径时按 930 张唯一原图逐一核对字节数与 SHA，不改图、不重编码。不是全盘哈希。

本次上传**不含原图或模型权重**。如另需复核历史 tokenizer/像素 Tensor，要单独准备冻结模型 OpenGVLab/InternVL3-8B，Revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9，以及原 Python 3.11.8 / Torch 2.6.0+cu124 / Transformers 4.49.0 CPU 环境；原 CPU Smoke 结果和执行源码均已归档。轻量脚本并未假称重跑 tokenizer、动态预处理或 GPU Smoke。

## 保留与边界

- 原始服务器目录和历史实验结果不修改；本次用独立发布目录组织新增材料。
- 不将 Late9 3 题外推为全量分数，不调整 Scorer，不承诺修复配置后达到 880。
- 公共官方 config.py 中含其上游仓库原有的示例 API 地址，属于公开源码快照；不含本服务器 SSH 地址、密码、密钥或 Token。
- 模型参数仅保留 Revision 与小型配置/校验回执；原图、权重、系统缓存及 SSH 文件均不在本目录。

