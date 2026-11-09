# InternVL3 Figure 3：GitHub 同步与完整证据恢复

本分支保存同一张 RTX 3090、同一批 3000 题的完整实验数据：原预注册 31 条件共 93,000 次生成，以及用户事后追加的 L20–L27 OFF 3000 次生成。后者为探索性分析，不能解释为预注册结果。

- 原实验报告：[report_zh.md](../internvl3_visual_writing_figure3_3000/report_zh.md)
- **最新扩展报告与重绘图**：[report_zh.md](../internvl3_visual_writing_l20_l27_3000/report_zh.md)、[三模型 PDF](../internvl3_visual_writing_l20_l27_3000/figure3_three_models.pdf)
- All-Writer：2710/3000，90.3333%；L17–L27 OFF：2654/3000，88.4667%；L20–L27 OFF：2700/3000，90.0000%。All-Writer 与 Native Full 的干预定义不同。
- 全部原始逐题文本、Gold、分数和 Token IDs 均在两个结果目录的 `per_question.csv` 中。全部 6000 个逐题检查点（包含原始 Prompt、输入绑定、逐步 Logits SHA、Cache 和 Mask 审计）也完整上传，采用流式 tar.gz 打包。

## 上传形式

`archive_manifest.json` 列出 4 个完整归档、目标目录、大小和 SHA256。原始检查点、历史参考文件、CPU 输入绑定及一份原始 pyc 校验对象采用压缩形式上传；没有删除、裁剪或改写原始内容。逐个归档成员已对照原始 SHA256SUMS 核验。解压后可以恢复两个原结果目录全部 6393 个冻结清单条目。

`publication_manifest.json` 与本目录 `SHA256SUMS` 覆盖本次发布的源文件和证据。原结果目录中的原 SHA256SUMS、协议、历史 `no_push` 验证状态均按完成时的原字节保留；后者描述生成验收时状态，并非本次同步状态。

模型权重、图像数据集、虚拟环境、系统缓存和所有密钥未上传；模型固定 Revision 与文件 SHA256、图像 SHA256 与来源均已记录。源码、协议、逐题预测和原始检查点已上传。恢复研究环境时仍需自行提供冻结的模型和 GQA 图片。

## 验证与恢复

在克隆本分支的仓库根目录执行（仅 Python 标准库，不加载模型或调用 GPU）：

```bash
# 验证所有已发布文件和全部压缩包成员，不解压
python vlm_research/results/internvl3_visual_writing_github_sync_20261110/restore_evidence.py --verify-only

# 如需重新运行 CPU 统计，先恢复缺失的检查点、参考记录和输入绑定
python vlm_research/results/internvl3_visual_writing_github_sync_20261110/restore_evidence.py
```

恢复器逐文件流式处理，仅写入缺失文件；已有文件必须匹配冻结 SHA256，否则停止，不覆盖。解压约需额外 0.5 GB 磁盘空间，内存不随整个数据集大小增长。原实验运行脚本仍保留实际服务器绝对路径；在其他机器复用时需配置相应模型、数据和结果路径，不能直接假定原路径存在。

所有打包、核验及 GitHub 上传均在服务器执行，未将数据复制到 Windows 本机。本次同步未运行任何 GPU 推理、新实验、训练或正式评测；持久化工作盘原件与模型/数据保留，未关闭服务器。仅上传独立分支 `codex/internvl3-visual-writing-figure3-3000`，不合并 main、不创建 PR。
