# ATE：视觉语言模型 Attention 与视觉信息流研究

当前完成第一阶段：在单张 RTX 4090 上验证 LLaVA 1.5 的完整 FP16 图文推理，并记录后续分析所需的模型结构。第二阶段尚未开始。

工程位于 [`vlm_research/`](vlm_research/)。环境安装、权重下载和常见问题见 [工程说明](vlm_research/README.md)，真实耗时、显存和回答见 [第一阶段实测报告](vlm_research/results/phase1_report.md)。

## 已验证基线

| 项目 | 结果 |
| --- | --- |
| 模型 | `llava-hf/llava-1.5-7b-hf`，约 7.063B 参数 |
| 环境 | Python 3.11.18、PyTorch 2.6.0+cu124、Transformers 4.49.0 |
| 推理 | 单卡 FP16、eager Attention、greedy decoding |
| GPU | NVIDIA RTX 4090，24GB |
| 图像描述生成耗时 | 1.262 秒；含预处理 1.274 秒，不含加载 |
| 峰值活跃张量显存 | 13.589 GiB |
| 重复推理 | 三次输出 token 一致，显存稳定，无 NaN/Inf 或 OOM |
| 视觉 token / 本次描述问题的总输入 token | 576 / 594 |

模型权重与 Conda 环境保存在本机，仓库提供固定版本、下载校验脚本及实验结果。新机器须先按工程说明创建环境并下载权重；`configs/model.json` 中的本地路径由下载脚本重新生成。

## 运行

完成安装和模型下载后：

```bash
conda activate vlm_research
cd vlm_research
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --image ./test_images/example.jpg \
  --question "What is in this image?"
CUDA_VISIBLE_DEVICES=0 python inspect_model.py
CUDA_VISIBLE_DEVICES=0 python validate.py
```

测试图片来源见 [SOURCE.md](vlm_research/test_images/SOURCE.md)。阶段一只验证推理流程，不把单张图片回答作为能力评测。没有实现 Attention Hook、Token Pruning 或训练；逐层 Attention 提取等待用户确认后再开展。
