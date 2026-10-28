# ATE：视觉语言模型 Attention 与视觉信息流研究

当前已完成 Phase1～Phase3.5：单卡FP16基线、逐层Attention/Token定位、全局及定向Mask、240图Fixed-K Physical Exit、ROI/Sink诊断与离线边界分析。历史报告及统计可直接复用。

**最新单案例Sink去偏/ROI增强已完成**：31条件真实7B、108项CPU回归和396审计通过。中层整体去偏变差，ROI增强微小改善仍错，深层Sink抑制未复现K24转正；仅L2～L4指定非ROI Sink抑制转正。单例FP16观察，[报告及全部图表](vlm_research/results/attention_sink_steering_case_15133706/README.md)。

**15133706单案例Attention继承已完成**：101项CPU回归及六条件真实7B验证通过。L23 λ.5/1使right Margin转正（+.02344/+.08594），L24未转正；单案例FP16观察，不作普遍机制结论。[完成报告](vlm_research/results/attention_inheritance_15133706/completed_20261028_2245/report.md)。

**新服务器接手从这里开始**：[交接文档](vlm_research/handoff/README.md) · [完整数据集/模型下载](vlm_research/handoff/DATASET_SETUP.md) · [Codex交接提示词](vlm_research/handoff/CODEX_PROMPT.md)。说明如何恢复全部GQA图片、校验252张冻结图、保留240图/48主图统计协议，以及读取已完成单案、按后续授权继续统计工作。

工程位于 [`vlm_research/`](vlm_research/)。环境安装、权重下载和常见问题见 [工程说明](vlm_research/README.md)，真实耗时、显存和回答见 [第一阶段实测报告](vlm_research/results/phase1_report.md)。

第二阶段结果见 [实验报告](vlm_research/results/phase2_attention/phase2_report.md)：32 层 Attention、逐 Head 统计、24×24 Patch 热图与两个问题的比较。原有环境、模型配置与第一阶段脚本未改变。

Phase 2.1 结果见 [位置分析报告](vlm_research/results/phase2_1_attention/phase2_1_report.md)：真实 Token 对齐、答案目标与预测 Query 的位置偏移、独立 BOS 统计、语义答案空间图与 Phase 2 数值一致性排查。第一、第二阶段既有结果保留。

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

测试图片来源见 [SOURCE.md](vlm_research/test_images/SOURCE.md)。阶段一只验证推理流程，不把单张图片回答作为能力评测。阶段二只做描述性 Attention 观察，没有实现 Attention Hook、Knockout、Token Pruning 或训练；不据此宣称视觉信息已被文本吸收或视觉 Token 可以安全删除。
