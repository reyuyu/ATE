# 新旧评测一致性复核

复核时间：2026-11-10 05:33，Asia/Shanghai。

## 结论

针对仓库旧 SVH-R 的冻结实验，五项已接受数据的样本与声明协议通过复核。
这不等于已经完成同硬件历史结果复现，也不等于与所有外部论文方法完全同条件。

| 数据集 | 数量 | 已核验的绑定 | 生成上限 |
|---|---:|---|---:|
| GQA testdev_balanced | 12,578 | 官方题号、Question、Gold、Image ID、完整 prepared user text | 32 |
| ScienceQA-IMG Test | 2,017 | 与原历史 Manifest 全字段逐题一致，含选项和原 Prompt | 1024 |
| TextVQA Validation Paper-OCR | 5,000 | 与原历史 Manifest 全字段逐题一致，不是 image-only 版本 | 128 |
| POPE Paper | 8,910 | 与原历史 Manifest 全字段逐题一致，2910/3000/3000 | 128 |
| MMBench-EN | 4,377 | 与原历史 Manifest 全字段逐题一致，保留 Circular 分组 | 1024 |

四项 Paper 数据的原始压缩 Manifest SHA256 与参考提交 `0414e553` 中的协议记录严格相同。
GQA 原始 prepared JSONL 与官方 Gold 的 SHA256 匹配 `71a5762` 的冻结协议；未混用旧 USER/crop 数据。
本次重算 7,213 个图片路径的文件 SHA256，均与冻结记录一致。

模型 Revision、FP16/eager、batch=1、Greedy、H1、Torch 和 Transformers 版本均匹配。
共享预处理和评分源码 SHA 一致：完整 Vicuna-v1 Prompt、RGB、CLIP mean 方形补边、336 像素处理。
TextVQA 使用同一十答案软评分器；MMBench 按冻结 Circular 规则最终聚合，不能拿行准确率直接对比 Circular 分数。
唯一预声明的算法变化仍为回看 L8-L23、相应持久视觉 KV 和 L24 前物理删行。
没有修改正在执行的推理源码、条件、输入或评分器。

## 尚不能声称严格配对的原因

1. GQA 旧正式对照实际运行在 RTX4090，本轮为 RTX3090。相同 FP16 软件协议不保证跨架构逐位相同。
2. 历史完整预测和输入 Tensor 逐题哈希尚未恢复。本轮的旧实现 Smoke 是同输入的实现检查，不能替代完整历史预测绑定。
3. 因此 58.3320% 与 59.2543% 的 -0.9222pp 是跨运行的汇总差值；不能断言全部由 Middle16 改动导致，更不能据此报告配对显著性。
4. 配对原对变错、原错变对和差值 CI 仍缺失，未制造结果。严格归因需要恢复绑定历史记录，并对跨硬件差异补充同机控制。
5. MME 的指定图片版本和 OCRBench 官方原图未验收，二者仍 BLOCKED；不能将其计作协议已一致。

机器可读结果见 `protocol_recheck.json`，CPU 复现入口为 `protocol_recheck.py`。
该检查没有加载模型、初始化 CUDA 或新增 GPU 生成。
