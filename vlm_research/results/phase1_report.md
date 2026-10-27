# 第一阶段实测报告

日期：2026-10-27（Asia/Shanghai）。第一阶段完成，未开展第二阶段。

## 环境与执行边界

- Conda 环境：`vlm_research`，路径 `/home/zsy/.conda/envs/vlm_research`。
- Python 3.11.18；PyTorch 2.6.0+cu124；torchvision 0.21.0+cu124；Transformers 4.49.0；CUDA 运行库 12.4；cuDNN 9.1.0。
- NVIDIA Driver 595.99.02；驱动最高支持 CUDA 13.2，可运行本项目 CUDA 12.4 wheel。无需系统 nvcc。
- 物理 GPU 0：RTX 4090。进程只见一张 GPU，FP16 CUDA 矩阵乘法测试成功。物理 GPU 1 的其他任务未被操作。
- 初始磁盘剩余约 1.6 TB，足够容纳权重、环境和缓存。
- `pip check` 成功，完整版本快照见 `requirements-lock.txt`；锁文件的离线安装 dry-run 通过。
- 环境问题已处理：下载命令绕过故障的本地代理；Conda 使用 classic solver 和 current_repodata；CUDA wheel 与普通依赖分开下载。系统代理和 base 环境未修改。
- HF 主站解析失败，使用 hf-mirror 传输同一官方转换仓库的固定 revision，并检查三个分片的 LFS SHA256。最终模型测试显式设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，不依赖运行时网络。

## 模型与加载

- 官方模型：`llava-hf/llava-1.5-7b-hf`，LLaVA 1.5，固定 revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。
- 完整模型参数：7,063,427,072（约 7.063B）；其中语言模型 6,738,939,904、视觉编码器 303,507,456、投影器 20,979,712。
- 所有参数均为 `torch.float16`，全部位于 `cuda:0`，device map 为 `{"": "cuda:0"}`。未量化、未 CPU offload。
- `eval()` + `inference_mode()`；视觉编码器和语言模型均使用 `eager` Attention。
- 实测加载耗时：4.159305 秒，不计入下表的生成耗时。
- 加载后 PyTorch 活跃张量显存：13473.648 MiB（13.158 GiB）。
- 加载后驱动侧本进程显存采样：13,938 MiB（13.611 GiB），包含 CUDA 上下文等额外开销。
- 验收中的最大 PyTorch 活跃张量显存峰值：13915.152 MiB（13.589 GiB）；分配器预留峰值 13,984 MiB（13.656 GiB）。
- 推理期间驱动侧本进程显存采样 14,476 MiB（14.137 GiB），这是采样值，不是连续测量的驱动显存峰值。

## A/B/C 结果

统一设置：本地单张 RGB 图片、图像处理输出 336×336、batch=1、max_new_tokens=64、do_sample=False、use_cache=True。

计时前后执行 CUDA synchronize。生成耗时仅包含 generate；预处理与生成包含读取图片、processor 和输入传输，不包含模型加载、解码及结果写盘。token 数包含生成的 EOS。

| 测试 | 新增 token | generate 秒 | 预处理与生成秒 | 峰值 allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| A_description | 51 | 1.261974 | 1.274135 | 13915.102 |
| B_detail | 7 | 0.195261 | 0.208296 | 13915.152 |
| C_repeat_1 | 51 | 1.022278 | 1.036054 | 13914.496 |
| C_repeat_2 | 51 | 1.022250 | 1.034322 | 13914.496 |
| C_repeat_3 | 51 | 1.025028 | 1.037096 | 13914.496 |

测试 A 问题：`What is in this image?`

> The image features a city street with a stop sign, a lion statue, and a red building. There is a black car driving down the street, and a person is walking nearby. The scene also includes a tree and a fire hydrant.

测试 B 问题：`What color is the stop sign?`

> The stop sign is red.

人工核对：A 对街景、STOP 标志、石狮、建筑、黑色汽车和行人的描述与图片相关，英文连贯；末尾 fire hydrant 在图片中无法明确确认，可能是模型幻觉。B 的红色属性与图片一致。这是功能验证，不是模型视觉能力评测。

C 在同一进程、同一模型实例中连续运行三次，没有在调用之间清空缓存。三个输出的 token ID 完全一致；每次清理输出后的 allocated 均为 13505.648 MiB，reserved 均为 13,984 MiB，未观察到持续增长。首次推理后 allocated 比刚加载时增加约 32 MiB，随后重复调用保持稳定；本阶段未追踪这部分分配来源。本短测不证明长期运行绝无泄漏。

最终 A/B/C 没有 CUDA OOM 或异常；每步生成词表分数均为有限值（无 NaN/Inf）。未通过修改非法分数来掩盖问题。

独立 CLI 也已执行成功：inference.py 生成耗时 1.241676 秒，预处理与生成 1.262031 秒。inspect_model.py 单独执行成功。原始数据见 validation.json、inference.json、model_structure.json 和 model_structure_cli.json。

## 真实模型结构与输入

| 项目 | 实测/已加载配置 |
| --- | --- |
| 模型类 | LlavaForConditionalGeneration |
| checkpoint 原始 Transformers 版本 | 4.36.0.dev0 |
| 运行时 Transformers 版本 | 4.49.0 |
| Vision Encoder | CLIPVisionModel，CLIP ViT-L/14，336px，24 层 |
| Language Model | LlamaForCausalLM，lmsys/vicuna-7b-v1.5 |
| LLM Transformer 层数 | 32 |
| Hidden Size | 4096 |
| Attention Heads | 32 |
| KV Heads | 32 |
| Vision Feature Dimension，投影前/后 | 1024 / 4096 |
| 视觉特征选择 | 第 -2 层，default，移除 CLS |
| 单图视觉 token | 576，24×24 |
| A/C 实际 LLM 输入 token | 594 = 576 视觉 + 18 文本/特殊 token |
| B 实际 LLM 输入 token | 595 = 576 视觉 + 19 文本/特殊 token |
| LLM 最大位置长度 | 4096 |

后续定位可参考以下已验证的模块路径，本阶段没有注册 Hook：

```text
vision_tower.vision_model.encoder.layers
multi_modal_projector
language_model.model.layers
```

## 阶段结论

已满足进入第二阶段的推理与结构前提。FP16 模型完整驻留单张 RTX 4090、单图推理正常、结构与视觉 token 数已确认、确定性重复检查通过。测试进程已退出，GPU 0 恢复空闲。等待用户确认推理结果后再开展逐层 Attention 提取；不自动开始第二阶段。
