# InternVL3-8B 单 GPU Layer Analysis Smoke

日期：2026-11-01。状态：**PASS，限本报告所列输入和 Smoke 范围**。

## 十项直接回答

1. **完整下载：是。** 官方 `OpenGVLab/InternVL3-8B`，4 个权重分片、685 个 BF16 Tensor；Safetensors 可打开，权重索引与实际 Tensor 完全匹配。实际模型目录占用 15,903,244,288 字节。没有提交权重到 Git。
2. **固定 HF Revision：`853e3a797a661694b1b8ece0cb72dc2b23e3dac9`。** 文件字节数和 SHA256 见 `model_files_sha256.json`，四个分片另列于下文。官方直连不可达，Revision 元数据通过 HF 镜像取得，不能声称独立直连核验了官方当前 HEAD。
3. **GPU1 单卡 BF16：通过。** `CUDA_VISIBLE_DEVICES=1`，进程内 `cuda:0` 对应 `GPU-b08489af-0a8a-eb48-55ba-8346733af417`；全部模型参数为 BF16 并在单卡，没有量化、CPU Offload 或跨卡切分。语言与视觉模块均关闭 FlashAttention，使用 Eager。
4. **真实语言 Decoder：28 层，L0–L27。** Qwen2 实现的语言骨干，Hidden Size 3584，28 个 Query Head、4 个 KV Head，GQA 分组数 7。视觉编码器另有 24 层，不能混用语言层坐标。
5. **`o_proj` Hook：通过。** 对两个真实样本分别验证 L2、L27 的 P−1 输出置零；目标行严格为零，其他 Prefill Query 不变。No-op 首 Token 全词表 Logits 最大绝对差为 **0**。
6. **P−1 与视觉布局：可精确定位。** 使用实际展开的官方 Chat Prompt 和 Token IDs；单图块 256 个视觉 Token，双图块加缩略图共 768 个。四个输入的 P−1 分别为 319、332、322、836，未使用固定索引。
7. **来源 Mass：通过有限 Smoke。** 在一张图的 L0/L27 只提取当前 P−1 行，立即在 CPU 归约，未保存全序列或全层 Attention 矩阵。五类 Key 互斥且完备，采集前后原生 Logits 最大绝对差为 **0**。
8. **真实 GQA 生成：通过。** 三个不同任务的单图块样本和一个不同宽高比的动态图块样本均输出非空答案、正常 EOS，无截断。这里只验证运行，不给出模型 Accuracy 结论。
9. **显存峰值：** `nvidia-smi` 定时采样的模型进程峰值 **16,044 MiB（约 15.67 GiB）**；CUDA allocated 峰值 **15,485.93 MiB**，reserved 峰值 **15,728 MiB**。采样峰值不是连续硬件测量，可能遗漏很短的瞬时峰值。
10. **后续 120 题扫描：具备基本技术条件，但尚未运行。** 必须继续使用动态布局、逐层当前行归约、完整异常与显存检查。此 Smoke 不保证更多图块、长上下文、全部层矩阵留存或来源 Hard Mask 的可行性。

## 下载与模型绑定

本地绝对路径：

```text
/data/lk/ATE-resources/models/internvl3-8b/853e3a797a661694b1b8ece0cb72dc2b23e3dac9
```

首先尝试官方 HF 固定 Revision 的 `snapshot_download`，因网络不可达失败。小文件从 `hf-mirror.com` 的同一固定 HF Revision 获取，并核对 Git Blob/LFS 身份。权重从官方 ModelScope `OpenGVLab/InternVL3-8B` 的固定 Revision `0908e0d61e20f8794538f8a9efe134e4d1e0e8d4` 获取：四个分片的实际 SHA256 和大小均与上述 HF 固定 Revision 的 LFS 元数据精确一致。没有以未经核验的不同模型替代。

| 权重分片 | 字节数 | SHA256 |
| --- | ---: | --- |
| model-00001-of-00004.safetensors | 4991123960 | `7ea1f92eaae35cb927e7c7b0f87568ccc046a2446aa68066dfdccb7ebbe0c7f0` |
| model-00002-of-00004.safetensors | 4958443072 | `05f9f1bf63d946d2963bd68c9d1ee7f93bdc5fc05a9b39a3fb91734d5ef4362d` |
| model-00003-of-00004.safetensors | 4796984024 | `904aa0f2fbdbf504cd7bce1a8ec5633ee4e3e1769b48c5b3e08369a23875aaa1` |
| model-00004-of-00004.safetensors | 1142280864 | `50e048a88254db95e0aa397a4e4012a353eaf650a18d7b291e1522d16e83d389` |

自定义 Python 源码在启用 `trust_remote_code=True` 前静态审阅；执行的五个本地缓存模块逐一与固定 Snapshot 字节校验。模型只从本地路径加载，没有执行其他 Revision 的代码。`download_validation.json` 的 `remote_code_executed=false` 仅指下载校验阶段，后续推理确实执行了已审阅的固定本地代码。

## 环境与真实结构

独立环境：`/data/lk/深度学习/.envs/ate-internvl3`。Python 3.11、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、Transformers 4.49.0、accelerate 1.4.0、timm 0.9.16、einops 0.8.1。完整实际版本见 `requirements-actual.txt` 和 `environment.json`。环境安装复用本地已校验 Wheel，缺失依赖通过清华 PyPI 镜像下载。旧 LLaVA/Qwen 环境包清单及 Transformers 源码哈希均保持不变。

- 实际模型类：`InternVLChatModel`，总参数 7,944,373,760。
- 视觉路径：`vision_model.encoder.layers[i]`，24 层 InternVisionModel，304,012,288 参数；448×448，Patch 14。
- 语言路径：`language_model.model.layers[i]`，28 层，7,612,820,992 参数。每层 `self_attn.q_proj/k_proj/v_proj/o_proj`、`input_layernorm`、`post_attention_layernorm` 和 `mlp` 均经实例遍历确认，完整逐层路径见 `architecture_audit.json`。
- 图像 CLS 去除后执行 `pixel_shuffle` v2、下采样比例 0.5，再经 `mlp1`：LayerNorm(4096)、Linear(4096→3584)、GELU、Linear(3584→3584)。投影结果替换真实 `<IMG_CONTEXT>` 对应的输入 Embedding，发生在语言 L0 前。**本架构没有 Qwen3-VL 的 DeepStack 多层注入。**
- 使用官方 `internvl2_5` Chat Template。实际输入不包含配置中的 BOS 151643；首个真实 `<|im_start|>`（151644）仅命名为 Start Anchor，不命名为 BOS。
- 实际 Chat EOS 是 `<|im_end|>`（151645），与语言配置中的 EOS 151643 区分；官方 Chat 接口按对话分隔符设置停止条件。

## 四个真实生成样本

样本采用预先固定的输入哈希规则选择，没有按答案或模型表现筛选。图像 SHA256、原问题、最终 Prompt、Token IDs 与分区映射见 Manifest、`token_layout_audit.json` 和 `smoke_generations.json`。

| 任务 / Question ID | max_num | 实际图块数 | 视觉 Token | 输入长度 | P−1 | 原生回答 | EOS |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ChooseAttr / 1127952 | 1 | 1 | 256 | 320 | 319 | Blue | 是 |
| ChooseRel / 05785689 | 1 | 1 | 256 | 333 | 332 | right | 是 |
| QueryAttr / 02357466 | 1 | 1 | 256 | 323 | 322 | right | 是 |
| ChooseAttr / 11534918 | 2 | 3 | 768 | 837 | 836 | Green | 是 |

`max_num=2` 的第四例实际产生两个动态图块和一个整图缩略图，因此是三个视觉图块，不是两个。预处理直接复用固定官方教程中的函数，RGB、448 Bicubic、官方 ImageNet 归一化，没有为通过测试而缩小分辨率。

## 正确性和数值审计

10 项 CPU 测试通过；真实 GPU 测试通过。成功执行时的冻结源码 Commit 为 `07e8825ccd926c1e1188fe3dab8ee7ac4ce50436`，协议包含源码、模型与输入哈希；最终结果提交不改变该执行代码。

- 使用官方 `chat` 真实前向捕获的多模态 Embedding 与原生参数进行 Hook 重放。重放与原生首 Token Logits 精确一致，No-op 也精确一致。
- 两个样本分别只测 L2 与 L27，不做层扫描。目标 P−1 `o_proj` 输出严格为零，全部其他 Prefill Query 的投影/层输出保持精确一致；MLP 均执行，实际 BF16 残差加法正确，28 层独立新建的 DynamicCache 长度和结构正常。
- L2/L27 关闭后全词表 Logits 确实发生变化，证明 Hook 有效；两个样本的首 Top-1 未变不能证明 Attention 冗余。
- Mass 来自未改写的官方 Qwen2 Eager 计算，观察包装原样返回结果。Key 分为 Anchor、Image、Question、Template、Self，互斥完备；Question 通过实际文本 Offset 对齐，并核验 Fast/Slow Tokenizer 的 Token IDs 相同。
- L0/L27 各 Head 行和最大误差分别为 0.00185698、0.001953125，低于预先固定的 BF16 舍入界 `2**-8 + 2e-6 = 0.00390825`，没有运行后放宽阈值。
- 正常与异常路径均测试 Hook 清理；模型权重、输入、执行的自定义源码及第三方 Transformers 文件未修改；无 NaN、Inf、OOM。

**保留的首次失败：** 初次加载成功，但脚本将整个主函数置于 `inference_mode`，导致审计读取模型参数 `_version` 时抛出 `Inference tensors do not track version counter`，发生在生成前。仅将主函数加载上下文改为 `no_grad`，保留前向推理模式；重新执行 CPU 测试并冻结源码后成功。失败日志与原协议完整保留在 `attempt1/`，未改变模型、样本、分辨率或数值容差。

## 资源、边界与停止

成功 GPU Worker 从加载到四次生成及 Hook/Mass 检查合计约 8.95 秒，**不包含下载、安装和准备时间，也不是正式性能基准**。显存定时采样及 CUDA 统计见 `gpu_memory_report.csv`、`gpu1_resource_log.json`、`smoke_generations.json` 和 Hook 审计。

模型进程 PID 2600201 已退出，GPU1 释放检查时仅 27 MiB 桌面占用，没有计算进程；没有终止其他任务。见 `gpu1_release.json`。GPU0/2/3 未用于本实验。

后续可在新协议下开展 120 题 Layer Mass 与 Single-Layer Sensitivity，但需要逐图动态映射、逐层及时归约和断点/异常处理。尚未验证来源 Hard Mask、多图、12 图块、长生成或大上下文；不能由本轮小样本保证其显存安全，更不能推断消融无损、Accuracy 或真实加速。

**本轮到此停止：未启动 120 题扫描、完整 GQA 基准、来源 Mask、窗口实验或训练。** 模型和原始 Tensor 缓存位于 Git 仓库外，Git 仅保存实现、配置、审计及报告。
