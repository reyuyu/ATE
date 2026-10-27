# LLaVA 第一阶段：单卡 FP16 推理基线

本工程用于验证 `llava-hf/llava-1.5-7b-hf` 在 RTX 4090 上的正常推理、结构和输入长度。使用 `LlavaForConditionalGeneration`、`AutoProcessor`、FP16、`eager` Attention、`eval()` 和 `torch.inference_mode()`。所有参数必须位于同一张 GPU；加载后会检查参数的设备和精度。

仅实现第一阶段。没有 Attention Hook、Attention Probability 提取、Token Pruning、量化或训练代码。结构记录中的模块路径供后续阶段参考。

## 文件

```text
vlm_research/
├── README.md
├── requirements.txt          # 固定直接依赖，包括匹配的 torch/torchvision
├── requirements-lock.txt     # 安装成功后的全部 pip 依赖快照
├── download_model.py         # 固定 revision、断点续传、权重 SHA256 校验
├── vlm_common.py             # 单卡加载、输入准备、计时和显存统计
├── inference.py
├── inspect_model.py
├── validate.py               # A/B/C；同一进程内重复三次
├── test_images/
│   ├── example.jpg
│   └── SOURCE.md
├── results/
└── configs/
```

模型文件保存在用户缓存目录，不放入 Git。当前主工程目录之外的 AFAC 文件未因本任务修改。

## 环境创建

```bash
conda create -n vlm_research python=3.11 -y
conda activate vlm_research
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install --index-url https://pypi.org/simple -r requirements.txt
python -m pip check
```

若当前 shell 尚未初始化 Conda，本机可先执行 `source /anaconda3/etc/profile.d/conda.sh`。

选择兼容性明确的稳定组合：PyTorch `2.6.0+cu124`、torchvision `0.21.0+cu124`、Transformers `4.49.0`。本阶段不安装 vLLM、DeepSpeed、FlashAttention。完整依赖版本见 `requirements-lock.txt`；复现已验证环境可用：

```bash
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install --index-url https://pypi.org/simple -r requirements-lock.txt
```

本机 NVIDIA 驱动为 `595.99.02`，`nvidia-smi` 显示最高支持 CUDA `13.2`，可向后兼容本项目使用的 CUDA `12.4` 运行库。`nvidia-smi` 的 CUDA 数字并不表示安装了对应 CUDA Toolkit。PyTorch CUDA wheel 带运行库，本阶段不依赖系统 `nvcc`。

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## 本机网络问题

初始环境中的本地 HTTP/HTTPS 代理引起 TLS EOF，Conda、pip、HF 请求失败。本任务只在下载命令中绕过代理，没有修改系统代理配置。需要时给命令加：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy NO_PROXY='*' no_proxy='*' \
  conda create -n vlm_research python=3.11 -y \
  --solver classic --repodata-fn current_repodata.json
```

pip 同样可以加上述 `env ...` 前缀。官方 PyTorch CUDA wheel 与普通 Python 依赖分两步安装，避免混合额外索引让普通依赖走较慢的 CDN。本次使用官方 PyTorch/NVIDIA CDN 下载 CUDA wheel，并核对公开索引中的 SHA256；普通 Python 包通过清华 PyPI 镜像下载，版本不变。可用 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` 替换普通依赖安装步骤中的 PyPI 地址。不要通过关闭证书校验来解决 TLS 问题。

## 下载模型

固定官方仓库的 revision：`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。不使用量化权重。

```bash
python download_model.py
```

本机无法直接解析 Hugging Face 主站，因此实际使用镜像传输同一官方转换仓库文件：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy python download_model.py --endpoint https://hf-mirror.com
```

脚本核对 revision，按 LFS SHA256 校验三个 safetensors 分片，支持下载中断后的续传，并在 `configs/model.json` 保存本地路径、文件大小和校验值。默认位置：

```text
~/.cache/huggingface/manual/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/
```

加载本地模型时使用 `local_files_only=True`，运行推理不需要访问网络。也可用 `--model /absolute/checkpoint/path` 指定完整的同模型目录。

## 模型加载与图像输入

核心加载设置：

```python
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    device_map={"": "cuda:0"},
    low_cpu_mem_usage=True,
    use_safetensors=True,
    local_files_only=True,
).eval()
```

程序默认设置 `CUDA_VISIBLE_DEVICES=0`，并要求进程只看到一张卡。若外部设置了该变量则尊重它；本次测试均显式设置为物理 GPU 0。`cuda:0` 是进程内的逻辑编号。

通过 `processor.apply_chat_template` 格式化图文对话，再由 processor 处理本地 RGB 图片和文本。在固定的 4.49.0 环境中保留 checkpoint 默认的图像处理器和 Fast tokenizer；加载时关于未来 `use_fast` 默认值的提示不影响本次推理。显式从配置设置 `patch_size` 和 `vision_feature_select_strategy`，并给 CLIP 设置一个附加 CLS token。`default` 特征选择会移除 CLS，336×336 图像、14×14 patch 最终展开成 576 个视觉 token；程序会对实际 `input_ids` 中的图像 token 计数并校验。实际 LLM 输入长度按具体问题计算，不写死。

## 单图推理

在本目录、激活环境后执行：

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --image ./test_images/example.jpg \
  --question "What is in this image?"

CUDA_VISIBLE_DEVICES=0 python inference.py \
  --image ./test_images/example.jpg \
  --question "What color is the stop sign?" \
  --output ./results/inference_detail.json
```

默认 `max_new_tokens=64`、`do_sample=False`、`use_cache=True`。可用 `--max-new-tokens` 修改生成上限。程序按输入 token 长度切片，只解码新增 token，不将 Prompt 作为答案打印。默认 JSON 保存到 `results/inference.json`。

计时在 CUDA 同步后开始/结束：`generation_seconds` 只包含 `generate()`，`preprocess_and_generation_seconds` 包含图片读取、预处理、输入传输及生成。模型加载时间单列；首次调用可能包含 CUDA 初始化及 kernel 的额外开销。为检查 NaN/Inf，生成时保留各步词表分数并检查有限性；这会增加少量显存，未提取 Attention。

显存单位均为 MiB（除报告中特别标记的 GiB）：`allocated` 是 PyTorch 活跃张量，`reserved` 是分配器预留，`peak_*` 是本次推理内峰值，均含驻留模型参数。`nvidia_smi_process_mib` 记录当前进程在驱动侧的显存，包含 CUDA 上下文等额外开销；它是采样值，不是峰值。显存统计不包含 GPU 1 的其他任务。

## 结构检查

```bash
CUDA_VISIBLE_DEVICES=0 python inspect_model.py \
  --image ./test_images/example.jpg \
  --question "What is in this image?"
```

输出并保存 `results/model_structure.json` 和 `results/model_modules.txt`，记录模型类、固定权重 revision、权重原始 Transformers 版本、当前库版本、精确参数数量、Vision Encoder、LLM、层数、hidden size、Attention/KV heads、投影前后视觉维度、视觉 token 数和实际 LLM 输入总长度。

结构字段从已加载模块、配置和真实 processor 输出读取。本阶段不注册 Hook。当前版本的模块位置见结构 JSON 中的 `module_paths_for_future_reference`。

## 验收 A/B/C

```bash
CUDA_VISIBLE_DEVICES=0 python validate.py
```

- A：`What is in this image?`，自然街景描述。
- B：`What color is the stop sign?`，检查明确物体属性。
- C：同一模型实例、同一输入连续生成三次，检查 token ID 完全一致、分数无 NaN/Inf、清理临时输出后的活跃显存稳定以及缓存显存是否趋稳。

A/B 后进行 C，C 的三次调用之间不执行 `empty_cache()`，不重新加载模型。允许活跃显存波动 1 MiB，检查后两次预留显存是否稳定。三次短测只能说明本次运行的稳定性，不能证明长时间运行绝无泄漏。单个正确回答不作为视觉能力评测。

完整结果保存在 `results/validation.json`，执行输出见 `results/validation.log`；人工核对图像相关性的结论与实测摘要见 `results/phase1_report.md`。

## 常见问题

- **CUDA 不可用**：确认激活了 `vlm_research`，运行 `nvidia-smi` 和上面的 torch 检查。确保安装的是 `+cu124` 构建，而非 CPU 构建。
- **Conda 网络失败**：检查本地代理；本机成功路径为绕过代理、classic solver 和 `current_repodata.json`。不需要更新 base 环境。
- **HF DNS / TLS 失败**：先修复代理，或使用记录在配置中的镜像传输官方文件；下载完成后可完全离线推理。
- **显存 OOM**：先检查是否确实使用空闲 GPU 0、是否有新进程占用；保持 batch=1、单图 336×336 和 64 个新增 token。检查输入和图片数量，先排查占用，不自动改用 4-bit 或 CPU offload。
- **图像 token 与特征数量不匹配**：使用固定的 Transformers 版本和完整 processor 文件；显式配置 patch size、CLIP CLS token 和特征选择策略；不要截断图像占位 token。
- **`torchvision::nms` / torchvision 导入错误**：使用配套的 `torch==2.6.0+cu124` 与 `torchvision==0.21.0+cu124`，不要混装不匹配的 CPU/CUDA 构建。
- **找不到 chat template / API 改变**：确认完整下载固定 revision 的 processor、tokenizer 和 `chat_template.json`，并使用本工程锁定版本。
- **输出包含 Prompt**：使用新增 token 切片逻辑；不要直接解码完整 `generate()` 序列。

参考：[HF LLaVA 文档](https://huggingface.co/docs/transformers/model_doc/llava)、[4.49.0 官方文档源文件](https://github.com/huggingface/transformers/blob/v4.49.0/docs/source/en/model_doc/llava.md)、[官方模型仓库](https://huggingface.co/llava-hf/llava-1.5-7b-hf)、[PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。
