# LLaVA：单卡推理、Attention 分析与 Visual Attention Knockout

本工程用于验证 `llava-hf/llava-1.5-7b-hf` 在 RTX 4090 上的正常推理、结构和输入长度。使用 `LlavaForConditionalGeneration`、`AutoProcessor`、FP16、`eager` Attention、`eval()` 和 `torch.inference_mode()`。所有参数必须位于同一张 GPU；加载后会检查参数的设备和精度。

第一阶段推理基线、第二阶段 Prompt Prefill Attention 与 Phase 2.1 Query Position 分析已完成。复用第一阶段加载和输入处理函数，通过 `outputs.attentions` 提取语言模型注意力。Phase 2.1 研究问题 Token、Answer-start 与 Teacher Forcing 的答案预测位置，该阶段没有注册 Hook。新增 Phase 3 为全局逐层 Text→Visual Attention Knockout：CPU 实现测试和真实输入审计已通过，**7B GPU 实验因显存被其他任务占用而尚未执行**。没有实施自由生成 Decode Attention、Token Pruning、量化或训练。环境与前三阶段既有代码/结果保持原样。

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
├── attention_probe.py        # 第二阶段：完整 Prompt Forward、验证、逐层逐 Head 统计和空间图
├── compare_attention.py      # 读取两个问题的保存结果，离线比较，不加载模型
├── test_attention_probe.py   # 统计量与空间距离的解析测试，不需要 GPU/权重
├── attention_position_probe.py # Phase 2.1：问题 Token 定位、答案预测偏移、因果分组及前向对照
├── compare_query_positions.py  # 离线位置热图、语义目标空间图与问题间对比
├── test_attention_positions.py # 真实 Tokenizer 离线对齐与因果/偏移测试
├── knockout_probe.py           # Phase 3：全局逐层视觉注意力阻断、teacher-forced NLL/KL/候选评分
├── test_knockout.py            # 当前 Transformers 小型随机 LLaVA + 真实 Tokenizer 的 CPU 测试
├── test_images/
│   ├── example.jpg
│   └── SOURCE.md
├── results/
└── configs/
```

模型文件保存在用户缓存目录，不放入 Git。当前主工程目录之外的 AFAC 文件未因本任务修改。

## Phase 3：逐层 Visual Attention Knockout（GPU 运行待资源空闲）

用户已授权每 15 分钟自动检查 GPU 并在资源可用时继续 Phase 3。服务器 cron 调用 `phase3_watch.py`，独立检查两张 GPU；任一卡 NVML free ≥16000 MiB 时选择该卡，再由原程序执行 CUDA 显存预检。检查期间不导入 torch、不加载权重、不操作其他进程。实验固定为现有全部 9 条 QA 与 8 个 cutoff，首次样本先做真实 7B Hook/Mask 校验；成功后执行回归测试并提交/推送，失败则保存诊断并停止。文件锁防止重复执行；完成、失败或中断状态都会停止轮询并移除本任务的 cron 条目，其他定时任务保留。

实时状态：`/home/zsy/.local/state/ate-phase3-watch/status.json`；事件与推送日志在同目录。结果发布后更新 [最新运行入口](results/phase3_knockout/LATEST.md)。可用 `/usr/bin/python3 /home/zsy/ATE/vlm_research/phase3_watch.py --status` 查询；尚未触发时可用 `--disable` 停止轮询。机器须保持开机且 cron 服务运行，不依赖聊天窗口；支持重启后的继续检查。运行被中断时保留部分数据，不盲目重跑。

当前状态和限制见 [Phase 3 报告](results/phase3_knockout/phase3_report.md)。本阶段已获用户授权，但目前只完成代码、CPU 测试、样本整理与真实 Processor 输入审计，不能声称完成 7B Cutoff Sweep 或找到融合饱和深度。

`knockout_probe.py` 继续调用 `vlm_common.load_model()` / `prepare_inputs()`，并复用 Phase 2.1 的 Question 定位、答案编码和直接追加 IDs。输入中的 576 个视觉位置由实际 `input_ids == image_token_index` 定位。对所有层安装 `register_forward_pre_hook(with_kwargs=True)`，层索引 ≥ K 时克隆原始 `[1,1,S,S]` additive mask，将所有后图像文本 Query→视觉 Key 的原合法连接置为 `-inf`；保留非目标连接、因果性及原位置。K=32 安装 no-op Hook，以便与未安装 Hook 的 Baseline 比较。Hook 始终在 context manager 退出时移除，拒绝 KV cache、Decode、非 eager backend 和不符合预期的 Mask。

先做单层阻断验证，再检查 K=32、K=0、K=16，全部 Attention 行的有限性/非负性/归一化、未来与 padding Key 为零、视觉 Query 不变、cutoff 之前层不变、Hook 清理及 output_attentions 开关等价。只保存统计和张量摘要，不保存完整 Attention 矩阵。CPU 随机小模型通过这些检查不等于 7B GPU 通过。

激活原环境，在本目录执行（不安装或下载任何依赖/权重）：

```bash
# 纯 CPU：本机已缓存 Processor 的真实输入、候选词和 Phase 2.1 前缀审计
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python knockout_probe.py --audit-inputs-only

# 纯 CPU：17 项新测试；加上前两阶段共 34 项
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest \
  test_attention_probe test_attention_positions test_knockout -v

# GPU 预检：不足 16000 MiB 可用显存时以退出码 2 停止，不尝试加载模型
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python knockout_probe.py --preflight-only

# 显存空闲后先完成原颜色问题的校验及全部 8 个 cutoff
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python knockout_probe.py --sample-ids street_color \
  --output-dir ./results/phase3_knockout_color_run

# 或一次加载、先颜色/描述，再执行全部 9 条相关样本
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python knockout_probe.py --output-dir ./results/phase3_knockout_gpu_run

# 完成之后从 JSON 离线重绘，不加载模型、不需要 GPU
python knockout_probe.py --replot-only \
  --output-dir ./results/phase3_knockout_gpu_run
```

`configs/phase3_samples.json` 固定 K=[0,8,12,16,20,24,28,32]：2 条原始任务加 7 条补充 QA。新增答案按用户指示由 AI 目视复核，明确 `human_verified=false`。数据只来自一张街景及其水平镜像，不是 9 张独立图片；镜像用于同问题、左右答案相反的空间配对，不用于 OCR。汽车颜色有黑/蓝反光歧义；原生成描述的 fire hydrant 不当作已验证事实。

所有 K 使用同一个 teacher-forced 输入，取 P−1+i 的 logits 预测 a_i；`red` 仍为 Query 598、Target 599。只对答案计算 mean/sum NLL 和 ΔNLL，逐 Token 保存 log P 与全词表 KL(Full || Knockout)。候选词在完整替换答案中重新分词并核对先前 Token 相同；多 Token 候选累加完整条件 log P，记录目标与最强竞争词的差值。不同候选统一右 padding 到相同长度并遮蔽 padding，避免候选间长度变化；这一候选专用长度可能与主参考答案 Forward 不同，候选分数只在候选控制内比较。不同长度候选的 joint log P 本身也带长度偏好，额外保存 Token 数与 mean log P。

GPU 执行后每个 `samples/<id>/` 输出 `correctness_validation.json`（首样本）、Question/Answer mapping、`metrics.json`、`token_statistics.csv`、`sample_statistics.csv`、候选评分与 Margin CSV，以及 NLL/关键 Token/候选 Margin/逐 Token 敏感性四类图。无候选词的任务不生成 Margin 图。JSON 保存每次同步 Forward 时间、allocated/reserved 峰值和 Hook 审计；CSV 也保存正式 Forward 时间和峰值。计时包括 Hook 的 Mask 检查/克隆，不能解释为推理加速。

程序不覆盖已有 `run.json` 或 `samples/`；复跑或部分失败续做应使用新的 output-dir。每个 cutoff 完成后写入 `completed=false` 检查点，全部完成才置 true。GPU 被占用时不会清理其他任务、量化或 offload。16000 MiB 是基于旧阶段约 14.5 GiB 峰值的保守启动门槛，不保证竞争任务变化或其他输入下绝不 OOM；异常记录具体阶段并保留已完成结果。

## 环境创建

本机已有 `vlm_research` 环境，继续实验只需激活，不要重复创建、下载或升级。下列安装步骤保留为第一阶段和新机器的复现记录。

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

## 第二阶段：Text-to-Visual Attention

实测报告：[phase2_report.md](results/phase2_attention/phase2_report.md)。只分析 batch=1、单图、完整 Prompt 的最后一个 Token Query，**不进行答案生成或 Decode Attention 提取**。这两个 Prompt 的末尾 Token 都是 `ASSISTANT:` 中的 `:`，不是问题中的目标词，也不是生成答案的 Token。

```bash
source /anaconda3/etc/profile.d/conda.sh
conda activate vlm_research
cd /home/zsy/ATE/vlm_research

# 根目录保存颜色问题；description/ 保存图像描述问题。
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python attention_probe.py \
  --image ./test_images/example.jpg \
  --question "What color is the stop sign?" \
  --output-dir ./results/phase2_attention

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python attention_probe.py \
  --image ./test_images/example.jpg \
  --question "What is in this image?" \
  --output-dir ./results/phase2_attention/description

# 以下命令读取已保存的小型结果，不加载模型或运行 GPU Forward。
python compare_attention.py \
  --first ./results/phase2_attention/description \
  --second ./results/phase2_attention \
  --output-dir ./results/phase2_attention/comparison

python -m unittest -v test_attention_probe.py
```

`attention_probe.py` 使用现有 `load_model()` / `prepare_inputs()`，保持 FP16、eager、eval、inference_mode 和整模型 cuda:0。对完全相同输入，先执行 `output_attentions=False`，再执行 `True`，两次都设 `use_cache=False, return_dict=True`。比较完整 Prompt logits 和最后位置的下一 Token argmax，不通过 Decode 序列来间接比较。同步计时不含模型加载；对照第一次 Forward 是冷启动，不能把它与后一次 Forward 的时间差解释为 Attention 开关带来的加速。

精确视觉索引来自 `input_ids == model.config.image_token_index`。可访问 Key 同时满足非 padding 和 `key_index <= query_index`。本次所有 576 个图像 Token 都可访问；实际输入长度为 594/595，不写死序列长度。每层检查 `[1,32,S,S]`、完整矩阵有限且非负、当前 Query 的可访问权重和在 `1 ± 0.002` 内，并检查图像占位和实际投影视觉特征 `[1,576,4096]` 一致。

每层每个 Head 保存以下指标（CSV 共 32×32=1024 行）：

| 字段 | 定义 |
| --- | --- |
| `visual_attention_mass` | 原始视觉 Key 权重之和 Mv |
| `nonvisual_attention_mass` | 所有可访问非视觉 Key 权重之和 Mn；包含 BOS、模板与 Query 自身 |
| `uniform_visual_baseline` | Nv / (Nv + Nn)；本次约 0.970/0.968 |
| `visual_attention_density_vs_uniform` | D = Mv / baseline；等于视觉每 Token 密度相对于全体均匀分布的比值 |
| `visual_nonvisual_density_ratio` | R = (Mv/Nv) / (Mn/Nn + 1e-12) |
| `visual_entropy_nats` | 对视觉内部条件分布 p=a/Mv 计算 -Σp ln p；额外保存除以 ln(Nv) 的归一化值 |
| `top10_visual_attention_mass` | 最大 ceil(0.1×Nv)=58 个 Patch 的原始权重之和；另存除以 Mv 的条件份额 |

D=1 和 R=1 分别是相应的等密度参考值。层曲线是逐 Head 指标的算术平均，**mean(R_head) 不等于先平均质量后求 R**；阴影为跨 Head 标准差，不是统计置信区间。视觉质量为 0 时，条件熵的存储值为 0，并以 `visual_entropy_defined=False` 标为未定义。

每个问题的输出目录均包含：

```text
attention_statistics.csv/json              # 逐 Head 指标、输入索引、定义、形状/数值验证、时间与显存
layer_statistics.csv                      # 逐层平均/标准差
layer_visual_attention_mass.png            # 原始视觉质量与 Token 数量均匀基线
layer_normalized_attention.png             # D 与 R 曲线，R 使用对数纵轴
head_layer_attention_heatmap.png            # Layer×Head 的质量、D、R（R 对数色标）
clip_processed_image.png                   # 实际 FP16 pixel_values 反归一化后的 336×336 RGB
spatial_attention_layer_XX.png             # 原始质量标注 + 24×24 图 + 336×336 叠图
spatial_attention_grid_layer_XX.png         # 不叠加图像、精确 24×24 像素的彩色 Grid
spatial_attention_layer_XX.npy              # Head 平均的原始 24×24 Patch 权重
query_visual_attention.npy                 # [32,32,576]，仅最后 Query 的视觉权重
query_nonvisual_attention.npy              # [32,32,Nn]，非视觉权重；Key 顺序见 JSON
```

空间层为 `[0,4,8,12,16,20,24,28,31]`（从 0 编号）。CLIP 对原图先按短边 Resize，再 Center Crop；叠图直接反归一化真实输入张量，包含 FP16 和 uint8 的舍入误差，不使用原图坐标。Patch 按 CLIP 卷积输出的行优先顺序恢复，CLS 已移除。空间图仅在显示时除以本层最大值，原始质量在面板标题中标注，原始概率另存 NPY；单独 24×24 PNG 的质量与色标应参照同层面板和统计表，不能独立比较亮度。

`comparison/` 保存两个问题的质量/D/R 曲线、空间叠图并排图、共享色标的条件空间差值图，以及每层 TV、JS 散度、余弦相似度的 CSV/JSON。空间比较先将每个 Head 平均的视觉图除以各自视觉总质量，方向为第二个问题减第一个问题；它只比较视觉内部空间分布，不替代原始质量。两个问题的 Token 数也不同，不能将差异完全归因于问题语义。

不向磁盘保存完整 S×S Attention。Forward 返回全部层后移除 ModelOutput 的引用，逐层只复制最后 Query 的小型向量到 CPU，聚合后释放该层 GPU 矩阵。记录 Forward 和验证阶段的 PyTorch allocated/reserved 峰值，驱动进程显存只是采样值。若 OOM，会在 `failure.json` 记录发生阶段和显存；先排查 GPU 占用/输入，再评估逐层提取，不自动量化或 CPU offload。

本阶段结论只描述最后模板 Token 的 Attention。BOS 集中与背景 Patch 热点不证明信息被文本吸收，也不证明任何层之后可安全删除视觉 Token。第二阶段完成后停止，未开展 Attention Knockout。

## Phase 2.1：Question 与 Answer Prediction

报告：[phase2_1_report.md](results/phase2_1_attention/phase2_1_report.md)。新阶段不覆盖 `results/phase2_attention/`，直接导入已有 `key_masks()`、`summarize_rows()`、`layer_summary()`、`make_plots()` 和空间距离函数。无需修改或重装环境。

```bash
conda activate vlm_research
cd /home/zsy/ATE/vlm_research

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python attention_position_probe.py \
  --image ./test_images/example.jpg \
  --question "What color is the stop sign?" \
  --answer "The stop sign is red." \
  --answer-source ground_truth_teacher_forced \
  --query-mode all \
  --content-phrases color "stop sign" \
  --semantic-answer-phrases red \
  --output-dir ./results/phase2_1_attention

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest -v test_attention_probe.py test_attention_positions.py
```

分析类型：

| `--query-mode` | 分析位置 |
| --- | --- |
| `all`（默认） | 全部问题 Token、Answer-start、全部答案预测 Query；需提供答案 |
| `question_all` | 每个问题 Token，另存无权重平均的描述性汇总 |
| `question_last` | 问题的最后 Token，通常是 `?`，不是模板末尾的 `:` |
| `question_content` | `--content-phrases` 手动指定的片段，允许一个词拆为多个子词 |
| `answer_start` | 原始 Prompt 最后位置 P−1，用于预测首个答案 Token |
| `answer_prediction` | 预测答案 a_i 的位置 P−1+i；需提供答案 |

问题模式可不传 `--answer`；即使只选择某一种模式，也会额外捕获 Answer-start 作一致性验证，并在 metadata 的 `validation_only` 标记。默认 `all` 会捕获所有问题 Token，因此原始需求中不含 `--content-phrases` 的命令也可运行；手动指定片段后才有 Content 视图。Answer-start 与预测 a0 是同一个物理 Query，只保存一份权重，允许属于多个分析视图。

Question 定位步骤为：用占位标记确定真实模板的 Question 字符范围；对原模板取得 Fast Tokenizer offset mapping；按真实图像 Token 数展开单个 `<image>` 的 Token ID/offset；验证展开后的 **全部 ID** 与 `prepare_inputs()` 完全一致。只选择 Question 范围内的 Token，验证合并解码与原问题逐字符一致，并覆盖全部字符；不能靠英文空格或单词数量定位。首版对问题/答案的首尾空白或模板特殊 Token 显式报错，避免边界歧义。

Teacher Forcing 只对 Answer 使用同一 Tokenizer、`add_special_tokens=False`，将答案 IDs 和全 1 mask 追加到已验证 Prompt 后。原 Prompt IDs/mask 前缀完全保留，`pixel_values` 是同一对象；不追加 BOS/EOS、不重复处理图像。默认位置编码和 `use_cache=False` 保持不变。

例如 B 的 `red` 是第 4 个答案 Token，目标位置 599，对应 **Query 598，输入 Token 是 `is`**。Query 599 则是在 `red` 已输入后预测 `.`，两者不得混淆。CSV 同时记录 Query、目标 ID/文本和目标位置。

答案来源必须正确标注：`ground_truth_teacher_forced` 表示使用调用方提供的参考答案，不意味着这里新增了人工标注或外部真值验证；`model_generated` 表示固定的模型输出轨迹。若与第一阶段归档匹配，脚本还会核对生成 ID（去掉归档 EOS）与重新编码后的 ID 完全相同。所有 Teacher Forcing 结果均限定于固定答案历史，不能直接称为自由生成 Decode 轨迹。

复现 A 时直接读取第一阶段真实回答：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
import json, subprocess, sys
from pathlib import Path
sample = json.loads(Path("results/validation.json").read_text())["tests"]["A_description"]
subprocess.run([
    sys.executable, "attention_position_probe.py",
    "--image", "./test_images/example.jpg",
    "--question", sample["question"], "--answer", sample["answer"],
    "--answer-source", "model_generated", "--content-phrases", "image",
    "--semantic-answer-phrases", "red", "stop sign",
    "--output-dir", "./results/phase2_1_attention/description",
], check=True)
PY

# 从已保存数据重绘，不加载模型。
python compare_query_positions.py --input-dir ./results/phase2_1_attention
python compare_query_positions.py \
  --first ./results/phase2_1_attention/description \
  --second ./results/phase2_1_attention \
  --output-dir ./results/phase2_1_attention/comparison
```

每个 Query 的可访问 Key 被互斥且完备地划分为 **visual / bos / question / prompt_template / previous_answer**；BOS 不算普通文本。`previous_answer` 包含当前 Query 对应的已输入答案 Token，但不包含它正在预测的目标 Token。额外列 `prompt_text`、`other_text` 是派生小计，不能再与五组相加。质量、D、R、条件熵与 Top-58 Share 沿用 Phase 2，但每个 Query 有自己的基线 Nv/N_accessible；Content/All 的无权重平均仅是描述性汇总。

一致性检查执行三次 Forward：原始 Prompt、完整 Teacher Forcing、与 Teacher Forcing 等长度但将答案全部替换为 padding 并遮蔽的 Prompt。B 的首次 Smoke Test 发现，跨序列长度比较的部分非视觉权重超过最初阈值。排查后：相同长度的两次 logits/Prompt Attention 完全一致，实际前缀因果 Mask 也相同；原始 Prompt 的视觉行与 Phase 2 完全一致，Teacher Forcing 的视觉与非视觉行均在误差检查范围内。这将该误差隔离到长度变化下的有限精度计算，而非答案泄漏。

脚本保留最初 `atol=5e-4, rtol=5e-3` 的诊断结果；跨长度比较另要求 `atol=3e-3, rtol=5e-3`、每行总 L1 误差≤0.02、下一 Token argmax 不变，并且等长度对照必须逐元素一致。Logits 检查阈值 `atol=0.02, rtol=0.002`，实际差异完整记录。任何必需检查失败会停止并写 `failure.json`，不忽略差异、不修改第三方实现。

主要输出（B 在根目录，A 在 `description/`）：

```text
position_metadata.json                  # 完整 Token/Query 映射、定义、输入与验收
question_token_mapping.csv
answer_prediction_mapping.csv           # 目标 Token 与前一位置 Query 的对齐
head_statistics.csv                     # 每层×每 Head×每个唯一 Query；包含五组质量
query_layer_statistics.csv              # 每个 Query 的 Head 平均与 SD
query_group_layer_statistics.csv/json   # 分析视图的描述性均值
selected_query_attention.npz            # FP16 [32,32,N_unique_query,N_sequence_keys]，只存选定行
forward_consistency.json                 # 原始/Teacher/等长度对照及 Phase 2 比较
layer_question_token_heatmap.png
layer_answer_prediction_heatmap.png      # 横轴是被预测 Token，并标明 q→target
question_last_vs_answer_start.png
query_type_comparison.png
query_spatial_comparison.png
query_specificity_by_layer.csv/png       # Query 间质量范围、视觉内部平均两两 JS/TV
semantic_answer_XX_word/                 # 每个匹配目标 Token 的九层空间图、原始 Grid、曲线
```

读取 Question→Visual 数组时可使用 `np.take` 保留轴顺序：先按 metadata 取得问题 Query 的 axis 索引，再对 `attention` 的 Query 轴（axis=2）及视觉 Key 轴（axis=3）执行 `np.take`，得到 `[32,32,Nq,576]`。未保存任何全层完整 S×S 矩阵。

空间叠图继续复用 Phase 2 的真实 CLIP 输入恢复与绘图函数，原始 Visual Mass 明确标注，单独缩放颜色只供查看空间形状。Query 特异性的 JS/TV 基于 Head 平均视觉图的条件分布，它舍弃总视觉质量，必须结合质量曲线解读。不同 Token 的可见历史不同，跨问题还有长度与位置混杂因素。本阶段只建议后续对照设计，**完成后停止，不实施 Knockout、Pruning 或训练**。

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
