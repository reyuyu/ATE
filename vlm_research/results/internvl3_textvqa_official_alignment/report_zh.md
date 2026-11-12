# InternVL3-8B TextVQA val 官方输入协议对齐

## 结论与实际完成范围

**状态：PARTIAL / BLOCKED_GPU_BUSY。不是已完成的官方成绩复现。**

2026-11-13（北京时间）已实际完成官方源码取证、TextVQA_VAL 文件校验、全部5,000题动态输入冻结、Late9动态长度适配及9项CPU测试。启动门禁发现四张3090均存在外部计算进程，因此未加载模型、未启动GPU Smoke、未生成Native或Late9答案。未抢占、终止或更改任何其他GPU任务，没有后台轮询。

| 条件 | 完成题数 | 官方Soft Accuracy | 相对80.2% | 相对本轮Native |
| --- | ---: | ---: | ---: | ---: |
| 本轮Native | 0/5,000 | NOT_RUN | NOT_MEASURED | 不适用 |
| 本轮SVH-R Late9（L19–L27） | 0/5,000 | NOT_RUN | 不适用 | NOT_MEASURED |

不能用历史固定单图块成绩填补上述空缺，不能宣布Late9数值验收通过，也不能声称已经复现80.2%。当前仅完成官方输入/评分配方的静态对齐，端到端官方协议对齐仍待Native验证。

## 官方证据与版本选择

官方文档归档为 `sources/internvl3_evaluation.html`。其中483、486行说明主要使用VLMEvalKit，并说明工具包、版本、环境与硬件可能导致差异；584–611行为VLMEvalKit使用入口。官方博客归档为 `sources/internvl3_blog.html`，TextVQA表中1172–1175行为InternVL3-8B及80.2。

在读取任何本轮模型预测前，审计了两个上游快照：

1. **冻结执行配方：`1cc6d16d34b8f164523388edb0a9642ef8ab7314`。** 2025-04-11首个注册InternVL3的官方VLMEvalKit提交。`release_sources/vlmeval/config.py:742–743` 将8B注册为`InternVLChat(..., version="V2.0")`，使用HF路径。
2. **仅作比较：`cb1b907680808ffb3c314b8a34e0c90a7a3db1de`。** 对应文档更新日期附近快照；`sources/vlmeval/config.py:827–828` 已为8B开启`use_lmdeploy=True`。不能悄悄将这一运行时与HF视为相同实现。

选择发布时HF快照的理由是可复核的发布时间和明确的HF入口，不是根据80.2%的目标分数选择版本。完整文件列表、Git blob和SHA256分别见`release_source_manifest.json`和`source_manifest.json`。

**证据边界：作者未在这些资料中公开80.2%对应的唯一运行Commit、完整依赖锁和Attention后端。** 因此冻结的是官方发布时HF配方，不宣称已经确定作者隐藏的原始运行环境。当前候选运行时为既有Torch2.6.0+cu124、Transformers4.49.0、BF16/Eager；它不冒充后来的LMDeploy配置。Native未实测前不能认定运行时差异无影响。

## 已冻结配方

| 项目 | 冻结值 | 可核查源码 |
| --- | --- | --- |
| 数据 | 官方`TextVQA_VAL`，5,000题，每题10个人工答案 | `release_sources/vlmeval/dataset/image_vqa.py:20,32,60–61` |
| 模型 | `OpenGVLab/InternVL3-8B`，revision `853e3a797a661694b1b8ece0cb72dc2b23e3dac9` | 本地冻结模型文件及`protocol.json` |
| 图像 | RGB，动态长宽比切图，448×448，ImageNet归一化 | `release_sources/vlmeval/vlm/internvl/utils.py:23–97` |
| 视觉预算 | `max_num=6`网格图块，附缩略图，最多7个实际图块、1,792个语言侧视觉Token | `internvl_chat.py:182–201`、`utils.py:83–95` |
| Prompt | 原题后接`Answer the question using a single word or phrase.`，无外部OCR | `internvl_chat.py:135–180` |
| Chat | 模型原生`internvl2_5`，原System/角色/分隔符 | 模型`conversation.py`、`modeling_internvl_chat.py:253–298` |
| 生成 | Greedy，`max_new_tokens=4096`；有效配置`temperature=0.7, top_p=0.95, do_sample=False`；自然EOS | `internvl_chat.py:118–120,274–298` |
| 评分 | 上游`process_line(method='vqa_score')`，十答案leave-one-out共识软分 | `dataset/image_vqa.py:60–61`、`dataset/utils/vqa_eval.py:239–268` |

实际完整展开Prompt、Token IDs和各类哈希见`prompt_examples.json`及5,000题的`sample_manifest.jsonl.gz`。EOS、pad ID、依赖版本和全部代码哈希见`protocol.json`。

本轮不会把原生模型config中的默认`max_dynamic_patch=12`误当作该Benchmark生效值；TextVQA在上述VLMEvalKit路径中明确走6图块上限。也不会将生成上限自行改为旧ATE的32或128。

### 数据实际校验

- 官方源码指定MD5：`b233b31f551bbf4056f2f955da3a92cd`，下载后完全匹配。
- TSV SHA256：`f9bd2e1518185f678a60b9a31fea5655f7533704fa7d7c9b3775bf0ca2a6b8d0`。
- 官方原域名存在TLS证书问题；实际从同一官方OSS对象的有效TLS地址下载，未禁用TLS校验。两个入口Content-Length/ETag匹配，最终以源码MD5验证内容。
- 全部5,000题的Question ID、Question和十答案与本地官方TextVQA标注逐项一致。
- 3,166个图片SHA簇；仅对完全相同图片去重存储，未删题或重采样。
- 实际图块数：7块4,043题，5块681题，3块238题，4块38题。
- 沿用官方TSV图片解码后PIL保存JPEG的语义，未直接以可能字节不同的旧ATE JPEG替换。每张实际图片字节与动态BF16像素Tensor分别哈希。
- 每题绑定图片、原生Prompt、Token IDs、实际视觉Token位置、Gold和生成参数；Manifest SHA256为`1c4097027de44d01bf7e95733cff92ed26c24ab4321403fb2db4cdab53b05369`。

## CPU测试与Late9适配

`cpu_tests_final.log`记录**9/9通过**：官方Prompt、Soft Accuracy、独立MMF评分器、动态切图、真实Tokenizer位置、动态KV/零Scratch、Late9权限、旧模块未被修改、图片簇Bootstrap。

首次CPU输入准备因不必要的pandas依赖失败，已移除该适配器依赖，未安装包或修改共享环境。首次评分测试暴露缺少上游标点函数导入，已改为加载固定上游实现。失败记录分别保留在`prepare_cpu_attempt1.log`和`cpu_tests_attempt1.log`，不改判为通过。

历史Late9物理引擎将**总视觉长度**硬编码为256，不能直接用于动态多图块。本轮独立适配模块仅将这类总长度转换为真实`len(roles['image'])`，不修改历史文件和共享模块全局值。所有转换位置和源码SHA记录在`dynamic_length_source_audit.json`。

保留的Late9结构为：P−1和全部真实Decode只在L19–L27读原始视觉KV；普通R写入仅L0–L19；答案Attention在L2–L5跳算；为计算Late9所需视觉KV保留相应视觉计算，退出位于L27之后；持久视觉KV只在9个回看层保留。不会将普通Dense Mask冒充Physical。

**CPU适配不等于GPU数值通过。** 新门禁为Native/No-op全部有效Logits严格`torch.equal`，Dense/Physical逐步Logits最大绝对误差不超过0.02且完整Token序列一致。历史Late9数值waiver不继承。短/中/长Smoke题号在预测前冻结为`34635`、`38948`、`36148`。任何失败或OOM均停止，不改图像预算、精度或阈值。

## 已实现但尚未执行的GPU流程

1. 单次检查真实空闲3090，不轮询、不抢占。
2. 3题Native/No-op Smoke，检查输入完全对应冻结Manifest。
3. 正式Native5,000题，逐题SQLite原子Checkpoint及记录SHA，保留原始答案、Token IDs、EOS、截断和输入哈希。
4. 独立重新评分；与80.2%的差距大于预声明1pp时停止并排查，不运行Late9。这是复现诊断门禁，不是调参或非劣效认定。
5. Native完整且通过复现检查后，运行3题动态Late9 Dense/Physical门禁。
6. 只有门禁通过才运行同输入Late9全部5,000题，逐题比较软分增减，按图片SHA整簇Bootstrap20,000次。

主分数始终用冻结VLMEvalKit评分器。额外从只读历史归档加载独立MMF/LLaVA TextVQA评分器，分别输出逐题分数；若归一化细节存在差异则显式记录，不能拿较高的评分器替换主结果。

`accuracy_summary.csv`和`per_question.csv`当前是带表头/NOT_RUN状态的准备产物，不是完成预测。`completion.json`明确标记PARTIAL。

## 资源阻塞与后续启动

2026-11-13 02:20:27（北京时间）门禁记录如下，详见`gpu_preflight.json`：

| GPU | UUID | 已占显存MiB | 外部任务 |
| --- | --- | ---: | --- |
| 0 | `GPU-a63c6362-dcfd-4a26-c44b-014b40127c89` | 12,912 | SPECAT |
| 1 | `GPU-b08489af-0a8a-eb48-55ba-8346733af417` | 12,912 | SPECAT |
| 2 | `GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6` | 2,542 | stable_spectral |
| 3 | `GPU-96bc853f-ceea-2404-c704-c48a3d1ca185` | 2,324 | stable_spectral |

GPU2/3显存较少也不代表空闲，它们有活跃计算进程。此次启动器返回2，模型加载次数0，CUDA初始化次数0，GPU Smoke0/3，正式预测0/10,000。

资源自然空闲后，在本工作树运行以下命令可执行已冻结的门控流程；每次启动都会重新检查哈希及GPU，不会跳过Native门禁：

```bash
cd /data/lk/ATE-worktrees/internvl3-textvqa-official-alignment
/data/lk/深度学习/.envs/ate-internvl3/bin/python -u -m vlm_research.internvl3_textvqa_official.run launch
```

本轮没有运行GQA、测速或层搜索，没有修改main、历史Late9工作树或共享环境。原始TSV、图片和未来SQLite位于本目录`server_only/`，已Git忽略；GitHub只包含源码、审计、冻结输入清单和状态，不上传权重或原始图片。

最终问题的当前答案：**Native得分未知；与80.2%的差值未知；Late9得分及损失未知；官方协议尚未完成GPU实测对齐，不能宣布成功。**
