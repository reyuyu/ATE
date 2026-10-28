# LLaVA：单卡推理、Attention 分析、Knockout 与 Physical Exit

本工程用于验证 `llava-hf/llava-1.5-7b-hf` 在 RTX 4090 上的正常推理、结构和输入长度。使用 `LlavaForConditionalGeneration`、`AutoProcessor`、FP16、`eager` Attention、`eval()` 和 `torch.inference_mode()`。所有参数必须位于同一张 GPU；加载后会检查参数的设备和精度。

第一阶段推理基线、第二阶段 Prompt Prefill Attention 与 Phase 2.1 Query Position 分析已完成。复用第一阶段加载和输入处理函数，通过 `outputs.attentions` 提取语言模型注意力。Phase 2.1 研究问题 Token、Answer-start 与 Teacher Forcing 的答案预测位置，该阶段没有注册 Hook。Phase 3 全局逐层 Text→Visual Attention Knockout 已完成真实 7B 小规模实测；Phase 3.1 新增物理视觉 Token 退出，已完成 12 张固定 GQA 图片的 Full/Mask/Physical 对照。没有实施自由生成 Decode Attention、Token Pruning、量化或训练。Phase 3.2 已扩展为 240 张新增独立图的固定 K 验证，Phase 3.3 已完成同240图的定向后缀Mask诊断；Phase 3.4 已完成240图ROI审计及48主样本+5独立案例的ROI/Sink小规模诊断，详见下文；环境与旧阶段既有代码/结果保持原样。

## 跨服务器交接

[完整交接](handoff/README.md) · [完整GQA数据/权重下载及路径适配](handoff/DATASET_SETUP.md) · [新服务器Codex提示词](handoff/CODEX_PROMPT.md)。涵盖已完成的单案例六条件结果、240图/48主图统计基础与缓存；按252张冻结原图SHA256恢复，不重采样。初始交接不新增GPU实验；后续单案例完成状态已同步。

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
├── physical_exit.py            # Phase 3.1：保留原始位置/RoPE 的真实隐藏状态删除
├── physical_exit_pilot.py      # 固定12张GQA图的Full/Mask/Physical对照及两张曲线
├── prepare_physical_exit_data.py # 官方CSV确定性采样、图片与源文件哈希
├── test_physical_exit.py       # 物理长度、位置、RoPE、数值等价和多Token评分测试
├── directed_suffix.py          # Phase 3.3：继承既有安全Mask Hook，定向Query→视觉Key阻断
├── directed_suffix_probe.py    # 冻结240图、真实GPU校验、完整候选评分与严格断点续跑
├── summarize_directed_suffix.py # 离线完整性校验、配对统计和三张核心图
├── test_directed_suffix.py     # 定向Mask方向、因果不变量、P−1、清理和续跑测试
├── roi_sink_common.py          # Phase 3.4：真实裁剪几何、冻结Sink规则、匹配背景和Patch Mask
├── prepare_roi_sink.py         # 240图ROI审计及按几何固定48主样本/5独立案例
├── roi_sink_probe.py           # 观察→冻结候选→7B Smoke→小规模定向阻断
├── roi_sink_precision.py       # 五案例固定55条件的FP32输出投影敏感性检查
├── summarize_roi_sink.py       # 离线完整性校验、分Head统计、配对效果和图表
├── test_roi_sink.py            # 小模型/真实Processor/匹配控制/统计和隔离测试
├── exit_boundary_analysis.py   # Phase 3.5：纯缓存CPU统计/图级Bootstrap/边界曲线
├── test_exit_boundary_analysis.py # 离线指标/偏移/并列/Bootstrap数值测试
├── test_images/
│   ├── example.jpg
│   └── SOURCE.md
├── results/
└── configs/
```

模型文件保存在用户缓存目录，不放入 Git。当前主工程目录之外的 AFAC 文件未因本任务修改。

## 单案例15133706：逐Head视觉Attention继承（真实GPU完成）

`attention_inheritance.py`及`attention_inheritance_pilot.py`仅在L24～L31、P−1→576视觉Keys，按λ=0/.5/1混合Full L23/L24的逐Head视觉条件分布，保留当前视觉Mass和非目标权重；实例包装结束/异常时恢复，不改库或旧实验。7项专项测试和全工程101项CPU回归通过。

[完成报告](results/attention_inheritance_15133706/completed_20261028_2245/report.md) · [对比图](results/attention_inheritance_15133706/completed_20261028_2245/margin_comparison.png) · [六条件CSV](results/attention_inheritance_15133706/completed_20261028_2245/per_condition_results.csv) · [验收](results/attention_inheritance_15133706/completed_20261028_2245/validation.json)。L23 λ0/.5/1 Margin为−.03125/+.0234375/+.0859375；L24为−.03125/−.03125/−.0390625。L23非零λ使right二选一预测转正，L24没有；历史Physical K24 +.0390625仅作旧对照，正确候选全词表绝对概率反而下降，不能视为一般能力改善。

两source λ0全词表logits精确Full，96层级审计通过；所有条件Question最终hidden、非目标连接及运行后原生Full精确不变，无NaN/Inf/OOM、库文件未改。最大行和误差.0002965，最大Mass相对误差.0005598。8审计超过旧固定条件L1告警，但全部通过逐元素公式FP16舍入及dtype推导界；首次失败、资源阻塞与旧partial结果完整保留。峰值allocated13640.731MiB/reserved13734MiB，15次诊断forward合计2.743861秒，非稳定性能计时。完成后停止，无扩样/训练。`summarize_attention_inheritance.py`仅用于历史partial，勿覆盖完成结果；独立复验须指定全新输出目录。

## Phase 3.5：Exit Boundary vs ROI Attention Alignment（离线完成）

补充两案例全景图：[六图与说明](results/phase3_5_two_case_visualization/README.md)。`two_case_attention_viz.py`只读取04132006/15133706缓存，为Q和P−1分别输出4×8全32层ROI热图及配对曲线；四热图共用同一视觉条件概率对数色标并标注原始视觉Mass。此补图按要求先平均Head再条件化，曲线使用同一口径，旧逐Head条件化统计不变。`CUDA_VISIBLE_DEVICES='' python two_case_attention_viz.py --verify-existing`可核验六图及输入哈希；不使用GPU或补Forward。

[报告](results/phase3_5_exit_boundary_viz/report.md)、[全部18张图](results/phase3_5_exit_boundary_viz/gallery.md)、[固定边界统计](results/phase3_5_exit_boundary_viz/fixed_boundary_summary.csv)位于独立 `results/phase3_5_exit_boundary_viz/`。仅配对既有Phase3.2 Physical Exit分数与Phase3.4 Full Attention缓存，48主图和5历史案例分开；没有GPU、模型Forward、网络补采或旧结果修改。

K边界严格对应Full的L(K−1)→L_K；组合图将K置于K−0.5。先逐Head条件化计算ROI面积富集/视觉归一化熵，再平均全部32Head，另记录原始Visual Mass和冻结非ROI Sink份额。固定K16/K24、5000次图级Bootstrap、低质量阈值、最佳K并列规则均保存在analysis_protocol.json。

主48的K16正Gain37图，其中Q/A的ROI下降31/27图；K24正Gain33图中仅4/2图ROI下降，熵却全部上升。固定K16 ROI–Gain Pearson Q/A为+.212/+.323，K24为−.130/−.251，方向不统一。主48在两K均保持Full的33/48正确、无分类翻转。五案例中15133706 K24恢复时Q/A ROI均上升，是明确反例。事后最佳K虽ROI负相关，但熵–Gain也负相关，且有事后选择/层深混杂，不证明退出边界机制或Sink解释。

```bash
cd /home/zsy/ATE/vlm_research
conda activate vlm_research
CUDA_VISIBLE_DEVICES='' python -m unittest test_exit_boundary_analysis -v
# 仅第一次、结果目录不存在时运行；拒绝覆盖已完成目录
# CUDA_VISIBLE_DEVICES='' python exit_boundary_analysis.py
CUDA_VISIBLE_DEVICES='' python exit_boundary_analysis.py --verify-existing
```

新增exit_boundary_analysis.py/test_exit_boundary_analysis.py；9项数值测试通过。636行边界变化、424行8K附表、固定/最佳K/敏感性统计、5套对齐/局部/空间图、3张主统计图和输入输出SHA256均保存。K_best仅从[12,16,20,24,28]事后选择，列出全部并列，K0/8/32仍在完整附表。已停止，不训练、不补Forward、不自动扩展其他阶段。

## Phase 3.4：ROI vs Attention Sink Causal Diagnostic（完成）

[完整报告](results/phase3_4_roi_sink_diagnostic/report.md)、[文献与官方源码核对](results/phase3_4_roi_sink_diagnostic/source_review.md)、[测试记录](results/phase3_4_roi_sink_diagnostic/test_report.md)位于独立目录 `results/phase3_4_roi_sink_diagnostic/`。未改变Phase1–3.3的Python、数据或结果，也未重装环境/下载模型。

- **ROI审计：** 真实CLIP shortest-edge Resize/336 Center Crop，240/240图pixel_values重构逐元素相同，行优先24×24映射经真实CLIP embedding验证。215图两对象可见、24图仅一对象可见、1图均不可见；14张原图/裁剪/Patch三联图保留。几何筛选后固定left/right各24张，不按预测选择；五历史案例单列。主集合有面积和裁剪选择偏差。
- **观察：** 32层×32Head，真实Question组与P−1答案预测行；两个中性提示识别持续热点，第三提示仅检查稳定性。候选在因果评分前冻结，不用ROI、答案或Margin调阈值。候选不是已证明的无关噪声；ROI交集另存。只保存所需行和范数/激活，不保存完整S×S。
- **因果：** 仅L8–15/L24–31、Q-only/A-only，对ROI和非ROI Sink各配三seed精确同数量背景。主48对照全部可用；案例0216546的12个ROI随机组合因背景不足不可用，明确标记。Sink控制空间bin全部匹配；ROI仅约61.61%，存在剩余混杂。保留所有576视觉Token，不是Physical Exit或推理加速。

主Full33/48（left24/24、right9/24）。中层Question ROI富集1.4896，33/48图>1；Answer仅0.9183。中层Q/A→ROI相对各自随机对照额外Margin Drop分别+0.9663/+0.1092 nats；Q条件一原对变错、一原错变对，正确率净值不变。深层Sink视觉内占比没有普遍升高；深层Q/A→非ROI Sink相对随机额外Drop仅−0.0029/−0.0087 nats，无主样本分类收益，也未恢复K24 Physical改善的两条问题。H1仅获有条件支持，H2强解释未获支持，不能把Attention相关性、Mask复合效应和Physical整体效应混为一谈。

GPU前83项CPU测试、最终85项通过；3图7B Smoke审计3072层，目标Attention严格0，预测/Question因果不变量/历史Full/Hook清理通过。无NaN/Inf/OOM。五案例55条件的CPU FP32最终投影检查未改变分类，但小配对效应方向可变；这不是FP32主干复验。正式阶段完成门计时416.054秒，峰值allocated13641.403MiB、reserved13736MiB，只作诊断计时。

```bash
cd /home/zsy/ATE/vlm_research
conda activate vlm_research
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' python -m unittest test_attention_probe test_attention_positions test_knockout \
  test_phase3_watch test_physical_exit test_fixed_k_exit test_directed_suffix test_roi_sink -v
# 仅当结果目录尚不存在时准备一次；已有目录会拒绝覆盖/重采样。
# CUDA_VISIBLE_DEVICES='' python prepare_roi_sink.py
CUDA_VISIBLE_DEVICES=0 python roi_sink_probe.py observe
CUDA_VISIBLE_DEVICES='' python roi_sink_probe.py freeze
# 先离线审阅ROI和Attention观察结果，再进入因果阶段。
CUDA_VISIBLE_DEVICES='' python summarize_roi_sink.py --observations-only
CUDA_VISIBLE_DEVICES=0 python roi_sink_probe.py smoke
CUDA_VISIBLE_DEVICES=0 python roi_sink_probe.py causal
CUDA_VISIBLE_DEVICES=0 python roi_sink_precision.py
CUDA_VISIBLE_DEVICES='' python summarize_roi_sink.py
```

`observe/smoke/causal`相同命令支持逐样本续跑，严格验证父Manifest、配置、图像/模型/旧源码、新实现SHA256和检查点校验和，已完成项跳过；禁止更改绑定实现来继续既有检查点。`roi_sink_common.py`、`roi_sink_probe.py`、`prepare_roi_sink.py`为已运行冻结代码。完整汇总还要求先生成precision检查结果。`precision`是五案例固定55条件的额外诊断，没有断点续跑需求。

完整逐样本CSV/JSON1749行（含12不可用案例组合），Head统计108544行在 `head_attention_statistics.csv.gz`（标准gzip CSV），并保存原始选定Attention行、ROI/候选清单、分组汇总、配对统计、曲线/Head热图/Overlay、数值检查和资源日志。首次观察调用的资源预检API错误发生于加载前，修复与失败日志保留，不影响冻结数据及后续成功实验。本轮停止于Phase3.4，不扩大240图全组合、不训练Gate/Router、不进行Decode；自动GPU轮询仍关闭。

## Phase 3.3：Directed Suffix Visual Attention Knockout（完成）

结果位于独立的 `results/phase3_3_directed_suffix/`；[完整报告](results/phase3_3_directed_suffix/report.md)、[测试记录](results/phase3_3_directed_suffix/test_report.md)、[配对汇总](results/phase3_3_directed_suffix/summary_by_condition.csv)。沿用Phase3.2冻结的同240张图片和Prompt/候选顺序，无重采样、无结果筛选。

72项CPU回归、8图真实7B Smoke及全部240图验收通过，3120行完整结果。Full187/240；K16 Q-only/A-only/Q+A/Global分别189/184/188/186，Full-correct翻错1/3/2/3条；K24分别189/187/189/189，均无Full-correct损失。三个历史Global翻错中，两个单路径各保留2条，Q+A保留1条；A-only另有2条Global未翻错的损失，因此不能概括为单路径普遍更安全。A-only Question hidden精确Full仍可答案翻错，QA与Q-only Question hidden精确相同仍有答案差异。

Global与历史Physical均通过既有容差；`04993832` K16的历史+0.0078125变为本轮0平局，造成Global比历史Physical少答对1条，报告单独保留，不能视为新的机制证据。正式1083.881秒、allocated峰值13778.161MiB；全Attention Smoke峰值14467.481MiB。所有条件保留602–613完整Token序列，不是加速实验。

- K=[16,24,32]，从L_K持续到L31；K32为无干预控制，另有原生Full。
- Q-only：真实Question Query→全部576视觉Key；A-only：**P−1**预测首个答案Token的Query→视觉Key，绝不是已输入答案P。
- Q+A取二者并集；Global使用原Phase3的所有合法后图像文本Query。行是target Query，列是source Key。
- 继承既有Causal Mask Clone和安全Hook清理，所有视觉Token/原始位置/32层计算保持不变；不是Physical Exit或加速实验。
- 单Token left/right仍按完整候选序列条件logP评分，Margin>0才算正确，平局算错。Question Drift取最终RMSNorm状态，逐Token数据在检查点中。

```bash
cd /home/zsy/ATE/vlm_research
conda activate vlm_research
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest test_attention_probe test_attention_positions test_knockout \
  test_phase3_watch test_physical_exit test_fixed_k_exit test_directed_suffix -v
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python directed_suffix_probe.py smoke
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python directed_suffix_probe.py run
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python summarize_directed_suffix.py
```

续跑时使用相同命令：先校验冻结Manifest、图片/模型/旧阶段源码与新实现SHA256、配置和检查点校验和，已完成样本跳过，拒绝覆盖、重复或错配。必须先通过Smoke完成门才能正式运行。只有全部240图/3120行完整时，汇总程序才生成全队列图表。不要为恢复而修改绑定源码或选择性删除检查点。

三张核心图为 `accuracy_and_flips.png`、`margin_drop_distribution.png`、`question_drift_answer_margin.png`。CSV/JSON记录全部样本及固定Full-correct子集；`historical_three_cases.csv`逐一对照用户指定的三个K16历史翻错问题；`global_vs_historical_physical.csv`单独记录FP16容差内可能存在的近零Margin分类差异。`compute_diagnostics.csv`仅为含Hook/检查开销的诊断计时，无独立预热重复、不宣称加速。

本轮止于Phase3.3，不自动扩展ROI、Router、KV Cache、Decode、训练或其他数据集；自动GPU轮询保持关闭。

## Phase 3.2：240 图 Fixed-K Physical Exit（完成）

[完整报告](results/phase3_2_fixed_k_exit/report.md)和[逐 K 汇总](results/phase3_2_fixed_k_exit/summary_by_k.csv)已保存。新增 240 张独立图片，left/right 各120，排除上一轮12图；种子20261029，先冻结 Manifest 再运行，无结果筛选。复用未修改的 `physical_exit.py`、加载器、Processor、真实 Question 定位和完整候选序列 logP；K=[0,8,12,16,20,24,28,32]。

Full=187/240（77.9167%）；K16=187/240，但原对变错3、原错变对3；K20=186/240（原对变错1）；K24=189/240，保留全部187条Full-correct。平均 Margin Drop 为负仍有退化尾部；left/right 存在明显偏好。Question Drift 在测试网格中随K增大而下降，较小Drift不保证不翻错。上游数据已筛选，候选准确率不等于自由生成GQA准确率，不支持通用安全退出层结论。

先通过62项CPU回归，再通过24图GPU Smoke、原12图×8K Mask/Physical等价性，最后完成240图正式配对实验。K32在全部240图与原生Full的有效文本hidden/logits精确一致；Smoke与正式运行的分数精确一致。原容差未放宽，环境/模型/第三方库及旧阶段代码结果未改变。正式阶段412.775秒，峰值allocated13776.610MiB、reserved13860MiB；未做独立预热重复，耗时仅诊断，不宣称稳定加速或Decode加速。

新增入口：`prepare_fixed_k_data.py`、`fixed_k_exit.py`、`summarize_fixed_k.py`、`test_fixed_k_exit.py`。结果目录 `results/phase3_2_fixed_k_exit/` 保存冻结数据/版本/权重与代码哈希、1920行CSV/JSON、逐Token漂移和逐层长度、四图、翻转/尾部个案及资源日志。

```bash
cd /home/zsy/ATE/vlm_research
conda activate vlm_research
# 只对尚不存在的新结果目录准备一次；已有冻结目录会拒绝覆盖。
# python prepare_fixed_k_data.py
CUDA_VISIBLE_DEVICES='' python -m unittest test_attention_probe test_attention_positions test_knockout test_phase3_watch test_physical_exit test_fixed_k_exit -v
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python fixed_k_exit.py smoke
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python fixed_k_exit.py validate
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python fixed_k_exit.py run
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python summarize_fixed_k.py
```

中断恢复：直接重跑相同阶段，先核验已完成检查点再跳过，未完成样本重做；原子文件和进程锁防止重复计数。绑定涵盖冻结Manifest、图片哈希、候选和ID、代码/安装源码、模型权重及版本；错配、损坏、重复K或不完整记录会拒绝。不要手改Manifest或删除已完成检查点来选有利样本；代码变化后须使用本实验提交的代码恢复。GPU预检失败会保存原因，不终止其他进程。自动轮询保持关闭。

本阶段仅固定K、无缓存Prefill/Teacher-forcing。完成后停止；不自动进入Phase3.3，不训练Router、不做Decode或更多数据集。

## Phase 3.1：Physical Visual Exit Pilot（完成）

已完成 [12 张固定 GQA 图片的实测报告](results/phase3_1_physical_exit_pilot/report.md)。图片全部来自服务器已有 GQA 目录，没有下载图片包；种子 20261028，left/right 各 6 条，12 张不同图片，无结果筛选。来源与 SHA256 见 [固定数据 manifest](configs/phase3_1_data/manifest.json) 和 [官方参考](configs/phase3_1_data/SOURCES.md)。参考 Cross-modal Information Flow 与 DyVTE，本实验不主张首次提出 Visual Exit。

K=[0,8,16,24,32]，在进入 L_K 前实际切除 576 个视觉 hidden states，随后 LLM 序列从 603–608 缩至 27–32；保留原始文本逻辑位置、RoPE cos/sin 和全部非视觉文本，裁剪 causal mask 及输出映射。仅无 KV Cache 的 Prefill/Teacher-forcing，不是 Decode 加速实现。Full 使用原生 Forward，Mask 复用 Phase 3，Physical 使用独立 Decoder 循环，不修改第三方源码或模型方法。

全部 Mask/Physical 比较通过预先固定容差：最大文本 logits 差 0.0859375、cosine distance 3.273e-5、Margin 差 0.015625 nats；全部 K32 与原生 Full 的 hidden/logits 完全一致。12 项新增 CPU 测试、合计 56 项回归测试及 pip check 通过。Full 候选正确 10/12；Physical K0/8 为 6/12，K16/24 为 10/12，Full-correct 子集分别为 6/10 和 10/10。K16 的全部样本 Question mean cosine drift≈0.05563，不能把候选正确性保持当作表征或语义完全不变。

[逐样本 CSV](results/phase3_1_physical_exit_pilot/per_sample_results.csv) · [完整 JSON 与位置/层长度](results/phase3_1_physical_exit_pilot/per_sample_results.json) · [Question 漂移图](results/phase3_1_physical_exit_pilot/question_representation_drift_vs_K.png) · [Answer Margin Drop 图](results/phase3_1_physical_exit_pilot/answer_margin_drop_vs_K.png)。JSON 同时保留全部12条和本次 Full 正确的10条，未丢弃原本错误样本。

完整实验 50.527 秒，峰值 allocated 13774.966 MiB、reserved 13862 MiB。时间包含加载/诊断/绘图；未做合理预热与重复计时，各模式还有观测存储开销差异，不宣称稳定加速收益。模型/环境和 Phase 1–3 结果未改动，GPU 0 已释放，自动轮询未恢复。当前在 Phase 3.1 停止，无 Router、Directed Attention 完整扫描、逐生成步退出、ROI 或滑动窗口实验。

CPU 测试（不需要 GPU）：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsy/.conda/envs/vlm_research/bin/python -m unittest test_physical_exit -v
```

本次实际执行命令（已完成的目录会拒绝覆盖，不要求原地重跑）：

```bash
# 取得指定官方仓库 revision 后，先固定数据；已有 manifest 不可替换
python prepare_physical_exit_data.py --source-repo /path/to/cross-modal-information-flow-in-MLLM \
  --image-root /data/wdy/datasets/MMEB-V2/image-tasks/MMEB/GQA \
  --output-dir configs/phase3_1_data
# 运行前固定选择哈希、容差与协议，再执行单卡离线实验
python physical_exit_pilot.py --freeze-only
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsy/.conda/envs/vlm_research/bin/python physical_exit_pilot.py
```

程序会验证图片哈希，缺图或变化时停止、不换样本。Mask/Physical 差异超出固定容差时保存 failure.json 并停止，不能解释成视觉机制或用 Mask 结果替代物理退出。

## Phase 3：逐层 Visual Attention Knockout（小规模 GPU 实测完成）

2026-10-27 18:17（Asia/Shanghai），应用户要求取消自动轮询，改为用户手动检查 GPU。已执行 `phase3_watch.py --disable`，确认状态为 `disabled`，本任务 cron 条目已移除，其他定时任务保留。取消时尚未自动启动 Phase 3；后续不会自动检查 GPU 或触发实验。轮询代码保留，但不得自行重新启用。

状态记录：`/home/zsy/.local/state/ate-phase3-watch/status.json`；[最新运行入口](results/phase3_knockout/LATEST.md)。

2026-10-27 23:10，用户明确要求继续后，在 GPU 0 手动执行现有全部 9 条 QA × 8 个 cutoff，真实 7B 正确性与 K=32 基线等价通过（logits 最大绝对差 0），44 项回归测试通过。峰值 allocated 14344.958 MiB，正式 Forward 峰值 allocated 13649.285 MiB，reserved 峰值 14544 MiB；包含加载/校验/绘图的完整实验 42.201 秒。没有更改环境、权重或旧阶段结果。

实测数据在 [manual_gpu_20261027_2310](results/phase3_knockout/manual_gpu_20261027_2310/phase3_report.md)，分析与解释边界见 [Phase 3 报告](results/phase3_knockout/phase3_report.md)。描述任务 K=16 的 Δmean NLL=0.509531，K=28 为 0.002213，但 stop Token 在 K=28 仍增加 0.435381 nats；各样本存在明显非单调变化。没有确定通用融合饱和深度，不能据此宣称安全删除视觉 Token。新增标注仍为 AI 复核，样本实际仅一张街景及其镜像。完成后停在 Phase 3，自动轮询保持关闭。

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

离线汇总已完成运行（不加载模型、不使用 GPU）：

```bash
python compare_knockout.py \
  --run-dir ./results/phase3_knockout/manual_gpu_20261027_2310
```

输出跨样本 NLL 热图、颜色/描述逐 Token 对比、真实资源汇总 JSON，以及与历史 Phase 2.1 同 Query Attention 的对照 CSV。后缀各层 Attention Mass 求和仅作描述，不是概率或单层因果效应。准备阶段原报告完整保存在 `results/phase3_knockout/phase3_preparation_report.md`。

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
