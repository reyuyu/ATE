# 发给另一台服务器 Codex 的交接提示词

以下代码块可直接复制为新会话的完整提示词：

```text
你现在接手我的 ATE 视觉语言模型 Attention / Visual Exit 研究。请实际读取GitHub文件，不依赖之前聊天，也不要从第一阶段重新开始。

仓库：https://github.com/reyuyu/ATE
已有实验代码基线commit：bcc3c3da2ff7d416cd988c83264fd3ffc18afaed；请获取main最新版本，必须包含 vlm_research/handoff/。
旧服务器工程：/home/zsy/ATE/vlm_research。新服务器目录可以不同，请自己检查当前系统、已有检出、Conda、驱动、GPU、磁盘、网络，不假定旧路径可用。

先按顺序完整阅读：
1. vlm_research/handoff/README.md（全项目状态、统计口径及已完成单案）
2. vlm_research/handoff/DATASET_SETUP.md（完整数据集/模型下载与路径适配）
3. vlm_research/results/attention_inheritance_15133706/completed_20261028_2245/report.md、validation.json；根目录旧partial不代表当前状态
3a. 最新 results/attention_sink_steering_case_15133706/README.md、report.md、protocol.json、per_condition_results.csv、paired_control_comparison.csv、heatmap_gallery.md；代码attention_sink_steering.py/attention_sink_steering_case.py/summarize_sink_steering_case.py
4. vlm_research/attention_inheritance.py、attention_inheritance_pilot.py、test_attention_inheritance.py
5. vlm_common.py、roi_sink_common.py、roi_sink_probe.py的inputs_and_history/score、attention_position_probe.py相关定位和physical_exit_pilot.py候选评分
6. Phase3.2/3.3/3.4/3.5的manifest、关键CSV、报告；交接README已列出准确入口。

数据恢复是必要任务，不能只恢复一个案例来声称统计实验已可接续：
- 原图不在GitHub。按DATASET_SETUP.md，从TIGER-Lab/MMEB-V2下载image-tasks/mmeb_v1.tar.gz，先固定和保存HF revision，再解出完整MMEB/GQA图片目录；不需要视频/训练数据。
- GitHub已有官方ChooseRel CSV（964题/785图）及冻结样本。当前统计基线是Phase3.2的240张独立图（left/right各120）+Phase3.1的12张验证图，合计252张；Phase3.4固定48主图和5历史案例来自同240图。
- 运行handoff/verify_recovery.py校验全部252张图的SHA256、尺寸/解码和仓库数据指纹。缺图/哈希不符必须记录，禁止换图、重采样、改Manifest伪装续算。完整GQA目录同时保留，方便后续统计实验；不自行开展其他数据集。
- 优先复用已存在模型/环境。确实缺少时只恢复同一revision的llava-hf/llava-1.5-7b-hf与原固定依赖，详见下载页，不盲目升级。
- 原环境：Python3.11.18、torch2.6.0+cu124、torchvision0.21.0+cu124、transformers4.49.0、FP16/eager、32层/32Head/576视觉Token，权重revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9。
- 新服务器只适配路径：显式load_model(model_path=本机目录)、运行时样本image_path映射、CPU子进程使用sys.executable。新建remote入口/测试launcher，不重写冻结Manifest、configs/model.json或历史结果，记录新代码hash与路径映射。

已完成单案例实验（读取结果，不默认重跑）：GQA 15133706（Girl/Boy）
Question: Is the young girl to the left or to the right of the boy?
Gold=right，候选顺序left/right不变。
在L24～L31，仅修改Answer-prediction P−1→576视觉Keys的Attention：逐Head继承原始Full的L23或L24视觉条件分布；λ=[0,0.5,1]，共6条件。
替换位于Softmax后、乘当前层V之前：m*((1−λ)*a_current/m + λ*p_ref_full)。保持当前视觉总Mass、非视觉连接和其他Query，λ0跳过赋值。用原实例局部包装；不改第三方源码、不用输出插值/Mask/Physical删除冒充该干预。
此输入Prompt长度603、P−1=602、候选位置603、视觉位置5..580；必须用真实input_ids和历史映射验证，不能选已经输入的答案Token作预测Query。

真实状态，不得误报：
- 历史统计阶段Phase1～3.5已完成，结果可直接复用，无需重跑240图来接手。
- 后续Sink去偏/ROI增强单案31条件也已完成：108回归/396审计通过。A去偏变差，ROI增强仍错、深层Sink抑制未转正；仅早层L2～L4 Sink×.5/.1转正+.0078125/+.1015625。三背景对照未转正；注意交集Patch364的重分配和弱Margin/FP16单例限制，不默认重跑、不据此扩样本。
- 单案已完成101项CPU回归、真实7B六条件与96层级审计。L23 λ0/.5/1 Margin为−.03125/+.0234375/+.0859375，L24为−.03125/−.03125/−.0390625。
- Native Full−.03125；Physical K24+.0390625仅为历史Phase3.2对照。本轮没有重跑Physical。
- 两source λ0全部P−1词表logits精确Full、Q hidden所有条件精确Full、非目标权重局部不变、收尾Native Full精确、库源码未改，无NaN/Inf/OOM。
- 首次因小Mass FP16固定条件L1阈值停止及资源阻塞历史仍在根partial；逐元素舍入和dtype界修正现已通过全部7B六条件，不修改干预公式，8审计超过原固定告警仍保留。
- L23非零λ转正但right绝对概率下降；该FP16弱Margin单案不证明通用能力改善或Sink/Physical机制。不要把根旧partial当作当前未完成状态。
- 完成运行峰值allocated13640.731MiB/reserved13734MiB；15次诊断forward合计2.743861秒，不宣称稳定速度。自动轮询仍关闭。

执行：先恢复/校验数据与环境→路径适配→CPU回归→读取已完成报告与原始分数。不要自行重跑7B；如我明确要求独立复验，再手动GPU预检（只暴露单卡、至少16000MiB空闲）→Native Full和六条件→末尾原生Full恢复/源码未改检查。后续统计目标需要按本次新会话明确授权。
所有新结果写入results/attention_inheritance_15133706/remote_<server>_<timestamp>/之类全新目录。原runner逐条件保存但不能自动从已配置目录续跑；失败保留，下一次用新目录，不能把不兼容运行拼成完成状态。
要求λ0对同机Full精确、行和/Mass通过现有容差、非目标连接bitwise未改、真实Question hidden因果不变、候选P−1输出一致、无NaN/Inf/OOM、临时包装清理。遇历史Full/cache跨机bitwise差异先保存数值差/环境并调查，不静默删断言或改容差。
输出六条件CSV/JSON、原始logits、审计、真实耗时/显存、Margin对比图和简短报告。观察和机制猜测分开；近零FP16 Margin和事后选例不支持Sink有害/解释Physical整体效应的结论。

后续统计工作也要能继承：
保留原240图Manifest、48主图/5案例隔离、完整候选序列logP（不加EOS）、Margin>0才正确（tie错）、Drop=Full−干预、Gain=−Drop、最终RMSNorm Q drift。所有题号按字符串保留前导零。
直接复用已有Phase3.2的1920行、Phase3.3的3120行与Phase3.4的53图Attention缓存；不可假称全部240图都有Attention。
NPZ attention轴[32,32,2,576]，group0是真实Q行均值，group1=P−1。逐Head条件化再平均与先Head平均再条件化不可混用。
后续用户要求统计扩展时，先冻结新协议和结果目录，再同图配对，报告overall/left/right正确率、两种翻转、Margin分布尾部、Q drift；Bootstrap按图，不按Head或层扩大样本数。不得把5历史案例混入48主统计或按预测筛有利图。当前没有240图继承runner或分数，不要声称已有。
单案已完成；根据我在新会话给出的统计目标继续，未给出目标时只完成恢复和验收。不要自行训练Router/Gate/GRPO、Decode/KV Cache或扩其他数据集。

我已授权按原项目流程提交并推送GitHub：保存真实状态，完成后更新README/report，commit/push并核验远端hash和文件；不要提交权重/原图全集/环境/凭据，不强推。若没有写权限，保留本地commit/明确缺少何种权限，不声称已上传。不要重复询问已经明确授权的常规操作。
请先给出核对到的仓库版本、数据/环境缺项和实际恢复步骤，然后继续执行能够完成的部分，不要只给我计划。
```
