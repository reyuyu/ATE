# Qwen3-VL 首答案 Token 视觉 Attention 敏感性

## 实际完成
240/240题，六任务各40；完整36层，共8,640/8,640个单层 Image OFF 条件。Question ID、Image ID和图片SHA256与LLaVA RQ2.1的240题全部一致。原始图片与问题复用，输入使用已经冻结的Qwen官方Chat Template，不冒充LLaVA的Token位置。

- 模型：Qwen/Qwen3-VL-8B-Instruct，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；BF16/eager；Transformers 4.57.6。
- 物理GPU0/2/3，每卡80题；未使用GPU1。冻结源码Commit：`e4b555e772caf34b0991f88e4065abee9fe32998`。
- 正式运行最长Shard：5.08分钟；各卡Smoke耗时见integrity.json。
- 仅原始Prefill P−1的目标层到实际Image Key施加Softmax前`-inf`，其他Query、MLP、视觉Token和DeepStack保持原生。没有自由生成、训练或其他消融。

## 核心观察
- Mean Signed LogP Loss最高的五层（nat）：L26=0.010851, L25=0.001554, L27=0.001352, L29=0.001113, L34=0.000417。
- Mean Absolute LogP Change最高的五层（nat）：L26=0.018743, L23=0.009431, L24=0.008039, L21=0.006660, L16=0.006245。
- 全部8,640条干预记录中，仅L26有1题发生首Token翻转（1/240，0.42%；CompareAttr，Question ID 1522541）；其余35层均为0。
- Signed均值允许负值；Absolute先逐题取绝对值再平均，不是Signed均值的绝对值。全词表FP32 log_softmax，目标固定为本题Native Full首Top-1，不使用Gold答案替换。

六任务的绝对变化峰值并不在同一层；L26的总体峰主要集中在CompareAttr，不能称为所有任务共同的敏感层。

| 任务 | Mean Absolute Change峰值层 | 该层均值（nat） |
| --- | --- | --- |
| ChooseAttr | L23 | 0.007943 |
| ChooseCat | L33 | 0.004066 |
| ChooseRel | L26 | 0.008170 |
| CompareAttr | L26 | 0.080146 |
| LogicalObj | L34 | 0.006980 |
| QueryAttr | L23 | 0.015027 |

## 正确性与动态视觉位置
实际Image Key由当前Processor展开后的image_token_id及image_grid_thw/merge_size交叉核验，不假定576个Token。每个正式干预均从原生Attention返回值检查全部Head的P−1→Image概率为0；保存实际Token数、Mask改变的元素数与BF16行和误差。

各GPU先通过真实Smoke：No-op和Full全部有效Prompt Logits逐元素精确一致；复用视觉特征与Native Full全部有效Logits精确一致；L0/L2/L17/L35的目标Mask有效，其他Prefill Query的Attention输出、层输出与最终Logits精确一致；更早层不变；三个DeepStack特征与实际注入精确核验；输入和权重不变；正常及异常退出Hook均清理。全部GPU通过后才开始正式扫描。

首次Smoke将全部位置LM Head投影和仅末位置投影跨形状比较，触发BF16精确一致断言，正式扫描未启动；原始失败日志保存在attempt0_projection_shape_check。修正为各自相同投影形状的Native/No-op严格逐元素比较，不放宽容差。两种投影形状的实测差异单列在validation.json，正式Full与Image OFF始终使用相同的仅末位置投影。

每题只计算一次原生视觉编码，将最终视觉特征和全部三个DeepStack特征一起原样克隆复用到各层条件；没有删除或遗漏DeepStack。每次使用独立无Cache Prefill，防止条件间KV污染。

## 解释边界
这是240题首Token的探索性直接路径扫描，不测量完整答案Accuracy。Image路径低敏感不能证明不需要视觉信息，其他Token残差及DeepStack仍可能携带视觉信息。Signed损失、绝对变化和Top1翻转为不同指标，负值/抵消可能使Signed曲线弱化；本轮重新计算全部数据，不复用旧图或旧消融结果。

## 文件
`per_question.csv`为8,640行逐题逐层数据；`layer_summary.csv`为36层总体统计；`task_layer_summary.csv`为六任务统计。`overall_sensitivity.png/pdf`和`task_*.png/pdf`均由本轮真实数据直接绘制，无平滑插值。`sample_manifest.json`保存全部动态输入映射及三卡分片，`shard_*/validation.json`、日志及`integrity.json`保存验证和执行记录。

Qwen环境本身未安装Matplotlib，首次汇总因此在绘图处停止；日志原样保留。最终仅在CPU汇总进程追加已有ate-vlm-research绘图库路径，模型包仍优先使用Qwen环境，未安装、升级或修改共享环境。复现方式和Matplotlib来源记录在plot_environment.json。

## GPU释放
三个Worker均正常退出，全部PID已不在GPU进程列表。物理GPU0/2/3最终均为27 MiB、0%利用率；GPU1的外部任务仍正常运行。见gpu_release.json。本轮到此结束，没有增加额外条件。
