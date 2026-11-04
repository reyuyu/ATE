# ATE-FullJoint：VQAv2快速验证

**COMPLETE**：60题×5条件=300条真实Greedy答案；20题×5条件各3次预热、5次正式计时。仅VQAv2，不扩样、不训练。

## 合体定义与数据
C0为真正Native Full。C1严格调用冻结Fast-P2：L20前删除视觉Hidden，P−1/Decode不直接读图，文本KV原地追加。C2仅增加Conditional Writer；C3仅增加物理A2；C4合体两者。
Conditional Writer严格使用644834d3的Fixed-Q-Conditional-Assistant-15：所有原Question Subword含标点；首词what/which或Question Token数≥15时加原ASSISTANT的完整4个Subword。其他R、P−1禁读图。仅L0–L19存在视觉写入。不是旧Semantic-k。
A2在L2–L5只投影保留Query的Q/o_proj并计算其Attention；K/V仍覆盖全部Prefill输入，Image/P−1输出置零，Decode不执行该四层Attention。Norm、Residual和全部MLP保留。最终Writer Mask先构造，再重新取A2 active_mask；没有绕过Conditional Mask。
L20之后物理删除576个视觉Hidden，持久KV只存需要的文本；被彻底跳过的四层不存无用KV。其余层Decode使用文本KV，保留原始逻辑位置和RoPE，无历史KV拼接或重复拷贝。
60题从已有冻结VQAv2-val2014本地500题按输入稳定哈希选取，60张不同图片；不是整个验证集的无偏随机样本，也不是新的独立确认集。官方10答案逐项对照原Annotations ZIP，图片SHA、Prompt IDs和输入张量哈希与历史记录一致。
五条件均新生成，Greedy最多32 Token、EOS停止。同一RTX3090、FP16/eager、Transformers4.49.0、同一权重和Processor。沿用Fast-P2的logits_to_keep=0，在同形状下严格检查Native/No-op及官方generate一致；最新规则仅复用接收者定义，不混入其历史答案或不同LM Head形状结果。

## 官方共识评分
|条件|软Accuracy %|ΔNative pp|EOS/60|截断|空答案|
|---|---:|---:|---:|---:|---:|
|C0|85.500|+0.000|60|0|0|
|C1|80.667|-4.833|60|0|0|
|C2|79.667|-5.833|60|0|0|
|C3|82.833|-2.667|60|0|0|
|C4|79.667|-5.833|60|0|0|

官方VQAEval标点、数字、冠词、缩写归一化加十标注leave-one-out共识；不是单字符串匹配，也不是简单min(count/3,1)。1/2/3个匹配分别得0.3/0.6/0.9。

|比较|Δ软分 pp|配对95%CI|软分下降/上升|满分丢失/获得|
|---|---:|---|---|---|
|C4−C1|-1.000|[-5.000, +2.000]|1/1|1/1|
|C4−C3|-3.167|[-8.333, +0.667]|3/1|2/1|
|C4−C0|-5.833|[-12.167, -0.667]|5/1|3/1|
|C2−C1|-1.000|[-5.000, +2.000]|1/1|1/1|
|C3−C1|+2.167|[-3.333, +7.667]|1/3|1/2|

20,000次Image SHA整簇配对Bootstrap，题目等权，固定seed=20261105。全部区间为探索性，不作非劣效/等价判断；VQAv2软分的下降/上升不能全部称为二值正确翻转。完整0↔1翻转另见CSV。

## 同卡真实性能
|条件|Prefill ms|GPU TTFT ms|E2E ms|Decode ms/token|vs Native加速|vs C1加速|vs C3加速|
|---|---:|---:|---:|---:|---:|---:|---:|
|C0|162.999|207.088|280.386|71.690|1.000×|0.811×|0.778×|
|C1|109.664|153.476|227.099|71.099|1.233×|1.000×|0.958×|
|C2|109.606|153.648|228.305|71.237|1.224×|0.994×|0.952×|
|C3|105.241|149.348|218.801|65.913|1.286×|1.044×|1.000×|
|C4|105.299|149.235|218.568|65.986|1.281×|1.041×|0.999×|

时延为逐题5次正式重复中位数再跨20题取中位数；加速为同题时延比的中位数，不必等于两列总体中位数之比。所有条件交错、每请求独立冷KV；Vision每次原生重算。图片I/O、CPU Processor及输入分区/Writer位置准备统一不计时；控制器/实际Mask/Cache构造计入同步墙钟E2E。
GPU TTFT沿用冻结首Forward CUDA Event口径，为Vision+Language Prefill，不包含全部CPU请求准备，不冒充完整墙钟首Token时延。自然EOS的输出长度变化可能改变E2E；同完整Token序列子集的配对加速和样本数另见latency_summary.csv。没有固定16/64测试，也没有缓存复用视觉特征。

|条件|Prefill矩阵TFLOPs|vs C1减少%|估算Prefill TFLOP/s|Live KV MiB|容量KV MiB|Peak allocated MiB|
|---|---:|---:|---:|---:|---:|---:|
|C0|8.2076|-58.154|50.446|304.000|304.000|13923.8|
|C1|5.1865|+0.000|47.296|16.000|30.750|13649.3|
|C2|5.1865|+0.000|47.395|16.000|30.750|13649.3|
|C3|5.0087|+3.423|47.663|14.000|26.906|13645.5|
|C4|5.0087|+3.423|47.669|14.000|26.906|13645.5|

实际Linear/Conv/QK/AV/bmm张量形状审计，2 FLOPs/MAC，包含Vision矩阵计算的独立项；不包含Norm/Softmax/RoPE等非矩阵操作。估算TFLOP/s为算术量除对应CUDA阶段时间，不是硬件计数器。硬件TFLOPS/DRAM计数器N/A，沿用ERR_NVGPUCTRPERM已知权限限制，不伪造硬件实测。
Conditional Writer仍是Dense Hard Mask，本身不省Dense矩阵FLOPs。逐题严格验证C2与C1、C4与C3的Prefill矩阵量相同，C4与C3的KV容量相同；不能把写入连接预算减少当成物理加速。实际省算来自L20删除视觉行、文本KV以及A2局部Query跳算。Live KV可随自然生成长度变化，不是额外的Token选择压缩；KV Live和容量分别计量，不能当作整卡显存减幅。

## 直接回答与边界
1. 四项已经组合在C4同一真实生成路径，三题Smoke通过；Dense参考仅验证、不参与速度排名。
2. C4软Accuracy=79.667%。
3. C4−C3=-3.167pp，配对95%CI [-8.333, +0.667]。
4. C4相对Native E2E=1.281×，时延减少21.95%。
5. C4相对C1/C3 E2E分别为1.041×/0.999×。
6. C4相对C1的Prefill矩阵量减少3.423%；Live KV为14.000MiB。C4与C3的Token选择不额外节省Dense矩阵计算。
7. 本轮不建议把完整C4作为优先扩规模候选：Conditional Writer相对C3没有额外矩阵量/KV容量/时延收益，软分差值为-3.167pp，且CI仍很宽。若下一轮决定扩大验证，优先复核更简单的C3是否稳定；这不是在本轮自动扩样或认定其非劣效。C4当前不能称为质量保持的完整加速方案。

## 正确性与异常
7项CPU测试、3/3真实GPU Smoke通过。Native/No-op及官方generate全返回有效Logits严格torch.equal；C1与冻结Fast-P2、C3与旧A2严格一致。C4与全Dense D1+I+A零输出参考，以及同物理交接的零输出参考，使用旧FP16 .10/.003 Logits、Hidden RMSE≤.01及Cosine距离≤1e-4，未放宽。
Smoke记录真实禁用概率为0、A2目标Q/o_proj行数、完整K/V依赖、32层MLP正常、L20后实际文本行数、KV逻辑位置及Hook正常/异常清理。所有Checkpoint哈希、输入、完整序列及官方软分重新核对。
EOS、截断、空答案和重复3gram启发式均保留在CSV，不删负例。重复启发式不等于模型失效。逐题每条件原始Shape和计时全部保留在Checkpoint。
正式GPU矩阵耗时5.39分钟（不含模型加载/Smoke/CPU冻结/汇总）；完整操作时间与文件/提交记录另存。
只占用启动前空闲的物理GPU2，GPU0/1外部任务未干扰，GPU3保持空闲；进程退出后保存gpu_release.json。main未修改。
