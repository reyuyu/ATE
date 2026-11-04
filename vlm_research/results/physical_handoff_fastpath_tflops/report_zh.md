# Visual Handoff Fast Path：60题质量与性能Pilot

## 范围与有效性

历史原生数值验收仍为PARTIAL，未被覆盖或改写。本轮只使用原冻结60题（每任务10题），不构成非劣效验证。
所有路径原精度/eager，无FP32稳定Linear、无Canonical Tile、无训练。Native Full保留直接视觉读取，D1及Fast的写入依赖与其不同；物理优化的主要功能配对是D1。
Native/No-op全部有效Logits严格一致；新Dense与旧Debug D1逐位一致；两模型各3题Mask/位置/Cache/Hook真实GPU Smoke通过。
每题每条件5次预热、20次计时；条件按轮次交错。E2E含每请求Mask/KV设置和冷Cache，不含磁盘IO/CPU图像预处理；不缓存视觉特征，每次都执行原生视觉编码器。
固定16/64为工程测试，可能越过EOS，不能代替自然短答案时延。Profiler和FLOPs形状采集与时延重复完全分离。

## 核心结果

|模型|条件|Accuracy|自然E2E ms|语言Prefill ms|KV Live MiB|KV Capacity MiB|总Peak Allocated MiB|
|---|---|---:|---:|---:|---:|---:|---:|
|llava|Native Full|83.33%|278.20|163.65|306.50|306.50|13931.6|
|llava|D1 Dense|85.00%|281.14|164.81|306.50|321.50|13959.4|
|llava|Fast-P1|86.67%|227.85|111.66|198.50|213.50|13836.5|
|llava|Fast-P2|86.67%|226.58|110.91|18.50|33.50|13653.5|
|qwen|Native Full|81.67%|255.42|106.81|29.11|29.11|16867.8|
|qwen|D1 Dense|81.67%|253.18|104.37|29.11|33.33|16868.0|
|qwen|Fast-P1|81.67%|251.94|103.54|19.48|23.70|16867.9|
|qwen|Fast-P2|81.67%|251.38|103.70|4.36|8.58|16867.9|

各列为60题各自20次重复中位数的跨题中位数；不把跨题中位数当成总批次耗时。p10/p90及逐题明细见CSV。

## FLOPs、TFLOPS与硬件计数器

`HARDWARE_TFLOPS_UNAVAILABLE`：Nsight Compute实际探测返回ERR_NVGPUCTRPERM，保留原日志。DRAM读写、GB/s、L2命中、SM/TC利用率、硬件浮点操作计数均未测得，CSV标N/A。未修改系统权限，未将SM百分比或理论峰值换算为实测TFLOPS。
Estimated Algorithmic TFLOPS来自独立执行形状账本的Linear、Conv、QK和AV FLOPs（每MAC=2）除以对应CUDA Event阶段时间；包含原生Vision、QKV、MLP、投影、LM Head，排除Norm/Softmax/激活/RoPE等非矩阵操作，不是硬件Achieved TFLOPS。

- llava Fast-P2相对Dense：语言Prefill估算矩阵FLOPs减少36.5%；语言Prefill加速1.486×；自然EOS E2E加速1.241×；相对Native E2E为1.228×。KV Live节省288.00MiB；总Peak Allocated下降2.19%。
- llava fixed16 Decode：Native Full 67.951ms/token；D1 Dense 68.135ms/token；Fast-P1 69.211ms/token；Fast-P2 68.401ms/token。
- llava fixed64 Decode：Native Full 70.087ms/token；D1 Dense 70.246ms/token；Fast-P1 70.136ms/token；Fast-P2 69.565ms/token。
- llava fixed64 prefill估算TFLOPS：Native Full 48.728；D1 Dense 48.456；Fast-P1 45.122；Fast-P2 45.376（非硬件计数器）。
- llava fixed64 decode估算TFLOPS：Native Full 0.193；D1 Dense 0.193；Fast-P1 0.192；Fast-P2 0.190（非硬件计数器）。
- qwen Fast-P2相对Dense：语言Prefill估算矩阵FLOPs减少33.3%；语言Prefill加速1.007×；自然EOS E2E加速1.007×；相对Native E2E为1.016×。KV Live节省24.75MiB；总Peak Allocated下降0.00%。
- qwen fixed16 Decode：Native Full 96.288ms/token；D1 Dense 95.137ms/token；Fast-P1 94.765ms/token；Fast-P2 94.136ms/token。
- qwen fixed64 Decode：Native Full 96.927ms/token；D1 Dense 95.767ms/token；Fast-P1 95.194ms/token；Fast-P2 94.702ms/token。
- qwen fixed64 prefill估算TFLOPS：Native Full 24.177；D1 Dense 24.709；Fast-P1 16.523；Fast-P2 16.544（非硬件计数器）。
- qwen fixed64 decode估算TFLOPS：Native Full 0.157；D1 Dense 0.159；Fast-P1 0.160；Fast-P2 0.160（非硬件计数器）。

## Cache、显存与开销解释

Fast-P1在写入截止点真正缩短Hidden，前段仍保留稠密KV。Fast-P2在Prefill直接gather非视觉KV到最终缓冲，不先持久保存整层视觉KV再复制删除；Decode原位写入，历史Cache拼接/整段复制为0。KV Live与预分配容量分别报告。
不存在跨请求KV共享；每个请求独立分配，测量含其开销。原始逻辑位置不压缩；Qwen原生DeepStack/mRoPE保留。KV管理GPU时间来自单独Profiler运行的CUDA Events，包括排队/启动开销，不能冒充纯内存带宽。
GPU进程显存快照见resource_audit.json；总Peak Allocated/Reserved包含权重、临时张量、缓存和工作空间，不能用KV减少比例代替总显存减少。peak_increment为相对请求开始的动态峰值，不是只含临时Tensor的精确分解。
KV缩小不必然让Decode同比加速：每步权重矩阵读取、GEMV/小GEMM、原生Python调度和kernel启动仍存在。独立Profiler提供CUDA活动数和CPU启动时间证据；硬件带宽计数器不可用，不能确定宣称带宽受限或计算受限。
矩阵运算变小可能降低估算TFLOPS，但同时减少工作量和时延；不以TFLOPS单独判定成功。若Decode无改善，保留负结果，不用强制64步掩盖自然短答案结果。

## 质量与数值风险

- llava D1 Dense：相对Native损害1题、纠正2题；完整Token序列变化3题；相对Dense序列变化0题；EOS未结束/截断0题、空答案0题。
- llava Fast-P1：相对Native损害0题、纠正2题；完整Token序列变化2题；相对Dense序列变化1题；EOS未结束/截断0题、空答案0题。
- llava Fast-P2：相对Native损害0题、纠正2题；完整Token序列变化2题；相对Dense序列变化1题；EOS未结束/截断0题、空答案0题。
- qwen D1 Dense：相对Native损害1题、纠正1题；完整Token序列变化2题；相对Dense序列变化0题；EOS未结束/截断0题、空答案0题。
- qwen Fast-P1：相对Native损害1题、纠正1题；完整Token序列变化2题；相对Dense序列变化0题；EOS未结束/截断0题、空答案0题。
- qwen Fast-P2：相对Native损害1题、纠正1题；完整Token序列变化2题；相对Dense序列变化0题；EOS未结束/截断0题、空答案0题。
物理路径固定容差的Smoke数值检查失败6项，详见equivalence_audit.csv。数值近似与Mask/位置/Cache语义检查分开；语义通过不代表原生数值等价，亦未为通过而放宽旧容差。

## 后续建议

只建议在明确接受本轮近似计算、逐题差异和性能条件后，进入固定240题质量验证；该验证仍需预声明质量门槛。当前不应直接宣称原生等价、非劣效、硬件TFLOPS或DRAM优化已获证明。本轮到此停止，不自动扩样。

## 可复现性

源码、模型权重清单、精度、依赖、原始图片/输入和协议哈希在protocol.json；逐题20次原始测量在checkpoints，聚合CSV与论文PNG/PDF同目录。只在/data运行，未使用/data1。

## 独立交付复核

冻结Census捕获torch.matmul但CLIP原生使用torch.bmm；此处不改运行源码或时延样本，按每题真实QKV执行Shape及调用数、固定Vision config补计CLIP两次bmm。详细推导和Config SHA见clip_bmm_flops_supplement.csv，主estimated_flops.csv及complete副本均已补齐。冻结Worker原始账本保存在estimated_flops_worker_raw.csv供核查；正文语言Prefill/Decode TFLOPS和图A不受该Vision补计影响。

process_vram_snapshots.csv记录每个未计时质量请求中KV仍存活时的进程显存；它是nvidia-smi快照而非全过程硬件峰值，不能冒充峰值。请求实际Peak Allocated/Reserved由未Profile的20次重复记录。任务分组与全部负面案例分别见task_accuracy_60.csv、negative_cases.csv。

llava独立Profiler最耗时活动保存在profile_review.json，不能从这些活动名直接推出DRAM带宽或Tensor Core利用率。

qwen独立Profiler最耗时活动保存在profile_review.json，不能从这些活动名直接推出DRAM带宽或Tensor Core利用率。

## 权重、历史基线及设备活动复核

两模型7个实际权重文件本轮重新SHA256，与冻结清单逐项一致。120个Native历史配对中Token序列变化0项；本轮一律使用新测Native，不拼接历史时延或质量结果。详见model_binding_audit.json、historical_native_audit.csv。

Figure D另给NVML GPU busy-time采样曲线及独立Profiler CUDA活动数。NVML的memory utilization表示采样窗口读写忙碌时间比例，不是DRAM带宽，也不是Tensor Core或SM硬件Throughput。采样覆盖两Worker的部分执行区间，不对单个条件归因。

更新范数/Softmax/激活等非矩阵操作，以及GEMM内部Tile补齐或其他底层指令，不包含在形状FLOPs账本中；减少的是张量形状对应的估算矩阵算术量，不能宣称硬件实际浮点指令同比减少。

CPU核心回归为两环境各6/6通过，独立FLOPs记账测试另2/2通过。后者主动隐藏所有CUDA设备，CPU Profiler启动时记录的gpuGetDeviceCount code 100为无可见GPU探测，不是实验GPU错误，两项测试实际均通过。

- llava：NVML平均设备忙碌40.5%，SM时钟中位数1755MHz。
- qwen：NVML平均设备忙碌37.4%，SM时钟中位数1920MHz。

厂商理论参考见hardware_reference.json；RTX3090的BF16/FP16且FP32累积Tensor Core稠密参考为71TFLOPS，FP16累积与结构化稀疏峰值另列，均不是实测，也未用于推导实际带宽或硬件TFLOPS。

## 分模型决策

LLaVA：存在可复现的自然短答案时延收益，值得在明确承认近似数值路径、预先冻结质量门槛后进入240题质量验证；不能以本轮60题的净纠正直接宣称非劣效或质量提升。

Qwen：KV显著下降，但自然短答案时延收益很小，且旧固定隐藏表示容差未通过。目前不优先扩大为240题“等价且加速”的验证；先明确性能证据和近似计算接受标准。这不是自动追加的任务，本轮到此停止。

## 统一仪表KV微测与执行耗时

四路径额外采用同样的轻量CUDA Events包裹Cache.update，避免把带完整Profiler的Fast时间与不带Profiler的Native时间直接比较。原完整Profiler测得的KV时间另列保留，不直接相除；两组结果都不属于正式Latency重复。

|模型|条件|Prefill KV管理 ms|15步Decode KV管理合计 ms|Decode历史拼接新分配数|
|---|---|---:|---:|---:|
|llava|D1 Dense|1.539|26.854|0|
|llava|Fast-P1|1.107|27.015|0|
|llava|Fast-P2|0.441|27.148|0|
|llava|Native Full|0.048|25.269|960|
|qwen|D1 Dense|0.720|31.234|0|
|qwen|Fast-P1|1.620|31.278|0|
|qwen|Fast-P2|1.589|31.341|0|
|qwen|Native Full|0.730|23.513|1080|

这些KV时间包括小操作的排队与启动，不是纯内存拷贝时间。即使移除历史KV拼接，仍有模型权重读取、QKV/MLP/LM Head及原生eager操作，不能据KV缩小直接预测Decode加速。

- llava：基准驱动墙钟39.72分钟；20次正式重复的请求时延相加为28.95分钟。前者含预热、质量/形状采集和检查点写入，不含实现、CPU测试、模型加载、Smoke、后处理和推送。两模型双GPU并行，不能将两卡时间相加当总墙钟。
- qwen：基准驱动墙钟44.50分钟；20次正式重复的请求时延相加为32.47分钟。前者含预热、质量/形状采集和检查点写入，不含实现、CPU测试、模型加载、Smoke、后处理和推送。两模型双GPU并行，不能将两卡时间相加当总墙钟。

本60题是历史探索工程队列，不是独立外部确认集。TFLOPS下降并不自动意味着优化失败：LLaVA的估算Prefill TFLOPS由Dense的48.456降至Fast-P2的45.376，但工作量下降且自然请求更快；Qwen则从24.709降至16.544，同时Latency近乎不变，不能将少算宣称为等幅加速。

另给平均请求时延口径，避免只看跨题中位数掩盖长输入差异（逐题先取20次中位数）：
- llava 相对Native Full：平均请求290.18→237.49ms，均值口径加速1.222×；逐题加速比P10/P90为1.186/1.239。
- llava 相对D1 Dense：平均请求293.14→237.49ms，均值口径加速1.234×；逐题加速比P10/P90为1.196/1.252。
- qwen 相对Native Full：平均请求282.37→270.53ms，均值口径加速1.044×；逐题加速比P10/P90为1.015/1.108。
- qwen 相对D1 Dense：平均请求280.00→270.53ms，均值口径加速1.035×；逐题加速比P10/P90为1.005/1.098。

Qwen不能笼统写成所有输入都无收益：部分较长动态视觉输入的节省更明显，但主中位数收益很小；不得据此事后挑选有利题目重新定义主结果。
