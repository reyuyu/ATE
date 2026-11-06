# Qwen3-VL：Vision SDPA / Text Eager 续测结果

**最终状态：STOPPED。** 本轮在两张已授权空闲RTX3090上完成两道短输入的Vision Eager/SDPA Native完整请求比较。两题所有27个Vision Block均确认使用实际CUDA Flash SDPA，Text保持36层Eager；两种后端生成Token完全一致且自然EOS结束。但4个有效Logits步骤的平均绝对差均超过事先冻结的0.05阈值，因此停止，未启动MME/SEED正式评估，也未运行最坏输入或Physical C3请求。不能据此报告C3的质量损失，不能宣称最坏输入OOM已解决。

## 固定条件与CPU审计

基于 `213d4e3be766444e1ecd9a90ac479836f7756e9b`，使用独立分支 `codex/qwen3vl-mme-seed-2048-vision-sdpa`。Qwen3-VL-8B-Instruct固定revision为 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；BF16、PyTorch 2.6.0+cu124、Transformers 4.57.6、batch1、Greedy、自然EOS、最多32个Token、`logits_to_keep=1`。原2048语言侧视觉Token上限、Processor、Prompt、模型文件及输入Tensor绑定保持不变。

完整复用前轮6811图片路径的真实Processor审计及16607题输入绑定，逐字节SHA与基线Commit一致；重新核验原始图片和模型文件SHA。MME2374问/1187路径、SEED14233条原始记录/5624路径，保留重复Question ID及整图分片。5项CPU测试通过，包括官方模态后端接口及只读SDPA观察器逐位一致的CPU测试。旧LLaVA结果、旧PARTIAL状态和main均未修改。

## 预声明数值门槛

GPU运行前冻结：Vision pooled与三个DeepStack特征Mean Abs≤0.02、Max Abs≤0.5；Native相同前缀的全部有效Logits Mean Abs≤0.05、Max Abs≤1.0。跨后端完整生成Token是否一致单独观察，不要求逐位Logits相等。同后端Native/HF/No-op仍须严格 `torch.equal`；该项因前置门禁失败尚未运行。上述门槛没有根据结果调整。

## 两张卡的实际短输入结果

| 问题键 | 有效步骤 | Logits Mean Abs | Logits Max Abs | Mean Abs门禁 |
|---|---|---:|---:|---|
| MME__00798 | first answer token | 0.100300 | 0.750000 | 未通过 |
| MME__00798 | EOS prediction | 0.060077 | 0.398438 | 未通过 |
| MME__02358 | first answer token | 0.057641 | 0.500000 | 未通过 |
| MME__02358 | EOS prediction | 0.072049 | 0.437500 | 未通过 |

`MME__00798`：264个ViT Patch、66个语言侧视觉Token；两种后端均输出 `Yes`，完整Token `[9454, 151645]`。`MME__02358`：260个Patch、65个视觉Token；均输出 `No`，Token `[2753, 151645]`。首Token和EOS预测均相同，不存在空答案或截断。这两道输入依据输入长度/视觉Token数事先选择，不能当作正式Benchmark样本分数。

全部8份视觉特征的误差通过既定阈值，Mean Abs最大0.002857、Max Abs最大0.154297；输出形状、Patch数、语言侧视觉Token数、三个DeepStack索引 `[8,16,24]` 及输入Tensor绑定保持不变。特征小误差在语言模型中形成了上述Logits差异。这里只确认观测，不能据此推断大样本Accuracy损失。

实际调用 `aten._scaled_dot_product_flash_attention.default` 共54次，每道SDPA请求27次；Text未调用SDPA，Math/cuDNN回退被官方上下文禁用。没有计时、Profiler、FLOPs测量或Kernel搜索。

短输入两卡Peak Reserved均为16.4414GiB，余量7.1469GiB；没有OOM或NaN/Inf。此结果只覆盖260/264 Patch，**不覆盖原OOM的8160/8192 Patch**。

## 初始审计器问题与修正记录

第一次尝试被新写的只读观察器误拦：`inference_mode` 暴露外层 `aten.scaled_dot_product_attention.default`，观察器尚未看到内部CUDA算子就拒绝了它。这是本轮审计器错误，不是模型或C3语义失败，也未发生OOM。完整日志、源码SHA及失败现场保存在 `attempts/attempt1_observer_composite_wrapper_failure/`。

修正仅让观察器进入PyTorch已注册的官方Composite分解以观察真实融合算子，不添加Attention算术或自定义Kernel，不改变后端选择和任何阈值。CPU观察前后结果严格相等；修正版本在GPU运行前重新测试、冻结并提交。之后真实GPU数值比较触发当前停止状态。审计器修正不能视为模型数值修复。

## Physical C3与正式质量评估状态

历史Physical-Original引擎保持原字节SHA `a719b590975e6282ad983fcd27b80f2e031c302caf47eaf54d50e1f3e36cca77`；保留L0–L21 R写入、L22前物理视觉Hidden退出、P−1/Decode禁图、L1–L6答案Attention物理跳算及原紧凑KV。没有加载Repaired、cuBLASLt或Zero-Key版本。

历史Physical/Dense F的严格门槛Mean Abs≤0.02、Max Abs≤0.5及Token一致没有被修改，历史不等价记录仍有效。本轮没有执行Physical请求，因而没有新的Mask/mRoPE/Cache/退出/跳算验收结论，不能将配置或CPU检查冒充GPU语义验收。

| 项目 | 实际状态 |
|---|---|
| 两卡短输入Native跨后端比较 | 已运行，数值门禁未通过 |
| 真实Vision Flash SDPA / Text Eager | 短输入已确认 |
| 最坏输入完整Native/Physical请求与显存门禁 | 未运行 |
| 同后端Native/HF/No-op严格一致 | 未运行 |
| Physical/Dense F语义及数值检查 | 未运行 |
| MME Native / Physical完整得分 | 未运行，正式Checkpoint 0/4748 |
| SEED Native / Physical完整ACC | 未运行，正式Checkpoint 0/28466 |
| 图片簇配对20000次Bootstrap和质量图 | 无完整预测，不计算 |

MME与SEED的CSV保留覆盖状态，不填入预测分数或零分。SEED仍冻结为本地HF完整候选平均条件负LogP排序、最终ACC，不冒充官方公布分数。没有转用自由生成或改变候选协议。

## 文件与资源释放

逐步骤Logits与特征误差分别见 `backend_logits_comparison.csv`、`backend_feature_comparison.csv`；真实运算符、显存及答案见 `actual_vision_backend.csv`、`smoke_memory.csv`、`backend_comparison_answers.csv`。图表PNG/PDF位于 `figures/`，只画真实观测的数值差异，不伪造质量图。

完整特征与Logits数组保存在服务器 `/data/lk/ATE-resources/experiments/qwen3vl_mme_seed_2048_vision_sdpa/`，路径和SHA见 `raw_array_sha_index.json`；这些数组不在GitHub。原始停止日志与失败现场均保留。

两张卡的本轮Worker均已退出，当前各27MiB，无计算进程；详见 `gpu_release_audit.json`。未启动正式评估、未追加GPU实验，也没有降低分辨率、改变精度或放宽数值门槛。

**本轮最终回答：** 目前只能确认短输入能够通过官方Vision Flash SDPA完成Native请求，且这两题的生成答案保持一致；无法确认最坏输入OOM是否解决。由于预声明跨后端Logits门禁失败，MME和SEED的Native/Physical完整质量差均尚未测得，不能回答C3损失多少质量，也不能判断退化是否集中于OCR或空间关系。
