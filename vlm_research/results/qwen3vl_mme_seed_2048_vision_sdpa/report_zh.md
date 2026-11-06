# Qwen3-VL：Vision SDPA / Text Eager 续测准备

状态：**CPU_PREPARED_AWAITING_GPU_AUTHORIZATION**。本报告仅记录已完成的CPU审计与冻结执行入口。GPU Smoke、实际CUDA后端、完整质量分数均尚未运行，不应将本状态解释为修复成功或评测完成。

## 已完成

- 基于 `213d4e3be766444e1ecd9a90ac479836f7756e9b` 创建独立分支 `codex/qwen3vl-mme-seed-2048-vision-sdpa`。旧 `mme_seed_physical_c3_quality` 的PARTIAL记录及全部LLaVA结果保留原样。
- 固定Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16、PyTorch 2.6.0+cu124、Transformers 4.57.6。实际模型为27个Vision Block、36个Text Decoder层。
- 官方模态配置字典及动态切换接口通过CPU小模型测试：Vision可设置SDPA或Eager，Text始终Eager。此测试证明配置接口可用，**不能证明GPU实际调用了高效SDPA**。
- 前轮真实Processor的全量审计与输入绑定按字节复用，包括2048语言侧视觉Token上限、6811个图片路径、全部16607条输入与候选Token。
- 重新核验全部6811个原始图片文件及固定模型文件SHA；原有SEED重复Question ID保留，以原始记录索引构建唯一键。
- 4项CPU测试通过：官方模态配置、原子Checkpoint恢复及损坏拒绝、全部题目/重复ID/分片覆盖、冻结数值和后端协议。

MME为2374问、1187图片路径；SEED为14233条原始图像问题记录、5624图片路径。相同图片SHA的所有问题与条件仍在同一Worker。

## 冻结执行和门禁

唯一允许的执行变更为官方Vision Attention改为SDPA。未修改历史Physical-Original引擎；仍保留L0–L21 R写入、L22前视觉Hidden物理退出、P−1/Decode禁图、L1–L6答案Attention物理跳算及原紧凑KV。没有加载Repaired、cuBLASLt或Zero-Key方案。

新的GPU入口在官方PyTorch上下文中只允许Flash/Efficient SDPA，禁用Math/cuDNN回退。只读ATen运算符记录用于确认全部27个Vision Block的真实CUDA路径；36层Text仍须Eager。这里描述的是已写好的门禁，实际通过与否须GPU验证。没有性能计时、Profiler、FLOPs测量或Kernel搜索。

短输入Vision Eager/SDPA比较在GPU运行前固定如下阈值：pooled及三个DeepStack特征的Mean Abs ≤0.02、Max Abs ≤0.5；Native相同生成前缀下的有效Logits Mean Abs ≤0.05、Max Abs ≤1.0。所有有效步骤逐项保存，生成Token是否相同单独报告。两种后端的输入Tensor必须不变；同一后端Native/HF/No-op仍要求所有有效Logits严格 `torch.equal`。

Physical与Dense F的历史严格门槛仍为Mean Abs ≤0.02、Max Abs ≤0.5及完整Token一致；既有失败没有被修改。本轮只允许在语义及资源审计通过后，以明确披露数值不等价限制的方式开展探索性质量评估。

最坏输入Smoke名单由真实输入长度、Patch数和完整候选路径长度确定，未读取新预测。包含前轮OOM的8160 Patch输入、8192 Patch最大输入、两个Benchmark最长实际输入、最长SEED候选路径及中短输入。每张卡均须通过完整Native和Physical请求、SEED候选独立冷启动与Cache分叉严格一致验证，并保留至少512MiB显存余量。

## 尚未运行

| 项目 | 状态 |
|---|---|
| 高效CUDA SDPA实际调用 | 未运行 |
| Vision后端数值比较 | 未运行 |
| 最坏输入显存及完整请求Smoke | 未运行 |
| Native/HF/No-op逐位一致 | 未运行 |
| Physical/Dense F语义及数值审计 | 未运行 |
| MME Native / Physical完整得分 | 未运行 |
| SEED Native / Physical完整ACC | 未运行 |
| 配对Bootstrap与质量图 | 等待完整预测 |

因此，目前不能回答Vision SDPA是否解决最坏输入OOM，也没有Qwen MME或SEED的实测质量损失可报告。不存在推测填入的分数。

## 后续执行边界

附件第九节要求“先确认本轮GPU授权及空闲状态”。已提交本轮最多两张空闲RTX3090的授权问题，当前尚未收到答复，未启动GPU模型或占用GPU。收到授权后需再次核查UUID、进程、显存及至少20GiB磁盘余量，然后先Smoke；两卡均通过才允许完整评估。任何OOM、非法Mask、Cache/位置错误、非有限数值或SHA不一致均停止，不自动改变配置。

MME继续使用冻结官方Acc、Acc+及Score；SEED继续使用本地冻结HF完整候选平均条件负LogP排序，最终指标为ACC，不冒充官方已公布分数。完整统计采用原冻结Image SHA整簇配对20000次Bootstrap。全部新输出位于本目录；旧结果不会被改写。
