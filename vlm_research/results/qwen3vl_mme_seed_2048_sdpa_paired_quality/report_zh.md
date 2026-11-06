# Qwen3-VL同Backend全量质量评估：CPU准备记录

状态：**CPU_ONLY_COMPLETE**。本轮尚未启动GPU请求，没有新的MME或SEED质量分数。

基于Commit `53e9cc1f52481de338ab1a3149a41b63eaa5d162`，独立分支为 `codex/qwen3vl-mme-seed-2048-sdpa-paired-quality`，独立Worktree为 `/data/lk/ATE-worktrees/qwen3vl-mme-seed-2048-sdpa-paired-quality`。

本轮研究问题是同一Vision SDPA / Text Eager后端下Native与Physical-Original的同题质量差。与历史实验不同：跨Backend误差仅为诊断，不作本轮停止门禁。这一差异已经在GPU预测前写入新 `protocol.json`，没有改变旧实验任何结果或门槛，也没有重新判定旧STOPPED记录。

## 冻结设置与数据

Qwen3-VL-8B-Instruct固定revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16、Transformers4.57.6、PyTorch2.6.0+cu124。27个Vision Block配置官方SDPA；36层Text维持Eager。仍使用原2048语言侧视觉Token预算、原Prompt、Processor、Token边界和模型权重。

全部16607条输入、6811个图片路径及已有真实Processor审计按字节复用；已重新核验原图片和模型SHA。MME2374问、1187路径；SEED14233条原始图像题、5624路径，重复Question ID完整保留，以原始记录索引区分。整张图片SHA不跨Worker。

严格复用原Physical-Original引擎：L0–L21 R写入，L22前视觉Hidden物理退出，P−1/Decode禁图，L1–L6答案Attention物理跳算，原生mRoPE、DeepStack、MLP/Norm/Residual及紧凑KV。没有引入Repaired、Zero-Key或cuBLASLt修改。

## CPU实际验证

8项测试通过：官方模态后端配置；只读SDPA观察器；完整输入/分片/重复ID；Checkpoint恢复及损坏拒绝；紧凑KV保存内容、追加位置和独立存储；全部36层Mask范围及禁图概率；MME Acc+及Score与SEED图片簇统计复算；新旧数值门禁边界记录。

这里的CPU测试不能替代真实7B模型的GPU验证。没有将配置成功标为真实CUDA调用成功。

## 已冻结GPU门禁

最坏输入包括8160/8192 Patch的MME输入、SEED最长实际输入和完整候选路径、中短输入。每个案例先完成完整Native和Physical请求及SEED全部候选，记录实际Tensor形状、后端、峰值显存和至少512MiB安全余量；之后检查同Backend Native/HF/No-op全部有效Logits严格 `torch.equal`、观察器不改变结果、完整输出与EOS、Mask/mRoPE/物理退出/真实跳算/独立KV与Hook清理。

Physical与Dense F继续按历史Mean Abs≤0.02、Max Abs≤0.5及Token一致门槛分别报告数值结果；语义审计失败必须停止，只有已知数值差异且语义正确才可按事前协议继续探索性质量测试。

跨Backend诊断在必要门禁之后单独执行，两个MME及两个SEED短案例均按输入长度/视觉Token数预先选择。保留旧0.05门槛的诊断失败，不把它重写为通过；报告特征、Logits、完整答案、SEED候选排名及小样本ACC漂移，不推断全量成绩。

## 正式评估与当前缺项

两卡Smoke均通过后才允许正式Native/Physical同题评估。MME沿用官方Yes/No及Acc、Acc+、Score；SEED沿用冻结HF完整候选平均条件负LogP排序，最终指标ACC，不冒充官方公布分数。全部完整预测后按图片SHA整簇配对20000次Bootstrap，MME单位Score Points、SEED单位pp。

当前GPU门禁、正式Checkpoint、完整质量表、配对CI和质量图均未运行，不填入预测成绩。尚不能回答最坏输入OOM是否解决，或Physical在两个Benchmark上损失多少质量。

附件第十节要求“正式GPU运行前，重新确认本轮用户授权、GPU UUID、空闲状态、显存和磁盘剩余空间”。已提交本轮最多两张空闲卡授权问题；等待明确回复期间仅做CPU准备。最近预检GPU均空闲，磁盘约32GiB；启动前仍需重查。没有GPU模型进程需要释放，历史数据和main均未修改。
