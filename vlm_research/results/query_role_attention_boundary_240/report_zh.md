# Query-Role Attention Dormancy：未通过Smoke，已停止

**状态：PARTIAL。正式实验为0/240题、0/960次生成，不能得出I/R/A角色必要性结论。**

## 已完成

- 基于固定历史Commit `bd171312fd0489e8bc027744b7a976367da11668` 建立独立Worktree和分支 `codex/query-role-attention-boundary-240`，没有修改main。
- 阅读五份指定历史报告及必要实现；明确区分o_proj输出消融、来源Hard Mask和真实物理跳过。
- 从原始3000题按输入哈希冻结240题，六任务各40题、240个不同Image SHA256。未按输出、正确性或敏感性选样。
- 核对真实Processor输入、576个视觉位置、动态Relay/P−1位置及FP16输入哈希；模型文件SHA256与历史绑定一致。
- 在独立模块实现固定L2–L5的Full/I-Bypass/R-Bypass/A-Bypass，以及独立真实Cache生成、原子检查点、统计与绘图代码。**实现存在不等于GPU验证已经通过。**
- 最终11项CPU测试通过：动态角色、层范围、Prefill/Decode定位、输出行作用范围、No-op与异常清理、Cache隔离、输入选样不依赖答案、signed loss和图片簇Bootstrap。

## GPU Smoke失败与停止原因

只使用物理GPU1，UUID `GPU-b08489af-0a8a-eb48-55ba-8346733af417`；CUDA只暴露该UUID，程序内为cuda:0。未干扰其他GPU或终止任何外部任务。

Smoke题目在GPU前按最短、最长、中位Prompt长度预选为3题。实际在第1题就停止，**0/3题完成全部Smoke**。

1. 初版 `71c42d75e7` 将HF原生generate仅返回的末Query Logits与自定义Full的整个Prefill矩阵直接比较，触发形状比较错误。原生/自定义完整生成Token在该断言前已一致。原日志、环境、协议和CPU结果完整保留。
2. 正式采集尚未开始，因此仅修正验证对齐，加入CPU回归并重新冻结到 `3744af84e8`。仍使用原模型、原FP16、原Prompt和严格逐位比较；不修改层、样本、精度或容差。
3. 第二次Smoke对齐预测Query后，原生HF和自定义Full的完整Token序列仍一致，**有效Logits的torch.equal仍失败**。按任务要求立即停止，没有继续I/R/A验证，也没有启动正式矩阵。

当前不能将差异归因于Attention Hook：失败发生在**无消融的Full基线兼容检查**。HF默认`logits_to_keep=1`与自定义返回所有位置的LM-head计算形状不同，可能影响FP16数值，但尚未隔离证实。本轮不追加GPU诊断实验。

首次断言没有保存差异向量，因此最大数值差异没有记录，不能编造“误差很小”或声称容差内通过。要求的Full/No-op全部有效Logits检查及三角色GPU正确性验证尚未完成；CPU测试不能替代它们。

## 结果覆盖

|条件|计划题数|正式已完成|Accuracy|配对95%CI|
|---|---:|---:|---|---|
|Full|240|0|未测|未计算|
|I-Bypass|240|0|未测|未计算|
|R-Bypass|240|0|未测|未计算|
|A-Bypass|240|0|未测|未计算|

`accuracy_summary.csv`、`task_accuracy.csv`和`paired_comparisons.csv`将缺失数值留空并明确标为not_run；`per_question.csv`只有表头，没有虚构逐题行。没有生成Accuracy图或任务热图，以免将占位图误认为真实实验结果。20,000次Bootstrap实现仅通过CPU合成测试，未对不存在的正式结果运行。

## 科学问题判定

- **L2–L5是否仅在部分Query角色下低敏感？本轮尚不能回答。**
- **Image Query、Relay Query、Answer Query谁更重要？未测，不能排序。**
- **R是否比I/A更有损害？未测，不能把历史来源Mask或完整Prefill结果冒充当前角色输出消融的结果。**
- **是否支持视觉写入/廉价答案读取分离？历史证据提供动机，但本轮没有新增可用机制证据。**
- **未来哪些运算可安全省略？本轮没有证据可宣称任何物理计算已可安全省略；代码本身仍计算QKV、Attention与o_proj，只把目标行输出置零。即使将来某角色低敏感，其供其他当前/未来Query读取的K/V也不能自动删除。**

I为576个视觉Query；R为图像后、P−1前的全部非视觉Query；A为原P−1及所有真实Decode Query，包含EOS预测。三组数量不同，即使未来得到Accuracy差异，也不能直接解释成单位Query贡献或唯一的视觉记忆写入功能。

## 交付与资源

代码、冻结样本/输入、两次预正式验证日志、失败堆栈、模型/环境/源码绑定和CPU测试记录均保留。**没有正式原始Logits数组，也没有960个完整生成检查点。**

GPU模型进程已经退出；实时GPU1检查为27 MiB、0%利用率、无计算进程，见`gpu_release.json`。本轮没有建立后台轮询或自动任务；未放宽容差、未改模型精度、未扩展条件。

后续需先单独处理原生/自定义Full兼容验证问题，再由用户决定是否重新启动本轮；本次到此停止。最终结果Commit以Git分支提交为准；不自动合并main。
