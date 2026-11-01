# Qwen3-VL：仅保留L26视觉读取的完整答案验证

复用上一轮240题（六类各40）及全部动态输入绑定；未重新采样。模型Revision、BF16/Eager、官方Chat Template与上一轮一致。仅原始Prefill P−1施加Image来源Hard Mask；其他Prefill Query、后续Decode和DeepStack全部原生。三个条件各自全新空KV Cache、Greedy最多32 Token，不复用扫描实验作为生成结果。

## 总体结果

|条件|正确/240|Accuracy|ΔFull (pp)|配对95%CI (pp)|原对→错|原错→对|首Token翻转|
|---|---:|---:|---:|---|---:|---:|---:|
|Full|217|90.417%|+0.000|[0.000, 0.000]|0|0|0|
|Keep L26|218|90.833%|+0.417|[-0.840, 2.049]|1|2|4|
|All OFF|218|90.833%|+0.417|[-1.255, 2.119]|2|3|6|

## 六任务结果

|任务|Full|Keep L26|All OFF|
|---|---:|---:|---:|
|ChooseAttr|85.0%|85.0%|85.0%|
|ChooseCat|95.0%|97.5%|97.5%|
|ChooseRel|92.5%|92.5%|92.5%|
|CompareAttr|90.0%|87.5%|87.5%|
|LogicalObj|95.0%|95.0%|95.0%|
|QueryAttr|85.0%|87.5%|87.5%|

## 结论与边界

Keep L26相对Full变化+0.417 pp，配对95%CI [-0.840, 2.049]；损害1题、纠正2题。首Token翻转与完整答案正确率分别报告，不能相互替代。
Keep L26与All OFF的总体及六任务Accuracy点估计均完全相同，但相对Full的错误翻转构成不同。本轮没有显示保留L26相比P−1全层Image OFF存在额外的Accuracy点估计收益，不能据此认定L26是不可替代的唯一视觉读取层。
本批题已用于挑选L26，因此本轮是同队列功能验证，不是独立确认。CI采用Image SHA256整簇配对Bootstrap（100,000次，seed=20261101），保留题目加权估计；主要比较为Keep L26−Full，All OFF为描述性参照。未预设非劣效/等价检验，CI含0不等于无损。
Keep L26只表示P−1在L26允许直接读取Image Key，不代表整个模型只需一层视觉融合。其他文本Query仍可读图、P−1残差仍可能携带视觉信息，Decode完全原生；All OFF也不是纯文本模型或删除图像。

## 正确性与执行

复用已验证的动态Image Key与Softmax前−inf实现。每个目标层均直接检查原生Attention概率为0；所有Mask在Prefill结束即移除，Decode无Mask调用。各卡Smoke对比原生Full、监控Full和No-op的生成Token及首Token Logits逐元素精确一致，检查非P−1 Query、DeepStack、异常Hook清理与36层Cache长度。每个生成条件使用独立原生DynamicCache，初始36层长度均0。输入和权重版本/存储不变。
正式运行最长Shard 0.87分钟；物理GPU0/2/3各80题，未使用GPU1。冻结代码Commit `ea07a18dde57d483c93ecc1263cde205fecb639f`。
EOS/截断/空答案：[('Full', 240, 0, 0), ('Keep L26', 240, 0, 0), ('All OFF', 240, 0, 0)]。
逐题生成文本、Token ID、正确性和翻转见per_question.csv；完整Mask/Cache审计见各Shard检查点。GPU释放另见gpu_release.json。本轮到此结束，不新增其他条件。
物理GPU0/2/3最终均为27 MiB、0%利用率，三个Worker正常退出；GPU1外部任务未干扰。
