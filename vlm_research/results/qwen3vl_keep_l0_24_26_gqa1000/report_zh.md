# Qwen3-VL Keep-L0–L24 / L26：固定1000题快速质量验证

在上一轮完全相同的1000道GQA val_balanced问题（918个Image SHA簇）上新增两个固定条件。所有层编号0-based、包含端点：Keep-L0–L24保留25层R读图；Keep-L0–L26保留27层。P−1与全部真实Decode始终在36层禁读原始Image Key；L1–L6 Attention正常。

模型为Qwen3-VL-8B-Instruct，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；BF16/Eager、Transformers 4.57.6、原Prompt/Processor、Greedy、max_new_tokens=32、logits_to_keep=1，每请求独立完整真实KV。只做Softmax前Hard Mask，Image Query及其余Attention、QKV、MLP、Residual、mRoPE、DeepStack与KV均保持原生。不进行物理删除、测速或新增搜索。

## Accuracy

|条件|正确/1000|Accuracy|相对Keep-L0–L21 Δpp|相对All-Writer Δpp|
|---|---:|---:|---:|---:|
|Native|694|69.400%|+10.900|+1.700|
|F|588|58.800%|+0.300|-8.900|
|Keep-L0-L21|585|58.500%|+0.000|-9.200|
|Keep-L0-L24|664|66.400%|+7.900|-1.300|
|Keep-L0-L26|669|66.900%|+8.400|-0.800|
|E|680|68.000%|+9.500|+0.300|
|All-Writer|677|67.700%|+9.200|+0.000|

Native、F、Keep-L0–L21、E、All-Writer均复用上一轮经审计记录，本轮新增2×1000份完整生成。F/E的L1–L6 P−1/Decode输出置零仅为历史对照；两个新Keep条件不包含这一干预。

## 预设配对比较

|候选−参考|Δpp|普通95%CI|参考对→错|参考错→对|答案变化|首Token翻转|
|---|---:|---|---:|---:|---:|---:|
|Keep-L0-L24−Keep-L0-L21|+7.900|[+6.052, +9.838]|10|89|186|181|
|Keep-L0-L26−Keep-L0-L21|+8.400|[+6.513, +10.338]|9|93|200|196|
|Keep-L0-L26−Keep-L0-L24|+0.500|[-0.199, +1.207]|4|9|31|30|
|Keep-L0-L24−All-Writer|-1.300|[-2.220, -0.404]|17|4|46|44|
|Keep-L0-L26−All-Writer|-0.800|[-1.529, -0.100]|11|3|28|27|
|Keep-L0-L24−Native|-3.000|[-4.654, -1.327]|50|20|145|134|
|Keep-L0-L26−Native|-2.500|[-4.137, -0.813]|47|22|139|129|

20,000次Image SHA整簇配对Bootstrap，seed=20261106，整簇重抽后按题数加权。区间是普通探索性95%CI，未作确认性多重检验；CI跨0不能解释为非劣效或无损。本1000题已用于研究开发，不是新独立确认集。保留官方types.semantic五类任务分组。

## 正确性与范围

5项CPU测试覆盖精确层端点、36层Prefill目标Mask、Decode全层禁图、非目标连接不变、样本及图像分片，以及新增层差集。两卡各选此前冻结Smoke中的最短/最长输入，共4题；通用Keep21完整Token及全部有效Logits严格复现历史Keep，Native/No-op有效Logits严格torch.equal，两个新条件禁止实际概率严格0、独立Cache存储与Hook清理通过。2000份正式新增生成均检查Logits有限和目标Mask数量。

所有新旧答案、完整Token、评分、EOS、截断、空输出、首Token变化及全部损害/恢复案例均保留。共0条输出异常，详见output_anomalies.csv。评分严格沿用原始解码文本的rstrip(".").lower()规则，不额外strip。

数据与源码基于`dd464a691c30a0e047ddfb59177ec6a8ad2a40be`；新执行源码冻结于`a5057c4fd6bca2b4cade5258749efeaa6060f11b`。历史记录、main不修改。所有结果完成后释放两卡并停止，不扩展其他层集合。
