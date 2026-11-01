# GQA3000：Only Text 与 Sparse Visual + Text

|条件|正确数/3000|Accuracy|
|---|---:|---:|
|Full|2627/3000|87.5667%|
|Only_Text|2612/3000|87.0667%|
|Sparse|2629/3000|87.6333%|

Only Text − Sparse = **-0.5667 pp**；图片SHA256整簇配对95%CI **[-1.0902, -0.0653] pp**。固定种子20261102、100,000次Bootstrap，保留题目加权，共2409个图片簇。未作非劣效或等价认定。

**简短结论：本批3,000题中，Sparse比Only Text净多答对17题，准确率高0.5667个百分点，普通配对95%区间支持这一方向。保留L14/L19的直接视觉访问优于完全取消该访问，但效应幅度较小。Sparse相对Full仅为点估计多对2题，不能据此声称优于Full、等价或无损。**

|任务|题数|Full|Only Text|Sparse|
|---|---:|---:|---:|---:|
|ChooseAttr|500|82.60%|82.40%|83.00%|
|ChooseCat|500|89.60%|89.20%|90.00%|
|ChooseRel|500|83.60%|82.60%|83.40%|
|CompareAttr|500|89.20%|87.60%|88.40%|
|LogicalObj|500|95.80%|95.80%|96.40%|
|QueryAttr|500|84.60%|84.80%|84.60%|

Only Text正确、Sparse错误：**23题**；Sparse正确、Only Text错误：**40题**；两者均正确2589题，均错误348题。

复用原六类各500题、LLaVA-1.5-7B固定revision、FP16/eager/Transformers4.49.0、原始Prompt、Greedy、独立KV Cache、EOS及最多32 Token，并沿用原GQA完整答案归一化评分。
Full全部3000题复用RQ2.3绑定的历史结果；Only Text复用RQ2.6的240题；Sparse只复用RQ2.4的P−1+Decode 240题，未复用RQ2.3的P−1-only Sparse。Question ID、完整样本/Prompt绑定、图片SHA256、模型和Mask定义均逐题核验。
直接调用未修改的RQ2.6 Only Text生成器和RQ2.4 Sparse生成器。Mask为Softmax前来源−inf；非P−1 Prefill读图正常，MLP/残差/位置及全部视觉Token保留。没有新算法、图表或额外条件。
三卡均通过简短No-op及Mask Smoke：历史Full/两策略Token与首步Logits精确一致，No-op全部有效Prefill/Decode Logits精确一致，非P−1 Prefill输出精确一致。仅使用GPU0/2/3；GPU1未使用。
正式新增5520次生成。三卡正式墙钟约12.33分钟；从Worktree创建至汇总约23.02分钟，不含最初阅读和之后推送。

三条件均为3,000/3,000正常到EOS，截断和空答案均为0。GPU0/2/3已回落至27MiB、0%利用率，GPU1的外部任务仍在运行且未被干预。

本轮比较为已有研究样本上的策略复核，不作为全新图像隔离确认集。来源Mask会重分配概率，不能直接解释为独立语义贡献；没有实际剪枝或加速结论。本轮到此停止。
