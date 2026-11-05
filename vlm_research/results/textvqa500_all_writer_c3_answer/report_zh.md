# LLaVA TextVQA500：全层视觉写入与Dense C3式答案读取

**N=49.68%，A=25.56%，B=25.70%。** 在本批固定的顺序干预中，较大的净质量差来自**答案阶段直接禁图**：N−A=+24.12pp，而A−B=-0.14pp。下面同时报告图片簇配对区间及所有负面结果，不将点估计当作普遍因果份额。

## 冻结样本、协议与复用

基于Commit `5627bc822a7827ae11edfaec84c31f6b73398e25`，原样复用c3v2_middle_visual_reaccess的500题、309个图片SHA簇，无新增OCR输入。所有Question ID、十个官方答案、原Prompt、完整Token、FP16图片Tensor及原图片SHA已在CPU逐题重建并与历史Native和原C3检查点输入绑定严格一致。

模型LLaVA-1.5-7B revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`，FP16/Eager、Transformers 4.49.0、Greedy、max_new_tokens=32、logits_to_keep=1，原generation_config EOS；每请求独立完整Native DynamicCache，所有32层保留Image Hidden和KV。评分调用已固定SHA的官方TextVQA十答案留一共识软评分器。没有重新抽题、筛除异常或修改Prompt。

|条件|全部32层R→Image|P−1/真实Decode→Image|L2–L5答案Attention输出|执行图|
|---|---|---|---|---|
|N|开放|全层开放|正常|Native，审计复用|
|A|开放|全层禁止|正常|Dense完整计算|
|B|开放|全层禁止|仅答案位置置零|Dense完整计算|
|历史C3-V2|仅L0–L19开放|全层禁止|物理跳算|L20退出Image Hidden，压缩KV|

层索引0-based。R为最后一个Image Token之后至原始P−1之前，包含问题及原模板。B不跳过L2–L5之前的任何Prefill Query，不直接修改R/Image Query输出；仅将o_proj的原始P−1行及每个真实Decode行置零。全部MLP、Norm、Residual、RoPE、QKV及完整KV更新仍正常。

未找到严格匹配的历史完整500题A/B条件。旧全层All-Writer的300题及500题物理C3/Middle结果无法替代本轮A/B。Native 49.68%和原物理C3-V2 22.34%的1000份历史记录经归档逐文件SHA、原Payload摘要、模型权重、源码和输入绑定复核后保留；Native另经六题真实Smoke完整Token及首Token Logits哈希严格复现。新增A/B各500题，Smoke不计入正式矩阵重复评分。原始历史归档仍位于继承目录，派生baselines仅保留质量字段及完整溯源，不把旧计时当作本轮测试。

## 完整答案软Accuracy

|条件|软分总和/500|Accuracy|相对N Δpp|相对N软分升/降/不变题数|
|---|---:|---:|---:|---|
|N|248.400/500|49.680%|+0.000|0/0/500|
|A|127.800/500|25.560%|-24.120|9/143/348|
|B|128.500/500|25.700%|-23.980|7/142/351|
|Historical-C3-V2|111.700/500|22.340%|-27.340|7/160/333|

软分是十答案共识分，不是二值正确题数；不能把分数上升/下降称为对错翻转。原版C3-V2单列为历史参考，其不同的写入限制和物理路径不能与B混同。

|比较|差值pp|普通配对95%CI|候选软分更高/更低/相同|完整Token序列变化|文本答案变化|首Token变化|
|---|---:|---|---|---:|---:|---:|
|N−A|+24.120|[+20.142, +28.189]|143/9/348|308|308|182|
|A−B|-0.140|[-1.317, +1.043]|7/6/487|109|109|48|
|N−B|+23.980|[+20.059, +27.992]|142/7/351|302|302|174|

## 质量损失归因

保留全部32层R视觉写入后，直接禁止答案位置读图的N−A为+24.120pp，95%CI [+20.142, +28.189]。在此禁图背景下，再把L2–L5答案Attention输出置零的A−B为-0.140pp，95%CI [-1.317, +1.043]。合计N−B为+23.980pp。

**直接回答：在本批TextVQA500中，即使全部32层R写入保持原生，也不能保留Native质量；主要观察损失来自P−1/Decode直接禁图。** B比A的点估计为+0.140pp，A−B区间跨0，本批未显示再置零L2–L5答案Attention的明确额外净损失。这不构成早层跳算严格无损、有效提升或非劣效证明。

这是固定顺序下的两个条件效应：A−B仅衡量已经禁图时的早层答案Attention输出作用。本轮没有新增“答案可读图但早层置零”的第四个条件，因此不能估计完整二因素交互，也不能推广为所有背景下独立相加的通路份额。N−A同时改变P−1和真实Decode，不能进一步区分两阶段各自贡献。B与历史物理C3的差别还混合了深层R写入及物理数值路径，不用它单独断言深层写入的纯效应。

## 统计、正确性与边界

20,000次按309个Image SHA簇配对Bootstrap，固定seed=20261106。每次有放回抽取图片簇，保留同图全部问题，按题目等权重新计算软Accuracy。主三项比较在GPU前冻结；图中及CSV中95%CI均为普通探索性点位区间，未作为校正后的确认性检验，CI跨0不代表无损或等效。500题为历史复用探索样本，不是独立确认集。

10项CPU测试通过。两卡各3题Smoke验证Native/No-op及HF生成全部有效Logits严格torch.equal；Native完整Token与首Logits哈希复现历史。32层所有P−1之前的Prefill Hidden在N/A/B间逐层严格相等，原生视觉特征一致，RoPE相同；禁止的实际Attention概率严格0，非目标Mask及o_proj行不变，全部32层MLP/Norm/Block完整执行，独立完整KV存储、Cache位置及Hook清理均通过。1000份正式生成均核验输入、逐步Logits有限、实际Mask、块计数、Cache长度和模型参数版本。

首次Smoke启动因独立加载器遗漏历史模块级CUBLAS_WORKSPACE_CONFIG而在图像编码运算前中止；失败现场保留于diagnostics/smoke_attempt1。补齐历史相同的:4096:8配置后通过全部Smoke，未改精度、输入、数值门槛或实验条件，正式矩阵只在通过后启动。执行环境补充见execution_environment.json。

所有EOS、截断、空答案、重复标记和负面案例完整保留。输出异常共5条，见output_anomalies.csv；重复为三Token子串启发式，不自动排除。逐题全部答案、完整Token、软分及配对变化见per_question.csv/generation_outputs.csv/paired_score_changes.csv；negative_cases.csv保留A/B相对Native全部软分下降题。

本轮没有运行任何测速、物理优化或额外层组合。原始模型/图片仍保留于服务器资源目录，其SHA和输入绑定在交付中可审计；GitHub包含完整答案和源码审计，不宣称包含模型权重或原始图像。提交独立分支、释放两张GPU后停止。
