# 七个完整Benchmark：Native / 原版Physical C3-V2

状态：2026-11-06 18:25（北京时间）启动。两卡POPE真实Smoke共6/6通过，
已进入POPE完整Native/Physical配对评测；尚无本轮正式完整split分数。

## 本轮范围

2026-11-06用户收紧执行范围：**暂时跳过GQA val_balanced 132,062题和
VQAv2 val2014 214,354题**，不自动续跑。当前只继续POPE、ScienceQA-IMG、
MMBench、TextVQA、MM-Vet，共20,612题、41,224条Native/Physical生成。
原七项Manifest和科学协议原样保留，已完成POPE记录精确续用，不重新抽样。
执行队列变更独立冻结在execution_scope/，不修改正在执行的Worker源码。

原始七项冻结清单包含下载审计通过的367,028条记录：POPE 9,000、ScienceQA-IMG
2,017、MMBench 4,377、TextVQA 5,000、MM-Vet 218、GQA 132,062、VQAv2
214,354。图片SHA整簇确定性分成两片，各183,514题；同题的Native与
Physical C3-V2在同一卡完成。原七项计划为734,056次生成；收紧后的本次
实际执行上限为41,224条，不将两项DEFERRED标记为已完成。

仅使用物理GPU2、GPU3，每卡一份固定FP16/eager LLaVA-1.5-7B；模型
revision、Tokenizer、Processor和全部权重文件已重新验证SHA256。
原版引擎与完成MME/SEED的213d4e3be766444e1ecd9a90ac479836f7756e9b
逐文件绑定，不替换为Dense评分或修复版算法。P-1/Decode禁图、L20前视觉
Hidden物理退出、L2-L5答案Attention物理跳算和Compact KV沿用原实现。

每个Benchmark先完成两卡各三道按输入长度选择的Smoke，两片均通过后
进入该Benchmark。Native/No-op及HF返回Logits要求严格torch.equal；
物理/对应Dense同前缀对照使用既有FP16阈值，不放宽。保存真实Mask概率、
QKV形状、Norm/MLP/Residual、RoPE、Cache、输入及异常清理检查。

## 评分边界

POPE使用冻结官方词法解析，三策略分别计算Accuracy/F1等；ScienceQA
保留真实Hint与选项顺序，不输入lecture/solution；TextVQA与VQAv2使用
各自十答案共识软分；GQA沿用已完成全val实验的预测转换规则。
MMBench使用固定VLMEvalKit的exact_matching与循环选项全轮命中协议，
不混用候选似然。MM-Vet只生成答案，未经授权不调用付费官方GPT评审，
最终标为GENERATED_AWAITING_OFFICIAL_SCORING，不虚构分数或区间。

## Qwen停止状态

后续Qwen Vision SDPA提交53e9cc1f52481de338ab1a3149a41b63eaa5d162
报告数值门禁失败，尚未验证最坏输入的完整Native/Physical安全性。因此
本轮不启动Qwen正式评分，不降低视觉上限、改变后端或放宽阈值。Qwen的
停止不取消已通过门禁的LLaVA任务。

## 检查点与状态

逐条件SQLite原子事务，完整输入绑定与记录哈希，精确恢复，不保存大型
Logits或图像数组。仅完整双条件split才汇总分数；完成后独立复算评分，
以图片SHA整簇进行20,000次配对Bootstrap，区间为探索性，不作无损认定。
全程不测速、不训练、不搜索策略，不修改main或历史Checkpoint。

每卡进度位于worker0/与worker1/，完成分项位于summaries/。资源门槛为
/data至少20GiB空闲，OOM/非有限输出/绑定或正确性失败会停止本任务
自己的两卡Worker。报告须以实际运行结果更新，当前不能声称全量完成。
