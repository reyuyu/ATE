# SPACE / NL视觉写入：GQA3000 CPU冻结与资源停止记录

**状态：STOP_NO_IDLE_GPU；实验尚未完成。** 初始检查四张GPU都有活动进程，没有真正空闲卡。本轮仅完成CPU输入审计、历史答案审计和设计冻结；没有GPU Smoke、模型加载、新答案生成或后台等待。不能据此判断SPACE/NL是否提高ACC。

## 已完成的输入核验

全部3000题、六任务各500题、2409张独特图片均复用原冻结Manifest。原Prompt、原生Tokenizer IDs/Offset、S集合及完整词组逐题核对；6000个边界位置全部符合预期。样本不重采、不按输出筛选。

- SPACE：原生ID **29871**，词片`▁`，单独解码是空串；它是Tokenizer的边界位置，不等于Prompt里有一个独立ASCII空格字符。
- NL：原生ID **13**，词片`<0x0A>`，解码为换行。
- 二者固定为R的前两个位置 **581、582**，Question从583开始；不包含Image或P−1。两者Offset可同为原始换行字符的跨度[4038,4039]，但Tokenizer位置不同。未向Prompt插入SPACE/NL文字。

N与D直接复用11fe70e中的完整3000题固定spaCy3.8.7/en_core_web_sm3.8.0解析，按原始Offset闭合到历史完整词组。原120题+12题的S/N/D映射全部严格一致；没有重新训练、替换分析器或按本轮结果修正can等词性。CPU共 **12项测试通过**，包含实际Hard Mask原语在真实样本布局上的Prefill和Decode连接范围检查；GPU概率及Logits检查仍未执行。

模板对照预先固定为Question之后、Instruction中的最早两个完整单Subword词法组，排除Question/B/Image/P−1，并要求均在S及N+D之外。实际全部3000题均选到 **Answer + the**，每题均新增2个Token，与B严格等预算；不加随机种子或临时换词。逐题位置见template_control_audit.csv。

## 冻结条件与待生成数量

九个标签为All-Writer、Semantic-k、S+SPACE、S+NL、S+SPACE+NL、Noun+Article、Noun+Article+SPACE+NL、S+Template-Control-2、N+D+Template-Control-2。All-Writer是答案阶段已禁止直接读图的历史基线，不是Native Full。

已逐份审计原6000个All/S检查点的Git冻结字节、图片/输入/接收集合、生成Token、评分、真实Cache协议及结果哈希。额外严格审计原132题Noun+Article记录并复用。没有复用120题Accuracy替代3000题结果。
七个新增条件共21000个标签；同题精确集合去重及历史复用后，仍需 **20868条新的完整Greedy生成**。本轮实际新增为0。token_plan.json保留全部标签、实际接收位置、canonical ID及复用出处。

## 当前可核验的历史ACC（不是新增条件结果）

| 子集 | 条件 | 历史可用/目标 | ACC |
|---|---|---:|---:|
| main3000 | All-Writer | 3000/3000 | 87.033% |
| main3000 | Semantic-k | 3000/3000 | 83.800% |
| main3000 | Noun+Article | 132/3000 | 未完成，不计算总体ACC |
| image_disjoint2640 | All-Writer | 2640/2640 | 87.121% |
| image_disjoint2640 | Semantic-k | 2640/2640 | 83.561% |
| image_disjoint2640 | Noun+Article | 131/2640 | 未完成，不计算总体ACC |
| pilot120 | All-Writer | 120/120 | 82.500% |
| pilot120 | Semantic-k | 120/120 | 81.667% |
| pilot120 | Noun+Article | 120/120 | 79.167% |
| historical12 | All-Writer | 12/12 | 100.000% |
| historical12 | Semantic-k | 12/12 | 0.000% |
| historical12 | Noun+Article | 12/12 | 50.000% |

与Semantic240问题及图片均不重叠的子集重新核验为2640题；不是独立外部验证。原pilot120与historical12单列，所有新条件仍待生成。historical_failure_cases.csv已列出12题×9条件，特别包括001020372、09662582、10763612、0160501；待生成答案明确写NOT_GENERATED，不当作空答案或错误。

## 预先声明的分析与执行门槛

主要指标只有原GQA完整答案ACC，不生成大小写敏感Gold NLL。五组比较按用户指定顺序冻结：S+B−S；N+D+B−N+D；S+B−S+模板；N+D+B−N+D+模板；S+SPACE−S+NL。20000次Image SHA整簇配对Bootstrap，seed20261104，保持同图所有题与条件同簇；报告普通探索性95%CI。主3000题五项比较额外报告Bonferroni调整99%区间（每项alpha=.01，family alpha=.05），不将CI跨0当作非劣效。其他子集与六任务另列描述性95%CI。

一/两Worker的预分片均已冻结；同图所有题和条件在同Worker。当前无设备绑定，也没有GPU执行入口：恢复时须先检查真实资源、冻结执行源码，再通过每卡真实Smoke的HF/No-op严格torch.equal、禁止概率严格0、非目标Mask不变、独立KV、32层MLP、权重/输入Hash及Hook清理，才可运行矩阵。当前不能把历史Smoke当作本轮通过。

## 资源与停止结论

初始快照：GPU0/3分别18611/18167MiB已用并有活动进程；GPU1/2虽只使用2335/2337MiB，也分别有活动进程，且利用率70%/53%，不满足真正空闲要求。四卡均未使用，没有杀其他进程、后台监控或自动重试。此快照仅代表本轮初始检查，不声称后续占用不变。

**SPACE和NL是否提升ACC、是否超过等预算模板、是否修复部分真实失败题：目前均不能回答，新增生成数为0。** 不能用此前较高Attention Mass或已完成120题的N+D结果替代本次因果比较。Accuracy对比图、Budget–Accuracy图、六任务差值热图及新增配对CI均待完整生成后制作；当前未制造空图或补填数值。

设计、名单、边界/词性/预算审计、历史答案及本次资源停止记录保存在独立分支；main和所有历史结果未修改。本轮按资源不足的停止条件结束，不追加搜索、训练或其他实验。
