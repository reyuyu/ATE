# 七个完整Benchmark：Native / 原版Physical C3-V2

状态：LLaVA本轮五项生成于2026-11-06 20:23（北京时间）全部结束。
共20,612题、41,224条完整答案；四项评分完成，MM-Vet已生成但待官方评分。
GPU2、GPU3均已释放，不再启动新增实验。GQA、VQAv2继续按用户要求暂缓，
Qwen仍因既有数值门禁未通过而停止，不将它们计入本轮完成数量。

## 实测结果

| Benchmark | 输入题数 | Native | Physical C3-V2 | 差值pp | 图片簇配对95%CI(pp) | 损害/纠正 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| POPE三策略合计 | 9,000 | 85.9111% | 84.8444% | -1.0667 | [-1.6111, -0.5222] | 177 / 81 |
| ScienceQA-IMG | 2,017 | 66.0387% | 64.6505% | -1.3882 | [-2.4131, -0.4890] | 42 / 14 |
| MMBench | 4,377 | 62.8401% | 62.5000% | -0.3401 | [-1.1815, 0.4318] | 14 / 10 |
| TextVQA | 5,000 | 48.1280% | 22.5100% | -25.6180 | [-26.9263, -24.3139] | 1,490 / 94 |
| MM-Vet | 218 | 待官方评分 | 待官方评分 | N/A | N/A | N/A |

MMBench主分数按官方循环选项全轮命中，统计单位为1,176个循环题组，不是
4,377条旋转记录；按原始图像SHA进一步整簇Bootstrap。记录级辅助分数为
72.5154% / 72.2870%，不与主分数混用。TextVQA为十答案软分，损害/纠正表示
逐题软分下降/上升，不将所有下降都叫作二值的原对变错。

| POPE策略 | Native Accuracy | C3 Accuracy | Native F1 | C3 F1 |
| --- | ---: | ---: | ---: | ---: |
| Random | 87.3667% | 85.9667% | 85.9785% | 84.0591% |
| Popular | 86.2667% | 85.2333% | 84.9415% | 83.3646% |
| Adversarial | 84.1000% | 83.3333% | 82.9704% | 81.6176% |

所有区间为20,000次图片SHA整簇配对Bootstrap，固定种子20261106，保留原始
评分单位权重；均为探索性区间，不进行事后非劣效或无损认定。MMBench区间
跨零不代表无损。POPE、ScienceQA有约1pp的点估计损害；TextVQA降幅25.62pp，
明确否定本轮原版C3在该任务上保持接近Native质量的说法。这里评估的是原版
不回看图像的C3，不是Middle9/Middle10等恢复视觉访问的后续变体，不能混用。
本轮未新增归因条件，因此不能从这张表唯一归因于视觉退出或答案Attention跳算。

### 输出异常和生成上限

ScienceQA的Native解析失败0题，C3解析失败26题，均保留为错误，没有改用候选
LogP或约束解码掩盖。TextVQA的Native空答案0、截断3、重复3-gram标记9题；
C3空答案21、截断28、重复标记49题。这些输出全部进入原始评分，未删除失败题。

MM-Vet沿用本轮冻结的max_new_tokens=32：Native有117/218题达到截断上限，
C3有123/218题达到上限。故“生成完成”仅指全部冻结请求执行完毕，不意味着
所有回答均已语义完整地结束。尚未取得官方分数；任何后续评审必须同时说明
这一较高截断率和长度限制。本轮不擅自修改生成上限或追加生成。

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
自己的两卡Worker。两片已正常以exit code 0结束；本轮30/30真实GPU Smoke通过。

## 耗时与可复核产物

首次POPE正式记录为18:25:46；本次范围收紧后的续跑于18:35:52启动；最后
正式记录为20:23:24，监督进程于20:23:33结束。首次记录至监督进程结束约
1小时57分47秒，包含中间按用户要求暂停并修改执行队列的时间；续跑阶段约
1小时47分41秒。这是任务墙钟耗时，不是性能Benchmark或算法加速比。

CPU收尾脚本为`execution_scope/finalize.py`，仅读取已完成记录，不加载模型
或启动GPU生成。它复核冻结源码/Manifest、SQLite完整性、41,224条记录SHA、
每题双条件输入绑定、图片簇分片、独立评分、30条Smoke及正常资源释放。

- `accuracy_summary.csv`、`summaries/*.json`：总体分数、配对区间、POPE策略及ScienceQA学科结果。
- `*_per_question.csv.gz`：每条件完整答案、Token IDs、逐题分数、EOS/截断/解析状态。
- `worker0_records.jsonl.gz`、`worker1_records.jsonl.gz`：完整可移植检查点、输入映射及逐记录SHA。
- `outcome_audit.csv`：逐条件EOS、截断、空答案、重复、解析失败计数。
- `score_changed_cases.csv`：全部逐题损害/纠正，包含Gold与两条件答案，不只展示成功案例。
- `validation.json`、`completion.json`、`execution_audit.json.gz`：最终复核、完整Smoke与执行日志、GPU释放记录。
- `artifact_hashes.json`：产物SHA256。原始SQLite留在本机，不重复提交大数据库。

MM-Vet的218题两条件答案完整保存，但没有官方分数；未调用付费外部评审。
GQA、VQAv2和Qwen不属于本轮已完成结果，不自动继续。
