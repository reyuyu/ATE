# Visual Attention Mass of Omitted Semantic Writers：资源预检停止

**状态：STOPPED_NO_TWO_IDLE_GPUS。不是完成的Attention观察实验。**

本轮没有加载模型、没有运行GPU Smoke、没有执行任何Native Prefill或All-Writer对照，也没有生成新答案。没有Attention Mass、Excess、Enrichment、热图或高Mass排序的实测结果。

## 为什么停止

本次新鲜预检中，GPU0与GPU1各有其他计算进程，显存各约1791 MiB；GPU2有其他计算进程，显存约1793 MiB；只有GPU3无计算进程、显存约27 MiB。因此不满足用户明确要求的“两张真正空闲GPU”。有足够剩余显存也不能把已占用GPU视为空闲。完整UUID、显存、利用率和进程快照见gpu_selection.json。

未抢占、未终止其他任务，未退化为单GPU运行，未轮询等待，也未设置后台任务。gpu_release.json记录本任务未申请任何GPU资源，不能当成后续设备空闲保证。

## 已完成的CPU工作

- 基于425286a2566cc6b0387460492a0f99566b142a38建立独立Worktree和分支codex/semantic-omitted-token-attention-mass；没有修改main或历史结果。
- 按原selected_cases.csv完整保留16例：12个All-Writer正确/Semantic错误的案例，4个Semantic纠正All-Writer的反例；无重新挑选。原顺序交替分配两个Worker，各8例、各6错误+2纠正。
- 核验原始图片SHA、Prompt、Native Tokenizer输入IDs/Offsets、Semantic完整词组与选择集合。原参考Commit和逐文件哈希保持一致。Question/Instruction/Assistant-template分类按冻结字符区间，不重新划分词组。
- 保留原始All-Writer及Semantic答案供查看，不把历史All-Writer当Native Full。
- 7项CPU测试通过：Uniform指标、Excess符号、Enrichment零分母/越界处理、固定案例数量、R与P−1分离/分组、均值排名和层段覆盖。
- 冻结采集入口及规则。GPU数值正确性仍未验证，不宣称源码已经通过真实Smoke。

输入卡片gallery.html / case_cards.pdf仅展示原始图片、历史答案、实际Semantic选中状态和完整词组位置。标题和每页均标明Attention尚未采集；没有橙框高Mass位置或虚构热图。

## 已冻结的观察与排名协议

Native Full使用原LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9、FP16/Eager、Transformers4.49.0、torch.inference_mode和真实output_attentions。所有原始连接开放，只在采集Hook中移走CPU统计并释放完整Attention矩阵，Attention输出不修改。独立真实KV，不运行Decode或完整答案生成。

R是最后Image Token之后至P−1之前的位置。P−1单列，永远不参与R排序。576个原始Image Key与实际因果合法Key数决定p0；Excess=Mass−p0；Enrichment按用户公式从每Head实际Mass计算，分母无定义或舍入导致Mass≥1时标明未定义，不裁剪、不制造无穷值。

词组主排名：在每案例、层段与角色内，先跨层和32个Head平均，再平均原完整词组的Subword Mass；保留中位数、最大值及逐Subword原值，但不按单Head最大值挑选。层段包括L0–3、L4–7、L8–11、L12–15、L16–19、L20–30、L31、L0–15、L0–19、L0–31。L31单列且明确其R输出在答案阶段禁读图时无法继续传播到第一答案位置。

橙框规则已预设为每案例L0–15内未选Question词组的相对Top5，不是绝对高Mass、Excess>0或因果重要性阈值。Question、Instruction、Assistant/template分别排名；P−1从未混入。

两个Smoke问题固定为各Worker首例001020372与14236062；计划比较Native/No-op有效Logits严格torch.equal，并比较Native Full与历史协议All-Writer的完整R→Image概率张量严格一致。计划总计16个主Prefill+6个验证Prefill，无Decode。这些均尚未运行。

名词标签仅加载上一轮已冻结的文本CPU POS结果（11fe70e82d6b608da21d9bd375ce49abf390d4d5，spaCy3.8.7/en_core_web_sm3.8.0），映射回原词组，只用于描述，不参与选题或Mass排名。冠词、关系词和Question末尾标签也已预声明。未依据模型结果修订语法规则。

## 本轮五个研究问题的回答

1. 哪些未选Token的Full视觉Mass最大：**未测，无法回答。**
2. 是否集中于冠词、名词、关系词、Question末尾、Instruction或Assistant：**未测，不能用原选择集合或上一轮恢复案例代替Attention观测。**
3. 集中在哪些层：**未测，没有层段排名。**
4. 错误与纠正案例是否不同：**未测。**
5. 哪些候选最值得Token-Swap：**本轮没有Attention证据支持候选优先级，未追加因果验证。**

high_mass_omitted_tokens.csv仅有Schema表头，零观测不是“没有高Mass遗漏词”的结论。逐Head CSV和Attention图未生成；token_mapping_audit.csv是真实CPU映射审计。所有未执行项目见observation_status.json。

Attention分配不等于因果重要性。未来即便观测到高Mass，也不能直接宣布它是Semantic回答错误的原因。本轮按资源停止边界交付，不扩样、不改变Semantic、不训练、不开展准确率搜索或追加GPU实验。
