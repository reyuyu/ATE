# L2-L4 Residual-only Pilot

64题，固定种子20261029；ChooseRel左右各16，ScienceQA-IMG16，TextVQA16；无表现筛选。
同一RTX3090、LLaVA-1.5-7B FP16/eager、Transformers4.49.0，全部条件本轮配对。
Full原生；P1-Joint仅原Prompt P-1同时置零L2/L3/L4的o_proj及MLP输出；Whole-Skip全部Token直接传入残差，不运行这些层Attention或MLP。
每一步重新计算完整前缀，无KV Cache，Greedy最多32Token，EOS停止；P1-Joint始终修改原P-1而非新生成位置。

| 数据集 | 题数 | Full | P1-Joint | Whole-Skip |
|---|---:|---:|---:|---:|
| gqa_chooserel（严格生成文本匹配） | 32 | 0.00% | 78.12% | 6.25% |
| scienceqa_img | 16 | 68.75% | 68.75% | 25.00% |
| textvqa | 16 | 53.75% | 53.12% | 40.62% |

| 数据集 | 条件 | 评分变化pp | 纠正/软分上升 | 损害/软分下降 | 平均Margin变化nats |
|---|---|---:|---:|---:|---:|
| gqa_chooserel | P1-Joint | +78.12 | 25 | 0 | +0.553955 |
| gqa_chooserel | Whole-Skip | +6.25 | 2 | 0 | -2.194580 |
| scienceqa_img | P1-Joint | +0.00 | 0 | 0 | -0.028320 |
| scienceqa_img | Whole-Skip | -43.75 | 0 | 7 | -2.261230 |
| textvqa | P1-Joint | -0.62 | 0 | 1 | N/A |
| textvqa | Whole-Skip | -13.12 | 0 | 3 | N/A |

## ChooseRel输出格式混淆与候选诊断

**ChooseRel的0%不表示Full没有关系判断能力。** 原Phase3.2的主要评分是完整候选LogP，而非自由生成短答案匹配。本轮保留原始问题Prompt，生成端复用现有GQA严格文本匹配函数；Full的32条答案均未严格等于单个left/right，但常给出包含方向词的完整句子。例如00543846的gold是left，Full生成`The brown bag is to the left of the girl.`，P1-Joint生成`Left`。原始Prompt没有强制短答案；不能根据这个结果添加短答指令或另选解析规则。

所以“纠正25题”仅指严格生成评分从0翻为1，混有输出格式改变，不得表述为纠正25个关系推理错误。Whole-Skip的2次生成得分上升也受同样限制。以下用已保存的原候选LogP补充判断，未增加GPU实验或更换评分协议：

| 数据集 | Full候选准确率 | P1-Joint候选准确率 | Whole-Skip候选准确率 |
|---|---:|---:|---:|
| ChooseRel | 27/32（84.375%） | 26/32（81.25%） | 16/32（50%） |
| ScienceQA-IMG | 11/16（68.75%） | 11/16（68.75%） | 6/16（37.5%） |

ScienceQA的候选排序与最终生成评分也不可互换：Whole-Skip的候选评分6/16高于实际生成选项字母评分4/16，主表仍报告真实生成分数。

## 简短结论

- P1-Joint在ScienceQA生成评分上持平，未纠正或损害任何题；TextVQA软评分下降0.625个百分点，仅1题下降。ChooseRel候选准确率从27/32降至26/32，尽管平均Margin增加0.553955；不能用平均Margin或格式相关的生成升分宣布推理改善。
- Whole-Skip在ScienceQA损害7题、生成分数下降43.75个百分点；TextVQA有3题软评分下降，平均下降13.125个百分点；ChooseRel候选准确率下降至16/32，平均Margin下降2.194580。该Pilot不支持将L2-L4整体视为可无损跳过。
- 仅省略原P-1更新与跳过全部Token整层，干预范围不同；结果只提示前者对这批题影响更小，不能据此证明全局等价、普遍无害，或识别具体信息路径。
- Whole-Skip有1条ScienceQA生成到32Token上限而未EOS；其余条件/数据集无截断。没有报告无Cache方法的加速比。

## 验证

- CPU测试与三数据集真实GPU Smoke通过；Full无Cache手动生成与原生有/无Cache生成Token一致。
- 两干预No-op的全部Logits与Full精确一致；独立计算零输出参考与两实现全部Logits精确一致，未放宽容差。
- P1-Joint每次前向只改固定原P-1的6个子层输出，其他Token局部输出不变；目标位置在L2/L3/L4各自严格保持层输入残差。
- Whole-Skip每层输入输出为同一个Tensor，全部Token精确恒等，目标层Attention/MLP调用为零；正常/异常后forward和Hook恢复。
- 输入和参数版本未变化，模型完整字节哈希前后相同；全部192条生成重新解码/计分，原始Logits独立重算LogP通过。
- 原始LogP重算最大误差：3.28e-14。逐题审计及答案见各数据集JSON和per_question.csv；原始NPZ仅本地保留。

## 边界

- 仅描述性Pilot，不推断总体等价或通用提升；没有训练、层扫描或额外消融，不报告无Cache方法加速比。
- ChooseRel复用原始问题和冻结240图来源，来源名含correct_question，不能解释为无偏GQA总体。
- ScienceQA评分为生成选项字母准确率；TextVQA为原10答案软评分，无OCR，软分上升/下降不是二值纠错。
- Margin仅用于ChooseRel/ScienceQA，完整候选全词表LogP求和，gold减最强wrong；TextVQA无虚构候选。
- 本轮Full均为同机无Cache重新运行，未混用历史Cache基线。

## 来源

- 执行代码冻结Commit：`3ea236213334e975e1b56119d4768c38e7a7077c`。
- 分支：`codex/l2-l4-residual-only-pilot`，基于已推送的`7e94cdb2af9b444007054ea09d36b8ce2981beb9`，未修改main或历史源码/结果。
- 完整样本、来源与代码哈希：`manifest.json`；环境及权重哈希：`environment.json`、`weight_audit.json`。
- 168项CPU回归测试：`cpu_tests.json`和`cpu_console.txt`；真实GPU检查：`smoke.json`；逐前向范围复核：`independent_review.json`及`review_results.py`。
- 逐题生成/评分与Margin：`per_question.csv`；两候选或多候选完整LogP和逐前向审计保存在各数据集的逐题JSON中。
