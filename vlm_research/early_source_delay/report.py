"""Generate an explicitly scoped Chinese report from the measured outputs."""
import numpy as np
from .protocol import OUT,KEYS,CONDITIONS,old,read


def write_report(result,taskwise,masses):
    rows=result['summary'];lookup={r['condition']:r for r in rows}
    interactions=result['source_interactions']
    def supported(r):return r['D_family95_low']>0
    sig=[r['source'] for r in interactions if supported(r)]
    harms=[f'{r["condition"]}（原对→错{r["correct_to_wrong"]}题，纠正{r["wrong_to_correct"]}题）' for r in rows if r['condition'].startswith('A') and r['correct_to_wrong']]
    unobserved=[r['condition'] for r in rows if r['condition'].startswith('A') and not r['correct_to_wrong']]
    comparison=[]
    for layer,last,question in [(0,'A1','A3'),(1,'A4','A6')]:
        a,b=lookup[last],lookup[question]
        comparison.append(f'L{layer}：P−1→Image造成{a["correct_to_wrong"]}题退化/{a["wrong_to_correct"]}题纠正；Question→Image造成{b["correct_to_wrong"]}题退化/{b["wrong_to_correct"]}题纠正')
    compensation=('当前交互为'+', '.join(sig)+'来源的额外依赖提供了有限支持，但只限本模型、协议和240题样本' if sig else
        '本轮两个条件性交互均未提供经两比较校正的正向支持，因此现有结果不足以确认Delayed Attention Compensation；这不是证明不存在补偿')
    lines=['# Early Source-Specific Attention Necessity and Delayed Compensation',
        '', '**状态：complete；仅240题探索，不是3,000题新增实验。**',
        f'六类各40题，{result["image_clusters"]}个图像SHA256聚类；复用240条Full完整答案，只新增2,640条干预完整生成。',
        '', '## 直接回答', '',
        '1. **L0/L1哪些来源有可观察贡献？** 本批观察到完整答案退化的路径为：'+('；'.join(harms) if harms else '无')+'。未观察到退化的路径为：'+('、'.join(unobserved) if unobserved else '无')+'。Accuracy退化证明该具体屏蔽在这些题上有因果影响；只有LogP变化时，只能说预测分布受到影响。A1–A6是探索性比较，不能因未翻转或CI包含0就断言不必要。',
        '2. **Question→Image与P−1→Image是否不同？** '+'；'.join(comparison)+'。前者改变多个真实Question Query，后者仅改变原始P−1。本轮未加入两路径组合或中介恢复条件，不能据此确认H3的间接视觉链。',
        f'3. **早层关闭后L2依赖是否显著增加？** 预声明正方向为D>0；两比较校正95%CI严格大于0的来源为：{", ".join(sig) if sig else "无"}。两个D的数值与区间见下表；正点估计但区间跨0不是确认性证据。',
        '4. **是否支持Delayed Attention Compensation？** '+compensation+'。Mass增大本身不构成补偿证据。即使D显著，也不能证明读取了特定视觉语义。替代解释包括残差/Query重定向、Softmax归一化竞争、层间冗余及输出阈值非线性。',
        '', '## 完整答案Accuracy与预测变化', '',
        '| 条件 | Accuracy | ΔFull(pp) [普通95%CI] | 原对→错 | 原错→对 | 首Token Flip | 平均LogP Loss(nat) |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["condition"]} | {r["correct"]}/240 = {r["accuracy_pct"]:.3f}% | {r["delta_full_pp"]:+.3f} [{r["delta_ci95_low"]:.3f}, {r["delta_ci95_high"]:.3f}] | {r["correct_to_wrong"]} | {r["wrong_to_correct"]} | {r["first_token_flips"]} | {r["mean_logp_loss"]:+.6f} |')
    lines += ['', 'Accuracy采用历史完整答案评分器，不是首Token命中率。LogP Loss = logP_Full(t_Full) − logP_condition(t_Full)，使用FP32全词表log_softmax；t_Full固定为历史Full首Token。有符号负值不是错误，不能将它误称Gold答案概率。',
        '普通、A六比较校正区间、绝对Accuracy区间、翻转/损害/纠正率及LogP Loss区间完整保存在 `accuracy_summary.csv`。零损益导致的退化Bootstrap区间不证明等价或功能不必要。',
        f'空答案数：{sum(r["empty"] for r in rows)}；截断答案数：{sum(r["truncated"] for r in rows)}。',
        '', '### 条件定义', '',
        '| 条件 | 原始Prefill干预 |', '|---|---|',
        '| A1 / A2 / A3 | L0：P−1→Image / P−1→Question / Question→Image |',
        '| A4 / A5 / A6 | L1：P−1→Image / P−1→Question / Question→Image |',
        '| EarlyOFF | L0、L1的o_proj输出仅原始P−1置零 |',
        '| L2Image / EarlyOFF_L2Image | 原生 / EarlyOFF基础上，L2 P−1→Image屏蔽 |',
        '| L2Question / EarlyOFF_L2Question | 原生 / EarlyOFF基础上，L2 P−1→Question屏蔽 |',
        '', '**所有Decode步骤均不追加干预**，但自然继承各条件自己的真实Prefill KV。MLP及其他路径保留。Question只含真实问题文本，不含答题指令、模板、BOS或特殊Token。A3/A6不是P−1-only消融。',
        '', '## 两个条件性交互', '',
        '`D_s = [Acc(EarlyOFF)−Acc(EarlyOFF_L2s)] − [Acc(Full)−Acc(L2s)]`。正值表示早层关闭后，该L2来源的额外功能必要性增加。',
        '', '| 来源 | 原生L2屏蔽损害(pp) | EarlyOFF下L2屏蔽损害(pp) | D(pp) | 普通95%CI | 两比较同时95%CI |',
        '|---|---:|---:|---:|---|---|']
    for r in interactions:
        lines.append(f'| {r["source"]} | {r["normal_source_harm_pp"]:+.3f} | {r["earlyoff_source_harm_pp"]:+.3f} | {r["D_pp"]:+.3f} | [{r["D_ci95_low"]:.3f}, {r["D_ci95_high"]:.3f}] | [{r["D_family95_low"]:.3f}, {r["D_family95_high"]:.3f}] |')
    lines += ['', '200,000次按图像SHA256整簇配对Bootstrap，题目加权，种子20261031。直接对逐题四格对比重采样，共用图级重采样；不相减独立CI。预声明比较族只有D_visual和D_question，报告双侧普通95%CI及Bonferroni同时95%CI；支持正方向要求校正下界>0。六个A对比单独视作探索族；任务分组不做全局确认性推断。',
        '**设计边界：** B只检验整体P−1-only EarlyOFF后的L2必要性；没有A1–A6各自与L2屏蔽的交叉条件，因此不能回答每个单独A路径之后的条件性必要性。没有为此擅自增加生成条件。',
        '', '## 六类结果', '', '| 类型（各40题） | '+' | '.join(KEYS)+' |', '|---|'+'---:|'*len(KEYS)]
    task_lookup={(r['group'],r['condition']):r for r in taskwise}
    for t in old.TASKS:lines.append('| '+t+' | '+' | '.join(f'{task_lookup[t,c]["accuracy_pct"]:.1f}%' for c in KEYS)+' |')
    lines += ['', '逐题错误交集、Full正确条件下的联合受损/并集及Jaccard见 `error_intersections.csv`；每题全部受损/纠正条件见 `per_question_error_patterns.csv`。不能把不同条件的净Accuracy相同理解为答案相同。',
        '', '## L2来源分配', '', '下表为240题在首答案预测P−1的原始Mass均值，先对Head取平均，再对题取平均。Question与Template分开，Template包含原答题指令。',
        '', '| 条件 | BOS(%) | Image(%) | Question(%) | Template(%) |', '|---|---:|---:|---:|---:|']
    for c in KEYS:
        vals=[r for r in masses if r['condition']==c and r['layer']==2]
        lines.append('| '+c+' | '+' | '.join(f'{100*np.mean([r[s+"_mass"] for r in vals]):.3f}' for s in ['bos','image','question','template'])+' |')
    lines += ['', '`layer_source_mass.csv`保留每题每条件L0–L5的四来源Mass、相对Full变化及有效Attention输出范数；`hidden_query_changes.csv`保存L2–L5的层输入/输出Hidden、post-RoPE Query和实际BOS K/V变化。全部条件BOS K/V逐位不变已断言；这不排除Query及其他Key竞争造成Mass变化。',
        '输出置零位于Attention权重计算之后，因此EarlyOFF的L0权重不变是预期现象，不是干预失败。Pathway Mask通过原生加性因果Mask克隆添加−inf；目标概率归零，保留其余合法边并由原生Softmax重新归一化。',
        '', '## 图表', '',
        '![Accuracy变化](figures/accuracy_effects.png)',
        '![条件性来源依赖](figures/conditional_source_dependence.png)',
        '![六类Accuracy](figures/task_accuracy.png)',
        '![L2–L5来源Mass变化](figures/layer_source_mass_changes.png)',
        '![逐题受损交集](figures/error_intersections.png)',
        '', '所有PNG都有同名矢量PDF。以下6个代表案例在结果产生前固定，每类按抽样哈希取第一题，不按正确性、Margin或变化选例。Patch图显示实际CLIP336裁剪；原始Image Mass写在标题，24×24分布为对Query/Head求原始均值后再视觉内部归一化，并明确采用对数色标。逐Head图为L2各条件BOS/Image/Question原始Mass，不是视觉内部概率。']
    for name in result['cases']:
        lines += ['',f'### {name}',f'![视觉Patch](cases/{name}/visual_patch_maps.png)',f'![逐Head来源Mass](cases/{name}/head_source_mass.png)']
    lines += ['', '## 复用、验证和审计', '',
        '- 历史版本：f1e661fd（Phase3.7路径代码/16例先导，只审查不重跑）、c67bc7d1（原3,000题L0/L1 OFF）、e495adf4（完整Full/评分/数据绑定）。Phase3.7使用的DirectedSuffixKnockout、knockout_mask及Token定位原语与本分支逐文件哈希一致；旧封装不能处理缓存且绑定两层，本轮仅在新目录增加单层、Prefill-only缓存时序适配。',
        f'- 成功采集冻结代码：`{result["code_commit"]}`；源码、样本、协议、CPU验证哈希均绑定到检查点。独立分支 `codex/early-source-delay-240`；不修改main或历史文件。',
        '- 固定LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager、Transformers4.49.0，Greedy/32/EOS2，原Prompt、评分器及条件独立KV Cache。环境和空闲GPU检查见smoke.json。',
        '- 哈希抽样精确定义、全部题号/图像SHA256/字符和Token位置/历史绑定保存在sample_manifest.json；真实Question逐字符重建精确，不包含末尾答题指令。',
        '- 复用240条Full完整答案；每题一次原生cached Prefill仅用于来源Mass/Hidden观测及历史首logits逐位核对，不是完整生成。EarlyOFF首logits逐题精确核对历史P−1+Decode OFF，但完整答案全部重新以P−1-only生成。',
        '- CPU审计与两例真实GPU Smoke先于正式240题。Smoke检查11条件No-op每步logits和初始KV指纹精确Full；目标权重归零、相同局部Hidden下非目标Query权重逐位不变；所有32层MLP正常、Prefill/Decode时序、独立HF生成、BOS KV、原生恢复及Hook/Input清理。',
        '- 首次CPU异常注入测试误替换Attention.forward，触发原有签名保护而不是预期的运行时异常；失败记录保留。改为在q_proj注入异常，并新增签名保护测试；没有禁用保护或放宽容差。其他失败如发生，同样记录于failures/和attempts/，不得将其伪装通过。',
        f'- 正式采集/观测阶段耗时约{result["completion"]["seconds"]/60:.1f}分钟，不含加载、Smoke和CPU统计；不据此推断算子加速。逐题逐条件原子检查点支持安全断点续跑。',
        '- 只使用预检空闲的物理GPU2，不终止其他任务；工作结束释放模型，释放记录见gpu_release.csv。未训练、未扩展至3,000题、未扫描额外层/窗口或额外A×B条件。',
        '', '## 交付文件', '',
        '`protocol.json` / `sample_manifest.json` / `execution_manifest.json`；`per_question.csv`（2,880行，含240条复用Full）；`accuracy_summary.csv` / `task_accuracy.csv`；`source_interactions.csv` / `task_source_interactions.csv`；`layer_source_mass.csv` / `hidden_query_changes.csv`；`error_intersections.csv` / `per_question_error_patterns.csv`；PNG/PDF、6例小型NPZ；CPU/GPU验证、运行日志、完成记录和result.json的产物哈希。',
        '', '**解释底线：** Attention Mass改变、被屏蔽路径的因果贡献、L2功能性补偿，以及具体视觉语义整合不是同一个结论。本轮240题是探索性样本，不把假说作为既定结论，也不把不显著等同于不存在。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
