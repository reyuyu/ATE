"""Chinese report generated from completed measured tables, never planned numbers."""
import json
from collections import Counter
import numpy as np
from benchmarks.dataset_registry import freeze_json,sha256
from .protocol import ROOT,OUT,PROTOCOL
from .analyze import csv_read


def table(rows,columns):
    def fmt(value):
        if isinstance(value,(float,np.floating)):return f'{value:.6g}'
        return str(value).replace('|','/').replace('\n',' ')
    return ['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(fmt(r.get(k,'')) for k in columns)+' |' for r in rows]


def main():
    stats=csv_read(OUT/'bootstrap_statistics.csv');full=csv_read(OUT/'full_generation_per_sample.csv');allrows=csv_read(OUT/'layerwise_generation_per_sample.csv')
    selection=json.loads((OUT/'sample_manifest.json').read_text())['selection'];worker=json.loads((OUT/'execution_manifest.json').read_text())
    by={(r['metric'],int(r['layer'])):r for r in stats}
    def values(metric):return [by[(metric,k)] for k in range(32)]
    def low(metric,k):return by[(metric,k)]['family_noninferior']=='True'
    candidates=[k for k in range(32) if all(low(m,k) for m in ['CIDEr','SPICE','TextVQA'])]
    nominal=[k for k in range(32) if all(by[(m,k)]['ordinary_noninferior']=='True' for m in ['CIDEr','SPICE','TextVQA'])]
    corrected_harm={m:[k for k in range(32) if float(by[(m,k)]['family95_high'])<0] for m in ['CIDEr','SPICE','TextVQA']}
    coco_summary={int(r['layer']):r for r in csv_read(OUT/'layerwise_coco_quality_summary.csv')}
    decode_content=json.loads((OUT/'textvqa_full_decode_content_audit.json').read_text())
    ranked={m:sorted(range(32),key=lambda k:float(by[(m,k)]['delta'])) for m in ['CIDEr','SPICE','TextVQA']}
    layers=lambda seq:', '.join('L'+str(k) for k in seq) or '无'
    score_rows=[]
    for k in range(32):
        score_rows.append(dict(layer='L'+str(k),CIDEr=float(by[('CIDEr',k)]['ablated_score']),dCIDEr=float(by[('CIDEr',k)]['delta']),
            SPICE=float(by[('SPICE',k)]['ablated_score']),dSPICE=float(by[('SPICE',k)]['delta']),TextVQA_pct=100*float(by[('TextVQA',k)]['ablated_score']),
            TextVQA_dpp=100*float(by[('TextVQA',k)]['delta']),all3_family_NI=k in candidates))
    def detail(m,k):
        r=by[(m,k)];scale=100 if m=='TextVQA' else 1
        return f"L{k}: Δ={float(r['delta'])*scale:+.6f}, 普通95%CI[{float(r['ci95_low'])*scale:+.6f},{float(r['ci95_high'])*scale:+.6f}], 96项校正区间[{float(r['family95_low'])*scale:+.6f},{float(r['family95_high'])*scale:+.6f}]"+('pp' if m=='TextVQA' else '')
    base={m:float(values(m)[0]['full_score']) for m in ['CIDEr','SPICE','TextVQA']}
    lines=['# 32层 Decode Attention 功能必要性与完整生成质量', '',
        '**状态：正式400个样本、33条件、13,200条真实生成及官方评分全部完成。**', '',
        '## 核心回答','',
        f"1. COCO CIDEr平均下降最大的5层：{layers(ranked['CIDEr'][:5])}；SPICE对应：{layers(ranked['SPICE'][:5])}。两指标不能互相替代，完整尾部和区间见CSV。",
        f"2. 两任务三端点均满足预设96项校正Bootstrap非劣效筛选的候选层：**{layers(candidates)}**。仅普通95%区间通过：{layers(nominal)}。这是探索候选，不等于安全证明。",
        '3. 单步原生Token LogP与完整自由生成质量是不同观测；下面提供历史9/10个实测离散层的描述性相关，不能补造其余层单步结果，也不能把相关当成因果。',
        f"4. 仅建议将以上跨任务校正候选（若有）提交下一阶段独立研究决策。本轮没有进行任何多层联合消融；单层低损害不保证联合安全。",
        '5. 200图探索、有限自动指标与稀有错误缺失，不能建立通用低损害或真实计算加速。普通区间包含0不是无损；校正筛选也不能取代独立确认。','',
        f"补充：三端点联合筛选未通过不等于所有层均有害。校正区间整体低于0的层分别为：CIDEr {layers(corrected_harm['CIDEr'])}；SPICE {layers(corrected_harm['SPICE'])}；TextVQA {layers(corrected_harm['TextVQA'])}。其它层可能是功效不足或指标分歧，应看逐层区间。",
        f"COCO L0-off出现{coco_summary[0]['truncated']}/200条64-Token截断，平均生成长度{float(coco_summary[0]['mean_length']):.2f}；质量损失伴随显著的终止异常。这不等于已分离出视觉事实错误的因果贡献。",'',
        '## 模型、数据、冻结协议','',
        f"- 基线提交`{PROTOCOL['base_commit']}`；冻结采集提交`{worker['code_commit']}`；模型`LLaVA-1.5-7B@{PROTOCOL['model_revision']}`，FP16/eager，Transformers4.49.0；未训练或改权重。",
        '- GPU0已有外部任务，未占用、终止或等待抢占。只使用物理GPU1–3，每卡独立模型；同一样本33条件固定同卡。',
        f"- COCO200张图，排除历史{selection['coco_historical_excluded_images']}个已用身份（包括Discovery、Token-role、Late Sources、Function Stage A/B/C及Smoke），另用1张独立Smoke。按固定种子20261030的输入哈希选样，不按输出筛选。",
        f"- TextVQA200题/200图，官方0.5.1验证集、每题完整10答案。可用本地不同图{selection['textvqa_local_unique_candidates']}，历史未用{selection['textvqa_unused_local_candidates']}，本次实际历史重叠{selection['textvqa_selected_historical_overlap']}；另1图Smoke。",
        '- 全部402图核查官方原始行、问题/参考答案、文件SHA256和解码RGB；不声称排除了所有感知近重复或数据分布偏差。来源、Prompt真实Token映射和划分先于GPU结果冻结。',
        '- COCO Prompt：`Describe the image in one sentence.`；最多64 Token。TextVQA原Question+冻结短答模板，最多32 Token，无额外OCR。Greedy、原位置与RoPE、正常KV Cache，EOS=2。','',
        '- 解码沿用原流程skip_special_tokens=True及首尾strip，所有条件相同；不替换生成内容。完整原始Token IDs另行保留，文本变化率与Token变化率分别记录。','',
        '## 干预与Cache语义','',
        '- 全Prompt Prefill的32层Attention全部正常；第一个Token始终由Native Prefill产生。只从后续真实Decode开始，每一步将指定单层当前Query的`o_proj`输出置零。其他Attention、所有MLP/LayerNorm/残差保持正常。',
        '- 使用真实cache_position、Prompt长度与该层历史Cache长度判定Decode，不只检查sequence_length=1。单Token Prefill CPU反例也覆盖。',
        '- 经Smoke证明数值等价后，每图计算一次正常Prefill，再为33条件分别深复制完整Cache；无存储别名。每条生成消费自身真实预测前缀，不接Full后续Token。父Cache指针/形状逐条件检查、SHA逐图前后核对。',
        '- 本轮是Attention输出更新消融，不是删除Transformer Block，不省去底层Attention计算，不报告速度收益。','',
        '## 正确性验证','',
        f"- {json.loads((OUT/'cpu_validation.json').read_text())['tests']}项CPU测试通过；目标32层、Hook异常清理、阶段判定、独立Cache、EOS/长度、固定选样、重复/哈希绑定及Bootstrap逻辑均覆盖。",
        f"- 另{json.loads((OUT/'analysis_cpu_validation.json').read_text())['tests']}项后处理/断点恢复测试通过。最初后处理测试因环境无SciPy而失败，已改为复用仓库原有相关函数；失败日志保留，没有安装/更改运行环境，也未修改GPU采集代码。",
        '- 每张GPU均对独立COCO/TextVQA Smoke验证：cached Full与原生model.generate Token完全一致；32层No-op的全部生成步有效Logits精确一致；全部32条件的Prefill末位有效Logits/完整Cache精确一致；指定输出确为零，32层Attention与MLP仍执行；Hook/Input/父Cache无污染。',
        '- L0/L14/L31进一步比较各自新算Prefill与共享Prefill的完整干预轨迹和Logits精确一致。三卡同样本Full Logits最大差异记录在gpu_smoke_validation.json，未放宽容差。',
        '- 正式400图均保存33条件检查点、代码/数据/结果哈希、首Token、逐图父Cache/Input审计。全表无非有限分数、NaN/Inf/OOM或跳过失败样本。','',
        '## 官方评分与统计','',
        f"Full：CIDEr **{base['CIDEr']:.6f}**，SPICE **{base['SPICE']:.6f}**，TextVQA十答案软评分 **{base['TextVQA']*100:.3f}%**。TextVQA不是严格二元准确率。",'',
        '- 同一200图的COCO人工参考、PTBTokenizer、CIDEr IDF和SPICE评分器固定用于全部条件。评分器/Java模型SHA与相同参考fixture已验证。未添加没有基础设施的CHAIR指标。',
        '- 按图片配对Bootstrap200,000次；TextVQA若多题同图按图片聚类（本次每图1题）。CIDEr区间基于固定参考IDF的逐图分数差，不在重采样中重定义IDF。',
        '- 主比较集合预冻结为32层×3端点=96项。报告普通双侧95%及Bonferroni家族95%区间；极端分位数alpha/192，每尾约52.08个Bootstrap抽样，存在有限Monte Carlo误差；百分位Bootstrap并非严格有限样本保证。',
        '- 非劣效阈值在本轮GPU结果之前冻结：原始CIDEr0.02、SPICE0.01、TextVQA0.01即1pp。判据为区间下界严格大于负阈值。完整负面结果保留，不以均值接近0宣告等价。',
        '- 即便某层在200图中没有观测损害，退化的经验Bootstrap也不能给未出现的稀有错误建立安全上界。本轮不扩大数据或自动执行独立确认。','',
        '## 全32层实测质量','']
    lines+=table(score_rows,['layer','CIDEr','dCIDEr','SPICE','dSPICE','TextVQA_pct','TextVQA_dpp','all3_family_NI'])
    lines+=['','## 层段分析','']
    for title,indices in [('早层L0-L8',range(9)),('中层L14-L16',[14,15,16]),('深层L26/L31',[26,31])]:
        lines+=['### '+title,'']
        for metric in ['CIDEr','SPICE','TextVQA']:
            order=sorted(indices,key=lambda k:float(by[(metric,k)]['delta']))
            lines.append('- '+metric+' 本层段下降最大：'+detail(metric,order[0]))
            lines.append('- '+metric+' 本层段下降最小：'+detail(metric,order[-1]))
        lines+=['不能从这些单层结果推断相邻层联合关闭的效果；不同任务的敏感峰必须分别看待。','']
    lines+=[f"早层L0–L8：COCO两端点均通过校正筛选的层为{layers([k for k in range(9) if low('CIDEr',k) and low('SPICE',k)])}；TextVQA通过的层为{layers([k for k in range(9) if low('TextVQA',k)])}。",
        '此前TextVQA L0–L2三层Decode窗口的损害不能被单层均值解释完毕；有无单层损害都不能证明三层联合无损。','']
    for k in [26,31]:
        lines.append(f"L{k}完整生成：CIDEr变化{float(by[('CIDEr',k)]['delta']):+.6f}，SPICE变化{float(by[('SPICE',k)]['delta']):+.6f}，TextVQA变化{100*float(by[('TextVQA',k)]['delta']):+.3f}pp；其方向和区间应与历史视觉词单步LogP高敏感分别解释，不直接等同。")
    lines+=['## 生成变化、EOS与TextVQA长度','']
    for unit in ['coco_caption','textvqa']:
        subset=[r for r in full if r['dataset']==unit];content=np.array([int(r['content_tokens']) for r in subset]);off=[r for r in allrows if r['dataset']==unit]
        lines.append(f"- {unit} Full内容Token：均值{content.mean():.3f}、中位数{np.median(content):.1f}、范围{content.min()}–{content.max()}；至少2个内容Token {sum(content>=2)}/200。Full EOS {sum(r['eos']=='True' for r in subset)}/200；32层总截断{sum(r['truncated']=='True' for r in off)}/6400。")
    lines+=['- TextVQA按Full内容长度1、2–3、4+分组的评分、增减题数及文本变化见`textvqa_length_analysis.csv`。1内容Token题的Decode主要预测EOS；这种局限不能证明对一般多Token回答不重要。',
        f"- 更严格的后续内容审计：{decode_content['with_substantive_decode']}/200题在t>=1存在非特殊且包含字母/数字的Token；{decode_content['only_eos_or_whitespace_punctuation_decode']}/200题后续仅EOS/空白/标点。此定义包含子词，不等于独立词数或人工语义判定。逐题证据见textvqa_full_decode_content_audit.csv。",
        '- `caption_change_analysis.csv`与逐题表保存首个不同Token、原生词类/词和替换Token。词类沿用冻结词表，是事后诊断，不是随机词类因果效应；Unknown/多义/标点/EOS均保留。',
        '- 分数提升不等于视觉事实改善，文本改变也不自动等于质量损害。提前EOS/短文本只描述，不当作性能或加速证据。','',
        '## 与历史预测敏感性的关系','',
        '- GQA首答案Token研究：Prefill P−1单次消融，先前L14–L16敏感；本轮所有Prefill完全正常，且任务为COCO/TextVQA，因此不是直接复现同一个量。',
        '- Token-role Decode研究：在Full真实前缀上单步关闭一层，然后停止；其原生Token LogP变化不包含生成轨迹反馈。本轮在干预后自身前缀持续Decode并用最终任务指标评价。',
        '- 历史9层Token-role与10层Late Sources的Whole-off均值，与本轮相同层的质量损失进行描述性Pearson/Spearman比较；不同图片队列、相互依赖的层，不能当成独立样本做因果或总体显著性推断。','']
    correlations=csv_read(OUT/'historical_quality_correlations.csv')
    lines+=table(correlations,['source','role','metric','layers','pearson','spearman']) if correlations else ['历史相关暂无可比数据，不能补造。']
    lines+=['','## 真实案例与语义核查','',
        'Figure6按预定展示规则从全体配对中选自动指标最大下降、最大上升、分数接近但文本改变，以及截断/语义风险候选。完整排序在quality_win_loss_cases.csv；此处选择用于解释，不用于统计选样。',
        '自动CIDEr/SPICE提升不构成人工确认。代理图像条件化复核见case_visual_review.md：最大CIDEr增益的L25-off主要省略与原图大体相容的服装细节，并非已证实纠正幻觉；公交车例仅增加逗号、核心语义不变；L0大象例存在重复字符/换行和未生成EOS的可核实异常。该复核不是独立人工盲审，不估计总体幻觉率。','',
        '## 图表','',
        '所有图PNG/PDF，L0–L31全部实测点，不平滑/插值；普通与家族区间在前三图分别显示。','']
    for name in ['figure1_coco_cider','figure2_coco_spice','figure3_textvqa','figure4_cross_task_sensitivity','figure5_caption_change_and_harm','figure6_actual_cases']:
        lines+=['!['+name+'](figures/'+name+'.png)','']
    lines+=['## 结论边界与下一阶段','',
        f"- 若开展下一阶段，优先预注册候选：{layers(candidates)}，先做独立确认，再考虑多层联合干预。没有通过校正阈值的层不能凭点估计认定安全。",
        '- 本轮没有通过三端点共同筛选的层，因此不建议直接将任何层组合当作低损害策略。如果仅用于设计未来独立单层确认，L8/L12的COCO点估计变化较小且TextVQA通过本轮校正阈值，可作为探索线索；这不是冻结筛选通过，也不是联合消融证据。本轮不启动后续实验。',
        '- 不同任务、生成长度、自动参考覆盖差异都可能导致层敏感性不同。单步预测敏感性、局部Attention关联、完整生成必要性和真实硬件加速必须分开。',
        '- 本次只在冻结LLaVA-1.5-7B上进行400样本探索；不推广跨模型、不训练分类器、不执行来源裁剪、不做多层联合或计算加速。','',
        '## 可复现产物','',
        '`experiment_config.json`、`sample_manifest.json`、`execution_manifest.json`、每卡环境与日志、CPU/GPU验证、全部逐条件检查点、官方评分器输出、逐题CSV、Bootstrap统计、历史对照、六图PNG/PDF、失败清单、完成清单与哈希。',
        '外置Smoke原始Logits仅用于审计，绝对路径和SHA在GPU验证记录中；不上传图像或模型权重。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    short=['# 中文研究总结','',f"Full: CIDEr {base['CIDEr']:.6f}; SPICE {base['SPICE']:.6f}; TextVQA {base['TextVQA']*100:.3f}%.",'',
        f"COCO CIDEr损害最大：{layers(ranked['CIDEr'][:5])}。SPICE损害最大：{layers(ranked['SPICE'][:5])}。TextVQA损害最大：{layers(ranked['TextVQA'][:5])}。",'',
        f"三端点96项校正非劣效候选：{layers(candidates)}；仅探索，不认定安全。",'',
        f"COCO L0-off有{coco_summary[0]['truncated']}/200条达到64-Token上限；不能从早层Prefill/单步低敏感性推出持续Decode关闭安全。",
        'L14–L16没有复现为COCO CIDEr损害峰。历史单步敏感性与完整生成质量没有稳定统一对应，详细离散层Pearson/Spearman及协议差异见报告。',
        '本轮400样本×33条件=13,200次真实完整生成；Prefill和首Token原生，仅后续Decode单层Attention输出置零。所有MLP保持，三卡同样本配对。',
        '单步原生Token LogP不等于自由生成质量；历史相关性见report.md。文本变化/平均损失小不证明语义安全，单层不推导多层，Hook置零不代表加速。',
        '已保存全部结果、负向尾部、普通与96比较校正区间、六组PNG/PDF及真实图像案例；下一步候选交由研究决策，本轮不自动扩展。']
    (OUT/'research_summary.md').write_text('\n'.join(short)+'\n')
    freeze_json(OUT/'main_findings.json',dict(full=base,largest_harm_layers={m:v[:5] for m,v in ranked.items()},family_noninferior_cross_task_candidates=candidates,
        ordinary_noninferior_cross_task_candidates=nominal,report_sha256=sha256(OUT/'report.md')))
    print('\n'.join(short),flush=True)


if __name__=='__main__':main()
