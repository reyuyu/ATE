import json,csv,sys,collections
from pathlib import Path
import numpy as np
P=Path(sys.argv[1]);D=json.loads((P/'analysis_data.json').read_text());C=D['cases'];G=list(csv.DictReader((P/'group_band_rankings.csv').open()));A=list(csv.DictReader((P/'case_scope_band_summary.csv').open()))
def display(r):return r['group_text'].strip() or ('[换行]' if '\n' in r['group_text'] else '[Image后空白]')
def main():
    done=json.loads((P/'gpu_completion.json').read_text());v=json.loads((P/'analysis_validation.json').read_text());lines=['# Native Full中的遗漏Semantic视觉接收者：16例Attention观察','',
    '**状态：COMPLETE。** 按用户后续明确授权，改为一张真正空闲的物理GPU3串行处理固定16例。原两卡资源停止记录保留在attempts/two_gpu_resource_stop。固定图片、Prompt、词组、Semantic集合与排序规则未改变。','',
    '本轮得到474,112行R逐层逐Head逐Subword观测，以及独立的16,384行P−1观测。仅采集Native Full Prefill，全部原始Attention连接开放；没有重跑3000题，没有Decode，没有新的完整答案或准确率搜索。卡片中的All-Writer/Semantic答案均来自历史记录，不是本轮Native Full生成。','',
    '## 五个问题的直接回答','',
    '**1. 哪些遗漏Token的视觉Mass最大？** 必须分层段和角色。L8–L11未选Question的最高例是10763612的名词can（0.2314），其次是09662582的train（0.1778）；L0–L15按均值排序靠前的是冠词the，例如08163732第一处the（0.2515）。L16–L19最高例为0481307的the（0.2556）。模板类最高常为Image之后的第一空白Token，不能把它误叫成ASSISTANT词。','',
    '**2. 集中在哪类Token？** 冠词经常靠前，但并不只出现在错误例；也观察到名词can/train、辅助词Is/Are/Do、个别问题末尾标点、Instruction及Assistant/template位置。大部分关系词在这些案例里原本已属于S，不能预设关系词被普遍漏选。具体原位置与选择状态都由冻结Manifest决定。','',
    '**3. 集中在哪些层？** 全案例原始Mass均值在L0–L3最高，不能据此把视觉写入定位成只有L8–L11。L8–L11有相对突出的遗漏名词；L16–L19冠词相对突出；L31更多出现句首辅助词和边界位置。各层段模式不同，并非单个集中写入窗口的证据。','',
    '**4. 错误与纠正案例是否不同？** 高Mass冠词并不专属于错误：L8–L11每例最高遗漏Question组为冠词的比例，错误例8/12、纠正反例3/4；L16–L19为10/12与4/4。两类共享明显模式。本批是事后固定案例，不做总体差异显著性推断，不能把16例中的频率外推为GQA3000比例。','',
    '**5. 后续等预算Token-Swap优先候选？** train @590（09662582，L8–L11）、a @600/the @584（08163732）、the @587（0481307）、? @595（03567478）以及Image后边界空白 @581值得作为不同角色的候选，在未来独立授权的对照实验里检验。每次仅更改接收集合，原Prompt保持完整；必须移除相同Token数的已选完整词组，并使用随机Swap及纠正反例作对照。can @588在10763612的Mass很高，但其原S为k=0，**无法执行保持原k的Swap**，不能将它包装成可行的等预算候选。真正ASSISTANT组有4个Subword，不能拆成单个A来补位。当前只提出候选，未执行任何Swap。','',
    '## 数量优势校正与关键数值','',
    '576个Image Key在这些R位置的Uniform Baseline接近1，因而原始Mass为0.2并不意味着按Key平均偏好图像。p0=576/实际合法Key数（本批均为q+1）；Excess=Mass−p0；Enrichment按每Head的用户公式计算。下面En为跨Head/层/Subword的有效Enrichment算术均值，不能与“先平均Mass再代入非线性公式”混淆。','',
    '| 案例 | 词组与位置 | 层段 | 原始Mass | Excess | Enrichment均值 |','|---|---|---|---:|---:|---:|']
    targets=[('10763612','can','L8-11',588),('09662582','train','L8-11',590),('08163732','a','L8-11',600),('0481307','the','L16-19',587),('001020372','the','L0-15',584)]
    for q,t,b,p in targets:
        r=next(r for r in G if r['question_id']==q and r['group_text']==t and r['band']==b and int(r['first_position'])==p);lines.append(f'| {q} | {t} @{p} | {b} | {float(r["mass_mean"]):.4f} | {float(r["excess_mean"]):+.4f} | {float(r["enrichment_mean"]):.4f} |')
    lines+=['','全部完整词组×层段的平均Excess均为负，Enrichment均值均小于1；个别Head仍可高于Uniform。相对Top5只表示该预定义池内更高，不等于绝对视觉偏好，更不等于因果必要性。','',
    '有1个原始Head/Query的ImageMass为1.0001424551（0160501，L11/H17，ASSISTANT的首Subword @606）。这是对FP16概率逐Key相加时的舍入现象，未裁剪或重归一化；按1−Mass公式其Enrichment不在有效域，CSV保留Mass/Excess并把Enrichment留空、写明原因。其余聚合的Enrichment均值仅使用有定义项，未制造无穷大。原生FP16行和误差容差沿用历史ROW_ATOL=0.002，记录实际误差；Logits及R→Image一致性仍严格torch.equal，未放宽容差。','',
    '## 角色与层段均值','',
    '以下为逐案例先平均、再跨案例平均的描述性数字。Selected Question只有15例（10763612的k=0，该组缺失而非填0），其余角色16例，因此不是同一分母的配对效应；Figure A保留每例点位及实际n。','',
    '| 层段 | Selected Q | Omitted Q | Instruction | ASSISTANT完整词 | 边界/换行 |','|---|---:|---:|---:|---:|---:|']
    for band in ['L0-3','L4-7','L8-11','L12-15','L16-19','L20-30','L31','L0-15','L0-19','L0-31']:
        vals=[np.mean([float(r['mass_mean']) for r in A if r['band']==band and r['scope']==scope and r['selected']==sel]) for scope,sel in [('Question','True'),('Question','False'),('Instruction','False'),('Assistant word','False'),('Boundary/separators','False')]];lines.append('| '+band+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
    lines+=['','Assistant/template原始角色池保持不变，另外将真正ASSISTANT词与边界/换行拆开做描述性显示，避免第一R空白的高Mass被错误归因于Assistant词。此分解不改变词组或冻结排名。','',
    '![Figure A](figure_A_selected_omitted.png)','','![Figure B](figure_B_heatmap_overview.png)','','![Figure C](figure_C_high_mass_omitted.png)','','## 逐题遗漏Question组Top1','',
    '| ID | 类型 | L8–L11 | L0–L15 | L16–L19 |','|---|---|---|---|---|']
    for c in C:
        q=c['case']['question_id'];cells=[q,c['case']['kind']]
        for band in ['L8-11','L0-15','L16-19']:
            r=next(r for r in G if r['question_id']==q and r['band']==band and r['scope']=='Question' and r['rank_mass']=='1');cells.append(f'{display(r)} @{r["first_position"]} ({float(r["mass_mean"]):.4f})')
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','完整每例、每层段、每角色Top5见high_mass_omitted_tokens.csv；全部排名与副指标排序见group_band_rankings.csv。逐Subword、跨Head均值/中位/最大值及逐层表分别保留，排名未使用单Head最大值。','',
    '## 正确性、资源和可复现性','',
    '固定16例来自425286a2566cc6b0387460492a0f99566b142a38的selected_cases.csv：12错误+4纠正。图片、Prompt、Native Tokenizer、Question Offset、Word Group和Semantic集合逐文件核对；全32层、32个Head的R位置为最后Image之后至P−1之前，P−1独立CSV。模型revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/Eager/Transformers4.49.0，logits_to_keep=1。','',
    '成功运行只加载一份模型到物理GPU3，CUDA_VISIBLE_DEVICES绑定UUID GPU-96bc853f-ceea-2404-c704-c48a3d1ca185。原case_manifest的rank0/1仅保留两卡计划元数据；用户授权后实际全部由worker0串行处理，没有遗漏或重复。所有前向计算在torch.inference_mode下进行，逐层提取实际output_attentions到CPU，完整矩阵不长期留在GPU；返回Attention输出、所有MLP、Residual、权重和输入保持原值。','',
    '两道Smoke固定001020372、14236062：Native plain/No-op所有有效Logits逐位一致；Native与All-Writer的全部32层×32Head×全部R×576Image Key概率张量严格torch.equal；原输入与视觉特征哈希不变，独立KV、Hook异常清理及全部Prefill概率非负/因果非法边概率0、行和正常通过。All-Writer只用于Smoke，主矩阵16例均为原生Native Full，不是Semantic Hard Mask。','',
    f'成功冻结运行共22次单Prefill前向：16个主观测+6个验证前向，0 Decode，0新答案；驱动墙钟约{done["wall_seconds"]:.2f}秒。前一次初始化审计失败中另有1次Native参考前向已执行、没有Attention观测进入统计；失败现场保留在attempts/serial_initialization_failure。模型初始化从inference_mode移出后重新冻结并重新检查空闲GPU，前向仍在该模式下，未放宽数值检查。两次模型实例先后退出，从未同时占两份显存。资源释放快照中GPU3约27MiB，其他任务未被抢占或终止。','',
    '## 文件与停止边界','',
    '- per_head_csv/：16份完整R逐题×层×Head×Subword原始数值，索引和SHA在per_head_index.json。P−1只在p_minus_one_per_head.csv，不混入R排序。','- token_layer_summary.csv、subword_band_summary.csv：跨Head及层段统计；group_layer_summary.csv、group_band_rankings.csv：完整词组聚合与排序。','- measurements/：小型Native Mass/合法Key数及Logits数组，足以重算全部指标；checkpoint含输入、权重审计和数组SHA。Smoke完整R→Image概率数组保存在服务器资源路径，路径与SHA见worker0/smoke_*.json，不声称GitHub包含这些大数组。','- gallery.html可离线打开，case_cards.pdf每例两页共32页；逐案例真实热图PNG位于heatmaps/；summary_figures.pdf含Figure A/B/C。全部图的CSV与绘图/渲染脚本均保留。','- 初始CPU-only预览已移入attempts/two_gpu_resource_stop，不作为当前Attention结果。没有把无数据状态当成低Mass。','',
    'L31的R Attention可以很高，但其R输出在当前因果计算图中不能继续传播到第一答案的P−1位置，不能据此宣布功能必要。可观测分配可能冗余、可由其他位置补偿，也可能是有用写入；仅本轮不能区分。','',
    '**结论：Semantic确实未覆盖一些在原生Full中分配较多图像Attention的接收者，包括少量被漏掉的名词、常见冠词及边界位置；这些模式也出现在Semantic纠正的反例中。它们是后续受控Swap的候选线索，不是“Semantic答错原因”的证明。** 本轮停止，不改变Semantic规则，不追加GPU实验、训练或准确率搜索。']
    (P/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT COMPLETE',flush=True)
if __name__=='__main__':main()
