"""Reproducible figures and Chinese interpretation; no model inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,sys,json,csv,collections,html,re,datetime
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(O))
from analysis import *
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):fig.tight_layout();fig.savefig(O/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(O/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def pt(ax,x,y,lo,hi,color,label=None,marker='o'):
 ax.errorbar(x,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt=marker,color=color,capsize=2,markersize=4,label=label)
def efffmt(r):return f"{float(r['delta_pp']):+.3f}pp [{float(r['CI95_low_pp']):+.3f}, {float(r['CI95_high_pp']):+.3f}]；Holm p={float(r['Holm_p']):.4f}"
def main():
 assert read(O/'analysis_validation.json')['passed'];s=loadcsv(O/'ocr_group_accuracy.csv');effects=loadcsv(O/'paired_ocr_effects.csv');inter=loadcsv(O/'ocr_vs_nonocr_interaction.csv');div=loadcsv(O/'decode_divergence.csv');ds=loadcsv(O/'decode_divergence_summary.csv');features=loadcsv(O/'descriptive_features.csv');labs=loadcsv(O/'ocr_labels.csv');history=loadcsv(H/'per_question.csv');by=collections.defaultdict(dict)
 for r in history:by[r['question_id']][r['condition']]=r
 groups=['OCR-Direct','OCR-Reasoning','Non-OCR Visual','Ambiguous'];short={'OCR-Direct':'OCR Direct','OCR-Reasoning':'OCR Reasoning','OCR-Combined':'OCR Combined','Non-OCR Visual':'Non-OCR','Ambiguous':'Ambiguous'};colors=['#444444','#999999','#E69F00','#009E73','#56B4E9','#CC79A7','#0072B2'];conds=['A','B','F','G','Both-12','Both-16','Both-24'];names=['Native A','No re-access B','P-1 only F','Decode only G','Both 12','Both 16','Both 24'];lookup={(r['group'],r['condition']):r for r in s}
 fig,ax=plt.subplots(figsize=(12,6))
 for j,c in enumerate(conds):
  for i,g in enumerate(groups):
   r=lookup[(g,c)];pt(ax,i+(j-3)*.105,float(r['accuracy_pct']),float(r['CI95_low_pct']),float(r['CI95_high_pct']),colors[j],names[j] if i==0 else None)
 ax.set_xticks(range(4),[short[g]+f"\nn={lookup[(g,'A')]['n_questions']}" for g in groups]);ax.set_ylabel('Official TextVQA consensus accuracy (%)');ax.set_ylim(-3,103);ax.set_title('A. Accuracy by frozen OCR-dependency category\nAI input-only labels; ordinary pointwise image-cluster 95% CI');ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.11),fontsize=9);save(fig,'figure_A_ocr_accuracy');csvwrite(O/'figure_A_plot_data.csv',[lookup[(g,c)] for g in groups for c in conds])
 fig,(ax,ix)=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1.25,1]});eindex={(r['group'],r['effect']):r for r in effects};en=['Full_reaccess_effect','Pminus1_effect','Decode_effect','Both16_effect'];enames=['A - B','F - B','G - B','Both16 - B'];glist=['OCR-Combined','OCR-Direct','OCR-Reasoning','Non-OCR Visual','Ambiguous']
 for i,g in enumerate(glist):
  for j,name in enumerate(en):
   r=eindex[(g,name)];x=float(r['delta_pp']);ax.errorbar(x,i+(j-1.5)*.15,xerr=[[x-float(r['CI95_low_pp'])],[float(r['CI95_high_pp'])-x]],fmt='o',color=colors[j+2],capsize=2,label=enames[j] if i==0 else None,markersize=4)
 ax.set_yticks(range(5),[short[g]+f" (n={lookup[(g,'A')]['n_questions']})" for g in glist]);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Paired re-access effect (percentage points)');ax.set_title('B1. Within-category effects');ax.legend(fontsize=8,ncol=2)
 for i,g in enumerate(['OCR-Combined','OCR-Direct','OCR-Reasoning']):
  for j,name in enumerate(en):
   r=next(r for r in inter if r['OCR_group']==g and r['effect']==name);x=float(r['interaction_pp']);ix.errorbar(x,i+(j-1.5)*.15,xerr=[[x-float(r['CI95_low_pp'])],[float(r['CI95_high_pp'])-x]],fmt='o',color=colors[j+2],capsize=2,markersize=4)
 ix.set_yticks(range(3),[short[g]+'\nminus Non-OCR' for g in ['OCR-Combined','OCR-Direct','OCR-Reasoning']]);ix.invert_yaxis();ix.axvline(0,color='gray',lw=.8);ix.set_xlabel('Difference of paired effects (pp)');ix.set_title('B2. Observed-label interactions');fig.suptitle('Ordinary 95% CIs; separate Holm families: 25 effects and 15 interactions',fontsize=11);save(fig,'figure_B_paired_effects');csvwrite(O/'figure_B_plot_data.csv',effects);csvwrite(O/'figure_B_interaction_data.csv',inter)
 fig,ax=plt.subplots(figsize=(10,6));ks=[0,2,4,8,12,16,24,32];cc=['B','E','Both-4','Both-8','Both-12','Both-16','Both-24','A'];curve=[];gcols=['#0072B2','#56B4E9','#E69F00','#009E73','#CC79A7'];gcur=['OCR-Combined','OCR-Direct','OCR-Reasoning','Non-OCR Visual','Ambiguous']
 for j,g in enumerate(gcur):
  for i,(k,c) in enumerate(zip(ks,cc)):
   r=lookup[(g,c)];curve.append(dict(layer_budget=k,**r));pt(ax,k+(j-2)*.19,float(r['accuracy_pct']),float(r['CI95_low_pct']),float(r['CI95_high_pct']),gcols[j],short[g]+f" (n={r['n_questions']})" if i==0 else None,marker=['o','s','^','D','x'][j])
 ax.set_xticks(ks);ax.set_ylim(-3,103);ax.set_xlabel('Allowed layers in both P-1 and Decode');ax.set_ylabel('Official TextVQA consensus accuracy (%)');ax.set_title('C. Measured budget points by frozen OCR category\nPointwise 95% CI; horizontal offsets only for visibility; no interpolation');ax.legend(fontsize=8,ncol=3);save(fig,'figure_C_ocr_budget');csvwrite(O/'figure_C_plot_data.csv',curve)
 fig,(ax,bx)=plt.subplots(1,2,figsize=(13,5.5));bins=['2','3-4','5+','unchanged'];bincols=['#E69F00','#56B4E9','#0072B2','#aaaaaa'];dplot=[]
 for i,g in enumerate(groups):
  dd=[r for r in div if r['category']==g];counts=[]
  for bn in bins:
   n=sum((r['divergence_kind']=='unchanged') if bn=='unchanged' else (r['first_divergence_position_1based']=='2') if bn=='2' else (r['first_divergence_position_1based'] in ['3','4']) if bn=='3-4' else (bool(r['first_divergence_position_1based']) and int(r['first_divergence_position_1based'])>=5) for r in dd);counts.append(n);dplot.append(dict(group=g,n=len(dd),first_divergence_bin=bn,count=n))
  bottom=0
  for j,n in enumerate(counts):ax.bar(i,n,bottom=bottom,color=bincols[j],label=bins[j] if i==0 else None);bottom+=n
 ax.set_xticks(range(4),[short[g]+f"\nn={lookup[(g,'A')]['n_questions']}" for g in groups]);ax.set_ylabel('Number of questions');ax.set_title('D1. First G/B token divergence (1-based)\nPosition 1: zero changes in all 300');ax.legend(title='Generation position',fontsize=8)
 metrics=['first_content_subword_changed','first_english_word_changed','any_token_divergence'];mn=['First visible subword','First English word proxy','Any token divergence']
 for j,m in enumerate(metrics):
  for i,g in enumerate(groups):
   r=next(r for r in ds if r['group']==g and r['metric']==m);pt(bx,i+(j-1)*.17,float(r['percentage']),float(r['CI95_low_pct']),float(r['CI95_high_pct']),colors[j+2],mn[j] if i==0 else None)
 bx.set_xticks(range(4),[short[g] for g in groups]);bx.set_ylim(-3,103);bx.set_ylabel('Changed questions (%)');bx.set_title('D2. Literal first token is not first answer word\nOrdinary image-cluster 95% CI');bx.legend(fontsize=8);save(fig,'figure_D_decode_divergence');csvwrite(O/'figure_D_plot_data.csv',dplot);csvwrite(O/'figure_D_change_data.csv',ds)
 # All length distributions use actual output records, not a synthetic averaged answer.
 lengths=[];lmap={r['question_id']:r for r in labs}
 for g in GROUPS+['All300']:
  use=[q for q in by if g=='All300' or lmap[q]['category']==g or (g=='OCR-Combined' and lmap[q]['category'].startswith('OCR-'))]
  for c in CONDS:
   rr=[by[q][c] for q in use]
   for kind,values in [('generated_tokens_excluding_EOS',[int(r['length'])-int(flag(r['eos'])) for r in rr]),('generated_whitespace_words',[len(r['answer'].split()) for r in rr])]:
    lengths.append(dict(group=g,condition=c,unit=kind,n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p25=float(np.percentile(values,25)),p75=float(np.percentile(values,75)),min=min(values),max=max(values)))
 csvwrite(O/'length_distributions.csv',lengths)
 # Case cards in Markdown and self-contained HTML, retaining actual predictions and soft scores.
 import base64
 cases=read(O/'case_selection.json')['cases'];md=['# OCR与视觉回看：冻结规则选择的案例','','每类按Question ID升序取第一个满足预定条件的案例；分类已先冻结。软分数不是二值正确率。完整损害记录见negative_cases.csv。',''];cards=[]
 for r in cases:
  if not r['question_id']:md+=['## '+r['category'],'','无符合冻结条件的案例。',''];cards.append('<article><h2>'+r['category']+'</h2><p>No qualifying case.</p></article>');continue
  md+=['## '+r['category']+' / '+r['question_id'],'',r['question'],'',f"类别：{r['classification']}。依据：{r['rationale']}",'',f"![原图缩略图]({r['thumbnail']})",'',f"原图：`{r['image_path']}`；SHA256 `{r['image_sha256']}`。",'', '官方10个参考答案：`'+json.dumps(r['answers'],ensure_ascii=False)+'`','', '| 条件 | 实际回答 | Soft Score | 相对B首处分叉（1-based） |','|---|---|---:|---:|']
  rows=[]
  for p in r['predictions']:
   md.append(f"| {p['condition']} | {p['answer'].replace('|',' / ')} | {p['score']:.1f} | {p['first_divergence_vs_B_1based'] or '相同'} |");rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [p['condition'],p['answer'],p['score'],p['first_divergence_vs_B_1based'] or 'unchanged'])+'</tr>')
  md+=[''];im=base64.b64encode((O/r['thumbnail']).read_bytes()).decode();cards.append('<article><h2>'+html.escape(r['category']+' / '+r['question_id'])+'</h2><img src="data:image/jpeg;base64,'+im+'"><h3>'+html.escape(r['question'])+'</h3><p>'+html.escape(r['classification']+': '+r['rationale'])+'</p><p>References: '+html.escape(json.dumps(r['answers']))+'</p><table><tr><th>Condition</th><th>Actual answer</th><th>Soft score</th><th>First divergence vs B</th></tr>'+''.join(rows)+'</table><p>Original SHA256: '+r['image_sha256']+'</p></article>')
 (O/'case_studies.md').write_text('\n'.join(md).rstrip()+'\n');(O/'case_gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>TextVQA OCR dependency cases</title><style>body{font:16px sans-serif;max-width:1100px;margin:25px auto;color:#222}article{padding:25px 0;border-bottom:1px solid #ccc;page-break-after:always}img{max-width:95%;max-height:480px;object-fit:contain}td,th{padding:7px;border:1px solid #ccc;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}p{overflow-wrap:anywhere}</style><h1>Frozen-case OCR dependency and visual re-access</h1><p>Input-only AI labels; not human ground truth. First qualifying Question ID per prespecified category.</p>'+''.join(cards))
 counts=collections.Counter(r['category'] for r in labs);lens=loadcsv(O/'decode_length_interaction.csv');dg={(r['group'],r['metric']):r for r in ds};di=next(r for r in inter if r['OCR_group']=='OCR-Combined' and r['effect']=='Full_reaccess_effect');nocr=[r for r in effects if r['group']=='Non-OCR Visual'];neg=loadcsv(O/'negative_cases.csv');ocrgain=sum(float(by[q]['A']['score'])-float(by[q]['B']['score']) for q in by if lmap[q]['category'].startswith('OCR-'));totalgain=sum(float(by[q]['A']['score'])-float(by[q]['B']['score']) for q in by);share=ocrgain/totalgain
 lines=['# TextVQA的OCR依赖性与视觉回看机制分析','', '本轮复用冻结TextVQA300和5040条历史记录，未加载LLaVA、未启动GPU、未生成任何新答案。官方TextVQA十人共识软评分重算差异为0。分类标签在逐题效应统计前冻结，全部300题保留。','', '## 标签、证据与限制','',f"按冻结协议，本次单Agent图像/问题复核标注：OCR-Direct {counts['OCR-Direct']}题，OCR-Reasoning {counts['OCR-Reasoning']}题，Non-OCR Visual {counts['Non-OCR Visual']}题，Ambiguous {counts['Ambiguous']}题。Combined为前两类共{counts['OCR-Direct']+counts['OCR-Reasoning']}题。每题对应一张不同图片。不能将这些AI标注称为已确证的人类真值；`human_review_queue.csv`保留高优先级复核项，全部标签均需要独立人类核验。",'', '先前总体成绩和旧报告中的少数案例已在本对话可见，所以不是完全独立盲审。此次300题分类界面移除了Gold、模型回答、评分及Attention；分类依据为冻结原图、原问题及随后准确对齐的公开OCR文字。先冻结规则，再完成全部图像复核和标签SHA，之后读取逐题结果统计；标签未因收益修改。没有训练分类器或调用新的OCR/视觉模型。','', '模拟时钟的指针读时、颜色和可见计数等可不依赖文字；具体数字/字母转录属于OCR。纯字母品牌标识按文字处理，具有独立图形Logo或外观识别路径者保留不确定性。空间定位后照抄文字属于Direct；读字后比较、计数、条件判断、双核转数量等属于Reasoning。40道Ambiguous不并入主OCR或Non-OCR；不能把TextVQA数据集名称当逐题OCR证明。','', '官方TextOCR验证JSON为40,028,022字节，SHA256 `0d04c1c9f5ba99e90be5b0df9ab34f6237b3fbc55443f263133d5c49b6435918`；292/300张图片在官方集合内。逐图检查Image ID、文件名stem、尺寸和标注image_id；原TextVQA图片字节按冻结SHA验证，没有下载任何图像归档。未核验独立TextOCR图像归档的字节同一性。8张不匹配不影响原图复核，OCR词面辅助指标标记未知。官方OCR中的`.`表示不可读/非英文，不是没有文字。来源：[TextOCR官方数据说明](https://textvqa.org/textocr/dataset/)、[官方验证标注](https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json)。','', '## 分组成绩与配对效应','', '| 组别 | n | A | B | F | G | Both12 | Both16 | Both24 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for g in ['All300']+GROUPS:lines.append('| '+g+' | '+lookup[(g,'A')]['n_questions']+' | '+' | '.join(f"{float(lookup[(g,c)]['accuracy_pct']):.3f}%" for c in conds)+' |')
 lines+=['','每组所有12个300题条件及120子集条件分别保存在`ocr_group_accuracy.csv`、`stage2_group_accuracy.csv`，完整答案/Token/首内容子词变化、EOS/截断/重复数量一并保留。120题结果不与300题混算。','', '| 组别 | A−B | F−B | G−B | Both16−B |','|---|---|---|---|---|']
 for g in GROUPS:lines.append('| '+g+' | '+' | '.join(efffmt(eindex[(g,n)]) for n in en)+' |')
 lines+=['', '区间单位均为百分点，20,000次Image SHA整簇配对Bootstrap，固定seed=20261105。CI为普通点位95%区间，不是同时区间；组内25项检验、交互15项检验各自Holm控制。组内p为整簇sign-flip加一修正，交互p为中心化整簇Bootstrap零假设近似，不是随机化OCR标签试验。Direct、Combined重叠，不能当独立重复证据。小组与分层存在宽区间/低功效。','', '## OCR是否解释更大的回看收益？','', '| OCR组−NonOCR | 效应 | 交互pp [95%CI] | Holm p |','|---|---|---|---:|']
 for r in inter:lines.append(f"| {r['OCR_group']} | {r['effect']} | {float(r['interaction_pp']):+.3f} [{float(r['CI95_low_pp']):+.3f}, {float(r['CI95_high_pp']):+.3f}] | {float(r['Holm_p']):.4f} |")
 lines+=['',f"OCR-Combined的A−B效应减Non-OCR为{float(di['interaction_pp']):+.3f}pp，95%CI [{float(di['CI95_low_pp']):+.3f}, {float(di['CI95_high_pp']):+.3f}]，Holm p={float(di['Holm_p']):.4f}。"+('区间支持本批观察标签下存在效应差异，但不是OCR依赖性的独立因果识别。' if float(di['CI95_low_pp'])>0 and float(di['Holm_p'])<.05 else '不能据此确认OCR题比Non-OCR题具有更大的视觉回看效应；证据不足不等于二者效应相同。'),'',f"按分组加总，Combined题贡献了全体A−B净软分差的{100*share:.2f}%（净贡献、可能被负贡献抵消，不是机制解释率）。但Combined本来占{100*(counts['OCR-Direct']+counts['OCR-Reasoning'])/300:.2f}%样本。贡献占比与组间效应交互必须分开看，不能仅因大多数收益出现在大多数OCR样本就宣称OCR驱动全部23.4pp。",'', '## Decode分叉、首内容子词与长度','',f"G/B的300题首位置原始Logits SHA及首Token ID全部一致；完整Token轨迹改变{dg[('All300','any_token_divergence')]['count']}/300，第一个可见内容子词改变{dg[('All300','first_content_subword_changed')]['count']}/300，首个英文词文本代理改变{dg[('All300','first_english_word_changed')]['count']}/300。分叉位置为包含EOS的1-based生成位置；EOS分叉和严格前缀延伸单独记录。首个Tokenizer位置、非空白内容子词和首英文词不是同一对象；首英文词只取保存输出中的连续ASCII字母代理，不能保证是完整语义答案词（特别是URL、数字、截断）。",'', '逐题`decode_divergence.csv`保留B/G实际文本、Token、首处分叉、EOS及分叉处词面。是否处于OCR片段只按与官方OCR词的规范化精确重叠做描述性代理，不能把无匹配解释为非OCR或OCR失效；没有建立字符级视觉指向。','', '| 组别 | Gold长度划分 | 短/长n | G−B长−短pp [95%CI] |','|---|---|---|---|']
 for r in lens:lines.append(f"| {r['group']} | {r['axis']} | {r['n_short']}/{r['n_long']} | {float(r['Decode_gain_long_minus_short_pp']):+.3f} [{float(r['CI95_low_pp']):+.3f}, {float(r['CI95_high_pp']):+.3f}] |")
 lines+=['', '长度分组使用10个官方Gold的modal原文（并列取首次）及原Tokenizer无special的CPU编码，绝不按生成成败或生成长度分组。原Tokenizer SHA见analysis_validation。`answer_length_analysis.csv`含单词/多词和单Token/多Token全部组的A/B/F/G成绩及效应；`length_distributions.csv`保留各输出长度分布。长度关联不等于Decode对每个新增答案Token的因果收益。','', 'F−G分组比较如下；P−1只有一个Query而Decode有可变多个真实Query，差异不能解释为单位Query贡献：','']
 for g in GROUPS:lines.append('- '+g+'：'+efffmt(eindex[(g,'Pminus1_minus_Decode')]))
 lines+=['', '## 混杂、反例与解释边界','', '预设短/长问题、modal参考单/多词、单/多Token、含数字/无数字、what/非what及公开可读文字数量均完整分层报告，含空组，见confound_stratified_effects.csv。它们是形式代理，非OCR真值。图像复杂度、任务难度、模型OCR能力没有被控制；未拟合回归、Router或分类器，不挑选显著分层。小Non-OCR/Reasoning组使条件分层更稀疏。','',f"所有预定配对中的负差记录{len(neg)}条（同题不同配对可重复，不能当不同题数）。案例按冻结QuestionID规则选择，见case_studies.md及离线case_gallery.html；每类无符合条件者如实标记。增加层数造成逐题损害并未删除，异常输出仍计入官方评分。",'', '关于GQA：这批Non-OCR标签下的表现可作为同模型的描述性参照，但没有按问题难度、输出长度、图像分布对齐GQA，不能从18道Non-OCR题证明与GQA机制一致。','', '在固定计算图内，A/B/F/G的Mask干预支持直接视觉回看会改变部分问题的完整答案；OCR类别是事后观察标签，其与干预效应的关联不能识别“OCR导致回看必要性”。还不能排除其他信息保留机制替代反复访问原视觉KV。Attention Mass不是此轮分类依据，也不等于必要性。本轮未证明最优动态策略、跨模型普适、质量非劣效或实际加速。','', '## 对十二个问题的回答','',f"1–4. 本复核标注Direct {counts['OCR-Direct']}、Reasoning {counts['OCR-Reasoning']}、Combined {counts['OCR-Direct']+counts['OCR-Reasoning']}、Non-OCR {counts['Non-OCR Visual']}（{counts['Non-OCR Visual']/3:.2f}%）、Ambiguous {counts['Ambiguous']}；均为需独立复核的AI标签，不称确证OCR题数。",'', '5–6. 直接看上表的A−B组间交互及区间，不能用组内显著/不显著相互比较替代交互检验。','', '7. 单/多Token及单/多词的G−B长短交互已全部报告；是否支持更强收益由区间判断，不预设长答案更受益。','', '8. 以分组F−G配对差及区间判断，不从各自点估计排名；样本量不足的组不能给阶段重要性排序。','', '9. OCR各组的8个真实Both预算点、相对B提升和相对A缺口见figure_C_plot_data.csv；最小观察预算只针对预设路径，不是全局最小层集合。','', '10. Non-OCR与GQA仅可描述性讨论，不能证明相同机制。','', '11. B仍答对的OCR题、F/G方向不同、Non-OCR回看改善及高预算损害均按规则保存，不隐藏反例。','', '12. 结论属于固定Mask干预下的观察性效应异质性分析，尚不能把23.4pp收益归因为OCR本身，更不能推导必须反复读取原视觉KV。','', '## 图表与复现','']
 for name in ['figure_A_ocr_accuracy','figure_B_paired_effects','figure_C_ocr_budget','figure_D_decode_divergence']:lines+=['![实测图]('+name+'.png)','']
 lines+=['全部图同时提供矢量PDF、PNG和绘图CSV，无平滑、无插值。运行本目录`analysis.py`后运行`plots_report.py`，最后`validate.py`，只读历史记录与本地Tokenizer；不运行准备Worktree脚本以避免重建分类冻结。`classification_freeze.json`与`preclassification_freeze.json`分别记录标签及规则冻结SHA。公开标注子集及原图受控引用均已保存；完整40MB官方JSON留在/data资源目录，其URL及SHA供重取。','', '本轮未请求、占用或释放任何GPU；resource_audit.json保留CPU环境和当前进程快照，仅供审计。main与旧结果未修改。']
 lead=f"本批结果支持**精确文字读取依赖与视觉回看收益相关**：OCR-Combined的A−B为24.545pp，Non-OCR为1.667pp，交互为22.879pp，普通95%CI [16.265,29.159]，15项Holm p=0.000750。Direct驱动了主要差异；Reasoning仅12题，证据不足。该关联基于单次AI输入复核标签，不能视为OCR本身的因果效应。"
 lines[4:4]=[lead,'']
 # Report actual subgroup recovery, including unstable denominators, rather than only pointing to a CSV.
 rec=loadcsv(O/'group_budget_recovery.csv');extra=['','## OCR分组的实际预算恢复','', '| 组别 | k=2 | k=4 | k=8 | k=12 | k=16 | k=24 | k=32 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for g in GROUPS:
  rr=[next(r for r in rec if r['group']==g and r['budget_layers']==str(k)) for k in [2,4,8,12,16,24,32]];extra.append('| '+g+' | '+' | '.join(f"{100*float(r['Full_gap_recovery_ratio']):.1f}%" if r['Full_gap_recovery_ratio'] else 'NA' for r in rr)+' |')
 extra+=['','以上是(A−B)缺口恢复比例的描述性点估计，不是Accuracy。Non-OCR仅1.667pp缺口，个别条件可超过100%，不代表质量更稳定；Reasoning与Non-OCR重采样时存在零/负分母，故不提供伪稳定的ratio CI，抽样次数记录于group_budget_recovery.csv。各条件Accuracy本身的普通CI仍完整提供。','', 'OCR-Combined在12/16/24层为30.165%/33.554%/39.050%，分别比Native 42.603%低12.438/9.050/3.554pp；24层尚未达到Full−2pp点估计目标。Direct 24层37.043%相对Full40.783%仍低3.739pp。','', '多Token与多词不可混同：Combined按原Tokenizer分组，多Token的G−B增益比单Token高9.261pp，普通CI [0.297,16.313]；按空格词数分组，多词反而比单词低8.764pp，CI [−16.026,−1.137]。这些均为描述性、未做长度分层多重校正，且单Token仅30题，不能挑其中一种宣称通用长答案机制。','', 'Reasoning和Non-OCR的G−B逐题软分差均为0，Bootstrap退化成[0,0]，只反映本样本没有观察到分差，不证明总体Decode无效。Non-OCR的A−B仅一题有+0.3软分，经验Bootstrap不能覆盖未观察到的失败类型；需结合18题规模解读。']
 lines+=extra
 (O/'report_zh.md').write_text('\n'.join(lines).rstrip()+'\n')
 write(O/'report_summary.json',dict(UTC=utc(),counts=dict(counts),Combined_share_of_total_net_A_minus_B=share,Combined_sample_share=(counts['OCR-Direct']+counts['OCR-Reasoning'])/300,full_interaction=di,all300_divergence={k:dg[('All300',k)] for k in ['any_token_divergence','first_content_subword_changed','first_english_word_changed']},no_new_inference=True))
 print('PLOTS_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
