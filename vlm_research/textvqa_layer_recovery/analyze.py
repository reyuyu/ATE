"""Fixed family statistics; every score and abnormal output is retained."""
from .common import *
from receiver_cross_dataset.analyze import paired
from receiver_cross_dataset.scoring import normalize
import numpy as np,collections,re,math
def holm(rows):
 previous=0.
 for j,r in enumerate(sorted(rows,key=lambda x:x['cluster_signflip_p'])):
  previous=max(previous,min(1.,(len(rows)-j)*r['cluster_signflip_p']));r['Holm_p_8']=previous
def load():
 assert read(OUT/'generation_validation.json')['passed']
 rows=read(OUT/'sample_manifest.json')['rows'];old=baseline_rows();by={};records=[];mass=[]
 for r in rows:
  q=r['question_id'];s=read(OUT/'input_specs'/(q+'.json'));by[q]={}
  for c in CONDITIONS:
   p=OLD/old[q]['checkpoint'] if c=='B' else OUT/'checkpoints'/q/(c+'.json');v=read(p);g=v['generation'];value=score(g['text'],r['answers']);assert abs(value-v['score'])<1e-12
   rr=dict(question_id=q,image_id=r['image_id'],image_sha256=r['image_sha256'],question=r['question'],answers=json.dumps(r['answers'],ensure_ascii=False),condition=c,condition_name=NAMES[c],answer=g['text'],score=value,token_ids=json.dumps(g['token_ids']),first_token=v['first_token'],length=g['length'],non_EOS_length=g['length']-int(g['eos']),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],R_tokens=len(s['R']),R_allowed_ratio=1.,Pminus1_allowed_layers=32 if c=='A' else 2 if c in ['C','E'] else 0,Decode_allowed_layers=32 if c=='A' else 2 if c in ['D','E'] else 0,checkpoint=str(p.relative_to(ROOT)),checkpoint_sha256=sha(p),origin='Audited historical B' if c=='B' else v['origin'])
   by[q][c]=rr;records.append(rr)
   for x in v.get('attention_mass',[]):mass.append(dict(question_id=q,condition=c,**{k:z for k,z in x.items() if k!='per_head_mass'}))
 for q,v in by.items():
  for c,r in v.items():r['score_delta_vs_B']=r['score']-v['B']['score'];r['score_delta_vs_A']=r['score']-v['A']['score'];r['answer_changed_vs_B']=r['answer']!=v['B']['answer'];r['token_sequence_changed_vs_B']=r['token_ids']!=v['B']['token_ids']
 csvwrite(OUT/'per_question.csv',records)
 for c in ['A','C','D','E']:csvwrite(OUT/('actual_attention_mass_'+c+'.csv'),[x for x in mass if x['condition']==c])
 return records,by,mass
def statistics(records,by):
 summary=[];images=[v['B']['image_sha256'] for v in by.values()]
 for c in CONDITIONS:
  rr=[v[c] for v in by.values()];vals=[r['score'] for r in rr];ci=paired(vals,np.zeros(len(vals)),images)
  summary.append(dict(condition=c,name=NAMES[c],n=len(rr),score_sum=sum(vals),accuracy_pct=100*float(np.mean(vals)),CI95_low_pct=ci['CI95_low_pp'],CI95_high_pct=ci['CI95_high_pp'],score_positive=sum(x>0 for x in vals),score_one=sum(abs(x-1)<1e-12 for x in vals),eos=sum(r['eos'] for r in rr),empty=sum(r['empty'] for r in rr),truncated=sum(r['truncated'] for r in rr),repeated_trigram=sum(r['repeated_trigram'] for r in rr),mean_tokens=float(np.mean([r['length'] for r in rr])),Pminus1_allowed_layers=rr[0]['Pminus1_allowed_layers'],Decode_allowed_layers=rr[0]['Decode_allowed_layers'],R_allowed_ratio=1.,scorer='Official TextVQA MMF consensus, ten annotations'))
 base=summary[1]['accuracy_pct'];full=summary[0]['accuracy_pct']
 for r in summary:r['delta_vs_B_pp']=r['accuracy_pct']-base;r['delta_vs_A_pp']=r['accuracy_pct']-full
 csvwrite(OUT/'accuracy_summary.csv',summary);comparisons=[]
 for a,b in COMPARISONS:
  aa=[v[a] for v in by.values()];bb=[v[b] for v in by.values()];p=paired([r['score'] for r in aa],[r['score'] for r in bb],images)
  p.update(left=a,right=b,answer_changed=sum(x['answer']!=y['answer'] for x,y in zip(aa,bb)),token_sequence_changed=sum(x['token_ids']!=y['token_ids'] for x,y in zip(aa,bb)),first_token_flip=sum(x['first_token']!=y['first_token'] for x,y in zip(aa,bb)),length_changed=sum(x['length']!=y['length'] for x,y in zip(aa,bb)),mean_length_delta=float(np.mean([x['length']-y['length'] for x,y in zip(aa,bb)])))
  for flag in ['eos','truncated','repeated_trigram','empty']:
   p[flag+'_gained']=sum(x[flag] and not y[flag] for x,y in zip(aa,bb));p[flag+'_lost']=sum(y[flag] and not x[flag] for x,y in zip(aa,bb));p[flag+'_changed']=p[flag+'_gained']+p[flag+'_lost']
  comparisons.append(p)
 inter=paired([v['E']['score']+v['B']['score'] for v in by.values()],[v['C']['score']+v['D']['score'] for v in by.values()],images);inter['formula']='E-C-D+B';holm(comparisons+[inter]);csvwrite(OUT/'paired_comparisons.csv',comparisons);csvwrite(OUT/'interaction.csv',[inter])
 cases=[]
 for q,v in by.items():
  for a,b in COMPARISONS:
   x,y=v[a],v[b];d=x['score']-y['score']
   if abs(d)>1e-12:cases.append(dict(question_id=q,image_sha256=x['image_sha256'],question=x['question'],answers=x['answers'],left=a,right=b,left_answer=x['answer'],right_answer=y['answer'],left_score=x['score'],right_score=y['score'],delta_score=d,kind='improvement' if d>0 else 'harm',left_truncated=x['truncated'],right_truncated=y['truncated']))
 fields=['question_id','image_sha256','question','answers','left','right','left_answer','right_answer','left_score','right_score','delta_score','kind','left_truncated','right_truncated']
 csvwrite(OUT/'changed_score_cases.csv',cases,fields);csvwrite(OUT/'harm_cases.csv',[r for r in cases if r['kind']=='harm'],fields);csvwrite(OUT/'E_vs_B_cases.csv',[r for r in cases if r['left']=='E' and r['right']=='B'],fields)
 abnormal=[r for r in records if r['empty'] or r['truncated'] or r['repeated_trigram'] or not r['eos']];csvwrite(OUT/'abnormal_outputs.csv',abnormal,fields=list(records[0]))
 return summary,comparisons,inter
def descriptions(by,mass):
 rows=[];assignments=[]
 for q,v in by.items():
  r=v['B'];words=re.findall(r"[A-Za-z]+",r['question']);first=words[0].lower() if words else 'other';length=r['non_EOS_length'];answers=json.loads(r['answers']);mode=collections.Counter(normalize(a) for a in answers).most_common(1)[0][0]
  groups=dict(first_word=first,historical_B_answer_length='1' if length<=1 else '2-4' if length<=4 else '5+',annotation_proxy='contains_digit' if any(ch.isdigit() for ch in mode) else 'alphabetic_or_other',historical_B_abnormal=str(r['truncated'] or r['repeated_trigram'] or r['empty']))
  assignments.append(dict(question_id=q,modal_normalized_annotation=mode,**groups))
  for axis,value in groups.items():
   for c,x in v.items():rows.append(dict(axis=axis,value=value,condition=c,score=x['score'],B_score=r['score'],A_score=v['A']['score']))
 grouped=collections.defaultdict(list)
 for r in rows:grouped[(r['axis'],r['value'],r['condition'])].append(r)
 table=[]
 for (axis,value,c),rr in grouped.items():table.append(dict(axis=axis,value=value,condition=c,n=len(rr),accuracy_pct=100*float(np.mean([x['score'] for x in rr])),delta_vs_B_pp=100*float(np.mean([x['score']-x['B_score'] for x in rr])),delta_vs_A_pp=100*float(np.mean([x['score']-x['A_score'] for x in rr]))))
 csvwrite(OUT/'descriptive_groups.csv',table);csvwrite(OUT/'descriptive_assignments.csv',assignments)
 budgets=[]
 for c in CONDITIONS:
  for phase in ['Pminus1','Decode']:
   rr=[x for x in mass if x['condition']==c and x['phase']==phase]
   for l in range(32):
    ss=[x for x in rr if x['layer']==l]
    if c=='B':
     # Old B records audited every forbidden probability across all32 layers and steps.
     total=sum(1 if phase=='Pminus1' else v['B']['length']-1 for v in by.values());masked=1.;mm=0.;source='Historical actual forbidden-probability max=0; mass inferred exact zero, not remeasured'
    else:total=len(ss);masked=sum(x['blocked'] for x in ss)/total if total else None;mm=float(np.mean([x['image_mass_mean'] for x in ss])) if total else None;source='Fresh actual attention measurements'
    budgets.append(dict(condition=c,phase=phase,layer=l,query_count=total,masked_original_image_connections_ratio=masked,actual_image_mass_headmean=mm,source=source))
 csvwrite(OUT/'phase_layer_budget_mass.csv',budgets)
 return table,budgets
def plots(summary,cmp,budgets):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 colors=['#444444','#0072B2','#E69F00','#CC79A7','#009E73'];vals=[r['accuracy_pct'] for r in summary];err=[[v-r['CI95_low_pct'] for v,r in zip(vals,summary)],[r['CI95_high_pct']-v for v,r in zip(vals,summary)]]
 fig,ax=plt.subplots(figsize=(8,4));ax.bar(range(5),vals,color=colors,yerr=err,capsize=4);ax.set_xticks(range(5),['A Native\nFull','B All-\nWriter','C P-1\n14+19','D Decode\n14+19','E Both\n14+19']);ax.set_ylabel('Official TextVQA consensus accuracy (%)');ax.set_ylim(0,max(55,max(vals)+10))
 for i,v in enumerate(vals):ax.text(i,v+3.5,f'{v:.2f}',ha='center')
 ax.set_title('Same frozen TextVQA300; no OCR text\nPointwise image-cluster bootstrap 95% CI');save(fig,'figure_accuracy')
 fig,ax=plt.subplots(figsize=(8,4.5))
 for j,r in enumerate(cmp):ax.errorbar(r['delta_pp'],j,xerr=[[r['delta_pp']-r['CI95_low_pp']],[r['CI95_high_pp']-r['delta_pp']]],fmt='o',capsize=4,color='#0072B2');ax.text(r['CI95_high_pp']+.3,j,f"p(Holm-8)={r['Holm_p_8']:.3g}",va='center',fontsize=8)
 ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(7),[r['left']+' - '+r['right'] for r in cmp]);ax.invert_yaxis();ax.set_xlabel('Consensus score difference (percentage points)');ax.set_title('Seven fixed paired comparisons\nPointwise 95% CI; Holm family includes interaction');ax.margins(x=.25);save(fig,'figure_paired_comparisons')
 fig,axes=plt.subplots(2,1,figsize=(11,5.8))
 for ax,phase in zip(axes,['Pminus1','Decode']):
  values=np.array([[next(x['actual_image_mass_headmean'] for x in budgets if x['condition']==c and x['phase']==phase and x['layer']==l) for l in range(32)] for c in CONDITIONS]);im=ax.imshow(values,aspect='auto',vmin=0,vmax=max(x['actual_image_mass_headmean'] or 0 for x in budgets),cmap='viridis');ax.set_yticks(range(5),CONDITIONS);ax.set_xticks(range(0,32,2));ax.set_title(phase+' mean measured image attention mass (head/query mean)');ax.set_xlabel('Zero-based decoder layer');fig.colorbar(im,ax=ax,label='Probability mass')
 save(fig,'figure_actual_image_mass')
 fig,ax=plt.subplots(figsize=(7,4))
 for i,r in enumerate(summary):
  fraction=(r['Pminus1_allowed_layers']+r['Decode_allowed_layers'])/64;ax.scatter(fraction*100,r['accuracy_pct'],color=colors[i],s=70);ax.annotate(r['condition'],(fraction*100,r['accuracy_pct']),xytext=(5,3 if i%2 else -13),textcoords='offset points')
 ax.set_xlabel('Allowed answer-phase layer permissions (unweighted %)');ax.set_ylabel('Official TextVQA accuracy (%)');ax.set_title('Functional phase/layer permissions, not R receiver budget\nDense QKV, MLP and KV updates retained');save(fig,'figure_functional_budget')
def report(summary,cmp,inter):
 lookup={r['condition']:r for r in summary};text=['# TextVQA部分视觉层恢复：固定L14/L19','',f'基线Commit：`{BASE}`。相同300题、相同图片字节、相同无OCR Prompt、相同LLaVA-1.5-7B冻结权重。主指标是官方TextVQA/MMF十人留一共识软评分，不是单个Gold的字符串准确率。未训练、未重新选层、未筛题。','', '## 同题结果','', '| 条件 | P−1开放层 | Decode开放层 | Accuracy % | 相对B pp | 截断 | 重复3gram |','|---|---|---|---:|---:|---:|---:|']
 for r in summary:text.append(f"| {r['condition']} {r['name']} | {'全部32' if r['condition']=='A' else '14、19' if r['condition'] in ['C','E'] else '无'} | {'全部32' if r['condition']=='A' else '14、19' if r['condition'] in ['D','E'] else '无'} | {r['accuracy_pct']:.3f} | {r['delta_vs_B_pp']:+.3f} | {r['truncated']} | {r['repeated_trigram']} |")
 text+=['','A是完全原生Full，取消所有本实验的视觉限制；B为严格审计复用的历史All-Writer（300条）。C/D/E仅恢复对应阶段的原始Image Key连接。全部R（含ASSISTANT四个Subword）在五条件均开放，Image Query、非图像边、MLP、Residual和完整真实KV更新保持原生。A/C/D/E共1200条新生成，Smoke通过后复用其中32条，所有1500个条件标签保留。','', '## 预声明配对统计','', '| 比较 | Δpp [普通95%CI] | Score升/降/不变 | 回答/Token序列改变 | Holm-8 p |','|---|---:|---:|---:|---:|']
 for r in cmp:text.append(f"| {r['left']}−{r['right']} | {r['delta_pp']:+.3f} [{r['CI95_low_pp']:+.3f}, {r['CI95_high_pp']:+.3f}] | {r['score_up']}/{r['score_down']}/{r['score_equal']} | {r['answer_changed']}/{r['token_sequence_changed']} | {r['Holm_p_8']:.5f} |")
 text+=['',f"交互项E−C−D+B：{inter['delta_pp']:+.3f}pp，95%CI [{inter['CI95_low_pp']:+.3f}, {inter['CI95_high_pp']:+.3f}]，Holm-8 p={inter['Holm_p_8']:.5f}。这仅表示当前软评分和干预定义下的非加性，不证明物理协同机制。",'', '300题按Image SHA256整簇配对Bootstrap 20,000次，固定seed=20261104；保留题目等权。普通95%CI为逐点区间，不是多重校正区间。双侧整簇符号翻转随机化20,000次、plus-one p；七组比较和交互共八项执行Holm校正，均在GPU运行前冻结。没有定义非劣效界限，不能以CI跨0声明等价或非劣效。','', '## 结论','']
 text += [f"同题真正Native Full为{lookup['A']['accuracy_pct']:.3f}%；B为{lookup['B']['accuracy_pct']:.3f}%。仅恢复P−1的L14/L19（C）相对B为{lookup['C']['delta_vs_B_pp']:+.3f}pp；仅恢复Decode（D）为{lookup['D']['delta_vs_B_pp']:+.3f}pp；两阶段联合（E）为{lookup['E']['delta_vs_B_pp']:+.3f}pp。E相对Native Full仍为{lookup['E']['delta_vs_A_pp']:+.3f}pp。"]
 for r in cmp:
  text.append(f"{r['left']}−{r['right']}："+('Holm校正后p<0.05，配对差异获得统计支持。' if r['Holm_p_8']<.05 else 'Holm校正后未达到0.05；只能作为当前样本的探索性观察。'))
 text+=['', '本轮只能检验GQA提出的固定两层在这批TextVQA、无OCR Prompt、FP16/Eager条件中的效果，不能证明L14/L19优于其他任意两层；未执行随机两层对照。官方常见TextVQA模型成绩使用的OCR输入/完整数据协议不同，不能与本轮直接作差。小样本、历史已研究图片及固定输入限制外推。','', '## 异常与失败归因','']
 for r in summary:text.append(f"{r['condition']}：EOS {r['eos']}/300，截断{r['truncated']}，重复3gram {r['repeated_trigram']}，空答案{r['empty']}，平均生成Token {r['mean_tokens']:.2f}（含EOS）。")
 text+=['', '全部异常进入主评分；条件间新增/消失的EOS、截断、重复和长度改变详见paired_comparisons.csv，逐条详见abnormal_outputs.csv。未把正常达到32 Token上限视作资源OOM或删题理由。','', 'descriptive_groups.csv按问题首词、历史B回答长度、标注中是否包含数字及历史异常分组；数字/字母仅是答案形式代理，不是经人工确认的视觉识别需求。E相对B所有改善与损害保存在E_vs_B_cases.csv；全部预声明比较的损害在harm_cases.csv。观察到错误恢复不等于已定位OCR、知识或语言能力的唯一故障。','', '## 正确性与工程预算','', 'CPU验证重建全部300题原Prompt、Token与FP16图像输入哈希，逐字节对照冻结Git记录，重新核对所有模型文件及图片SHA。8题GPU Smoke包含历史异常：HF原生/No-op和A的全部有效Logits严格torch.equal；B与冻结引擎全部有效Logits严格相等，历史Token和首Token向量SHA匹配。Mask非目标元素不变，禁止实际概率精确为0，P−1随真实Prompt确定、Decode随真实Cache移动。层名映射、32层MLP、视觉特征、参数版本、独立Cache及正常/异常Hook清理均审计。未放宽历史行归一化数值容差。','', '实际注意力Mass按Head均值/中位数/最大/最小逐层逐真实查询记录；不要求开放层每步Mass必须为正。B未重跑：其Mass为历史实际禁止概率全0所推出的精确零，表中明确标注来源。phase_layer_budget_mass.csv按实际P−1/Decode调用给出Mask比例和Mass。所有条件R开放比例均100%；新恢复的是答案阶段层级权限，不能与R Token稀疏预算混作一个指标。','', '本轮仍执行完整QKV、Attention、MLP、Residual及KV更新；记录时间和显存属于此实现的运行审计，不证明FLOPs、延迟或显存优化。gpu_run_complete.json给出GPU阶段时间及峰值显存，gpu_release.json给出进程退出后资源。图片和14GB模型保留在服务器资源目录，Git交付不包含这些大文件；Manifest保留原始路径和SHA。','', '图表PNG/矢量PDF由analyze.py重现；CSV保存全部绘图数值。推理执行源码哈希在source_manifest.json，结果索引在artifact_index.csv。完成本轮后停止，未追加训练、层搜索或其他GPU实验。']
 (OUT/'report_zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
def main():
 records,by,mass=load();summary,cmp,inter=statistics(records,by);table,budgets=descriptions(by,mass);plots(summary,cmp,budgets);report(summary,cmp,inter)
 print(json.dumps(dict(summary=summary,paired=cmp,interaction=inter),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
