"""Prespecified question-weighted image-cluster statistics and publication figures."""
from .common import *
import numpy as np,collections,math
def paired(a,b,images,seed=SEED):
 d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float);unique=sorted(set(images));ix={s:i for i,s in enumerate(unique)};ind=np.array([ix[s] for s in images]);tot=np.bincount(ind,weights=d,minlength=len(unique));counts=np.bincount(ind,minlength=len(unique));rng=np.random.default_rng(seed);boot=[];extreme=0
 for start in range(0,20000,200):
  n=min(200,20000-start);weights=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=n);boot.extend((weights@tot)/(weights@counts)*100)
  signs=rng.choice([-1,1],size=(n,len(unique)));extreme+=int(np.count_nonzero(np.abs(signs@tot)>=abs(tot.sum())-1e-12))
 lo,hi=np.percentile(boot,[2.5,97.5])
 return dict(delta_pp=float(d.mean()*100),CI95_low_pp=float(lo),CI95_high_pp=float(hi),score_up=int((d>1e-12).sum()),score_down=int((d< -1e-12).sum()),score_equal=int((abs(d)<=1e-12).sum()),cluster_signflip_p=(extreme+1)/20001,n_questions=len(d),n_image_clusters=len(unique))
def loadrows(dataset):
 d=OUT/dataset;rows=list(csv.DictReader((d/'per_question.csv').open()));by=collections.defaultdict(dict)
 for r in rows:
  r['score']=float(r['score']);r['R_ratio']=float(r['R_ratio']);r['k']=int(r['k']);r['R']=int(r['R']);by[r['question_id']][r['condition']]=r
 assert all(set(v)==set(CONDITIONS) for v in by.values())
 return rows,by
def analyze(dataset):
 d=OUT/dataset;assert read(d/'validation.json')['passed'];rows,by=loadrows(dataset);summary=[]
 for name in CONDITIONS:
  rr=[v[name] for v in by.values()];sc=np.array([r['score'] for r in rr]);images=[r['image_sha256'] for r in rr];ci=paired(sc,np.zeros(len(sc)),images)
  summary.append(dict(dataset=dataset,condition=name,n=len(rr),score_sum=float(sc.sum()),VQA_accuracy=float(sc.mean()*100),CI95_low_pct=ci['CI95_low_pp'],CI95_high_pct=ci['CI95_high_pp'],mean_R_ratio=float(np.mean([r['R_ratio'] for r in rr])),pooled_R_ratio=sum(r['k'] for r in rr)/sum(r['R'] for r in rr),k_mean=float(np.mean([r['k'] for r in rr])),scorer='Official VQA soft consensus; ten annotators'))
 allacc=summary[0]['VQA_accuracy']
 for r in summary:r['delta_vs_All_pp']=r['VQA_accuracy']-allacc
 csvwrite(d/'accuracy_summary.csv',summary)
 comparisons=[]
 for left,right in read(OUT/'protocol.json')['comparisons']:
  aa=[v[left] for v in by.values()];bb=[v[right] for v in by.values()];r=paired([v['score'] for v in aa],[v['score'] for v in bb],[v['image_sha256'] for v in aa]);r.update(dataset=dataset,left=left,right=right,answer_changed=sum(x['answer']!=y['answer'] for x,y in zip(aa,bb)),first_token_flip=sum(x['first_token']!=y['first_token'] for x,y in zip(aa,bb)));comparisons.append(r)
 prev=0
 for j,r in enumerate(sorted(comparisons,key=lambda r:r['cluster_signflip_p'])):prev=max(prev,min(1.,(6-j)*r['cluster_signflip_p']));r['Holm_p_6within_dataset']=prev
 csvwrite(d/'paired_comparisons.csv',comparisons)
 types=[];triggers={r['question_id']:r for r in csv.DictReader((d/'trigger_analysis.csv').open())}
 for axis in ['answer_type','first_word_group','trigger15','trigger18']:
  for value in sorted({triggers[q][axis] for q in by}):
   sub=[v for q,v in by.items() if triggers[q][axis]==value]
   for name in CONDITIONS:
    vals=[v[name]['score'] for v in sub];deltas=[v[name]['score']-v['Question-Only']['score'] for v in sub]
    types.append(dict(dataset=dataset,axis=axis,value=value,condition=name,n=len(sub),accuracy=100*float(np.mean(vals)),delta_vs_Q_pp=100*float(np.mean(deltas)),score_up=sum(x>1e-12 for x in deltas),score_down=sum(x< -1e-12 for x in deltas)))
 csvwrite(d/'answer_type_accuracy.csv',types)
 cases=[]
 for q,v in by.items():
  for left,right in read(OUT/'protocol.json')['comparisons']:
   a,b=v[left],v[right];diff=a['score']-b['score']
   if abs(diff)>1e-12:cases.append(dict(question_id=q,image_sha256=a['image_sha256'],question=a['question'],answers=a['answers'],left=left,right=right,left_answer=a['answer'],right_answer=b['answer'],left_score=a['score'],right_score=b['score'],delta_score=diff,kind='harm' if diff<0 else 'improvement'))
 csvwrite(d/'changed_score_cases.csv',cases,fields=['question_id','image_sha256','question','answers','left','right','left_answer','right_answer','left_score','right_score','delta_score','kind'])
 csvwrite(d/'harm_cases.csv',[r for r in cases if r['kind']=='harm'],fields=['question_id','image_sha256','question','answers','left','right','left_answer','right_answer','left_score','right_score','delta_score','kind'])
 # Known non-overlap subset is explicit rather than inferred from dataset names.
 subset=[v for v in by.values() if v['All-Writer']['metadata_overlap']=='False' and v['All-Writer']['SHA_overlap']=='False'];over=[]
 for name in CONDITIONS:over.append(dict(condition=name,n=len(subset),accuracy=100*float(np.mean([v[name]['score'] for v in subset])) if subset else None,scope='No known metadata or exact-byte overlap with historical GQA3000; re-encoded identity may remain unknown'))
 csvwrite(d/'known_nonoverlap_accuracy.csv',over)
 return summary,comparisons,types,by
def gqa_structure():
 prior=OUT.parent/'semantic_single_gpu_search3h';ids=read(prior/'split_manifest.json')['metadata'];r=[];budget=[]
 f15=read(prior/'phase6_fixed_compositions/winner_freeze.json')['positions'];f18=read(prior/'phase7_fixed_refinements/winner_freeze.json')['positions']
 from .selection import firstword,trigger
 for q in sorted(ids):
  s=read(pathlib.Path('/data1/likai/ATE-worktrees/gqa5h-adaptive-token-receiver-search/vlm_research/results/gqa5h_receiver_search/input_specs')/(q+'.json'));w=firstword(s['question']);n=len(s['Q']);cat=w if w in ['what','which','how'] else 'yes/no' if w in ['is','are','does','do'] else 'where/who/why' if w in ['where','who','why'] else 'other'
  r.append(dict(question_id=q,first_word=w,first_word_group=cat,Q_tokens=n,R_tokens=len(s['R']),trigger15=trigger(s['question'],n,15),trigger18=trigger(s['question'],n,18)))
  for name,positions in [('All-Writer',s['R']),('Question-Only',s['Q']),('Threshold-15',f15[q]),('Threshold-18',f18[q])]:budget.append(dict(question_id=q,condition=name,k=len(positions),R=len(s['R']),R_ratio=len(positions)/len(s['R'])))
 csvwrite(OUT/'cross_dataset/gqa_historical_structure.csv',r);csvwrite(OUT/'cross_dataset/gqa_historical_budgets.csv',budget)
 return r,budget
def plots(results):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 D=OUT/'cross_dataset';D.mkdir(exist_ok=True);gstructure,gbudget=gqa_structure();names=['All-Writer','Question-Only','Threshold-15','Threshold-18','Budget-Matched Random'];short=['All','Q only','T15','T18','Random'];colors=['#606060','#0072B2','#D55E00','#009E73','#CC79A7']
 def save(fig,name):fig.savefig(D/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(D/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,1+len(results),figsize=(5*(1+len(results)),4));axes=np.atleast_1d(axes)
 axes[0].bar(range(4),[87.033333,85.833333,86.6,86.533333],color=colors[:4]);axes[0].set_title('GQA3000 (historical, explored)\nExact-match accuracy');axes[0].set_xticks(range(4),short[:4]);axes[0].set_ylabel('Accuracy (%)');axes[0].set_ylim(0,100)
 for ax,(dataset,(summary,cmp,_,by)) in zip(axes[1:],results.items()):
  vals=[x['VQA_accuracy'] for x in summary];err=np.array([[v-x['CI95_low_pct'] for v,x in zip(vals,summary)],[x['CI95_high_pct']-v for v,x in zip(vals,summary)]]);ax.bar(range(5),vals,color=colors,yerr=err,capsize=3);ax.set_xticks(range(5),short,rotation=20);ax.set_title(f'{dataset} (n={len(by)})\nOfficial VQA consensus score');ax.set_ylim(0,100)
 fig.suptitle('A. Frozen receiver rules across datasets\nDifferent scorers and datasets: absolute values do not measure a common difficulty scale',fontsize=12,y=1.09);fig.tight_layout();save(fig,'figure_A_cross_dataset_accuracy')
 structures={'GQA3000':gstructure}
 for dataset in results:structures[dataset]=list(csv.DictReader((OUT/dataset/'trigger_analysis.csv').open()))
 fig,axes=plt.subplots(1,3,figsize=(15,4));groups=['what','which','how','yes/no','where/who/why','other'];x=np.arange(len(structures));bottom=np.zeros(len(x))
 for category in groups:
  v=np.array([100*sum(r['first_word_group']==category for r in rows)/len(rows) for rows in structures.values()]);axes[0].bar(x,v,bottom=bottom,label=category);bottom+=v
 axes[0].set_xticks(x,list(structures),rotation=15);axes[0].set_ylabel('Questions (%)');axes[0].legend(fontsize=7);axes[0].set_title('First question word')
 axes[1].boxplot([[int(r['Q_tokens']) for r in rows] for rows in structures.values()],tick_labels=list(structures),showfliers=True);axes[1].axhline(15,color=colors[2],ls='--',lw=1);axes[1].axhline(18,color=colors[3],ls=':',lw=1);axes[1].set_title('Native Question-token length');axes[1].set_ylabel('Tokenizer subwords');axes[1].tick_params(axis='x',rotation=15)
 for k,t in enumerate([15,18]):
  v=[100*sum(str(r['trigger'+str(t)])=='True' for r in rows)/len(rows) for rows in structures.values()];axes[2].bar(x+(k-.5)*.3,v,width=.3,label=f'Threshold-{t}')
 axes[2].set_xticks(x,list(structures),rotation=15);axes[2].set_ylabel('Triggered questions (%)');axes[2].legend();axes[2].set_title('Frozen OR-trigger coverage');fig.suptitle('B. Question structure and receiver trigger distribution');fig.tight_layout();save(fig,'figure_B_question_structure')
 fig,axes=plt.subplots(1,len(results),figsize=(6*len(results),4));axes=np.atleast_1d(axes)
 for ax,(dataset,(_,cmp,_,by)) in zip(axes,results.items()):
  q=next(iter(by.values()));additional=[]
  for name in ['Threshold-15','Threshold-18','Budget-Matched Random']:
   a=[v[name]['score'] for v in by.values()];b=[v['Question-Only']['score'] for v in by.values()];r=paired(a,b,[v['All-Writer']['image_sha256'] for v in by.values()]);additional.append(dict(dataset=dataset,condition=name,**r))
  csvwrite(D/(dataset+'_gain_plot_data.csv'),additional)
  for i,r in enumerate(additional):ax.errorbar(i,r['delta_pp'],yerr=[[r['delta_pp']-r['CI95_low_pp']],[r['CI95_high_pp']-r['delta_pp']]],fmt='o',color=colors[i+2],capsize=5)
  ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(3),['T15','T18','Random']);ax.set_ylabel('VQA score difference (pp)');ax.set_title(dataset)
 fig.suptitle('C. Gains over Question-Only\nPaired image-cluster bootstrap, pointwise exploratory 95% CI');fig.tight_layout();save(fig,'figure_C_gain_over_question')
 fig,axes=plt.subplots(1,len(results),figsize=(6*len(results),4));axes=np.atleast_1d(axes)
 for ax,(dataset,(summary,_,_,_)) in zip(axes,results.items()):
  for i,r in enumerate(summary):ax.scatter(r['mean_R_ratio']*100,r['VQA_accuracy'],color=colors[i],label=short[i],s=60);ax.annotate(short[i],(r['mean_R_ratio']*100,r['VQA_accuracy']),xytext=(4,5+7*(i%2)),textcoords='offset points',fontsize=8)
  ax.set_xlabel('Mean allowed R-to-image receiver budget (%)');ax.set_ylabel('Official VQA accuracy (%)');ax.set_title(dataset);ax.grid(alpha=.2)
 fig.suptitle('D. Functional receiver budget vs. answer quality\nDense computation retained; budget is not FLOPs or latency');fig.tight_layout();save(fig,'figure_D_budget_accuracy')
def report(results):
 lines=['# 固定Question-Conditional Assistant规则的跨数据集验证','',f'执行源码基线：`644834d3e22194b4982758dabd59d3e479ef8ce3`。本轮仅用LLaVA-1.5-7B、单张RTX3090（UUID `{UUID}`），没有训练、调阈值或修改Prompt。15/18阈值来自反复探索的GQA3000，本轮仅在冻结的新问题上验证。','', '## 协议和数据','', '所有32层P−1与真实Decode都禁止直接读取原始Image Key；Question始终开放，触发时额外开放原生ASSISTANT的4个完整Subword。原生Full与这里All-Writer不同。五个条件标签完整保留；同题仅输入哈希与接收位置完全相同者共享生成，缓存均由该唯一生成从空开始独立构建。','', 'VQAv2使用官方VQAEval归一化与10人留一共识，TextVQA使用固定MMF/EvalAI评分器。主要指标是平均VQA软分数，不是单Gold字符串正确率；1/2/3人匹配得0.3/0.6/0.9分。GQA历史87.0333%、85.8333%、86.6000%、86.5333%为Exact Match，不能跨评分协议直接比较绝对能力。','', '数据按固定SHA顺序选取，每图一道；优先排除VG→COCO/Flickr已知历史GQA3000图片及字节SHA重叠。未匹配不能排除重新编码、缺失元数据或预训练重叠。详细来源、图像SHA与下载记录见dataset_sources、sample_manifest和dataset_overlap_audit。','', '随机触发对照在完全相同Question Token长度与R长度分层内置换T15标签，种子20261104，触发题数、额外4-Token数量、平均k/R及合并k/R均匹配。某些分层没有可置换的触发差异，须以changed_labels判断实际控制对比强度。','', '## 结果','']
 for dataset,(summary,cmp,types,by) in results.items():
  lines += [f'### {dataset}（{len(by)}题）','', '| 条件 | VQA Accuracy % | 相对All pp | 平均R预算 % | 合并R预算 % |','|---|---:|---:|---:|---:|']
  for r in summary:lines.append(f"| {r['condition']} | {r['VQA_accuracy']:.3f} | {r['delta_vs_All_pp']:+.3f} | {100*r['mean_R_ratio']:.3f} | {100*r['pooled_R_ratio']:.3f} |")
  lines += ['', '| 配对比较 | 差值pp [普通95%CI] | Score升/降/不变 | Holm p |','|---|---:|---:|---:|']
  for r in cmp:lines.append(f"| {r['left']} − {r['right']} | {r['delta_pp']:+.3f} [{r['CI95_low_pp']:+.3f}, {r['CI95_high_pp']:+.3f}] | {r['score_up']}/{r['score_down']}/{r['score_equal']} | {r['Holm_p_6within_dataset']:.4f} |")
  trigger=list(csv.DictReader((OUT/dataset/'trigger_analysis.csv').open()));n=len(trigger);length=np.array([int(r['Q_tokens']) for r in trigger]);audit=read(OUT/dataset/'random_trigger_audit.json');overlap=read(OUT/dataset/'dataset_overlap_audit.json');validation=read(OUT/dataset/'validation.json')
  counts=collections.Counter(r['first_word_group'] for r in trigger)
  lines += ['', f"问题结构：what {100*counts['what']/n:.2f}%，which {100*counts['which']/n:.2f}%，how {100*counts['how']/n:.2f}%，yes/no首词 {100*counts['yes/no']/n:.2f}%，where/who/why {100*counts['where/who/why']/n:.2f}%。Question长度均值{length.mean():.2f}、中位数{np.median(length):.1f}，范围{length.min()}–{length.max()}；≥15占{100*np.mean(length>=15):.2f}%，≥18占{100*np.mean(length>=18):.2f}%。T15/T18触发率{100*sum(r['trigger15']=='True' for r in trigger)/n:.2f}%/{100*sum(r['trigger18']=='True' for r in trigger)/n:.2f}%。随机与T15实际不同标签{audit['changed_labels']}题。",'', f"重叠审计：已知元数据重叠{overlap['metadata_overlap']}题，字节SHA重叠{overlap['SHA_overlap']}题；未发现匹配不等于完全图像独立。输出异常（按五个逻辑条件计数）：{validation['abnormal']}。",'', '各答案类型、首词和触发子组的Score增加/损害完整保存在answer_type_accuracy.csv；全部预设比较的损害例保存在harm_cases.csv。']
  for threshold in [15,18]:
   q=next(r for r in cmp if r['left']==f'Threshold-{threshold}' and r['right']=='Question-Only');random=next(r for r in cmp if r['left']==f'Threshold-{threshold}' and r['right']=='Budget-Matched Random')
   control_note='与T15等预算的随机' if threshold==15 else 'T15预算匹配的随机（与T18不保证等预算）'
   lines += ['',f"解释：Threshold-{threshold}相对Question-Only点估计{'提高' if q['delta_pp']>0 else '降低' if q['delta_pp']<0 else '不变'}{abs(q['delta_pp']):.3f}pp；相对{control_note}为{random['delta_pp']:+.3f}pp。"+('没有观察到超过随机的正向点估计，不能宣称语言触发机制已验证。' if random['delta_pp']<=0 else '点估计优于随机，但仍需结合区间和六比较Holm校正，不能仅据点估计宣布机制成立。')]
  lines += ['']
 lines += ['## 统计、正确性与边界','', '两个数据集分别统计，不合并Accuracy。每项20,000次Image SHA整簇配对Bootstrap，保持问题等权，seed20261104；表中为普通点位95%CI。每数据集固定六比较的双侧簇符号翻转p值另做Holm校正。CI跨0不是等效或非劣效证明，本轮未定义/检验质量非劣效界限。数据集规模有限、只一组随机种子，不能把单数据集差异泛化为普适机制。','', 'CPU审计逐题原生Offset、完整词组、R/Q/I/P−1和真实ASSISTANT四Subword，并对历史两条规则各3000题逐位置回归。真实GPU Smoke检查Native/No-op全部有效Logits严格torch.equal、禁止概率严格0、非目标Mask不变、32层MLP、原图特征不变且参与计算、独立KV和异常Hook清理。自由生成使用logits_to_keep=1，保留EOS；冻结输入与源码SHA逐题复核。没有保存冗余全词表Logits归档。','', 'ASSISTANT是LLaVA原生模板中的接收位置，本实验没有换模板或模型，因而任何收益仍可能强烈依赖LLaVA模板；没有证明跨模板或跨模型成立。阈值来自GQA搜索，新数据上的点估计、随机控制及不确定区间共同判断是否存在GQA特化，不能因外部数据集名称不同就排除图像或预训练重叠。','', '本轮只验证功能性视觉接收者。密集Attention、完整QKV/MLP/Residual/KV都保留，预算不是计算/延迟节省。是否纳入Physical Visual Handoff须以外部质量、随机控制证据与单独工程验证共同决定；本轮不执行加速、训练或追加搜索。','', '## 图表和复现','', 'cross_dataset/figure_A–D包含PNG及矢量PDF；图A明确区分GQA Exact Match与VQA共识。作图数据来自全部冻结样本，图B区分GPU样本与完整验证集CPU首词统计。完整原始生成、Token IDs、输入/Checkpoint索引及评分保存在各数据集目录。','', '运行：在vlm_research目录下，依次 `python -m receiver_cross_dataset.prepare`、`python -m receiver_cross_dataset.run`、`python -m receiver_cross_dataset.analyze`。数据准备脚本data_acquisition.py只缓存所需图像；源文件与依赖哈希见source_manifest.json/protocol.json。','', '官方来源：[VQAv2下载](https://visualqa.org/download.html)、[官方VQA评分](https://visualqa.org/evaluation.html)、[VQAEval实现](https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py)、[TextVQA官方数据](https://huggingface.co/datasets/facebook/textvqa)、[MMF评分](https://github.com/facebookresearch/mmf/blob/main/mmf/utils/m4c_evaluators.py)。']
 (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
def main():
 results={}
 for dataset in ['vqa_v2','textvqa']:
  if (OUT/dataset/'completion.json').exists() and read(OUT/dataset/'completion.json')['status']=='GENERATED_AND_VERIFIED':results[dataset]=analyze(dataset)
 assert 'vqa_v2' in results
 plots(results);report(results)
 for file in ['accuracy_summary.csv','paired_comparisons.csv','per_question.csv','answer_type_accuracy.csv','question_structure_summary.csv','trigger_analysis.csv','writer_budget.csv']:
  combined=[]
  for dataset in results:
   for r in csv.DictReader((OUT/dataset/file).open()):combined.append({'dataset':dataset,**r})
  csvwrite(OUT/file,combined)
 write(OUT/'sample_manifest.json',dict(datasets={d:dict(path=f'{d}/sample_manifest.json',sha256=sha(OUT/d/'sample_manifest.json'),n=len(result[3])) for d,result in results.items()}))
 write(OUT/'random_trigger_audit.json',{d:read(OUT/d/'random_trigger_audit.json') for d in results});write(OUT/'dataset_overlap_audit.json',{d:read(OUT/d/'dataset_overlap_audit.json') for d in results})
 write(OUT/'validation.json',dict(passed=True,UTC=utc(),datasets={d:read(OUT/d/'validation.json') for d in results},source_hashes_verified=sources()==read(OUT/'source_manifest.json')['files'],all_statistics_recomputed=True,bootstrap_replicates=20000,scorers_distinct=True))
 write(OUT/'completion.json',dict(status='COMPLETE' if len(results)==2 else 'PARTIAL',UTC=utc(),datasets=list(results),logical_records=sum(len(r[3])*5 for r in results.values()),unique_generations=sum(read(OUT/d/'completion.json')['unique_generations'] for d in results),no_training=True,no_tuning=True))
 print(json.dumps({d:result[0] for d,result in results.items()}),flush=True)
if __name__=='__main__':main()
