"""Frozen estimands, image-cluster resampling, official TextVQA soft scores."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import json,csv,hashlib,contextlib,collections,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from receiver_cross_dataset.common import read,write,sha,utc
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
OUT=ROOT/'results/textvqa_decode_l0_l1_factorial';CONDITIONS=read(OUT/'protocol.json')['conditions'];NAMES=['Native','Decode-None','A','B','C','D'];assert set(NAMES)==set(CONDITIONS);PROTO=read(OUT/'protocol.json');SEED=PROTO['seed'];REPS=PROTO['bootstrap_repetitions'];EV=TextVQAAccuracyEvaluator()
AUX={'B-A':{'B':1,'A':-1},'C-A':{'C':1,'A':-1},'D-B':{'D':1,'B':-1},'D-C':{'D':1,'C':-1},'C-B':{'C':1,'B':-1},'D-C-B+A':{'D':1,'C':-1,'B':-1,'A':1},**{c+'-Native':{c:1,'Native':-1} for c in ['A','B','C','D']}}
def csvout(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def official(text,answers):
 with open(os.devnull,'w') as z,contextlib.redirect_stderr(z):return EV.eval_pred_list([dict(pred_answer=text,gt_answers=answers)])
def cluster_arrays(x,images):
 labels=sorted(set(images));idx={h:i for i,h in enumerate(labels)};count=np.zeros(len(labels));sums=np.zeros((len(labels),x.shape[1]))
 for row,h in zip(x,images):k=idx[h];count[k]+=1;sums[k]+=row
 return sums,count
def bootstrap(x,images,reps=REPS,seed=SEED):
 sums,counts=cluster_arrays(x,images);rng=np.random.default_rng(seed);out=[]
 for start in range(0,reps,256):
  ids=rng.integers(0,len(counts),size=(min(256,reps-start),len(counts)));out.append(sums[ids].sum(1)/counts[ids].sum(1)[:,None])
 return np.vstack(out)
def signflip_p(x,images,reps=REPS,seed=SEED+1):
 sums,counts=cluster_arrays(x,images);rng=np.random.default_rng(seed);obs=abs(sums.sum(0));ge=np.zeros(x.shape[1],dtype=int)
 for start in range(0,reps,512):
  signs=rng.integers(0,2,size=(min(512,reps-start),len(counts)))*2-1;ge+=(abs(signs@sums)>=obs[None]-1e-12).sum(0)
 return (ge+1)/(reps+1)
def holm(p):
 order=np.argsort(p);ans=np.empty(len(p));running=0
 for rank,k in enumerate(order):running=max(running,min(1.,(len(p)-rank)*p[k]));ans[k]=running
 return ans
def contrast(x,weights):return sum(x[:,NAMES.index(c)]*w for c,w in weights.items())
def load():
 rows=read(OUT/'sample_manifest.json')['rows'];reuse={(r['question_id'],r['condition']):r for r in read(OUT/'historical_reuse_audit.json')['checkpoints']};allv={};idx=[];flat=[];checks=collections.Counter()
 for r in rows:
  q=r['question_id'];allv[q]={}
  for c in NAMES:
   reused=(q,c) in reuse;path=Path(reuse[(q,c)]['checkpoint_path']) if reused else OUT/'checkpoints'/q/(c+'.json');v=read(path);h=sha(path);g=v['generation'];s=official(g['text'],r['answers']);assert abs(s-v['score'])<1e-12 and v['input_hashes']==r['input_hashes'] and v['image_sha256']==r['image_sha256'];assert g['audit']['passed'] and g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1 and len(g['token_ids'])==g['length'];assert v['hook_cleanup'];assert g['eos']==(g['token_ids'][-1]==2) and g['truncated']==(len(g['token_ids'])==32 and not g['eos']);assert all(a['absolute_query']==g['query_layout']['prompt_length']+i-1 and a['after']==g['query_layout']['prompt_length']+i for i,a in enumerate(g['cache_lengths']))
   if reused:assert h==reuse[(q,c)]['sha256'];checks['reused']+=1
   else:
    assert v['protocol_sha256']==sha(OUT/'protocol.json') and v['source_manifest_sha256']==sha(OUT/'execution_source_manifest.json') and v['allowed_decode_image_layers']==CONDITIONS[c];assert v['input_spec_sha256']==r['input_spec_sha256'];assert v['runtime_git_commit']==read(OUT/'execution_freeze.json')['commit'];a=g['audit'];assert a['prefill_masked_queries_by_layer']==[0]*32;assert a['decode_masked_queries_by_layer']==[0 if k in CONDITIONS[c] else g['length']-1 for k in range(32)];assert v['worker']==read(OUT/'shards.json')['by_question'][q];checks['fresh']+=1
   checks['EOS']+=int(g['eos']);checks['empty']+=int(g['empty']);checks['truncated']+=int(g['truncated']);checks['repeated_trigram']+=int(g['repeated_trigram']);allv[q][c]=v
   idx.append(dict(question_id=q,cohort=r['cohort'],condition=c,path=str(path),sha256=h,reused=reused,origin_commit='4408b862e4760b6aadcb5135d2640f422778938b' if reused else v['runtime_git_commit']))
   flat.append(dict(cohort=r['cohort'],question_id=q,image_id=r['image_id'],image_sha256=r['image_sha256'],question=r['question'],answers=json.dumps(r['answers']),condition=c,allowed_layers=json.dumps(CONDITIONS[c]),retained_layers=len(CONDITIONS[c]),prediction=g['text'],token_ids=json.dumps(g['token_ids']),score=s,eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],first_logits_sha256=v['first_logits_sha256'],reused=reused,checkpoint_sha256=h))
  assert len({allv[q][c]['first_logits_sha256'] for c in NAMES})==1,'All six first predictions must be Native-identical'
 assert checks['fresh']==5400 and checks['reused']==1200
 write(OUT/'checkpoint_index.json',dict(checkpoints=idx,counts=dict(checks),all_scores_recomputed=True));csvout('per_question.csv',flat);csvout('all_error_outputs.csv',[r for r in flat if r['score']<1.-1e-12])
 return rows,allv,checks
def main():
 rows,allv,checks=load();summaries=[];pairs=[];interactions=[];neg=[];data={};ni={}
 for cohort in ['old300','new_images']:
  rr=[r for r in rows if r['cohort']==cohort];images=[r['image_sha256'] for r in rr];x=np.array([[allv[r['question_id']][c]['score'] for c in NAMES] for r in rr]);boot=bootstrap(x,images);data[cohort]=(rr,x,boot)
  for j,c in enumerate(NAMES):
   lo,hi=np.quantile(boot[:,j]*100,[.025,.975]);summaries.append(dict(cohort=cohort,questions=len(rr),images=len(set(images)),condition=c,retained_decode_layers=len(CONDITIONS[c]),score_sum=float(x[:,j].sum()),accuracy_percent=float(x[:,j].mean()*100),CI95_low_percent=float(lo),CI95_high_percent=float(hi),delta_Native_pp=float((x[:,j]-x[:,0]).mean()*100),reused=int(cohort=='old300' and c in ['A','B','Native','Decode-None'])))
  diffs=np.column_stack([contrast(x,w) for w in AUX.values()]);p=signflip_p(diffs,images);adjusted=holm(p)
  defs={'A-D':{'A':1,'D':-1},**AUX,'Native-Decode-None':{'Native':1,'Decode-None':-1}}
  for name,w in defs.items():
   d=contrast(x,w);b=contrast(boot,w)*100;lo,hi=np.quantile(b,[.025,.975]);aux=name in AUX;k=list(AUX).index(name) if aux else None
   row=dict(cohort=cohort,comparison=name,questions=len(rr),images=len(set(images)),delta_pp=float(d.mean()*100),CI95_low_pp=float(lo),CI95_high_pp=float(hi),improved=int((d>1e-12).sum()),declined=int((d< -1e-12).sum()),unchanged=int((abs(d)<=1e-12).sum()),p_two_sided=float(p[k]) if aux else '',p_Holm_10=float(adjusted[k]) if aux else '',family='fixed10 auxiliary' if aux else ('primary NI new only' if name=='A-D' and cohort=='new_images' else 'descriptive'))
   if name!='D-C-B+A':
    pos=[c for c,z in w.items() if z==1][0];ref=[c for c,z in w.items() if z==-1][0]
    row['answer_changed']=sum(allv[r['question_id']][pos]['generation']['text']!=allv[r['question_id']][ref]['generation']['text'] for r in rr)
    row['token_sequence_changed']=sum(allv[r['question_id']][pos]['generation']['token_ids']!=allv[r['question_id']][ref]['generation']['token_ids'] for r in rr)
    row['first_token_flipped']=sum(allv[r['question_id']][pos]['generation']['token_ids'][0]!=allv[r['question_id']][ref]['generation']['token_ids'][0] for r in rr)
   else:row.update(answer_changed='',token_sequence_changed='',first_token_flipped='')
   pairs.append(row)
   if name=='D-C-B+A':interactions.append(row)
   if name=='A-D':
    lower=float(np.quantile(b,.05));ni[cohort]=dict(delta_A_minus_D_pp=row['delta_pp'],one_sided95_lower_pp=lower,margin_pp=-1.,supports_noninferiority=bool(lower> -1.) if cohort=='new_images' else None,confirmatory=cohort=='new_images')
   if name!='D-C-B+A':
    pos=[c for c,z in w.items() if z==1][0];ref=[c for c,z in w.items() if z==-1][0]
    for r,di in zip(rr,d):
     if abs(di)>1e-12:
      q=r['question_id'];vp=allv[q][pos];vr=allv[q][ref];ps=set(CONDITIONS[pos]);rs=set(CONDITIONS[ref]);neg.append(dict(cohort=cohort,comparison=name,question_id=q,image_sha256=r['image_sha256'],question=r['question'],answers=json.dumps(r['answers']),candidate=pos,reference=ref,candidate_answer=vp['generation']['text'],reference_answer=vr['generation']['text'],candidate_score=vp['score'],reference_score=vr['score'],delta_score=float(di),direction='declined' if di<0 else 'improved',nonmonotonic=bool((di<0 and ps>rs) or (di>0 and ps<rs))))
 csvout('accuracy_summary.csv',summaries);csvout('paired_comparisons.csv',pairs);csvout('factorial_interactions.csv',interactions);csvout('negative_cases.csv',neg);csvout('nonmonotonic_cases.csv',[r for r in neg if r['nonmonotonic']]);write(OUT/'noninferiority.json',ni)
 plots(data,pairs);write_report(data,summaries,pairs,ni,checks)
 write(OUT/'analysis_validation.json',dict(passed=True,UTC=utc(),all6600_scores_recomputed=True,all_six_first_logits_SHA_equal=True,raw_checkpoint_counts=dict(checks),bootstrap_repetitions=REPS,seed=SEED,question_weighted_cluster_resampling=True,holm_family_size=10,noninferiority=ni))
 print(json.dumps(dict(accuracy=summaries,noninferiority=ni),indent=2),flush=True)
def savefig(fig,name):
 fig.savefig(OUT/'figures'/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(OUT/'figures'/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def plots(data,pairs):
 (OUT/'figures').mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 colors=['#4C78A8','#F58518','#54A24B','#B279A2'];cs=['A','B','C','D'];rr,x,boot=data['new_images'];mean=x.mean(0)*100
 fig,ax=plt.subplots(figsize=(7.4,4.3));ys=[mean[NAMES.index(c)] for c in cs];ax.bar(range(4),ys,color=colors,width=.6)
 for i,c in enumerate(cs):
  low,high=np.quantile(boot[:,NAMES.index(c)]*100,[.025,.975]);ax.errorbar(i,ys[i],yerr=[[ys[i]-low],[high-ys[i]]],fmt='none',color='black',capsize=4);ax.text(i,high+1.1,f'{ys[i]:.4f}%',ha='center')
 ax.axhline(mean[0],ls='--',color='#333333',label=f'Native Full ({mean[0]:.2f}%)');ax.axhline(mean[1],ls=':',color='#777777',label=f'Decode None ({mean[1]:.2f}%)');ax.set(xticks=range(4),xticklabels=['A: L11–20\n10 layers','B: +L0\n11 layers','C: +L1\n11 layers','D: +L0+L1\n12 layers'],ylabel='TextVQA soft accuracy (%)',title=f'New-image validation: {len(rr)} questions');ax.set_ylim(0,max(mean)+12);ax.legend(frameon=False,loc='lower right');fig.tight_layout();savefig(fig,'accuracy_layer_sets')
 fig,axs=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
 for ax,(cohort,(rr,x,boot)) in zip(axs,data.items()):
  m=x.mean(0)*100
  for cs2,label,color in [(['A','B'],'L1 absent','#4C78A8'),(['C','D'],'L1 present','#F58518')]:ax.plot([0,1],[m[NAMES.index(c)] for c in cs2],'-o',label=label,color=color)
  ax.set(xticks=[0,1],xticklabels=['L0 absent','L0 present'],title=f'{cohort}: n={len(rr)}',ylabel='TextVQA soft accuracy (%)');ax.legend(frameon=False)
 fig.suptitle('L0 × L1 interaction, middle layers L11–20 always retained');fig.tight_layout();savefig(fig,'factorial_interaction')
 fig,ax=plt.subplots(figsize=(8,6));names=['A-D']+list(AUX)
 for off,cohort,color in [(-.15,'old300','#999999'),(.15,'new_images','#4C78A8')]:
  z=[next(p for p in pairs if p['cohort']==cohort and p['comparison']==n) for n in names];y=np.arange(len(names))+off;val=np.array([p['delta_pp'] for p in z]);lo=np.array([p['CI95_low_pp'] for p in z]);hi=np.array([p['CI95_high_pp'] for p in z]);ax.errorbar(val,y,xerr=[val-lo,hi-val],fmt='o',color=color,capsize=3,label=cohort)
 ax.axvline(0,color='black',lw=1);ax.axvline(-1,color='#AA4444',ls=':',label='−1pp NI margin (A−D only)');ax.set(yticks=range(len(names)),yticklabels=names,xlabel='Paired accuracy difference (pp); pointwise 95% CI');ax.invert_yaxis();ax.legend(frameon=False);fig.tight_layout();savefig(fig,'paired_differences')
 fig,ax=plt.subplots(figsize=(8,4.2));pos=np.arange(6)
 for off,cohort,color in [(-.18,'old300','#999999'),(.18,'new_images','#4C78A8')]:
  rr,x,b=data[cohort];ax.bar(pos+off,x.mean(0)*100,width=.34,label=f'{cohort} (n={len(rr)})',color=color)
 ax.set(xticks=pos,xticklabels=NAMES,ylabel='TextVQA soft accuracy (%)',title='Historical300 vs new-image validation; same fixed conditions');ax.set_ylim(0,max(v[1].mean(0).max()*100 for v in data.values())+14);ax.legend(frameon=False);fig.tight_layout();savefig(fig,'old_vs_new')
def write_report(data,summaries,pairs,ni,checks):
 def s(cohort,c):return next(z for z in summaries if z['cohort']==cohort and z['condition']==c)
 def p(cohort,c):return next(z for z in pairs if z['cohort']==cohort and z['comparison']==c)
 def ef(c):
  z=p('new_images',c);return f"{z['delta_pp']:+.3f}pp，95%CI [{z['CI95_low_pp']:+.3f}, {z['CI95_high_pp']:+.3f}]；提高/下降/不变 {z['improved']}/{z['declined']}/{z['unchanged']}，Holm p={z['p_Holm_10']}"
 lines=['# TextVQA：L0/L1是否是中层视觉回看的必要补充？','',f'本轮固定旧300题探索与新800题（531张图片）验证。所有六条件整个Prefill及P−1均原生；仅在真实增量Decode用Softmax前−inf屏蔽关闭层的576个原始Image Key，完整视觉KV始终保留。','', '| 条件 | Decode开放层 | 层数 | 旧300软ACC | 新800软ACC | 新集相对Native |','|---|---|---:|---:|---:|---:|']
 for c in ['Native','Decode-None','A','B','C','D']:
  layertext={'Native':'L0–L31','Decode-None':'无','A':'L11–L20','B':'L0 + L11–L20','C':'L1 + L11–L20','D':'L0、L1 + L11–L20'}[c]
  z=s('new_images',c);lines.append(f"| {c} | {layertext} | {len(CONDITIONS[c])} | {s('old300',c)['accuracy_percent']:.4f}% | {z['accuracy_percent']:.4f}% | {z['delta_Native_pp']:+.4f}pp |")
 v=ni['new_images'];decision='达到预声明的−1pp非劣效标准' if v['supports_noninferiority'] else '未达到预声明的−1pp非劣效标准；证据不足以支持10层替代12层'
 lines+=['','## 预声明主检验', '',f"新图片A−D={v['delta_A_minus_D_pp']:+.3f}pp，按图片SHA整簇20,000次配对Bootstrap的单侧95%下界={v['one_sided95_lower_pp']:+.3f}pp。只有下界严格大于−1.000pp才支持非劣。本轮{decision}。相对D的结论不能代替相对Native的质量判断。",'', '## 直接回答研究问题','',f"**Q1：仅L11–L20是否足够？** A的新集ACC={s('new_images','A')['accuracy_percent']:.3f}%，相对Native {s('new_images','A')['delta_Native_pp']:+.3f}pp。{decision}，不称其相对Native无损。",'',f"**Q2：L0增益能否由L1替代？** 无L1时加L0（B−A）：{ef('B-A')}。无L0时加L1（C−A）：{ef('C-A')}。同预算C−B：{ef('C-B')}。这组固定比较限定了可支持的替代程度，不能由区间跨0声称等价。",'',f"**Q3：同时保留L0/L1是否有额外收益？** D−B：{ef('D-B')}。D−C：{ef('D-C')}。交互D−C−B+A：{ef('D-C-B+A')}。",'',f"**Q4：同11层预算哪个更好？** B={s('new_images','B')['accuracy_percent']:.3f}%，C={s('new_images','C')['accuracy_percent']:.3f}%；点估计较高者为{'L0版本B' if s('new_images','B')['accuracy_percent']>s('new_images','C')['accuracy_percent'] else ('L1版本C' if s('new_images','C')['accuracy_percent']>s('new_images','B')['accuracy_percent'] else '并列')}；显著性与不确定性见上面的C−B及Holm校正。",'', '**Q5：新图片是否复现？** 两个队列的全部配对效应和区间均保留在paired_comparisons.csv及对照图中；旧300是机制探索，新800在预测前冻结且与发现到的历史图片ID及字节SHA均不重叠。按具体效应的方向和区间判断复现，不将同源公开验证集称为外部独立数据集。','', '## 数据、绑定与正确性','', '官方TextVQA v0.5.1验证集5000题。先审计所有可用工作区冻结样本及实际历史检查点身份：历史候选池中的未推理图片不自动视为已使用，候选清单另行保留；历史TextVQA与TextCaps已执行图片均排除。按冻结SHA规则取整张图片的全部问题，得到531图800题；不读取新预测来挑题。完整排除来源和SHA在image_overlap_audit.json及historical_image_inventory.json。无法排除未公开历史或重新编码的近重复。','', '旧300的A/B/Native/Decode-None共1200份记录精确复用：输入、图片SHA、官方评分、协议/源码绑定及基础Commit文件字节逐一核验。新增旧300 C/D=600条；新800×6=4800条，共5400条新正式生成。两卡Smoke额外回放单独审计，拥有分片的有效Smoke可保留为正式记录。','', '8项CPU测试；两卡各4道真实7B Smoke，Native/HF/No-op全部有效Logits严格torch.equal。A/B与冻结历史引擎同输入全部有效Logits严格一致，同时与历史完整Token及首向量SHA匹配。历史文件仅保存首Logits SHA，未保存全步向量，不能把引擎回放等同于已归档全步历史向量。六条件首预测向量均与Native逐位一致；实际禁用概率严格0、非目标Mask原样、位置与独立Cache、完整32层MLP/Norm/Block、视觉特征与权重版本及正常/异常Hook清理均验证。','',f"所有6600条条件标签完整，复用{checks['reused']}，新增{checks['fresh']}。EOS={checks['EOS']}，截断={checks['truncated']}，空答案={checks['empty']}，重复3gram={checks['repeated_trigram']}。异常不按答案剔除。",'', '## 统计与解释边界','', '采用官方十答案共识软Accuracy；提高/下降是逐题软分的变化，不等价于简单正确题数。按Image SHA整簇有放回Bootstrap20000次，seed20261105；每次保留同图全部问题，并以重抽到的问题总数为分母，保持题均权。所有普通95%CI是点位探索性区间。主NI为新集A−D一项；辅助10项比较每个队列独立固定家族，用20000次整图配对符号翻转双侧Monte Carlo p及Holm校正。交互是D−C−B+A，不把单项显著与否代替交互检验。','', 'negative_cases.csv保留全部配对软分变化，包括增加开放层却下降的非单调例，不只展示支持中层故事的题。完整原始输出、Token与每请求Cache/Mask审计位于检查点，SHA索引绑定来源。此实验只衡量功能依赖，未删除KV、未减少实际计算，不从层数或Mask比例推出加速收益；物理算法可行性需另行验证。','', '## 图表','', '![新图片主结果](figures/accuracy_layer_sets.png)','','![二因素交互](figures/factorial_interaction.png)','','![配对差异](figures/paired_differences.png)','','![旧300与新800](figures/old_vs_new.png)','','执行协议、源码冻结Commit、GPU预检、两卡Smoke、Checkpoint索引及释放记录均保留。本轮完成后停止，不开展新层搜索或物理加速实现。','']
 lines.insert(2,f"**主结果：新800题上，仅L11–L20的A为{s('new_images','A')['accuracy_percent']:.3f}%，D为{s('new_images','D')['accuracy_percent']:.3f}%；A−D={v['delta_A_minus_D_pp']:+.3f}pp，单侧95%下界={v['one_sided95_lower_pp']:+.3f}pp，{decision}。A相对Native仍有{s('new_images','A')['delta_Native_pp']:+.3f}pp的点估计差距。**\n")
 lines+=['## 复现判定与非单调案例','', '旧300题中，四个稀疏集合的最高点是B（43.133%），L0相对A增加1.067pp；新800题的最高点变为A（47.350%），B−A反而为−0.713pp。因此旧样本中“L0有正收益”的方向未得到新图片复现。新集也没有建立L1可替代L0的证据：C−B为−0.538pp，普通95%CI跨0，Holm p=0.746。', '', '同时保留L0/L1的D在新集低于A和B，D−B为−0.413pp；D−C仅+0.125pp。辅助校正后没有证据支持额外L0/L1是中层回看的必要补充。交互为+0.838pp，95%CI [−0.258,+2.000]，Holm p=0.746，不能确立非加性交互。', '', '新集主比较A−D逐题软分提高16题、下降5题、不变779题；68题完整答案改变。也就是说，额外开放L0和L1可使部分问题变好，也可使更多问题变差，性能不随开放层数单调增加。完整变化、非单调例及所有非满分输出分别保留在negative_cases.csv、nonmonotonic_cases.csv、all_error_outputs.csv。', '', '相对Native的A−Native普通95%CI为[−3.100,−0.062]pp；该辅助比较Holm p=0.307，既不能从未显著推导无损，也不能把相对D的非劣效结论转移到Native。旧/新队列来自不同图片分布，不通过直接比较两个队列的绝对ACC来评价泛化提升。', '', '下表仅描述预定效应的方向，不是新增检验族：', '', '| 配对效应 | 旧300 Δpp | 新800 Δpp | 点估计方向 |','|---|---:|---:|---|']
 for name in ['A-D','B-A','C-A','D-B','D-C','C-B','D-C-B+A']:
  a=p('old300',name)['delta_pp'];b=p('new_images',name)['delta_pp'];lines.append(f"| {name} | {a:+.3f} | {b:+.3f} | {'同向' if np.sign(a)==np.sign(b) else '不同向'} |")
 lines+=['','该表是固定效应的复现对照，不是额外确认性假设族。证据强弱应结合新集配对CI及预声明校正；方向相同不等于效应大小已得到确认。','']
 (OUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
