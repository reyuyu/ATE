"""Independent CPU scoring, paired cluster bootstrap and one scientific figure."""
import os,sys,json,sqlite3,gzip,csv,hashlib,importlib.util,statistics,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/'code'))
from common import *
import numpy as np
def independent_score(ev,text,gold):
 pred=ev.answer_processor(text);answers=[ev.answer_processor(x) for x in gold]
 return sum(min(1.0,sum(a==pred for j,a in enumerate(answers) if j!=i)/3.0) for i in range(10))/10.0
def main():
 assert (OUT/'generation_complete.json').exists()
 verify()
 source=read(OUT/'source_manifest.json')
 for mapping in [source['official_transformers_source'],source['scorer']]:
  for path,expected in mapping.items():assert sha(path)==expected,('UPSTREAM_SOURCE_CHANGED',path)
 db=sqlite3.connect('file:'+str(RUN/'checkpoints/predictions.sqlite')+'?mode=ro',uri=True)
 rows=read(OUT/'sample_manifest.json');cs=configs();names=list(cs);idx={n:i for i,n in enumerate(names)}
 spec=importlib.util.spec_from_file_location('frozen_scoring',RUN/'references/textvqa_eval.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);ev=mod.TextVQAAccuracyEvaluator()
 complete=[];excluded=[];matrix=[];stats={n:dict(empty=0,truncated=0,eos=0,lengths=[]) for n in names}
 total=0;actual=0;blocked=0;kvcount=0
 fields=['key','question_id','condition','status','common_subset','image_id','image_sha256','allowed_layers','input_binding','question','prompt_text','gold','text','score','normalized_prediction','token_ids','generated_tokens','decode_steps','eos','truncated','empty','prefill_logits_sha256','prefill_first_token','gpu_uuid','model_revision','protocol_sha256','checkpoint_record_sha256']
 (OUT/'exports').mkdir(exist_ok=True)
 qf=gzip.open(OUT/'per_question.csv.gz','wt',newline='',encoding='utf-8');writer=csv.DictWriter(qf,fields);writer.writeheader()
 rf=gzip.open(OUT/'predictions.jsonl.gz','wt',encoding='utf-8')
 kf=gzip.open(OUT/'kv_layout_per_layer.csv.gz','wt',newline='',encoding='utf-8');kw=csv.DictWriter(kf,['key','condition','layer','key_tokens','visual_key_tokens','key_shape','value_shape','key_dtype','value_dtype','live_bytes']);kw.writeheader()
 for key,row in rows.items():
  records={}
  for condition,checksum,raw in db.execute('select condition,checksum,record from predictions where key=?',(key,)):
   assert hashlib.sha256(raw.encode()).hexdigest()==checksum
   r=json.loads(raw);assert r['condition']==condition and r['key']==key and r['gpu_uuid']==GPU and r['model_revision']==S['revision']
   assert r['input_binding']==row['input_binding'] and r['image_sha256']==row['image_sha256'] and r['gold']==row['gold']
   assert r['protocol_sha256']==sha(OUT/'protocol.json') and r['source_manifest_sha256']==sha(OUT/'source_manifest.json')
   records[condition]=(r,checksum)
  assert set(records)==set(names),(key,set(records))
  common=all(records[n][0]['status']=='COMPLETE' for n in names)
  if common:
   complete.append(key);matrix.append([records[n][0]['score'] for n in names])
   assert len({records[n][0]['prefill_logits_sha256'] for n in names})==1
   assert len({records[n][0]['prefill_first_token'] for n in names})==1
  else:
   excluded.append(dict(key=key,question_id=row['question_id'],image_id=row['image_id'],image_sha256=row['image_sha256'],conditions_blocked=[n for n in names if records[n][0]['status']!='COMPLETE'],reason='capacity OOM; no zero score, no replacement'))
  for name in names:
   r,ch=records[name];total+=1;actual+=int(r['generation_executed'])
   if r['status']=='COMPLETE':
    predicted=ev.answer_processor(r['text']);score=ev._compute_answer_scores(row['gold']).get(predicted,0.)
    other=independent_score(ev,r['text'],row['gold'])
    assert abs(score-other)<1e-12 and abs(score-r['score'])<1e-12,(key,name,score,other,r['score'])
    assert len(r['token_ids'])<=32 and r['generated_tokens']==len(r['token_ids']) and r['decode_steps']==len(r['token_ids'])-1
    if common:
     for f in ['empty','truncated','eos']:stats[name][f]+=int(r[f])
     stats[name]['lengths'].append(r['generated_tokens'])
    for layout in r['kv_layout']:
     assert layout['visual_key_tokens']==row['image_tokens']
     kw.writerow(dict(key=key,condition=name,**{k:json.dumps(v) if isinstance(v,list) else v for k,v in layout.items()}));kvcount+=1
   else:
    assert r['status']=='BLOCKED_OOM' and r['score'] is None;blocked+=1;predicted=''
   rf.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
   export={f:r.get(f) for f in fields};export.update(common_subset=common,question=row['question'],prompt_text=row['prompt_text'],normalized_prediction=predicted,checkpoint_record_sha256=ch)
   writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in export.items()})
 qf.close();rf.close();kf.close();db.close()
 assert total==15000 and len(complete)+len(excluded)==500 and len(complete)>0
 scores=np.asarray(matrix,dtype=np.float64);means=scores.mean(0)*100
 clusters=sorted({rows[k]['image_sha256'] for k in complete});ci={h:i for i,h in enumerate(clusters)}
 sums=np.zeros((len(clusters),len(names)));counts=np.zeros(len(clusters))
 for k,score in zip(complete,scores):
  j=ci[rows[k]['image_sha256']];sums[j]+=score;counts[j]+=1
 rng=np.random.default_rng(20261112);boot=np.empty((20000,len(names)))
 for start in range(0,20000,250):
  w=rng.multinomial(len(clusters),np.ones(len(clusters))/len(clusters),size=min(250,20000-start))
  boot[start:start+len(w)]=(w@sums)/(w@counts)[:,None]*100
 summary=[];native=idx['Native'];zero=idx['NoImage']
 for name in names:
  j=idx[name];lo,hi=np.percentile(boot[:,j],[2.5,97.5]);lens=stats[name].pop('lengths')
  summary.append(dict(condition=name,k=len(cs[name]),layers=cs[name],N_common=len(complete),manifest_N=500,excluded_N=len(excluded),image_clusters=len(clusters),accuracy_percent=means[j],CI95_low=lo,CI95_high=hi,loss_vs_Native_pp=means[native]-means[j],recovery_from_NoImage_percent=(means[j]-means[zero])/(means[native]-means[zero])*100 if means[native]!=means[zero] else None,length_P50=float(np.median(lens)),length_P95=float(np.percentile(lens,95)),length_max=max(lens),**stats[name],source='new current3090 predictions; common non-OOM subset',GPU=GPU))
 csvwrite(OUT/'accuracy_summary.csv',summary)
 comparisons=[]
 pairs=[(n,'Native') for n in names if n!='Native']+[(n,'NoImage') for n in names if n not in ['Native','NoImage']]
 for k in S['budgets'][1:-1]:
  pairs += [(a+str(k),b+str(k)) for a,b in itertools.combinations(['Early','Middle','Late','Random'],2)]
 for a,b in pairs:
  d=boot[:,idx[a]]-boot[:,idx[b]];lo,hi=np.percentile(d,[2.5,97.5]);x=scores[:,idx[a]];y=scores[:,idx[b]]
  comparisons.append(dict(A=a,B=b,delta_A_minus_B_pp=means[idx[a]]-means[idx[b]],CI95_low=lo,CI95_high=hi,N=len(complete),image_clusters=len(clusters),bootstrap_replicates=20000,improved_questions=int((x>y).sum()),harmed_questions=int((x<y).sum()),unchanged_questions=int((x==y).sum()),multiplicity='ordinary95%CI, no multiple-comparison adjustment, exploratory'))
 csvwrite(OUT/'paired_comparisons.csv',comparisons)
 plotrows=[]
 for family in ['Early','Middle','Late','Random']:
  for k in S['budgets']:
   name='NoImage' if k==0 else 'Native' if k==36 else family+str(k)
   r=next(r for r in summary if r['condition']==name);plotrows.append(dict(family=family,**r))
 csvwrite(OUT/'plot_data.csv',plotrows)
 write(OUT/'common_subset.json',dict(manifest_N=500,N=len(complete),image_clusters=len(clusters),complete_keys=complete,excluded=excluded,policy='same successful intersection across all30 conditions; not a complete500 accuracy'))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 write(OUT/'analysis_environment.json',dict(python=sys.version,executable=sys.executable,numpy_version=np.__version__,numpy_import=np.__file__,matplotlib_version=matplotlib.__version__,matplotlib_import=matplotlib.__file__,scope='CPU scoring/statistics/plot only; frozen GPU runtime untouched',CUDA_used=False))
 plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42})
 fig,ax=plt.subplots(figsize=(5.4,3.8))
 colors={'Early':'#6b8aaf','Middle':'#0c8f95','Late':'#e3833a','Random':'#9858b5'};markers={'Early':'s','Middle':'o','Late':'^','Random':'D'}
 for family in colors:
  data=[r for r in plotrows if r['family']==family]
  ax.plot([r['k'] for r in data],[r['accuracy_percent'] for r in data],label=family,color=colors[family],marker=markers[family],markersize=4,linewidth=1.5,markerfacecolor='white')
 ax.axhline(means[native],color='#768397',ls='--',lw=1)
 ax.axhline(means[zero],color='#a3acb8',ls=':',lw=.9)
 ax.set_xticks(S['budgets']);ax.set_xlabel('Number of decode visual re-access layers');ax.set_ylabel('TextVQA accuracy (%)')
 ax.set_title('Qwen3-VL-8B-Instruct',fontsize=11)
 ax.text(36,means[native]+.35,f'Native Full ({means[native]:.2f}%)',ha='right',fontsize=8,color='#566272')
 ax.set_ylim(float(means.min())-1.5,float(means.max())+2.5)
 ax.grid(axis='y',color='#dce2e8',lw=.6);ax.spines[['top','right']].set_visible(False)
 ax.legend(ncol=4,loc='lower center',bbox_to_anchor=(.5,1.09),frameon=False,handlelength=1.6,columnspacing=1)
 fig.text(.5,.015,f'Image-only; common non-OOM subset {len(complete)}/500; native prefill/P−1; greedy ≤32 tokens',ha='center',fontsize=7)
 fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(OUT/'figure_accuracy.png',dpi=250);fig.savefig(OUT/'figure_accuracy.pdf');plt.close(fig)
 validation=dict(status='PASS_COMMON_SUBSET' if excluded else 'PASS_FULL500',planned_condition_records=15000,actual_condition_records=total,executed_generation_attempts=actual,successful_generations=total-blocked,blocked_records=blocked,N_common=len(complete),manifest_N=500,excluded_questions=len(excluded),image_clusters=len(clusters),independent_official_scoring_equal=True,all_record_checksums_equal=True,all_input_gold_bindings_equal=True,all30_points_same_subset=True,first_token_native_in_all_conditions=True,all36_dense_KV_layout_checked=True,KV_rows=kvcount,bootstrap_replicates=20000,source_fingerprints_unchanged=True,GPU=GPU,analysis_source_sha256=sha(Path(__file__)))
 write(OUT/'validation.json',validation)
 progress=read(OUT/'progress.json');progress.update(status=validation['status'],successful_generations=total-blocked,blocked_records=blocked,N_common=len(complete));write(OUT/'progress.json',progress)
 lines=['# Qwen3-VL Figure7：Decode视觉回看层预算曲线','',f"完成30个固定条件；原500题保留，统一有效交集 **{len(complete)}/500题、{len(clusters)}个图片簇**。共{total-blocked}次成功生成，{blocked}条BLOCKED记录。OOM未补零、未换题；本报告不是完整500题分数。",'',f"Native Full **{means[native]:.2f}%**；Decode全层禁图 **{means[zero]:.2f}%**。Prefill和P−1在全部36层保持原生；所有条件首输出Token一致。",'','| k | Early | Middle | Late | Random |','|---:|---:|---:|---:|---:|']
 for k in S['budgets']:
  vals=[next(r['accuracy_percent'] for r in plotrows if r['family']==family and r['k']==k) for family in ['Early','Middle','Late','Random']]
  lines.append('| '+str(k)+' | '+' | '.join(f'{x:.2f}%' for x in vals)+' |')
 lines += ['','同预算配对差值（percentage points；图片SHA整簇20000次Bootstrap，普通95%CI，未经多重校正）：','','| k | Middle−Early | Middle−Late | Middle−Random |','|---:|---:|---:|---:|']
 for k in S['budgets'][1:-1]:
  cells=[]
  for b in ['Early','Late','Random']:
   a='Middle'+str(k);b+=str(k);d=boot[:,idx[a]]-boot[:,idx[b]];lo,hi=np.percentile(d,[2.5,97.5]);cells.append(f'{means[idx[a]]-means[idx[b]]:+.2f} [{lo:+.2f}, {hi:+.2f}]')
  lines.append('| '+str(k)+' | '+' | '.join(cells)+' |')
 lines += ['','协议与限制：',
 '- 与Figure7对应的原始证据为9b540dc6d5b7cce53c930fdfb322479f7478c534，完全复用问题、原图、Image-only、十答案Gold与顺序。Figure7原始LLaVA500准确率不直接充当本轮Qwen基线。',
 '- Qwen固定Revision '+S['revision']+'，BF16/Eager，Transformers4.57.6；Qwen原生Chat、动态Processor、DeepStack与mRoPE。原图不缩小，视觉Token数不同于LLaVA的576。',
 '- 36层采用用户指定的四等分及各区间中点，共9点：'+str(S['budgets'])+'；中点向下取整；Middle连续嵌套窗口start=(36+4−k)//2；Random固定单种子20261108，一次排列，不择优。',
 '- 仅真实Decode禁止原始Image Key访问；Prefill及P−1全Native。没有SVH Attention跳算、Hidden退出或物理KV压缩，不能据此宣称FLOPs、显存或延迟节省。',
 '- Smoke确认Native/No-op Logits逐位相等，HF原生生成与手工真实Decode Token一致；独立Cache克隆和独立重算Prefill的所有Smoke Logits/Token逐位相等；禁止图像概率严格为0。模型在正式工作进程只加载一次。',
 '- OOM策略经用户确认，GitHub历史来源详见oom_policy_audit_zh.md。各点使用同一完整成功交集；OOM与分辨率相关，可能引入缺失偏差。不得把子集结果称为完整Figure7的500题复现。',
 '- 36759的原图已实测在原生Vision Eager阶段OOM；36760与之共用相同SHA的图片，沿用该图容量失败证据，未再重复GPU尝试。逐条记录保留original_failed_key、generation_executed与原始错误；不把共享证据声称为独立测量。',
 '- 全部离散点均为真实生成，无平滑、补点或结果驱动选层。区间为探索性普通95%CI；CI跨0不证明等价。只有一个随机排列，不代表随机选层总体期望。',
 '- 不运行其他Benchmark、质量测试或性能Profiling。完整检查点保留本地SSD并归档到持久工作盘；GitHub仅同步轻量代码、协议、数据和图。','', '排除的原始题：']
 for r in excluded:lines.append('- '+r['question_id']+'；图片SHA '+r['image_sha256']+'；'+r['reason'])
 lines += ['','逐点CI、相对Native损失、相对全禁图恢复比例见accuracy_summary.csv；全部同预算配对比较见paired_comparisons.csv；EOS、空输出、截断及长度P50/P95/最大值同表保存。','', '图片SHA整簇配对方法：以共同有效集的图片簇有放回重采样，簇内全部问题一起纳入，每次按实际抽取的问题数计算软评分均值；全部条件使用同一重采样权重。','', '核心路径：', '- 本地执行与完整SQLite：'+str(RUN),'- 持久结果：'+S['root']+'/'+S['result']]
 (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(validation=validation,accuracy=[{k:r[k] for k in ['condition','accuracy_percent']} for r in summary]),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
