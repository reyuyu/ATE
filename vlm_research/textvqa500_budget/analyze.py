"""One paired image-cluster bootstrap and one discrete scientific figure."""
import csv,gzip,json,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from ..benchmarks.scoring import textvqa_score

def write_csv(path,rows):
 with Path(path).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(x):return tuple(float(v) for v in np.percentile(x,[2.5,97.5]))
def holm(rows):
 n=len(rows);previous=0
 for rank,row in enumerate(sorted(rows,key=lambda r:r['p_unadjusted'])):
  previous=max(previous,min(1,(n-rank)*row['p_unadjusted']));row['p_holm_family']=previous

def main():
 assert read(OUT/'run_status.json')['status']=='COMPLETE'
 assert read(OUT/'gpu_smoke.json')['status']=='PASS'
 protocol=read(OUT/'protocol.json');samples=load_samples();keys=list(samples)
 with gzip.open(OUT/'reused_predictions.json.gz','rt',encoding='utf-8') as f:reuse=json.load(f)
 records=[];scores={};qualities={}
 for condition in CONDITIONS:
  rr=[]
  if condition in REUSE:
   for r in reuse[condition]:
    rr.append(dict(key=r['key'],condition=condition,answer=r['answer'],score=float(r['score']),token_ids=json.loads(r['token_ids']),eos=r['eos'].lower()=='true',empty=r['empty'].lower()=='true',truncated=r['truncated'].lower()=='true',first_top1_token=int(r['first_top1_token']),input_binding_sha256=r['input_binding_sha256'],source='reused',source_commit=protocol['reuse'][condition]['commit']))
  else:
   for key in keys:
    wrap=read(OUT/'checkpoints'/condition/(key+'.json'));p=wrap['payload'];assert wrap['sha256']==digest(p)
    g=p['generation'];rr.append(dict(key=key,condition=condition,answer=g['text'],score=p['score'],token_ids=g['token_ids'],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],first_top1_token=g['first_top1_token'],input_binding_sha256=p['input_binding_sha256'],source='new',source_commit=''))
  bykey={r['key']:r for r in rr};assert len(bykey)==500
  rr=[bykey[k] for k in keys]
  for r in rr:
   s=samples[r['key']];assert abs(r['score']-textvqa_score(r['answer'],s['answer']))<1e-12
   r.update(question_id=s['question_id'],image_sha256=s['image_sha256'])
  records.extend(rr);scores[condition]=np.array([r['score'] for r in rr]);qualities[condition]={field:sum(bool(r[field]) for r in rr) for field in ['eos','empty','truncated']}
 conditions=list(CONDITIONS);X=np.stack([scores[c] for c in conditions],axis=1)
 clusters=sorted({s['image_sha256'] for s in samples.values()});indices={v:i for i,v in enumerate(clusters)}
 sums=np.zeros((len(clusters),len(conditions)));counts=np.zeros(len(clusters))
 for i,key in enumerate(keys):
  j=indices[samples[key]['image_sha256']];sums[j]+=X[i];counts[j]+=1
 rng=np.random.default_rng(20261108);weights=rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=20000)
 bootstrap=(weights@sums)/(weights@counts)[:,None]
 means=X.mean(0);col={c:i for i,c in enumerate(conditions)};n=col['Native'];z=col['NoImage']
 assert means[n]!=means[z]
 summary=[]
 for c in conditions:
  j=col[c];lo,hi=ci(100*bootstrap[:,j]);loss=100*(means[n]-means[j]);ll,lh=ci(100*(bootstrap[:,n]-bootstrap[:,j]))
  summary.append(dict(condition=c,k=len(CONDITIONS[c]),layers='none' if not CONDITIONS[c] else f'L{min(CONDITIONS[c])}-L{max(CONDITIONS[c])}',questions=500,image_clusters=len(clusters),accuracy_pct=100*means[j],ci95_low_pct=lo,ci95_high_pct=hi,native_loss_pp=loss,native_loss_ci_low_pp=ll,native_loss_ci_high_pp=lh,recovery_fraction_vs_NoImage=(means[j]-means[z])/(means[n]-means[z]),source='reused' if c in REUSE else 'new',source_branch={'middle':'codex/middle10-decode-early-late-recovery','l11':'codex/textvqa-decode-l11-l24-500'}.get(REUSE.get(c,('',))[0],''),source_commit=protocol['reuse'][c]['commit'] if c in REUSE else '',execution_freeze_sha256=sha(OUT/'execution_freeze.json'),**qualities[c]))
 pairs=[]
 for k in BUDGETS[1:-1]:
  for left,right in [('Middle','Early'),('Middle','Late'),('Early','Late')]:
   a,b=left+str(k),right+str(k);delta=100*(means[col[a]]-means[col[b]]);bs=100*(bootstrap[:,col[a]]-bootstrap[:,col[b]]);lo,hi=ci(bs)
   p=(1+np.count_nonzero(np.abs(bs-delta)>=abs(delta)))/(len(bs)+1)
   pairs.append(dict(k=k,left=a,right=b,difference_pp=delta,ci95_low_pp=lo,ci95_high_pp=hi,p_unadjusted=p,p_holm_family=0.,family='primary9' if k in [14,16,21] else 'secondary12',improved_questions=int(np.count_nonzero(scores[a]>scores[b])),harmed_questions=int(np.count_nonzero(scores[a]<scores[b])),equal_questions=int(np.count_nonzero(scores[a]==scores[b])) ))
 for family in ['primary9','secondary12']:holm([r for r in pairs if r['family']==family])
 lookup={r['condition']:r for r in summary};plot=[]
 for region in ['Early','Middle','Late']:
  for k in BUDGETS:
   c='NoImage' if k==0 else 'Native' if k==32 else region+str(k)
   plot.append(dict(region=region,**lookup[c],highlight=(region=='Middle' and k==14)))
 write_csv(OUT/'accuracy_summary.csv',summary);write_csv(OUT/'paired_comparisons.csv',pairs);write_csv(OUT/'plot_data.csv',plot)
 with gzip.open(OUT/'per_question.jsonl.gz','wt',encoding='utf-8') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
 fig,ax=plt.subplots(figsize=(8.0,4.8),layout='constrained')
 colors={'Early':'#1f77b4','Middle':'#d1495b','Late':'#30966a'}
 for region,marker in [('Early','o'),('Middle','s'),('Late','^')]:
  rr=[r for r in plot if r['region']==region];y=np.array([r['accuracy_pct'] for r in rr]);low=np.array([r['ci95_low_pct'] for r in rr]);high=np.array([r['ci95_high_pct'] for r in rr])
  ax.plot(BUDGETS,y,marker=marker,color=colors[region],label=region,linewidth=1.6,markersize=4.6)
  ax.errorbar(BUDGETS,y,yerr=[y-low,high-y],fmt='none',ecolor=colors[region],alpha=.35,elinewidth=.9,capsize=2)
 r=lookup['Middle14'];ax.scatter([14],[r['accuracy_pct']],s=105,facecolors='none',edgecolors=colors['Middle'],linewidths=1.6,zorder=5)
 ax.annotate('Middle k=14: L11-L24',xy=(14,r['accuracy_pct']),xytext=(17,r['accuracy_pct']+4.0),arrowprops={'arrowstyle':'->','color':colors['Middle']},fontsize=9,color=colors['Middle'])
 for k,c in [(0,'NoImage'),(32,'Native')]:ax.scatter([k],[lookup[c]['accuracy_pct']],color='#202020',s=28,zorder=6)
 ax.set(xlabel='Number of Decode Visual Re-access Layers',ylabel='TextVQA Soft Accuracy (%)',title='TextVQA Accuracy vs. Number of Decode Visual Re-access Layers',xticks=BUDGETS,xlim=(-.8,32.8))
 ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,ncol=3,loc='lower right')
 fig.text(.01,-.015,'500 frozen questions; 309 image clusters. Ordinary pointwise 95% cluster-bootstrap CIs. Dense masks only.',fontsize=8)
 fig.savefig(OUT/'figure_accuracy.png',dpi=300,bbox_inches='tight');fig.savefig(OUT/'figure_accuracy.pdf',bbox_inches='tight');plt.close(fig)
 # Adjacent increases are retained even when they reduce accuracy; no smoothing.
 adjacent=[]
 for region in ['Early','Middle','Late']:
  cc=['NoImage']+[region+str(k) for k in BUDGETS[1:-1]]+['Native']
  for a,b in zip(cc,cc[1:]):
   bs=100*(bootstrap[:,col[b]]-bootstrap[:,col[a]]);lo,hi=ci(bs)
   adjacent.append(dict(region=region,from_condition=a,to_condition=b,difference_pp=100*(means[col[b]]-means[col[a]]),ci95_low_pp=lo,ci95_high_pp=hi,harmed_questions=int(np.sum(scores[b]<scores[a])),improved_questions=int(np.sum(scores[b]>scores[a]))))
 write_csv(OUT/'adjacent_changes.csv',adjacent)
 harm=[]
 record_map={(r['condition'],r['key']):r for r in records}
 for k in [14,16,21]:
  for other in ['Early','Late']:
   a,b='Middle'+str(k),other+str(k)
   for i,key in enumerate(keys):
    if scores[a][i]!=scores[b][i]:harm.append(dict(k=k,key=key,image_sha256=samples[key]['image_sha256'],question=samples[key]['question'],left=a,right=b,left_score=float(scores[a][i]),right_score=float(scores[b][i]),difference=float(scores[a][i]-scores[b][i]),left_answer=record_map[a,key]['answer'],right_answer=record_map[b,key]['answer'],gold_answers=json.dumps(samples[key]['answer'],ensure_ascii=False)))
 write_csv(OUT/'priority_disagreements.csv',harm)
 lines=['# TextVQA500 Early / Middle / Late Decode 视觉回看预算实验','',f'冻结 500 题、309 个 Image SHA 簇；模型 {REV}，FP16 / Eager / Transformers 4.49.0 / Torch 2.6.0+cu124。全部 Prefill（含 P−1）保持 Native，只在真正 Decode 对原始 Image Key 使用 Softmax 前 −inf Hard Mask。Dense 功能性实验，不代表 FLOPs、KV 容量或延迟节省。','',f'CPU 严格输入核验和 GPU 正确性 Smoke 均 PASS。4 个历史点复用（2,000 条），19 个新条件共 9,500 条；共 23 个唯一测量条件。没有重跑 500 题 Native，没有混入旧 300 题联合限制实验。','', '## 全部测量点','', '| 条件 | k | 层集合 | Accuracy（95% CI） | 相对 Native 损失 pp | 相对禁图恢复比例 | 来源 |','|---|---:|---|---|---:|---:|---|']
 for r in sorted(summary,key=lambda r:(r['k'],r['condition'])):lines.append(f"| {r['condition']} | {r['k']} | {r['layers']} | {r['accuracy_pct']:.2f}% [{r['ci95_low_pct']:.2f}, {r['ci95_high_pct']:.2f}] | {r['native_loss_pp']:.2f} | {r['recovery_fraction_vs_NoImage']:.3f} | {r['source']} |")
 lines+=['','恢复比例 = (条件 Accuracy − k0 Accuracy) / (Native Accuracy − k0 Accuracy)，不截断至 [0,1]。准确率为官方十答案共识软分的逐题均值。','', '## 同预算配对比较','', '差值为左侧减右侧（百分点）。所有问题共同按 Image SHA 整簇重采样 20,000 次（seed=20261108），每次仍按问题等权计算均值。下列为普通、未作同时覆盖校正的 95% percentile CI。预先重点比较 k=14/16/21 的 9 项；其余 12 项为次要探索比较。分别在两组内报告 Holm 校正的中心化双侧 Bootstrap p 值。CI 跨 0 表示存在不确定性，不能证明等价；探索结果不应解释为独立确认性检验。','', '| k | 比较 | 配对差值 pp（95% CI） | Holm p |','|---:|---|---|---:|']
 for r in pairs:lines.append(f"| {r['k']} | {r['left']} − {r['right']} | {r['difference_pp']:.2f} [{r['ci95_low_pp']:.2f}, {r['ci95_high_pp']:.2f}] | {r['p_holm_family']:.4f} |")
 lines+=['','## 重点预算与不确定性','']
 for k in [14,16,21]:
  rr=[r for r in pairs if r['k']==k and r['left'].startswith('Middle')]
  lines.append(f"k={k}："+'；'.join(f"Middle 相对 {r['right']} 为 {r['difference_pp']:+.2f} pp（95% CI {r['ci95_low_pp']:+.2f} 至 {r['ci95_high_pp']:+.2f}）" for r in rr)+'。')
 declines=[r for r in adjacent if r['difference_pp']<0]
 lines+=['','观测到的非单调相邻预算变化：'+('；'.join(f"{r['from_condition']} → {r['to_condition']}：{r['difference_pp']:+.2f} pp，CI [{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}]" for r in declines) if declines else '无负向点估计')+'。逐题受损/改善计数见 adjacent_changes.csv；重点预算不同区域的全部逐题差异见 priority_disagreements.csv。不能从总体均值推断每题都受益。','', '## 复用与可复现性','']
 lines[-3:-3]=['','以下为事后描述的损害案例（按冻结题序取 k=14 前两个 Middle 低于另一窗口的例子，不用于选择层集合）：','']+[f"- {h['key']}：{h['question']}；Middle「{h['left_answer']}」（软分 {h['left_score']:.2f}），{h['right']}「{h['right_answer']}」（软分 {h['right_score']:.2f}）；十答案 Gold：{h['gold_answers']}。" for h in [r for r in harm if r['k']==14 and r['difference']<0][:2]]
 for c,r in protocol['reuse'].items():lines.append(f"- {c}：{r['evidence']} / {r['label']}，证据 Commit `{r['commit']}`。")
 lines+=['','4 个指定证据分支的 500 题 Manifest 和顺序完全一致。恢复后逐题 Question ID、原始问题、Image SHA 库存、十答案有序 Gold、渲染 Prompt、展开 Token，以及 FP16 图像张量哈希均严格匹配历史绑定；历史预测均以冻结官方评分器重算核对。历史 SHA 库存由此前已验证的备份 ZIP SHA 及元数据绑定，本轮没有重新计算全部图片 SHA。图表数据逐点标注来源、历史 Commit 和执行冻结 SHA。','', 'GPU Smoke 覆盖 Native/No-op 全 Token 与全 Logit 一致、真实 Decode 边界、全部 19 个新层集合、禁图概率精确为 0、非图 Mask 保持不变、Prefill QKV 与完整 KV 一致、原生 DynamicCache 与全部 MLP/QKV 调用、首答案 Token/Logit Native 一致、EOS 和评分流程；2 题对 4 个复用点的历史 Token 一致性通过。详见 gpu_smoke.json。所有生成保留原生全部 32 层真实 KV；没有进行性能 Profiling。','', '历史 L11–L20（45.14%）、L12–L24（47.34%）及 L10–L24（48.10%）不属于本次冻结 Middle 曲线，不连接到主图，也不参与同预算比较。','', '唯一主图：figure_accuracy.png / figure_accuracy.pdf。逐题预测：per_question.jsonl.gz；大型可恢复检查点留在服务器 checkpoints/，不提交 Git。输入/执行冻结、运行环境、状态及 Smoke 证据均保留。','', '失败项：无（如有未完成条件，分析程序拒绝生成本报告）。任务完成后停止，不自动启动 Physical C3 或其他实验。']
 (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 atomic(OUT/'analysis_status.json',dict(status='PASS',conditions=23,new_generations=9500,reused_generations=2000,questions=500,clusters=309,bootstrap_replicates=20000,bootstrap_seed=20261108))
 print('ANALYSIS COMPLETE',json.dumps({r['condition']:round(r['accuracy_pct'],4) for r in summary}),flush=True)
if __name__=='__main__':main()
