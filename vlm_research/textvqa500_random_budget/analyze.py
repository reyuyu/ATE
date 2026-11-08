"""One four-curve figure; paired image-cluster uncertainty, conditional on one seed."""
from .common import *
import csv,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..benchmarks.scoring import textvqa_score

def write_csv(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(x):return tuple(float(v) for v in np.percentile(x,[2.5,97.5]))

def main():
 assert read(OUT/'run_status.json')['status']=='COMPLETE' and read(OUT/'gpu_smoke.json')['status']=='PASS'
 analysis_spec=read(OUT/'analysis_specification.json');assert analysis_spec['replicates']==20000
 protocol=read(OUT/'protocol.json');samples=load_samples();keys=list(samples);hist=load_history();all_rows={};records=[]
 conditions=['NoImage','Native','Native4090']+[r+str(k) for r in ['Early','Middle','Late','Random'] for k in protocol['budgets'][1:-1]]
 freeze=read(OUT/'execution_freeze.json');run_id=digest(freeze)
 for path,h in freeze['source_hashes'].items():assert sha(ROOT/path)==h,path
 bindings=read(LOCAL/'input_bindings.json')
 for c in conditions:
  rr=[]
  if c.startswith('Random') or c=='Native4090':
   for key in keys:
    w=read(LOCAL/'checkpoints'/c/(key+'.json'));p=w['payload'];assert w['sha256']==digest(p) and p['run_id'] in [run_id]+protocol['checkpoint_reuse']['accepted_parent_run_ids'];g=p['generation']
    s=samples[key];assert p['key']==key and p['condition']==c and p['question_id']==s['question_id'] and p['image_sha256']==s['image_sha256'] and p['input_binding_sha256']==bindings[key]
    if c.startswith('Random'):assert g['first_top1_token']==read(LOCAL/'checkpoints'/'Native4090'/(key+'.json'))['payload']['generation']['first_top1_token']
    rr.append(dict(key=key,condition=c,answer=g['text'],score=p['score'],token_ids=g['token_ids'],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],first_top1_token=g['first_top1_token'],input_binding_sha256=p['input_binding_sha256'],source='new_RTX4090',source_commit='',generation_source_commit=''))
  else:
   rr=[dict(hist[c][key],source='historical_RTX3090',generation_source_commit=hist[c][key].get('source_commit','') or REF,source_commit=REF) for key in keys]
  assert len(rr)==500 and [r['key'] for r in rr]==keys
  for row in rr:
   s=samples[row['key']];assert abs(row['score']-textvqa_score(row['answer'],s['answer']))<1e-12
   row.update(question_id=s['question_id'],image_sha256=s['image_sha256'])
  records+=rr;all_rows[c]=rr
 assert len(records)==15500
 scores={c:np.array([r['score'] for r in all_rows[c]]) for c in conditions};X=np.stack([scores[c] for c in conditions],1)
 clusters=sorted({s['image_sha256'] for s in samples.values()});assert len(clusters)==309;index={s:i for i,s in enumerate(clusters)}
 sums=np.zeros((309,len(conditions)));counts=np.zeros(309)
 for i,key in enumerate(keys):j=index[samples[key]['image_sha256']];sums[j]+=X[i];counts[j]+=1
 rng=np.random.default_rng(20261108);weights=rng.multinomial(309,np.full(309,1/309),size=20000);bootstrap=(weights@sums)/(weights@counts)[:,None]
 means=X.mean(0);col={c:i for i,c in enumerate(conditions)};n=col['Native'];z=col['NoImage'];summary=[]
 for c in conditions:
  j=col[c];nn=col['Native4090'] if c.startswith('Random') or c=='Native4090' else n;lo,hi=ci(100*bootstrap[:,j]);ll,lh=ci(100*(bootstrap[:,nn]-bootstrap[:,j]));keep=protocol['decode_image_layers'][c]
  summary.append(dict(condition=c,k=len(keep),layers=json.dumps(keep),questions=500,image_clusters=309,accuracy_pct=100*means[j],ci95_low_pct=lo,ci95_high_pct=hi,native_reference_condition=conditions[nn],native_loss_pp=100*(means[nn]-means[j]),native_loss_ci_low_pp=ll,native_loss_ci_high_pp=lh,recovery_fraction_vs_NoImage=(means[j]-means[z])/(means[nn]-means[z]),source='new_RTX4090' if c.startswith('Random') or c=='Native4090' else 'historical_RTX3090',source_commit='' if c.startswith('Random') or c=='Native4090' else REF,execution_freeze_sha256=sha(OUT/'execution_freeze.json'),eos=sum(r['eos'] for r in all_rows[c]),empty=sum(r['empty'] for r in all_rows[c]),truncated=sum(r['truncated'] for r in all_rows[c])))
 with (OUT/'historical_accuracy_summary.csv').open() as f:
  for old in csv.DictReader(f):assert abs(float(old['accuracy_pct'])-100*means[col[old['condition']]])<1e-10
 cal_a,cal_b=col['Native4090'],col['Native'];cal_delta=100*(means[cal_a]-means[cal_b]);cal_lo,cal_hi=ci(100*(bootstrap[:,cal_a]-bootstrap[:,cal_b]));calibration=dict(status='MEASURED',questions=500,native_3090_accuracy_pct=100*means[cal_b],native_4090_accuracy_pct=100*means[cal_a],difference_pp=cal_delta,ci95_low_pp=cal_lo,ci95_high_pp=cal_hi,first_token_changed=sum(a['first_top1_token']!=b['first_top1_token'] for a,b in zip(all_rows['Native4090'],all_rows['Native'])),complete_token_sequence_changed=sum(a['token_ids']!=b['token_ids'] for a,b in zip(all_rows['Native4090'],all_rows['Native'])),question_score_changed=int(np.sum(scores['Native4090']!=scores['Native'])),user_authorized=True,old_curves_not_shifted=True)
 atomic(OUT/'native_cross_hardware_calibration.json',calibration)
 pairs=[]
 for k in protocol['budgets'][1:-1]:
  for region in ['Early','Middle','Late']:
   a,b='Random'+str(k),region+str(k);delta=100*(means[col[a]]-means[col[b]]);bs=100*(bootstrap[:,col[a]]-bootstrap[:,col[b]]);lo,hi=ci(bs)
   pairs.append(dict(k=k,left=a,right=b,difference_pp=delta,ci95_low_pp=lo,ci95_high_pp=hi,p_unadjusted=(1+np.count_nonzero(abs(bs-delta)>=abs(delta)))/20001,p_holm_21=0.,emphasized=k in [14,16,21],improved_questions=int(np.sum(scores[a]>scores[b])),harmed_questions=int(np.sum(scores[a]<scores[b]))))
 previous=0
 for rank,row in enumerate(sorted(pairs,key=lambda r:r['p_unadjusted'])):previous=max(previous,min(1,(21-rank)*row['p_unadjusted']));row['p_holm_21']=previous
 lookup={r['condition']:r for r in summary};plot=[]
 for region in ['Early','Middle','Late','Random']:
  for k in protocol['budgets']:
   c='NoImage' if k==0 else ('Native4090' if region=='Random' else 'Native') if k==32 else region+str(k);plot.append(dict(region=region,**lookup[c],random_seed=20261108 if region=='Random' else ''))
 write_csv(OUT/'accuracy_summary.csv',summary);write_csv(OUT/'paired_comparisons.csv',pairs);write_csv(OUT/'plot_data.csv',plot)
 with gzip.open(OUT/'per_question.jsonl.gz','wt',encoding='utf-8') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
 fig,ax=plt.subplots(figsize=(8.7,5.4));fig.subplots_adjust(left=.09,right=.99,bottom=.22,top=.91);colors={'Early':'#1f77b4','Middle':'#d1495b','Late':'#30966a','Random':'#7955a3'}
 for region,marker in [('Early','o'),('Middle','s'),('Late','^'),('Random','D')]:
  rr=[r for r in plot if r['region']==region];y=np.array([r['accuracy_pct'] for r in rr]);low=np.array([r['ci95_low_pct'] for r in rr]);high=np.array([r['ci95_high_pct'] for r in rr])
  ax.plot(protocol['budgets'],y,marker=marker,color=colors[region],label='Random (seed=20261108)' if region=='Random' else region,linewidth=1.7,markersize=4.7,linestyle='--' if region=='Random' else '-')
  ax.errorbar(protocol['budgets'],y,yerr=[y-low,high-y],fmt='none',ecolor=colors[region],alpha=.25,elinewidth=.8,capsize=2)
 r=lookup['Middle14'];ax.scatter([14],[r['accuracy_pct']],s=105,facecolors='none',edgecolors=colors['Middle'],linewidths=1.6,zorder=5)
 ax.annotate('Middle k=14: L11-L24',xy=(14,r['accuracy_pct']),xytext=(17,54.5),arrowprops={'arrowstyle':'->','color':colors['Middle']},fontsize=9,color=colors['Middle'])
 for k,c in [(0,'NoImage'),(32,'Native')]:ax.scatter([k],[lookup[c]['accuracy_pct']],color='#202020',s=28,zorder=6)
 ax.set(xlabel='Number of Decode Visual Re-access Layers',ylabel='TextVQA Soft Accuracy (%)',title='TextVQA Accuracy vs. Number of Decode Visual Re-access Layers',xticks=protocol['budgets'],xlim=(-.8,32.8))
 ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,ncol=2,loc='lower right')
 fig.text(.015,.015,'500 frozen image-only questions; 309 image clusters. Pointwise 95% CIs; Random CI conditional on one seed.\nDense decode masks only. Historical curves: RTX3090; Random / calibrated Native: RTX4090. No-image endpoint reused from RTX3090.',fontsize=7.4)
 fig.savefig(OUT/'figure_accuracy.png',dpi=300,bbox_inches='tight');fig.savefig(OUT/'figure_accuracy.pdf',bbox_inches='tight');plt.close(fig)
 adjacent=[];chain=['NoImage']+['Random'+str(k) for k in protocol['budgets'][1:-1]]+['Native4090']
 for a,b in zip(chain,chain[1:]):
  lo,hi=ci(100*(bootstrap[:,col[b]]-bootstrap[:,col[a]]));adjacent.append(dict(from_condition=a,to_condition=b,difference_pp=100*(means[col[b]]-means[col[a]]),ci95_low_pp=lo,ci95_high_pp=hi,improved_questions=int(np.sum(scores[b]>scores[a])),harmed_questions=int(np.sum(scores[b]<scores[a]))))
 write_csv(OUT/'random_adjacent_changes.csv',adjacent)
 lines=['# TextVQA500：补充 Random 层预算曲线','','完成7个Random预算条件×500题=3,500条，加用户授权4090 Native校准500条，正式新增4,000条。固定seed=20261108，只抽一次32层排列，各k取排列前k层；所有题共用各预算冻结集合，集合嵌套。没有搜索种子或挑选最佳结果。','','同一历史500题、309图像簇，Image-only；完整Prefill及P−1保持Native，仅真正Decode禁止未开放层读取原始Image Key。FP16/Eager、Transformers4.49.0、固定LLaVA Revision、greedy/cap32及官方十答案软评分全部保持。Native/禁图端点及Early/Middle/Late共23个历史条件复用，来源Commit `'+REF+'`。','','历史曲线在RTX3090生成；本次Random在RTX4090生成。所有500题的Question ID、Gold顺序、图片SHA库存、Prompt及FP16输入张量绑定严格一致；GPU Smoke两题Native/NoImage/Middle14完整Token与历史一致，Native与No-op全Logit/KV一致，7种Random集合通过Decode边界、首答案、Prefill、KV及禁图概率0检查。','','| k | Early % | Middle % | Late % | Random %（95% CI） |','|---:|---:|---:|---:|---|']
 for k in protocol['budgets']:
  cs={r:'NoImage' if k==0 else ('Native4090' if r=='Random' else 'Native') if k==32 else r+str(k) for r in ['Early','Middle','Late','Random']};rr=lookup[cs['Random']]
  lines.append(f"| {k} | {lookup[cs['Early']]['accuracy_pct']:.2f} | {lookup[cs['Middle']]['accuracy_pct']:.2f} | {lookup[cs['Late']]['accuracy_pct']:.2f} | {rr['accuracy_pct']:.2f} [{rr['ci95_low_pct']:.2f}, {rr['ci95_high_pct']:.2f}] |")
 lines += ['',f"4090 Native校准：{calibration['native_4090_accuracy_pct']:.2f}%，3090历史Native {calibration['native_3090_accuracy_pct']:.2f}%；同题配对差值{cal_delta:+.2f}pp（95% CI {cal_lo:+.2f}至{cal_hi:+.2f}）。首Token变化{calibration['first_token_changed']}/500，完整Token序列变化{calibration['complete_token_sequence_changed']}/500，逐题软分变化{calibration['question_score_changed']}/500。",'', 'Random k32采用真实4090 Native校准结果，其他三条曲线保持历史3090端点。Random的Native损失以Native4090为参照；k0继续复用历史NoImage。所有Accuracy均为原始测量，未按校准差值平移；同预算跨卡比较保留运行硬件差异这一限制。']
 lines+=['','本次固定随机集合在全部七个中间预算的样本均值均低于Middle。k=14/16/21时，Middle减Random分别为+6.64/+6.96/+3.64pp，对应普通95%配对CI为[+4.23,+9.21]/[+4.36,+9.70]/[+1.57,+5.80]pp，21项Holm校正p为0.0010/0.0010/0.0112。该结果只针对这一冻结随机排列；跨卡差异及未覆盖种子间波动限制结论的推广。','','普通95% CI：同题配对、按Image SHA整簇重采样20,000次（seed20261108），按问题等权均值。CI条件于这一个层排列，反映题目/图像簇重采样，未涵盖随机种子之间波动。21项Random对Early/Middle/Late比较统一报告Holm校正p；普通CI未作同时覆盖校正。CI跨0不能证明等价。冻结protocol含两处多重比较描述不一致，统计前说明analysis_specification.json明确采用更保守的21项统一Holm族，未改写生成冻结记录。','','| k | 比较（Random减区域） | 差值pp（95% CI） | Holm p（21项） |','|---:|---|---|---:|']
 for r in pairs:
  if r['emphasized']:lines.append(f"| {r['k']} | {r['left']} − {r['right']} | {r['difference_pp']:+.2f} [{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}] | {r['p_holm_21']:.4f} |")
 lines+=['','Random各预算相对同卡Native4090的损失、相对历史Decode禁图端点的恢复比例：','']
 for k in protocol['budgets'][1:-1]:
  rr=lookup['Random'+str(k)];lines.append(f"- k={k}：{rr['accuracy_pct']:.2f}%，Native损失{rr['native_loss_pp']:.2f}pp，恢复比例{rr['recovery_fraction_vs_NoImage']:.3f}；开放层{protocol['decode_image_layers']['Random'+str(k)]}。")
 declines=[r for r in adjacent if r['difference_pp']<0];lines+=['','Random非单调变化：'+('；'.join(f"{r['from_condition']}→{r['to_condition']} {r['difference_pp']:+.2f}pp" for r in declines) if declines else '本次离散测量点未出现均值下降')+'。逐题损害/改善数量及配对CI见random_adjacent_changes.csv与paired_comparisons.csv。','', '单张主图为figure_accuracy.png/pdf；所有离散点直接连接，未平滑或补造测量。Dense功能性实验只描述Decode视觉访问权限预算；未测量物理FLOPs、KV或Latency节省。','', 'Native4090为追加校准500题；禁图端点沿用3090，未追加NoImage500。历史23个条件的Accuracy与原CSV完全一致。全部15,500条逐题记录（历史11,500 + 新增4,000）保存于per_question.jsonl.gz。新生成的可恢复检查点存于本地实例盘并单独归档，Git仅同步精简证据。','', '执行曾因跨卡历史首Token变化停止；Native/No-op/Random诊断确认同卡一致后，用户授权追加Native500。原255条Random4检查点保留原run_id并通过父冻结记录复用，未修改生成算法、随机集合或旧记录。最终未完成条件/非有限值/协议错误：无。完成后停止。']
 new_records=[r for c in conditions if c.startswith('Random') or c=='Native4090' for r in all_rows[c]]
 lines+=['',f"新增4,000条均非空；{sum(r['eos'] for r in new_records)}条正常EOS，{sum(r['truncated'] for r in new_records)}条按冻结max_new_tokens=32上限截止，均按相同官方评分流程保留，未延长生成。"]
 (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
 atomic(OUT/'analysis_status.json',dict(status='PASS',conditions=31,new_generations=4000,reused_generations=11500,native_calibration_generations=500,questions=500,image_clusters=309,random_seed=20261108,bootstrap_replicates=20000,bootstrap_seed=20261108,analysis_specification_sha256=sha(OUT/'analysis_specification.json')))
 print('RANDOM_ANALYSIS_COMPLETE',json.dumps({c:round(100*means[col[c]],4) for c in conditions if c.startswith('Random')}),flush=True)
if __name__=='__main__':main()
