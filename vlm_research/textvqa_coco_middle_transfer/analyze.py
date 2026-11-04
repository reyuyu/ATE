"""Report verified reuse and explicitly missing transfer results; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,csv,json,hashlib,collections,datetime,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .policy import OUT as O,LAYER_SETS,NEW,COMPARISONS,SEED,ROOT
H=ROOT/'results/textvqa_visual_reaccess_budget_exploration'
def rows(p):return list(csv.DictReader(p.open()))
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rr):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
def main():
 assert read(O/'validation.json')['status']=='BLOCKED_NO_FREE_GPU'
 audit=read(O/'historical_reuse_audit.json');f=read(O/'classification_freeze.json');assert sha(O/'ocr_labels.csv')==f['labels_sha256']
 historical=[r for r in rows(H/'per_question.csv') if r['condition'] in ['A','B','F']]
 for r in historical:r['origin']='audited historical reuse';r['new_generation']=False
 csvwrite(O/'per_question.csv',historical)
 by=collections.defaultdict(dict)
 for r in historical:by[r['question_id']][r['condition']]=r
 order=sorted(by,key=lambda q:by[q]['A']['image_sha256']);assert len(order)==300
 scores=np.array([[float(by[q][c]['score']) for c in ['A','F','B']] for q in order]);labels={r['question_id']:r['category'] for r in rows(O/'ocr_labels.csv')}
 rng=np.random.default_rng(SEED);weights=np.concatenate([rng.multinomial(300,np.ones(300)/300,size=500) for _ in range(40)])
 conf=lambda x:tuple(float(v) for v in np.percentile(x,[2.5,97.5]))
 summary=[];groups=[]
 for group in ['All300','OCR-Direct','OCR-Reasoning','Non-OCR Visual','Ambiguous','OCR-Combined']:
  idx=np.array([group=='All300' or labels[q]==group or (group=='OCR-Combined' and labels[q].startswith('OCR-')) for q in order]);n=int(idx.sum());denom=weights@idx.astype(float);valid=denom>0;bs=(weights[valid]@(scores*idx[:,None]))/denom[valid,None];mean=scores[idx].mean(0)
  for c in ['A','F','B']+NEW:
   j=['A','F','B'].index(c) if c in ['A','F','B'] else None;lo,hi=conf(100*bs[:,j]) if j is not None else ('','')
   r=dict(group=group,condition=c,name=LAYER_SETS[c]['name'],n_available=n if j is not None else 0,n_target=n,accuracy_pct=100*mean[j] if j is not None else '',CI95_low_pct=lo,CI95_high_pct=hi,delta_vs_F_pp=100*(mean[j]-mean[1]) if j is not None else '',delta_vs_A_pp=100*(mean[j]-mean[0]) if j is not None else '',Decode_layers=len(LAYER_SETS[c]['Decode']),origin=LAYER_SETS[c]['origin'],status='AUDITED_HISTORICAL' if j is not None else 'NOT_RUN_NO_FREE_GPU',valid_bootstrap_draws=int(valid.sum()) if j is not None else 0)
   (summary if group=='All300' else groups).append(r)
 csvwrite(O/'accuracy_summary.csv',summary);csvwrite(O/'ocr_group_accuracy.csv',groups)
 paired=[dict(left=a,right=b,left_name=LAYER_SETS[a]['name'],right_name=LAYER_SETS[b]['name'],n_available=0,n_target=300,delta_pp='',CI95_low_pp='',CI95_high_pp='',raw_p='',Holm_p='',status='NOT_RUN_NO_FREE_GPU',note='12 vs 11 Decode layers' if b=='E' else 'same 11-layer budget' if b in ['D','R'] else '') for a,b in COMPARISONS]
 csvwrite(O/'paired_comparisons.csv',paired)
 # Historical-only paired intervals are independent of the absent new conditions.
 bs=weights@scores/300;histpairs=[]
 for a,b in [('A','F'),('F','B'),('A','B')]:
  i,j=['A','F','B'].index(a),['A','F','B'].index(b);lo,hi=conf(100*(bs[:,i]-bs[:,j]));d=scores[:,i]-scores[:,j]
  histpairs.append(dict(left=a,right=b,n=300,delta_pp=100*float(d.mean()),CI95_low_pp=lo,CI95_high_pp=hi,score_up=int((d>1e-12).sum()),score_down=int((d< -1e-12).sum()),origin='audited historical reuse only'))
 csvwrite(O/'historical_baseline_comparisons.csv',histpairs)
 lookup={r['condition']:r for r in summary};display=['A','F','D','C','E','R','B'];colors={'A':'#333333','F':'#999999','B':'#bbbbbb'}
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
 fig,ax=plt.subplots(figsize=(10,5.2))
 for k,c in enumerate(display):
  r=lookup[c]
  if r['accuracy_pct']=='':ax.text(k,7,'NOT RUN',ha='center',va='center',rotation=90,color='#777777',fontsize=10);continue
  v=r['accuracy_pct'];ax.bar(k,v,color=colors[c],width=.55);ax.errorbar(k,v,yerr=[[v-r['CI95_low_pct']],[r['CI95_high_pct']-v]],fmt='none',capsize=4,color='black');ax.text(k,r['CI95_high_pct']+1,f'{v:.3f}%',ha='center',fontsize=9)
 ax.set_xticks(range(len(display)),['Native\nFull','P-1 Full\nDecode None','COCO\nEarly','COCO\nMiddle','COCO\nLate','Random\n11 layers','Both\nNone'])
 ax.set_ylim(0,65);ax.set_ylabel('Official TextVQA consensus accuracy (%)');ax.set_title('TextVQA300 COCO transfer — resource-blocked preflight\nOnly historical baselines measured; pointwise image-cluster 95% CI');ax.spines[['right','top']].set_visible(False);ax.set_axisbelow(True);ax.yaxis.grid(alpha=.2);fig.tight_layout();fig.savefig(O/'accuracy_comparison.png',dpi=200);fig.savefig(O/'accuracy_comparison.pdf');plt.close(fig)
 snap=read(O/'resource_preflight.json');random_layers=LAYER_SETS['R']['Decode'];counts=read(O/'classification_freeze.json')['counts']
 lines=['# COCO中层Decode视觉回看迁移至TextVQA：资源阻塞预检','','**状态：BLOCKED_NO_FREE_GPU；实验未完成。四个新条件均未运行，不能报告COCO-Middle迁移后的Accuracy。**','', '已创建独立Worktree和分支，完成历史基线、输入、OCR标签和CPU Mask审计。本次GPU检查中4张RTX3090均有其他计算进程，各占约13.5–14.1GiB；没有真正空闲设备。未加载LLaVA，未启动GPU Worker，未创建后台等待/重试任务。按单张真正空闲3090的约束停止在GPU Smoke之前。','', '## 已核验与未执行范围','', '复用完全相同TextVQA300，原始Manifest字节SHA不变，逐图SHA核验通过；900条A/F/B历史答案用官方十人共识软评分重算差异为0，全部原检查点SHA、Token、文本和输入哈希一致。没有重复生成历史基线。冻结OCR标签逐字节复用原Commit，不重新分类。','', '| 条件 | Decode读图层 | n已完成/目标 | ACC | 相对Full Δpp | 来源 |','|---|---|---|---:|---:|---|']
 for c in display:
  r=lookup[c];measured=r['accuracy_pct']!='';acc=f"{r['accuracy_pct']:.3f}%" if measured else '未测';delta=f"{r['delta_vs_A_pp']:+.3f}" if measured else 'NA'
  lines.append(f"| {r['name']} | {r['Decode_layers']} | {r['n_available']}/300 | {acc} | {delta} | {'已审计历史复用' if measured else 'GPU资源阻塞'} |")
 lines+=['', 'A的Prefill/P−1与Decode全32层读图；F的Prefill/P−1全开放而Decode完全禁读；B的P−1/Decode均禁读。四个新增条件的整个Prefill保持原生，包括P−1；只在真实增量Decode按层Hard Mask原始Image Key。保留全部KV、其他文本Attention、MLP、Residual和RoPE，不做o_proj消融。','', f"Random-11预先固定seed={SEED}，Python random.Random(seed).sample(range(1,32),10)，另保留L0，具体层为`{random_layers}`。不依据Accuracy选择种子。Middle/Early/Random各11层，Late12层，后者多一层的预算差必须保留。",'', '历史COCO的层集合与本轮C/D/E逐项严格相同；COCO的Caption生成上限96 Token，本轮TextVQA冻结32 Token，不能复用COCO指标或把COCO结果当TextVQA结果。','', '## 冻结OCR分组','', '| 类别 | n | Native A | Decode None F | Both None B | 新条件 |','|---|---:|---:|---:|---:|---|']
 for g in counts:
  rr={r['condition']:r for r in groups if r['group']==g};lines.append('| '+g+' | '+str(counts[g])+' | '+' | '.join(f"{rr[c]['accuracy_pct']:.3f}%" for c in ['A','F','B'])+' | 未运行 |')
 lines+=['', '标签是上轮输入屏蔽式AI复核标签，非独立人类真值；Reasoning仅12题、Non-OCR仅18题，均需保持小样本与分类误差限制。Middle在OCR-Direct的恢复尚无观测，不能用COCO或历史Both预算条件代替。','', '## 正确性与统计预设','', '10项CPU测试通过：C/D/E精确层集合、Random预算复现、所有新条件完整Prefill Mask逐位不变、Decode只修改Image边、实际CPU Softmax禁止概率严格0、保留层Mask逐位原生、No-op不改变Mask、A/F/B阶段语义、位置边界/文本连接保护以及CUDA未初始化。这些是Mask策略/张量测试，不是7B GPU Smoke。','', '真实GPU的Native/No-op有效Logits torch.equal、每个新条件首位置与Native严格一致、模型实际禁止Attention概率0、允许层实际读图、独立KV、32层MLP、输入/视觉特征/参数哈希、位置与Hook清理、完整EOS检查均未执行，validation.json明确标记NOT_RUN，不能将历史Smoke借作本轮通过证明。','', f'冻结Image SHA整簇配对Bootstrap 20,000次，seed={SEED}，普通95%CI；C−A、C−F、C−D、C−R、C−E五项作为主要比较，用整簇配对sign-flip的加一修正p后Holm控制。OCR组分析为描述性区间，不做未经校正的显著性宣称；没有观察结果前不进行层集合调整或另选种子。新比较均留空而非填0，不能将缺失视作无损。','', '## 对六个研究问题的当前回答','', '1. COCO-Middle的TextVQA Accuracy：未测。','2. 相对Decode None恢复：未测；已核验F=31.400%，A=45.233%，两者相差13.833pp。','3. 与Native Full缺口：未测。','4. 是否优于Early和Random-11：未测，不能判断。','5. OCR-Direct是否明显恢复：未测，不能判断。','6. 是否值得实现物理视觉KV加速：本轮尚无迁移结果，不能据此支持开发；Hard Mask本身也没有减少实际KV存储或证明加速。','', '## 资源现场','', '```text',snap['inventory'],'','GPU UUID / PID / 显存：',snap['compute_processes'],'```','',f"/data预检可用空间{snap['disk']['free']/1024**3:.2f}GiB，空间足够；阻塞原因是GPU占用。没有抢占、杀进程或跨卡加载，没有本轮GPU显存需要释放。",'', '## 文件与恢复边界','', 'accuracy_summary.csv、ocr_group_accuracy.csv中已测项是历史复用，新项明确NOT_RUN_NO_FREE_GPU；per_question.csv仅含900条已核验A/F/B记录，不伪造1200条新答案。paired_comparisons.csv保留5项冻结比较及缺失状态。图中的NOT RUN不表示Accuracy为0。','', '![历史基线与未运行条件](accuracy_comparison.png)','', '新条件尚无检查点或真实Smoke，不得标记COMPLETE。恢复时需重新检查实际GPU占用，取得一张空闲3090后运行真实Smoke，再按冻结层集合单卡串行完成1200条；本次未设置自动恢复。新包engine.py只为复用已验证Mask函数绑定本轮策略，尚未交付或验收正式GPU采集入口。','', '历史与main不修改；本次仅提交资源阻塞下的预检、协议、可审阅CPU策略源码及历史审计，不声称完成迁移实验。']
 (O/'report_zh.md').write_text('\n'.join(lines)+'\n')
 # Independently check every available subgroup point against the delivered OCR study.
 previous={(r['group'],r['condition']):float(r['accuracy_pct']) for r in rows(ROOT/'results/textvqa_ocr_reaccess_mechanism/ocr_group_accuracy.csv')}
 for r in groups:
  if r['status']=='AUDITED_HISTORICAL':assert abs(r['accuracy_pct']-previous[(r['group'],r['condition'])])<1e-10
 assert len(historical)==900 and not any(r['condition'] in NEW for r in historical)
 assert all(r['accuracy_pct']=='' and r['n_available']==0 for r in summary if r['condition'] in NEW)
 write(O/'report_validation.json',dict(passed=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),historical_subgroup_accuracy_matches_frozen_OCR_report=True,no_new_results_fabricated=True,new_comparisons_explicitly_missing=True,per_question_records=900,new_generations=0,experiment_complete=False))
 print('CPU_PRECHECK_REPORT_READY; GPU_GENERATIONS=0; STATUS=BLOCKED_NO_FREE_GPU')
if __name__=='__main__':main()
