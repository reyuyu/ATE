"""CPU-only full-cohort verification, paired descriptive statistics and report."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,gzip,hashlib,importlib.util,json,sqlite3,subprocess,sys
import numpy as np
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/v2drop_prefix_svh_accuracy_3090';V=R/'vlm_research/results/v2drop_quick_reproduction'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jl(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def csv_out(name,rr):
 if not rr:return
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
p=read(O/'protocol.json');state=read(O/'progress.json');smoke=read(O/'physical_smoke.json');audit=read(O/'baseline_audit.json')
with gzip.open(O/'input_manifest.jsonl.gz','rt') as f:samples=[json.loads(l) for l in f]
lookup={s['key']:s for s in samples};binding=digest(p)
spec=importlib.util.spec_from_file_location('unmodified_official_textvqa',V/'frozen_scoring_sources/m4c_evaluator.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);e=m.TextVQAAccuracyEvaluator()
def score(s,text):
 if s['benchmark']=='GQA':parsed=text.rstrip('.').lower();return parsed,float(parsed==s['answer'])
 parsed=e.answer_processor(text);aa=[e.answer_processor(x) for x in s['answer']];matches=aa.count(parsed)
 value=sum(min((matches-int(a==parsed))/3,1) for a in aa)/10
 assert abs(value-e._compute_answer_scores(s['answer']).get(parsed,0.0))<1e-12
 return parsed,value
baseline=jl(O/'baseline_reuse.jsonl');assert len(baseline)==600
records={};allrows=[]
for r in baseline:
 s=lookup[r['key']];parsed,value=score(s,r['raw_output']);assert parsed==r['parsed'] and abs(value-r['score'])<1e-10
 assert r['gold']==s['answer'] and r['image_sha256']==s['image_sha256'] and r['prompt_sha256']==s['prompt_sha256']
 allrows.append(r)
with sqlite3.connect('file:'+str(O/'predictions.sqlite')+'?mode=ro',uri=True) as db:
 assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
 for key,payload,checksum in db.execute('SELECT key,payload,sha256 FROM predictions'):
  r=json.loads(payload);s=lookup[key];assert digest(r)==checksum and r['binding']==binding and r['sample_sha256']==s['sample_sha256']
  assert r['gold']==s['answer'] and r['input_hashes']==s['input_hashes']
  g=r['generation'];parsed,value=score(s,g['text']);assert parsed==r['scoring']['parsed'] and abs(value-r['scoring']['score'])<1e-10
  assert g['empty']==(not bool(g['text'])) and g['truncated']==(len(g['token_ids'])==s['max_new_tokens'] and not g['eos'])
  assert [x['layer'] for x in r['pruning_events']]==[3,17] and [x['visual_after'] for x in r['pruning_events']]==[288,144]
  records[key]=r;allrows.append(dict(key=key,benchmark=s['benchmark'],question_id=s['question_id'],method=p['condition'],raw_output=g['text'],token_ids=g['token_ids'],parsed=parsed,score=value,gold=s['answer'],empty=g['empty'],truncated=g['truncated'],image_sha256=s['image_sha256'],prompt_sha256=s['prompt_sha256'],input_hashes=s['input_hashes'],sample_sha256=s['sample_sha256']))
assert len(records)==state['completed']
complete=state['status']=='COMPLETE' and len(records)==200 and smoke['status']=='PASS'
if complete:assert set(records)==set(lookup)
methods=['Native','V2Drop','llava-SVH',p['condition']];summary=[]
by={(r['method'],r['key']):r for r in allrows};assert len(by)==len(allrows)
for method in methods:
 for bench in ['GQA','TextVQA']:
  rows=[r for r in allrows if r['method']==method and r['benchmark']==bench]
  summary.append(dict(method=method,benchmark=bench,completed=len(rows),expected=100,
   accuracy_percent=100*sum(r['score'] for r in rows)/100 if len(rows)==100 else None,
   score_sum=sum(r['score'] for r in rows) if len(rows)==100 else None,
   empty=sum(r['empty'] for r in rows),truncated=sum(r['truncated'] for r in rows),source='NEW_FUSION_ONLY' if method==p['condition'] else 'REUSED_ARCHIVE'))
paired=[];changed=[];wide=[]
def bootstrap(differences,groups):
 bins=defaultdict(list)
 for d,g in zip(differences,groups):bins[g].append(d)
 sums=np.array([sum(v) for v in bins.values()]);counts=np.array([len(v) for v in bins.values()])
 rng=np.random.default_rng(20261109);index=rng.integers(0,len(bins),size=(10000,len(bins)))
 estimates=100*sums[index].sum(1)/counts[index].sum(1)
 return dict(low=float(np.quantile(estimates,.025)),high=float(np.quantile(estimates,.975)),clusters=len(bins),replicates=10000,seed=20261109,unit='Image SHA256 cluster',resample='Paired image clusters with replacement, question-weighted mean within each resample')
if complete:
 for bench in ['GQA','TextVQA']:
  cohort=[s for s in samples if s['benchmark']==bench]
  for control in ['llava-SVH','Native','V2Drop']:
   rr=[(s,by[p['condition'],s['key']],by[control,s['key']]) for s in cohort]
   diffs=[a['score']-b['score'] for s,a,b in rr]
   interval=bootstrap(diffs,[s['image_sha256'] for s,a,b in rr])
   paired.append(dict(benchmark=bench,comparison=p['condition']+' minus '+control,delta_pp=100*sum(diffs)/100,
    score_improved=sum(d>0 for d in diffs),score_worsened=sum(d<0 for d in diffs),score_unchanged=sum(d==0 for d in diffs),
    full_credit_to_zero=sum(b['score']==1 and a['score']==0 for s,a,b in rr),zero_to_full_credit=sum(b['score']==0 and a['score']==1 for s,a,b in rr),
    positive_credit_to_zero=sum(b['score']>0 and a['score']==0 for s,a,b in rr),zero_to_positive_credit=sum(b['score']==0 and a['score']>0 for s,a,b in rr),
    answers_changed=sum(a['parsed']!=b['parsed'] for s,a,b in rr),CI95_low_pp=interval['low'],CI95_high_pp=interval['high'],image_clusters=interval['clusters']))
  for s in cohort:
   r=dict(key=s['key'],benchmark=bench,question_id=s['question_id'],image_sha256=s['image_sha256'])
   for method in methods:
    v=by[method,s['key']];r[method+'_output']=v['raw_output'];r[method+'_score']=v['score']
   r['fusion_minus_SVH']=by[p['condition'],s['key']]['score']-by['llava-SVH',s['key']]['score'];wide.append(r)
   if r['fusion_minus_SVH']!=0:changed.append(r)
csv_out('accuracy_summary.csv',summary);csv_out('paired_comparisons.csv',paired);csv_out('per_question_comparison.csv',wide);csv_out('SVH_changed_score_cases.csv',changed)
with gzip.GzipFile(filename=str(O/'all_methods_predictions.jsonl.gz'),mode='wb',mtime=0) as out:
 for r in allrows:out.write((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode())
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used,memory.free','--format=csv,noheader'],text=True).strip()
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
assert not processes,'Experiment process must have released GPU before finalizing'
for name,h in p['frozen_sources'].items():assert sha(R/name)==h,name
result=dict(status='COMPLETE' if complete else state['status'],GPU_physical_smoke=smoke['status'],new_accuracy_generations=len(records),expected=200,
 reused_accuracy_predictions=600,model_loads=state['model_loads'],accuracy=summary,paired_comparisons=paired,
 bootstrap=dict(replicates=10000,seed=20261109,cluster='image_sha256',CI='percentile 95%, paired image-cluster bootstrap',scope='Exploratory fixed 100-question cohorts, not equivalence or noninferiority'),
 source_commit=p['code_commit'],new_executor_identity='Uncommitted isolated source; content SHA256 in protocol.json, not part of historical Git commit',
 model_revision=p['revision'],GPU_after=gpu,GPU_processes_after=processes,GPU_released=True,frozen_sources_unchanged=True,no_Native_or_V2_accuracy_regeneration=True,no_GitHub_push=True,
 scorer='Frozen official GQA conversion/exact match; unchanged author m4c_evaluator ten-answer soft score plus independent leave-one-out audit',
 pruning_budget_caution='Fusion uses L3/L17 only then SVH visual exit before L20; full V2Drop also prunes L22, so budgets differ',
 failure=read(O/'failure.json') if (O/'failure.json').exists() else None)
result['smoke_GPU_generation_requests']=4*len(smoke['cases'])
result['total_GPU_generation_requests']=len(records)+result['smoke_GPU_generation_requests']
result['accuracy_conclusion']='Fixed sample does not establish absence of additional loss: GQA -5.0 pp versus SVH, TextVQA +1.7 pp; exploratory intervals do not prove equivalence' if complete else 'NOT_VALIDATED'
save('score_summary.json',result)
lookup_summary={(r['method'],r['benchmark']):r for r in summary}
lines=['# V²Drop-prefix + llava-SVH：固定 200 题准确率验证','',f"**状态：{result['status']}**；Physical Smoke：{smoke['status']}；融合准确率生成 {len(records)}/200。Native、V²Drop、SVH 的 600 条预测全部复用历史存档。",'',
 '| Method | GQA100 | TextVQA100 Paper-OCR |','|---|---:|---:|']
for method in methods:
 vals=[lookup_summary[method,b]['accuracy_percent'] for b in ['GQA','TextVQA']]
 label={'Native':'Native Full','V2Drop':'V²Drop','V2Drop-prefix+SVH':'V²Drop-prefix + SVH'}.get(method,method)
 lines.append('| '+label+' | '+' | '.join('NOT_VALIDATED' if v is None else f'{v:.1f}%' for v in vals)+' |')
if complete:lines+=['','本次固定样本不支持“融合没有额外准确率损失”的结论：GQA 相对 SVH 下降 5.0 pp，TextVQA 上升 1.7 pp。数值与结构检查通过说明实现满足已验收语义；不意味着准确率等价。没有为改善结果调整层集合、Prompt 或样本。']
lines+=['','GQA 为官方 exact-match；TextVQA 为十答案 VQA Soft Accuracy，保留空输出和软分，不转成普通二值准确率。此表仅为冻结 GQA100 + TextVQA100 的探索性检查，不是完整 Benchmark 成绩。',
 '',f"实际 GPU/释放状态：{gpu}；计算进程：{'无' if not processes else processes}。模型加载 {state['model_loads']} 次；FP16 / Eager / torch 2.6.0+cu124 / Transformers 4.49.0；模型 Revision `{p['revision']}`。",
 f"冻结研究基线 Git Commit：`{p['code_commit']}`。新融合执行器为独立目录中的未提交代码，具体内容 SHA256 记录于 protocol.json；没有改冻结核心、GitHub main 或 Push。",'',
 '协议绑定：严格复用原始 200 行 input_manifest.jsonl.gz（SHA256 '+p['manifest_sha256']+'），不是 efficiency_keys 的 30 行。Question ID、Gold、图像 SHA256、Prompt、Tokenizer、实际 FP16 Pixel Values 与输入 tensor 哈希均核对；GQA 上限 32、TextVQA 上限 128，Greedy、EOS=2。',
 '历史 SVH：从完整 GQA 官方对齐 results.sqlite 与 TextVQA Paper-OCR predictions.jsonl.gz 提取同一 100+100 题。已独立重算 600 条存档分数。TextVQA 历史像素字段为转 FP16 前的 FP32 哈希，两种精度阶段均经 CPU 逐题验证，不是输入差异；历史 SVH TextVQA 的 2 条空输出原样保留。', '',
 '| Benchmark | Fusion − 对照 | Δ (pp) | 配对 95% CI (pp) | 分数改善 / 恶化 |','|---|---|---:|---|---:|']
for c in paired:lines.append(f"| {c['benchmark']} | {c['comparison'].split(' minus ')[-1]} | {c['delta_pp']:+.1f} | [{c['CI95_low_pp']:+.2f}, {c['CI95_high_pp']:+.2f}] | {c['score_improved']} / {c['score_worsened']} |")
lines+=['','CI：按同一批题的 Image SHA256 簇成对重采样 10,000 次，seed=20261109，百分位 95% CI。样本很小，区间覆盖零不能解释为等价或非劣效。与完整 V²Drop 的差值仅作描述，因为剪枝预算不同。']
for c in paired:
 if c['comparison'].endswith('llava-SVH'):
  if c['benchmark']=='GQA':lines.append(f"GQA 对 SVH：正确→错误 {c['full_credit_to_zero']} 题；错误→正确 {c['zero_to_full_credit']} 题。")
  else:lines.append(f"TextVQA 对 SVH：软分提高 {c['score_improved']} 题、降低 {c['score_worsened']} 题；满分→零分 {c['full_credit_to_zero']} 题、零分→满分 {c['zero_to_full_credit']} 题。另保存任意正分→零分/零分→正分计数，不把软评分强行二值化。")
lines+=['','| Method | Benchmark | 生成/复用条数 | 空输出 | 截断 |','|---|---|---:|---:|---:|']
for r in summary:lines.append(f"| {r['method']} | {r['benchmark']} | {r['completed']} | {r['empty']} | {r['truncated']} |")
if smoke['status']=='PASS':
 max_rmse=max(c['strict_numerical_gate']['hidden_relative_rmse'] for c in smoke['cases']);max_abs=max(c['strict_numerical_gate']['max_abs_logits'] for c in smoke['cases'])
 lines+=['',f"4 图真实 GPU Smoke 全通过：576→288→144→0；L3/L17 实际 Hidden 行删除、原始文本位置保持，RoPE 精确取原始位置；R 在允许层对剩余视觉 Key 的实际 Attention 概率为正；P−1/Decode 视觉访问为零；L2–L5 答案 Q/K/V/O 实际跳算，Norm/MLP/Residual 正常；持久原始视觉 KV 为零。关闭剪枝与冻结 SVH 的 Hidden/Logits 逐位一致。",
  f"数值参考只用于 4 图各 3 Token Smoke，使用物理路径选出的同一视觉索引、完整形状 dense 参考来验证物理压缩；200 题使用真实 Physical 实现。原门禁 atol=0.1、rtol=0.003、Hidden relative RMSE≤0.01、cosine≤1e-4 均未修改；最大 RMSE={max_rmse:.9f}，最大绝对 Logit 差={max_abs:.9f}，短序列输出相同。",
  '官方选择算法：直接调用既有 hf_adapter.py 从固定官方 Commit 的 AST 提取的 L2 norm/top-k 选择器；实际图像起点动态传入，并与未改动固定 35 的原始选择块核对。融合只调用 L3/L17；不执行 L22。融合前层受到 SVH 的影响，因此不要求与独立 V²Drop 选择同一索引。']
if result['failure']:lines+=['','停止原因：'+result['failure']['error']]
lines+=['','逐题原始输出、分数、绑定、选择索引保存在 predictions.sqlite / predictions.jsonl；每题事务与 progress.json 支持恢复，最终 all_methods_predictions.jsonl.gz 和 per_question_comparison.csv 保存四组逐题比较。没有重新生成 Native/V²Drop，也未补跑 SVH 正式样本；只为要求的 Smoke 做短 SVH/数值参考。',
 f"GPU 请求合计 {result['total_GPU_generation_requests']} 次：{len(records)} 次融合自然 EOS 准确率请求，以及 {result['smoke_GPU_generation_requests']} 次三 Token Smoke 请求（每张图各 Physical / Dense 参考 / 关闭剪枝 / 冻结 SVH）；Smoke 不计入 200 题准确率。",
 '没有运行 FLOPs、KV 容量效率、TPOT、延迟测试、额外数据集或层搜索。服务器未关闭。Windows 真实 SHA256 备份结果另见 windows_backup_receipt.json。']
(O/'report_zh.md').write_text('\n'.join(lines)+'\n')
names=['report_zh.md','score_summary.json','accuracy_summary.csv','baseline_audit.json','baseline_audit.csv','baseline_summary.csv','baseline_reuse.jsonl','input_manifest.jsonl.gz','protocol.json','progress.json','physical_smoke.json','cpu_checks.json','fusion_engine.py','run_fusion.py','cpu_checks.py','audit_baselines.py','finalize_fusion.py','predictions.sqlite','predictions.jsonl','all_methods_predictions.jsonl.gz','scorer_source/llava/eval/m4c_evaluator.py']
for name in ['paired_comparisons.csv','per_question_comparison.csv','SVH_changed_score_cases.csv','failure.json']:
 if (O/name).exists():names.append(name)
save('artifact_manifest.json',dict(files={n:dict(bytes=(O/n).stat().st_size,sha256=sha(O/n)) for n in names},weights_datasets_images_excluded=True))
print(json.dumps(dict(status=result['status'],accuracy=summary,paired=paired,GPU=gpu,GPU_processes=processes),ensure_ascii=False))
