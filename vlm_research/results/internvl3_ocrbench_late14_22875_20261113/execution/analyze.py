"""CPU-only audit of one extra Late14 condition; reuse previous six frozen baselines."""
import sys,json,gzip,csv,shutil,subprocess,hashlib,collections
from pathlib import Path
O=Path('/home/featurize/ate-experiments/internvl3-ocrbench-late14-22875-20261113')
OLD=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
sys.path.insert(0,str(OLD));from run_common import sha,atomic,score
import numpy as np
from transformers import AutoTokenizer
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
P=json.loads((O/'protocol.json').read_text());pb=sha(O/'protocol.json');rows=json.loads((O/'selected_rows.json').read_text());byid={r['question_id']:r for r in rows};ids=P['ids'];method='Late14_L14_27'
assert json.loads((O/'progress.json').read_text())['status']=='COMPLETE_PENDING_CPU_AUDIT'
assert P['configs']=={method:list(range(14,28))}
pub=O/'publication/vlm_research/results/internvl3_ocrbench_late14_22875_20261113';pub.mkdir(parents=True,exist_ok=True)
for p,h in P['source_SHA256'].items():assert sha(OLD/'late9_sources'/p)==h,p
baseline=json.loads((O/'baselines.json').read_text());assert sha(O/'baselines.json')==P['baselines_sha256']
tok=AutoTokenizer.from_pretrained('/home/featurize/ate-local-models/InternVL3-8B',trust_remote_code=True,local_files_only=True,use_fast=False)
new=[];files=sorted((O/'predictions'/method).glob('*.json'));assert len(files)==100
for f in files:
 x=json.loads(f.read_text());q=x['question_id'];r=byid[q];g=x['generation'];assert x['protocol_sha256']==pb
 assert g['reaccess_layers']==list(range(14,28)) and g['cutoff']==28 and g['independent_cache']
 assert [a['layer'] for a in g['kv_layout'] if a['visual_kv']]==list(range(14,28))
 text=tok.decode(g['token_ids'],skip_special_tokens=True).split('<|im_end|>')[0].strip();assert text==g['text']
 assert len(g['token_ids'])==g['generated_tokens']<=128 and g['eos']==(g['token_ids'][-1]==151645)
 assert g['truncated']==(len(g['token_ids'])==128 and not g['eos']) and g['empty']==(not text)
 b=x['input_binding_details'];assert x['image_sha256']==r['image_sha256']==b['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']==b['prompt_sha256']
 s=score(q,text);assert s['official_score']==x['official_score']==x['independent_score']
 new.append(dict(method=method,question_id=q,category=r['category'],dataset_name=r['dataset_name'],image_sha256=r['image_sha256'],prompt_sha256=r['prompt_sha256'],input_binding=x['input_binding'],text=text,token_ids=g['token_ids'],official_score=s['official_score'],independent_score=s['independent_score'],eos=g['eos'],truncated=g['truncated'],empty=g['empty']))
assert {x['question_id'] for x in new}==set(ids)
for name,rs in baseline.items():
 assert len(rs)==100 and {x['question_id'] for x in rs}==set(ids)
 for x in rs:
  r=byid[x['question_id']];assert x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']
  assert x['gold']==(r['gold'] if isinstance(r['gold'],list) else [r['gold']])
  assert score(x['question_id'],x['text'])['official_score']==x['official_score']==x['independent_score']
data=dict(baseline);data[method]=new;imap={m:{r['question_id']:r for r in rs} for m,rs in data.items()}
order=['Native','Physical_D','Late9',method,'Late16_L12_27','Late20_L8_27','All24_ExceptL2_5']
table=[dict(method=m,n=100,correct=sum(x['official_score'] for x in data[m]),empty=sum(x['empty'] for x in data[m]),truncated=sum(x['truncated'] for x in data[m]),new_generations=100 if m==method else 0) for m in order]
def csvout(name,rs):
 with (pub/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
csvout('scores.csv',table)
csvout('per_question_scores.csv',[dict(question_id=q,image_sha256=byid[q]['image_sha256'],**{m:imap[m][q]['official_score'] for m in order}) for q in ids])
category=[]
for m in order:
 for c in sorted(P['allocation']):
  xs=[x for x in data[m] if x['category']==c];category.append(dict(method=m,category=c,n=len(xs),correct=sum(x['official_score'] for x in xs)))
csvout('category_scores.csv',category)
clusters=sorted({r['image_sha256'] for r in rows});cq={s:[r['question_id'] for r in rows if r['image_sha256']==s] for s in clusters}
sizes=np.array([len(cq[s]) for s in clusters]);boot=np.random.default_rng(20261113).integers(0,len(clusters),size=(10000,len(clusters)))
paired=[];cases=[]
for ref in ['Native','Late9','Late16_L12_27','Late20_L8_27','All24_ExceptL2_5']:
 diffs={q:imap[method][q]['official_score']-imap[ref][q]['official_score'] for q in ids}
 sums=np.array([sum(diffs[q] for q in cq[s]) for s in clusters]);ci=np.quantile(100*sums[boot].sum(1)/sizes[boot].sum(1),[.025,.975])
 paired.append(dict(method=method,reference=ref,delta_pp=sum(diffs.values()),wrong_to_right=sum(v==1 for v in diffs.values()),right_to_wrong=sum(v==-1 for v in diffs.values()),changed_text=sum(imap[method][q]['text']!=imap[ref][q]['text'] for q in ids),ci95_low_pp=float(ci[0]),ci95_high_pp=float(ci[1]),image_clusters=len(clusters),bootstrap_replicates=10000,seed=20261113,multiplicity='unadjusted sequential exploratory'))
 for q in ids:
  if diffs[q]:cases.append(dict(reference=ref,question_id=q,category=byid[q]['category'],question=byid[q]['question'],gold=json.dumps(byid[q]['gold'],ensure_ascii=False),reference_output=imap[ref][q]['text'],Late14_output=imap[method][q]['text'],score_change=diffs[q]))
csvout('paired_statistics.csv',paired);csvout('changed_score_cases.csv',cases)
with gzip.GzipFile(str(pub/'predictions_Late14.jsonl.gz'),'wb',mtime=0) as f:
 for x in new:f.write((json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n').encode())
with gzip.GzipFile(str(pub/'inputs_light.jsonl.gz'),'wb',mtime=0) as f:
 for r in rows:
  d={k:r[k] for k in ['question_id','question','image_id','image_sha256','prompt','prompt_sha256','input_ids','roles','category','dataset_name','max_new_tokens']};f.write((json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n').encode())
for f in ['protocol.json','gold.json','expected_bindings.json','cpu_audit.json','smoke.json','environment.json']:
 shutil.copy2(O/f,pub/f)
shutil.copytree(O/'numerical',pub/'numerical',dirs_exist_ok=True)
(pub/'execution').mkdir(exist_ok=True)
for f in ['prepare.py','run.py','analyze.py']:shutil.copy2(O/f,pub/'execution'/f)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.free','--format=csv,noheader'],text=True).strip()
summary=dict(status='COMPLETE_EXPLORATORY_QUALITY',scores=table,paired=paired,new_quality_generations=100,reused_predictions=600,new_dense_reference_requests=1,cpu_token_decode_checks=100,independent_scoring_checks=700,source_files_verified=len(P['source_SHA256']),GPU_after=gpu,previous_result_commit=P['baseline_commit'],scope='Same frozen OCRBench100; sequential exploratory follow-up, not independent validation or full1000 result. ATE single448/256visual, NOT official dynamic24.')
atomic(pub/'summary.json',summary)
n=next(x['correct'] for x in table if x['method']==method)
report=['# InternVL3-8B Late14：同一OCRBench100追加验证','',f'状态：**COMPLETE_EXPLORATORY_QUALITY**。Late14明确为0-based L14–L27，共14层回看。本轮仅新增该条件100次生成，得分 **{n}/100**；其他六组600条逐题输出全部复用。','',
 '| 方法 | 回看层（0-based） | 得分 /100 | 空输出 | 截断 |','|---|---|---:|---:|---:|']
windows={'Native':'原生完整','Physical_D':'无回看','Late9':'L19–L27',method:'L14–L27','Late16_L12_27':'L12–L27','Late20_L8_27':'L8–L27','All24_ExceptL2_5':'L0–L1及L6–L27'}
for x in table:report.append(f"| {x['method']} | {windows[x['method']]} | {x['correct']} | {x['empty']} | {x['truncated']} |")
report+=['','## 配对比较','','| 对照 | Late14减对照（pp） | 错→对 | 对→错 | 图片簇Bootstrap 95% CI（pp） |','|---|---:|---:|---:|---|']
for x in paired:report.append(f"| {x['reference']} | {x['delta_pp']:+d} | {x['wrong_to_right']} | {x['right_to_wrong']} | [{x['ci95_low_pp']:.2f}, {x['ci95_high_pp']:.2f}] |")
report+=['', '配对Bootstrap按98个图片SHA簇重采样10000次，seed=20261113，未做多重比较校正。Late14是看到此前100题结果后的追加窗口，因此本次仍是同批样本上的探索，不能当作独立验证、全量OCRBench成绩或统计非劣效。差异不显著也不等于方法等价。',
 '', '## 输入与实现不变', '', f"完整复用[上一轮提交](https://github.com/reyuyu/ATE/commit/{P['baseline_commit']})的100题ID、顺序、图片SHA、Gold、Prompt、input_ids、BF16 pixel_values、选择seed和10类比例；未重新挑选样本。只把P−1与Decode允许视觉回看的层集合设为L14–L27。R读图边界仍为20、答案Attention跳算仍为L2–L5、cutoff仍为28。原始终端Physical核心59个来源文件SHA256逐项核验，未修改冻结源码。",
 '', '模型OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager、Torch2.6.0+cu124、Transformers4.49.0、batch1、greedy、max_new_tokens128、EOS151645；ATE_SVH_1TILE_SCORER_ALIGNED，单448×448图块/256视觉Token，原Chat Template与短答后缀不变。模型只加载一次，模型/数据/环境/运行文件均在本地盘。',
 '', '## Smoke、数值与评分', '', '4题Smoke为0000、0560、0743、0926，全部通过，并计入100题。真实视觉输入、允许层视觉Attention概率>0、禁止层及L20后R视觉Attention=0、L2–L5答案Attention/QKV跳算、Norm/MLP、RoPE原始位置、独立KV和K集合选择性持久视觉KV均通过，视觉Hidden在L27后/Final Norm前删除。',
 '', '首题0000只生成2Token（包含EOS），另做1次同前缀Dense参考；两步logits与共享Hidden逐元素一致。原阈值未改；该极短参考不能消除历史全局数值等价失败。本轮保留EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER，不声称全局等价。OOM、NaN及执行异常为0。',
 '', '700条新旧预测均在CPU使用冻结OCRBench v1官方评分循环与独立包含匹配核算；普通数据大小写不敏感，HME100k区分大小写并去除空格/换行，全部一致。新增100条Token IDs离线解码与保存原文一致；没有丢弃错误、空输出或截断。旧600条不重复上传完整内容，其来源文件哈希见protocol.json、配对分数见per_question_scores.csv，新增逐题原文见predictions_Late14.jsonl.gz。',
 '', '终端cutoff28意味着全部28个语言层Prefill仍处理视觉Hidden；非保留层使用瞬时零scratch保持Attention归约宽度，仅对持久视觉KV作物理选择性保留。本轮只验证质量，不测试也不声称FLOPs/KV/延迟收益。',
 '', f'真实GPU及释放后状态：{gpu}；无遗留计算进程。不关闭服务器，不增加窗口、样本或Benchmark。',
 '', '完整原始记录、逐题审计和日志会保存在独立归档，Windows和持续盘的真实SHA256核验见backup_receipt.json；轻量结果在独立Git分支，main和旧结果不修改。']
(pub/'report_zh.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
pr=json.loads((O/'progress.json').read_text());pr.update(status='COMPLETE_EXPLORATORY_QUALITY',cpu_token_decode_checks=100,independent_scoring_checks=700);atomic(O/'progress.json',pr);shutil.copy2(O/'progress.json',pub/'progress.json')
atomic(O/'analysis_done.json',dict(status='PASS',new_quality=100,reused=600,publication=str(pub)))
print(json.dumps(summary,ensure_ascii=False,indent=2))
