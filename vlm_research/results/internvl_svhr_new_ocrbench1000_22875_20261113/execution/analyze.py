"""CPU-only full coverage audit: exact original1000, reused100, missing900 only."""
import sys,json,gzip,csv,shutil,subprocess,hashlib,collections
from pathlib import Path
O=Path('/home/featurize/ate-experiments/internvl-svh-r-new-ocrbench1000-22875-20261113');L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
sys.path.insert(0,str(L));from run_common import sha,atomic,score,RAW,A
import numpy as np
from transformers import AutoTokenizer
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'Wait for the only GPU process to exit'
assert json.loads((O/'progress.json').read_text())['status']=='COMPLETE_GENERATION_PENDING_CPU_AUDIT'
cfg=json.loads((O/'configuration.json').read_text());P=json.loads((O/'protocol.json').read_text());pb=sha(O/'protocol.json');assert sha(O/'configuration.json')==P['configuration_sha256']
rows=json.loads((O/'frozen_rows.json').read_text());byid={r['question_id']:r for r in rows};ids=[r['question_id'] for r in rows];assert ids==[f'{i:04d}' for i in range(1000)]
expected={r['question_id']:r for r in json.loads((O/'expected_bindings.json').read_text())['bindings']}
inherited=json.loads((O/'inherited_late14_100.json').read_text());new={p.stem:json.loads(p.read_text()) for p in (O/'predictions').glob('*.json')}
assert len(inherited)==100 and len(new)==900 and not(set(inherited)&set(new)) and set(inherited)|set(new)==set(ids)
for q,info in P['inherited_source_SHA256'].items():assert sha(info['path'])==info['sha256']
for p,h in P['source_SHA256'].items():assert sha(L/'late9_sources'/p)==h,p
tok=AutoTokenizer.from_pretrained('/home/featurize/ate-local-models/InternVL3-8B',trust_remote_code=True,local_files_only=True,use_fast=False)
pred=[];audits=[]
for q in ids:
 x=inherited[q] if q in inherited else new[q];r=byid[q];raw=RAW[int(q)];g=x['generation'];b=x['input_binding_details']
 assert r['question']==raw['question'] and r['gold']==raw['answers'] and r['official_id']==raw['id'] and r['category']==raw['type']
 assert g['reaccess_layers']==list(range(14,28)) and g['cutoff']==28 and g['independent_cache']
 assert [a['layer'] for a in g['kv_layout'] if a['visual_kv']]==list(range(14,28))
 for k in ['pixels_bf16_sha256','input_ids_sha256','prompt_sha256','image_sha256']:assert b[k]==expected[q][k],(q,k)
 if q in new:assert x['protocol_sha256']==pb and x['configuration_name']==cfg['configuration_name']
 text=tok.decode(g['token_ids'],skip_special_tokens=True).split('<|im_end|>')[0].strip();assert text==g['text']
 assert len(g['token_ids'])==g['generated_tokens']<=128 and g['eos']==(g['token_ids'][-1]==151645)
 assert g['truncated']==(len(g['token_ids'])==128 and not g['eos']) and g['empty']==(not text)
 s=score(q,text);assert s['official_score']==x['official_score']==x['independent_score']
 pred.append(dict(configuration_name=cfg['configuration_name'],method=cfg['configuration_name'],question_id=q,official_id=r['official_id'],category=r['category'],dataset_name=r['dataset_name'],image_sha256=r['image_sha256'],prompt_sha256=r['prompt_sha256'],input_binding_sha256=x['input_binding'],input_binding_details=b,text=text,token_ids=g['token_ids'],official_score=s['official_score'],independent_score=s['independent_score'],eos=g['eos'],truncated=g['truncated'],empty=g['empty'],origin='INHERITED_LATE14_100' if q in inherited else 'NEW_MISSING900',source_commit=P['inherited_commit'] if q in inherited else None,source_record=P['inherited_source_SHA256'][q]['path'] if q in inherited else str(O/'predictions'/(q+'.json'))))
 if q in P['smoke_ids']:audits.append(dict(question_id=q,audit=g['audit'],kv_layout=g['kv_layout']))
baseline={};baseline_hash={};B=L/'publication/vlm_research/results/ocrbench_ate_frozen_completion_22875_20261113/predictions'
for m in ['Native','Physical_D','Late9']:
 f=B/f'InternVL3_8B__{m}.jsonl.gz';rs=[json.loads(s) for s in gzip.open(f,'rt')];assert len(rs)==1000 and {r['question_id'] for r in rs}==set(ids)
 for x in rs:
  r=byid[x['question_id']];assert x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']
  assert x['gold']==(r['gold'] if isinstance(r['gold'],list) else [r['gold']])
  assert score(x['question_id'],x['text'])['official_score']==x['official_score']==x['independent_score']
 baseline[m]=rs;baseline_hash[m]=dict(path=str(f),sha256=sha(f),commit=P['baseline_commit'])
data=dict(baseline);data[cfg['configuration_name']]=pred;imap={m:{x['question_id']:x for x in rs} for m,rs in data.items()}
table=[dict(method=m,n=1000,score_0_1000=sum(x['official_score'] for x in rs),accuracy_percent=sum(x['official_score'] for x in rs)/10,empty=sum(x['empty'] for x in rs),truncated=sum(x['truncated'] for x in rs),new_generations=900 if m==cfg['configuration_name'] else 0,reused=100 if m==cfg['configuration_name'] else 1000) for m,rs in data.items()]
pub=O/'publication/vlm_research/results/internvl_svhr_new_ocrbench1000_22875_20261113';pub.mkdir(parents=True,exist_ok=True)
def csvout(name,rs):
 with (pub/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
csvout('scores.csv',table);csvout('per_question_scores.csv',[dict(question_id=q,image_sha256=byid[q]['image_sha256'],**{m:imap[m][q]['official_score'] for m in data}) for q in ids])
categories=sorted({r['category'] for r in rows});catrows=[]
for m,rs in data.items():
 for c in categories:
  xs=[x for x in rs if x['category']==c];catrows.append(dict(method=m,category=c,n=len(xs),correct=sum(x['official_score'] for x in xs)))
csvout('category_scores.csv',catrows)
clusters=sorted({r['image_sha256'] for r in rows});assert len(clusters)==930;cq={s:[r['question_id'] for r in rows if r['image_sha256']==s] for s in clusters};sizes=np.array([len(cq[s]) for s in clusters]);boots=np.random.default_rng(20261113).integers(0,len(clusters),size=(10000,len(clusters)))
paired=[];cases=[];name=cfg['configuration_name']
for ref in ['Native','Late9']:
 diffs={q:imap[name][q]['official_score']-imap[ref][q]['official_score'] for q in ids};sums=np.array([sum(diffs[q] for q in cq[s]) for s in clusters]);ci=np.quantile(100*sums[boots].sum(1)/sizes[boots].sum(1),[.025,.975])
 paired.append(dict(method=name,reference=ref,n=1000,delta_points=sum(diffs.values()),delta_pp=sum(diffs.values())/10,wrong_to_right=sum(v==1 for v in diffs.values()),right_to_wrong=sum(v==-1 for v in diffs.values()),changed_text=sum(imap[name][q]['text']!=imap[ref][q]['text'] for q in ids),ci95_low_pp=float(ci[0]),ci95_high_pp=float(ci[1]),image_clusters=930,bootstrap_replicates=10000,seed=20261113,multiplicity='unadjusted exploratory; configuration selected using100 overlapping questions'))
 for q in ids:
  if diffs[q]:cases.append(dict(reference=ref,question_id=q,category=byid[q]['category'],question=byid[q]['question'],gold=json.dumps(byid[q]['gold'],ensure_ascii=False),reference_output=imap[ref][q]['text'],new_output=imap[name][q]['text'],score_change=diffs[q]))
csvout('paired_statistics.csv',paired);csvout('changed_score_cases.csv',cases)
def gzout(path,rs):
 with gzip.GzipFile(str(pub/path),'wb',mtime=0) as f:
  for x in rs:f.write((json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n').encode())
gzout('predictions_interVL_SVH_R_new.jsonl.gz',pred);gzout('inputs_light.jsonl.gz',[{k:r[k] for k in ['question_id','official_id','question','image_id','image_sha256','prompt','prompt_sha256','input_ids','roles','category','dataset_name','max_new_tokens']} for r in rows]);gzout('smoke_physical_audits.jsonl.gz',audits)
atomic(pub/'gold.json',[dict(question_id=r['question_id'],gold=r['gold']) for r in rows]);atomic(pub/'baseline_sources.json',baseline_hash)
for f in ['protocol.json','cpu_audit.json','smoke.json','environment.json','prereg_github_verification.json']:shutil.copy2(O/f,pub/f)
(pub/'execution').mkdir(exist_ok=True)
for f in ['prepare.py','run.py','analyze.py']:shutil.copy2(O/f,pub/'execution'/f)
shutil.copy2(L/'run_common.py',pub/'execution/run_common.py')
(pub/'scoring').mkdir(exist_ok=True);scorer=A/'official/Yuliang-Liu__MultimodalOCR/OCRBench/example.py';shutil.copy2(scorer,pub/'scoring/official_OCRBench_v1_example.py')
atomic(pub/'scorer_sources.json',dict(official_source_path=str(scorer),official_source_sha256=sha(scorer),helper_source_path=str(L/'run_common.py'),helper_sha256=sha(L/'run_common.py'),execution='Select and compile the unchanged official result-scoring loop by AST, then independently normalize/match each answer; both must agree'))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.free','--format=csv,noheader'],text=True).strip();newscore=sum(x['official_score'] for x in pred)
summary=dict(status='COMPLETE_FULL_COVERAGE_WITH_NUMERICAL_WAIVER',configuration_name=name,coverage=1000,expected=1000,unique_images=930,new_quality_generations=900,inherited_predictions=100,score_0_1000=newscore,accuracy_percent=newscore/10,scores=table,paired=paired,independent_scoring_checks=4000,cpu_token_decode_checks=1000,empty=sum(x['empty'] for x in pred),truncated=sum(x['truncated'] for x in pred),GPU_after=gpu,numerical_status=P['numerical_status'],official_dynamic24_protocol=False,preregistration_commit=json.loads((O/'prereg_commit_receipt.json').read_text())['commit'])
atomic(pub/'summary.json',summary)
report=[f'# {name}：OCRBench v1 全1000题','',f'状态：**COMPLETE_FULL_COVERAGE_WITH_NUMERICAL_WAIVER**。实际完成1000/1000题、930张唯一原图，得分 **{newscore}/1000（{newscore/10:.1f}%）**。继承上一轮Late14的100条原始预测，仅新增缺失900条；没有重跑Native、Physical-D、Late9或其他条件。','',
 '## 命名与配置','','正式配置名按用户指定保留为`interVL-SVH-R-new`。模型为InternVL3-8B，Late14=L14–L27（0-based，共14层）允许P−1与真实Decode回看原始视觉KV；R仍在L0–L19读图，L2–L5答案Attention/QKV物理跳算，视觉Hidden在L27后/Final Norm前退出，选择性持久KV与独立Cache不变。命名不改变算法，也不代表全局数值等价门禁已经通过。',
 '',f"[运行前命名与协议提交](https://github.com/reyuyu/ATE/commit/{summary['preregistration_commit']})；配置文件为`vlm_research/configs/interVL-SVH-R-new.json`，SHA256={sha(O/'configuration.json')}。",'',
 '## 全集正式分数（原ATE单图块协议）','','| 方法 | 覆盖 | 得分 /1000 | Accuracy | 空输出 | 截断 |','|---|---:|---:|---:|---:|---:|']
for x in table:report.append(f"| {x['method']} | {x['n']} | {x['score_0_1000']} | {x['accuracy_percent']:.1f}% | {x['empty']} | {x['truncated']} |")
report+=['','对照均为同一模型Revision、1000题、930张原图、单448图块/256视觉Token、Prompt、生成上限与评分器；只复用已有逐题结果，不使用官方动态24图块新协议，不把不同协议公开论文分数当作直接对照。','','## 配对差异','','| 对照 | 新方法减对照（分） | 差值pp | 错→对 | 对→错 | 图片SHA簇Bootstrap 95% CI（pp） |','|---|---:|---:|---:|---:|---|']
for x in paired:report.append(f"| {x['reference']} | {x['delta_points']:+d} | {x['delta_pp']:+.1f} | {x['wrong_to_right']} | {x['right_to_wrong']} | [{x['ci95_low_pp']:.2f}, {x['ci95_high_pp']:.2f}] |")
report+=['','930个图片SHA簇、10000次配对Bootstrap、seed=20261113，未做多重比较校正。Late14是依据此前100题探索选择的配置，且该100题包含在本次1000题中；因此全量覆盖并不等于独立留出验证，也不能由未显著差异推断统计等价或非劣效。逐题改善与恶化均保留。','','## 输入、评分与Smoke核验','','原1000题Question ID、顺序、Question、Gold、Image ID及类别全部与原始OCRBench注释一致。930张图片沿用此前可信SHA清单，不重复全盘哈希；生成时逐题核对实际图片、Prompt、input_ids和BF16 pixel_values绑定。继承100条的原文件SHA及旧协议都逐项复核，未重新生成。Gold与模型输入独立保存，Gold未拼入Prompt。',
 '', '4道新题0001、0500、0700、0900的真实GPU结构Smoke通过，并计入新增900条。允许回看层视觉Attention概率>0，禁止层及L20后R的视觉Attention=0；物理Attention跳算、Norm/MLP、RoPE原始位置、独立Cache、持久视觉KV层集合以及Hidden退出均通过。无OOM、NaN或结构异常。此前Late14首题2步Dense短参考一致的记录仅作为局部证据，本轮不重复生成该参考；原严格阈值与历史失败、EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER继续保留，不能宣称全局数值等价。',
 '', '所有1000条原文和Token IDs保存，CPU重新解码一致；包括三个历史对照在内的4000条预测，冻结OCRBench v1官方评分循环与独立包含匹配实现逐项一致。普通类别小写化、首尾空白处理及换行转空格；HME100k区分大小写并去除空格/换行。错误、空输出、截断均未删除或用0分伪造补齐。',
 '', '## 环境与边界','','模型OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9；BF16/Eager、PyTorch2.6.0+cu124、Transformers4.49.0、batch1、greedy、max_new_tokens128、EOS151645、原Chat Template及短答后缀。实际模型仅加载一次；数据、模型、环境和运行目录均在本地盘。冻结59个来源文件SHA在运行前后核验，核心未修改。',
 '', '该终端实现的全部28层Prefill仍处理视觉Hidden，非保留层用瞬时零scratch保持原Attention归约宽度，持久原始视觉KV则物理选择性保留。未进行KV/FLOPs/TPOT/延迟实验，不把回看层减少直接当作真实效率改善。',
 '',f'实际GPU及释放后状态：{gpu}。无残留计算进程，不关闭服务器，不启动其他实验。',
 '', '独立分支同步配置、全集轻量预测、输入哈希、评分、对照统计及外围执行源码；完整原始记录、逐题结构与日志单独归档，真实Windows和持续盘SHA256备份回执见backup_receipt.json。']
(pub/'report_zh.md').write_text('\n'.join(report)+'\n',encoding='utf-8');(pub/'README_zh.md').write_text(f'# {name}\n\n当前状态：COMPLETE_FULL_COVERAGE_WITH_NUMERICAL_WAIVER。OCRBench v1覆盖1000/1000，得分{newscore}/1000；继承100、新增900。\n\n完整证据见[中文报告](report_zh.md)、[汇总](summary.json)、[运行前协议](protocol.json)及[命名配置](../../configs/interVL-SVH-R-new.json)。历史数值等价豁免保留；不是官方动态24图块协议。\n',encoding='utf-8')
pr=json.loads((O/'progress.json').read_text());pr.update(status=summary['status'],independent_scoring_checks=4000,cpu_token_decode_checks=1000);atomic(O/'progress.json',pr);shutil.copy2(O/'progress.json',pub/'progress.json');atomic(O/'analysis_done.json',dict(status='PASS',score_0_1000=newscore,coverage=1000,publication=str(pub)))
print(json.dumps(summary,ensure_ascii=False,indent=2))
