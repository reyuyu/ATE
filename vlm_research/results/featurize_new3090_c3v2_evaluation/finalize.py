"""CPU-only final artifact checks and independent aggregation of saved scores."""
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import ast,hashlib,json,math,sqlite3,subprocess
from pathlib import Path
repo=Path('/home/featurize/work/ATE')
out=repo/'vlm_research/results/featurize_new3090_c3v2_evaluation'
auditdir=repo/'vlm_research/results/featurize_new3090_scoring_audit'
meta=json.loads((out/'run_metadata.json').read_text())
assert meta['status']=='COMPLETE',meta['status']
summary=json.loads((out/'score_summary.json').read_text())
audit=json.loads((auditdir/'audit_summary.json').read_text())
assert audit['overall_status']=='PASS' and audit['records_score_exact']==audit['records_audited']==15
counts=meta['config']['expected_counts'];totals=Counter();values=defaultdict(list)
mme_tasks=defaultdict(list);mme_pairs=defaultdict(list);mmgroups=defaultdict(list)
def digest(r):return hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def save(p,r):
    tmp=p.with_name(p.name+'.finalize.tmp')
    tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def cmd(args):return subprocess.check_output(args,cwd=repo,text=True,timeout=40).strip()
db=sqlite3.connect((out/'predictions.sqlite').as_uri()+'?mode=ro',uri=True)
assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
for key,bench,payload,checksum in db.execute('SELECT key,benchmark,payload,sha256 FROM results'):
    r=json.loads(payload);assert digest(r)==checksum and r['binding']==meta['binding']
    assert r['condition']=='Physical-C3' and r['key']==key and r['benchmark']==bench
    v=r['scoring']['score'];assert 0<=v<=1
    assert bench=='TextVQA' or v in [0,1]
    assert r['generation']['all_effective_logits_finite']
    totals[bench]+=1;values[bench].append(v)
    if bench=='MME':
        task=r['image_id'].split('/')[0];mme_tasks[task].append(v);mme_pairs[(task,r['image_id'])].append(v)
    if bench=='MMBench':mmgroups[int(r['normalized_question_id'])%1000000].append(v)
db.close();assert dict(totals)==counts and sum(totals.values())==22678
assert all(len(g)==2 for g in mme_pairs.values()) and len(mme_pairs)==1187
task_aggregate={}
for task,vs in mme_tasks.items():
    pairs=[p for (t,image),p in mme_pairs.items() if t==task]
    a=100*sum(vs)/len(vs);p=100*sum(all(g) for g in pairs)/len(pairs)
    task_aggregate[task]=dict(accuracy=a,accuracy_plus=p,score=a+p)
for s in summary['benchmarks']['MME']['tasks']:
    for k in ['accuracy','accuracy_plus','score']:assert math.isclose(s[k],task_aggregate[s['task']][k],abs_tol=1e-10)
assert math.isclose(sum(x['score'] for x in task_aggregate.values()),summary['benchmarks']['MME']['scores']['Total'],abs_tol=1e-10)
circular=100*sum(all(vs) for vs in mmgroups.values())/len(mmgroups)
assert math.isclose(circular,summary['benchmarks']['MMBench']['value'],abs_tol=1e-10)
for b in ['ScienceQA-IMG','POPE','TextVQA']:
    assert math.isclose(100*sum(values[b])/len(values[b]),summary['benchmarks'][b]['value'],abs_tol=1e-10)
pope_strategies=summary['benchmarks']['POPE']['strategies']
assert math.isclose(sum(pope_strategies[k]['official_generation_accuracy']*n for k,n in dict(random=2910,popular=3000,adversarial=3000).items())/8910*100,
    summary['benchmarks']['POPE']['value'],abs_tol=1e-10)
for strategy,n in dict(random=2910,popular=3000,adversarial=3000).items():
    ss=pope_strategies[strategy]
    assert sum(ss[k] for k in ['tp','tn','fp','fn'])==n
    assert math.isclose(ss['official_generation_accuracy'],(ss['tp']+ss['tn'])/n,abs_tol=1e-10)
assert cmd(['git','rev-parse','HEAD'])==meta['config']['code_commit']
assert not cmd(['git','status','--porcelain','--untracked-files=no'])
for p,h in meta['config']['source_sha256'].items():assert hashlib.sha256((repo/p).read_bytes()).hexdigest()==h
external=Path('/home/featurize/data/ate-recovery/sources/Information-Horizon/llava/eval/m4c_evaluator.py')
vendor=repo/'vlm_research/benchmarks/_vendor/m4c_evaluator.py'
external_sha=hashlib.sha256(external.read_bytes()).hexdigest()
vendor_sha=hashlib.sha256(vendor.read_bytes()).hexdigest()
assert external_sha=='aa1d450ac2717da6c41dd86cd5fdafb0b1f3fd49fbd7c4bb6482e857f4e479c2'
assert vendor_sha=='8b49e1033e29d551c0bdd86d9149e2b54370e4cd17e52176ac9061b3521f0774'
external_tree=ast.parse(external.read_text())
cls=next(n for n in external_tree.body if isinstance(n,ast.ClassDef) and n.name=='TextVQAAccuracyEvaluator')
cls.body=[n for n in cls.body if not (isinstance(n,ast.FunctionDef) and n.name=='best_answer')]
assert ast.dump(external_tree,include_attributes=False)==ast.dump(ast.parse(vendor.read_text()),include_attributes=False)
author_root=external.parents[2]
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=author_root,text=True).strip()=='75909b2936a13d7214e83514f0bc0cb9cc91139e'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=author_root,text=True).strip()
progress=json.loads((out/'progress.json').read_text());assert progress['completed']==22678
progress['status']='COMPLETE';save(out/'progress.json',progress)
summary.pop('all_saved_scores_independently_recomputed',None)
summary['all_saved_scores_recomputed_with_frozen_scorer']=True
summary['independent_aggregation_check']=dict(status='PASS',records=22678,
    MME_14_tasks_and_1187_pairs=True,MMBench_circular_groups=len(mmgroups),
    method='Independent arithmetic from saved per-question scores; all row payload SHAs verified',
    native_15_prediction_independent_scoring_audit=str(auditdir/'audit_summary.json'))
save(out/'score_summary.json',summary)
resource=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
meta['final_verification']=dict(status='PASS',at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
    all_record_hashes_verified=True,independent_aggregate_check=True,
    frozen_inference_sources_unchanged=True,tracked_worktree_unchanged=True,
    gpu_compute_processes_after=resource,not_pushed=True)
meta['native_scoring_audit_result']=dict(status='COMPLETE',scores_correct='15/15',
    scoring_mismatches=[],input_or_prompt_mapping_issue=False,
    GQA_hash_repair_changes_only_independent_harness=True)
meta['remote_refs_verified']=dict(alignment='e281754ef17b3accc09d6938b5b9f020048b2aec',gqa='57e37ce024b1b2adf9fa24fc24aabec32a1fed08')
meta['integration_notes']=['Preflight Circular completeness follows exact frozen Manifest membership, not an assumption that rotation count equals option count; source data and frozen scoring were unchanged.',
    'Frozen five-benchmark formal inputs use paper prepare pad-to-square; generic recovery processor bridge is not substituted.',
    'Post-run verification helper was corrected to read official_generation_accuracy from the existing POPE summary; frozen scoring, raw predictions and scores were not modified.']
meta['textvqa_scorer_provenance']=dict(status='PASS',actual_path=str(external),actual_sha256=external_sha,
    frozen_vendor_sha256=vendor_sha,author_commit='75909b2936a13d7214e83514f0bc0cb9cc91139e',
    all_module_AST_equal_after_removing_unused_best_answer=True,
    difference='Author source adds an unused best_answer helper; normalization and soft-scoring methods are unchanged.')
save(out/'run_metadata.json',meta)
report=(out/'report_zh.md').read_text()
report+='\n最终 CPU 核对：全部 22,678 条记录 SHA、配置绑定和题数验证通过；独立从已存逐题得分重算 14 项 MME 的 ACC/ACC+、全部 Circular 组以及其余三项均分，与冻结汇总一致。该项是独立聚合核账；另行的 Native 15 条审计独立验证实际答案解析与评分规则。\n\nNative 评分审计：COMPLETE，15/15 原得分正确。TextVQA 34602 的十答案留一软分为 0.9；MME 单对图不能推导全量 Total；MMBench 四轮组正确。此前 GQA 哈希修复只影响独立入口的 SHA 编码，冻结数据、Prompt、推理与评分语义未变。\n\n未发现本轮需要修改的冻结 Scorer、数据映射或 Prompt 问题；已保留 MME 单图版本差异。首次本轮入口的 Circular 组绑定检查已按冻结 Manifest 定义完整成员，未改数据或评分逻辑。\n\n五项正式评测已完成，未启动其他实验或 Push。GPU 退出后计算进程列表：'+(resource or '空')+'。\n'
report+='\nTextVQA 外部评分文件已额外核对：作者源码与冻结 vendor 的差异仅为本轮未调用的 best_answer 辅助方法；移除此方法后的整个模块 AST 一致，答案规范化与十答案软评分实现均未改变。两份文件 SHA 和作者 Commit 已记入 run_metadata.json。\n'
(out/'report_zh.md').write_text(report)
print('FINAL_VERIFIED',json.dumps(dict(status=meta['status'],counts=dict(totals),scores=summary['benchmarks'],
    peak=meta['gpu'],gpu_after=resource,files={p.name:p.stat().st_size for p in out.iterdir() if p.is_file()}),ensure_ascii=False))
