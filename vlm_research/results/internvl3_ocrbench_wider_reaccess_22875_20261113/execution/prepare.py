"""Freeze a prediction-blind proportional OCRBench100 and reuse complete baselines."""
import sys, collections, gzip, json, hashlib
from pathlib import Path
OLD=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
O=Path('/home/featurize/ate-experiments/internvl3-ocrbench-wider-reaccess-22875-20261113')
sys.path.insert(0,str(OLD))
from run_common import sha, digest, atomic, score, RAW, D
R=OLD/'results/internvl3'; rows=json.loads((R/'frozen_rows.json').read_text())
assert len(rows)==len(RAW)==1000
assert sha(D/'OCRBench.json')=='df75957e75ea052e9f38f2c67a4f6676103998517345cf49ef7d03b62c835864'
assert not (O/'protocol.json').exists(), 'Do not overwrite a frozen protocol'
seed='internvl3-ocrbench100-wider-reaccess-20261113-v1'
groups=collections.defaultdict(list)
for r in rows:
 q=int(r['question_id']); raw=RAW[q]
 assert r['question']==raw['question'] and r['official_id']==raw['id'] and r['category']==raw['type']
 assert r['gold']==raw['answers'], ('GOLD_MISMATCH',q)
 groups[r['category']].append(r)
def rank(r):return hashlib.sha256((seed+'|'+r['question_id']+'|'+r['image_sha256']).encode()).hexdigest()
chosen=[]; allocation={}
for category,rs in sorted(groups.items()):
 assert len(rs)%10==0
 n=len(rs)//10; allocation[category]=dict(population=len(rs),selected=n)
 chosen.extend(sorted(rs,key=rank)[:n])
chosen.sort(key=lambda r:int(r['question_id'])); assert len(chosen)==100
ids=[r['question_id'] for r in chosen]; byid={r['question_id']:r for r in chosen}
paths=json.loads((R/'input_paths.json').read_text());verified={r['question_id']:r for r in json.loads((R/'inputs_verified.json').read_text())['bindings']}
for r in chosen:
 assert sha(paths[r['image_sha256']])==r['image_sha256']
 assert len(r['roles']['image'])==256 and len(r['input_ids'])==r['input_tokens']
 assert r['prompt_sha256']==hashlib.sha256(r['prompt'].encode()).hexdigest()
v=[verified[q] for q in ids]
old_protocol=json.loads((R/'protocol.json').read_text())
for p,h in old_protocol['source_SHA256'].items(): assert sha(OLD/'late9_sources'/p)==h, p
all24=json.loads((O/'history_all24_source_manifest.json').read_text())['runtime_source_sha256']
core='vlm_research/internvl3_svh_r_terminal_windows_textvqa/engine.py'
assert sha(OLD/'late9_sources'/core)==all24[core]
baseline={}; source_hashes={}
P=OLD/'publication/vlm_research/results/ocrbench_ate_frozen_completion_22875_20261113/predictions'
for method in ['Native','Physical_D','Late9']:
 p=P/f'InternVL3_8B__{method}.jsonl.gz';source_hashes[method]=dict(path=str(p),sha256=sha(p))
 full=[json.loads(line) for line in gzip.open(p,'rt')];assert len(full)==1000 and len({x['question_id'] for x in full})==1000
 baseline[method]=[x for x in full if x['question_id'] in byid]; assert len(baseline[method])==100
 for x in baseline[method]:
  r=byid[x['question_id']]
  gold=r['gold'] if isinstance(r['gold'],list) else [r['gold']]
  assert x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256'] and x['gold']==gold
  assert x['official_score']==x['independent_score']==score(x['question_id'],x['text'])['official_score']
smoke_categories=['Regular Text Recognition','Doc-oriented VQA','Key Information Extraction','Handwritten Mathematical Expression Recognition']
smoke_ids=[min((r for r in chosen if r['category']==c),key=rank)['question_id'] for c in smoke_categories]
configs={'Late16_L12_27':list(range(12,28)), 'Late20_L8_27':list(range(8,28)), 'All24_ExceptL2_5':[0,1]+list(range(6,28))}
protocol=dict(status='FIXED_EXPLORATORY_QUALITY_ONLY',model='OpenGVLab/InternVL3-8B',revision='853e3a797a661694b1b8ece0cb72dc2b23e3dac9',
 baseline_commit='a4fb52e3aa4d57385bd1108832c58a28471ecea3',selection_seed=seed,selection_rule='Per official type sort SHA256(seed|zero-padded question_id|image_sha256), take exactly population/10; restore original question order. No predictions or gold used in ranking.',
 allocation=allocation,ids=ids,smoke_ids=smoke_ids,configs=configs,cutoff=28,R_image_permission_boundary=20,skip_answer_attention=[2,3,4,5],
 dtype='bfloat16',attention='eager',torch='2.6.0+cu124',transformers='4.49.0',batch=1,greedy=True,max_new_tokens=128,eos_token_id=151645,
 input_protocol='ATE_SVH_1TILE_SCORER_ALIGNED',image_shape=[1,3,448,448],visual_tokens=256,prompt_suffix='Give only the requested text or answer. Do not explain.',
 scorer='Frozen official OCRBench v1; case-insensitive answer substring except HME100k case-sensitive with whitespace removal; independent implementation required.',
 source_SHA256=old_protocol['source_SHA256'],baselines=source_hashes,expected_new_quality_requests=300,baseline_regeneration=0,
 numerical_status='Historical EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER preserved. A short same-prefix Dense reference on the first smoke question per config reports unchanged strict thresholds; this does not erase historical failures or establish global equivalence.',
 numerical_thresholds=dict(logit_max_abs=.125,logit_mean_abs=.01,shared_hidden_relative_RMS=.015625,top1_equal=True),
 numerical_reference_max_steps=8,hard_stop='OOM, nonfinite, input/binding mismatch, wrong layers, cache/RoPE/position/skip/exit errors or hook leak; preserve partial predictions.',
 physical_scope='Unchanged terminal-cutoff28 engine; physical selective persistent visual KV. All 28 blocks still process visual Hidden in prefill; removal is afterL27 before final norm. Transient masked zero scratch preserves reduction width. No efficiency claims.',
 audit_history='All24 TextVQA128=54.765625%, Late9=50.703125%; these are Image-only TextVQA results, not OCRBench. Broader early-only R14 did not recover quality.',
 resources='All model/data/env/output on local disk; persistent disk only light evidence/backup. No core modifications, downloads, Native reruns, full benchmark or timing.')
atomic(O/'selected_rows.json',chosen);atomic(O/'gold.json',[dict(question_id=r['question_id'],gold=r['gold']) for r in chosen])
atomic(O/'expected_bindings.json',v);atomic(O/'input_paths.json',{r['image_sha256']:paths[r['image_sha256']] for r in chosen});atomic(O/'baselines.json',baseline)
protocol['selected_rows_sha256']=sha(O/'selected_rows.json');protocol['baselines_sha256']=sha(O/'baselines.json')
atomic(O/'protocol.json',protocol)
atomic(O/'cpu_audit.json',dict(status='PASS',selected=100,unique_images=len({r['image_sha256'] for r in chosen}),source_files_verified=len(protocol['source_SHA256']),model_weights_rehashed=False,baseline_predictions_reused=300,annotation_sha256=sha(D/'OCRBench.json'),allocation=allocation,gold_check='Historical normalized predictions store singleton answers as one-element lists; compared to scalar frozen Gold as a list without changing any accepted answer or scorer. Initial preparation stopped before GPU on that serialization assertion.'))
print('CPU_PREPARE_PASS',json.dumps(dict(smoke_ids=smoke_ids,baselines={k:sum(x['official_score'] for x in v) for k,v in baseline.items()},selected=100),ensure_ascii=False),flush=True)
