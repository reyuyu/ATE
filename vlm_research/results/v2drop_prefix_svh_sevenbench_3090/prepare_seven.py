"""CPU freeze; reuse 2,574 verified predictions; never changes source results."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from pathlib import Path
from collections import Counter,defaultdict
import gzip,hashlib,json,sqlite3,sys,ast
R=Path('/home/featurize/work/ATE');B=R/'vlm_research/results';O=B/'v2drop_prefix_svh_sevenbench_3090';F=B/'v2drop_prefix_svh_accuracy_3090';FM=B/'v2drop_prefix_svh_mme_full_3090';T=B/'featurize_new3090_c3v2_evaluation';G=B/'featurize_4090_gqa_official_aligned_threeway';S=B/'featurize_new3090_c3v2_reaccess_sevenbench';V=B/'v2drop_quick_reproduction';D=Path('/home/featurize/data/ate-recovery/normalized')
sys.path.insert(0,str(R))
from vlm_research.information_horizon_alignment import score as scorer
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def gz(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def writegz(name,rows):
 with gzip.GzipFile(filename=str(O/name),mode='wb',mtime=0) as f:
  for r in rows:f.write((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode())
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert O.is_dir() and not (O/'protocol.json').exists() and not (O/'input_manifest.jsonl.gz').exists()
fp=read(F/'protocol.json');assert read(F/'physical_smoke.json')['status']=='PASS' and read(FM/'score_summary.json')['status']=='COMPLETE'
for n,h in fp['frozen_sources'].items():assert sha(R/n)==h,n
for n,h in fp['sources'].items():assert sha(F/n)==h,n
assert sha(G/'input_manifest.jsonl.gz')=='0ddd9d4112e7fe7ce9cd77b8ce48b603581c0abf6a3feefca4de4446587caeb8'
assert sha(T/'mapped_manifest.jsonl.gz')=='f62d6906de26c4556ef35815a55ac47e25230cae6697ced8f3896d7113b93741'
cfg=read(T/'run_metadata.json')['config']
input_receipts={}
for b,v in cfg['inputs'].items():
 p=D/v['view']/'inputs.jsonl';q=D/v['view']/'answers.json'
 assert sha(p)==v['inputs_sha256'] and sha(q)==v['answers_sha256'],b
 input_receipts[b]=dict(inputs=str(p),answers=str(q),inputs_sha256=sha(p),answers_sha256=sha(q))
samples=[dict(s,benchmark='GQA',key='GQA:'+str(s['question_id']),max_new_tokens=32,input_view='GQA-official-aligned') for s in gz(G/'input_manifest.jsonl.gz')]
paper=gz(T/'mapped_manifest.jsonl.gz');samples+=paper
ocr=[s for s in gz(S/'mapped_manifest.jsonl.gz') if s['benchmark']=='OCRBench-v1'];samples+=ocr
current_ocr={s['key']:s for s in [json.loads(l) for l in (D/'OCRBench-v1/inputs.jsonl').read_text().splitlines()]}
for s in ocr:
 c=current_ocr[s['key']]
 assert all(s[k]==c[k] for k in ['answer','question','prompt','image_sha256','image_path','official_annotation_index','dataset_name','category'])
assert sha(S/'ocr_official_LLaVA1_5.py')=='c480b271cf71e18848aa637c9f6e5d5f8b22fdcd3b969cb63167b4fbeb450a68'
gqa_current={str(s['question_id']):s for s in [json.loads(l) for l in (D/'GQA-testdev-balanced/inputs.jsonl').read_text().splitlines()]};gqa_gold=read(D/'GQA-testdev-balanced/answers.json')
for s in samples[:12578]:
 c=gqa_current[s['question_id']]
 assert s['answer']==gqa_gold[s['question_id']] and all(s[k]==c[k] for k in ['question_id','question','image_id','image_path'])
 assert len([x for x in s['input_ids'] if x==32000])==576 and s['split']=='testdev_balanced'
expected={'GQA':12578,'ScienceQA-IMG':2017,'TextVQA':5000,'POPE':8910,'MME':2374,'MMBench':4377,'OCRBench-v1':1000}
assert dict(Counter(s['benchmark'] for s in samples))==expected
by={s['key']:s for s in samples};assert len(samples)==len(by)==36256
assert all(Path(s['image_path']).is_file() for s in samples)
caps={'GQA':32,'ScienceQA-IMG':1024,'TextVQA':128,'POPE':128,'MME':128,'MMBench':1024,'OCRBench-v1':128}
assert all(s['max_new_tokens']==caps[s['benchmark']] for s in samples)
assert all(s.get('paper_ocr') and len(s['answer'])==10 for s in samples if s['benchmark']=='TextVQA')
pope=dict(Counter(s['strategy'] for s in samples if s['benchmark']=='POPE'));assert pope=={'random':2910,'popular':3000,'adversarial':3000}
groups=defaultdict(list);pairs=defaultdict(list)
for s in samples:
 if s['benchmark']=='MMBench':groups[s['circular_group']].append(s)
 if s['benchmark']=='MME':pairs[s['pair_id']].append(s)
assert len(groups)==1176 and sum(map(len,groups.values()))==4377
assert len(pairs)==1187 and all(len(v)==2 for v in pairs.values())
baseline_inputs={s['key']:s for s in gz(F/'input_manifest.jsonl.gz')};reuse=[]
for directory,count in [(F,200),(FM,2374)]:
 protocol=read(directory/'protocol.json')
 with sqlite3.connect('file:'+str(directory/'predictions.sqlite')+'?mode=ro',uri=True) as db:
  assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
  records=[]
  for key,payload,h in db.execute('SELECT key,payload,sha256 FROM predictions'):
   r=json.loads(payload);s=by[key];assert digest(r)==h and r['binding']==digest(protocol) and r['gold']==s['answer'] and r['image_sha256']==s['image_sha256']
   assert r['condition']=='V2Drop-prefix+SVH' and r['generation']['all_effective_logits_finite']
   if directory==F:
    original=baseline_inputs[key]
    assert all(original[k]==s[k] for k in ['question_id','prompt','answer','image_sha256','max_new_tokens'])
    if s['benchmark']=='GQA':assert original['input_ids']==s['input_ids'] and original['input_hashes']==s['input_hashes']
   else:assert r['sample_sha256']==digest(s) and r['same_input_binding_as_SVH']
   assert [e['layer'] for e in r['pruning_events']]==[3,17] and [e['visual_after'] for e in r['pruning_events']]==[288,144]
   records.append(dict(key=key,source_directory=str(directory),source_record_sha256=h,source_protocol_sha256=sha(directory/'protocol.json'),record=r))
  assert len(records)==count;reuse+=records
assert len(reuse)==2574 and len({r['key'] for r in reuse})==2574
scorer.AUTHOR=F/'scorer_source';scorer.text_evaluator.cache_clear();scorer.text_evaluator();scorer.matcher()
for r in reuse:
 s=by[r['key']];text=r['record']['generation']['text']
 if s['benchmark']=='GQA':assert r['record']['scoring']['score']==float(text.rstrip('.').lower()==s['answer'])
 else:assert scorer.score(s,text)==r['record']['scoring']
# Freeze the exact old per-input bindings (FP32 pixel digest for Paper inputs).
refs=[r for r in gz(T/'predictions.jsonl.gz')]+[r for r in gz(S/'predictions.jsonl.gz') if r['benchmark']=='OCRBench-v1']
writegz('reference_bindings.jsonl.gz',[{k:r[k] for k in ['key','gold','prompt_sha256','input_ids_sha256','pixel_values_sha256','expanded_length']} for r in refs])
writegz('input_manifest.jsonl.gz',samples);writegz('reused_predictions.jsonl.gz',reuse)
vp=read(V/'protocol.json');assert len(vp['efficiency_keys'])==30 and all(k in baseline_inputs and k in by for k in vp['efficiency_keys'])
smoke=[]
for b in ['ScienceQA-IMG','POPE','MMBench','OCRBench-v1']:
 first=next(s for s in samples if s['benchmark']==b)
 if b=='MMBench':smoke += [s['key'] for s in groups[first['circular_group']]]
 elif b=='POPE':smoke += [next(s['key'] for s in samples if s['benchmark']==b and s['strategy']==v) for v in ['adversarial','popular','random']]
 else:smoke.append(first['key'])
audit=dict(status='PASS',expected=expected,total=36256,reused=2574,new_required=33682,POPE_strategies=pope,MME_pairs=1187,MMBench_Circular_groups=1176,
 all_image_paths_present=True,no_full_image_rehash=True,inputs=input_receipts,source_GQA_manifest_sha256=sha(G/'input_manifest.jsonl.gz'),source_paper_manifest_sha256=sha(T/'mapped_manifest.jsonl.gz'),
 source_OCR_manifest_sha256=sha(S/'mapped_manifest.jsonl.gz'),OCR_current_inputs_sha256=sha(D/'OCRBench-v1/inputs.jsonl'),OCR_official_source_sha256=sha(S/'ocr_official_LLaVA1_5.py'),
 inherited_strict_physical_gate=dict(status='PASS',path=str(F/'physical_smoke.json'),sha256=sha(F/'physical_smoke.json'),same_GPU=True,unchanged_threshold=True),
 frozen_sources=fp['frozen_sources'],fusion_sources={n:sha(F/n) for n in ['fusion_engine.py','run_fusion.py']},model=fp['model'],revision=fp['revision'],code_commit=fp['code_commit'],
 max_new_tokens=caps,integration_smoke_keys=smoke,efficiency_keys=vp['efficiency_keys'],efficiency_lengths=[32,128],efficiency_primary_length=128,
 known_MME_image_difference=read(FM/'input_audit.json')['known_MME_image_difference'])
save('input_audit.json',audit)
print('SEVEN_CPU_FREEZE_PASS',json.dumps({k:audit[k] for k in ['total','expected','reused','new_required','integration_smoke_keys']},ensure_ascii=False))
