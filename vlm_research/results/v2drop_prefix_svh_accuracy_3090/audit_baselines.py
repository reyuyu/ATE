"""CPU-only archive audit. No model imports, downloads, or prediction changes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import csv,gzip,hashlib,importlib.util,json,shutil,sqlite3,subprocess,sys
from collections import Counter
from pathlib import Path
R=Path('/home/featurize/work/ATE'); B=R/'vlm_research/results'
O=B/'v2drop_prefix_svh_accuracy_3090'; V=B/'v2drop_quick_reproduction'
G=B/'featurize_4090_gqa_official_aligned_threeway'; T=B/'featurize_new3090_c3v2_evaluation'
H=B/'featurize_new3090_recovery/history/cvpr2026-information-horizon-alignment/vlm_research/results/cvpr2026_information_horizon_alignment'
D=Path('/home/featurize/data/ate-recovery')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def digest(x,compact=False):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,**({'separators':(',',':')} if compact else {})).encode()).hexdigest()
def rows(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open()) as f:
  for l in f:yield json.loads(l)
def save(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jl(name,values):
 with (O/name).open('w') as f:
  for v in values:f.write(json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def require(ok,reason):
 if not ok:raise AssertionError(reason)
O.mkdir(exist_ok=True)
require(not (O/'predictions.sqlite').exists(),'Never replace an active fusion result')
p=read(V/'protocol.json'); manifest=list(rows(V/'input_manifest.jsonl.gz')); lookup={s['key']:s for s in manifest}
require(sha(V/'input_manifest.jsonl.gz')==p['manifest_sha256'],'V2 manifest checksum')
require(len(manifest)==len(lookup)==200 and Counter(s['benchmark'] for s in manifest)=={'GQA':100,'TextVQA':100},'Accuracy cohort must be 100+100')
sources={rel:sha(R/rel) for rel in p['frozen_ATE_sources']}
require(sources==p['frozen_ATE_sources'],'Frozen source changed')
artifacts={str(V/'protocol.json'):sha(V/'protocol.json'),str(V/'input_manifest.jsonl.gz'):sha(V/'input_manifest.jsonl.gz')}
for path,expected in [(G/'input_manifest.jsonl.gz',p['source_manifests']['GQA']),(H/'sample_manifest.jsonl.gz',p['source_manifests']['TextVQA'])]:
 require(sha(path)==expected,str(path)+' source manifest changed');artifacts[str(path)]=expected
cohorts={'GQA':list(rows(G/'input_manifest.jsonl.gz')),'TextVQA':[s for s in rows(H/'sample_manifest.jsonl.gz') if s['benchmark']=='TextVQA']}
require({b:len(v) for b,v in cohorts.items()}=={'GQA':12578,'TextVQA':5000},'Full source cohort sizes')
source_rows={}
for bench,cohort in cohorts.items():
 rank=lambda s:hashlib.sha256((p['seed']+'|'+bench+'|'+str(s['question_id'])).encode()).hexdigest()
 selected=sorted(cohort,key=rank)[:100]
 actual=[s for s in manifest if s['benchmark']==bench]
 require([str(s['question_id']) for s in selected]==[s['question_id'] for s in actual],bench+' fixed sample selection')
 for i,(s,a) in enumerate(zip(selected,actual)):
  require(a['selection_rank']==i and a['selection_hash']==rank(s),a['key']+' rank')
  require(all(s[k]==a[k] for k in ['question_id','image_id','image_sha256','question','prompt','answer']),a['key']+' source input')
  require(a['prompt_sha256']==hashlib.sha256(a['prompt'].encode()).hexdigest(),a['key']+' prompt hash')
  require(a['sample_sha256']==digest({k:v for k,v in a.items() if k!='sample_sha256'},True),a['key']+' sample binding')
  require(a['input_ids'].count(32000)==576 and a['input_ids'].index(32000)==a['image_token_start'],a['key']+' image tokens')
  require(Path(a['image_path']).is_file(),a['key']+' restored image path')
  source_rows[a['key']]=s
require(len(p['efficiency_keys'])==30 and set(p['efficiency_keys'])<set(lookup),'30-input efficiency set must not replace 200-input Accuracy cohort')
current={};gold={}
for bench,view in [('GQA','GQA-testdev-balanced'),('TextVQA','TextVQA-OCR')]:
 current[bench]={str(s['question_id']):s for s in rows(D/'normalized'/view/'inputs.jsonl')}
 gold[bench]=read(D/'normalized'/view/'answers.json')
 for s in [s for s in manifest if s['benchmark']==bench]:
  c=current[bench][s['question_id']]
  require(c['question']==s['question'] and c['image_id']==s['image_id'] and gold[bench][s['question_id']]==s['answer'],s['key']+' current Gold/image mapping')
  if bench=='TextVQA':require(c['prompt']==s['prompt'] and c['paper_ocr'] and 'Reference OCR token:' in s['prompt'] and len(s['answer'])==10,s['key']+' Paper-OCR ten-answer protocol')
spec=importlib.util.spec_from_file_location('unchanged_textvqa_scorer',V/'frozen_scoring_sources/m4c_evaluator.py');sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
evaluator=sc.TextVQAAccuracyEvaluator()
def score(bench,text,answer):
 if bench=='GQA':
  parsed=text.rstrip('.').lower();return parsed,float(parsed==answer)
 parsed=evaluator.answer_processor(text);normalized=[evaluator.answer_processor(x) for x in answer]
 # Independent leave-one-annotator-out count, compared with unchanged official scorer.
 matches=sum(a==parsed for a in normalized)
 value=sum(min((matches-int(a==parsed))/3,1) for a in normalized)/10
 official=evaluator._compute_answer_scores(answer).get(parsed,0.0)
 require(abs(value-official)<1e-12,'Independent TextVQA soft-score disagreement')
 return parsed,value
all_records=[];audit=[]
def add(s,method,raw,tokens,parsed,saved_score,empty,truncated,provenance):
 actual_parsed,value=score(s['benchmark'],raw,s['answer'])
 require(actual_parsed==parsed and abs(value-saved_score)<1e-10,s['key']+' '+method+' parsed/score')
 require(bool(empty)==(not bool(raw)),s['key']+' empty flag')
 require(bool(truncated)==(len(tokens)==s['max_new_tokens'] and tokens[-1]!=2),s['key']+' truncation flag')
 require(len(tokens)>0 and len(tokens)<=s['max_new_tokens'],s['key']+' output token count')
 r=dict(key=s['key'],benchmark=s['benchmark'],question_id=s['question_id'],method=method,raw_output=raw,token_ids=tokens,
   parsed=actual_parsed,score=value,gold=s['answer'],empty=empty,truncated=truncated,image_sha256=s['image_sha256'],
   prompt_sha256=s['prompt_sha256'],input_hashes=s['input_hashes'],sample_sha256=s['sample_sha256'],source=provenance)
 all_records.append(r);audit.append(dict(key=s['key'],method=method,saved_score=saved_score,recomputed_score=value,input_binding='PASS',score_audit='PASS'))
pred=list(rows(V/'predictions_light.jsonl'));require(len(pred)==400,'Archived Native/V2 prediction count')
require(len({(r['key'],r['condition']) for r in pred})==400,'Duplicate baseline prediction')
for r in pred:
 s=lookup[r['key']];require(r['condition'] in ('Native','V2Drop'),r['condition'])
 require(r['question_id']==s['question_id'] and r['gold']==s['answer'] and all(r[k]==s[k] for k in ['sample_sha256','image_sha256','prompt_sha256']),r['key']+' Native/V2 record binding')
 add(s,r['condition'],r['raw_output'],r['token_ids'],r['parsed'],r['score'],r['empty'],r['truncated'],dict(artifact=str(V/'predictions_light.jsonl'),condition=r['condition']))
with sqlite3.connect('file:'+str(G/'results.sqlite')+'?mode=ro',uri=True) as db:
 for s in [s for s in manifest if s['benchmark']=='GQA']:
  found=db.execute('SELECT payload,sha256 FROM results WHERE qid=? AND condition=?',(s['question_id'],'C3-V2')).fetchall();require(len(found)==1,s['key']+' SVH archive')
  payload,checksum=found[0];r=json.loads(payload)
  require(checksum in (digest(r),digest(r,True)),s['key']+' stored record checksum')
  require(r['gold']==s['answer'] and r['image_sha256']==s['image_sha256'] and r['input_hashes']==s['input_hashes'] and r['prompt_sha256']==s['prompt_sha256'],s['key']+' SVH same GQA tensors')
  gen=r['generation'];add(s,'llava-SVH',gen['text'],gen['token_ids'],r['parsed'],r['score'],gen['empty'],gen['truncated'],dict(artifact=str(G/'results.sqlite'),condition='C3-V2',record_sha256=checksum))
textpred={r['key']:r for r in rows(T/'predictions.jsonl.gz') if r.get('benchmark')=='TextVQA' and r['key'] in lookup}
textmanifest={r['key']:r for r in rows(T/'mapped_manifest.jsonl.gz') if r.get('benchmark')=='TextVQA' and r['key'] in lookup}
require(len(textpred)==len(textmanifest)==100,'Text SVH cohort coverage')
for s in [s for s in manifest if s['benchmark']=='TextVQA']:
 r=textpred[s['key']];m=textmanifest[s['key']]
 require(r['gold']==s['answer'] and r['question_id']==s['question_id'] and r['image_sha256']==s['image_sha256'] and m['prompt']==s['prompt'] and m['answer']==s['answer'],s['key']+' Text SVH input')
 require(r['prompt_sha256']==digest(s['prompt']) and r['input_ids_sha256']==digest(s['input_ids']),s['key']+' Text SVH identical prompt and token IDs')
 # Historical SVH stored FP32 pixels before its .to(float16); V2 stored actual FP16.
 acceptance=read(B/'featurize_20520_environment_restore/resource_acceptance.json')
 require(acceptance['status']=='PASS' and acceptance['historical_TextVQA_FP32_and_current_FP16_pixel_hashes_verified']==100,'CPU pixel-stage proof missing')
 verified={c['key']:c for c in acceptance['cases']}
 require(verified[s['key']]['pixel_values_fp16_sha256']==s['input_hashes']['pixel_values'],'Restored FP16 pixel-stage proof')
 require(sha(T/'predictions.jsonl.gz')=='efe842a6ec01abf455dce17fec0b5b7ab34c221a560a27add377e8ad3c19a363','Historical FP32-hash proof artifact changed')
 require(m['input_view']=='TextVQA-OCR' and m['paper_ocr'] and m['max_new_tokens']==s['max_new_tokens']==128,s['key']+' Text SVH Paper-OCR/cap')
 gen=r['generation'];add(s,'llava-SVH',gen['text'],gen['token_ids'],r['scoring']['parsed'],r['scoring']['score'],gen['empty'],gen['truncated'],dict(artifact=str(T/'predictions.jsonl.gz'),condition=r['condition'],source_binding=r['binding']))
gp=read(G/'protocol.json');tm=read(T/'run_metadata.json')
require(gp['revision']==p['revision'],'GQA model revision')
require(tm['config']['model_revision']==p['revision'] and tm['config']['dtype']=='float16' and tm['config']['attention']=='eager' and tm['config']['greedy'] and tm['config']['num_beams']==1,'Text SVH model/backend/generation')
require(tm['config']['max_new_tokens']['TextVQA']==128 and tm['config']['original_manifest_sha256']==p['source_manifests']['TextVQA'],'Text SVH input/cap')
# Freeze source metadata in output, retaining exact historical field names.
for x in (V/'protocol.json',G/'protocol.json',T/'run_metadata.json',V/'predictions_light.jsonl',T/'predictions.jsonl.gz',T/'mapped_manifest.jsonl.gz',V/'frozen_scoring_sources/m4c_evaluator.py'):
 artifacts[str(x)]=sha(x)
summary=[]
for method in ['Native','V2Drop','llava-SVH']:
 for bench in ['GQA','TextVQA']:
  rr=[r for r in all_records if r['method']==method and r['benchmark']==bench];require(len(rr)==100,'Exact 100 per baseline')
  summary.append(dict(method=method,benchmark=bench,questions=100,score_sum=sum(r['score'] for r in rr),accuracy_percent=sum(r['score'] for r in rr),empty=sum(r['empty'] for r in rr),truncated=sum(r['truncated'] for r in rr),source='REUSED_ARCHIVE_NO_GPU'))
require([round(r['accuracy_percent'],7) for r in summary[:4]]==[59.0,54.4,61.0,53.4],'Expected archived baseline scores')
shutil.copy2(V/'input_manifest.jsonl.gz',O/'input_manifest.jsonl.gz');jl('baseline_reuse.jsonl',all_records)
for filename,values in [('baseline_summary.csv',summary),('baseline_audit.csv',audit)]:
 with (O/filename).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
save('baseline_audit.json',dict(status='PASS',archived_records_audited=600,selected_inputs=200,baseline_summary=summary,
 selection_rule_verified_against_full_manifests=True,efficiency_subset_not_used=True,TextVQA_independent_leave_one_out=True,
 GQA_official_exact_match=True,all_selected_current_Gold_mappings_verified=True,all_selected_image_paths_present=True,
 tensor_hashes_match_archived_SVH=True,current_images_not_rehashed='Prior restored archive integrity reused; no full image SHA scan',
 model_weights_status='RESTORED_PINNED_REVISION',GPU_generation_started=False,TextVQA_pixel_hash_stage='Historical SVH FP32 and V2 actual FP16 both independently matched by prior CPU restoration acceptance',source_metadata=dict(v2_protocol=p,gqa_protocol=gp,text_run_metadata=tm),artifacts=artifacts,frozen_ATE_sources=sources))
print(json.dumps(dict(status='PASS',inputs=200,archived_records=600,scores=summary),ensure_ascii=False))
