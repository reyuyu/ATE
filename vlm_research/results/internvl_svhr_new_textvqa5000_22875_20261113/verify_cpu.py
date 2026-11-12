"""CPU-only portable audit. Optional tqdm fallback affects progress display only."""
from pathlib import Path
import sys,json,gzip,csv,hashlib,struct
try:
 import tqdm
except ModuleNotFoundError:
 # The unchanged official module imports tqdm for its unused bulk-evaluation progress bar.
 # Score and normalization methods do not call tqdm. No scorer logic is replaced.
 import types
 progress_module=types.ModuleType('tqdm');progress_module.tqdm=lambda iterable,*args,**kwargs:iterable;sys.modules['tqdm']=progress_module
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for x in json.loads((P/'integrity_manifest.json').read_text(encoding='utf-8')):
 p=P/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
protocol=json.loads((P/'protocol.json').read_text(encoding='utf-8'));summary=json.loads((P/'summary.json').read_text(encoding='utf-8'))
assert sha(P/'configuration.json')==protocol['configuration_sha256'] and sha(P/'evaluation_config.json')==protocol['evaluation_config_sha256']
cfg=json.loads((P/'configuration.json').read_text(encoding='utf-8'));assert cfg['configuration_name']=='interVL-SVH-R-new' and cfg['reaccess_layers']==list(range(14,28)) and cfg['cutoff']==28
inputs=[json.loads(x) for x in gzip.open(P/'inputs_light.jsonl.gz','rt',encoding='utf-8')];byid={x['question_id']:x for x in inputs};gold={x['question_id']:x['gold'] for x in json.loads((P/'gold.json').read_text(encoding='utf-8'))}
pred=[json.loads(x) for x in gzip.open(P/'predictions_interVL_SVH_R_new.jsonl.gz','rt',encoding='utf-8')];baseline=[json.loads(x) for x in gzip.open(P/'baseline_predictions.jsonl.gz','rt',encoding='utf-8')]
assert len(inputs)==len(gold)==len(pred)==len(byid)==5000 and set(byid)==set(gold)=={x['question_id'] for x in pred}
assert [x['question_id'] for x in inputs]==[x['question_id'] for x in pred] and len({x['image_sha256'] for x in inputs})==3166
sys.path.insert(0,str(P/'execution'));from scoring import Scorer
assert sha(P/'scoring/official_textvqa_eval.py')=='f6fbe6fc4e2d6029d0a6cd792a25246cb537a11f88fa57c293d3f133f57302ef'
s=Scorer([dict(question_id=q,gold=a) for q,a in gold.items()],P/'scoring/official_textvqa_eval.py')
data={summary['configuration_name']:pred,'Native':[],'Physical-D':[]}
for x in baseline:data[x['method']].append(x)
assert len(baseline)==10000 and all(len(xs)==5000 and {x['question_id'] for x in xs}==set(byid) for xs in data.values())
for method,xs in data.items():
 for x in xs:
  q=x['question_id'];r=byid[q];assert x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']
  score=s.score(q,x['text']);assert abs(x['official_score']-score['official_score'])<1e-12 and abs(x['independent_score']-score['independent_score'])<1e-12
  assert x['eos']==(x['token_ids'][-1]==151645) and len(x['token_ids'])<=32
  assert x['empty']==(not x['text']) and x['truncated']==(len(x['token_ids'])==32 and not x['eos'])
  assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256'] and len(r['roles']['image'])==256
  if method==summary['configuration_name']:
   b=x['input_binding_details'];assert b['prompt_sha256']==r['prompt_sha256'] and b['image_sha256']==r['image_sha256']
   assert hashlib.sha256(struct.pack('<'+'q'*len(r['input_ids']),*r['input_ids'])).hexdigest()==b['input_ids_sha256']
scores=list(csv.DictReader((P/'per_question_scores.csv').open(encoding='utf-8-sig')));assert len(scores)==5000 and {x['question_id'] for x in scores}==set(byid)
maps={m:{x['question_id']:x for x in xs} for m,xs in data.items()}
for x in scores:
 assert x['image_sha256']==byid[x['question_id']]['image_sha256']
 for m in data:assert abs(float(x[m])-maps[m][x['question_id']]['official_score'])<1e-12
for row in summary['scores']:
 if row['method']=='Late9':assert row['accuracy_percent']==protocol['late9_published_percent'] and row['source'].startswith('PUBLISHED_AGGREGATE_ONLY@');continue
 accuracy=100*sum(x['official_score'] for x in data[row['method']])/5000;assert abs(accuracy-row['accuracy_percent'])<1e-10
assert abs(100*sum(x['official_score'] for x in pred)/5000-summary['accuracy_percent'])<1e-10
for row in summary['paired']:
 delta=[maps[summary['configuration_name']][q]['official_score']-maps[row['reference']][q]['official_score'] for q in byid]
 assert abs(sum(delta)/50-row['delta_pp'])<1e-10 and sum(x>1e-12 for x in delta)==row['soft_score_improved'] and sum(x< -1e-12 for x in delta)==row['soft_score_declined']
print(json.dumps(dict(status='PASS_CPU',coverage=5000,unique_images=3166,independent_prediction_scoring=15000,accuracy_percent=summary['accuracy_percent'],normalization_checks=s.normalization_checks),ensure_ascii=False))
