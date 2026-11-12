"""Standalone stdlib audit: named config hash,1000 coverage and independent official scoring."""
from pathlib import Path
import json,csv,gzip,hashlib
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for x in json.loads((P/'integrity_manifest.json').read_text(encoding='utf-8')):
 p=P/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
protocol=json.loads((P/'protocol.json').read_text(encoding='utf-8'));summary=json.loads((P/'summary.json').read_text(encoding='utf-8'))
C=P.parent.parent/'configs/interVL-SVH-R-new.json';assert sha(C)==protocol['configuration_sha256'];cfg=json.loads(C.read_text(encoding='utf-8'))
assert cfg['configuration_name']=='interVL-SVH-R-new' and cfg['reaccess_layers']==list(range(14,28)) and cfg['cutoff']==28
inputs={x['question_id']:x for x in map(json.loads,gzip.open(P/'inputs_light.jsonl.gz','rt',encoding='utf-8'))};gold={x['question_id']:x['gold'] for x in json.loads((P/'gold.json').read_text(encoding='utf-8'))}
pred=[json.loads(x) for x in gzip.open(P/'predictions_interVL_SVH_R_new.jsonl.gz','rt',encoding='utf-8')];ids={f'{i:04d}' for i in range(1000)}
assert len(inputs)==len(gold)==len(pred)==1000 and set(inputs)==set(gold)=={x['question_id'] for x in pred}==ids
assert len({x['image_sha256'] for x in pred})==930
for x in pred:
 r=inputs[x['question_id']];ans=gold[x['question_id']];ans=ans if isinstance(ans,list) else [ans]
 assert x['configuration_name']==cfg['configuration_name'] and x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']
 assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256'] and len(r['roles']['image'])==256
 def norm(s):
  s=s.strip().replace('\n',' ');return s.replace(' ','') if x['dataset_name']=='HME100k' else s.lower()
 assert int(any(norm(str(a)) in norm(x['text']) for a in ans))==x['official_score']==x['independent_score']
 assert x['eos']==(x['token_ids'][-1]==151645) and len(x['token_ids'])<=128
 assert x['empty']==(not x['text']) and x['truncated']==(len(x['token_ids'])==128 and not x['eos'])
assert sum(x['origin']=='INHERITED_LATE14_100' for x in pred)==100 and sum(x['origin']=='NEW_MISSING900' for x in pred)==900
scores=list(csv.DictReader((P/'per_question_scores.csv').open(encoding='utf-8-sig')));assert len(scores)==1000 and {x['question_id'] for x in scores}==ids
for x in scores:assert x['image_sha256']==inputs[x['question_id']]['image_sha256']
for s in summary['scores']:assert sum(int(x[s['method']]) for x in scores)==s['score_0_1000']
assert sum(x['official_score'] for x in pred)==summary['score_0_1000']==sum(int(x[cfg['configuration_name']]) for x in scores)
print(json.dumps(dict(status='PASS_CPU',coverage=1000,unique_images=930,independent_new_condition_scoring=1000,baseline_score_rows_checked=3000,score_0_1000=summary['score_0_1000'],new=900,reused=100),ensure_ascii=False))
