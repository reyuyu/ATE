"""Stdlib-only integrity and independent OCRBench v1 scoring of the added100."""
from pathlib import Path
import json,csv,gzip,hashlib
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for m in json.loads((P/'integrity_manifest.json').read_text(encoding='utf-8')):
 p=P/m['path'];assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'],m['path']
protocol=json.loads((P/'protocol.json').read_text(encoding='utf-8'));summary=json.loads((P/'summary.json').read_text(encoding='utf-8'))
assert protocol['configs']=={'Late14_L14_27':list(range(14,28))}
inputs={x['question_id']:x for x in map(json.loads,gzip.open(P/'inputs_light.jsonl.gz','rt',encoding='utf-8'))}
gold={x['question_id']:x['gold'] for x in json.loads((P/'gold.json').read_text(encoding='utf-8'))}
pred=[json.loads(x) for x in gzip.open(P/'predictions_Late14.jsonl.gz','rt',encoding='utf-8')]
assert len(inputs)==len(gold)==len(pred)==100 and set(inputs)==set(protocol['ids'])=={x['question_id'] for x in pred}
for x in pred:
 r=inputs[x['question_id']];answers=gold[x['question_id']];answers=answers if isinstance(answers,list) else [answers]
 assert x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']
 assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
 def norm(s):
  s=s.strip().replace('\n',' ');return s.replace(' ','') if x['dataset_name']=='HME100k' else s.lower()
 assert int(any(norm(str(a)) in norm(x['text']) for a in answers))==x['official_score']==x['independent_score']
 assert x['eos']==(x['token_ids'][-1]==151645) and len(x['token_ids'])<=128
scores=list(csv.DictReader((P/'per_question_scores.csv').open(encoding='utf-8-sig')))
assert len(scores)==100 and {x['question_id'] for x in scores}==set(inputs)
for x in scores:assert x['image_sha256']==inputs[x['question_id']]['image_sha256']
for s in summary['scores']:assert sum(int(x[s['method']]) for x in scores)==s['correct']
assert sum(x['official_score'] for x in pred)==sum(int(x['Late14_L14_27']) for x in scores)
print(json.dumps(dict(status='PASS_CPU',new_predictions_independently_scored=100,prior_score_rows_checked=600,inputs=100,scores={x['method']:x['correct'] for x in summary['scores']}),ensure_ascii=False))
