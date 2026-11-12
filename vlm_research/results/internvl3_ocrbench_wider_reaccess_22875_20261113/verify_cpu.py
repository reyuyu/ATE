"""Standalone stdlib audit: hashes, coverage, Gold bindings and independent OCRBench v1 scores."""
from pathlib import Path
import json,gzip,hashlib
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for m in json.loads((P/'integrity_manifest.json').read_text(encoding='utf-8')):
 p=P/m['path'];assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'],m['path']
protocol=json.loads((P/'protocol.json').read_text(encoding='utf-8'));summary=json.loads((P/'summary.json').read_text(encoding='utf-8'))
inputs={x['question_id']:x for x in map(json.loads,gzip.open(P/'inputs_light.jsonl.gz','rt',encoding='utf-8'))}
gold={x['question_id']:x['gold'] for x in json.loads((P/'gold.json').read_text(encoding='utf-8'))}
assert len(inputs)==len(gold)==100 and set(inputs)==set(protocol['ids'])
assert sum(x['selected'] for x in protocol['allocation'].values())==100
scores={x['method']:x['correct'] for x in summary['scores']};checked=0
for p in sorted((P/'predictions').glob('*.jsonl.gz')):
 rows=[json.loads(x) for x in gzip.open(p,'rt',encoding='utf-8')];assert len(rows)==100 and len({x['question_id'] for x in rows})==100
 assert {x['question_id'] for x in rows}==set(inputs)
 for x in rows:
  r=inputs[x['question_id']];answers=gold[x['question_id']];answers=answers if isinstance(answers,list) else [answers]
  assert x['gold']==answers and x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']
  assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
  def norm(s):
   s=s.strip().replace('\n',' ');return s.replace(' ','') if x['dataset_name']=='HME100k' else s.lower()
  score=int(any(norm(str(a)) in norm(x['text']) for a in answers))
  assert score==x['official_score']==x['independent_score'];checked+=1
 assert sum(x['official_score'] for x in rows)==scores[p.name.removesuffix('.jsonl.gz')]
assert checked==600
print(json.dumps(dict(status='PASS_CPU',predictions_checked=checked,inputs=100,methods=len(scores),scores=scores),ensure_ascii=False))
