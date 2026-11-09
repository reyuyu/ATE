"""Execute only the author baseline's pure-Python Accuracy aggregation on saved answers."""
from pathlib import Path
import ast,csv,gzip,hashlib,json
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
root=O/'official_sources_extra';manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));entry=next(r for r in manifest if r['path']=='main.py');p=root/'stanfordnlp__mac-network/main.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['SHA256'];tree=ast.parse(p.read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='avg');ctx={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ctx);author_avg=ctx['avg'];results=[]
for fname,benchmark,expected,encoded in [('historical_per_question.csv.gz','HISTORICAL_GQA1000_REPRODUCTION',1000,False),('per_question.csv.gz','GQA',12578,True)]:
 p=O/fname
 if not p.exists():continue
 with gzip.open(p,'rt',encoding='utf-8',newline='') as f:rows=[r for r in csv.DictReader(f) if r['benchmark']==benchmark]
 for c in ['Native','Dense-F','Physical-Original']:
  selected=[r for r in rows if r['condition']==c]
  if not selected:continue
  data=[dict(prediction=r['parsed'],answer=json.loads(r['gold']) if encoded else r['gold']) for r in selected];author=author_avg(data,'acc');stored=sum(float(r['score']) for r in selected)/len(selected);assert abs(author-stored)<1e-12
  results.append(dict(benchmark=benchmark,condition=c,checked=len(selected),expected=expected,complete=len(selected)==expected,author_avg_equals_independent_offline_score=True,observed_accuracy=author))
(O/'gqa_author_crosscheck.json').write_text(json.dumps(dict(status='PASS',source=entry,scope='Author-linked pure Python full-answer exact-match aggregation; not a claim to run the 784MB full auxiliary-metric package',predictions_or_Gold_changed=False,checks=results),indent=2)+'\n',encoding='utf-8');print(json.dumps(results),flush=True)
