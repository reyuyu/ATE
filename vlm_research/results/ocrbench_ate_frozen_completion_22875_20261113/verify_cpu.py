"""CPU-only audit of published OCRBench records. No models or datasets required."""
import ast,collections,gzip,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
def norm(v,math):
 v=str(v).strip().replace('\n',' ')
 return v.replace(' ','') if math else v.lower()
def run():
 files=json.loads((P/'integrity_manifest.json').read_text())['files']
 for r in files:assert hashlib.sha256((P/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
 inputs=[json.loads(s) for s in (P/'input_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
 assert len(inputs)==1000 and [r['question_id'] for r in inputs]==[f'{i:04d}' for i in range(1000)]
 assert len({r['image_sha256'] for r in inputs})==930
 by={r['question_id']:r for r in inputs};summary=json.loads((P/'score_summary.json').read_text());actual=[]
 tree=ast.parse((P/'official_ocrbench_v1_example.py').read_text(encoding='utf-8'))
 loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and "data[i]['result']" in ast.unparse(n) and 'data_type' in ast.unparse(n) and 'dataset_name' in ast.unparse(n))
 code=compile(ast.Module(body=[loop],type_ignores=[]),'<frozen official OCRBench v1>','exec')
 for f in sorted((P/'predictions').glob('*.jsonl.gz')):
  rows=[json.loads(s) for s in gzip.open(f,'rt',encoding='utf-8')];seen=set();total=0
  for r in rows:
   q=r['question_id'];assert q not in seen;seen.add(q);i=by[q]
   assert all(r[k]==i[k] for k in ['image_sha256','image_id','question','category','dataset_name'])
   answers=i['gold'] if isinstance(i['gold'],list) else [i['gold']];assert r['gold']==answers
   data=[dict(dataset_name=i['dataset_name'],type=i['category'],answers=i['gold'],predict=r['text'])];exec(code,{'data':data});off=int(data[0]['result'])
   independent=int(any(norm(a,i['dataset_name']=='HME100k') in norm(r['text'],i['dataset_name']=='HME100k') for a in answers))
   assert off==independent==r['official_score']==r['independent_score'];total+=off
  if rows:
   s=next(s for s in summary['results'] if (s['model'],s['method'])==(rows[0]['model'],rows[0]['method']))
   assert len(rows)==s['coverage'] and total==s['partial_correct']
   if s['complete']:assert len(rows)==1000 and total==s['score_0_1000']
   actual.append(dict(model=s['model'],method=s['method'],coverage=len(rows),score=s['score_0_1000']))
 print(json.dumps(dict(status='PASS',files_verified=len(files),questions=1000,images=930,results=actual),ensure_ascii=False,indent=2))
if __name__=='__main__':run()
