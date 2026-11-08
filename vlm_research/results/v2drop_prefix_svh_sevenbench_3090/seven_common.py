"""Isolated seven-benchmark orchestration, unchanged frozen scorers."""
from pathlib import Path
import gzip,hashlib,json,sys,re,os
R=Path('/home/featurize/work/ATE');B=R/'vlm_research/results';O=B/'v2drop_prefix_svh_sevenbench_3090';F=B/'v2drop_prefix_svh_accuracy_3090';FM=B/'v2drop_prefix_svh_mme_full_3090';V=B/'v2drop_quick_reproduction';M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
sys.path[:0]=[str(O),str(F),str(R)]
from vlm_research.information_horizon_alignment import score as scorer
from vlm_research.gqa_testdev_c3.common import prediction
scorer.AUTHOR=F/'scorer_source';scorer.text_evaluator.cache_clear()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def old_digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def gz(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def save(n,x):
 p=O/n;t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def score(s,text):
 b=s['benchmark']
 if b=='GQA':
  parsed=prediction(text);assert parsed==text.rstrip('.').lower()
  return dict(parsed=parsed,score=float(parsed==s['answer']))
 if b=='OCRBench-v1':
  answers=s['answer'] if isinstance(s['answer'],list) else [s['answer']]
  def normalize(x):return x.strip().replace('\n',' ').replace(' ','') if s['dataset_name']=='HME100k' else x.lower().strip().replace('\n',' ')
  p=normalize(text);return dict(score=float(any(normalize(a) in p for a in answers)),parsed=p,protocol='official-v1-substring')
 value=scorer.score(s,text)
 if b=='TextVQA':
  e=scorer.text_evaluator();p=e.answer_processor(text);answers=[e.answer_processor(x) for x in s['answer']];m=answers.count(p)
  assert len(answers)==10 and abs(value['score']-sum(min((m-int(a==p))/3,1) for a in answers)/10)<1e-12
 elif b=='POPE':
  p='no' if set(text.split('.')[0].replace(',','').split(' ')).intersection({'No','not','no'}) else 'yes'
  assert value['parsed']==p and value['score']==float(p==s['answer'])
 elif b=='ScienceQA-IMG':
  hits=re.findall(r'The answer is ([A-Z]).',text)
  p=text if len(text)==1 and text in 'ABCDE' else text[0] if len(text)>=3 and text[0] in 'ABCDE' and text[1:3]=='. ' else hits[0] if len(hits)==1 else 'FAILED'
  ix='ABCDE'.index(p) if p in s['options'] else -1
  assert value['parsed']==ix and value['score']==float(ix==s['answer'])
 elif b=='MME':
  low=text.lower();p=low if low in ['yes','no'] else 'yes' if 'yes' in low[:4] else 'no' if 'no' in low[:4] else 'other'
  assert value['parsed']==p and value['score']==float(p==s['answer'])
 return value
