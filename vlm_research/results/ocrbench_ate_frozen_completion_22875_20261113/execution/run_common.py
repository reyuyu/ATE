import ast,copy,csv,gzip,hashlib,json,os,subprocess,time
from pathlib import Path
L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
D=Path('/home/featurize/ate-local-data/bundles/OCRBench-v1')
A=Path('/home/featurize/ate-experiments/internvl3-ocrbench-native-official-22875-20261113/audit')
RAW=json.loads((D/'OCRBench.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def score(qid,text):
 r=RAW[int(qid)];tree=ast.parse((A/'official/Yuliang-Liu__MultimodalOCR/OCRBench/example.py').read_text())
 loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and "data[i]['result']" in ast.unparse(n) and 'data_type' in ast.unparse(n) and 'dataset_name' in ast.unparse(n))
 data=[dict(copy.deepcopy(r),predict=text)];exec(compile(ast.Module(body=[loop],type_ignores=[]),'<official OCRBench v1>','exec'),{'data':data})
 def norm(s):
  s=s.strip().replace('\n',' ');return s.replace(' ','') if r['dataset_name']=='HME100k' else s.lower()
 answers=r['answers'] if isinstance(r['answers'],list) else [r['answers']]
 independent=int(any(norm(str(a)) in norm(text) for a in answers));assert independent==data[0]['result']
 return dict(official_score=int(data[0]['result']),independent_score=independent)
def gpu_idle():
 p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip();assert not p,p
 return subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.free','--format=csv,noheader'],text=True).strip()
def load_records(folder):
 folder.mkdir(parents=True,exist_ok=True)
 return {p.stem:json.loads(p.read_text()) for p in folder.glob('*.json')}
