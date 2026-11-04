import os,pathlib,json,hashlib,csv,datetime,subprocess,shutil
from receiver_cross_dataset.common import read,sha,digest,write,csvwrite,tensors_hash,hardware,MODEL
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'results/textvqa_visual_reaccess_budget_exploration'
HIST=ROOT/'results/textvqa_partial_visual_layer_recovery'
CROSS=ROOT/'results/receiver_cross_dataset_validation'
PRIVATE=pathlib.Path('/data/lk/ATE-resources/experiments/textvqa_visual_reaccess_budget_exploration')
BASE='a628c042e1e358d619b595f503811692dc27f126'
UUIDS=['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
KS=[0,2,4,8,12,16,24,32];SEED=20261104;RANDOM_SEEDS=[202611041,202611042,202611043]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def guard():
 assert not (OUT/'STOP.json').exists(),'Saved STOP requires diagnosis'
 assert shutil.disk_usage('/data').free>20*1024**3,'20GiB reserve exhausted'
def layer_design():
 import numpy as np
 order=[14,19]
 while len(order)<32:
  candidates=[l for l in range(32) if l not in order];order.append(min(candidates,key=lambda l:(-min(abs(l-x) for x in order),l)))
 sets={str(k):sorted(order[:k]) if k else [] for k in KS};all32=list(range(32));conditions={}
 for c,p,d in [('A',all32,all32),('B',[],[]),('C',[14,19],[]),('D',[],[14,19]),('E',[14,19],[14,19]),('F',all32,[]),('G',[],all32)]:conditions[c]=dict(Pminus1=p,Decode=d,scope=300,kind='history' if c in 'ABCDE' else 'phase_full')
 for k in [4,8,12,16,24]:conditions['Both-'+str(k)]=dict(Pminus1=sets[str(k)],Decode=sets[str(k)],scope=300,kind='nested_both')
 for k in [4,8,16]:
  conditions['Ponly-'+str(k)]=dict(Pminus1=sets[str(k)],Decode=[],scope=120,kind='phase_split')
  conditions['Donly-'+str(k)]=dict(Pminus1=[],Decode=sets[str(k)],scope=120,kind='phase_split')
 for k in [8,16]:
  for j,seed in enumerate(RANDOM_SEEDS):
   selected=sorted(np.random.default_rng(seed+100*k).choice(32,k,replace=False).tolist());conditions[f'Random-{k}-s{j+1}']=dict(Pminus1=selected,Decode=selected,scope=120,kind='random_both',seed=seed+100*k)
 return dict(rule='Start14/19; repeatedly maximize nearest selected-layer absolute distance, tie lower index',addition_order=order,nested_sets=sets,conditions=conditions,curve_aliases={str(k):'B' if k==0 else 'E' if k==2 else 'A' if k==32 else 'Both-'+str(k) for k in KS},random_seeds=RANDOM_SEEDS)
DESIGN=layer_design();CONDITIONS=list(DESIGN['conditions']);NEW=[c for c in CONDITIONS if c not in 'ABCDE'];STAGE1=[c for c in NEW if DESIGN['conditions'][c]['scope']==300];STAGE2=[c for c in NEW if DESIGN['conditions'][c]['scope']==120]
def sources():
 files=list(pathlib.Path(__file__).parent.glob('*.py'))
 for rel in read(HIST/'source_manifest.json')['files']:files.append(ROOT/rel)
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
def score(text,answers):
 from receiver_cross_dataset.scoring import score as official
 return official('textvqa',text,answers)
def historical_rows():
 by={}
 for r in csv.DictReader((HIST/'per_question.csv').open()):by.setdefault(r['question_id'],{})[r['condition']]=r
 return by
def oldcp(q,c):return read(ROOT/historical_rows()[q][c]['checkpoint'])
def comparison_design():
 def add(rows,a,b):
  if a!=b and not any({x,y}=={a,b} for x,y in rows):rows.append((a,b))
 first=[];second=[];alias=DESIGN['curve_aliases']
 add(first,'A','B')
 for c in ['F','G']+[alias[str(k)] for k in [2,4,8,12,16,24]]:
  for base in ['B','A']:add(first,c,base)
 add(first,'F','G')
 for low,high in zip(KS,KS[1:]):add(first,alias[str(high)],alias[str(low)])
 for k in [2,4,8,16,32]:
  p='C' if k==2 else 'F' if k==32 else 'Ponly-'+str(k);d='D' if k==2 else 'G' if k==32 else 'Donly-'+str(k);both=alias[str(k)]
  for c in [p,d,both]:
   for base in ['B','A']:add(second,c,base)
  for a,b in [(p,d),(both,p),(both,d)]:add(second,a,b)
 for k in [8,16]:
  random=[f'Random-{k}-s{j+1}' for j in range(3)]
  for c in random:
   add(second,alias[str(k)],c)
   for base in ['B','A']:add(second,c,base)
  add(second,alias[str(k)],'RandomMean-'+str(k))
  for j,c in enumerate(random):
   for d in random[j+1:]:add(second,c,d)
 return dict(stage1=first,stage2=second)
