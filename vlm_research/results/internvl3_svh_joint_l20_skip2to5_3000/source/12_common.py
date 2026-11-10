from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[2]
OLD=Path('/home/featurize/work/ATE-internvl3-figure3-3000')
P=OLD/'vlm_research/results/internvl3_visual_writing_figure3_3000'
B=OLD/'vlm_research/results/internvl3_visual_writing_l20_l27_3000'
OUT=ROOT/'vlm_research/results/internvl3_svh_joint_l20_skip2to5_3000'
RUN=Path('/home/featurize/ate-runs/internvl3-svh-joint-l20-skip2to5-3000')
A_CP=Path('/home/featurize/ate-runs/internvl3-figure3-3000/checkpoints')
B_CP=Path('/home/featurize/ate-runs/internvl3-figure3-3000/l20_l27_3000/checkpoints')
MODEL=Path('/home/featurize/ate-local-models/InternVL3-8B')
REV='853e3a797a661694b1b8ece0cb72dc2b23e3dac9'
PY='/home/featurize/.venvs/ate-frozen/bin/python'
TASKS=['ChooseAttr','ChooseCat','ChooseRel','CompareAttr','LogicalObj','QueryAttr']
CONDITIONS=['A','B','C','D']
def read(d,n):return json.loads((d/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def save(d,n,v):
 d.mkdir(parents=True,exist_ok=True);p=d/n;t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def command(*a):return subprocess.check_output(a,text=True).strip()
def key(s):return s['task']+'__'+s['question_id']+'.json'
def score(text,gold):return int(text.rstrip('.').lower()==gold.lower())
def old_records(s,ha,hb):
 k=key(s);pa=A_CP/k;pb=B_CP/k
 assert sha(pa)==ha[k] and sha(pb)==hb[k]
 a=read(A_CP,k);b=read(B_CP,k)
 assert a['complete'] and b['complete'] and a['input_binding']==b['input_binding']
 assert a['sample_sha256']==b['sample_sha256']==digest(s)
 assert a['image_sha256']==b['image_sha256']==s['image_sha256']
 assert a['question_id']==b['question_id']==s['question_id'] and a['task']==b['task']==s['task']
 assert a['gold']==b['gold']==s['answer']
 return a,b
