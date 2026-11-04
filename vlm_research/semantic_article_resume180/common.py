import csv,datetime,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
import numpy as np
import torch
from query_role_attention.common import MODEL,REVISION,TASKS
from gqa3000_token_visual_writing.common import atomic_json,atomic_npz
from paper_rq1.scorers import gqa_prediction

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'results/semantic_article_resume180'
P=PREP/'run_20261104'
ROUND2=Path('/data1/likai/ATE-worktrees/receiver-search-continuation/vlm_research/results/receiver_search_round2')
OLDP=Path('/data1/likai/ATE-worktrees/gqa5h-adaptive-token-receiver-search/vlm_research/results/gqa5h_receiver_search')
OLDPRIVATE=Path('/data1/likai/ATE-resources/experiments/gqa5h_receiver_search')
BASE=Path('/data/lk/ATE-worktrees/gqa360-structural-receiver-selection/vlm_research')
HIST=BASE/'results/gqa3000_token_visual_writing'
PRIVATE=Path('/data1/likai/ATE-resources/experiments/semantic_article_resume180_run')
HISTORY_COMMIT='01c82b1d39360fc73bca3d10d12886672ffee370'
SMOKE_IDS=['00150746','11697551','11127723']
BASELINES=['All-Writer','Question-Only','Semantic-first-Q80','Semantic-k','Question-Last-Q80','Question-Random-20261103-Q80','Question-Random-20261107-Q80','Semantic-first-Q60']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def writecsv(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def event(kind,**kw):
    with (P/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,utc=utc(),monotonic=time.monotonic(),**kw),ensure_ascii=False)+'\n')
def remaining(gpu=False):return read(P/'clock.json')['gpu_stop_monotonic' if gpu else 'deadline_monotonic']-time.monotonic()
def guard(gpu=False,reserve=0):
    assert not (P/'STOP.json').exists(),read(P/'STOP.json')
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==read(P/'clock.json')['boot_id']
    assert remaining(gpu)>reserve,'Time budget exhausted'
def authorized(q):
    splits=read(P/'split_manifest.json');s=next((name for name,ids in splits['ids'].items() if q in ids),None)
    assert s is not None or q in SMOKE_IDS
    return s
def sample(q):
    authorized(q);return read(OLDPRIVATE/'samples'/f'{q}.json')
def gold(q):
    authorized(q);return read(OLDPRIVATE/'gold'/f'{q}.json')['answer']
def spec(q):return read(OLDP/'input_specs'/f'{q}.json')
def canonical(positions):return digest(positions)[:24]
def result_path(q,positions):return P/'checkpoints'/q/(canonical(positions)+'.json')
def old_result_path(q,positions):return OLDP/'checkpoints'/q/(canonical(positions)+'.json')
def source_hashes():
    files=list(Path(__file__).parent.glob('*.py'))
    # Every shared inference dependency is explicitly bound by the historical source map.
    from gqa360_structural_receiver_selection.run import source_hashes as inherited
    return {**{str(BASE/n):h for n,h in inherited().items()},**{str(f):sha(f) for f in files if f.name in ['common.py','worker.py','selection.py','prepare.py','tests.py','fit.py']}}
def verify_sources():
    p=read(P/'protocol.json')
    for f,h in p['shared_sources'].items():assert sha(f)==h,f
    assert sha(P/'split_manifest.json')==p['split_sha256']
    return p

class History:
    def __init__(self):
        self.shards=read(HIST/'shards.json')['image_assignments'];self.inputs=read(HIST/'input_hashes.json');self.cache={};self.files={}
        self.git=subprocess.Popen(['git','cat-file','--batch'],cwd=BASE.parent,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    def committed(self,path):
        name=str(Path(path).relative_to(BASE.parent));self.git.stdin.write((HISTORY_COMMIT+':'+name+'\n').encode());self.git.stdin.flush();head=self.git.stdout.readline().decode().strip().split();assert len(head)==3,head
        blob=self.git.stdout.read(int(head[2]));assert self.git.stdout.read(1)==b'\n' and blob==Path(path).read_bytes(),name;self.files[name]=sha(path)
    def get(self,q,label,positions=None):
        authorized(q);key=(q,label)
        if key in self.cache:return self.cache[key]
        s=sample(q);sp=spec(q);actual='Semantic-first-Q100' if label=='Question-Only' else label
        assert actual in sp['historical_writers'];w=sp['historical_writers'][actual]
        if positions is not None:assert positions==w
        uid=sp['historical_aliases'][actual];path=HIST/f'shard{self.shards[s["image_sha256"]]}/checkpoints'/(s['task']+'__'+q)/(uid+'.json');self.committed(path);r=read(path);g=r['generation']
        assert r['writers']==g['writer_positions']==w and r['input_hashes']==sp['input_hashes']==self.inputs[s['task']+'__'+q]
        assert g['logits_to_keep']==1 and g['own_cache'] and g['own_prefix'] and g['audit']['passed'] and g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==gold(q))
        assert r['sample_sha256']==sp['original_sample_sha256']
        # The immutable raw record is hash-audited once per distinct checkpoint.
        if str(path) not in self.files or 'raw:'+str(path) not in self.files:
            assert sha(r['raw']['path'])==r['raw']['sha256'];self.files['raw:'+str(path)]=r['raw']['sha256']
        out=dict(question_id=q,condition=label,writers=w,generation=g,reused=True,source=dict(path=str(path),sha256=sha(path),commit=HISTORY_COMMIT,raw_sha256=r['raw']['sha256']),input_hashes=r['input_hashes']);self.cache[key]=out;return out
    def close(self):self.git.stdin.close();self.git.wait()

def score_record(q,condition,writers,history=None):
    authorized(q);sp=spec(q)
    for n,w in sp['historical_writers'].items():
        if w==writers:
            h=history or History();r=h.get(q,n,writers)
            if history is None:h.close()
            return dict(r,condition=condition)
    for oldbase,commit in [(ROUND2,'1df71f42ffa318e38021e26b9ad71027e2e3cf88'),(OLDP,'a1c9d81db89f59142a28295f2f1eeaa2b93e8fe6')]:
        old=oldbase/'checkpoints'/q/(canonical(writers)+'.json')
        if old.exists():
            r=read(old);assert r['writers']==writers and r['input_hashes']==sp['input_hashes'] and sha(r['raw']['path'])==r['raw']['sha256'];g=dict(r['generation']);g['correct']=int(g['prediction']==gold(q));return dict(r,generation=g,condition=condition,reused=True,source=dict(path=str(old),sha256=sha(old),commit=commit))
    r=read(result_path(q,writers));assert r['writers']==writers and r['input_hashes']==sp['input_hashes'];g=dict(r['generation']);g['correct']=int(g['prediction']==gold(q));return dict(r,generation=g,condition=condition,reused=False)
