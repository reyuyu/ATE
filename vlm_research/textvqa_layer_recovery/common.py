import os,pathlib,hashlib,json,csv,datetime,subprocess,shutil
from receiver_cross_dataset.common import read,sha,digest,write,csvwrite,tensors_hash,hardware,MODEL,UUID
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'results/textvqa_partial_visual_layer_recovery'
OLD=ROOT/'results/receiver_cross_dataset_validation'
PRIVATE=pathlib.Path('/data/lk/ATE-resources/experiments/textvqa_partial_visual_layer_recovery')
BASE='b483cdc524ce568fdcff5a871b1608b4aeecb322'
CONDITIONS=['A','B','C','D','E']
NAMES=dict(A='Native Full',B='All-Writer',C='Pminus1 Keep14+19',D='Decode Keep14+19',E='Both Keep14+19')
LAYERS=[14,19]
SEED=20261104
COMPARISONS=[('A','B'),('C','B'),('D','B'),('E','B'),('E','C'),('E','D'),('E','A')]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def guard():
 assert not (OUT/'STOP.json').exists(),'Saved STOP record requires diagnosis'
 assert shutil.disk_usage('/data').free>20*1024**3,'Disk reserve below20GiB'
def sources():
 paths=list(pathlib.Path(__file__).parent.glob('*.py'))
 for rel in read(OLD/'source_manifest.json')['files']:paths.append(ROOT/rel)
 for s in ['gqa3000_token_visual_writing/engine.py','semantic_visual_writers/engine.py','query_role_attention/engine.py','query_role_smoke_fix/run.py','query_role_formal/common.py','vlm_common.py','knockout_probe.py','paper_rq1/engine.py','benchmarks/scoring.py','benchmarks/_vendor/m4c_evaluator.py','receiver_cross_dataset/common.py','receiver_cross_dataset/scoring.py','receiver_cross_dataset/analyze.py','receiver_cross_dataset/prepare.py']:
  paths.append(ROOT/s)
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
def baseline_rows():
 return {r['question_id']:r for r in csv.DictReader((OLD/'textvqa/per_question.csv').open()) if r['condition']=='All-Writer'}
def baseline(q):
 row=baseline_rows()[q];return read(OLD/row['checkpoint'])
def score(text,answers):
 from receiver_cross_dataset.scoring import score as official
 return official('textvqa',text,answers)
