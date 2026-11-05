"""Frozen 2x2 Decode-only visual access; Prefill including P-1 stays native."""
import pathlib,time,shutil,json
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'results/textvqa_decode_l0_l1_factorial'
INPUTS=OUT/'input_specs'
SEED=20261105
UUIDS=['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
CONDITIONS={'Native':list(range(32)),'Decode-None':[], 'A':list(range(11,21)), 'B':[0]+list(range(11,21)), 'C':[1]+list(range(11,21)), 'D':[0,1]+list(range(11,21))}
def query_rows(layout,c,layer,past=0,noop=False):
 return [0] if past and not noop and layer not in CONDITIONS[c] else []
def guard():
 assert not (OUT/'STOP.json').exists(),'STOP present'
 assert shutil.disk_usage('/data').free>20*1024**3,'Disk reserve below 20GiB'
 p=json.loads((OUT/'protocol.json').read_text())
 if time.time()>=p['gpu_deadline_epoch']:raise TimeoutError('Frozen GPU cutoff; reserve final30min')
