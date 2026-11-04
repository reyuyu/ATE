"""Frozen COCO-to-TextVQA transfer policy; no model or GPU imports."""
import random,pathlib,shutil
SEED=20261105
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'results/textvqa_coco_middle_transfer'
ALL=list(range(32))
LAYER_SETS={
 'A':dict(name='Native Full',Pminus1=ALL,Decode=ALL,origin='historical'),
 'F':dict(name='Pminus1 Full / Decode None',Pminus1=ALL,Decode=[],origin='historical'),
 'B':dict(name='Both None',Pminus1=[],Decode=[],origin='historical'),
 'C':dict(name='COCO-Middle',Pminus1=ALL,Decode=[0]+list(range(11,21)),origin='new'),
 'D':dict(name='COCO-Early',Pminus1=ALL,Decode=list(range(11)),origin='new'),
 'E':dict(name='COCO-Late',Pminus1=ALL,Decode=[0]+list(range(21,32)),origin='new'),
 'R':dict(name='Random-11',Pminus1=ALL,Decode=sorted([0]+random.Random(SEED).sample(range(1,32),10)),origin='new'),
}
NEW=['C','D','E','R']
COMPARISONS=[('C','A'),('C','F'),('C','D'),('C','R'),('C','E')]
def guard():
 assert not (OUT/'STOP.json').exists(),'STOP present; diagnose before resuming'
 assert shutil.disk_usage('/data').free>20*1024**3,'Insufficient /data disk reserve'
