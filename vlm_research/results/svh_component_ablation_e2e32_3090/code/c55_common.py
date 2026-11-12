"""Independent component-efficiency extension; old results remain read-only."""
import sys
from pathlib import Path
ROOT=Path('/home/featurize/work/ATE')
OLD=ROOT/'vlm_research/results/svh_ablation_efficiency_3090'
sys.path.insert(0,str(OLD/'code'))
from s55_common import *
N=ROOT/'vlm_research/results/svh_component_ablation_e2e32_3090'
CONDITIONS=['Native','wo_selective','wo_bypass','SVH']
LABELS=dict(Native='Native Full',wo_selective='SVH w/o Selective Transfer',wo_bypass='SVH w/o Attention Bypass',SVH='SVH Full')
def verify():
 verify_sources()
 for path,h in read(N/'source_manifest.json')['files'].items():assert sha(path)==h,path
def safety():
 assert not (N/'STOP.flag').exists()
 return guard()
