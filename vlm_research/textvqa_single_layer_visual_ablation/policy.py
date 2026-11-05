"""Prespecified single-layer TextVQA intervention table, CPU-only."""
import pathlib,time,shutil,json
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'results/textvqa_single_layer_visual_ablation'
HIST=pathlib.Path('/data/lk/ATE-worktrees/textvqa-coco-middle-transfer/vlm_research')
INPUTS=HIST/'results/textvqa_visual_reaccess_budget_exploration/input_specs'
SEED=20261105
MIDDLE=[0]+list(range(11,21))
UUIDS=['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
CONDITIONS={}
for family in ['All-Text','Ponly','Donly','Both']:
 for layer in range(32):CONDITIONS[f'{family}-L{layer}']=dict(family=family,layer=layer,reference='Native')
for layer in MIDDLE:CONDITIONS[f'Middle-minus-L{layer}']=dict(family='Middle-minus',layer=layer,reference='Middle')
for layer in range(32):CONDITIONS[f'Addback-L{layer}']=dict(family='Addback',layer=layer,reference='Decode-None')
for name in ['Native','Middle','Decode-None']:CONDITIONS[name]=dict(family=name,layer=None,reference=name)
NATIVE=[c for c,v in CONDITIONS.items() if v['family'] in ['All-Text','Ponly','Donly','Both']]
PRIMARY=[c for c,v in CONDITIONS.items() if v['family'] in ['All-Text','Both','Middle-minus']]
SECONDARY=[c for c,v in CONDITIONS.items() if v['family'] in ['Ponly','Donly']]
OPTIONAL=[c for c,v in CONDITIONS.items() if v['family']=='Addback']
def query_rows(layout,c,layer,past=0,noop=False):
 if noop:return []
 d=CONDITIONS[c];f=d['family'];target=d['layer'];p=layout['prompt_length']-1
 if f=='Native':return []
 if f in ['Middle','Middle-minus','Decode-None','Addback']:
  allowed=MIDDLE if f=='Middle' else ([k for k in MIDDLE if k!=target] if f=='Middle-minus' else ([target] if f=='Addback' else []))
  return [0] if past and layer not in allowed else []
 if layer!=target:return []
 if past:return [0] if f in ['All-Text','Donly','Both'] else []
 return layout['R']+[p] if f=='All-Text' else ([p] if f in ['Ponly','Both'] else [])
def guard():
 assert not (OUT/'STOP.json').exists(),'STOP present'
 assert shutil.disk_usage('/data').free>15*1024**3,'Disk reserve below 15GiB'
 p=json.loads((OUT/'protocol.json').read_text())
 if time.time()>=p['gpu_deadline_epoch']:raise TimeoutError('Prespecified GPU deadline reached; reserve final 30 minutes')
