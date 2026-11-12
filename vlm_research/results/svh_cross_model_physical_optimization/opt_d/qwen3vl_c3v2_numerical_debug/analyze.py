"""CPU alignment and error attribution of archived numerical probes."""
import csv,collections,gc,sys
import torch
from .common import *
from .probe import ORDER,align,metrics
def table(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['status']);w.writeheader();w.writerows(rows)
def main(rank=0):
 torch.set_num_threads(1);archive={};index=read(OUT/f'shard{rank}/tensor_index.json');layout=read(OUT/'sample_manifest.json')['input_bindings'][ORIGINAL];text=[i for i in range(layout['input_length']) if i not in layout['image_positions']]
 for c in CONDITIONS:
  p=PRIVATE/f'shard{rank}/{c}_prefill_probes.pt';assert sha(p)==next(x['sha256'] for x in index if x['path']==str(p));archive[c]=torch.load(p,weights_only=True)
 rows=[];position=[];firsts={}
 for a,b in PAIRS:
  pair=a+'__'+b;left=archive[a];right=archive[b];allrows=[]
  for layer in range(38):
   for stage in ORDER:
    key=f'L{layer:02d}/{stage}'
    if key not in left['records'] or key not in right['records']:continue
    x=left['records'][key];y=right['records'][key];x['retained_positions']=text
    for scope in ['text','p_minus_1','all_common']:
     u,v,positions=align(x,y,scope)
     if u is None:continue
     m=metrics(u,v);row=dict(comparison=pair,layer=layer,stage=stage,scope=scope,common_queries=len(positions),reference_physical_shape=str(x['shape']),candidate_physical_shape=str(y['shape']),**m);rows.append(row);allrows.append(row)
  ptext=[r for r in allrows if r['scope']=='text' and not r['equal']]
  first=ptext[0] if ptext else None;large=next((r for r in ptext if r['mean_abs'] is not None and r['mean_abs']>1e-3),None)
  firsts[pair]=dict(first_nonzero_text=first,first_mean_above_1e3=large)
  for l,r in zip(left['position_records'],right['position_records']):
   lp=l['cache_position'];rp=r['cache_position'];common=sorted(set(lp)&set(rp));li=torch.tensor([lp.index(p) for p in common]);ri=torch.tensor([rp.index(p) for p in common]);cos_equal=torch.equal(l['cos'].index_select(1,li),r['cos'].index_select(1,ri));sin_equal=torch.equal(l['sin'].index_select(1,li),r['sin'].index_select(1,ri))
   ids_equal=torch.equal(torch.tensor(l['position_ids']).index_select(-1,li),torch.tensor(r['position_ids']).index_select(-1,ri))
   maskleft=left['records'].get(f"L{l['layer']:02d}/effective_mask");maskright=right['records'].get(f"L{r['layer']:02d}/effective_mask")
   allowed=None
   if maskleft and maskright:
    maskleft['retained_positions']=text;u,v,pos=align(maskleft,maskright,'text');allowed=bool(torch.equal(u==0,v==0))
   position.append(dict(comparison=pair,layer=l['layer'],common_positions=common,position_ids_equal=ids_equal,cos_equal=cos_equal,sin_equal=sin_equal,common_text_allowed_mask_equal=allowed))
  print(pair,'FIRST',first,'AMPLIFICATION',large,flush=True)
 table('attention_component_differences.csv',rows)
 table('layerwise_hidden_differences.csv',[r for r in rows if r['stage'] in ['layer_input','attention_residual','layer_output','final_norm','lm_head_input']])
 write(OUT/'first_divergence_summary.json',firsts)
 write(OUT/'position_and_mask_audit.json',dict(comparisons=position,all_common_position_ids_equal=all(x['position_ids_equal'] for x in position),all_common_rotary_equal=all(x['cos_equal'] and x['sin_equal'] for x in position),all_common_mask_permissions_equal=all(x['common_text_allowed_mask_equal'] is not False for x in position),deepstack_equal={a+'__'+b:all(torch.equal(x['features'],y['features']) and torch.equal(x['visual_mask'],y['visual_mask']) for x,y in zip(archive[a]['deepstack'],archive[b]['deepstack'])) for a,b in PAIRS}))
 factors=read(OUT/f'shard{rank}/factor_isolation.json');clean=[]
 for r in factors:clean.append({k:v for k,v in r.items() if k!='top20_union'})
 table('factor_isolation.csv',clean)
 csvfile=OUT/f'shard{rank}/bf16_shape_sensitivity.csv'
 if csvfile.exists():(OUT/'bf16_shape_sensitivity.csv').write_bytes(csvfile.read_bytes())
 print('CPU analysis complete',len(rows),'component comparisons',flush=True)
if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 0)
