"""CPU-only shape reconstruction from measured cache rows, bytes and smoke B/H/D.

Does not claim fresh tensor.shape sampling for all efficiency requests: the
frozen original audit records actual token dimensions and storage bytes, while
smoke records the actual K/V batch, head and channel dimensions. Validate every
repeat against those independent measurements without another GPU inference.
"""
from pathlib import Path
import hashlib,json
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (O/'efficiency_completion.json').exists()
 dims=set();smoke_files=[]
 for p in sorted((O/'smoke').glob('*.json')):
  smoke_files.append(dict(path=str(p.relative_to(O)),SHA256=sha(p)))
  for r in read(p)['results'].values():
   for e in r['audit']['projection_shapes']:
    if e['op']=='QK':
     assert e['key']==e['value'];shape=e['key'];dims.add((shape[0],shape[1],shape[3]))
 assert dims=={(1,8,128)},dims
 batch,heads,channel=next(iter(dims));element_bytes=2;per_row=2*batch*heads*channel*element_bytes
 manifest=read(O/'historical_inputs.json');outputs=[];checked=0
 for q in manifest['benchmark_question_ids']:
  layout=manifest['input_bindings'][q];text=layout['input_length']-layout['image_tokens']
  for regime in ['natural','fixed16','fixed32']:
   for c in ['Native','Dense-F','Physical-Original']:
    expected=None;shape_rows=[]
    for rep in range(5):
     p=O/'efficiency_raw'/f'{q}__{regime}__{c}__{rep}.json';r=read(p)['result'];kv=r['kv'];checked+=1
     assert len(kv['layers'])==36
     assert kv['live_bytes']==sum(x['live_bytes'] for x in kv['layers'])
     assert kv['capacity_bytes']==sum(x['capacity_bytes'] for x in kv['layers'])
     if expected is None:expected=kv
     else:assert expected==kv,('Repeat cache-layout disagreement',q,regime,c,rep)
     for row in kv['layers']:
      absent=c=='Physical-Original' and 1<=row['layer']<=6
      live=0 if absent else (text if c=='Physical-Original' else layout['input_length'])+r['generated_tokens']-1
      capacity=0 if absent else text+r['max_new_tokens']-1 if c=='Physical-Original' else live
      assert row['live_tokens']==live and row['capacity_tokens']==capacity
      assert row['live_bytes']==per_row*live and row['capacity_bytes']==per_row*capacity
      if rep==0:shape_rows.append(dict(**row,K_live_shape=None if absent else [batch,heads,live,channel],V_live_shape=None if absent else [batch,heads,live,channel],K_capacity_shape=None if absent else [batch,heads,capacity,channel],V_capacity_shape=None if absent else [batch,heads,capacity,channel],element_bytes=element_bytes,raw_visual_KV_present=c!='Physical-Original'))
    outputs.append(dict(question_id=q,regime=regime,condition=c,repeat_layouts_identical=True,layers=shape_rows))
 result=dict(status='PASS',checked_requests=checked,checked_layouts=len(outputs),shape_provenance='Full 4D representation reconstructed from original audit actual token dimensions/storage bytes and smoke actual K/V B/H/D. Not an added direct tensor.shape probe for all 2160 requests.',smoke_sources=smoke_files,actual_smoke_BHD=[batch,heads,channel],element_bytes=element_bytes,all_live_and_storage_bytes_exactly_match=True,all_repeat_cache_layouts_identical=True,layouts=outputs)
 (O/'kv_tensor_shapes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('KV_SHAPE_CPU_CROSSCHECK_PASS',checked,len(outputs),flush=True)
if __name__=='__main__':main()
