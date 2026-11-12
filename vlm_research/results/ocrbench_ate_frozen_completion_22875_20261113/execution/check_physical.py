import json,sys
from run_common import *
O=L/'results/llava';P=json.loads((O/'protocol.json').read_text());cpu=json.loads((O/'cpu_bindings.json').read_text());out=[]
for q in P['smoke_ids']:
 r=json.loads((O/'predictions/Physical-C3'/f'{q}.json').read_text());g=r['generation'];ids=cpu[q]['input_ids'];n=len(ids);im={i for i,t in enumerate(ids) if t==32000};text=set(range(n))-im
 assert len(im)==576 and g['independent_cache'] and g['all_effective_logits_finite']
 assert not g['kv_audit']['retained_image_layers'] and g['kv_audit']['historical_cache_copies']==0
 for row in g['kv_audit']['layers']:
  if row['layer'] in range(2,6):assert row['used_tokens']==0
  else:assert not im.intersection(row['original_positions']) and text.issubset(row['original_positions']) and row['used_tokens']==len(row['original_positions'])
 for a in g['attention_audit']:
  assert a['blocked_probability_max']==0 and a['output_finite'] and a['row_sum_error']<.005
  if a['prefill']:
   # attention_audit records compact tensor row indices; visual_audit below
   # explicitly records original RoPE/query positions after physical gathering.
   expected=set(range(n if a['layer']<20 else len(text)))
   if a['layer'] in range(2,6):expected-={n-1}
   assert set(a['query_positions'])==expected,(q,a['layer'])
 for a in g['visual_audit']:
  assert a['answer_image_mass']==0
  if a['prefill']:assert set(a['original_query_positions'])==(set(range(n)) if a['layer']<20 else text)
  if not a['prefill'] or a['layer']>=20:assert not a['image_keys_present'] and a['relay_image_mass']==0
 assert any(a['prefill'] and a['layer']<20 and a['relay_image_mass']>0 for a in g['visual_audit'])
 out.append(dict(question_id=q,text=g['text'],score=r['official_score'],image_tokens=576,physical_exit_before_layer=20,skip_layers=[2,3,4,5],original_text_positions_preserved=True,answer_visual_access_zero=True,persistent_visual_KV_zero=True,row_sum_error_max=max(a['row_sum_error'] for a in g['attention_audit']),all_logits_finite=True,independent_KV=True))
atomic(O/'smoke_independent_check.json',dict(status='PASS',checked=len(out),checks=out,numerical_threshold_unchanged=True,new_dense_equivalence_not_claimed=True,checker_correction='Initial CPU assertion misread compact query_positions as original positions at L20; corrected to original_query_positions in visual_audit. Frozen execution and predictions unchanged.'));print('LLAVA_PHYSICAL_INDEPENDENT_CHECK_PASS',len(out))
