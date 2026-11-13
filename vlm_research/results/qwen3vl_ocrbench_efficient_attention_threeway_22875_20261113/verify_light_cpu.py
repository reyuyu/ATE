"""Portable, dependency-free CPU audit of the published 3000 predictions.

Usage: python -B verify_light_cpu.py /path/to/threeway/result_directory
No images, weights, GPU or network are used. Full token-decode and physical
execution checks remain in final_independent_audit.json/raw archive.
"""
import sys,json,hashlib,datetime
from pathlib import Path
D=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
P=json.loads((D/'protocol.json').read_text(encoding='utf-8'));pb=hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest()
inputs={r['question_id']:r for r in map(json.loads,(D/'inputs_light.jsonl').read_text(encoding='utf-8').splitlines())}
assert list(inputs)==[f'{i:04d}' for i in range(1000)] and len({r['image_sha256'] for r in inputs.values()})==930
all_data={};summary=[]
for c in ['Native','Physical-Original','SVH-R']:
 data={r['question_id']:r for r in map(json.loads,(D/(c+'_predictions_light.jsonl')).read_text(encoding='utf-8').splitlines())};assert list(data)==list(inputs)
 for q,r in data.items():
  x=inputs[q];assert r['condition']==c and r['model_revision']==P['revision'] and r['attention_backend']=='SDPA_EFFICIENT_ATTENTION_ONLY'
  assert r['image_sha256']==x['image_sha256'] and r['input_tensor_hashes']==x['input_tensor_hashes'] and r['input_layout_sha256']==x['input_layout_sha256']==P['all_layout_hashes'][q]
  assert r['protocol_sha256']==(pb if c=='SVH-R' else P['base_protocol_sha256'])
  assert r['gold'] and isinstance(r['gold'],list)
  text=r['prediction'].strip().replace('\n',' ')
  if x['dataset_name']=='HME100k':
   text=text.replace(' ','');gold=[a.strip().replace('\n',' ').replace(' ','') for a in r['gold']]
  else:
   text=text.lower();gold=[a.strip().replace('\n',' ').lower() for a in r['gold']]
  value=1 if any(a in text for a in gold) else 0
  assert r['official_score']==r['independent_score']==value,(c,q)
 all_data[c]=data;summary.append(dict(method=c,score_out_of_1000=sum(r['official_score'] for r in data.values()),coverage=1000))
for q in inputs:assert all_data['Native'][q]['gold']==all_data['Physical-Original'][q]['gold']==all_data['SVH-R'][q]['gold']
result=dict(status='PASS',portable_CPU_predictions_checked=3000,unique_original_images=930,inputs_identical_all_three=1000,scores_recomputed_without_original_scorer_import=True,all_Gold_lists_agree=True,summaries=summary,no_model_or_GPU_or_network=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(D/'light_verification_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
