"""Named Late14 full OCRBench: generate only missing900, reuse prior100 exactly."""
import os,sys,json,time,gc,gzip,shutil,traceback
from pathlib import Path
O=Path('/home/featurize/ate-experiments/internvl-svh-r-new-ocrbench1000-22875-20261113')
L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
sys.path.insert(0,str(L));from run_common import sha,atomic,score,gpu_idle
sys.path.insert(0,str(L/'late9_sources'))
import torch
from vlm_research.internvl3_svh_seven_benchmarks import worker as original
from vlm_research.internvl3_svh_r_terminal_windows_textvqa import engine as late
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
P=json.loads((O/'protocol.json').read_text());pb=sha(O/'protocol.json');cfg=json.loads((O/'configuration.json').read_text());rows=json.loads((O/'frozen_rows.json').read_text());byid={r['question_id']:r for r in rows}
paths=json.loads((O/'input_paths.json').read_text());v={r['question_id']:r for r in json.loads((O/'expected_bindings.json').read_text())['bindings']};inherited=json.loads((O/'inherited_late14_100.json').read_text())
F=O/'predictions';F.mkdir(exist_ok=True);records={f.stem:json.loads(f.read_text()) for f in F.glob('*.json')};start=time.time();model=None;current=None
assert len(inherited)==100 and cfg['reaccess_layers']==list(range(14,28)) and P['configuration_sha256']==sha(O/'configuration.json')
for q,x in records.items():assert q not in inherited and x['protocol_sha256']==pb and q in byid
def progress(status,**kw):atomic(O/'progress.json',dict(status=status,configuration_name=cfg['configuration_name'],expected=1000,complete=100+len(records),reused=100,new_completed=len(records),new_expected=900,elapsed_s=time.time()-start,**kw))
def checkpoint():
 folder=Path('/home/featurize/work/ate-results/internvl-svh-r-new-ocrbench1000-22875-20261113');folder.mkdir(parents=True,exist_ok=True)
 local=O/'checkpoint.jsonl.gz';tmp=local.with_suffix('.tmp')
 with gzip.open(tmp,'wt',encoding='utf-8') as f:
  for q in sorted(records):f.write(json.dumps(records[q],ensure_ascii=False,separators=(',',':'))+'\n')
 tmp.replace(local);shutil.copy2(local,folder/'checkpoint.jsonl.gz.tmp');(folder/'checkpoint.jsonl.gz.tmp').replace(folder/'checkpoint.jsonl.gz')
 atomic(folder/'checkpoint_receipt.json',dict(configuration_name=cfg['configuration_name'],protocol_sha256=pb,new_completed=len(records),inherited=100,sha256=sha(local)))
try:
 for p,h in P['source_SHA256'].items():assert sha(L/'late9_sources'/p)==h,p
 for f,k in [('frozen_rows.json','rows_sha256'),('expected_bindings.json','bindings_sha256'),('inherited_late14_100.json','inherited_records_sha256')]:assert sha(O/f)==P[k]
 progress('LOADING_MODEL',gpu=gpu_idle());original.MODEL=Path('/home/featurize/ate-local-models/InternVL3-8B');original.GPU=os.environ['CUDA_VISIBLE_DEVICES']
 model,tok=original.load_model();inputs=original.Inputs(model,tok);atomic(O/'environment.json',dict(original.environment(model,tok),GPU_UUID=os.environ['CUDA_VISIBLE_DEVICES'],model_loads=1,runner_sha256=sha(O/'run.py')))
 def infer(q,audit):
  global current
  current=q
  if q in inherited or q in records:return
  r=byid[q];ready,ib,b=inputs.ready(dict(r,image_path=paths[r['image_sha256']]))
  for k in ['pixels_bf16_sha256','input_ids_sha256','prompt_sha256','image_sha256']:assert b[k]==v[q][k],(q,k)
  g,s,c=late.generate(model,tok,ready,'Late14_L14_27',list(range(14,28)),max_tokens=128,audit=audit,physical=True,cutoff=28)
  assert g['independent_cache'] and g['reaccess_layers']==list(range(14,28)) and g['cutoff']==28
  assert [a['layer'] for a in g['kv_layout'] if a['visual_kv']]==list(range(14,28))
  if audit:
   a=g['audit'];assert a['hidden_exit']==[dict(before=r['input_tokens'],after=r['input_tokens']-256,removed=256)]
   assert a['exit_stage']=='after_L27_before_final_norm' and a['R_image_permission_boundary']==20 and a['finite'] and a['original_logical_positions']
   assert a['allowed_image_probability_samples'] and all(x[2]>0 for x in a['allowed_image_probability_samples'])
   assert all(x[2]==0 for x in a['blocked_image_probability_samples']+a['R_image_probability_samples'])
  x=dict(configuration_name=cfg['configuration_name'],method='interVL-SVH-R-new',question_id=q,category=r['category'],dataset_name=r['dataset_name'],image_sha256=r['image_sha256'],prompt_sha256=r['prompt_sha256'],input_binding=ib,input_binding_details=b,protocol_sha256=pb,generation=g,**score(q,g['text']))
  atomic(F/(q+'.json'),x);records[q]=x;del s,c,ready
  progress('SMOKE' if audit else 'RUNNING',last_id=q)
  if len(records)%50==0:checkpoint();print('FULL_PROGRESS',len(records)+100,'/1000',flush=True)
 for q in P['smoke_ids']:infer(q,True)
 atomic(O/'smoke.json',dict(status='PASS_STRUCTURAL',new_ids=P['smoke_ids'],count=4,reused_as_full_predictions=True,prior_late14_smoke_commit=P['inherited_commit'],numerical_status=P['numerical_status'],no_baseline_generation=True))
 print('SMOKE_PASS_NEW4',P['smoke_ids'],flush=True)
 for r in rows:infer(r['question_id'],False)
 assert len(records)==900 and not(set(records)&set(inherited));checkpoint();progress('COMPLETE_GENERATION_PENDING_CPU_AUDIT');print('FULL1000_REUSE100_NEW900_COMPLETE',flush=True)
except BaseException as e:
 atomic(O/'failure.json',dict(status='BLOCKED',question_id=current,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));progress('BLOCKED',question_id=current,error=str(e));checkpoint();traceback.print_exc();raise
finally:
 model=None;gc.collect();torch.cuda.empty_cache();print('GPU_RELEASE_ON_PROCESS_EXIT',flush=True)
