import sys,gc,os,time,traceback,json,csv
from run_common import *
sys.path.insert(0,str(L/'late9_sources'))
import torch
from vlm_research.internvl3_svh_seven_benchmarks import worker as original
from vlm_research.internvl3_svhr_late9_sevenbench.worker import generate
from vlm_research.internvl3_svhr_late9_sevenbench.common import KEEP
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
O=L/'results/internvl3';protocol=json.loads((O/'protocol.json').read_text());binding=sha(O/'protocol.json');start=time.time()
rows=json.loads((O/'frozen_rows.json').read_text());paths=json.loads((O/'input_paths.json').read_text())
inputs_verified={r['question_id']:r for r in json.loads((O/'inputs_verified.json').read_text())['bindings']}
hist={r['question_id']:r for r in json.loads((O/'historical_late9_3.json').read_text())}
records=load_records(O/'predictions');model=None;current=None
def progress(status,**kw):atomic(O/'progress.json',dict(status=status,complete=len(records)+3,expected=1000,new_generated=len(records),historical_reused=3,elapsed_s=time.time()-start,**kw))
try:
 for path,h in protocol['source_SHA256'].items():assert sha(L/'late9_sources'/path)==h
 gpu=gpu_idle();progress('LOADING_MODEL',gpu=gpu)
 original.MODEL=Path('/home/featurize/ate-local-models/InternVL3-8B');model,tok=original.load_model()
 assert model.language_model.config._attn_implementation=='eager'
 inputs=original.Inputs(model,tok)
 atomic(O/'environment.json',dict(original.environment(model,tok),GPU_UUID=os.environ['CUDA_VISIBLE_DEVICES'],model_loads=1))
 for q,r in records.items():assert r['protocol_sha256']==binding and q not in hist
 def infer(q,audit):
  global current
  current=q
  if q in records:return records[q]
  assert q not in hist,'Never regenerate historical completed IDs'
  r=rows[int(q)];ready,ib,b=inputs.ready(dict(r,image_path=paths[r['image_sha256']]))
  v=inputs_verified[q]
  assert b['pixels_bf16_sha256']==v['pixels_bf16_sha256'] and b['input_ids_sha256']==v['input_ids_sha256']
  torch.cuda.reset_peak_memory_stats();t=time.time()
  result,scores,control=generate(model,tok,ready,'late9',128,audit=audit)
  assert result['independent_cache'] and result['reaccess_layers']==KEEP and result['cutoff']==28
  assert [x['layer'] for x in result['kv_layout'] if x['visual_kv']]==KEEP
  if audit:
   a=result['audit'];assert a['hidden_exit']==[dict(before=r['input_tokens'],after=r['input_tokens']-256,removed=256)]
   assert a['exit_stage']=='after_L27_before_final_norm' and a['R_image_permission_boundary']==20
   assert a['allowed_image_probability_samples'] and all(x[2]>0 for x in a['allowed_image_probability_samples'])
   assert all(x[2]==0 for x in a['blocked_image_probability_samples']+a['R_image_probability_samples'])
  record=dict(model='InternVL3-8B',method='Late9',question_id=q,input_binding=ib,input_binding_details=b,
      image_sha256=r['image_sha256'],prompt_sha256=r['prompt_sha256'],protocol_sha256=binding,generation=result,
      **score(q,result['text']),origin='NEW_GPU_MISSING_ONLY',peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,elapsed_s=time.time()-t)
  atomic(O/'predictions'/(q+'.json'),record);records[q]=record
  del scores,control,ready;progress('SMOKE' if audit else 'RUNNING',last_id=q)
  if audit or len(records)%50==0:print('INTERNVL_PROGRESS',len(records)+3,'/1000',q,flush=True)
  return record
 for q in protocol['smoke_ids']:infer(q,True)
 atomic(O/'smoke.json',dict(status='PASS_STRUCTURAL_INHERITED_NUMERICAL_WAIVER',ids=protocol['smoke_ids'],reused_as_formal=True,no_native_regeneration=True,no_historical_late9_regeneration=True,numerical_status='Original EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER retained; no new numerical equivalence claim'))
 for r in rows:
  if r['question_id'] not in hist:infer(r['question_id'],False)
 assert len(records)==997
 progress('COMPLETE_PENDING_INDEPENDENT_AUDIT');print('INTERNVL_LATE9_COMPLETE_NEW997_REUSE3',flush=True)
except BaseException as e:
 atomic(O/'failure.json',dict(status='BLOCKED',question_id=current,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));progress('BLOCKED',current_id=current,error=str(e));traceback.print_exc();raise
finally:
 model=None;gc.collect();torch.cuda.empty_cache();print('INTERNVL_GPU_RELEASE_ON_PROCESS_EXIT',flush=True)
