"""User-authorized quality-first continuation. Strict numerical failures stay recorded."""
import gc,json,os,platform,subprocess,sys,traceback
import torch,transformers
from transformers import AutoTokenizer
from .common import *
from . import engine as p
from .run import load,quality,efficiency
from vlm_research.internvl3_visual_writing_figure3_3000 import engine as b
def main():
 model=None;monitor=None
 protocol=read(RUN/'protocol.json');manifest=read(RUN/'sample_manifest.json');ph=sha(RUN/'protocol.json')
 assert protocol['user_override']['continue_despite_numerical_gate_failure']
 assert protocol['user_override']['exact_user_instruction']=='0.0004不大吧，在小测试集上得分差不多就行'
 original=read(RUN/'engineering_attempts/attempt01_frozen_gate_failure/smoke_validation.json')
 assert original['status']=='FAILED' and all(s['complete_tokens_equal'] for s in original['samples'])
 assert read(RUN/'cpu_tests.json')['status']=='PASS'
 for path,h in read(RUN/'source_manifest.json').items():assert sha(path)==h,path
 sh=sha(RUN/'source_manifest.json')
 assert sha(RUN/'sample_manifest.json')==protocol['sample_manifest_sha256']
 assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
 assert command('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')==GPU
 assert not command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader')
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 save(RUN/'active.json',dict(pid=os.getpid(),stage='USER_AUTHORIZED_CONTINUATION',protocol_sha256=ph))
 environment=read(RUN/'environment.json')
 environment.update(source_commit=command('git','rev-parse','HEAD'),source_manifest_sha256=sh,
  protocol_sha256=ph,continuation_user_override=True,worker_pid=os.getpid(),python_executable=sys.executable)
 save(RUN/'environment.json',environment)
 try:
  log=(RUN/'gpu_telemetry_continuation.csv').open('w')
  monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,name,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used','--format=csv','-l','2'],stdout=log,stderr=subprocess.STDOUT)
  model=load();tok=AutoTokenizer.from_pretrained(MODEL,use_fast=False,trust_remote_code=True,local_files_only=True)
  loader=b.official_preprocessor(MODEL);versions=[(x._version,x.data_ptr()) for x in model.parameters()]
  qid=manifest['smoke_question_ids'][2];s=next(x for x in manifest['samples'] if x['question_id']==qid)
  prep=b.prepare(model,tok,loader,s);ready=p.gpu_ready(model,tok,loader,s)
  audits=[];layouts=[];tokens={}
  for method in METHODS:
   row,_,control=p.generate(model,tok,ready,method,fixed=True,audit=True,prepared=prep)
   assert row['generated_tokens']==32 and row['audit']['real_decode']==31
   assert row['audit']['norm_calls']==32*28*2 and row['audit']['MLP_calls']==32*28
   save(RUN/'operational_smoke'/('fixed32_'+method+'.json'),row)
   audits.append(row['audit']);tokens[method]=row['token_ids']
   layouts.extend(dict(question_id=qid,audit_fixed32=True,**x) for x in row['kv_layout'])
  # Identical tokens through timed vision decomposition, without audit hooks.
  clean,_,_=p.generate(model,tok,ready,'Native',fixed=True,prepared=prep)
  timed,_,_=p.generate(model,tok,ready,'Native',fixed=True,timed=True)
  assert clean['token_ids']==timed['token_ids']==tokens['Native']
  save(RUN/'smoke_validation.json',dict(status='PASS_WITH_USER_NUMERICAL_WAIVER',
   operational_correctness='PASS',strict_numerical_gate='FAILED',strict_numerical_equivalence_claim=False,
   original_smoke=original,user_override=protocol['user_override'],fixed32_audits=audits,
   real_decode_steps=31,native_clean_timed_tokens_equal=True,protocol_sha256=ph))
  save(RUN/'physical_execution_audit.json',dict(original_short_smoke=read(RUN/'engineering_attempts/attempt01_frozen_gate_failure/physical_execution_audit.json'),
   fixed32_audits=audits,strict_numerical_gate='FAILED',user_waiver=True))
  csvwrite(RUN/'kv_layout.csv',layouts,['question_id','audit_fixed32','method','layer','logical_length','physical_length','visual_kv','k_shape','v_shape','dtype','live_bytes','capacity_bytes','skip_attention'])
  del prep,ready,control
  print('OPERATIONAL_SMOKE_PASS_STRICT_NUMERIC_FAILED_USER_WAIVER',flush=True)
  quality(model,tok,loader,manifest,ph,sh)
  efficiency(model,tok,loader,manifest,ph,sh)
  assert versions==[(x._version,x.data_ptr()) for x in model.parameters()]
  save(RUN/'execution_status.json',dict(status='COMPLETE_WITH_NUMERICAL_WAIVER',quality_requests=360,
   efficiency_requests=450,strict_numerical_gate='FAILED',user_authorized_quality_first=True))
 except BaseException as exc:
  save(RUN/'continuation_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),protocol_sha256=ph))
  save(RUN/'execution_status.json',dict(status='PARTIAL',reason=repr(exc),strict_numerical_gate='FAILED'));raise
 finally:
  model=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  if monitor is not None:monitor.terminate();monitor.wait(timeout=10);log.close()
  save(RUN/'gpu_release.json',dict(worker_pid=os.getpid(),model_deleted=True,allocated_MiB=torch.cuda.memory_allocated()/1048576 if torch.cuda.is_initialized() else 0,context_released_on_process_exit=True))
if __name__=='__main__':main()
