"""Two fresh four-condition32 sessions; actual K/V and unchanged frozen timer."""
from c55_common import *
import argparse,gc,statistics,sqlite3,traceback,fcntl
import torch
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.physical_handoff_fastpath.engine import Layout
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
from s55_ablation_engine import component
import c55_runtime as runtime

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--session',type=int,choices=(1,2),required=True);session=ap.parse_args().session
 out=N/('session'+str(session));out.mkdir(exist_ok=False);lock=(N/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if session==2:assert read(N/'session1/progress.json')['status'] in ('COMPLETE','COMPLETE_WITH_PERFORMANCE_WARNINGS')
 a=None;db=None;monitors=[];state=dict(status='RUNNING',phase='PRECHECK',session=session,pid=os.getpid(),formal_completed=0,formal_expected=120,requests_completed=0,started_ns=time.time_ns());warnings=[];calibrations=[];baselines={};previous_end=None
 def status():save(out/'progress.json',dict(state,updated_ns=time.time_ns()))
 try:
  status();verify();safety();a=load_model()
  for mode in ('gpu','cpu','processes'):
   with (out/(mode+'_monitor.log')).open('x') as log:monitors.append(subprocess.Popen([str(P),'-B',str(OLD/'code/s55_monitor.py'),str(out),str(os.getpid()),UUID,mode],stdout=log,stderr=subprocess.STDOUT))
  save(out/'environment.json',dict(GPU=gpu(),GPU_detail=command(['nvidia-smi','-q']),session=session,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,transformers=__import__('transformers').__version__,CPU_affinity=sorted(os.sched_getaffinity(0)),intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads(),protocol_sha256=sha(N/'protocol.json'),model_receipt_sha256=sha(M/'.ate_revision.json')))
  samples=lines(N/'input_manifest.jsonl.gz');mapping=read(N/'input_path_mapping.json');questions=read(U/'question_positions.json') if (U/'question_positions.json').exists() else read(R/'vlm_research/results/unified_e2e32_eight_methods_3090_repeatability_v2/session1/question_positions.json');prepared={};layouts={}
  for s in samples:
   raw=dict(input_ids=torch.tensor([s['input_ids']],dtype=torch.long),attention_mask=torch.ones((1,len(s['input_ids'])),dtype=torch.long),pixel_values=frozen.paper_pixels(mapping[s['key']],a.image_processor).half())
   assert {k:tensor_hash(v) for k,v in raw.items()}==s['input_hashes'];prepared[s['key']]=raw
   ids=s['input_ids'];image=tuple(i for i,v in enumerate(ids) if v==32000);text=tuple(i for i,v in enumerate(ids) if v!=32000);relay=tuple(range(image[-1]+1,len(ids)-1));q=tuple(questions[s['key']]) if s['benchmark']=='GQA' else relay;layouts[s['key']]=Layout(len(ids),image,text,relay,q,20)
  db=sqlite3.connect(out/'checkpoint.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE requests(id INTEGER PRIMARY KEY,category TEXT,key TEXT,condition TEXT,payload TEXT,sha256 TEXT)');db.execute('CREATE UNIQUE INDEX formal_unique ON requests(key,condition) WHERE category="formal"');db.commit()
  allrecords=[];attempts=(out/'attempts.jsonl').open('x')
  def request(key,condition,category,smoke=False,reference=False,noop=False,capture=False,**extra):
   nonlocal previous_end
   assert all(p.poll() is None for p in monitors) and not (out/'STOP.flag').exists()
   a.inputs={k:v.to('cuda:0') for k,v in prepared[key].items()};layout=layouts[key];rid=len(allrecords)+1;captured=[];handle=None
   if capture:handle=a.model.register_forward_hook(lambda m,args,g:captured.append(g.logits[:,-1].detach().float().cpu().clone()))
   attempts.write(json.dumps(dict(id=rid,key=key,condition=condition,category=category,at_ns=time.time_ns(),**extra))+'\n');attempts.flush();os.fsync(attempts.fileno())
   cpu0=time.thread_time_ns();sched0=[int(v) for v in Path(f'/proc/self/task/{os.getpid()}/schedstat').read_text().split()];begin=time.time_ns();torch.manual_seed(0)
   try:g=runtime.ref.generate(a,layout,'FastVNoop' if noop else condition,smoke=smoke) if reference or noop else runtime.generate(a,layout,condition,smoke=smoke)
   finally:
    if handle:handle.remove()
   end=time.time_ns();sched1=[int(v) for v in Path(f'/proc/self/task/{os.getpid()}/schedstat').read_text().split()]
   assert g['output_tokens']==32 and len(g['token_ids'])==32 and g['real_decode_calls']==31 and g['prefill_calls']==g['vision_forward_calls']==1
   assert g['all_logits_finite'] and g['logits_to_keep']==1 and g['use_cache']
   assert abs(g['E2E_ms']-g['TTFT_ms']-31*g['TPOT_ms'])<1e-6
   assert abs(statistics.mean(g['step_event_ms'][1:])-g['CUDA_Event_Decode_TPOT_ms'])<1e-6
   assert g['cache']['live_bytes']==g['cache']['storage_bytes'] and g['cache']['cached_decode_inputs']==31
   assert not any(l._forward_hooks or l._forward_pre_hooks for l in a.layers) and not a.model._forward_hooks
   early=statistics.median(g['step_event_ms'][1:6]);late=statistics.median(g['step_event_ms'][6:]);jump=late/early-1
   r=dict(id=rid,session=session,GPU_UUID=UUID,category=category,key=key,condition=condition,begin_ns=begin,end_ns=end,gap_s=None if previous_end is None else (begin-previous_end)/1e9,thread_CPU_ms=(time.thread_time_ns()-cpu0)/1e6,thread_runqueue_ms=(sched1[1]-sched0[1])/1e6,early_vs_late_decode_change=jump,input_hashes=next(s['input_hashes'] for s in samples if s['key']==key),generation=g,**extra);previous_end=end
   if abs(jump)>.05:warnings.append(dict(type='within_request_event_decode_change_over5pct',request_id=rid,category=category,relative_change=jump))
   with db:db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?)',(rid,category,key,condition,json.dumps(r,ensure_ascii=False,allow_nan=False),digest(r)))
   allrecords.append(r);state['requests_completed']=len(allrecords)
   if category=='formal':state['formal_completed']+=1
   status();return (r,captured) if capture else r
  probes=[next(s['key'] for s in samples if s['benchmark']==b) for b in ('GQA','TextVQA')]
  state['phase']='SMOKE';status();smokes=[]
  for key in probes:
   for condition in CONDITIONS:
    current,logs=request(key,condition,'smoke',smoke=True,capture=True,check='component_physical32')
    if condition in ('Native','SVH'):
     old,oldlogs=request(key,condition,'smoke',smoke=True,capture=True,reference=True,check='unchanged_original32')
     assert current['generation']['token_ids']==old['generation']['token_ids'] and len(logs)==len(oldlogs)==32
     assert all(torch.equal(x,y) for x,y in zip(logs,oldlogs))
    if condition=='Native':
     noop,nlogs=request(key,condition,'smoke',smoke=True,capture=True,noop=True,check='NativeNoop32')
     assert current['generation']['token_ids']==noop['generation']['token_ids'] and all(torch.equal(x,y) for x,y in zip(logs,nlogs))
    # Exact32 quality-path numerical shadow. The supplied prefix is the actual
    # greedy runtime output, and every shadow argmax must reproduce that prefix.
    with torch.inference_mode(),component(condition):
     q,qlog=frozen.hf_generate(a,layouts[key],'HF-Native' if condition=='Native' else 'Physical-C3',32,capture=True,fixed_prefix=current['generation']['token_ids'])
    assert q['token_ids']==current['generation']['token_ids'] and len(qlog)==len(logs)==32
    assert all(torch.equal(x,y) for x,y in zip(logs,qlog))
    assert [int(x.argmax(-1)) for x in qlog]==current['generation']['token_ids']
    smokes.append(dict(key=key,condition=condition,status='PASS',physical_RoPE_QKV='PASS',quality_path_all32_logits_bitwise_exact=True,shadow_prefix='Actual runtime greedy tokens; all shadow argmax equal; not Gold; Smoke only',tokens=current['generation']['token_ids'],per_step_logits_sha256=[tensor_hash(x) for x in logs],KV_bytes=current['generation']['cache']['live_bytes']))
  save(out/'smoke_validation.json',dict(status='PASS',cases=smokes,NativeNoop_all32_logits_exact=True,formal_hooks=False))
  state['phase']='FIXED_WARMUP';status()
  for i in range(40):request(probes[0],'Native','warmup',warmup_phase='Native40',index=i)
  for repeat,key in enumerate(probes):
   for condition in CONDITIONS:request(key,condition,'warmup',warmup_phase='each_condition2',repeat=repeat)
  state['phase']='BASELINE';status()
  for repeat in range(10):
   for key in probes:request(key,'Native','baseline',repeat=repeat)
  for key in probes:
   rs=[r['generation'] for r in allrecords if r['category']=='baseline' and r['key']==key];baselines[key]={m:statistics.median(r[m] for r in rs) for m in ('E2E_ms','TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms')}
  warm=[r['generation']['E2E_ms'] for r in allrecords if r['category']=='warmup' and r.get('warmup_phase')=='Native40'];cv=statistics.stdev(warm)/statistics.mean(warm)
  if cv>.02:warnings.append(dict(type='Native40_warmup_CV_over2pct',cv=cv))
  save(out/'warmup_baseline.json',dict(fixed_counts=dict(Native32=40,each_condition=2,dual_probe_baseline=20),Native40_CV=cv,baseline=baselines,no_adaptive_warmup=True))
  state['phase']='FORMAL';status();print('COMPONENT_FORMAL_START',session,flush=True)
  for index,s in enumerate(samples):
   offset=index%4;order=CONDITIONS[offset:]+CONDITIONS[:offset]
   for condition in order:request(s['key'],condition,'formal',input_index=index)
   if (index+1)%5==0:
    check=[]
    for repeat in range(3):
     for key in probes:check.append(request(key,'Native','calibration',window=(index+1)//5,repeat=repeat))
    for key in probes:
     g=[r['generation'] for r in check if r['key']==key];med={m:statistics.median(r[m] for r in g) for m in baselines[key]};drift={m:med[m]/baselines[key][m]-1 for m in med};item=dict(window=(index+1)//5,key=key,median=med,baseline=baselines[key],relative_drift=drift,warn=any(abs(drift[m])>.02 for m in ('E2E_ms','TPOT_ms')));calibrations.append(item)
     if item['warn']:warnings.append(dict(type='calibration_over2pct',**item))
    save(out/'calibration.json',dict(status='WARN' if any(x['warn'] for x in calibrations) else 'PASS',checks=calibrations,baseline_not_reset=True));verify();safety();print('COMPONENT_PROGRESS',session,state['formal_completed'],'/120','calibration_WARN',sum(x['warn'] for x in calibrations),flush=True)
  assert state['formal_completed']==120 and db.execute('PRAGMA quick_check').fetchone()[0]=='ok';db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
  state.update(status='COMPLETE_WITH_PERFORMANCE_WARNINGS' if warnings else 'COMPLETE',phase='COMPLETE',finished_ns=time.time_ns(),performance_warnings=len(warnings));status();save(out/'performance_warnings.json',warnings)
 except BaseException as ex:
  state.update(status='BLOCKED' if state['formal_completed']==0 else 'PARTIAL',error=repr(ex),traceback=traceback.format_exc(),failed_ns=time.time_ns());status();print(traceback.format_exc(),flush=True);raise
 finally:
  (out/'monitor.stop').write_text(str(time.time_ns()))
  for p in monitors:
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.terminate()
  if db:db.close()
  a=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  save(out/'resource_release.json',dict(at_ns=time.time_ns(),pre_exit_allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0,context_released_on_exit=True))
if __name__=='__main__':main()
