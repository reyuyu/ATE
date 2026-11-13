"""Only the two authorized Original conditions; per-request atomic persistence."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import gc,hashlib,json,os,time,traceback
import torch
from benchmark_inputs import prepare,digest
from scorers import score
O=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval/vlm_research/results/qwen3vl_c3_3090_official_eval')
def read(p):return json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def run_seven(model,processor,call,basebinding):
 manifest=read(O/'input_manifest.json');rows=manifest['rows'];manifest_sha=hashlib.sha256((O/'input_manifest.json').read_bytes()).hexdigest()
 groups=defaultdict(list)
 for s in rows:groups[(s['benchmark'],s['protocol_label'])].append(s)
 statuses=[]
 for (benchmark,label),samples in groups.items():
  started=time.monotonic();completed=0;failure=None;folder=O/'benchmark_predictions'/benchmark/label
  for s in samples:
   try:
    x,l=prepare(processor,s)
    b=dict(**basebinding,sample_SHA256=digest(s),input_SHA256=digest(l),input_manifest_SHA256=manifest_sha,request_SHA256=s['request_SHA256'])
    request_dir=O/'request_predictions'/s['request_SHA256'][:2]
    # Inputs are prepared exactly once, then used by both conditions unchanged.
    for c in ['Native','Physical-Original']:
     request_path=request_dir/(s['request_SHA256']+'__'+c+'.json')
     shared_binding=dict(**basebinding,input_SHA256=digest(l),request_SHA256=s['request_SHA256'])
     if request_path.exists():
      record=read(request_path);assert record['binding']==shared_binding,'Request tensor binding changed'
     else:
      result,_=call(x,l,c,max_tokens=s['max_new_tokens'])
      record=dict(binding=shared_binding,result=result,input_layout=l,at_UTC=datetime.now(timezone.utc).isoformat())
      write(request_path,record)
     path=folder/(hashlib.sha256(s['key'].encode()).hexdigest()+'__'+c+'.json')
     index=dict(key=s['key'],question_id=s['question_id'],binding=b,request_path=str(request_path.relative_to(O)),gold=s['answer'],image_SHA256=s['image_sha256'],generation_SHA256=digest(record),new_GPU_prediction=True,scoring=score(s,record['result']['text']))
     if path.exists():assert read(path)==index
     else:write(path,index)
    completed+=1
    if completed%10==0:print('SEVEN_QUALITY',benchmark,label,completed,'/',len(samples),flush=True)
    if completed%10==0 or completed==len(samples):write(O/'progress.json',dict(status='RUNNING',phase='SEVEN_BENCHMARK_QUALITY',benchmark=benchmark,protocol_label=label,completed_questions=completed,total_questions=len(samples),elapsed_seconds=time.monotonic()-started,at_UTC=datetime.now(timezone.utc).isoformat()))
    del x
   except torch.cuda.OutOfMemoryError as e:
    # No resolution/backend retry. Preserve failed ID and stop this full view.
    failure=dict(status='BLOCKED',question_id=s['question_id'],key=s['key'],error='CUDA_OUT_OF_MEMORY',detail=str(e),traceback=traceback.format_exc(),input_length=l.get('input_length') if 'l' in locals() else None,image_tokens=l.get('image_tokens') if 'l' in locals() else None,protocol_unchanged=True)
    write(folder/'failure.json',failure)
    if 'x' in locals():del x
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    print('BENCHMARK_OOM_BLOCKED',benchmark,label,s['question_id'],flush=True);break
   if completed%50==0:gc.collect()
  status=dict(benchmark=benchmark,protocol_label=label,status='COMPLETE' if failure is None else 'BLOCKED',completed_questions=completed,total_questions=len(samples),failure=failure)
  statuses.append(status);write(folder/'completion.json',status);write(O/'seven_progress.json',dict(status='RUNNING',views=statuses))
 write(O/'seven_completion.json',dict(status='COMPLETE' if all(s['status']=='COMPLETE' for s in statuses) else 'PARTIAL',views=statuses))
