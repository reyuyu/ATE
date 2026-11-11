"""Run one frozen cohort with a real native prefill and independent cache clones."""
import sys,time,traceback,gc,importlib.util,os,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common import *
from engine import *
import smoke
def scorer():
 spec=importlib.util.spec_from_file_location('frozen_textvqa_eval',RUN/'references/textvqa_eval.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.TextVQAAccuracyEvaluator()
def score(s,text):
 return s._compute_answer_scores(text['gold']).get(s.answer_processor(text['prediction']),0.)
def store(db,key,condition,r):
 raw=json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))
 h=hashlib.sha256(raw.encode()).hexdigest()
 with db:db.execute('insert into predictions values(?,?,?,?)',(key,condition,h,raw))
def base_record(key,row,name):
 return dict(key=key,question_id=row['question_id'],image_sha256=row['image_sha256'],image_id=row['image_id'],condition=name,allowed_layers=configs()[name],input_binding=row['input_binding'],prompt_text_sha256=read(OUT/'input_bindings.json')[key]['prompt_text_sha256'],gold=row['gold'],gpu_uuid=GPU,model_revision=S['revision'],protocol_sha256=sha(OUT/'protocol.json'),source_manifest_sha256=sha(OUT/'source_manifest.json'),sample_manifest_sha256=sha(OUT/'sample_manifest.json'))
def main():
 lock=(RUN/'manager.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (RUN/'checkpoints').mkdir(exist_ok=True);verify();g=gpu()
 model=load_model();processor=AutoProcessor.from_pretrained(S['model'],local_files_only=True)
 started=time.time()
 with torch.inference_mode():
  if not (OUT/'smoke_summary.json').exists():smoke.run(model,processor)
  summary=read(OUT/'smoke_summary.json');assert summary['status']=='PASS'
  db=connect();evaler=scorer();rows=read(OUT/'sample_manifest.json');cs=configs()
  blocked={r['image_sha256']:r for r in summary['capacity_blocked']}
  done={(k,c) for k,c in db.execute('select key,condition from predictions')}
  for index,(key,row) in enumerate(rows.items()):
   verify();gpu(own=True)
   pending=[c for c in cs if (key,c) not in done]
   if not pending:continue
   inputs=first=base=rope=cache=None
   fail=blocked.get(row['image_sha256'])
   if fail is None:
    try:
     row,inputs=prepared(key)
     first,base,rope=prefill(model,inputs)
     assert all(k.shape[-2]==row['input_tokens'] for k,v in pairs(base))
    except torch.OutOfMemoryError as exc:
     fail=dict(key=key,image_sha256=row['image_sha256'],stage='native_shared_prefill',status='BLOCKED_OOM',error=str(exc),traceback=traceback.format_exc())
     blocked[row['image_sha256']]=fail
     inputs=first=base=rope=None;gc.collect();torch.cuda.empty_cache()
   names=pending[index%len(pending):]+pending[:index%len(pending)] if pending else []
   for name in names:
    record=base_record(key,row,name)
    if fail is not None:
     record.update(status='BLOCKED_OOM',generation_executed=False,score=None,blocked_evidence=fail)
    else:
     try:
      permission=DecodeImagePermission(model,row['image_positions'],cs[name])
      with permission:
       cache=clone_cache(model,base)
       got,cache,_=decode(model,inputs,first,cache,rope,permission,processor)
       kv=layout(cache,row['image_tokens'])
       assert all(r['key_tokens']==row['input_tokens']+got['decode_steps'] for r in kv)
       record.update(got,status='COMPLETE',generation_executed=True,kv_layout=kv,prefill_shared=True,cache_independent=True)
       record['score']=score(evaler,dict(gold=row['gold'],prediction=got['text']))
      cache=None
     except torch.OutOfMemoryError as exc:
      fail=dict(key=key,image_sha256=row['image_sha256'],condition=name,stage='decode_or_cache_clone',status='BLOCKED_OOM',error=str(exc),traceback=traceback.format_exc())
      blocked[row['image_sha256']]=fail
      record.update(status='BLOCKED_OOM',generation_executed=True,score=None,blocked_evidence=fail)
      cache=None;gc.collect();torch.cuda.empty_cache()
    store(db,key,name,record);done.add((key,name))
   inputs=first=base=rope=cache=None;gc.collect();torch.cuda.empty_cache()
   counts=dict(db.execute('select json_extract(record,"$.status"),count(*) from predictions group by 1'))
   progress=dict(status='RUNNING',questions_processed=index+1,questions_planned=500,condition_records=len(done),planned_condition_records=15000,states=counts,elapsed_seconds=time.time()-started,last_key=key,GPU=GPU,blocked_images=list(blocked),updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
   write(OUT/'progress.json',progress)
   print('PROGRESS',json.dumps(progress),flush=True)
  assert db.execute('select count(*) from predictions').fetchone()[0]==15000
  db.execute('pragma wal_checkpoint(TRUNCATE)');db.close()
 write(OUT/'generation_complete.json',dict(status='GENERATION_COMPLETE_WITH_BLOCKED_RETAINED',elapsed_seconds=time.time()-started,all500_manifest_questions_attempted=True,total_condition_records=15000,GPU=GPU))
 del model;gc.collect();torch.cuda.empty_cache()
 print('GENERATION_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  write(OUT/'failure.json',dict(status='FAILED_HARD_STOP',error=str(exc),traceback=traceback.format_exc(),time=time.time()))
  raise
