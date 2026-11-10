"""Frozen 2x2 functional validation. No physical skipping, benchmarking, or GitHub push."""
import argparse,csv,gc,json,os,platform,shutil,subprocess,sys,time,traceback
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import torch,transformers
from .common import *
from . import engine as j
e=j.base
def cpu_tests(bindings):
 mask_cases=zero_cases=0
 for idx in [525,1928,1127]:
  role=bindings[idx]['roles'];n=role['n'];lo,hi=role['image'][0],role['image'][-1]+1
  original=torch.full((1,1,n,n),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16)
  original.masked_fill_(torch.ones((n,n),dtype=torch.bool).tril(),0)
  for c in CONDITIONS:
   for layer in range(28):
    off=layer not in j.writers(c);m=e.edit_mask(original,role,True,off)
    want=torch.zeros_like(original,dtype=torch.bool);want[:,:,n-1,lo:hi]=True
    if off:want[:,:,hi:n-1,lo:hi]=True
    assert torch.equal(m!=original,want) and torch.isneginf(m[want]).all() and (m==0).any(-1).all()
    assert torch.equal(m[:,:,:hi,:],original[:,:,:hi,:]);mask_cases+=1
   for prefill in [True,False]:
    qlen=n if prefill else 1;x=torch.arange(qlen*8,dtype=torch.float32).reshape(1,qlen,8).to(torch.bfloat16)
    old=x.clone();z,q=j.zero_answer(x,role,prefill)
    assert torch.equal(x,old) and torch.count_nonzero(z[:,q:q+1])==0
    assert torch.equal(x[:,:q],z[:,:q]);zero_cases+=1
   dec=torch.zeros((1,1,1,n+3),dtype=torch.bfloat16);masked=e.edit_mask(dec,role,False,False)
   assert int((masked!=dec).sum())==256 and torch.isneginf(masked[:,:,:,lo:hi]).all()
 assert j.writers('A')==j.writers('C')==list(range(28))
 assert j.writers('B')==j.writers('D')==list(range(20))
 return dict(passed=True,mask_layer_cases=mask_cases,zero_output_cases=zero_cases,original_inputs_unchanged=True,
  non_target_prefill_exact=True,no_layer_search=True,zero_layers=list(range(2,6)))
def baseline_csv(p,condition):
 result={}
 with (p/'per_question.csv').open() as f:
  for r in csv.DictReader(f):
   if r['condition']==condition:
    k=r['task']+'__'+r['question_id']+'.json';assert k not in result;result[k]=r
 assert len(result)==3000
 return result
def prepare():
 torch.set_num_threads(4)
 assert not OUT.exists() and not RUN.exists(),'Refusing to overwrite an existing experiment'
 assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
 assert read(P,'validation.json')['passed'] and read(B,'validation.json')['passed']
 for p,h in read(P,'source_manifest.json').items():assert sha(p)==h,p
 assert sha(ROOT/'vlm_research/internvl3_visual_writing_figure3_3000/engine.py')==sha(OLD/'vlm_research/internvl3_visual_writing_figure3_3000/engine.py')
 gpu=command('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')
 assert not command('nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader')
 assert read(P,'protocol.json')['revision']==read(B,'protocol.json')['revision']==REV
 assert read(P,'protocol.json')['gpu_uuid']==read(B,'protocol.json')['GPU_UUID']==gpu
 assert sha(P/'sample_manifest.json')==sha(B/'sample_manifest.json')
 samples=read(P,'sample_manifest.json')['samples'];bindings=read(P,'cpu_input_bindings.json')
 assert len(samples)==len(bindings)==3000 and len({s['image_sha256'] for s in samples})==2409
 assert Counter(s['task'] for s in samples)==Counter({t:500 for t in TASKS})
 ha=read(P,'checkpoint_hashes.json');hb=read(B,'checkpoint_hashes.json')
 ca=baseline_csv(P,'All-Writer');cb=baseline_csv(B,'Keep-L0-L19')
 sums=dict(A=0,B=0)
 for idx,s in enumerate(samples):
  a,b=old_records(s,ha,hb);k=key(s);bind=bindings[idx]
  assert a['GPU_UUID']==b['GPU_UUID']==gpu and a['protocol_sha256']==sha(P/'protocol.json') and b['protocol_sha256']==sha(B/'protocol.json')
  assert a['input']['prompt']==b['input']['prompt']==bind['prompt']
  assert a['input']['input_ids']==b['input']['input_ids']==bind['input_ids']
  assert a['input']['roles']==b['input']['roles']==bind['roles']
  assert a['input']==b['input'] and e.digest(a['input'])==a['input_binding']
  assert Path(s['actual_image_path']).is_file()
  for c,row,ref in [('A',a['conditions']['All-Writer'],ca[k]),('B',b['generation'],cb[k])]:
   assert row['prediction']==row['text'].rstrip('.').lower()
   assert score(row['text'],s['answer'])==int(row['correct'])==int(ref['correct'])
   assert row['text']==ref['text'] and row['prediction']==ref['prediction'] and row['token_ids']==json.loads(ref['token_ids'])
   assert ref['image_sha256']==s['image_sha256'] and ref['gold']==s['answer'] and ref['input_binding']==a['input_binding']
   assert row['writer_layers']==j.writers(c) and row['P1_Decode_image_off'] and row['independent_fresh_cache']
   assert row['mask_audit']['blocked_probability_max']==0 and row['final_cache_lengths']==[bind['roles']['n']+len(row['token_ids'])-1]*28
   sums[c]+=int(row['correct'])
 tests=cpu_tests(bindings)
 OUT.mkdir(parents=True);RUN.mkdir();(RUN/'checkpoints').mkdir();(OUT/'checkpoints').mkdir()
 for name in ['sample_manifest.json','cpu_input_bindings.json']:shutil.copy2(P/name,RUN/name)
 shutil.copy2(P/'sample_manifest.json',OUT/'sample_manifest.json')
 save(OUT,'cpu_tests.json',tests);save(RUN,'cpu_tests.json',tests)
 save(OUT,'baseline_reuse_audit.json',dict(passed=True,questions=3000,image_clusters=2409,correct_counts=sums,
  all_CSV_Checkpoint_scores_tokens_match=True,all_checkpoint_SHA_match=True,all_prompt_image_gold_order_match=True,
  input_bindings_identical=True,same_GPU_UUID=gpu,same_revision=REV,complete_answers=True,no_new_image_SHA_scan=True,
  A_directory=str(P),B_directory=str(B),source_GitHub_commit='1cdcf6b2a0b6b5606b00789961904043849cbbdc'))
 save(RUN,'baseline_hashes.json',dict(A=ha,B=hb));shutil.copy2(RUN/'baseline_hashes.json',OUT/'baseline_hashes.json')
 sources=dict(read(P,'source_manifest.json'))
 for p in Path(__file__).parent.glob('*.py'):sources[str(p.resolve())]=sha(p)
 sources[str(ROOT/'vlm_research/internvl3_visual_writing_figure3_3000/engine.py')]=sha(ROOT/'vlm_research/internvl3_visual_writing_figure3_3000/engine.py')
 save(OUT,'source_manifest.json',sources);shutil.copy2(OUT/'source_manifest.json',RUN/'source_manifest.json')
 protocol=dict(experiment='InternVL3 SVH 2x2 functional joint validation',model='OpenGVLab/InternVL3-8B',revision=REV,
  GPU_UUID=gpu,torch='2.6.0+cu124',transformers='4.49.0',dtype='BF16',attention='eager',batch_size=1,
  greedy=True,max_new_tokens=32,normal_EOS=True,logits_to_keep=1,max_num=1,visual_tokens=256,use_cache=True,
  questions=3000,image_clusters=2409,task_questions={t:500 for t in TASKS},
  conditions={c:dict(R_to_image_layers=j.writers(c),answer_o_proj_output_zero_layers=list(range(2,6)) if c in ('C','D') else []) for c in CONDITIONS},
  P_minus_1_and_all_decode_to_image='OFF in all 28 layers',image_queries='native',
  R_definition='Non-image queries after the last image token and strictly before original P-1, including template tokens',
  functional_output_zero=True,physical_attention_skip=False,physical_Hidden_pruning=False,KV_compression=False,
  all_norm_MLP_residual_and_dense_cache_native=True,attention_zero_point='self_attn.o_proj output, only P-1 and true Decode rows',
  baseline_A=str(P),baseline_B=str(B),baseline_A_protocol_sha256=sha(P/'protocol.json'),baseline_B_protocol_sha256=sha(B/'protocol.json'),
  baseline_A_predictions_sha256=sha(P/'per_question.csv'),baseline_B_predictions_sha256=sha(B/'per_question.csv'),
  sample_manifest_sha256=sha(P/'sample_manifest.json'),source_manifest_sha256=sha(OUT/'source_manifest.json'),
  planned_new_generations=6000,baseline_replay_policy='Reuse A/B iff three same-input complete-token and all effective-logit replay checks match. Otherwise record mismatch and rerun A/B on this GPU; never mix numeric protocols.',
  input_mismatch_policy='Stop on prompt/image/gold/role mismatch',
  comparisons=['B-A','C-A','D-B','D-A'],interaction='(D-B)-(C-A)',
  bootstrap='20000 whole-image-SHA cluster paired resamples; question weighted ordinary 95% intervals',bootstrap_seed=20261110,
  multiplicity='Exploratory, five contrasts; ordinary CIs unadjusted, no confirmatory noninferiority claim',
  quality_reference_pp=-1.0,L20_candidate_post_hoc=True,no_layer_search=True,no_training=True,no_quality_benchmarks=True,no_E2E_measurement=True,
  no_GitHub_push=True,independent_worktree=str(ROOT))
 save(OUT,'protocol.json',protocol);shutil.copy2(OUT/'protocol.json',RUN/'protocol.json')
 save(OUT,'recovery.json',dict(runtime=str(RUN),persistent=str(OUT),resume_command=PY+' -u -m vlm_research.internvl3_svh_joint_3000.run supervise',protocol_sha256=sha(OUT/'protocol.json')))
 print('JOINT_CPU_PASS',json.dumps(dict(baseline_correct=sums,tests=tests,protocol_sha256=sha(OUT/'protocol.json'))),flush=True)
def smoke(model,tok,loader,samples,bindings,hashes):
 checks=[];replay=True
 for idx in [525,1928,1127]:
  s=samples[idx];a,b=old_records(s,hashes['A'],hashes['B']);prep=e.prepare(model,tok,loader,s)
  assert prep['input_binding']==a['input_binding']==b['input_binding']
  assert prep['prompt']==bindings[idx]['prompt'] and prep['binding']['roles']==bindings[idx]['roles']
  native,ns,_=j.generate(model,tok,prep,native=True)
  with e.OfficialChatCapture(model) as official:
   answer=model.chat(tok,prep['pixels'],prep['request'],dict(do_sample=False,num_beams=1,max_new_tokens=32,
    logits_to_keep=1,pad_token_id=tok.convert_tokens_to_ids(model.conv_template.sep.strip())))
  assert official.ids==native['token_ids'] and len(ns)==len(official.scores) and all(torch.equal(x,y) for x,y in zip(ns,official.scores))
  assert answer.strip()==native['text'] and official.input_ids[0].tolist()==prep['binding']['input_ids']
  assert torch.equal(official.embeddings,prep['inputs']['inputs_embeds'].cpu())
  noop,scores,_=j.generate(model,tok,prep,noop=True,audit=True,trace=True)
  assert noop['token_ids']==native['token_ids'] and len(ns)==len(scores) and all(torch.equal(x,y) for x,y in zip(ns,scores))
  outputs={};traces={}
  for c in CONDITIONS:
   row,sc,tr=j.generate(model,tok,prep,c,audit=True,trace=True)
   assert not row['empty'] and row['mask_audit']['real_decode']>=1
   assert row['mask_audit']['blocked_probability_max']==0 and row['independent_fresh_cache']
   assert row['answer_attention_output_zero_layers']==(list(range(2,6)) if c in ('C','D') else [])
   outputs[c]=row;traces[c]=tr
  for x,y in [('A','C'),('B','D')]:assert traces[x]['prefix']==traces[y]['prefix'],'Non-answer Prefill output changed'
  matches={}
  for c,old in [('A',a['conditions']['All-Writer']),('B',b['generation'])]:
   matches[c]=dict(tokens_equal=outputs[c]['token_ids']==old['token_ids'],all_logits_sha_equal=outputs[c]['all_step_logits_sha256']==old['all_step_logits_sha256'])
   replay=replay and all(matches[c].values())
  footprint=e.hooks_count(model)
  from transformers.cache_utils import DynamicCache
  try:
   with patch.object(e,'writers',j.writers):
    with j.JointMask(model,prep['roles'],'D',DynamicCache()):raise RuntimeError('joint-cleanup-test')
  except RuntimeError as err:assert str(err)=='joint-cleanup-test'
  assert e.hooks_count(model)==footprint
  checks.append(dict(index=idx,question_id=s['question_id'],input_binding=prep['input_binding'],native_noop_official_all_logits_torch_equal=True,
   baseline_replay=matches,all_non_answer_prefill_outputs_exact_A_vs_C_B_vs_D=True,
   audits={c:outputs[c]['mask_audit'] for c in CONDITIONS},exception_hooks_clean=True))
  print('JOINT_SMOKE_PASS',idx,'REPLAY',matches,flush=True)
 save(RUN,'smoke_summary.json',dict(passed=True,samples=checks,all_baseline_replays_exact=replay,protocol_sha256=sha(RUN/'protocol.json')))
 decision=dict(reuse_A_B=replay,formal_conditions=['C','D'] if replay else CONDITIONS,target_new_requests=6000 if replay else 12000,
  reason='All same-GPU input, complete-token and effective-logit replays matched' if replay else 'Numerical replay mismatch; A/B will be regenerated on the current GPU',GPU_UUID=read(RUN,'protocol.json')['GPU_UUID'])
 prior=RUN/'baseline_reuse_decision.json'
 if prior.exists():assert read(RUN,prior.name)==decision,'Resume baseline decision changed'
 else:save(RUN,prior.name,decision)
 return decision
@torch.no_grad()
def run():
 p=read(RUN,'protocol.json');ph=sha(RUN/'protocol.json');model=None;started=time.time()
 assert torch.__version__==p['torch'] and transformers.__version__==p['transformers']
 assert sha(RUN/'source_manifest.json')==p['source_manifest_sha256']
 for src,h in read(RUN,'source_manifest.json').items():assert sha(src)==h,src
 assert command('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')==p['GPU_UUID']
 assert not command('nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader')
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 try:
  save(RUN,'environment.json',dict(python=platform.python_version(),python_executable=sys.executable,
   torch=torch.__version__,torch_import=torch.__file__,transformers=transformers.__version__,transformers_import=transformers.__file__,
   cuda=torch.version.cuda,GPU=command('nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader'),
   source_manifest_sha256=sha(RUN/'source_manifest.json'),model_revision=REV,no_system_or_power_changes=True))
  from transformers import AutoModel,AutoTokenizer
  model=AutoModel.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,
   low_cpu_mem_usage=True,use_flash_attn=False,device_map={'':'cuda:0'},use_safetensors=True).eval()
  assert len(model.language_model.model.layers)==28 and model.num_image_token==256
  assert model.language_model.config._attn_implementation=='eager'
  assert {x.dtype for x in model.parameters()}=={torch.bfloat16} and {str(x.device) for x in model.parameters()}=={'cuda:0'}
  assert not any(x.attn.use_flash_attn for x in model.vision_model.encoder.layers)
  tok=AutoTokenizer.from_pretrained(MODEL,use_fast=False,local_files_only=True);loader=e.official_preprocessor(MODEL)
  samples=read(RUN,'sample_manifest.json')['samples'];bindings=read(RUN,'cpu_input_bindings.json');hashes=read(RUN,'baseline_hashes.json')
  versions=[(x._version,x.data_ptr()) for x in model.parameters()];footprint=e.hooks_count(model)
  decision=smoke(model,tok,loader,samples,bindings,hashes);dh=sha(RUN/'baseline_reuse_decision.json')
  count=new=0;formal_start=time.time()
  for idx,s in enumerate(samples):
   k=key(s);dest=RUN/'checkpoints'/k;a,b=old_records(s,hashes['A'],hashes['B'])
   if dest.exists():
    record=read(RUN/'checkpoints',k)
    assert record['protocol_sha256']==ph and record['sample_sha256']==digest(s) and record['GPU_UUID']==p['GPU_UUID'] and record['decision_sha256']==dh
    assert record['index']==idx and record['parent_checkpoint_SHA256']==dict(A=hashes['A'][k],B=hashes['B'][k])
   else:
    record=dict(index=idx,task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],gold=s['answer'],
     protocol_sha256=ph,sample_sha256=digest(s),GPU_UUID=p['GPU_UUID'],decision_sha256=dh,
     parent_checkpoint_SHA256=dict(A=hashes['A'][k],B=hashes['B'][k]),conditions={},complete=False)
    if decision['reuse_A_B']:
     for c,row in [('A',a['conditions']['All-Writer']),('B',b['generation'])]:
      record['conditions'][c]=dict(row,condition=c,reused=True,source_checkpoint_sha256=record['parent_checkpoint_SHA256'][c],
       answer_attention_output_zero_layers=[])
   missing=[c for c in decision['formal_conditions'] if c not in record['conditions']]
   if missing:
    prep=e.prepare(model,tok,loader,s)
    assert prep['input_binding']==a['input_binding']==b['input_binding']
    assert prep['binding']['prompt']==bindings[idx]['prompt'] and prep['binding']['input_ids']==bindings[idx]['input_ids'] and prep['roles']==bindings[idx]['roles']
    record.update(input_binding=prep['input_binding'],input=prep['binding'])
    for c in missing:
     row,sc,_=j.generate(model,tok,prep,c)
     assert row['mask_audit']['blocked_probability_max']==0
     row.update(correct=score(row['text'],s['answer']),reused=False)
     record['conditions'][c]=row;record['complete']=set(record['conditions'])==set(CONDITIONS)
     save(RUN/'checkpoints',k,record);new+=1;del sc
    del prep
   assert record['complete'] and set(record['conditions'])==set(CONDITIONS)
   count+=len(decision['formal_conditions']);del a,b,record
   if (idx+1)%5==0 or idx==0:
    assert versions==[(x._version,x.data_ptr()) for x in model.parameters()] and e.hooks_count(model)==footprint
    assert command('nvidia-smi','--query-gpu=uuid','--format=csv,noheader')==p['GPU_UUID']
    progress=dict(status='RUNNING',questions_complete=idx+1,questions_target=3000,formal_new_requests_complete=count,
     formal_new_requests_target=decision['target_new_requests'],new_this_process=new,reused_baseline_rows=2*(idx+1) if decision['reuse_A_B'] else 0,
     seconds=time.time()-formal_start,GPU_UUID=p['GPU_UUID'])
    save(RUN,'progress.json',progress);print('JOINT_PROGRESS',json.dumps(progress),flush=True)
  assert count==decision['target_new_requests'] and e.hooks_count(model)==footprint
  assert versions==[(x._version,x.data_ptr()) for x in model.parameters()]
  save(RUN,'generation_complete.json',dict(passed=True,questions=3000,new_requests=count,total_analysis_rows=12000,
   baseline_A_B_reused=decision['reuse_A_B'],GPU_UUID=p['GPU_UUID'],protocol_sha256=ph,
   weights_unchanged=True,hooks_clean=True,no_nan_inf_oom=True,formal_seconds=time.time()-formal_start))
 except BaseException:
  save(RUN,'failure.json',dict(status='STOPPED',traceback=traceback.format_exc(),protocol_sha256=ph,pid=os.getpid()));raise
 finally:
  if model is not None:del model
  gc.collect();torch.cuda.empty_cache();save(RUN,'gpu_release.json',dict(worker_finished=True,pid=os.getpid(),elapsed_seconds=time.time()-started))
def supervise():
 RUN.mkdir(exist_ok=True)
 proc=subprocess.Popen([PY,'-u','-m','vlm_research.internvl3_svh_joint_3000.run','run'],cwd=ROOT,
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false'),stdout=(RUN/'worker.log').open('ab'),
  stderr=subprocess.STDOUT,start_new_session=True)
 save(RUN,'active.json',dict(worker=proc.pid,supervisor=os.getpid(),runtime=str(RUN),persistent=str(OUT)));seen={}
 while True:
  alive=proc.poll() is None
  for src in (RUN/'checkpoints').glob('*.json'):
   st=src.stat();mark=(st.st_mtime_ns,st.st_size)
   if seen.get(src.name)==mark:continue
   raw=src.read_bytes();dest=OUT/'checkpoints'/src.name;tmp=dest.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(dest)
   assert sha(dest)==hashlib.sha256(raw).hexdigest();seen[src.name]=mark
  for src in RUN.glob('*.json'):
   if src.name in ('sample_manifest.json','cpu_input_bindings.json'):continue
   dest=OUT/src.name
   if not dest.exists() or sha(dest)!=sha(src):shutil.copy2(src,dest)
  save(RUN,'mirror_progress.json',dict(files=len(seen),worker_alive=alive,updated=time.time()))
  if not alive:break
  time.sleep(10)
 shutil.copy2(RUN/'worker.log',OUT/'worker.log')
 assert proc.returncode==0 and read(RUN,'generation_complete.json')['passed'],'Worker failed; diagnostic evidence retained'
 hashes={p.name:sha(p) for p in (RUN/'checkpoints').glob('*.json')}
 assert len(hashes)==3000 and all(sha(OUT/'checkpoints'/n)==h for n,h in hashes.items())
 save(OUT,'persistence_complete.json',dict(passed=True,files=3000,SHA256=hashes))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 subprocess.run([PY,'-u','-m','vlm_research.internvl3_svh_joint_3000.analyze'],cwd=ROOT,env=env,check=True)
 print('JOINT_ALL_COMPLETE_REVIEW_PENDING',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','supervise']);globals()[ap.parse_args().mode]()
