"""One model load: three Native checks and A/B/C on 200 frozen questions."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',
 CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import sys,json,hashlib,subprocess,gc,time,traceback
import torch,transformers
from transformers import LlamaTokenizer
from protocol_helpers import O,D,M,official_functions,configure,make_inputs
R=Path('/home/featurize/work/ATE');sys.path.insert(0,str(R))
from vlm_research.information_horizon_alignment import engine as frozen
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.gqa_testdev_c3 import engine as gqa_engine,common
from vlm_research.physical_handoff_fastpath.engine import Layout
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(n,v):
 p=O/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def cmd(*a):return subprocess.check_output(a,cwd=R,text=True).strip()
def layout(x):
 ids=x['input_ids'][0].tolist();image=tuple(i for i,v in enumerate(ids) if v==32000);text=tuple(i for i,v in enumerate(ids) if v!=32000)
 assert len(image)==576;relay=tuple(range(image[-1]+1,len(ids)-1));return Layout(len(ids),image,text,relay,relay,20)
def hashes(x):return {k:tensor_hash(v.to(torch.float16) if v.is_floating_point() else v) for k,v in x.items()}

def main():
 cpu=read(O/'cpu_audit_summary.json');p=read(O/'gpu_sample_plan.json');samples=p['samples'];lookup={s['question_id']:s for s in samples}
 assert cpu['status']=='PASS' and not cpu['CUDA_initialized'] and (cpu['images_with_crop_pad_pixel_difference']>0)
 assert len(samples)==len(lookup)==200 and len(p['smoke_ids'])==3
 assert sha(O/'protocol_helpers.py')==p['helper_sha256'] and sha(R/'vlm_research/gqa_testdev_c3/engine.py')==p['frozen_gqa_engine_sha256']
 for n,h in p['sources'].items():assert sha(O/n)==h
 assert cmd('git','rev-parse','HEAD')=='e281754ef17b3accc09d6938b5b9f020048b2aec' and not cmd('git','status','--porcelain','--untracked-files=no')
 assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0' and M.resolve()==Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
 assert not (O/'gpu_progress.json').exists(),'Do not re-generate a started cohort'
 state={'status':'RUNNING','started_at':now(),'unique_fixed_questions':200,'expected_control_generations':600,'completed_control_generations':0,
  'native_smoke':[],'model_loads':0,'sample_plan_sha256':sha(O/'gpu_sample_plan.json'),'cpu_audit_sha256':sha(O/'cpu_audit_summary.json'),
  'dtype':'float16','attention':'eager','backend':'Unmodified HF Native; frozen gqa_engine.generate(label=Native)',
  'max_new_tokens':32,'EOS':2,'greedy':True,'batch_size':1,'conditions':p['conditions'],'no_C3_execution':True,
  'code_commit':cmd('git','rev-parse','HEAD'),'model_revision':p['model_revision']}
 adapter=None;vision_calls=[0];hook=None;phase='GPU_LOAD';started=time.monotonic()
 try:
  assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader')
  assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4090'
  assert torch.cuda.mem_get_info()[0]>20*2**30
  state['gpu_before_load']=cmd('nvidia-smi','--query-gpu=name,uuid,driver_version,memory.free','--format=csv,noheader')
  torch.cuda.reset_peak_memory_stats();loader.MODELS=dict(loader.MODELS,llava=M);state['model_loads']=1;save('gpu_progress.json',state)
  print('LOADING_ONE_NATIVE_FP16_EAGER_MODEL',flush=True);adapter=frozen.load('B')
  assert {str(v.device) for v in adapter.model.parameters()}=={'cuda:0'} and {v.dtype for v in adapter.model.parameters()}=={torch.float16}
  assert adapter.model.config.text_config._attn_implementation==adapter.model.config.vision_config._attn_implementation=='eager'
  def vision(module,args,output):
   assert output.last_hidden_state.device.type=='cuda';vision_calls[0]+=1
  hook=adapter.model.vision_tower.register_forward_hook(vision)
  conv,functions=official_functions();tokenizer=LlamaTokenizer.from_pretrained(D/'audit_support/gqa_native_protocol/official_tokenizer',local_files_only=True,legacy=True)
  phase='NATIVE_SMOKE'
  for q in p['smoke_ids']:
   s=lookup[q];ref=p['historical_native'][q];x,prompt=make_inputs(s,'HISTORICAL',adapter.processor,tokenizer,conv,functions)
   assert hashes(x)==ref['input_binding']['input_hashes']
   assert common.digest(x['input_ids'][0].tolist())==ref['input_binding']['prompt_ids_sha256']
   a_layout=layout(x);adapter.inputs={k:v.to('cuda:0',dtype=torch.float16) if v.is_floating_point() else v.to('cuda:0') for k,v in x.items()}
   before=vision_calls[0];generation,*_=gqa_engine.generate(adapter,a_layout,'Native',32)
   assert vision_calls[0]==before+1
   match=generation['token_ids']==ref['token_ids'] and generation['text']==ref['text']
   record={'question_id':q,'structural':s['task'],'generation':generation,'historical_text':ref['text'],'historical_token_ids':ref['token_ids'],'input_tensors_exact':True,'tokens_and_text_exact':match,'vision_forward_calls':1}
   state['native_smoke'].append(record);save('gpu_progress.json',state);print('NATIVE_SMOKE',q,repr(generation['text']),'HISTORICAL_MATCH',match,flush=True)
   assert match,'Native hardware smoke differs from archived baseline; stop targeted comparisons to avoid a confounded result'
   adapter.inputs={}
  phase='ABC_CONTROLS'
  with (O/'gpu_predictions.jsonl').open('x') as file:
   for i,s in enumerate(samples):
    q=s['question_id'];ref=p['historical_native'][q]
    base,_=make_inputs(s,'HISTORICAL',adapter.processor,tokenizer,conv,functions);assert hashes(base)==ref['input_binding']['input_hashes']
    variants={c:make_inputs(s,c,adapter.processor,tokenizer,conv,functions) for c in ['A','B','C']}
    assert torch.equal(variants['A'][0]['pixel_values'],base['pixel_values'])
    assert torch.equal(variants['B'][0]['input_ids'],base['input_ids'])
    assert torch.equal(variants['C'][0]['input_ids'],variants['A'][0]['input_ids']) and torch.equal(variants['C'][0]['pixel_values'],variants['B'][0]['pixel_values'])
    for condition,(x,prompt) in variants.items():
     a_layout=layout(x);binding=hashes(x);adapter.inputs={k:v.to('cuda:0',dtype=torch.float16) if v.is_floating_point() else v.to('cuda:0') for k,v in x.items()}
     before=vision_calls[0];generation,*_=gqa_engine.generate(adapter,a_layout,'Native',32)
     assert vision_calls[0]==before+1 and generation['all_effective_logits_finite']
     parsed=common.prediction(generation['text']);score=common.score(s,generation['text'])
     r={'question_id':q,'condition':condition,'image_id':s['image_id'],'structural':s['task'],'semantic':s['types']['semantic'],'question':s['question'],'gold':s['answer'],
      'generation':generation,'parsed':parsed,'score':score,'historical_native_text':ref['text'],'historical_native_parsed':ref['parsed'],'historical_native_score':ref['score'],
      'historical_native_tokens':ref['token_ids'],'delta_correct':score-ref['score'],'answer_changed':parsed!=ref['parsed'],'prompt_sha256':common.digest(prompt),
      'input_hashes':binding,'expanded_length':len(x['input_ids'][0]),'sample_plan_sha256':state['sample_plan_sha256'],'vision_forward_calls':1}
     file.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n');file.flush();os.fsync(file.fileno())
     state['completed_control_generations']+=1;save('gpu_progress.json',state);adapter.inputs={}
    if (i+1)%20==0:print('ABC_PROGRESS',i+1,'/200 questions',state['completed_control_generations'],'/600 generations',round(time.monotonic()-started,1),'seconds',flush=True)
  assert state['completed_control_generations']==600 and vision_calls[0]==603
  state['status']='COMPLETE'
 except BaseException as exc:
  state['status']='BLOCKED';state['failure']={'phase':phase,'error':repr(exc),'traceback':traceback.format_exc()[-8000:]};print('STOPPED',traceback.format_exc(),flush=True)
 finally:
  if hook:hook.remove()
  state['vision_forward_calls_total']=vision_calls[0]
  if torch.cuda.is_initialized():
   state['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;state['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
   adapter=None;gc.collect();torch.cuda.empty_cache();state['allocated_mib_before_process_exit']=torch.cuda.memory_allocated()/2**20
  state['finished_at']=now();state['elapsed_seconds']=time.monotonic()-started;save('gpu_progress.json',state)
  print('GPU_FINAL',state['status'],state['completed_control_generations'],'/600',flush=True)
 if state['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
