import os,sys,json,pathlib,hashlib,subprocess,datetime,traceback,time,gc
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from vlm_research.qwen3vl_early_windows.experiment import prepare,MODEL,hook_counts
from .engine import generate,CONDITIONS,SKIP

W=pathlib.Path(__file__).resolve().parents[2]
O=W/'vlm_research/results/qwen3vl_c3v2_physical_transfer'
P=pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_physical_transfer')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n');os.replace(tmp,p)
def deadline():
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(read(O/'time_budget.json')['GPU_cutoff_UTC'])
 assert not (O/'STOP.json').exists(),'Other worker validation failed'
def score(s,text):
 if s['dataset']=='GQA':return float(text.rstrip('.').lower()==s['answer'])
 from .official_vqa_eval import VQAEval
 from types import SimpleNamespace
 evaluator=VQAEval(SimpleNamespace(getQuesIds=lambda:[]),None)
 answer=text.replace('\n',' ').replace('\t',' ').strip()
 gold=[x.replace('\n',' ').replace('\t',' ').strip() for x in s['answers']]
 # Pinned ATE VQAv2 submission normalization: same official helpers on both.
 answer=evaluator.processDigitArticle(evaluator.processPunctuation(answer))
 gold=[evaluator.processDigitArticle(evaluator.processPunctuation(x)) for x in gold]
 return sum(min(1.,sum(a==answer for j,a in enumerate(gold) if j!=i)/3.) for i in range(10))/10
def getinput(processor,s,spec):
 inputs,layout=prepare(processor,s)
 assert layout==spec,'Input mismatch'
 return inputs,layout
def compare(a,b):
 assert len(a)==len(b),'Different generated length'
 rows=[]
 for i,(x,y) in enumerate(zip(a,b)):
  delta=(x-y).abs();r=dict(step=i,max_abs=float(delta.max()),mean_abs=float(delta.mean()),top1_equal=int(x.argmax())==int(y.argmax()))
  assert r['max_abs']<=.5 and r['mean_abs']<=.02 and r['top1_equal'],r
  rows.append(r)
 return rows
def main(rank,phase):
 torch.set_num_threads(1)
 protocol=read(O/'protocol.json');bindings=read(O/'input_bindings.json');manifest=read(O/'sample_manifest.json');selection=read(O/'execution_selection.json')
 allrows=manifest['samples']+manifest['diagnostic_only'];lookup={s['id']:s for s in allrows}
 frozen=read(O/'source_manifest.json')
 for name,h in frozen['files'].items():assert sha(W/name)==h,name
 assert sha(O/'sample_manifest.json')==protocol['data_manifest_sha256']
 atomic(O/f'shard{rank}/environment.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cuda_visible=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,transformers=__import__('transformers').__version__,source_commit=frozen['commit']))
 deadline()
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
 initial=hook_counts(model)
 if phase=='smoke':
  results=[]
  for key in selection['smoke']:
   deadline();s=lookup[key];inputs,layout=getinput(processor,s,bindings[key])
   native,nlog=generate(model,processor,inputs,layout,'N',True,True)
   noop,no=generate(model,processor,inputs,layout,'N',True,True,noop=True)
   assert native['token_ids']==noop['token_ids'] and all(torch.equal(a,b) for a,b in zip(nlog,no))
   # Native manual incremental decoding matches official HF generation exactly.
   with torch.inference_mode():
    hf=model.generate(**{k:v.to('cuda:0') for k,v in inputs.items()},do_sample=False,max_new_tokens=32,use_cache=True,return_dict_in_generate=True,output_scores=True)
   assert hf.sequences[0,layout['input_length']:].tolist()==native['token_ids']
   assert len(hf.scores)==len(nlog) and all(torch.equal(a,b[0].float().cpu()) for a,b in zip(nlog,hf.scores))
   del hf
   outputs={'N':native};vectors={}
   for c in ['F','P','B','C','W22-functional','A16-functional']:
    r,log=generate(model,processor,inputs,layout,c,True,True)
    outputs[c]=r;vectors[c]=log
    assert hook_counts(model)==initial,'Hook leak'
    atomic(O/f'shard{rank}/smoke/{key}__{c}.json',dict(sample=s,result=r))
   comparisons={}
   for a,b in [('F','C'),('F','B'),('W22-functional','P')]:
    assert outputs[a]['token_ids']==outputs[b]['token_ids'],(a,b,'Token mismatch',outputs[a]['token_ids'],outputs[b]['token_ids'])
    comparisons[a+'__'+b]=compare(vectors[a],vectors[b])
   for c in ['P','C']:
    layers=outputs[c]['audit']['layers']
    assert all(r['hidden_rows']==(layout['input_length'] if r['layer']<22 else layout['input_length']-layout['image_tokens']) for r in layers if r['step']==0)
    assert outputs[c]['audit']['deepstack']==[0,1,2]
   for c in ['B','C']:
    kv=outputs[c]['audit']['kv']['layers'];assert all(kv[i]['live_tokens']==0 and kv[i]['capacity_bytes']==0 for i in SKIP)
    shapes=outputs[c]['audit']['projection_shapes']
    assert all(not(r['layer'] in SKIP and r['step']>0 and r['op'] in ['q_proj','k_proj','v_proj','o_proj','QK','AV']) for r in shapes)
   results.append(dict(id=key,image_tokens=layout['image_tokens'],input_length=layout['input_length'],native_noop_equal=True,native_hf_equal=True,comparisons=comparisons,conditions={c:dict(token_ids=r['token_ids'],text=r['text'],eos=r['eos'],kv=r['audit']['kv']) for c,r in outputs.items()}))
   atomic(O/f'shard{rank}/smoke_validation.json',dict(status='RUNNING',results=results))
  # Exception path restores every patch/hook too.
  from .engine import Execution
  try:
   with Execution(model,layout,'C',audit=True).context():raise RuntimeError('injected-cleanup-test')
  except RuntimeError as error:assert str(error)=='injected-cleanup-test'
  assert hook_counts(model)==initial
  atomic(O/f'shard{rank}/smoke_validation.json',dict(status='PASS',results=results,exception_cleanup=True))
 else:
  assert all(read(O/f'shard{i}/smoke_validation.json')['status']=='PASS' for i in range(2))
  if phase=='pilot':keys=[k for k in selection['pilot'] if lookup[k]['shard']==rank]
  elif phase=='quality':keys=[s['id'] for s in manifest['samples'] if s['shard']==rank]
  elif phase=='benchmark':keys=[k for k in selection['benchmark'] if lookup[k]['shard']==rank]
  else:raise ValueError(phase)
  for j,key in enumerate(keys):
   deadline();s=lookup[key];inputs,layout=getinput(processor,s,bindings[key])
   if phase=='benchmark':
    for rep in range(8):
     order=CONDITIONS[rep%5:]+CONDITIONS[:rep%5]
     for c in order:
      path=O/f'shard{rank}/benchmark/{key}__{c}__{rep}.json'
      if path.exists():continue
      deadline();r,_=generate(model,processor,inputs,layout,c)
      atomic(path,dict(sample_id=key,dataset=s['dataset'],image_sha256=s['image_sha256'],repeat=rep,warmup=rep<3,result=r,source_commit=frozen['commit']))
    # Full execution shape audit is outside performance timing.
    for c in CONDITIONS:
     path=O/f'shard{rank}/physical_audit/{key}__{c}.json'
     if not path.exists():
      r,_=generate(model,processor,inputs,layout,c,True)
      atomic(path,dict(sample=s,result=r))
   else:
    outputs={};logs={}
    for c in CONDITIONS:
     path=O/f'shard{rank}/{phase}/{key}__{c}.json'
     if path.exists():outputs[c]=read(path)['result'];continue
     deadline();r,log=generate(model,processor,inputs,layout,c,audit=phase=='pilot',capture=phase=='pilot')
     r['score']=score(s,r['text']);outputs[c]=r;logs[c]=log
     atomic(path,dict(sample=s,result=r,source_commit=frozen['commit'],input_hash=bindings[key]['input_tensor_hashes']))
    if phase=='pilot' and logs:
     assert outputs['F']['token_ids']==outputs['C']['token_ids'],'Pilot F/C generated tokens diverge'
     compare(logs['F'],logs['C'])
   assert hook_counts(model)==initial
   print(phase,rank,j+1,len(keys),key,flush=True)
  atomic(O/f'shard{rank}/{phase}_completion.json',dict(status='PASS',samples=len(keys),at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 rank=int(sys.argv[1]);phase=sys.argv[2]
 try:main(rank,phase)
 except BaseException as error:
  detail=dict(status='FAILED',rank=rank,phase=phase,error=repr(error),traceback=traceback.format_exc(),at=datetime.datetime.now(datetime.timezone.utc).isoformat())
  atomic(O/f'shard{rank}/{phase}_failure.json',detail);atomic(O/'STOP.json',detail);raise
