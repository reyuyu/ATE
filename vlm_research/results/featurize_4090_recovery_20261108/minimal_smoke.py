"""Recovery only: three frozen cases, one model load, no formal cohorts."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import sys,json,hashlib,gzip,gc,traceback,subprocess,time
from datetime import datetime,timezone
from dataclasses import replace
O=Path(__file__).resolve().parent;R=O.parents[2];M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
sys.path[:0]=[str(R),str(O)]
import torch,transformers
from transformers import LlamaTokenizer
from vlm_research.information_horizon_alignment import engine as frozen,score as scorer
from vlm_research.mme_seed_quality import engine as loader
from vlm_research.gqa_testdev_c3 import engine,common
from vlm_research.fastp2_bypass_writer.run import operators
from vlm_research.physical_visual_handoff.engine import equivalence
from vlm_research.physical_visual_handoff.common import TOLERANCES
from vlm_research.information_horizon_alignment.common import tensor_sha,digest as paper_digest
scorer.AUTHOR=Path('/home/featurize/data/ate-recovery/sources/Information-Horizon')
from protocol_helpers import official_functions,make_inputs
from smoke_helpers import hashes,layout_from,structural_trace,digest

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 p=O/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def concise(g):return {k:v for k,v in g.items() if k not in ['attention_audit','visual_audit','kv_audit'] and not k.startswith('_')}

def main():
 assert transformers.__version__=='4.49.0' and torch.__version__=='2.6.0+cu124'
 for p,h in read(O/'frozen_sources.json').items():assert sha(R/p)==h,p
 idle=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not idle,('GPU busy',idle)
 cases=read(O/'smoke_cases.json');samples=[('GQA-official-aligned-vicuna-pad',cases['gqa_official'],None)]+[(c['sample']['benchmark']+'-paper',c['sample'],c['reference_3090']) for c in cases['paper']]
 for protocol,s,ref in samples:
  assert Path(s['image_path']).is_file(),s['image_path'];assert sha(s['image_path'])==s['image_sha256']
  view='GQA-testdev-balanced' if protocol.startswith('GQA') else s['input_view'];root=Path('/home/featurize/data/ate-recovery/normalized')/view
  gold=read(root/'answers.json');qid=s['question_id'] if protocol.startswith('GQA') else s['normalized_question_id'];assert gold[qid]==s['answer']
  # Bind restored row, original question and image to the selected frozen case.
  row=next(r for r in map(json.loads,(root/'inputs.jsonl').read_text().splitlines()) if r['question_id']==qid)
  assert row['question']==s['question'] and row['image_path']==s['image_path']
  if not protocol.startswith('GQA'):assert row['prompt']==s['prompt'] and row['user_text']==s['user_text']
 result={'status':'RUNNING','started_at':datetime.now(timezone.utc).isoformat(),'model_loads':0,'scope':'three frozen samples only; no formal accuracy','strict_thresholds':TOLERANCES['llava'],'checks':[],'generations':[],'historical_warning':{'commit':'d6157216199f6e2bb6c03557c05317649fe9c7d6','POPE_hidden_relative_rmse':0.010154514573514462,'strict_threshold':0.01,'historical_status':'FAIL, preserved'}}
 save('physical_smoke_summary.json',result)
 a=None
 try:
  loader.MODELS['llava']=M;a=frozen.load('B');result['model_loads']=1
  assert len(a.layers)==32 and a.model.config.text_config._attn_implementation=='eager'
  assert all(p.device.type=='cuda' and p.dtype==torch.float16 for p in a.model.parameters())
  conv,functions=official_functions();tok=LlamaTokenizer.from_pretrained(M,local_files_only=True,legacy=True)
  torch.cuda.reset_peak_memory_stats();full_audits=[]
  for idx,(protocol,s,ref) in enumerate(samples):
   if protocol.startswith('GQA'):
    x,prompt=make_inputs(s,'C',a.processor,tok,conv,functions)
    assert prompt==s['prompt'] and x['input_ids'][0].tolist()==s['input_ids'] and hashes(x)==s['input_hashes']
    a.inputs={k:v.to('cuda:0',dtype=torch.float16) if v.is_floating_point() else v.to('cuda:0') for k,v in x.items()}
    l=layout_from(s['input_ids'],s['question_positions']);cap=32
   else:
    l,b=frozen.prepare(a,s);assert sha(s['image_path'])==ref['image_sha256']
    assert b['pixel_values_sha256']==ref['pixel_values_sha256'] and paper_digest(b['input_ids'])==ref['input_ids_sha256']
    assert paper_digest(s['prompt'])==ref['prompt_sha256'];cap=s['max_new_tokens']
   binding=hashes(a.inputs);native=engine.generate(a,l,'Native',cap,capture=True)
   assert native[0]['all_effective_logits_finite'] and not native[0]['empty'] and native[0]['eos'] and not native[0]['truncated']
   if idx==0:
    noop=engine.generate(a,l,'Native',cap,capture=True,noop=True)
    assert native[0]['token_ids']==noop[0]['token_ids'] and all(torch.equal(x,y) for x,y in zip(native[1],noop[1]))
    with torch.inference_mode():hf=a.model.generate(**a.inputs,do_sample=False,num_beams=1,max_new_tokens=cap,use_cache=True,logits_to_keep=1,return_dict_in_generate=True,output_scores=True)
    assert hf.sequences[0,l.length:].tolist()==native[0]['token_ids'] and len(hf.scores)==len(native[1])
    assert all(torch.equal(x,y.float().cpu()) for x,y in zip(native[1],hf.scores))
    result['native_HF_and_noop_exact']=True;del hf,noop
   score=lambda g: {'score':common.score(s,g['text']),'parsed':common.prediction(g['text'])} if protocol.startswith('GQA') else scorer.score(s,g['text'])
   result['generations'].append({'protocol':protocol,'question_id':s['question_id'],'condition':'Native','generation':concise(native[0]),'scoring':score(native[0]),'input_hashes':binding,'gold':s['answer']});del native
   for label in ('C3-V2','C3-V2+L11-L24'):
    keep=common.KEEP[label];cutoff=common.CUTOFF[label];ll=replace(l,cutoff=cutoff);before=binding
    with operators(a) as (calls,mlp,shapes),structural_trace(a,ll) as trace:
     actual=engine.generate(a,ll,label,3,capture=True,audit=True,fixed=True)
    g=actual[0];steps=len(g['token_ids']);assert steps==3
    assert trace['vision_calls']==trace['projector_calls']==1 and trace['image_embedding_exact']
    assert trace['norm_calls']==trace['post_norm_calls']==trace['mlp_calls']==[steps]*32 and trace['skip_residual_checks']==4*steps
    for i in range(32):assert mlp[i]==steps
    for i in (2,3,4,5):
     assert all(calls[i,op]==1 for op in ('q_proj','k_proj','v_proj','o_proj'))
     assert {r['op']:r['shape'][1] for r in shapes if r['layer']==i}=={'q_proj':ll.length-1,'k_proj':ll.length,'v_proj':ll.length,'o_proj':ll.length-1}
    assert all(r['blocked_probability_max']==0 for r in g['attention_audit'])
    events=g['visual_audit'];assert events
    assert all(e['image_keys_permitted']==(e['layer'] in keep) and ((e['answer_image_mass']>0)==(e['layer'] in keep)) for e in events)
    assert all(e['relay_image_mass']==0 for e in events if e['prefill'] and e['layer']>=20)
    assert any(e['relay_image_mass']>0 for e in events if e['prefill'] and e['layer']<20)
    assert all(not e['image_keys_present'] for e in events if (not e['prefill'] and e['layer'] not in keep) or (e['prefill'] and e['layer']>=cutoff))
    kv=g['kv_audit'];assert kv['retained_image_layers']==list(keep) and kv['historical_cache_copies']==kv['append_concatenations']==0
    for row in kv['layers']:
     if row['layer'] in (2,3,4,5):assert row['used_tokens']==row['storage_bytes']==0
     else:
      expected=(list(range(ll.length)) if row['layer'] in keep else list(ll.text))+list(range(ll.length,ll.length+steps-1))
      assert row['original_positions']==expected and row['used_tokens']==len(expected)
    # Same-semantics numerical reference only, never a formal dense cohort.
    dense=engine.generate(a,ll,label,3,capture=True,audit=True,dense=True,fixed=True)
    gate=equivalence('llava',dense[1],actual[1],dense[3],actual[3])
    assert g['token_ids']==dense[0]['token_ids'] and gate['all_argmax_equal'],'Short output mismatch'
    assert len(actual[4])==len(dense[4])==3
    assert all(x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x) for x,y in zip(actual[4],dense[4]))
    assert hashes(a.inputs)==before,'Input changed during physical execution'
    assert g['all_effective_logits_finite'] and dense[0]['all_effective_logits_finite']
    record={'protocol':protocol,'question_id':s['question_id'],'condition':label,'structural_status':'PASS','strict_numerical_status':'PASS' if gate['passed'] else 'FAIL','numerical_equivalence':gate,'short_tokens':g['token_ids'],'short_text':g['text'],'short_output_dense_exact':True,'rope_exact':True,'input_tensors_unchanged':True,'visual_hidden_exit':cutoff,'retained_visual_KV_layers':list(keep),'attention_L2_L5_physically_skipped':True,'norm_MLP_residual_checks':'PASS','vision_calls':trace['vision_calls'],'projector_calls':trace['projector_calls'],'all_logits_finite':True}
    result['checks'].append(record)
    full_audits.append(dict(record,trace=trace,projection_shapes=shapes,visual_audit=g['visual_audit'],KV_audit=g['kv_audit']))
    print('PHYSICAL_CHECK',protocol,label,json.dumps(gate),flush=True)
    save('physical_smoke_summary.json',result);del actual,dense
    # Complete answer/EOS/scorer only; no dense reference full generation.
    generated=engine.generate(a,ll,label,cap)[0]
    assert generated['all_effective_logits_finite'] and generated['eos'] and not generated['empty'] and not generated['truncated']
    item={'protocol':protocol,'question_id':s['question_id'],'condition':label,'generation':concise(generated),'scoring':score(generated),'gold':s['answer']}
    if ref and label=='C3-V2':
     item['historical_tokens_exact']=generated['token_ids']==ref['generation']['token_ids'];item['historical_score_exact']=score(generated)==ref['scoring'];item['reference_commit']='2c5d1820679b5eaad844e06db9cf46dcad23316f'
     assert item['historical_tokens_exact'] and item['historical_score_exact'],'Historical migration output/scorer mismatch'
    result['generations'].append(item);assert hashes(a.inputs)==binding
    print('COMPLETE_ANSWER',protocol,label,repr(generated['text']),flush=True)
   save('physical_smoke_summary.json',result)
  with gzip.open(O/'physical_audit.json.gz','wt',encoding='utf-8') as f:json.dump(full_audits,f,ensure_ascii=False,allow_nan=False)
  result['status']='READY' if all(x['strict_numerical_status']=='PASS' for x in result['checks']) else 'READY_WITH_WARNING'
  result['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30;result['peak_reserved_GiB']=torch.cuda.max_memory_reserved()/2**30
 except Exception:
  result['status']='BLOCKED';result['error']=traceback.format_exc();raise
 finally:
  result['finished_at']=datetime.now(timezone.utc).isoformat();save('physical_smoke_summary.json',result)
  if a is not None:
   a.inputs={};del a
  gc.collect();torch.cuda.empty_cache()
 print('RECOVERY_SMOKE_RESULT',result['status'],flush=True)

if __name__=='__main__':main()
