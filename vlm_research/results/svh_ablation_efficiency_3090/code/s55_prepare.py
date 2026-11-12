"""CPU input/reuse audit, protocol freeze and semantic unit checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from s55_common import *
from collections import Counter
from types import SimpleNamespace
import ast,torch
import ate_runtime
from vlm_research.information_horizon_alignment import engine as q,score as scorer
from vlm_research.information_horizon_alignment.common import tensor_sha
from vlm_research.p1_layerwise_accuracy.engine import tensor_hash
from vlm_research.physical_handoff_fastpath.engine import Layout
from s55_ablation_engine import NoSelectiveTransfer,NoAttentionBypass

def main():
 assert not (O/'protocol.json').exists(),'Refuse replacing a frozen protocol'
 (O/'ablation').mkdir(exist_ok=True);(O/'generation_length').mkdir(exist_ok=True)
 p=read(Q/'run_metadata.json');cfg=p['config'];hist=lines(Q/'mapped_manifest.jsonl.gz');pred=lines(Q/'predictions.jsonl.gz')
 assert p['status']=='COMPLETE' and p['physical_smoke_status']=='PASS'
 assert cfg['model_revision']==REV and cfg['dtype']=='float16' and cfg['attention']=='eager'
 for path,h in cfg['source_sha256'].items():assert sha(R/path)==h,path
 target=[s for s in hist if s['benchmark'] in ('ScienceQA-IMG','POPE')]
 refs={s['key']:s for s in pred if s['benchmark'] in ('ScienceQA-IMG','POPE')}
 assert Counter(s['benchmark'] for s in target)=={'ScienceQA-IMG':2017,'POPE':8910}
 assert len(refs)==10927 and all(r['condition']=='Physical-C3' for r in refs.values())
 processor=ate_runtime.processor_cpu();image_checks={};pixels={};rows=[];bindings=[];reused=[]
 for bench,view in [('ScienceQA-IMG','ScienceQA-IMG-paper'),('POPE','POPE-paper')]:
  current={s['question_id']:s for s in ate_runtime.iter_inputs(view)};gold=ate_runtime.load_answers(view)
  for s in [x for x in target if x['benchmark']==bench]:
   c=current[s['normalized_question_id']];g=gold[s['normalized_question_id']];ref=refs[s['key']]
   for k in ['question','prompt','user_text','image_id']:assert c[k]==s[k],(s['key'],k)
   assert g==s['answer']==ref['gold']
   path=c['image_path']
   if path not in image_checks:
    image_checks[path]=sha(path);px=q.paper_pixels(path,processor.image_processor);pixels[path]=tensor_sha(px)
   assert image_checks[path]==s['image_sha256']==ref['image_sha256']
   ids=[]
   for t in q.image_ids(s['prompt'],processor.tokenizer,-200):ids.extend([32000]*576 if t==-200 else [t])
   assert digest(ids)==ref['input_ids_sha256'] and digest(s['prompt'])==ref['prompt_sha256']
   assert pixels[path]==ref['pixel_values_sha256'] and len(ids)==ref['expanded_length']
   assert ref['generation']['logits_to_keep']==1 and ref['generation']['independent_cache'] and ref['generation']['all_effective_logits_finite']
   assert scorer.score(s,ref['generation']['text'])==ref['scoring']
   assert s['max_new_tokens']==cfg['max_new_tokens'][bench]==(1024 if bench=='ScienceQA-IMG' else 128)
   inp={k:v for k,v in c.items() if k not in ('answer','answers','gold','label','lecture','solution','correct_answer','answer_idx')}
   inp.update(key=s['key'],benchmark=bench,max_new_tokens=s['max_new_tokens'],image_sha256=s['image_sha256'])
   if bench=='ScienceQA-IMG':inp['options']=s['options']
   if bench=='POPE':assert inp['strategy']==s['strategy']
   rows.append(inp);bindings.append(dict(key=s['key'],image_sha256=image_checks[path],pixel_values_sha256=pixels[path],input_ids_sha256=digest(ids),prompt_sha256=digest(s['prompt']),expanded_length=len(ids)))
   reused.append(dict(key=s['key'],benchmark=bench,condition='SVH',generation=ref['generation'],scoring=ref['scoring'],source_directory=str(Q),source_record_sha256=digest(ref)))
   if len(rows)%1000==0:print('CPU_QUALITY_VERIFIED',len(rows),flush=True)
 # Neither Gold nor answer-derived fields are stored in the model-input manifest.
 writegz(O/'ablation/inputs.jsonl.gz',rows);save(O/'ablation/answers.json',{s['key']:s['answer'] for s in target})
 writegz(O/'ablation/input_verification.jsonl.gz',bindings);writegz(O/'ablation/reused_svh.jsonl.gz',reused)
 assert Counter(s['strategy'] for s in rows if s['benchmark']=='POPE')=={'random':2910,'popular':3000,'adversarial':3000}
 # Existing fixed30 inputs and exact preprocessed tensor bytes.
 fixed=lines(U/'input_manifest.jsonl.gz');mapping=read(REST/'frozen30_image_path_mapping.json')
 print('MAPPING_TYPE',type(mapping).__name__,str(mapping)[:1000],flush=True)
 if 'records' in mapping:mapping={r['key']:r.get('local_image_path',r.get('image_path')) for r in mapping['records']}
 checks=[]
 for s in fixed:
  path=mapping[s['key']]
  if isinstance(path,dict):path=path.get('local_image_path',path.get('new_path',path.get('image_path')))
  assert sha(path)==s['image_sha256']
  raw=dict(input_ids=torch.tensor([s['input_ids']],dtype=torch.long),attention_mask=torch.ones((1,len(s['input_ids'])),dtype=torch.long),pixel_values=q.paper_pixels(path,processor.image_processor).half())
  assert {k:tensor_hash(v) for k,v in raw.items()}==s['input_hashes']
  checks.append(dict(key=s['key'],image_path=path,input_hashes=s['input_hashes']))
 writegz(O/'generation_length/input_manifest.jsonl.gz',fixed);save(O/'generation_length/input_path_mapping.json',{c['key']:c['image_path'] for c in checks})
 save(O/'generation_length/input_verification.json',dict(status='PASS',count=30,records=checks))
 # Small genuine CPU mask/cache tests use a dummy backbone without GPU/model load.
 dummy=SimpleNamespace(kind='llava',model=SimpleNamespace(language_model=SimpleNamespace(model=None)),inputs={'input_ids':torch.zeros((1,9),dtype=torch.long)},layers=[torch.nn.Linear(1,1,dtype=torch.float16) for _ in range(32)])
 layout=Layout(9,(2,3,4),(0,1,5,6,7,8),(5,6,7),(5,6,7),20)
 full=__import__('vlm_research.c3v2_middle_reaccess.engine',fromlist=['MiddleWriter']).MiddleWriter(dummy,layout,(),4)
 ns=NoSelectiveTransfer(dummy,layout,(),4);nb=NoAttentionBypass(dummy,layout,(),4)
 for i in range(32):
  assert (ns.prefill_mask(i)[0,0,list(layout.relay),:][:,list(layout.image)]==0).all()
  assert torch.isneginf(ns.prefill_mask(i)[0,0,-1,list(layout.image)]).all()
 assert ns.layout.cutoff==32 and ns.physical and ns.physical_skip
 assert nb.layout.cutoff==20 and nb.physical and not nb.physical_skip and nb.cache.absent==()
 assert full.physical_skip and all(torch.equal(full.prefill_mask(i),nb.prefill_mask(i)) for i in range(32))
 assert NoSelectiveTransfer.forward is NoAttentionBypass.forward is full.forward.__func__
 assert NoSelectiveTransfer.context is NoAttentionBypass.context is type(full).context
 save(O/'cpu_tests.json',dict(status='PASS',all_quality_inputs_exact=10927,all_fixed30_tensors_exact=30,all_layers_transfer_mask_verified=True,answer_image_block_verified=True,non_target_masks_identical=True,inherited_decoder_and_attention_forward_unchanged=True,cuda_initialized=torch.cuda.is_initialized()))
 assert not torch.cuda.is_initialized()
 # Freeze narrow edits required to generalize the existing real timer from32toN.
 original=(U/'code/e2e32_runtime.py').read_text();new=original
 changes=[('def enabled(self):return self.step in (0,1,2,3,4,5,6,7,8,31)','def enabled(self):return self.step in (0,1,2,3,4,5,6,7,8,31)'),('def generate(a,layout,method,smoke=False):','def generate(a,layout,method,smoke=False,length=32):'),('length=32;n=layout.length;','n=layout.length;'),('len(ids)==32 and calls==1','len(ids)==length and calls==1'),('kv_snapshot(cache,ctrl,layout,method,32)','kv_snapshot(cache,ctrl,layout,method,length)'),('output_tokens=32,real_decode_calls=31','output_tokens=length,real_decode_calls=length-1'),('TPOT_ms=(end-first)*1000/31,CUDA_Event_Decode_TPOT_ms=sum(ms[1:])/31','TPOT_ms=(end-first)*1000/(length-1),CUDA_Event_Decode_TPOT_ms=sum(ms[1:])/(length-1)')]
 changes=[(x.replace('length=32;n=layout.length;','length=32;n=layout.length;device') if x=='length=32;n=layout.length;' else x,y.replace('n=layout.length;','n=layout.length;device') if x=='length=32;n=layout.length;' else y) for x,y in changes]
 for before,after in changes:
  assert before in new,before;new=new.replace(before,after)
 (O/'code/s55_length_runtime.py').write_text(new)
 import difflib
 (O/'generation_length/runtime_length_only.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile=str(U/'code/e2e32_runtime.py'),tofile='new_independent_length_runtime.py')))
 # Set parameter to32 and reverse constant substitutions; request AST must be exact.
 restored=new
 for before,after in reversed(changes):
  if before!=after:restored=restored.replace(after,before)
 assert ast.dump(ast.parse(restored))==ast.dump(ast.parse(original))
 save(O/'generation_length/runtime_ast_audit.json',dict(status='PASS',only_length_parameter_and_dependent_constants_changed=True,reverse_transform_AST_exact=True,frozen_runtime_sha256=sha(U/'code/e2e32_runtime.py'),new_runtime_sha256=sha(O/'code/s55_length_runtime.py')))
 protocol=dict(status='FROZEN',created_ns=time.time_ns(),gpu_uuid=UUID,model='llava-hf/llava-1.5-7b-hf',revision=REV,precision='FP16',attention='eager',batch=1,greedy=True,quality=dict(conditions=['Native','wo_selective','wo_bypass','SVH'],new_conditions=['Native','wo_selective','wo_bypass'],reuse='Full SVH only; all10927 image/Prompt/Processor/ID/Gold/input hashes and source/generation/scorer checked',counts={'ScienceQA-IMG':2017,'POPE':8910},pope_subsets={'random':2910,'popular':3000,'adversarial':3000},caps={'ScienceQA-IMG':1024,'POPE':128},stop='official EOS2, no forced output length; independent per-request KV',no_selective='Text relays read image at ALL32Prefill layers; no early exit; answer noimage and L2..5 physical bypass unchanged',no_bypass='TransferL0..19/exit20/removal unchanged; normal answer Attention and real text KV restored L2..5'),length=dict(methods=['Native','SVH','SVH-R'],lengths=[16,32,64,128],sessions=2,inputs=30,formal_per_session=360,ignore_eos=True,timing='Frozen4090 generate lifecycle; GPU tensors ready; controller/positions/mask initialization included; after context cleanup+CUDA sync; CPUlogs/KVaudit excluded',tpot='(E2E-TTFT)/(N-1)',kv='Actual live K/V tensor bytes after Nth output; N-1 decode inputs; NOT total GPU memory',warmup_native32=40,warmup_each_length_method=2,baseline_dual_probes_each=10,calibration_every_inputs=5,calibration_dual_probes_each=3,probe_length=32,performance_gate='WARN only >2%; no resets/deletions/retries; correctness/OOM/hardware safety hard stop',schedule='For each input: rotate/reverse four length groups by input index; cyclic method rotation by input+length index; Session2 exactly same schedule'),history_32_reuse=False,history_32_reason='OldGPU UUID differs and session2 drift15.09%; rerun only Native/SVH/SVHR on current GPU with all lengths',no_training=True,no_other_benchmarks=True,no_paper_edit=True,no_push=True)
 save(O/'protocol.json',protocol)
 files={str(p):sha(p) for p in (O/'code').glob('*.py')}
 for sub in ['physical_handoff_fastpath','ate_fulljoint_v2','accuracy_autoresearch','c3v2_middle_reaccess','information_horizon_alignment','mme_seed_quality','p1_layerwise_accuracy','physical_visual_handoff']:
  for p in (R/'vlm_research'/sub).glob('*.py'):files[str(p)]=sha(p)
 files[str(U/'code/e2e32_runtime.py')]=sha(U/'code/e2e32_runtime.py')
 save(O/'source_manifest.json',dict(files=files,model_receipt_sha256=sha(M/'.ate_revision.json'),quality_history_metadata_sha256=sha(Q/'run_metadata.json'),quality_history_predictions_sha256=sha(Q/'predictions.jsonl.gz'),git_HEAD=command(['git','-C',str(R),'rev-parse','HEAD'])))
 print('CPU_PREPARE_ALL_PASS',flush=True)
if __name__=='__main__':main()
