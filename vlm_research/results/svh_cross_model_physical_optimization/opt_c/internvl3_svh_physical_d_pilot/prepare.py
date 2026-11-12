"""CPU-only preregistration: new held-out IDs/images, numerical gates, and efficiency subset."""
import importlib.util,json,os,types
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from .common import *
from vlm_research.internvl3_visual_writing_figure3_3000 import engine as base

def main():
 RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 assert not (RUN/'active.json').exists()
 old=read(OLD/'sample_manifest.json')
 oldsamples=old['samples'] if isinstance(old,dict) else old
 oq={s['question_id'] for s in oldsamples};oi={s['image_sha256'] for s in oldsamples}
 rows=[json.loads(x) for x in (DATA/'inputs.jsonl').read_text().splitlines()]
 answers=read(DATA/'answers.json');dm=read(DATA/'manifest.json')
 assert sha(DATA/'inputs.jsonl')==dm['inputs_sha256'] and sha(DATA/'answers.json')==dm['answers_sha256']
 module_spec=importlib.util.spec_from_file_location('pilot_official_conversation',MODEL/'conversation.py')
 convmod=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(convmod)
 config=read(MODEL/'config.json')
 conv=convmod.get_conv_template(config['template'])
 dummy=types.SimpleNamespace(conv_template=conv,system_message=conv.system_message)
 tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,use_fast=False)
 typepairs=[('choose','attr'),('choose','cat'),('choose','rel'),('compare','attr'),('logical','obj'),('query','attr')]
 used=set();samples=[];hashes={};counts={}
 for task,pair in zip(TASKS,typepairs):
  candidates=[r for r in rows if (r['types']['structural'],r['types']['semantic'])==pair and r['question_id'] not in oq]
  candidates.sort(key=lambda r:digest(['InternVL3-Physical-D-pilot-v1',r['question_id']]))
  chosen=[]
  for r in candidates:
   p=Path(r['image_path'])
   if not p.is_file():continue
   if str(p) not in hashes:hashes[str(p)]=sha(p)
   image_sha=hashes[str(p)]
   if image_sha in oi or image_sha in used:continue
   s=dict(task=task,question_id=r['question_id'],image_id=r['image_id'],question=r['question'],
    answer=answers[r['question_id']],image_sha256=image_sha,actual_image_path=str(p),
    source_split=r['split'],source_types=r['types'])
   _,prompt,ids,roles=base.prepare_cpu(dummy,tokenizer,s)
   s.update(prompt=prompt,input_ids=ids[0].tolist(),roles=roles,input_length=roles['n'])
   chosen.append(s);used.add(image_sha)
   if len(chosen)==20:break
  assert len(chosen)==20,(task,len(chosen))
  samples.extend(chosen);counts[task]=len(candidates)
 assert len(samples)==120 and len({s['image_sha256'] for s in samples})==120
 efficiency=[]
 for task in TASKS:
  part=sorted([s for s in samples if s['task']==task],key=lambda s:(s['input_length'],s['question_id']))
  efficiency.extend(part[i]['question_id'] for i in (0,4,9,14,19))
 lengths=sorted(range(120),key=lambda i:(samples[i]['input_length'],samples[i]['question_id']))
 # Six fixed smoke inputs independent of predictions; two shortest, two central, two longest.
 smoke=[samples[lengths[i]]['question_id'] for i in (0,1,59,60,118,119)]
 manifest=dict(samples=samples,efficiency_question_ids=efficiency,smoke_question_ids=smoke,
  selection='SHA256 sort of fixed tag and QID; six tasks in declared order; reject historical QID or image SHA and global duplicate image',
  official_dataset_manifest_sha256=sha(DATA/'manifest.json'),
  old_manifest_sha256=sha(OLD/'sample_manifest.json'),old_questions=len(oq),old_image_clusters=len(oi),
  question_overlap=0,image_sha_overlap=0,unique_new_images=120,
  additional_exploration_scope='Published original Figure3 and L20 extension plus latest Dense joint share the same historical 3000 manifest. No claim about unprovided cohorts.',
  source_candidates=counts,efficiency_selection='5/task, input length ranks 0,4,9,14,19 among frozen 20; 30 distinct images',
  image_files_hashed=len(hashes))
 save(RUN/'sample_manifest.json',manifest)
 gate=dict(native_noop='all effective logits torch.equal; full token IDs exact',
  historical_D='complete tokens and SHA256 of every float32 effective logit exact on three historical samples',
  physical_vs_dense=dict(all_finite=True,per_step_logits_max_abs_le=0.125,per_step_logits_mean_abs_le=0.01,
   per_layer_shared_hidden_relative_RMS_le=0.015625,all_step_top1_equal=True,complete_tokens_equal=True),
  hidden_relative_RMS='RMS(physical-dense)/max(RMS(dense),1e-6), aligned by original logical positions; all shared rows',
  rationale='Predeclared BF16 engineering tolerances: absolute logits 0.125 and mean 0.01; layer hidden RMS difference <=1/64. These are numerical agreement gates, not quality noninferiority thresholds.',
  failure_policy='Stop formal quality and efficiency on failure. Diagnose earliest mismatch; preserve each attempt. No post-hoc tolerance changes.')
 protocol=dict(experiment='InternVL3-SVH Physical-D pilot',status='FROZEN_BEFORE_GPU',
  model='OpenGVLab/InternVL3-8B',revision=REV,dtype='BF16',attention='eager',torch='2.6.0+cu124',
  transformers='4.49.0',GPU_UUID=GPU,batch_size=1,greedy=True,max_num=1,visual_tokens=256,
  max_new_tokens=32,logits_to_keep=1,use_cache=True,quality_normal_EOS=True,methods=METHODS,
  Dense_D='Frozen joint D: answer->image blocked all28; R->image blocked L20-27; L2-5 answer o_proj output zero; dense KV retained',
  Physical_D='Image Hidden physically removed before L20; L2-5 P-1 and Decode excluded from QKV/QK/AV/o_proj; original positions retained; persistent original image KV removed all layers; no persistent KV at L2-5',
  Native='Official unmodified full model, including answer direct image access',
  numerical_gate=gate,sample_manifest_sha256=sha(RUN/'sample_manifest.json'),quality_requests=360,
  efficiency_inputs=30,efficiency_repeats=5,efficiency_formal_requests=450,
  warmup='2 requests per method before timing; every input gets 5 independently restarted requests/method',
  efficiency_output_tokens=32,efficiency_prefill=1,efficiency_real_decode=31,efficiency_ignore_EOS=True,
  timer='CUDA-synchronized perf_counter from GPU-ready pixels/IDs through vision/projector/injection/prefill/31decode and transient cleanup. Persistent cache kept live for measurement. Disk/tokenization/preprocessing/H2D/model load/output serialization excluded.',
  KV='actual tensor numel*element_size; capacity unique live underlying storage.nbytes, never allocator peak or theoretical token count',
  efficiency_order='input-major, repeat-major; rotate three-method order by input+repeat',
  audit_outside_timing=True,bootstrap_replicates=20000,bootstrap_seed=20261110,
  bootstrap='paired image-SHA cluster resampling, question-weighted; ordinary exploratory 95% CI',
  no_layer_search=True,no_training=True,no_seven_benchmarks=True,no_3000_rerun=True,
  source_parent=command('git','rev-parse','HEAD'),GitHub_branch='codex/internvl3-svh-physical-d-pilot-3090')
 save(RUN/'protocol.json',protocol)
 modelref=read(OLD/'model_manifest_reference.json')
 assert sha(modelref['model_verification_file'])==modelref['model_verification_sha256']
 save(RUN/'model_manifest.json',dict(model_path=str(MODEL),revision=REV,prior_verification=modelref,
  config_sha256=sha(MODEL/'config.json'),weight_rehash_or_download=False))
 for name in ['sample_manifest.json','protocol.json','model_manifest.json']:
  (OUT/name).write_bytes((RUN/name).read_bytes())
 print(json.dumps(dict(new_samples=len(samples),efficiency=len(efficiency),smoke=smoke,protocol_sha256=sha(RUN/'protocol.json'),overlap=0)))
if __name__=='__main__':main()
