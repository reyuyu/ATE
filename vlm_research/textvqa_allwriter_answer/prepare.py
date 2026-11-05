"""CPU-only provenance audit and freeze; never initializes a model or CUDA."""
import tarfile,shutil,collections,csv
from .common import *

QUALITY_KEYS=['text','token_ids','eos','truncated','empty','repeated','content_tokens','decode_steps','first_logits_sha256','first_top1_token']

def main():
 import torch
 torch.set_num_threads(1);assert git('merge-base','HEAD',BASE)==BASE and shutil.disk_usage('/data').free>20*2**30
 OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'protocol.json').exists(),'Frozen protocol already exists; do not overwrite'
 shutil.copyfile(HISTORY/'sample_manifest.json',OUT/'sample_manifest.json');samples=read(OUT/'sample_manifest.json');assert len(samples)==500
 freeze=read(HISTORY/'execution_freeze.json');assert sha(HISTORY/'sample_manifest.json')==freeze['sample_manifest_sha256']
 for path,h in freeze['sources'].items():assert sha(ROOT/path)==h,path
 for path,h in freeze['weights'].items():assert sha(path)==h,path
 package=read(HISTORY/'packaging.json');assert sha(HISTORY/package['archive'])==package['archive_sha256'];expected=read(HISTORY/'result_hashes.json')
 baselines={};index=[]
 with tarfile.open(HISTORY/package['archive']) as archive:
  for member in archive.getmembers():
   parts=member.name.split('/')
   if len(parts)!=3 or parts[0]!='checkpoints' or parts[1] not in samples or parts[2] not in ['Native.json','C3-V2.json']:continue
   data=archive.extractfile(member).read();h=hashlib.sha256(data).hexdigest();assert h==expected[member.name]
   d=json.loads(data);r=d['payload'];assert digest(r)==d['sha256'];s=samples[parts[1]]
   assert r['sample_sha256']==digest(s) and r['sources']==freeze['sources'];assert len(s['answer'])==10
   assert abs(r['score']-textvqa_score(r['generation']['text'],s['answer']))<1e-12
   c='N' if parts[2]=='Native.json' else 'Historical-C3-V2';g={k:r['generation'][k] for k in QUALITY_KEYS}
   record=dict(key=parts[1],condition=c,input_binding=r['input_binding'],sample_sha256=r['sample_sha256'],generation=g,score=r['score'],provenance=dict(archive=str((HISTORY/package['archive']).relative_to(ROOT)),archive_sha256=package['archive_sha256'],member=member.name,member_sha256=h,payload_sha256=d['sha256'],execution_freeze_sha256=sha(HISTORY/'execution_freeze.json')))
   atomic(OUT/f'baselines/{parts[1]}/{c}.json',record);baselines[parts[1],c]=record;index.append(dict(key=parts[1],condition=c,member=member.name,member_sha256=h))
 assert len(baselines)==1000
 p=processor_cpu();inputs={};groups=collections.defaultdict(list);images={}
 for j,(key,s) in enumerate(sorted(samples.items())):
  if s['image_path'] not in images:assert sha(s['image_path'])==s['image_sha256'];images[s['image_path']]=s['image_sha256']
  x,binding=prepare_input(p,s);assert binding==baselines[key,'N']['input_binding']==baselines[key,'Historical-C3-V2']['input_binding']
  inputs[key]=binding;groups[s['image_sha256']].append(key)
  if j%100==0:print('CPU_INPUT_AUDIT',j,flush=True)
 assert len(groups)==309
 scores={c:sum(r['score'] for (k,cc),r in baselines.items() if cc==c)/5 for c in ['N','Historical-C3-V2']};assert abs(scores['N']-49.68)<1e-9 and abs(scores['Historical-C3-V2']-22.34)<1e-9
 shard={};loads=[0,0]
 for h,keys in sorted(groups.items(),key=lambda kv:(-len(kv[1]),digest([SEED,kv[0]]))):
  rank=min(range(2),key=lambda r:(loads[r],r));shard[h]=rank;loads[rank]+=len(keys)
 smoke=[]
 for rank in range(2):
  keys=sorted([k for k,s in samples.items() if shard[s['image_sha256']]==rank],key=lambda k:(len(samples[k]['input_mapping']['prompt_ids']),k))
  smoke.extend([keys[0],keys[len(keys)//2],keys[-1]])
 atomic(OUT/'input_bindings.json',inputs);atomic(OUT/'partition.json',dict(image_shard=shard,counts=loads,smoke=smoke))
 atomic(OUT/'model_manifest.json',dict(revision=MODEL.name,files={str(f):sha(f) for f in sorted(MODEL.iterdir()) if f.is_file()}))
 atomic(OUT/'baseline_validation.json',dict(status='CPU_PASSED_GPU_REPRODUCTION_PENDING',at=now(),records=1000,scores_percent=scores,questions=500,images=309,images_rehashed=309,all_prompt_token_and_pixel_tensor_hashes_exact=True,all_source_weight_and_record_hashes_verified=True,raw_archive_sha256=package['archive_sha256'],raw_archive_members_preserved=True,A_B_reuse=False,A_B_reason='No strictly matching full500 A/B records found. Existing full-depth no-answer-image cohort has300 questions; existing500 C3/Middle conditions restrict R writing and/or physically remove image hidden/KV.'))
 atomic(OUT/'protocol.json',dict(status='FROZEN_BEFORE_GPU',at=now(),base_commit=BASE,revision=MODEL.name,dtype='float16',attention='eager',transformers='4.49.0',greedy=True,max_new_tokens=32,logits_to_keep=1,EOS='unchanged model.generation_config.eos_token_id',n=500,images=309,conditions=dict(N='native, all connections; reuse audited history after Smoke',A='R->Image all32 ON; P-1/Decode->Image all32 OFF; attention outputs native',B='same as A plus P-1/Decode o_proj output zero only at L2-L5'),layers_zero_based=True,R='last image token +1 through original P-1 excluded; original template included',B_zero_layers=[2,3,4,5],dense_full_native_KV=True,no_physical_image_exit=True,unchanged_MLP_Norm_Residual_RoPE=True,scorer='pinned official TextVQA ten-answer leave-one-out consensus evaluator',sample_manifest_sha256=sha(OUT/'sample_manifest.json'),input_bindings_sha256=sha(OUT/'input_bindings.json'),model_manifest_sha256=sha(OUT/'model_manifest.json'),smoke_ids=smoke,primary_comparisons=[['N','A'],['A','B'],['N','B']],bootstrap=dict(repetitions=20000,seed=SEED,cluster='Image SHA256',question_weighted=True,confidence=.95,exploratory=True),quality_only=True,no_timing_physical_optimization_or_search=True,history_C3_reference_only=True,gpu_uuids=UUIDS))
 atomic(OUT/'cpu_preflight.json',dict(at=now(),disk_free=shutil.disk_usage('/data').free,main_before=git('rev-parse','main'),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv']).decode(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']).decode(),no_GPU_model_loaded=True))
 print('CPU_FROZEN',len(samples),len(groups),loads,scores,smoke)

if __name__=='__main__':main()
