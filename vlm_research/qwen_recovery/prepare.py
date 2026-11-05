"""CPU audit of the exact previous 1000 questions and immutable baselines."""
import shutil,collections,csv,subprocess,os
from .common import *
def main():
 import torch
 from transformers import AutoProcessor
 from ..qwen3vl_early_windows.experiment import prepare,gqa_prediction
 torch.set_num_threads(1);assert git('merge-base','HEAD',BASE)==BASE;assert shutil.disk_usage('/data').free>20*2**30
 OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);os.chmod(PRIVATE,0o700)
 for name in ['sample_manifest.json','model_manifest.json']:shutil.copyfile(HIST/name,OUT/name)
 m=read(OUT/'sample_manifest.json');assert len(m['samples'])==1000
 oldsource=read(HIST/'source_manifest.json');oldmodelsha=sha(HIST/'model_manifest.json');oldprotocolsha=sha(HIST/'protocol.json');rows=[];counts=collections.Counter()
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
 for i,s in enumerate(m['samples']):
  x,l=prepare(processor,s);q=s['question_id'];assert l==m['input_bindings'][q]
  for label,c in [('Native','Native'),('F','Dense-F')]:
   p=HIST/f'checkpoints/rank{s["rank"]}/{q}__{c}.json';d=read(p);r=d['result'];b=d['binding']
   assert b['question_id']==q and b['image_sha256']==s['image_sha256'] and b['input_sha256']==digest(l)
   assert b['source_manifest_sha256']==digest(oldsource) and b['model_manifest_sha256']==oldmodelsha and b['protocol_sha256']==oldprotocolsha
   assert r['correct']==(gqa_prediction(r['text'])==s['gold']);counts[label]+=r['correct']
   rows.append(dict(question_id=q,condition=label,source_condition=c,path=str(p.relative_to(ROOT)),sha256=sha(p),correct=r['correct'],text=r['text'],token_ids=json.dumps(r['token_ids']),input_sha256=digest(l),image_sha256=s['image_sha256']))
  if i%100==0:print('AUDIT',i,flush=True)
  del x
 assert counts==dict(Native=694,F=588)
 with (OUT/'baseline_index.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 write(OUT/'baseline_validation.json',dict(status='PASSED',records=2000,correct_counts=counts,historical_commit=BASE,all_inputs_reconstructed_strict=True,all_baseline_records_hash_bound=True,sample_manifest_sha256=sha(OUT/'sample_manifest.json'),original_source_manifest_sha256=sha(HIST/'source_manifest.json'),at=now()))
 write(OUT/'protocol.json',dict(status='FROZEN_BEFORE_NEW_GPU_PREDICTIONS',at=now(),base_commit=BASE,model_revision=MODEL.name,dtype='bfloat16',attention='eager',transformers='4.57.6',greedy=True,max_new_tokens=32,logits_to_keep=1,independent_dense_cache=True,conditions=CONDITIONS,condition_specs=SPECS,layer_index='0-based; Qwen3-VL has 36 decoder layers',skip_layers=list(SKIP),late_layers=list(range(22,36)),restore_visual_only_non_skip_layers=True,R='non-visual positions strictly after last original Image token and before P-1',sample_manifest_sha256=sha(OUT/'sample_manifest.json'),baseline_index_sha256=sha(OUT/'baseline_index.csv'),n=1000,images=918,sampling='EXACT previous manifest, no new sampling or filtering',smoke='same previous 20 frozen IDs, 10 per worker, all five interventions plus native/no-op/F reproduction',gpu_uuids=UUIDS,bootstrap=dict(unit='Image SHA256 whole cluster',replicates=20000,seed=SEED,confidence=.95,exploratory=True,comparisons=[(c,b) for c in CONDITIONS for b in ['F','Native']],additional_frozen_comparisons=[['C','A'],['C','B']],simultaneous_F_family='Bonferroni percentile CI, 5 comparisons: quantiles .005/.995',best_selection='highest full 1000-question accuracy; ties in fixed A,B,C,D,E order'),continuation_gate=dict(minimum_improvement_pp=3.0,paired_95ci_lower_strictly_above=0,interpretation='User gate evaluated on best condition pointwise paired CI; simultaneous CI additionally reported. Even if passed, no physical implementation or timing in this round.'),prohibitions=['E2E or any latency benchmark','FLOPs calculation','performance or peak memory benchmark','kernel search','additional conditions or layer combinations','prompt/tokenizer/model-weight changes','training'],accuracy_only=True))
 write(OUT/'preflight.json',dict(at=now(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv']).decode(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']).decode(),disk_free=shutil.disk_usage('/data').free,main_before=git('rev-parse','main'),authorization_status='PENDING_THIS_ROUND'))
 print('CPU_FROZEN',counts,flush=True)
if __name__=='__main__':main()
