import shutil,csv,collections,os,subprocess
from types import SimpleNamespace
from .common import *
def main():
 import torch
 from transformers import AutoProcessor,AutoConfig
 from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
 from ..qwen3vl_early_windows.experiment import prepare,tensor_digest
 from ..qwen3vl_semantic_visual_writers.engine import roles
 torch.set_num_threads(1);assert git('merge-base','HEAD',BASE)==BASE;assert shutil.disk_usage('/data').free>20*2**30
 OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);os.chmod(PRIVATE,0o700)
 for name in ['sample_manifest.json','model_manifest.json']:shutil.copyfile(RECOVERY/name,OUT/name)
 m=read(OUT/'sample_manifest.json');assert len(m['samples'])==1000
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);config=AutoConfig.from_pretrained(MODEL,local_files_only=True);specs={};rows=[];counts=collections.Counter();strip_differences=[]
 for i,s in enumerate(m['samples']):
  q=s['question_id'];x,l=prepare(processor,s);assert l==m['input_bindings'][q]
  mapping=roles(l,processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'));assert mapping['R']==list(range(l['image_positions'][-1]+1,l['input_length']-1))
  ids=x['input_ids'];positions,delta=Qwen3VLModel.get_rope_index(SimpleNamespace(config=config),ids,x['image_grid_thw'],attention_mask=x['attention_mask'])
  specs[q]=dict(layout=l,roles=mapping,mrope=dict(position_ids_sha256=tensor_digest(positions),delta=int(delta.item())))
  for c in ['Native','F','D','E']:
   p=baseline_path(s,c);d=read(p);r=d['result'];assert d['binding']['input_sha256']==digest(l) and d['binding']['image_sha256']==s['image_sha256']
   assert r['correct']==(r['text'].rstrip('.').lower()==s['gold']);counts[c]+=r['correct']
   if r['text']!=r['text'].strip():strip_differences.append(dict(question_id=q,condition=c,text=r['text']))
   rows.append(dict(question_id=q,condition=c,path=str(p.relative_to(ROOT)),sha256=sha(p),correct=r['correct'],text=r['text'],token_ids=json.dumps(r['token_ids'])))
  if i%100==0:print('CPU_AUDIT',i,flush=True)
  del x
 assert counts==dict(Native=694,F=588,D=585,E=680)
 with (OUT/'baseline_index.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 write(OUT/'input_specs.json',specs)
 old=read(HISTORY/'sample_manifest.json')['samples'];oldq={s['question_id'] for s in old};oldim={s['image_sha256'] for s in old}
 write(OUT/'historical_cohort_audit.json',dict(historical_questions=len(old),current_questions=1000,current_images=918,shared_question_ids=len(oldq&{s['question_id'] for s in m['samples']}),current_questions_with_historical_images=sum(s['image_sha256'] in oldim for s in m['samples']),historical_task_counts=dict(collections.Counter(s['task'] for s in old)),current_official_task_counts=dict(collections.Counter(s['task'] for s in m['samples'])),interpretation='Different frozen cohorts; historical 3000 accuracy is context only, never substituted for the current 1000'))
 write(OUT/'baseline_validation.json',dict(status='PASSED',records=4000,correct_counts=counts,all_current_inputs_reconstructed=True,old_keep_D_rule_semantically_same=True,scoring_rule='raw tokenizer.decode(ids,skip_special_tokens=True).rstrip(".").lower(); same as current1000',historical_whitespace_rule='historical wrapper also returns a stripped text; preserve that in a separate field and score current1000 using raw decode',baseline_texts_changed_by_strip=strip_differences,at=now()))
 write(OUT/'protocol.json',dict(status='FROZEN_BEFORE_GPU',at=now(),base_commit=BASE,model_revision=MODEL.name,dtype='bfloat16',attention='eager',transformers='4.57.6',greedy=True,max_new_tokens=32,logits_to_keep=1,conditions=['Keep-L0-L21','All-Writer'],keep_open_R_image_layers=list(range(22)),all_writer_open_R_image_layers=list(range(36)),P1_Decode_image_blocked_layers=list(range(36)),early_L1_L6_attention_normal=True,no_o_proj_bypass=True,no_physical_exit=True,independent_dense_native_KV=True,mask_implementation='Unmodified historical qwen3vl_keep_l0_21.engine.generate / qwen3vl_visual_writing_necessity.engine.generate',sample_manifest_sha256=sha(OUT/'sample_manifest.json'),input_specs_sha256=sha(OUT/'input_specs.json'),baseline_index_sha256=sha(OUT/'baseline_index.csv'),smoke_ids=m['smoke_question_ids'],smoke_n=20,reuse_D_only_if='All 20 historical Keep versus latest D valid logits torch.equal AND complete token IDs identical AND original D checkpoint tokens reproduced',if_equivalence_fails='Do not reuse D. Diagnose mismatch; if both mask/input/cache paths remain valid, generate the same two requested historical conditions on all1000, keeping D as a separately labeled reference. Stop on invalid mask/cache/NaN/OOM.',n=1000,images=918,gpu_uuids=UUIDS,quality_only=True,bootstrap=dict(repetitions=20000,seed=SEED,cluster='Image SHA256',question_weighted=True,primary=['All-Writer','Keep-L0-L21'],secondary=[['All-Writer','F'],['All-Writer','E'],['Keep-L0-L21','F'],['Keep-L0-L21','E'],['All-Writer','Native']],confidence=.95,exploratory=True),no_timing_FLOPs_optimization_or_condition_search=True))
 write(OUT/'preflight.json',dict(at=now(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv']).decode(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']).decode(),disk_free=shutil.disk_usage('/data').free,main_before=git('rev-parse','main')))
 print('FROZEN',counts)
if __name__=='__main__':main()
