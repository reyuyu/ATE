import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import torch,sys,collections
from transformers import AutoProcessor
from receiver_cross_dataset.prepare import cpu_inputs
torch.set_num_threads(1)
def git_verified(p):
 data=subprocess.check_output(['git','-C',str(ROOT.parent),'show',BASE+':'+str(p.relative_to(ROOT.parent))]);assert data==p.read_bytes();return sha(p)
def main():
 guard();assert read(OUT/'CPU_test_suite.json')['passed'];oldm=HIST/'sample_manifest.json';git_verified(oldm);manifest=read(oldm);rows=manifest['rows'];assert len(rows)==300 and len({r['image_sha256'] for r in rows})==300
 # Sampling uses ONLY IDs and image SHAs; no scores or predictions enter the rule.
 subset=sorted(rows,key=lambda r:hashlib.sha256(f"{SEED}:{r['question_id']}:{r['image_sha256']}".encode()).hexdigest())[:120];selected={r['question_id'] for r in subset};shards={}
 for group in [subset,[r for r in rows if r['question_id'] not in selected]]:
  for j,r in enumerate(sorted(group,key=lambda r:r['image_sha256'])):shards[r['question_id']]=j%2
 assert collections.Counter(shards.values())=={0:150,1:150} and collections.Counter(shards[r['question_id']] for r in subset)=={0:60,1:60}
 write(OUT/'subset_120.json',dict(seed=SEED,rule='SHA256(seed:QuestionID:ImageSHA) lowest120; no predictions/scores used',question_ids=[r['question_id'] for r in subset],not_independent_confirmation=True))
 write(OUT/'shards.json',dict(UUIDS=UUIDS,rule='ImageSHA ascending alternating workers, within frozen120 and remaining180 separately; all questions/images unique',assignments=shards,counts=[150,150],subset_counts=[60,60]))
 design=layer_design();assert design['nested_sets']['2']==[14,19] and len(design['addition_order'])==32
 write(OUT/'layer_sets.json',design)
 for rel,h in read(HIST/'source_manifest.json')['files'].items():assert sha(ROOT/rel)==h
 model=read(HIST/'model_manifest.json');git_verified(HIST/'model_manifest.json')
 for name,meta in model['files'].items():assert sha(MODEL/name)==meta['sha256'],name
 write(OUT/'model_manifest.json',dict(**model,reverified_for_current_task_UTC=utc()))
 proc=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True);proc.patch_size=14;proc.num_additional_image_tokens=1;proc.vision_feature_select_strategy='default';proc.tokenizer.padding_side='left'
 history=historical_rows();indices=[];specs={};newrows=[]
 for j,r in enumerate(rows):
  q=r['question_id'];p=HIST/'input_specs'/(q+'.json');git_verified(p);s=read(p);new=cpu_inputs(proc,r)
  assert all(digest(new[k])==digest(s[k]) for k in ['prompt_text','prompt_ids','Q','R','I','ASSISTANT','ASSISTANT_ids','input_hashes','layout','groups','offsets'])
  assert sha(r['image_path'])==r['image_sha256'] and len(r['answers'])==10 and s['ASSISTANT_ids']==[319,1799,9047,13566]
  for c in 'ABCDE':
   rr=history[q][c];cp=ROOT/rr['checkpoint'];git_verified(cp);v=read(cp);g=v['generation'];assert sha(cp)==rr['checkpoint_sha256'] and v['input_hashes']==s['input_hashes'] and v['image_sha256']==r['image_sha256']
   assert abs(v['score']-score(g['text'],r['answers']))<1e-12 and g['audit']['passed'] and v['weights_unchanged'] and v['image_features']==oldcp(q,'B')['image_features']
   assert g['own_cache'] and g['logits_to_keep']==1
   indices.append(dict(question_id=q,condition=c,path=str(cp.relative_to(ROOT)),sha256=sha(cp),score=v['score'],origin_commit=BASE))
  target=OUT/'input_specs'/(q+'.json');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);specs[q]=sha(target)
  newrows.append(dict(**r,worker=shards[q],stage2_subset=q in selected))
  if (j+1)%50==0:print('CPU_INPUT_REUSE_AUDIT',j+1,flush=True)
 write(OUT/'sample_manifest.json',dict(status='FROZEN',source_commit=BASE,historical_manifest_sha256=sha(oldm),n=300,rows=newrows))
 csvwrite(OUT/'historical_reuse_index.csv',indices)
 historical_acc={c:100*sum(float(history[q][c]['score']) for q in history)/300 for c in 'ABCDE'}
 for c,expected in zip('ABCDE',[45.23333333333333,21.83333333333333,25.2,24.5,27.5]):assert abs(historical_acc[c]-expected)<1e-9
 # Three previously frozen cases; last is historical abnormal B, retained unchanged.
 oldsmoke=read(HIST/'smoke_selection.json')['question_ids'];smoke=[oldsmoke[0],oldsmoke[1],oldsmoke[-1]]
 write(OUT/'smoke_selection.json',dict(question_ids=smoke,both_workers_same_cases=True,rule='First two and last question in historical frozen smoke list; includes prior abnormal output'))
 comparisons=comparison_design();deps={n:__import__(n).__version__ for n in ['torch','transformers','numpy','PIL']};assert deps['transformers']=='4.49.0'
 protocol=dict(UTC=utc(),base_commit=BASE,branch='codex/textvqa-visual-reaccess-budget',model=str(MODEL),revision=MODEL.name,dependencies=deps,python=sys.version,UUIDS=UUIDS,model_manifest_sha256=sha(OUT/'model_manifest.json'),layer_sets_sha256=sha(OUT/'layer_sets.json'),subset_sha256=sha(OUT/'subset_120.json'),shards_sha256=sha(OUT/'shards.json'),conditions=design['conditions'],curve_aliases=design['curve_aliases'],native_A='Truly native all paths open; observed hooks must be exact no-op',other_queries='All original R incl four ASSISTANT subwords, all Image Queries and non-target mask entries native',mask='Only Pminus1/true Decode to original Image Keys; -inf softmax-pre; open layer rows native unchanged',generation=dict(FP16=True,eager=True,greedy=True,max_new_tokens=32,EOS=2,logits_to_keep=1,independent_real_KV=True),no_extra_OCR=True,no_prompt_changes=True,new_stage1=2100,new_stage2=1440,total_new=3540,historical_reuse=1500,canonical_logical_records=5040,comparisons=comparisons,Holm='Separate prespecified families: all stage1 comparisons on300; all stage2 comparisons on120. Ordinary95 CI pointwise not simultaneous.',family_sizes={k:len(v) for k,v in comparisons.items()},bootstrap_replicates=20000,bootstrap_seed=SEED,bootstrap_unit='Image SHA whole clusters; equal question mean; paired differences',p_value='Two-sided image-cluster sign-flip 20000 draws plus-one',random_mean='Average three seeds within each question before statistics; not360 independent examples',descriptive_targets=dict(gap_recovery=[.5,.75,.9],Full_minus_pp=2),noninferiority=False,subset_not_independent=True,no_training=True,no_search=True,no_physical_speedup_claims=True,phase_barrier='Each worker completes its full stage1 shard, then waits for both stage1_done files before frozen stage2; no adaptive selection',disk_reserve_GiB=20)
 write(OUT/'protocol.json',protocol)
 protocol['Figure_F_selection']='First frozen-manifest case in each of six predefined categories; recovery/harm tests Both8 then Both16, all cases retained; no data/condition changes based on cards'
 write(OUT/'protocol.json',protocol)
 # Features based on reference annotation form; never used for mask selection.
 from receiver_cross_dataset.scoring import normalize
 desc=[]
 import re
 for r in newrows:
  words=re.findall('[A-Za-z]+',r['question']);modal=collections.Counter(normalize(a) for a in r['answers']).most_common(1)[0][0];first=words[0].lower() if words else 'other'
  desc.append(dict(question_id=r['question_id'],first_word=first,reference_length='1' if len(modal.split())<=1 else '2-3' if len(modal.split())<=3 else '4+',answer_form='digit_related' if any(x.isdigit() for x in modal) else 'alphabetic_word' if any(x.isalpha() for x in modal) else 'other',modal_reference=modal))
 csvwrite(OUT/'descriptive_features.csv',desc)
 write(OUT/'cpu_validation.json',dict(passed=True,UTC=utc(),n=300,reused_records=1500,all_Git_bytes_verified=True,all_image_input_prompt_token_offset_hashes_exact=True,model_files_rehashed=True,historical_accuracies=historical_acc,input_specs_sha256=specs,manifest_sha256=sha(OUT/'sample_manifest.json'),shards_disjoint_complete=True,layer_sets_nested=True,subset_selected_without_scores=True,no_GPU_initialized=not torch.cuda.is_initialized()))
 write(OUT/'source_manifest.json',dict(UTC=utc(),files=sources(),protocol_sha256=sha(OUT/'protocol.json')))
 print('CPU_FREEZE_PASS',json.dumps(protocol['family_sizes']),flush=True)
if __name__=='__main__':main()
