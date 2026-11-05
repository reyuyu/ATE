import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from pathlib import Path
import sys,json,hashlib,subprocess,datetime,contextlib,collections
W=Path('/data/lk/ATE-worktrees/textvqa-decode-l0-l1-factorial');R=W/'vlm_research';sys.path.insert(0,str(R))
from receiver_cross_dataset.common import read,write,sha,tensors_hash,MODEL,utc
from transformers import AutoProcessor
from PIL import Image
from query_role_attention.engine import roles
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
from textvqa_decode_l0_l1_factorial.policy import CONDITIONS,OUT,INPUTS,UUIDS
P=Path('/data/lk/ATE-resources/experiments/textvqa_decode_l0_l1_factorial');H=Path('/data/lk/ATE-worktrees/textvqa-single-layer-visual-ablation/vlm_research/results/textvqa_single_layer_visual_ablation');BASE='4408b862e4760b6aadcb5135d2640f422778938b'
selection=read(P/'data_selection.json');old=selection['old_rows'];new=selection['new_rows'];ev=TextVQAAccuracyEvaluator()
def score(text,answers):
 with open(os.devnull,'w') as z,contextlib.redirect_stderr(z):return ev.eval_pred_list([dict(pred_answer=text,gt_answers=answers)])
model=read(H/'execution_model_manifest.json')
for name,v in model['files'].items():assert sha(Path(model['path'])/name)==v['sha256'],name
write(OUT/'execution_model_manifest.json',model)
frozen=read(H/'execution_source_manifest.json')
for name,h in frozen['files'].items():assert sha(name)==h,name
print('MODEL_AND_LEGACY_SOURCE_SHA_PASS',flush=True)
proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True);proc.patch_size=14;proc.num_additional_image_tokens=1;proc.vision_feature_select_strategy='default';proc.tokenizer.padding_side='left'
reuse0={(r['question_id'],r['condition']):r for r in read(H/'historical_reuse_audit.json')['checkpoints']};reuse=[];rows=[]
batch=subprocess.Popen(['git','cat-file','--batch'],cwd=W,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git_sha(path):
 rel='vlm_research/'+str(path).split('/vlm_research/',1)[1];batch.stdin.write((BASE+':'+rel+'\n').encode());batch.stdin.flush();line=batch.stdout.readline().decode();assert ' missing' not in line,line
 n=int(line.split()[2]);data=batch.stdout.read(n);assert batch.stdout.read(1)==b'\n';return hashlib.sha256(data).hexdigest()
for n,r0 in enumerate(old+new):
 r=dict(r0);q=r['question_id'];isold=n<len(old);r['cohort']='old300' if isold else 'new_images';assert sha(r['image_path'])==r['image_sha256']
 payload=r['question']+'\nAnswer the question using a single word or phrase.'
 prompt=proc.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=payload)])],add_generation_prompt=True)
 with Image.open(r['image_path']) as image:inputs=proc(images=image.convert('RGB'),text=prompt,return_tensors='pt')
 inputs['pixel_values']=inputs['pixel_values'].half();s=dict(question_id=q,question=r['question'],prompt_text=prompt,prompt_ids=inputs['input_ids'][0].tolist(),input_hashes=tensors_hash(inputs),layout=roles(inputs,32000));assert len(s['layout']['I'])==576
 if isold:
  assert s['input_hashes']==r['input_hashes'];op=Path('/data/lk/ATE-worktrees/textvqa-coco-middle-transfer/vlm_research/results/textvqa_visual_reaccess_budget_exploration/input_specs')/(q+'.json');assert sha(op)==r['input_spec_sha256'] and sha(op)==git_sha(op);assert s['prompt_ids']==read(op)['prompt_ids']
  for c,legacy in [('A','Middle-minus-L0'),('B','Middle'),('Native','Native'),('Decode-None','Decode-None')]:
   if c=='A':path=H/'checkpoints'/q/'Middle-minus-L0.json';expected=None
   else:path=Path(reuse0[(q,legacy)]['checkpoint_path']);expected=reuse0[(q,legacy)]['sha256']
   h=sha(path);v=read(path);assert h==git_sha(path) and (expected is None or h==expected);assert v['input_hashes']==s['input_hashes'] and v['image_sha256']==r['image_sha256'] and v['generation']['audit']['passed'];assert abs(score(v['generation']['text'],r['answers'])-v['score'])<1e-12
   g=v['generation'];assert g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1
   if c=='A':assert v['runtime_git_commit']=='f8f6f0550b6b0575ce6eb2f973f37b98c3367280' and v['definition']==dict(family='Middle-minus',layer=0,reference='Middle')
   if c in ['B','Decode-None']:assert v['allowed_layers']['Decode']==CONDITIONS[c] and v['allowed_layers']['Pminus1']==list(range(32))
   source_hash=v.get('source_manifest_sha256',v.get('execution_source_manifest_sha256'));protocol_hash=v.get('protocol_sha256',v.get('execution_protocol_sha256'));root=path.parents[2]
   assert any(sha(root/name)==source_hash for name in ['source_manifest.json','execution_source_manifest.json'] if (root/name).exists())
   assert any(sha(root/name)==protocol_hash for name in ['protocol.json','execution_protocol.json'] if (root/name).exists())
   reuse.append(dict(question_id=q,condition=c,legacy_condition=legacy,checkpoint_path=str(path),sha256=h,input_hashes=s['input_hashes'],source_manifest_sha256=source_hash,protocol_sha256=protocol_hash,score=v['score'],first_logits_sha256=v['first_logits_sha256'],git_base_blob_exact=True))
 write(INPUTS/(q+'.json'),s);r.update(input_hashes=s['input_hashes'],input_spec_sha256=sha(INPUTS/(q+'.json')));rows.append(r)
 if (n+1)%100==0:print('CPU_INPUT_AUDIT',n+1,flush=True)
batch.stdin.close();batch.wait();assert batch.returncode==0
for c,expected in [('A',42.06666666666667),('B',43.13333333333333),('Native',45.23333333333333),('Decode-None',31.4)]:assert abs(sum(x['score'] for x in reuse if x['condition']==c)/3-expected)<1e-9
write(OUT/'historical_reuse_audit.json',dict(passed=True,source_commit=BASE,checkpoints=reuse,all1200_git_blob_SHA_exact=True,all_scores_recomputed=True,all300_native_inputs_rebuilt=True,legacy_source_files_verified=len(frozen['files']),historical_logit_limit='Historical checkpoint stores first-logits SHA and full token IDs, not all step vectors. Full valid logits compared against frozen historical engine replay in Smoke.'))
write(OUT/'sample_manifest.json',dict(rows=rows,old_questions=len(old),new_questions=len(new),official_annotation_sha256=selection['annotation_sha256'],frozen_before_new_predictions=True))
by={};shardcounts=[0,0];groups=collections.defaultdict(list)
for r in rows:groups[r['image_sha256']].append(r['question_id'])
for h,qs in sorted(groups.items(),key=lambda kv:(-len(kv[1]),kv[0])):
 worker=min(range(2),key=lambda i:(shardcounts[i],i));shardcounts[worker]+=len(qs)
 for q in qs:by[q]=worker
assert len(by)==len(rows) and all(len({by[q] for q in qs})==1 for qs in groups.values())
write(OUT/'shards.json',dict(by_question=by,worker_question_counts=shardcounts,whole_image_no_intersection=True,UUIDS=UUIDS))
write(OUT/'layer_sets.json',{c:dict(allowed_decode_image_layers=v,allowed_prefill_image_layers=list(range(32)),retained_decode_layers=len(v)) for c,v in CONDITIONS.items()})
write(OUT/'smoke_selection.json',dict(question_ids=['39419','34726','39145',new[0]['question_id']],rule='Three previously fixed Smoke IDs plus first SHA-selected new question, frozen before predictions'))
start=datetime.datetime(2026,11,5,10,tzinfo=datetime.timezone.utc).timestamp()
protocol=dict(name='TextVQA Decode L0 x L1 Factorial',base_commit=BASE,model_revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',dtype='float16',attention='eager',transformers='4.49.0',greedy=True,max_new_tokens=32,logits_to_keep=1,Prefill='Native all32 including P-1, R and Image Query',intervention='Only actual incremental Decode Query to original576 Image Key; pre-softmax -inf on closed layers',conditions=CONDITIONS,old300_reuse=['A','B','Native','Decode-None'],old300_new=['C','D'],new_images_all_six_fresh=True,stage_order=['both_GPU_smoke','new_images_all6','old300_CD'],training=False,OCR_prompt=False,physical_KV_deletion=False,start_epoch=start,deadline_epoch=start+6*3600,gpu_deadline_epoch=start+5.5*3600,bootstrap_repetitions=20000,seed=20261105,primary=dict(cohort='new_images',contrast='A-D',NI_margin_pp=-1.,one_sided_confidence=.95,lower_bound='5th percentile of image-cluster paired percentile Bootstrap',support='lower bound strictly greater than -1pp'),auxiliary=dict(family=['B-A','C-A','D-B','D-C','C-B','D-C-B+A','A-Native','B-Native','C-Native','D-Native'],multiplicity='Holm two-sided image-cluster sign-flip Monte Carlo p values,20000 draws,+1 correction; separate fixed10-test family per cohort',ordinary_CI='Unadjusted pointwise percentile95%, not simultaneous or NI evidence'),bootstrap_estimator='Equal weight per question; resample image SHA clusters with replacement, retain every question, divide resampled score sums by resampled question count',scorer='Official TextVQAAccuracyEvaluator ten-answer consensus soft score',manifest_sha256=sha(OUT/'sample_manifest.json'),shards_sha256=sha(OUT/'shards.json'),model_manifest_sha256=sha(OUT/'execution_model_manifest.json'),layer_sets_sha256=sha(OUT/'layer_sets.json'),annotation_sha256=selection['annotation_sha256'])
write(OUT/'protocol.json',protocol)
print('FREEZE_CPU_DONE',len(rows),shardcounts,flush=True)
