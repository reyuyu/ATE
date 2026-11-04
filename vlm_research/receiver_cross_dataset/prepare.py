import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
from .selection import *
import torch,collections,re,sys,platform
torch.set_num_threads(1)
from transformers import AutoProcessor
from PIL import Image
from semantic_visual_writers.groups import token_groups,intact
from query_role_attention.engine import roles
from gqa3000_token_visual_writing.engine import cpu_mask
def cpu_inputs(proc,r):
 text=r['question']+'\nAnswer the question using a single word or phrase.'
 prompt=proc.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=text)])],add_generation_prompt=True)
 with Image.open(r['image_path']) as im:inputs=proc(images=im.convert('RGB'),text=prompt,return_tensors='pt')
 inputs['pixel_values']=inputs['pixel_values'].half()
 layout=roles(inputs);ids=inputs['input_ids'][0].tolist()
 groups,Q,words=token_groups(proc.tokenizer,r['question'],prompt,ids,layout['I'],layout['R'])
 A=groups[-1]['positions'];assert len(A)==4 and groups[-1]['words']==['ASSISTANT'] and groups[-1]['decoded'].strip()=='ASSISTANT'
 assert layout['prompt_length']-1 not in Q+A and ids[-1]==29901
 assert intact(Q,groups) and len(layout['I'])==576
 return dict(question_id=r['question_id'],question=r['question'],prompt_text=prompt,prompt_ids=ids,Q=Q,R=layout['R'],I=layout['I'],ASSISTANT=A,ASSISTANT_ids=[ids[p] for p in A],groups=groups,offsets=proc.tokenizer(prompt.replace('<image>','<image>'*576),return_offsets_mapping=True)['offset_mapping'],input_hashes=tensors_hash(inputs),layout=layout)
def structure(specs,manifest):
 rows=[];byid={r['question_id']:r for r in manifest}
 for s in specs:
  n=len(s['Q']);w=firstword(s['question']);cat=w if w in ['what','which','how'] else 'yes/no' if w in ['is','are','does','do'] else 'where/who/why' if w in ['where','who','why'] else 'other'
  rows.append(dict(question_id=s['question_id'],first_word=w,first_word_group=cat,Q_tokens=n,R_tokens=len(s['R']),what_which=w in ['what','which'],length_ge15=n>=15,length_ge18=n>=18,trigger15=trigger(s['question'],n,15),trigger18=trigger(s['question'],n,18),answer_type=byid[s['question_id']]['answer_type']))
 return rows
def main():
 guard();assert not torch.cuda.is_initialized()
 assert read(OUT/'CPU_test_suite.json')['passed'] and read(OUT/'scorer_validation.json')['passed']
 proc=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
 proc.patch_size=14;proc.num_additional_image_tokens=1;proc.vision_feature_select_strategy='default';proc.tokenizer.padding_side='left'
 prior=OUT.parent/'semantic_single_gpu_search3h'
 historical=[]
 for threshold,phase in [(15,'phase6_fixed_compositions'),(18,'phase7_fixed_refinements')]:
  frozen=read(prior/phase/'winner_freeze.json')['positions'];passed=0
  for q,pos in frozen.items():
   s=read(pathlib.Path('/data1/likai/ATE-worktrees/gqa5h-adaptive-token-receiver-search/vlm_research/results/gqa5h_receiver_search/input_specs')/(q+'.json'))
   actual=sorted(set(s['Q'])|set(s['groups'][-1]['positions'])) if trigger(s['question'],len(s['Q']),threshold) else s['Q']
   assert actual==pos,(q,threshold);passed+=1
  historical.append(dict(threshold=threshold,questions=passed,all_exact=True,winner_sha256=sha(prior/phase/'winner_freeze.json')))
 # Official consensus fixtures distinguish leave-one-out scoring from min(count/3,1).
 from .scoring import score,normalize
 for n,expected in [(0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)]:
  answers=['white']*n+['black']*(10-n)
  for dataset in ['vqa_v2','textvqa']:assert abs(score(dataset,'White',answers)-expected)<1e-12
 assert normalize('The two cats.')=='2 cats' and normalize('yes!')=='yes'
 audits=[]
 for dataset in ['vqa_v2','textvqa']:
  d=OUT/dataset;m=read(d/'sample_manifest.json')
  if m['status']!='FROZEN':continue
  specs=[cpu_inputs(proc,r) for r in m['rows']];labels,audit=random_labels(specs);write(d/'random_trigger_audit.json',audit)
  budgets=[]
  for s in specs:
   s['writers']=masks(s,labels[s['question_id']]);assert s['ASSISTANT_ids']==[319,1799,9047,13566]
   # Exact edge audits for prefill and real decode, representative layout per input.
   for name,w in s['writers'].items():
    assert intact(w,s['groups']);a,b,blocked=cpu_mask(s['layout'],w)
    assert torch.equal(a[0,0][~blocked],b[0,0][~blocked]) and torch.isneginf(b[0,0][blocked]).all()
    assert int(blocked.sum())==(len(s['R'])-len(w)+1)*576
    a,b,blocked=cpu_mask(s['layout'],w,s['layout']['prompt_length']);assert int(blocked.sum())==576
    budgets.append(dict(question_id=s['question_id'],condition=name,k=len(w),R=len(s['R']),Q=len(s['Q']),R_ratio=len(w)/len(s['R']),positions=json.dumps(w)))
   write(d/'input_specs'/(s['question_id']+'.json'),s)
  rows=structure(specs,m['rows']);csvwrite(d/'trigger_analysis.csv',rows);csvwrite(d/'writer_budget.csv',budgets)
  summary=[]
  for field in ['first_word_group','Q_tokens','R_tokens','what_which','length_ge15','length_ge18','trigger15','trigger18','answer_type']:
   for value,n in sorted(collections.Counter(str(r[field]) for r in rows).items()):summary.append(dict(field=field,value=value,count=n,ratio=n/len(rows),scope='Frozen GPU sample'))
  csvwrite(d/'question_structure_summary.csv',summary)
  write(d/'CPU_validation.json',dict(passed=True,n=len(specs),all_native_offsets_exact=True,ASSISTANT_ids=specs[0]['ASSISTANT_ids'],ASSISTANT_subwords=4,Pminus1_excluded=True,whole_groups=True,mask_scope_exact=True,manifest_sha256=sha(d/'sample_manifest.json'),input_specs_sha256={s['question_id']:sha(d/'input_specs'/(s['question_id']+'.json')) for s in specs},no_GPU_initialized=not torch.cuda.is_initialized()))
  audits.append(dict(dataset=dataset,n=len(specs)))
  print('CPU_FREEZE',dataset,len(specs),flush=True)
 write(OUT/'cpu_validation.json',dict(passed=True,historical_rule_regressions=historical,scoring_fixtures=True,datasets=audits,UTC=utc()))
 deps={n:__import__(n).__version__ for n in ['torch','transformers','numpy','PIL']};assert deps['transformers']=='4.49.0'
 write(OUT/'protocol.json',dict(UTC=utc(),base_commit='644834d3e22194b4982758dabd59d3e479ef8ce3',model=str(MODEL),revision=MODEL.name,dependencies=deps,python=sys.version,conditions=CONDITIONS,thresholds=[15,18],firstwords=['what','which'],Question_definition='Native tokenizer offset overlap with original Question character span; whole groups; native Question subwords including punctuation',ASSISTANT_subwords=4,prompt='Original Question + newline + Answer the question using a single word or phrase.; unchanged native chat template',all32layers=True,Pminus1_decode_image_blocked=True,FP16=True,eager=True,greedy=True,max_new_tokens=32,logits_to_keep=1,one_GPU_UUID=UUID,seed=SEED,bootstrap_replicates=20000,bootstrap_unit='Image SHA256 cluster; question-equal weighted mean score',comparison_families='Six fixed comparisons per dataset separately; ordinary percentile95 CI; two-sided paired image-cluster signflip p (20000 draws) with Holm correction within six-comparison family; exploratory, not noninferiority',comparisons=[['Threshold-15','Question-Only'],['Threshold-18','Question-Only'],['Threshold-15','All-Writer'],['Threshold-18','All-Writer'],['Threshold-15','Budget-Matched Random'],['Threshold-18','Budget-Matched Random']],scorers=dict(vqa_v2='Official VQAEval punctuation/number/article/contraction normalization applied to both prediction and all ten annotations, then exact annotator leave-one-out consensus',textvqa='Pinned official MMF EvalAIAnswerProcessor and TextVQAAccuracyEvaluator'),deduplication='Only exact same question/input tensor hashes and identical writer positions; preserve every logical condition label',no_training=True,no_tuning=True,resource_limits=dict(disk_reserve_GiB=20,new_download_cap_GiB=4),source_hashes=sources()))
 write(OUT/'source_manifest.json',dict(UTC=utc(),files=sources(),CPU_manifest_sha256=sha(OUT/'protocol.json')))
 print('CPU_ALL_PASS',flush=True)
if __name__=='__main__':main()
