import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import torch,numpy as np,sys
from transformers import AutoProcessor
from receiver_cross_dataset.prepare import cpu_inputs
torch.set_num_threads(1)
def git_verified(p):
 rel=str(p.relative_to(ROOT.parent));expected=subprocess.check_output(['git','-C',str(ROOT.parent),'show',BASE+':'+rel]);assert p.read_bytes()==expected
 return sha(p)
def main():
 guard();assert read(OUT/'CPU_test_suite.json')['passed']
 oldm=OLD/'textvqa/sample_manifest.json';git_verified(oldm);m=read(oldm);assert len(m['rows'])==300
 oldrows=baseline_rows();assert len(oldrows)==300
 model=read(OLD/'model_manifest.json');git_verified(OLD/'model_manifest.json')
 for name,meta in model['files'].items():assert sha(MODEL/name)==meta['sha256'],name
 write(OUT/'model_manifest.json',dict(**model,reverified_UTC=utc(),historical_manifest_sha256=sha(OLD/'model_manifest.json')))
 proc=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True);proc.patch_size=14;proc.num_additional_image_tokens=1;proc.vision_feature_select_strategy='default';proc.tokenizer.padding_side='left'
 rows=[];indices=[];specsha={};examples=[]
 for j,r in enumerate(m['rows']):
  q=r['question_id'];p=OLD/'textvqa/input_specs'/(q+'.json');git_verified(p);s=read(p);new=cpu_inputs(proc,r)
  assert all(new[k]==s[k] for k in ['prompt_text','prompt_ids','Q','R','I','ASSISTANT','ASSISTANT_ids','input_hashes','layout'])
  assert all(digest(new[k])==digest(s[k]) for k in ['groups','offsets'])
  assert sha(r['image_path'])==r['image_sha256'] and len(r['answers'])==10
  assert s['ASSISTANT_ids']==[319,1799,9047,13566] and set(s['ASSISTANT'])<=set(s['R'])
  assert s['layout']['A']==[s['layout']['prompt_length']-1] and not set(s['R'])&set(s['layout']['A'])
  cp=OLD/oldrows[q]['checkpoint'];git_verified(cp);c=read(cp);g=c['generation']
  assert c['writers']==s['R'] and c['input_hashes']==s['input_hashes'] and c['image_sha256']==r['image_sha256']
  assert c['source_manifest_sha256']==sha(OLD/'source_manifest.json') and c['input_spec_sha256']==sha(p)
  assert abs(c['score']-score(g['text'],r['answers']))<1e-12 and abs(float(oldrows[q]['score'])-c['score'])<1e-12
  assert c['weights_unchanged'] and c['hook_cleanup'] and c['input_hashes_unchanged'] and g['audit']['passed'] and all(x==0 for x in g['audit']['forbidden_max_by_layer'])
  assert g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1 and c['MLP_calls']==[g['length']]*32
  target=OUT/'input_specs'/(q+'.json');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);specsha[q]=sha(target)
  rows.append(dict(**r,input_hashes=s['input_hashes'],input_spec_sha256=specsha[q],prompt_ids_sha256=digest(s['prompt_ids'])))
  indices.append(dict(question_id=q,path=str(cp.relative_to(ROOT)),sha256=sha(cp),score=c['score'],answer=g['text'],token_ids=json.dumps(g['token_ids']),canonical_commit=BASE))
  if g['truncated'] or g['repeated_trigram']:examples.append(q)
  if (j+1)%50==0:print('CPU_INPUTS',j+1,flush=True)
 assert len(set(r['question_id'] for r in rows))==300
 write(OUT/'sample_manifest.json',dict(status='FROZEN',UTC=utc(),n=300,source_commit=BASE,historical_manifest_sha256=sha(oldm),rows=rows))
 csvwrite(OUT/'baseline_reuse_index.csv',indices)
 smoke=read(OLD/'textvqa/smoke_selection.json')['question_ids'];extra=next((q for q in examples if q not in smoke),None)
 if extra:smoke.append(extra)
 write(OUT/'smoke_selection.json',dict(question_ids=smoke,rule='Original seven frozen smoke questions, plus first manifest-order historical truncated/repeated B example not already included. No formal sample changes.',historical_abnormal_added=extra))
 deps={n:__import__(n).__version__ for n in ['torch','transformers','numpy','PIL']};assert deps['transformers']=='4.49.0'
 protocol=dict(UTC=utc(),base_commit=BASE,branch='codex/textvqa-partial-visual-layer-recovery',model=str(MODEL),revision=MODEL.name,model_manifest_sha256=sha(OUT/'model_manifest.json'),dependencies=deps,python=sys.version,conditions=NAMES,fixed_layers=LAYERS,layer_index='zero based language decoder',native_A='All original native connections including Pminus1 and every decode layer; zero experiment masks',B='All native R including four ASSISTANT subwords open at all32 layers; Pminus1/decode image edges forbidden all32',C='B except Pminus1 image read allowed only14/19',D='B except true decode image read allowed only14/19',E='B except both phases image read allowed only14/19',unchanged='Original image queries, all R, non-image edges, MLP, residual, QKV, positions, full independent real KV and prefix',generation=dict(FP16=True,eager=True,greedy=True,max_new_tokens=32,logits_to_keep=1,EOS=2),prompt='Exactly frozen historical no-OCR prompt and tokenizer; no added OCR text',mask='Softmax-pre -inf on original image keys only; restored layers use native mask without rescaling',GPU_UUID=UUID,n_questions=300,new_generations=1200,reused_B=300,comparisons=COMPARISONS,interaction='E-C-D+B on per-question official soft score',bootstrap_replicates=20000,bootstrap_seed=SEED,bootstrap_unit='Image SHA256 whole cluster, question-equal mean',test='Two-sided paired image-cluster sign-flip randomization; 20000 draws; plus-one p',Holm_family='Eight fixed tests: seven prescribed pairwise comparisons plus interaction; ordinary pointwise 95% bootstrap CI are not simultaneous CI',scorer='Pinned official TextVQA/MMF leave-one-out10-annotation consensus; no single gold exact match or NLL',no_noninferiority_margin=True,abnormal_policy='Retain every capped/repeated/empty/EOS output in all statistics',no_training=True,no_search=True,disk_reserve_GiB=20)
 write(OUT/'protocol.json',protocol)
 write(OUT/'CPU_validation.json',dict(passed=True,UTC=utc(),n=300,canonical_git_bytes_verified=True,model_files_sha256_verified=True,images_verified=True,original_prompt_tokens_offsets_exact=True,all_input_tensor_hashes_rebuilt_exact=True,ASSISTANT_four_subwords_preserved=True,baseline_checkpoints_reuse_verified=300,input_specs_sha256=specsha,manifest_sha256=sha(OUT/'sample_manifest.json'),no_GPU_initialized=not torch.cuda.is_initialized()))
 write(OUT/'source_manifest.json',dict(UTC=utc(),files=sources(),protocol_sha256=sha(OUT/'protocol.json')))
 print('CPU_FREEZE_PASS',flush=True)
if __name__=='__main__':main()
