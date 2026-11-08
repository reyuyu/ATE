"""CPU-only strict historical cohort, processor, scorer and reuse audit."""
import csv,gzip,json,tarfile,platform,subprocess
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor
from .common import *
from ..p1_layerwise_accuracy.engine import tensor_hash
from ..benchmarks.scoring import textvqa_score

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 samples=load_samples();assert len(samples)==500
 keys=list(samples);assert keys==sorted(keys)
 commits={n:(AUDIT/n/'source_commit.txt').read_text().strip() for n in ['middle','l11','l10','l12']}
 source_checks={}
 for name in commits:
  m=json.loads(gzip.decompress((AUDIT/name/'sample_manifest.json.gz').read_bytes()))
  assert list(m)==keys and m==samples,('manifest',name)
  p=read(AUDIT/name/'protocol.json')
  assert p['revision']==REV and p['transformers']=='4.49.0' and p['attention']=='eager' and p['dtype']=='FP16' and p['greedy'] and p['max_new_tokens']==32
  sources=p.get('inference_sources',p.get('source_hashes',{}))
  good=[]
  for path,expected in sources.items():
   actual=subprocess.check_output(['git','show',commits[name]+':'+path],cwd=ROOT)
   assert hashlib.sha256(actual).hexdigest()==expected,(name,path)
   good.append(path)
  source_checks[name]={'verified_files':len(good),'protocol_sha256':sha(AUDIT/name/'protocol.json')}
 rows=[json.loads(x) for x in (DATA/'inputs.jsonl').read_text().splitlines()]
 current={str(x['question_id']):x for x in rows};assert len(current)==5000
 gold=read(DATA/'answers.json');inventory=read(BUNDLE/'image_sha256.json')
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
 processor.patch_size=14;processor.num_additional_image_tokens=1;processor.vision_feature_select_strategy='default';processor.tokenizer.padding_side='left'
 # Historical bindings are authenticated by payload SHA and the CSV binding SHA.
 bindings={};raw_count=0
 archive=ROOT/'vlm_research/results/middle10_decode_early_late_recovery/raw_quality_checkpoints.tar.gz'
 with tarfile.open(archive) as t:
  for member in t:
   if member.isfile() and member.name.endswith('/Early.json'):
    wrap=json.load(t.extractfile(member));raw=wrap['payload'];assert digest(raw)==wrap['sha256']
    assert raw['key'] in samples and raw['sample_sha256']==digest(samples[raw['key']])
    bindings[raw['key']]=raw['input_binding'];raw_count+=1
 assert len(bindings)==raw_count==500
 cpu_bindings={};mapping={};pixels={}
 for index,key in enumerate(keys):
  s=samples[key];r=current[s['question_id']]
  assert r['image_id']==s['image_id'] and r['question']==s['question']
  assert gold[s['question_id']]==s['answer'] and len(s['answer'])==10
  assert s['prompt_text']==s['question']+'\nAnswer the question using a single word or phrase.'
  assert digest(s['input_mapping']['prompt_ids'])==s['prompt_sha256']
  image=Path(r['image_path']);rel=str(image.relative_to(BUNDLE));meta=inventory[rel]
  assert meta['sha256']==s['image_sha256'] and image.is_file() and image.stat().st_size==meta['bytes']
  chat=processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=s['prompt_text'])])],tokenize=False,add_generation_prompt=True)
  assert chat==s['input_mapping']['rendered_prompt']
  with Image.open(image) as im:
   inputs=processor(images=im.convert('RGB'),text=chat,return_tensors='pt').to('cpu',dtype=torch.float16)
  assert inputs['input_ids'][0].tolist()==s['input_mapping']['prompt_ids']
  hashes={k:tensor_hash(v) for k,v in inputs.items()}
  assert hashes==bindings[key]['input_hashes'],('input_tensors',key,hashes,bindings[key]['input_hashes'])
  m=s['input_mapping']
  binding=dict(input_hashes=hashes,prompt_ids=m['prompt_ids'],Q=m['parts']['Q'],ASSISTANT=m['parts']['ASSISTANT'],Hint=m['parts']['Hint'],Options=m['parts']['Options'],writers=m['R'],R=m['R'],Pminus1=m['Pminus1'])
  assert binding==bindings[key]
  cpu_bindings[key]=digest(binding);mapping[key]=str(image)
  if index%100==0:print('CPU inputs verified',index+1,flush=True)
 historical={};all_rows={}
 for name in commits:
  rr=list(csv.DictReader((AUDIT/name/'per_question.csv').open()))
  lookup={}
  for r in rr:
   r.setdefault('first_top1_token',str(json.loads(r['token_ids'])[0]))
   assert int(r['first_top1_token'])==json.loads(r['token_ids'])[0]
   key=r['key'];assert key in samples
   assert r['question_id']==samples[key]['question_id'] and r['image_sha256']==samples[key]['image_sha256']
   assert json.loads(r['gold_answers'])==samples[key]['answer']
   assert r['input_binding_sha256']==cpu_bindings[key]
   assert abs(textvqa_score(r['answer'],samples[key]['answer'])-float(r['score']))<1e-12
   assert processor.tokenizer.decode(json.loads(r['token_ids']),skip_special_tokens=True).strip()==r['answer']
   assert (key,r['condition']) not in lookup
   lookup[key,r['condition']]=r
  all_rows[name]=lookup
 for condition,(name,label) in REUSE.items():
  # Fail closed if the historical label is missing; no automatic baseline rerun.
  selected=[all_rows[name][key,label] for key in keys]
  assert len(selected)==500
  historical[condition]=selected
 for name,lookup in all_rows.items():
  for condition,(origin,label) in REUSE.items():
   for key in keys:
    if (key,label) in lookup:
     a=lookup[key,label];b=all_rows[origin][key,label]
     assert all(a[f]==b[f] for f in ['answer','score','token_ids','input_binding_sha256','first_top1_token','first_logits_sha256'])
 atomic(OUT/'input_audit.json',dict(status='PASS',questions=500,image_clusters=len({s['image_sha256'] for s in samples.values()}),order_sha256=digest(keys),manifest_sha256=sha(AUDIT/'middle/sample_manifest.json.gz'),image_verification='Trusted SHA inventory bound to previously SHA-verified backup archive; selected image byte sizes and all 500 FP16 processor tensor hashes matched historical bindings. No global image SHA recomputation.',all_500_input_tensor_bindings='exact',all_four_manifests='exact',gold='all ordered ten answers exact',prompt='Image-only exact',historical_predictions_official_rescore='PASS',source_commits=commits,source_checks=source_checks))
 atomic(AUDIT/'runtime_mapping.json',mapping);atomic(AUDIT/'input_bindings.json',cpu_bindings)
 with gzip.open(OUT/'reused_predictions.json.gz','wt',encoding='utf-8') as f:json.dump(historical,f,ensure_ascii=False,sort_keys=True)
 atomic(OUT/'protocol.json',dict(experiment='TextVQA500 Early/Middle/Late Decode visual re-access layer budget',exploratory=True,model='llava-hf/llava-1.5-7b-hf',revision=REV,dtype='FP16',attention='eager',transformers='4.49.0',torch='2.6.0+cu124',greedy=True,max_new_tokens=32,logits_to_keep=1,questions=500,image_clusters=309,order_sha256=digest(keys),budgets=BUDGETS,decode_image_layers=CONDITIONS,prefill_including_Pminus1='Native in all32 layers',intervention='Only true autoregressive Decode attention logits on original image keys are hard-masked to -inf outside allowed layers. Native HF attention, RoPE, QKV, MLP, residual and real DynamicCache otherwise unchanged.',scorer='Pinned official TextVQA M4C ten-answer consensus soft score',reuse={k:dict(evidence=n,label=l,commit=commits[n]) for k,(n,l) in REUSE.items()},new_conditions=[c for c in CONDITIONS if c not in REUSE],max_new_generations=9500,new_condition_order='NoImage, then Early increasing k, Middle increasing k excluding reused14, Late increasing k excluding reused21',bootstrap=dict(unit='Image SHA cluster, paired across conditions, equal question weighted mean',replicates=20000,seed=20261108,ci='ordinary pointwise percentile 95%; not simultaneous',comparison_family='21 same-budget pairwise tests; 9 prespecified primary contrasts at k14/16/21, remaining12 secondary',multiplicity='Report ordinary unadjusted CIs and two-sided centered-bootstrap p; Holm adjustment separately for primary9 and secondary12, labeled exploratory.'),no_physical_speedup_claim=True,old_300_joint_restriction_excluded=True,stop_on=['input mismatch','protocol error','OOM','non-finite logits'],source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
 print('STRICT CPU AUDIT PASS; reuse', {c:sum(float(r['score']) for r in rr)/5 for c,rr in historical.items()},flush=True)
if __name__=='__main__':main()
