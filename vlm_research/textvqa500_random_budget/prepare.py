"""CPU verification of the exact frozen cohort and all 500 FP16 input bindings."""
from .common import *
from collections import Counter
import subprocess,torch
from PIL import Image
from transformers import AutoProcessor
from ..p1_layerwise_accuracy.engine import tensor_hash

def main():
 p=read(OUT/'protocol.json');samples=load_samples();keys=list(samples);assert len(keys)==500 and keys==sorted(keys)
 old=read(OUT/'historical_protocol.json');assert digest(keys)==old['order_sha256'] and p['budgets']==old['budgets']
 assert sha(OUT/'sample_manifest.json.gz')==read(OUT/'historical_input_audit.json')['manifest_sha256']
 h=load_history();assert len(h)==23 and all(set(v)==set(keys) for v in h.values())
 rows={r['question_id']:r for r in map(json.loads,(DATA/'inputs.jsonl').read_text().splitlines())};gold=read(DATA/'answers.json');assert len(rows)==5000
 inventory=read(BUNDLE/'image_sha256.json');proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
 proc.patch_size=14;proc.num_additional_image_tokens=1;proc.vision_feature_select_strategy='default';proc.tokenizer.padding_side='left'
 paths={};bindings={}
 for i,(key,s) in enumerate(samples.items()):
  r=rows[s['question_id']];assert r['question']==s['question'] and r['image_id']==s['image_id'] and gold[s['question_id']]==s['answer'] and len(s['answer'])==10
  assert s['prompt_text']==s['question']+'\nAnswer the question using a single word or phrase.'
  path=Path(r['image_path']);meta=inventory[str(path.resolve().relative_to(BUNDLE.resolve()))]
  assert meta['sha256']==s['image_sha256'] and path.stat().st_size==meta['bytes']
  chat=proc.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=s['prompt_text'])])],tokenize=False,add_generation_prompt=True)
  assert chat==s['input_mapping']['rendered_prompt']
  with Image.open(path) as im:x=proc(images=im.convert('RGB'),text=chat,return_tensors='pt').to('cpu',dtype=torch.float16)
  m=s['input_mapping'];assert x['input_ids'][0].tolist()==m['prompt_ids'] and digest(m['prompt_ids'])==s['prompt_sha256']
  b=dict(input_hashes={k:tensor_hash(v) for k,v in x.items()},prompt_ids=m['prompt_ids'],Q=m['parts']['Q'],ASSISTANT=m['parts']['ASSISTANT'],Hint=m['parts']['Hint'],Options=m['parts']['Options'],writers=m['R'],R=m['R'],Pminus1=m['Pminus1'])
  expected=digest(b);assert all(v[key]['input_binding_sha256']==expected and v[key]['question_id']==s['question_id'] and v[key]['image_sha256']==s['image_sha256'] for v in h.values()),('binding',key)
  paths[key]=str(path);bindings[key]=expected
  if i%100==0:print('EXACT_FROZEN_INPUTS',i+1,flush=True)
 checks={}
 deps=['middle10_attribution/engine.py','middle10_attribution/common.py','p1_layerwise_accuracy/engine.py','p1_layerwise_accuracy/common.py','vlm_common.py','knockout_probe.py','benchmarks/scoring.py','benchmarks/_vendor/m4c_evaluator.py','c3v2_middle_reaccess/engine.py','ate_fulljoint_v2/engine.py','physical_handoff_fastpath/engine.py','middle10_recovery/engine.py']
 deps += ['textvqa500_budget/'+f.name for f in (ROOT/'vlm_research/textvqa500_budget').glob('*.py')]
 for name in deps:
  rel='vlm_research/'+name;frozen=subprocess.check_output(['git','show',REF+':'+rel],cwd=ROOT);assert hashlib.sha256(frozen).hexdigest()==sha(ROOT/rel),rel;checks[rel]=sha(ROOT/rel)
 LOCAL.mkdir(exist_ok=True);atomic(LOCAL/'runtime_mapping.json',paths);atomic(LOCAL/'input_bindings.json',bindings)
 atomic(OUT/'input_audit.json',dict(status='PASS',questions=500,image_clusters=len({s['image_sha256'] for s in samples.values()}),image_only=True,question_gold_order_exact=True,all_500_input_tensor_bindings_exact=True,manifest_sha256=sha(OUT/'sample_manifest.json.gz'),order_sha256=digest(keys),historical_result_sha256=sha(OUT/'historical_per_question.jsonl.gz'),trusted_image_inventory='Bound to SHA-verified original backup ZIP; SHA and sizes matched; no global image SHA recomputation',frozen_source_hashes=checks))
 print('RANDOM_CPU_PREFLIGHT_PASS',len(checks),flush=True)
if __name__=='__main__':main()
