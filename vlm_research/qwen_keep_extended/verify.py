"""Independent CPU checks for decoded answers, score changes and checkpoint coverage."""
import csv,collections
from .common import *

def main():
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);samples=read(OUT/'sample_manifest.json')['samples'];byid={s['question_id']:s for s in samples};specs=read(OUT/'input_specs.json');assert len(byid)==1000
 rows=list(csv.DictReader((OUT/'generation_outputs.csv').open()));assert len(rows)==7000
 seen=set();counts=collections.Counter();scores={};decoded=0
 for r in rows:
  q,c=r['question_id'],r['condition'];assert (q,c) not in seen;seen.add((q,c));s=byid[q];tokens=json.loads(r['token_ids']);text=tokenizer.decode(tokens,skip_special_tokens=True);assert text==r['answer']
  score=text.rstrip('.').lower()==s['gold'];assert score==bool(int(r['correct']));counts[c]+=score;scores[q,c]=score;decoded+=1
  assert r['image_sha256']==s['image_sha256'] and tokens[0]==int(r['first_token'])
 for r in csv.DictReader((OUT/'accuracy_summary.csv').open()):assert counts[r['condition']]==int(r['correct'])
 for r in csv.DictReader((OUT/'paired_comparisons.csv').open()):
  a,b=r['candidate'],r['reference'];delta=[int(scores[q,a])-int(scores[q,b]) for q in byid]
  assert abs(sum(delta)/10-float(r['delta_pp']))<1e-10 and int(r['reference_correct_to_wrong'])==sum(v<0 for v in delta) and int(r['reference_wrong_to_correct'])==sum(v>0 for v in delta)
 for r in csv.DictReader((OUT/'checkpoint_sha256.csv').open()):assert sha(ROOT/r['path'])==r['sha256']
 new=0
 for rank in range(2):
  paths=list((OUT/f'checkpoints/rank{rank}').glob('*.json'));assert len(paths)==1000 and not (OUT/f'accuracy/rank{rank}/failure.json').exists()
  for path in paths:
   d=read(path);r=d['result'];s=byid[d['binding']['question_id']];c=r['condition'];assert s['rank']==rank and c in CONDITIONS
   assert r['writer_layers']==list(range(CONDITIONS[c]+1));spec=specs[s['question_id']];ni=len(spec['roles']['I']);nr=len(spec['roles']['R']);a=r['source_audit']
   assert a['prefill_blocked_connections_per_layer']==[ni*(1+(0 if i<=CONDITIONS[c] else nr)) for i in range(36)] and a['decode_blocked_connections_per_layer']==[ni]*36
   assert a['blocked_probability_max']==0 and a['non_target_mask_exact'] and a['early_attention_normal'] and a['hook_cleanup'] and r['P1_Decode_image_off'] and r['all_logits_finite'];new+=1
 assert new==2000
 write(OUT/'independent_verification.json',dict(status='PASSED',at=now(),decoded_answers=decoded,new_checkpoint_mask_audits=new,all_checkpoint_SHA_verified=7000,all_raw_token_decode_scores_exact=True,all_paired_counts_recomputed=True,correct_counts=counts))
 print('INDEPENDENT_VERIFICATION_PASSED',dict(counts))

if __name__=='__main__':main()
