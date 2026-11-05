"""Independent CPU audit of every decoded answer, official score and target event."""
import csv,collections,tarfile,hashlib
from .common import *

def main():
 p=processor_cpu();samples=read(OUT/'sample_manifest.json');bindings=read(OUT/'input_bindings.json');part=read(OUT/'partition.json');rows=list(csv.DictReader((OUT/'generation_outputs.csv').open()));assert len(rows)==2000
 seen=set();totals=collections.defaultdict(float);scores={}
 for r in rows:
  key,c=r['key'],r['condition'];assert (key,c) not in seen;seen.add((key,c));s=samples[key];tokens=json.loads(r['token_ids']);text=p.tokenizer.decode(tokens,skip_special_tokens=True).strip();assert text==r['answer']
  score=float(textvqa_score(text,s['answer']));assert abs(score-float(r['soft_score']))<1e-12;totals[c]+=score;scores[key,c]=score
  assert r['image_sha256']==s['image_sha256'] and int(r['first_token'])==tokens[0]
 for r in csv.DictReader((OUT/'accuracy_summary.csv').open()):assert abs(totals[r['condition']]/5-float(r['accuracy_percent']))<1e-10
 for r in csv.DictReader((OUT/'paired_comparisons.csv').open()):
  a,b=r['candidate'],r['reference'];d=[scores[k,a]-scores[k,b] for k in samples];assert abs(sum(d)/5-float(r['delta_pp']))<1e-10
  assert int(r['candidate_score_higher'])==sum(v>1e-12 for v in d) and int(r['candidate_score_lower'])==sum(v< -1e-12 for v in d)
 for r in csv.DictReader((OUT/'checkpoint_sha256.csv').open()):assert sha(ROOT/r['path'])==r['sha256']
 audited=0
 for key,s in samples.items():
  n=len(s['input_mapping']['prompt_ids'])
  for c in ['A','B']:
   d=read(OUT/f'checkpoints/{key}/{c}.json');g=d['generation'];a=g['audit'];t=len(g['token_ids']);assert d['binding']['input_sha256']==digest(bindings[key]) and d['binding']['sample_sha256']==digest(s)
   assert a['all_image_KV_retained'] and a['all32_R_image_open'] and a['all_non_target_masks_exact'] and a['all_non_target_o_proj_outputs_exact'] and a['hooks_clean']
   assert a['blocked_probability_max']==0 and g['dense_cache_lengths']==[n+t-1]*32 and g['all_logits_finite']
   assert all(count==[t]*32 for count in a['all_blocks_MLP_norm_counts'].values())
   assert len(a['layer_events'])==t*32 and len(a['answer_zero_events'])==t*4
   for e in a['layer_events']:assert e['blocked_connections_per_head']==576 and e['blocked_probability_max']==0 and e['query_length']==(1 if e['step'] else n) and e['key_length']==n+e['step']
   for e in a['answer_zero_events']:assert e['layer'] in [2,3,4,5] and e['active']==(c=='B') and e['original_query']==n+e['step']-1
   audited+=1
 native_reproduced=0;valid_vectors=0
 for rank in range(2):
  assert not (OUT/f'accuracy/rank{rank}/failure.json').exists()
  for path in (OUT/f'smoke/rank{rank}').glob('textvqa__*.json'):
   r=read(path);assert all(r['checks'].values());native_reproduced+=1;valid_vectors+=r['native_valid_logit_vectors']
 assert native_reproduced==6
 archive=HISTORY/'raw_checkpoints.tar.gz';package=read(HISTORY/'packaging.json');assert sha(archive)==package['archive_sha256'];expected=read(HISTORY/'result_hashes.json');verified=0
 with tarfile.open(archive) as tar:
  for member in tar.getmembers():
   parts=member.name.split('/')
   if len(parts)!=3 or parts[0]!='checkpoints' or parts[1] not in samples or parts[2] not in ['Native.json','C3-V2.json']:continue
   data=tar.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==expected[member.name];old=json.loads(data);assert digest(old['payload'])==old['sha256']
   c='N' if parts[2]=='Native.json' else 'Historical-C3-V2';new=read(OUT/f'baselines/{parts[1]}/{c}.json');assert old['payload']['generation']['token_ids']==new['generation']['token_ids'] and old['payload']['score']==new['score'] and old['payload']['input_binding']==new['input_binding'];verified+=1
 assert verified==1000
 atomic(OUT/'independent_verification.json',dict(status='PASSED',at=now(),decoded_answer_and_official_score_records=2000,new_dense_mask_cache_norm_block_audits=audited,original_archive_records_reverified=verified,all_checkpoint_SHA_verified=2000,native_history_smoke_reproductions=6,native_strict_valid_logit_vectors=valid_vectors,paired_soft_score_changes_recomputed=True,accuracy_percent={c:v/5 for c,v in totals.items()}))
 print('INDEPENDENT_VERIFICATION_PASSED',dict(totals))

if __name__=='__main__':main()
