"""Independent CPU delivery check: raw decoding, scores, coverage and references."""
import csv,collections
from .common import *

def main():
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 m=read(OUT/'sample_manifest.json');samples=m['samples'];byid={s['question_id']:s for s in samples};assert len(byid)==1000
 rows=list(csv.DictReader((OUT/'generation_outputs.csv').open()));assert len(rows)==5000
 seen=set();counts=collections.Counter();scores={};decoding_diffs=[]
 for r in rows:
  q=r['question_id'];c=r['condition'];assert (q,c) not in seen;seen.add((q,c));s=byid[q];tokens=json.loads(r['token_ids'])
  text=tokenizer.decode(tokens,skip_special_tokens=True);assert text==r['answer'],(q,c,'decode changed')
  score=text.rstrip('.').lower()==s['gold'];assert score==bool(int(r['correct']));counts[c]+=score;scores[q,c]=score
  assert r['image_sha256']==s['image_sha256'] and int(r['first_token'])==tokens[0]
  if text!=text.strip():decoding_diffs.append(dict(question_id=q,condition=c,raw_text=text))
 expected={'Native':694,'F':588,'Keep-L0-L21':585,'E':680}
 for c,k in expected.items():assert counts[c]==k
 for r in csv.DictReader((OUT/'accuracy_summary.csv').open()):assert counts[r['condition']]==int(r['correct'])
 for r in csv.DictReader((OUT/'paired_comparisons.csv').open()):
  a,b=r['candidate'],r['reference'];delta=[int(scores[q,a])-int(scores[q,b]) for q in byid]
  assert abs(float(r['delta_pp'])-sum(delta)/10)<1e-10
  assert int(r['reference_correct_to_wrong'])==sum(v<0 for v in delta) and int(r['reference_wrong_to_correct'])==sum(v>0 for v in delta)
 index=list(csv.DictReader((OUT/'checkpoint_sha256.csv').open()));assert len(index)==5000
 for r in index:assert sha(ROOT/r['path'])==r['sha256']
 source_bases={}
 for origin in [RECOVERY,PRIOR]:
  frozen=read(origin/'source_manifest.json')
  for path,h in frozen['files'].items():assert sha(ROOT/path)==h
  source_bases[str(origin)]=dict(source_manifest_sha256=digest(frozen),protocol_sha256=sha(origin/'protocol.json'),model_manifest_sha256=sha(origin/'model_manifest.json'),sample_manifest_sha256=sha(origin/'sample_manifest.json'))
 for s in samples:
  for c in ['Native','F','D','E']:
   binding=read(baseline_path(s,c))['binding'];origin=RECOVERY if c in ['D','E'] else PRIOR
   for key,h in source_bases[str(origin)].items():assert binding[key]==h
 shards=[{s['image_sha256'] for s in samples if s['rank']==rank} for rank in range(2)];assert not shards[0]&shards[1]
 for rank in range(2):
  files=list((OUT/f'checkpoints/rank{rank}').glob('*.json'));assert len(files)==500
  assert not (OUT/f'accuracy/rank{rank}/failure.json').exists()
 smoke_n=0;vectors=0
 for s in samples:
  if s['question_id'] not in m['smoke_question_ids']:continue
  d=read(OUT/f'smoke/rank{s["rank"]}/{s["question_id"]}.json');smoke_n+=1
  assert d['comparison']['strict_equal'] and d['comparison']['complete_token_equal']
  for t in d['comparison']['steps']:vectors+=1;assert t['torch_equal'] and t['max_abs']==0 and t['mean_abs']==0
  assert d['logit_sha256']['Keep']==d['logit_sha256']['D']
 assert smoke_n==20
 result=dict(status='PASSED',at=now(),decoded_records=5000,all_raw_token_decodes_exact=True,correct_counts=counts,paired_harm_recovery_recomputed=True,checkpoint_files_SHA256_checked=5000,baseline_source_protocol_model_manifest_bindings_checked=4000,new_generation_files=1000,smoke_cases=20,strict_logit_vectors=vectors,image_shards_disjoint=True,non_stripped_text_records=decoding_diffs)
 write(OUT/'independent_verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
