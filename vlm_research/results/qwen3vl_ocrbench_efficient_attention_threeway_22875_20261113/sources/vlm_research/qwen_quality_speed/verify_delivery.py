"""Independent checks of exported answers, pairing, timing aggregation and hashes."""
import csv,collections,math
import numpy as np
from .common import *

def rows(name):
 with (OUT/name).open(newline='') as f:return list(csv.DictReader(f))
def main():
 check_sources();m=read(OUT/'sample_manifest.json');samples=m['samples'];v=read(OUT/'validation.json');assert v['status']=='PASSED'
 pp=rows('per_question.csv');assert len(pp)==1000 and len({r['question_id'] for r in pp})==1000
 lookup={r['question_id']:r for r in pp};samplemap={r['question_id']:r for r in samples}
 expected={c:0 for c in CONDITIONS}
 for q,s in samplemap.items():
  p=lookup[q];assert p['image_sha256']==s['image_sha256']
  for c in CONDITIONS:
   raw=read(OUT/f'checkpoints/rank{s["rank"]}/{q}__{c}.json')['result'];assert p[c+'__text']==raw['text']
   score=int(raw['text'].rstrip('.').lower()==s['gold']);assert score==int(p[c+'__correct'] in ['True','1'])
   assert json.loads(p[c+'__token_ids'])==raw['token_ids'];expected[c]+=score
 for r in rows('accuracy_summary.csv'):
  assert int(r['correct'])==expected[r['condition']]
  assert abs(float(r['accuracy_percent'])-expected[r['condition']]/10)<1e-10
 for r in rows('paired_comparisons.csv'):
  ca,cb=r['candidate'],r['reference'];harm=recovery=disagree=0
  for p in pp:
   a=p[ca+'__correct']=='True';b=p[cb+'__correct']=='True';harm+=b and not a;recovery+=a and not b;disagree+=p[ca+'__prediction']!=p[cb+'__prediction']
  assert harm==int(r['reference_correct_to_wrong']) and recovery==int(r['reference_wrong_to_correct'])
  assert disagree==int(r['normalized_answer_disagreements'])
  assert abs(float(r['delta_pp'])-(recovery-harm)/10)<1e-9
 for index in ['checkpoint_sha256.csv','benchmark_sha256.csv']:
  for r in rows(index):assert sha(ROOT/r['path'])==r['sha256'],r['path']
 raw=rows('benchmark_raw.csv');assert len(raw)==1920;groups=collections.defaultdict(list)
 for r in raw:
  key=(r['question_id'],r['condition'],r['regime']);groups[key].append(r)
  for k in ['e2e_wall_ms','ttft_gpu_ms','vision_ms','prefill_language_ms','decode_gpu_ms']:assert math.isfinite(float(r[k])) and float(r[k])>=0
  if r['regime']=='fixed16':assert int(r['generated_tokens'])==16
 assert len(groups)==384 and all(len(g)==5 for g in groups.values())
 med={(q,c,g):float(np.median([float(r['e2e_wall_ms']) for r in rr])) for (q,c,g),rr in groups.items()}
 for r in rows('benchmark_summary.csv'):
  qs=m['benchmark_question_ids'];g=r['regime'];c=r['condition'];speed=sum(med[q,'Native',g] for q in qs)/sum(med[q,c,g] for q in qs)
  assert abs(speed-float(r['e2e_speedup_native']))<1e-12
 write(OUT/'independent_delivery_verification.json',dict(status='PASSED',at=now(),quality_answers=4000,benchmark_repeats=1920,answer_score_recomputed=True,raw_csv_exact=True,all_harms_recoveries_recomputed=True,raw_sha256_verified=True,repeated_timing_medians_and_speed_recomputed=True,finite_positive_timing=True,actual_correct_counts=expected))
 print('INDEPENDENT_VERIFICATION_PASSED',expected)
if __name__=='__main__':main()
