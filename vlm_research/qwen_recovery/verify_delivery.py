"""Independent re-scoring, harm/recovery counting, raw SHA and gate verification."""
import csv
from .common import *
ALL=['Native','F']+CONDITIONS
def rows(name):
 with (OUT/name).open(newline='') as f:return list(csv.DictReader(f))
def main():
 check_sources();m=read(OUT/'sample_manifest.json');pp=rows('per_question.csv');assert len(pp)==1000 and len({r['question_id'] for r in pp})==1000
 lookup={s['question_id']:s for s in m['samples']};counts={c:0 for c in ALL}
 for r in rows('checkpoint_sha256.csv'):
  p=ROOT/r['path'];assert sha(p)==r['sha256'],r['path']
  d=read(p);v=d['result'];s=lookup[r['question_id']];c=r['condition'];score=v['text'].rstrip('.').lower()==s['gold'];assert score==v['correct'];counts[c]+=score
  if c in CONDITIONS:
   assert v['all_logits_finite'] and v['audit']['dense_cache_all_layers']
   assert not set(v).intersection({'timing','e2e_ms','flops','memory','peak_allocated'})
 for r in pp:
  for c in ALL:
   s=lookup[r['question_id']];score=r[c+'__text'].rstrip('.').lower()==s['gold'];assert score==(r[c+'__correct']=='True')
  assert r['B__first_token']==r['F__first_token'] and r['C__first_token']==r['A__first_token']
 for r in rows('accuracy_summary.csv'):
  assert int(r['correct'])==counts[r['condition']];assert abs(float(r['accuracy_percent'])-counts[r['condition']]/10)<1e-10
 for r in rows('paired_comparisons.csv'):
  a,b=r['candidate'],r['reference'];harm=sum(p[b+'__correct']=='True' and p[a+'__correct']=='False' for p in pp);rec=sum(p[b+'__correct']=='False' and p[a+'__correct']=='True' for p in pp)
  assert harm==int(r['reference_correct_to_wrong']) and rec==int(r['reference_wrong_to_correct']);assert abs(float(r['delta_pp'])-(rec-harm)/10)<1e-10
 best=max(CONDITIONS,key=counts.get);g=read(OUT/'restoration_gate.json');assert best==g['best_condition'];assert g['user_accuracy_gate_passed']==((counts[best]-counts['F']>=30) and g['pointwise_95ci_low']>0)
 assert counts['Native']==694 and counts['F']==588
 assert len(list((OUT/'checkpoints').glob('rank*/*.json')))==5000
 assert not (OUT/'benchmark').exists()
 write(OUT/'independent_delivery_verification.json',dict(status='PASSED',at=now(),independently_rescored_records=7000,new_records=5000,baseline_records=2000,all_raw_SHA256_valid=True,all_CSV_answers_match_raw=True,all_harm_recovery_counts_recomputed=True,gate_recomputed=True,correct_counts=counts,no_performance_records_or_fields=True))
 print('INDEPENDENT_VERIFICATION_PASSED',counts)
if __name__=='__main__':main()
