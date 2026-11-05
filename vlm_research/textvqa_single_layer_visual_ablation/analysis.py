"""CPU-only official scoring and whole-image paired inference of real records."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import pathlib,csv,json,collections,contextlib,hashlib,sys,datetime
import numpy as np
from .policy import *
from receiver_cross_dataset.common import read,write,sha
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
from tokenizers import Tokenizer
REPS=20000;EPS=1e-12
def csvwrite(path,rows,fields=None):
 rows=list(rows)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or (list(rows[0]) if rows else ['status']),lineterminator='\n');w.writeheader();w.writerows(rows)
def loadcsv(path):return list(csv.DictReader(path.open()))
def CI(a):return [float(x) for x in np.percentile(a,[2.5,97.5])]
def holm(p,total=None):
 p=np.array(p);order=np.argsort(p,kind='stable');m=len(p) if total is None else total
 ans=np.empty(len(p));ans[order]=np.minimum(1,np.maximum.accumulate((m-np.arange(len(p)))*p[order]));return ans
def paired_stats(data,baseline):
 n,m=data.shape;diff=data-baseline[:,None];rng=np.random.default_rng(SEED)
 # Image hashes are unique by input audit: this is exactly a whole-image cluster bootstrap.
 weights=rng.multinomial(n,np.full(n,1/n),size=REPS);bs=weights@data/n;bd=weights@diff/n
 rng=np.random.default_rng(SEED);observed=np.abs(diff.sum(0));extreme=np.zeros(m,dtype=np.int64)
 for j in range(0,REPS,500):
  signs=2*rng.integers(0,2,size=(min(500,REPS-j),n))-1
  extreme+=(np.abs(signs@diff)>=observed[None]-EPS).sum(0)
 return bs,bd,(extreme+1)/(REPS+1)
def main():
 manifest=read(OUT/'sample_manifest.json');rows=manifest['rows'];mr={r['question_id']:r for r in rows};labels={r['question_id']:r['category'] for r in loadcsv(OUT/'ocr_labels.csv')}
 assert len(rows)==len(mr)==len({r['image_sha256'] for r in rows})==300
 assert sha(OUT/'sample_manifest.json')==read(OUT/'protocol.json')['manifest_sha256']
 assert sha(OUT/'ocr_labels.csv')==read(OUT/'classification_freeze.json')['labels_sha256']
 source_hash=sha(OUT/'execution_source_manifest.json');protocol_hash=sha(OUT/'protocol.json');shards=read(OUT/'shards.json');by=collections.defaultdict(dict);index=[];records=[];ev=TextVQAAccuracyEvaluator();maxdiff=0
 tok=Tokenizer.from_file(read(OUT/'execution_model_manifest.json')['path']+'/tokenizer.json')
 entries=read(OUT/'historical_reuse_audit.json')['checkpoints']
 entries=[dict(e,path=e['checkpoint_path'],fresh=False) for e in entries]
 for qdir in sorted((OUT/'checkpoints').glob('*')):
  if qdir.is_dir():
   for p in sorted(qdir.glob('*.json')):entries.append(dict(question_id=qdir.name,condition=p.stem,path=str(p),sha256=sha(p),fresh=True))
 for e in entries:
  q,c=e['question_id'],e['condition'];p=pathlib.Path(e['path']);v=read(p);g=v['generation'];r=mr[q];assert sha(p)==e['sha256']
  assert v['question_id']==q and v['image_sha256']==r['image_sha256'] and v['input_hashes']==r['input_hashes']
  with open(os.devnull,'w') as z,contextlib.redirect_stderr(z):s=ev.eval_pred_list([dict(pred_answer=g['text'],gt_answers=r['answers'])])
  maxdiff=max(maxdiff,abs(s-v['score']));assert maxdiff<EPS
  ids=g['token_ids'];assert 0<len(ids)<=32 and 2 not in ids[:-1] and (ids[-1]==2 or len(ids)==32)
  assert bool(g['eos'])==(ids[-1]==2) and bool(g['truncated'])==(len(ids)==32 and ids[-1]!=2)
  pos=next((j for j in range(len(ids)) if tok.decode(ids[:j+1],skip_special_tokens=True).strip()),None)
  spec=read(INPUTS/(q+'.json'));assert sha(INPUTS/(q+'.json'))==r['input_spec_sha256'] and spec['input_hashes']==v['input_hashes']
  if e['fresh']:
   assert v['condition']==c and v['definition']==CONDITIONS[c] and v['worker']==shards['by_question'][q]
   assert v['protocol_sha256']==protocol_hash and v['source_manifest_sha256']==source_hash and v['generation']['audit']['passed'] and v['hook_cleanup']
   layout=g['query_layout'];n=layout['prompt_length'];a=g['audit'];expected_prefill=[len(query_rows(layout,c,k)) for k in range(32)];expected_decode=[sum(len(query_rows(layout,c,k,n+j)) for j in range(len(ids)-1)) for k in range(32)]
   assert a['prefill_masked_queries_by_layer']==expected_prefill and a['decode_masked_queries_by_layer']==expected_decode and a['masked_edges_per_head']==576*sum(expected_prefill+expected_decode)
   assert a['layer_calls']==32*len(ids) and a['positions_and_cache_lengths_exact']
   for j,x in enumerate(g['cache_lengths']):assert x['before']==(0 if j==0 else n+j-1) and x['after']==n+j
  rr=dict(question_id=q,image_sha256=r['image_sha256'],question=r['question'],OCR_category=labels[q],condition=c,family=CONDITIONS[c]['family'] if c in CONDITIONS else c,layer=CONDITIONS[c]['layer'] if c in CONDITIONS and CONDITIONS[c]['layer'] is not None else '',reference=CONDITIONS[c]['reference'] if c in CONDITIONS else '',origin='new generation' if e['fresh'] else 'SHA-audited historical reuse',generation_origin=v.get('origin','fresh') if e['fresh'] else 'historical reuse; no new baseline generation',worker=v.get('worker','historical'),answer=g['text'],answers=json.dumps(r['answers'],ensure_ascii=False),token_ids=json.dumps(ids),score=s,length=len(ids),eos=g['eos'],truncated=g['truncated'],empty=g['empty'],repeated_trigram=g['repeated_trigram'],first_token=ids[0],first_content_position_1based=pos+1 if pos is not None else '',first_content_subword=ids[pos] if pos is not None else '',first_logits_sha256=v['first_logits_sha256'],input_spec_sha256=r['input_spec_sha256'],checkpoint_path=str(p),checkpoint_sha256=e['sha256'])
  assert c not in by[q];by[q][c]=rr;records.append(rr)
  if e['fresh']:index.append(dict(question_id=q,condition=c,path=str(p.relative_to(OUT)),sha256=e['sha256'],worker=v['worker']))
 for q,cc in by.items():
  for c,v in cc.items():
   if c in CONDITIONS and CONDITIONS[c]['family'] in ['Donly','Middle-minus','Addback']:assert v['first_logits_sha256']==cc['Native']['first_logits_sha256']
  for k in range(32):
   a,b=f'Ponly-L{k}',f'Both-L{k}'
   if a in cc and b in cc:assert cc[a]['first_logits_sha256']==cc[b]['first_logits_sha256']
 for r in records:
  q=r['question_id'];ref=r['reference'] if r['reference'] in by[q] else 'Native';b=by[q][ref]
  r.update(delta_score_vs_reference=r['score']-b['score'],answer_changed=r['answer']!=b['answer'],first_token_changed=r['first_token']!=b['first_token'],first_content_changed=r['first_content_subword']!=b['first_content_subword'])
 csvwrite(OUT/'per_question.csv',records);csvwrite(OUT/'checkpoint_index.csv',index)
 negatives=[r for r in records if r['origin']=='new generation' and r['delta_score_vs_reference']< -EPS]
 positives=[r for r in records if r['origin']=='new generation' and r['delta_score_vs_reference']>EPS]
 csvwrite(OUT/'negative_cases.csv',negatives,list(records[0]));csvwrite(OUT/'positive_cases.csv',positives,list(records[0]))
 sample_sets=dict(Pilot120=shards['pilot120'],Full300=list(mr));stats=[];groups=[];baselines=[];coverage=[]
 for c in NATIVE+[x for x in PRIMARY if x.startswith('Middle')]+OPTIONAL:
  available=[q for q in mr if c in by[q]];coverage.append(dict(condition=c,family=CONDITIONS[c]['family'],layer=CONDITIONS[c]['layer'],n_completed=len(available),n_pilot_completed=sum(q in available for q in shards['pilot120']),target=300,status='COMPLETE300' if len(available)==300 else 'PARTIAL'))
 for sample,qs0 in sample_sets.items():
  qs=sorted(qs0,key=lambda q:mr[q]['image_sha256']);n=len(qs)
  for ref in ['Native','Middle','Decode-None','Both-None']:
   b=np.array([by[q][ref]['score'] for q in qs]);rng=np.random.default_rng(SEED);bs=rng.multinomial(n,np.full(n,1/n),size=REPS)@b/n;lo,hi=CI(100*bs)
   baselines.append(dict(sample_set=sample,condition=ref,n=n,accuracy_pct=100*b.mean(),CI95_low_pct=lo,CI95_high_pct=hi,eos=sum(by[q][ref]['eos'] for q in qs),truncated=sum(by[q][ref]['truncated'] for q in qs),empty=sum(by[q][ref]['empty'] for q in qs),repeated_trigram=sum(by[q][ref]['repeated_trigram'] for q in qs)))
  for family,cs0,ref,m in [('Native128',NATIVE,'Native',128),('Middle11',[c for c in PRIMARY if c.startswith('Middle')],'Middle',11),('Addback32',OPTIONAL,'Decode-None',32)]:
   cs=[c for c in cs0 if all(c in by[q] for q in qs)]
   if not cs:continue
   data=np.array([[by[q][c]['score'] for c in cs] for q in qs]);b=np.array([by[q][ref]['score'] for q in qs]);bs,bd,p=paired_stats(data,b);adj=holm(p,m)
   for j,c in enumerate(cs):
    rr=[by[q][c] for q in qs];br=[by[q][ref] for q in qs];d=data[:,j]-b;lo,hi=CI(100*bs[:,j]);dl,dh=CI(100*bd[:,j]);definition=CONDITIONS[c]
    r=dict(sample_set=sample,n=n,n_images=n,condition=c,family=definition['family'],layer=definition['layer'],reference=ref,accuracy_pct=100*float(data[:,j].mean()),accuracy_CI95_low_pct=lo,accuracy_CI95_high_pct=hi,reference_accuracy_pct=100*float(b.mean()),delta_pp=100*float(d.mean()),CI95_low_pp=dl,CI95_high_pp=dh,score_down=int((d< -EPS).sum()),score_up=int((d>EPS).sum()),score_same=int((abs(d)<=EPS).sum()),full1_to_below1=int(((b==1)&(data[:,j]<1)).sum()),below1_to_full1=int(((b<1)&(data[:,j]==1)).sum()),answer_changes=sum(a['answer']!=b0['answer'] for a,b0 in zip(rr,br)),first_token_changes=sum(a['first_token']!=b0['first_token'] for a,b0 in zip(rr,br)),first_content_changes=sum(a['first_content_subword']!=b0['first_content_subword'] for a,b0 in zip(rr,br)),eos=sum(a['eos'] for a in rr),truncated=sum(a['truncated'] for a in rr),empty=sum(a['empty'] for a in rr),repeated_trigram=sum(a['repeated_trigram'] for a in rr),mean_generated_tokens=float(np.mean([a['length'] for a in rr])),p_two_sided=float(p[j]),p_Holm=float(adj[j]),multiplicity_family=f'{sample}:{family}',family_size=m,complete_comparisons_in_family=len(cs),bootstrap_repetitions=REPS,bootstrap_seed=SEED,CI_type='ordinary pointwise percentile95; Holm applies to p separately',exploratory=True)
    stats.append(r)
   # Descriptive fixed labels; reuse whole-image weights with ratio subgroup denominators.
   rng=np.random.default_rng(SEED);weights=rng.multinomial(n,np.full(n,1/n),size=REPS)
   for group in sorted(set(labels.values())):
    use=np.array([labels[q]==group for q in qs]);ng=int(use.sum())
    if not ng:continue
    den=weights@use.astype(float);valid=den>0;bg=(weights[valid]@(data*use[:,None]))/den[valid,None];baseg=weights[valid]@(b*use)/den[valid]
    for j,c in enumerate(cs):
     lo,hi=CI(100*(bg[:,j]-baseg));d=(data[:,j]-b)[use]
     groups.append(dict(sample_set=sample,OCR_group=group,n=ng,condition=c,family=CONDITIONS[c]['family'],layer=CONDITIONS[c]['layer'],reference=ref,accuracy_pct=100*float(data[use,j].mean()),reference_accuracy_pct=100*float(b[use].mean()),delta_pp=100*float(d.mean()),CI95_low_pp=lo,CI95_high_pp=hi,score_down=int((d< -EPS).sum()),score_up=int((d>EPS).sum()),valid_bootstrap_draws=int(valid.sum()),interpretation='descriptive frozen prior AI labels, no independent human-ground-truth claim'))
 csvwrite(OUT/'accuracy_by_layer.csv',stats);csvwrite(OUT/'paired_comparisons.csv',stats)
 csvwrite(OUT/'middle_leave_one_out.csv',[r for r in stats if r['family']=='Middle-minus'])
 csvwrite(OUT/'baseline_accuracy.csv',baselines);csvwrite(OUT/'ocr_group_accuracy.csv',groups);csvwrite(OUT/'coverage.csv',coverage)
 # Observed single-layer phase interaction, not an additional confirmatory family.
 interactions=[]
 for sample,qs0 in sample_sets.items():
  qs=sorted(qs0,key=lambda q:mr[q]['image_sha256']);n=len(qs);rng=np.random.default_rng(SEED);weights=rng.multinomial(n,np.full(n,1/n),size=REPS)
  for layer in range(32):
   cs=[f'{f}-L{layer}' for f in ['All-Text','Ponly','Donly','Both']]
   if not all(all(c in by[q] for c in cs) for q in qs):continue
   v=np.array([[by[q][c]['score'] for c in cs]+[by[q]['Native']['score']] for q in qs]);inter=v[:,3]-v[:,1]-v[:,2]+v[:,4];relay=v[:,0]-v[:,3]
   il,ih=CI(100*(weights@inter/n));rl,rh=CI(100*(weights@relay/n))
   interactions.append(dict(sample_set=sample,layer=layer,n=n,Both_minus_Ponly_minus_Donly_plus_Native_pp=100*inter.mean(),interaction_CI95_low_pp=il,interaction_CI95_high_pp=ih,AllText_minus_Both_pp=100*relay.mean(),relay_context_CI95_low_pp=rl,relay_context_CI95_high_pp=rh,note='posthoc descriptive phase contrast; R effect conditional on same-layer Pminus1+Decode removal, not R-only experiment'))
 csvwrite(OUT/'phase_interactions.csv',interactions)
 mandatory=[c for c in NATIVE]+[c for c in PRIMARY if c.startswith('Middle')];complete=sum(all(c in by[q] for q in mr) for c in mandatory)
 smoke_calls=0;optional_smoke_calls=0
 fast_replays={'All-Text-L0','Ponly-L15','Donly-L31','Both-L8','Middle-minus-L11','Addback-L0'}
 for i in [0,1]:
  for case in read(OUT/f'gpu_smoke_worker{i}.json')['records']:
   smoke_calls+=len(case['cases'])+2+len(fast_replays&{v['condition'] for v in case['cases']})
  p=OUT/f'optional_smoke_worker{i}.json'
  if p.exists():
   v=read(p);optional_smoke_calls+=len(v['questions'])*len(v['conditions'])
 retained_smoke=sum(r['origin']=='new generation' and 'smoke' in r['generation_origin'] for r in records)
 write(OUT/'generation_accounting.json',dict(new_matrix_checkpoints=len(index),retained_smoke_in_matrix=retained_smoke,fresh_formal_outside_smoke=len(index)-retained_smoke,mandatory_dual_GPU_smoke_trajectories=smoke_calls,optional_dual_GPU_smoke_trajectories=optional_smoke_calls,total_model_generation_trajectories=len(index)-retained_smoke+smoke_calls+optional_smoke_calls,historical_reused_scoring_records=1200,control_replays_not_added_as_scoring_samples=True,duplicate_smoke_inputs_across_workers='explicitly required same3 smoke, no duplicate formal matrix record'))
 write(OUT/'analysis_validation.json',dict(passed=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_checkpoints=len(index),historical_records=1200,total_records=len(records),score_max_difference=maxdiff,mandatory_conditions_complete300=complete,mandatory_target_conditions=139,mandatory_target_records=41700,optional_conditions_complete300=sum(all(c in by[q] for q in mr) for c in OPTIONAL),negative_rows=len(negatives),negative_unique_questions=len({r['question_id'] for r in negatives}),positive_rows=len(positives),all_masks_structurally_match_frozen_policy=True,all_first_logits_constraints_exact=True,all_checkpoint_inputs_protocol_source_SHA_verified=True,actual_curve_points=len(stats)))
 print(json.dumps(read(OUT/'analysis_validation.json')),flush=True)
if __name__=='__main__':main()
