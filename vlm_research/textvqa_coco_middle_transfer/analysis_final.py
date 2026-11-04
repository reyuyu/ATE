"""Official scoring and prespecified whole-image paired inference; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import pathlib,csv,json,hashlib,collections,datetime,contextlib,sys
import numpy as np
from .policy import ROOT,OUT as O,LAYER_SETS,NEW,COMPARISONS,SEED
from receiver_cross_dataset.common import read,write,sha
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
from tokenizers import Tokenizer
H=ROOT/'results/textvqa_visual_reaccess_budget_exploration';CONDITIONS=['A','F','D','C','E','R','B'];REPS=20000
def rows(p):return list(csv.DictReader(p.open()))
def csvwrite(p,rr):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def CI(x):return tuple(float(v) for v in np.percentile(x,[2.5,97.5]))
def groupmask(g,qs,lb):return np.array([g=='All300' or lb[q]==g or (g=='OCR-Combined' and lb[q].startswith('OCR-')) for q in qs])
def pvalue_family(data):
 # One sign per whole image and draw. All actual images here are unique.
 rng=np.random.default_rng(SEED);observed=np.abs(data.sum(0));extreme=np.zeros(data.shape[1],dtype=np.int64)
 for start in range(0,REPS,500):
  signs=2*rng.integers(0,2,size=(min(500,REPS-start),data.shape[0]))-1
  extreme+=(np.abs(signs@data)>=observed[None]-1e-12).sum(0)
 return (extreme+1)/(REPS+1)
def main():
 assert read(O/'generation_validation.json')['passed'] and read(O/'generation_completion.json')['new_checkpoints']==1200
 assert read(O/'gpu_smoke_validation.json')['passed'] and read(O/'gpu_smoke_validation.json')['n']==3
 assert sha(O/'ocr_labels.csv')==read(O/'classification_freeze.json')['labels_sha256']
 assert sha(O/'layer_sets.json')==read(O/'protocol.json')['layer_sets_sha256']
 manifest=read(O/'sample_manifest.json')['rows'];mr={r['question_id']:r for r in manifest};lb={r['question_id']:r['category'] for r in rows(O/'ocr_labels.csv')}
 assert len(mr)==300 and set(mr)==set(lb)
 reuse={(r['question_id'],r['condition']):r for r in read(O/'historical_reuse_audit.json')['checkpoints']}
 index={(r['question_id'],r['condition']):r for r in rows(O/'checkpoint_index.csv')};assert len(index)==1200
 assert set(index)=={(q,c) for q in mr for c in NEW}
 ev=TextVQAAccuracyEvaluator();records=[];by=collections.defaultdict(dict);maskrows=[];maxdiff=0
 tok=Tokenizer.from_file(read(O/'execution_model_manifest.json')['path']+'/tokenizer.json')
 with open(os.devnull,'w') as null,contextlib.redirect_stderr(null):
  for q,r in mr.items():
   for c in CONDITIONS:
    entry=index[(q,c)] if c in NEW else reuse[(q,c)];path=O/entry['path'] if c in NEW else pathlib.Path(entry['checkpoint_path']);assert sha(path)==entry['sha256'];v=read(path);g=v['generation'];s=ev.eval_pred_list([dict(pred_answer=g['text'],gt_answers=r['answers'])]);maxdiff=max(maxdiff,abs(s-v['score']));assert maxdiff<1e-12
    assert v['question_id']==q and v['image_sha256']==r['image_sha256']
    if 'condition' in v:assert v['condition']==c
    else:assert c=='B' and v['writers']==g['query_layout']['R'] and g['writer_positions']==g['query_layout']['R']
    ids=g['token_ids'];assert 0<len(ids)<=32 and 2 not in ids[:-1] and (ids[-1]==2 or len(ids)==32)
    idx=next((i for i in range(len(ids)) if tok.decode(ids[:i+1],skip_special_tokens=True).strip()),None)
    record=dict(question_id=q,image_sha256=r['image_sha256'],image_path=r['image_path'],question=r['question'],OCR_category=lb[q],condition=c,condition_name=LAYER_SETS[c]['name'],origin='fresh generation' if c in NEW else 'audited historical reuse',answer=g['text'],answers=json.dumps(r['answers'],ensure_ascii=False),score=s,token_ids=json.dumps(ids),length=len(ids),non_EOS_length=len(ids)-int(ids[-1]==2),eos=g['eos'],truncated=g['truncated'],empty=g['empty'],repeated_trigram=g['repeated_trigram'],first_token=ids[0],first_content_position_1based=idx+1 if idx is not None else '',first_content_subword=ids[idx] if idx is not None else '',first_logits_sha256=v['first_logits_sha256'],Pminus1_allowed_layers=len(LAYER_SETS[c]['Pminus1']),Decode_allowed_layers=len(LAYER_SETS[c]['Decode']),Decode_allowed_layer_ratio=len(LAYER_SETS[c]['Decode'])/32,actual_Decode_queries=len(ids)-1,actual_allowed_Decode_image_edges=(len(ids)-1)*len(LAYER_SETS[c]['Decode'])*576,checkpoint_path=str(path),checkpoint_sha256=sha(path),input_spec_sha256=sha(H/'input_specs'/(q+'.json')))
    records.append(record);by[q][c]=record
    if c in NEW:
     assert v['execution_protocol_sha256']==sha(O/'execution_protocol.json') and v['execution_source_manifest_sha256']==sha(O/'execution_source_manifest.json') and v['allowed_layers']==LAYER_SETS[c]
     assert v['input_hashes']==read(H/'input_specs'/(q+'.json'))['input_hashes']
     assert v['first_logits_sha256']==read(reuse[(q,'A')]['checkpoint_path'])['first_logits_sha256']
     assert v['generation']['audit']['passed'] and v['hook_cleanup'] and v['weights_unchanged'] and v['MLP_calls']==[len(ids)]*32 and v['empty_cache_at_start']
     mass=v['attention_mass'];assert len(mass)==32*len(ids)
     assert all(row['forbidden_probability_max']==0 for row in mass)
     assert all(not row['blocked'] for row in mass if row['phase']=='Pminus1')
     assert all(row['blocked']==(row['layer'] not in LAYER_SETS[c]['Decode']) for row in mass if row['phase']=='Decode')
     maskrows.append(dict(question_id=q,condition=c,total_layer_calls=len(mass),prefill_blocked_rows=sum(row['blocked'] for row in mass if row['phase']=='Pminus1'),Decode_blocked_rows=sum(row['blocked'] for row in mass if row['phase']=='Decode'),max_forbidden_probability=max(row['forbidden_probability_max'] for row in mass),max_row_sum_error=max(row['row_sum_error'] for row in mass),first_logits_Native_exact=True,all32_MLP=True,hook_cleanup=True,weights_unchanged=True,own_cache=True))
 csvwrite(O/'per_question.csv',records);csvwrite(O/'mask_audit_summary.csv',maskrows)
 qs=sorted(by,key=lambda q:by[q]['A']['image_sha256']);images=[by[q]['A']['image_sha256'] for q in qs];assert len(set(images))==300
 data=np.array([[by[q][c]['score'] for c in CONDITIONS] for q in qs]);col={c:i for i,c in enumerate(CONDITIONS)}
 # Whole-image cluster weights are shared across every condition and subgroup.
 rng=np.random.default_rng(SEED);weights=np.concatenate([rng.multinomial(300,np.ones(300)/300,size=500) for _ in range(REPS//500)])
 groups=['All300','OCR-Direct','OCR-Reasoning','Non-OCR Visual','Ambiguous','OCR-Combined'];summary=[];grouprows=[];recoveries=[];groupcomparisons=[];bootcache={}
 for group in groups:
  use=groupmask(group,qs,lb);n=int(use.sum());den=weights@use.astype(float);valid=den>0;bs=(weights[valid]@(data*use[:,None]))/den[valid,None];mean=data[use].mean(0);bootcache[group]=bs
  for c in CONDITIONS:
   j=col[c];rr=[by[q][c] for q,u in zip(qs,use) if u];lo,hi=CI(100*bs[:,j]);a=mean[j]-mean[col['A']];f=mean[j]-mean[col['F']]
   r=dict(group=group,condition=c,name=LAYER_SETS[c]['name'],n_questions=n,n_images=len({r['image_sha256'] for r in rr}),accuracy_pct=100*mean[j],CI95_low_pct=lo,CI95_high_pct=hi,delta_vs_A_pp=100*a,delta_vs_F_pp=100*f,Decode_layers=len(LAYER_SETS[c]['Decode']),Decode_layer_ratio=len(LAYER_SETS[c]['Decode'])/32,mean_actual_Decode_queries=float(np.mean([r['actual_Decode_queries'] for r in rr])),origin=LAYER_SETS[c]['origin'],answer_changes_vs_A=sum(by[r['question_id']]['A']['answer']!=r['answer'] for r in rr),first_token_changes_vs_A=sum(by[r['question_id']]['A']['first_token']!=r['first_token'] for r in rr),first_content_changes_vs_A=sum(by[r['question_id']]['A']['first_content_subword']!=r['first_content_subword'] for r in rr),eos=sum(bool(r['eos']) for r in rr),truncated=sum(bool(r['truncated']) for r in rr),empty=sum(bool(r['empty']) for r in rr),repeated_trigram=sum(bool(r['repeated_trigram']) for r in rr),mean_generated_tokens=float(np.mean([r['length'] for r in rr])),valid_bootstrap_draws=int(valid.sum()))
   (summary if group=='All300' else grouprows).append(r)
   gap=mean[col['A']]-mean[col['F']];bgap=bs[:,col['A']]-bs[:,col['F']];good=bgap>1e-12;ratio=f/gap if gap>1e-12 else None;interval=CI((bs[:,j]-bs[:,col['F']])/bgap) if good.all() else ('','')
   recoveries.append(dict(group=group,condition=c,n=n,gain_vs_Decode_None_pp=100*f,Full_minus_None_pp=100*gap,Full_gap_recovery_ratio=ratio,CI95_low_ratio=interval[0],CI95_high_ratio=interval[1],zero_or_negative_denominator_draws=int((~good).sum()),ratio_note='descriptive; CI omitted if any resampled denominator nonpositive'))
  for a,b in COMPARISONS:
   j,k=col[a],col[b];d=data[use,j]-data[use,k];lo,hi=CI(100*(bs[:,j]-bs[:,k]))
   groupcomparisons.append(dict(group=group,left=a,right=b,n_questions=n,delta_pp=100*float(d.mean()),CI95_low_pp=lo,CI95_high_pp=hi,score_up=int((d>1e-12).sum()),score_down=int((d< -1e-12).sum()),full1_to_below1=int(((data[use,k]==1)&(data[use,j]<1)).sum()),below1_to_full1=int(((data[use,k]<1)&(data[use,j]==1)).sum()),status='MEASURED',inference='prespecified main family' if group=='All300' else 'descriptive ordinary subgroup CI; no subgroup significance claim'))
 primary=[r for r in groupcomparisons if r['group']=='All300'];diffs=np.array([data[:,col[a]]-data[:,col[b]] for a,b in COMPARISONS]).T;p=pvalue_family(diffs)
 for r,v in zip(primary,p):r['raw_p']=float(v)
 prev=0
 for j,r in enumerate(sorted(primary,key=lambda r:r['raw_p'])):prev=max(prev,min(1,(len(primary)-j)*r['raw_p']));r['Holm_p']=float(prev);r['family_size']=len(primary)
 csvwrite(O/'accuracy_summary.csv',summary);csvwrite(O/'ocr_group_accuracy.csv',grouprows);csvwrite(O/'paired_comparisons.csv',primary);csvwrite(O/'ocr_group_paired_comparisons.csv',[r for r in groupcomparisons if r['group']!='All300']);csvwrite(O/'group_gap_recovery.csv',recoveries)
 pairs=list(dict.fromkeys([(c,b) for c in NEW for b in ['A','F']]+COMPARISONS));negative=[]
 for q in qs:
  for a,b in pairs:
   x,y=by[q][a],by[q][b]
   if x['score']>=y['score']-1e-12:continue
   negative.append(dict(question_id=q,OCR_category=lb[q],question=mr[q]['question'],image_sha256=mr[q]['image_sha256'],image_path=mr[q]['image_path'],answers=json.dumps(mr[q]['answers'],ensure_ascii=False),left=a,right=b,left_answer=x['answer'],right_answer=y['answer'],left_score=x['score'],right_score=y['score'],delta_score=x['score']-y['score'],left_tokens=x['token_ids'],right_tokens=y['token_ids']))
 csvwrite(O/'negative_cases.csv',negative)
 previous={(r['group'],r['condition']):float(r['accuracy_pct']) for r in rows(ROOT/'results/textvqa_ocr_reaccess_mechanism/ocr_group_accuracy.csv')}
 for r in grouprows:
  if r['condition'] in ['A','F','B']:assert abs(r['accuracy_pct']-previous[(r['group'],r['condition'])])<1e-10
 assert len(records)==2100 and len(maskrows)==1200 and all(len(v)==7 for v in by.values())
 assert all(by[q][c]['first_logits_sha256']==by[q]['A']['first_logits_sha256'] for q in qs for c in NEW)
 write(O/'analysis_validation.json',dict(passed=True,UTC=utc(),records=2100,new_records=1200,historical_records=900,questions=300,image_clusters=300,max_official_score_difference=maxdiff,all_new_checkpoint_hashes_inputs_protocol_sources_verified=True,all1200_first_logits_Native_A_exact=True,all1200_actual_mask_audits_passed=True,OCR_labels_unchanged=True,all_reused_group_accuracy_matches_previous=True,bootstrap=20000,seed=SEED,primary_Holm_family_size=5,negative_records=len(negative),negative_unique_questions=len({r['question_id'] for r in negative}),no_training_no_search=True))
 print(json.dumps(dict(status='ANALYSIS_PASS',accuracy={r['condition']:r['accuracy_pct'] for r in summary},primary=primary,negative_records=len(negative))),flush=True)
if __name__=='__main__':main()
