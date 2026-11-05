"""Independent CPU verification; never requests CUDA or alters inference inputs."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import sys,json,csv,collections,subprocess
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from receiver_cross_dataset.common import read,write,sha,utc
from benchmarks._vendor.m4c_evaluator import EvalAIAnswerProcessor
O=R/'results/textvqa_decode_l0_l1_factorial';p=read(O/'protocol.json');model=read(O/'execution_model_manifest.json');source=read(O/'execution_source_manifest.json');m=read(O/'sample_manifest.json')['rows'];by={r['question_id']:r for r in m};norm=EvalAIAnswerProcessor()
def rows(n):
 with (O/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 assert read(O/'analysis_validation.json')['passed'] and all(sha(n)==h for n,h in source['files'].items());assert sha(O/'protocol.json')==source['protocol_sha256'];assert sha(O/'sample_manifest.json')==p['manifest_sha256'];assert sha(O/'shards.json')==p['shards_sha256'];assert sha(O/'execution_model_manifest.json')==p['model_manifest_sha256'];assert not (O/'STOP.json').exists()
 smoke=[read(O/f'gpu_smoke_worker{i}.json') for i in [0,1]]
 for v in smoke:assert v['passed'] and len(v['records'])==4
 for r0,r1 in zip(smoke[0]['records'],smoke[1]['records']):
  assert r0['question_id']==r1['question_id']
  for a,b in zip(r0['cases'],r1['cases']):assert a['condition']==b['condition'] and a['token_ids']==b['token_ids'] and a['all_valid_logits_sha256']==b['all_valid_logits_sha256'];assert a['audit']['passed'] and a['audit']['forbidden_probability_max']==0 and a['audit']['non_target_masks_exact']
 write(O/'gpu_smoke_validation.json',dict(passed=True,UTC=utc(),both_card_records=['gpu_smoke_worker0.json','gpu_smoke_worker1.json'],four_fixed_cases_each=True,Native_HF_noop_all_valid_logits_strict=True,cross_GPU_all_valid_logit_SHA_equal=True,legacy_A_B_full_logits_replay_strict=True,forbidden_probability_exact_zero=True,full_Prefill_and_Pminus1_native=True,all_six_first_logits_Native_equal=True,all32_MLP_Norm_Block=True,position_RoPE_cache_hook_audits=True,historical_archive_limit='All historical IDs and first-logit SHA matched; full-step equality is frozen-engine replay, not archived historical vectors.'))
 flat=rows('per_question.csv');assert len(flat)==6600 and len({(r['question_id'],r['condition']) for r in flat})==6600;maxscore=0
 for r in flat:
  g=by[r['question_id']]['answers'];pred=norm(r['prediction']);count=sum(norm(a)==pred for a in g);s=(count*min(1.,max(0,count-1)/3)+(10-count)*min(1.,count/3))/10;maxscore=max(maxscore,abs(s-float(r['score'])));assert maxscore<1e-12
 oldimages={r['image_sha256'] for r in m if r['cohort']=='old300'};newimages={r['image_sha256'] for r in m if r['cohort']=='new_images'};assert len(oldimages)==300 and len(newimages)==531 and not oldimages&newimages
 hist=read(O/'historical_image_inventory.json');assert not newimages&set(hist['excluded_image_sha256']);assert not {r['image_id'] for r in m if r['cohort']=='new_images'}&set(hist['excluded_image_ids']);assert read(O/'all_worktree_paper_checkpoint_identity_audit.json')['passed']
 annpath=Path('/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/sources/textvqa_annotations.json');assert sha(annpath)==p['annotation_sha256'];ann=read(annpath)['data'];official={str(r['question_id']):r for r in ann}
 for r in m:
  original=official[r['question_id']];assert r['question']==original['question'] and r['answers']==original['answers'] and r['image_id']==original['image_id']
 selectedIDs={r['image_id'] for r in m if r['cohort']=='new_images'};assert {str(r['question_id']) for r in ann if r['image_id'] in selectedIDs}=={r['question_id'] for r in m if r['cohort']=='new_images'}
 shard=read(O/'shards.json')['by_question'];hworkers=collections.defaultdict(set)
 for r in m:
  hworkers[r['image_sha256']].add(shard[r['question_id']]);assert sha(r['image_path'])==r['image_sha256'];assert sha(O/'input_specs'/(r['question_id']+'.json'))==r['input_spec_sha256']
 assert all(len(s)==1 for s in hworkers.values())
 # Independently reconstruct question-weighted image bootstrap from the delivered CSV.
 comparisons=rows('paired_comparisons.csv');ni=read(O/'noninferiority.json');maxstat=0
 for cohort in ['old300','new_images']:
  z=[r for r in m if r['cohort']==cohort];names=list(p['conditions']);lookup={(r['question_id'],r['condition']):float(r['score']) for r in flat};matrix=np.array([[lookup[(r['question_id'],c)] for c in names] for r in z]);labels=sorted({r['image_sha256'] for r in z});gidx={h:i for i,h in enumerate(labels)};sums=np.zeros((len(labels),6));counts=np.zeros(len(labels))
  for r,a in zip(z,matrix):k=gidx[r['image_sha256']];sums[k]+=a;counts[k]+=1
  rng=np.random.default_rng(p['seed']);samples=[]
  for start in range(0,20000,256):
   ix=rng.integers(0,len(labels),(min(256,20000-start),len(labels)));samples.append(np.sum(sums[ix],axis=1)/np.sum(counts[ix],axis=1)[:,None])
  b=np.concatenate(samples)
  summary=rows('accuracy_summary.csv')
  for c in names:
   sr=next(r for r in summary if r['cohort']==cohort and r['condition']==c);assert abs(float(sr['accuracy_percent'])-matrix[:,names.index(c)].mean()*100)<1e-10;assert abs(float(sr['delta_Native_pp'])-(matrix[:,names.index(c)]-matrix[:,names.index('Native')]).mean()*100)<1e-10
  weights={'A-D':dict(A=1,D=-1),'B-A':dict(B=1,A=-1),'C-A':dict(C=1,A=-1),'D-B':dict(D=1,B=-1),'D-C':dict(D=1,C=-1),'C-B':dict(C=1,B=-1),'D-C-B+A':dict(D=1,C=-1,B=-1,A=1),**{c+'-Native':{c:1,'Native':-1} for c in 'ABCD'},'Native-Decode-None':{'Native':1,'Decode-None':-1}}
  for name,w in weights.items():
   d=sum(matrix[:,names.index(c)]*v for c,v in w.items());bd=sum(b[:,names.index(c)]*v for c,v in w.items())*100;row=next(r for r in comparisons if r['cohort']==cohort and r['comparison']==name);expect=[d.mean()*100,*np.quantile(bd,[.025,.975])];actual=[float(row[k]) for k in ['delta_pp','CI95_low_pp','CI95_high_pp']];maxstat=max(maxstat,max(abs(np.array(expect)-actual)));assert maxstat<1e-10;assert [int(row[k]) for k in ['improved','declined','unchanged']]==[int((d>1e-12).sum()),int((d< -1e-12).sum()),int((abs(d)<=1e-12).sum())]
   if name=='A-D':assert abs(float(np.quantile(bd,.05))-ni[cohort]['one_sided95_lower_pp'])<1e-10
  aux=[r for r in comparisons if r['cohort']==cohort and r['family']=='fixed10 auxiliary'];assert len(aux)==10;sort=sorted(aux,key=lambda r:float(r['p_two_sided']));prev=0
  for i,r in enumerate(sort):prev=max(prev,min(1,(10-i)*float(r['p_two_sided'])));assert abs(prev-float(r['p_Holm_10']))<1e-12
 assert ni['new_images']['supports_noninferiority']==(ni['new_images']['one_sided95_lower_pp']> -1.)
 figures=[]
 for f in sorted((O/'figures').glob('*.pdf')):
  info=subprocess.check_output(['pdfinfo',str(f)],text=True);assert int(next(line.split(':',1)[1].strip() for line in info.splitlines() if line.startswith('Pages:')))==1;txt=subprocess.check_output(['pdftotext',str(f),'-'],text=True);assert len(txt.strip())>80;figures.append(dict(file=f.name,sha256=sha(f),one_page=True,text_characters=len(txt.strip())))
 assert len(figures)==4
 write(O/'validation.json',dict(passed=True,UTC=utc(),model_revision=model['revision'],source_manifest_all_files_unchanged=True,all6600_official_scores_independent_closed_form_max_difference=maxscore,all_paired_CI_independent_recompute_max_difference=maxstat,all_Holm_values_recomputed=True,main_NI_decision_recomputed=True,new_images=531,new_questions=800,historical_image_ID_and_SHA_overlap=0,all_available_paper_checkpoint_copies_identity_audited=True,shards_no_duplicate_or_intersection=True,all_input_and_image_hashes_rechecked=True,all_six_first_logits_identical=True,cross_GPU_smoke_exact=True,pdf_metadata_and_text=figures,tests='8 inference policy plus4 statistical CPU tests; bothGPU4-case real7B Smoke'))
 print('FINAL_CPU_VALIDATION_PASS',maxscore,maxstat,flush=True)
if __name__=='__main__':main()
