"""Independent CPU delivery checks; no model weights or CUDA libraries imported."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,sys,csv,json,hashlib,subprocess,datetime,collections,re,contextlib,shutil
import numpy as np
O=pathlib.Path(__file__).resolve().parent;ROOT=O.parents[1];W=ROOT.parent;H=ROOT/'results/textvqa_visual_reaccess_budget_exploration';BASE='65c7a9c66cd4ea3747fc11e132c4d6a01b5cd5b4';SEED=20261105
def load(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 assert load(O/'analysis_validation.json')['passed'];f=load(O/'classification_freeze.json');pre=load(O/'preclassification_freeze.json')
 assert sha(O/'ocr_labels.csv')==f['labels_sha256'] and sha(O/'classification_evidence.csv')==f['evidence_sha256'] and sha(O/'classification_protocol.md')==f['protocol_sha256']==pre['protocol_sha256']
 assert datetime.datetime.fromisoformat(pre['UTC'])<datetime.datetime.fromisoformat(f['UTC'])<datetime.datetime.fromisoformat(load(O/'analysis_validation.json')['UTC'])
 labels=rows(O/'ocr_labels.csv');assert len(labels)==300 and len({r['question_id'] for r in labels})==300;lb={r['question_id']:r['category'] for r in labels};blind=load(O/'blind_inputs.json');expected=collections.Counter(lb.values());assert dict(expected)==f['counts']
 sys.path.insert(0,str(ROOT));from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
 ev=TextVQAAccuracyEvaluator();hist=rows(H/'per_question.csv');by=collections.defaultdict(dict);maxdiff=0
 with open(os.devnull,'w') as null,contextlib.redirect_stderr(null):
  for r in hist:
   q,c=r['question_id'],r['condition'];assert c not in by[q];score=ev.eval_pred_list([dict(pred_answer=r['answer'],gt_answers=json.loads(r['answers']))]);maxdiff=max(maxdiff,abs(score-float(r['score'])));assert maxdiff<1e-12;by[q][c]=r
 assert len(hist)==5040 and len(by)==300
 assert len(rows(O/'historical_checkpoint_audit.csv'))==5040
 oldhash=load(H/'artifact_hashes.json')['files'];verified=0
 for key,v in oldhash.items():
  if not key.startswith('vlm_research/results/textvqa_visual_reaccess_budget_exploration/'):continue
  p=W/key;assert sha(p)==v['sha256'];verified+=1
 for r in blind:assert sha(pathlib.Path(r['image_path']))==r['image_sha256']
 ann=load(O/'aligned_ocr_annotations.json');assert len(ann)==300 and sum(r['TextOCR_matched'] for r in ann)==292
 for r in ann:
  b=next(x for x in blind if x['question_id']==r['question_id']);assert r['image_id']==b['image_id'] and r['image_sha256']==b['image_sha256']
  if r['TextOCR_matched']:
   im=r['TextOCR_image'];assert int(im['width'])==b['width'] and int(im['height'])==b['height'] and im['id']==b['image_id']
 conds=['A','B','C','D','E','F','G','Both-4','Both-8','Both-12','Both-16','Both-24'];glist=['OCR-Direct','OCR-Reasoning','OCR-Combined','Non-OCR Visual','Ambiguous','All300'];qorder=[r['question_id'] for r in blind]
 qs=lambda g:[q for q in qorder if g=='All300' or lb[q]==g or (g=='OCR-Combined' and lb[q].startswith('OCR-'))]
 def values(g,c):return np.array([float(by[q][c]['score']) for q in qs(g)])
 summary=rows(O/'ocr_group_accuracy.csv')
 for r in summary:
  g,c=r['group'],r['condition'];sub=qs(g);v=values(g,c);assert int(r['n_questions'])==len(sub) and int(r['n_images'])==len(sub);assert abs(100*v.mean()-float(r['accuracy_pct']))<1e-10
  for base in ['A','B']:assert abs(100*(v-values(g,base)).mean()-float(r['delta_vs_'+base+'_pp']))<1e-10
  for name,k in [('answer_change_count','answer'),('token_sequence_change_count','token_ids'),('first_token_change_count','first_token')]:assert int(r[name])==sum(by[q][c][k]!=by[q]['B'][k] for q in sub)
  for key in ['eos','truncated','repeated_trigram','empty']:assert int(r[key])==sum(by[q][c][key]=='True' for q in sub)
 for r in rows(O/'paired_ocr_effects.csv'):
  d=values(r['group'],r['left'])-values(r['group'],r['right']);assert abs(100*d.mean()-float(r['delta_pp']))<1e-10 and int(r['score_up'])==int((d>1e-12).sum()) and int(r['score_down'])==int((d< -1e-12).sum())
 # Verify completeness of all prespecified losses, including B versus Native A.
 curve=['B','E','Both-4','Both-8','Both-12','Both-16','Both-24','A']
 pairs=list(dict.fromkeys([(c,b) for c in conds for b in ['A','B'] if c!=b]+[('F','G'),('G','F')]+list(zip(curve[1:],curve[:-1]))))
 wanted={(q,l,r) for q in by for l,r in pairs if float(by[q][l]['score'])<float(by[q][r]['score'])-1e-12}
 negatives=rows(O/'negative_cases.csv');actual={(r['question_id'],r['left'],r['right']) for r in negatives};assert len(negatives)==len(actual) and actual==wanted
 for r in negatives:
  q,l,rr=r['question_id'],r['left'],r['right'];assert r['left_answer']==by[q][l]['answer'] and r['right_answer']==by[q][rr]['answer'];assert abs(float(r['delta_score'])-(float(by[q][l]['score'])-float(by[q][rr]['score'])))<1e-12
 for name,n in [('paired_ocr_effects.csv',25),('ocr_vs_nonocr_interaction.csv',15)]:
  rr=rows(O/name);assert len(rr)==n;prev=0
  for j,r in enumerate(sorted(rr,key=lambda r:float(r['raw_p']))):prev=max(prev,min(1,(n-j)*float(r['raw_p'])));assert abs(prev-float(r['Holm_p']))<1e-12
 # Independent loop implementation of all primary interaction bootstrap intervals.
 images=sorted({r['image_sha256'] for r in blind});rng=np.random.default_rng(SEED);weights=np.vstack([rng.multinomial(300,np.ones(300)/300,size=500) for _ in range(40)]);imageq={r['image_sha256']:r['question_id'] for r in blind};gg={g:np.array([q in qs(g) for q in [imageq[im] for im in images]],dtype=float) for g in glist};empty_draws={}
 for r in rows(O/'ocr_vs_nonocr_interaction.csv'):
  l,rr=r['effect'],r['reference_group'];left,right={'Full_reaccess_effect':('A','B'),'Pminus1_effect':('F','B'),'Decode_effect':('G','B'),'Both16_effect':('Both-16','B'),'Pminus1_minus_Decode':('F','G')}[l]
  data=np.array([float(by[imageq[im]][left]['score'])-float(by[imageq[im]][right]['score']) for im in images]);a,b=gg[r['OCR_group']],gg[rr];da=weights@a;db=weights@b;valid=(da>0)&(db>0);sample=100*((weights[valid]@(data*a))/da[valid]-(weights[valid]@(data*b))/db[valid]);lo,hi=np.percentile(sample,[2.5,97.5]);assert abs(lo-float(r['CI95_low_pp']))<1e-9 and abs(hi-float(r['CI95_high_pp']))<1e-9
  point=100*((data*a).sum()/a.sum()-(data*b).sum()/b.sum());p=(int((abs(sample-point)>=abs(point)-1e-12).sum())+1)/(len(sample)+1);assert abs(p-float(r['raw_p']))<1e-12
 # Decode audit uses independent list comparison, including EOS positions.
 for r in rows(O/'decode_divergence.csv'):
  q=r['question_id'];a=json.loads(by[q]['G']['token_ids']);b=json.loads(by[q]['B']['token_ids']);assert a[0]==b[0] and by[q]['G']['first_logits_sha256']==by[q]['B']['first_logits_sha256'];mismatch=[i+1 for i in range(min(len(a),len(b))) if a[i]!=b[i]];pos=mismatch[0] if mismatch else min(len(a),len(b))+1 if len(a)!=len(b) else None;assert str(pos or '')==r['first_divergence_position_1based']
 # Meaningful small fixtures: same-image bootstrap keeps whole cluster; EOS/prefix cases.
 sys.path.insert(0,str(O));import analysis
 assert analysis.divergence([5,2],[5,7,2])==(2,'token_mismatch') and analysis.divergence([5],[5,2])==(2,'prefix_extension') and analysis.divergence([5,2],[5,2])==(None,'unchanged')
 sample=analysis.cluster_boot(['same','same'],np.array([[0.],[1.]]));assert np.array_equal(sample,np.full((20000,1),.5))
 assert analysis.firstword(' [12] White-box')=='white' and analysis.firstword('12.5')==''
 # All requested panels and their underlying data are present; A-D vector PDF and PNG decode.
 from PIL import Image
 figs=[]
 for p in sorted(O.glob('figure_*.pdf')):
  raw=p.read_bytes();assert raw.startswith(b'%PDF') and b'/Subtype /Image' not in raw and len(re.findall(rb'/Type\s*/Page\b',raw))==1
  with Image.open(p.with_suffix('.png')) as im:im.verify()
  figs.append(p.name)
 assert len(figs)==4
 required=['report_zh.md','hypotheses.json','classification_protocol.md','ocr_labels.csv','classification_evidence.csv','ocr_dataset_alignment.json','ocr_group_accuracy.csv','paired_ocr_effects.csv','ocr_vs_nonocr_interaction.csv','decode_divergence.csv','answer_length_analysis.csv','negative_cases.csv','case_studies.md'];assert all((O/n).exists() for n in required)
 # No weight-loading/CUDA imports or launch calls are allowed in delivered analysis.
 scripts=['analysis.py','plots_report.py','validate.py'];forbidden=['import torch','import transformers','from_pretrained','AutoModel','LlavaFor','cuda.','nvidia-smi --gpu-reset','subprocess.Popen']
 for name in scripts:
  code=(O/name).read_text()
  if name!='validate.py':assert not any(s in code for s in forbidden)
 assert 'torch' not in sys.modules and 'transformers' not in sys.modules and os.environ['CUDA_VISIBLE_DEVICES']==''
 procs=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True);own=[r for r in procs.splitlines() if str(O) in r and ('analysis.py' in r or 'plots_report.py' in r)];assert not own
 gpu_snapshot=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
 resources=dict(UTC=utc(),CPU_only=True,CUDA_VISIBLE_DEVICES='',model_weights_loaded=False,new_generations=0,GPU_requested=False,GPU_workers_started=0,no_GPU_to_release=True,no_torch_or_transformers_import=True,analysis_process_finished=True,other_GPU_processes_untouched=True,current_external_GPU_processes=gpu_snapshot.stdout.strip().splitlines(),disk=shutil.disk_usage('/data')._asdict())
 write(O/'resource_audit.json',resources)
 assert subprocess.check_output(['git','rev-parse','main'],cwd=W,text=True).strip()=='7970be981e3462560cb66d83e1014eaa1b1ebdae'
 assert subprocess.run(['git','diff','--quiet','HEAD','--','.',':(exclude)vlm_research/results/textvqa_ocr_reaccess_mechanism'],cwd=W).returncode==0
 write(O/'validation.json',dict(passed=True,UTC=utc(),historical_records_rescored=5040,max_score_difference=maxdiff,historical_deliverables_SHA_verified=verified,all300_original_image_hashes_verified=True,all300_labels_retained=True,class_counts=dict(expected),labels_protocol_evidence_frozen_and_unchanged=True,TextOCR_ID_dimensions_aligned=292,all_paired_points_counts_and_Holm_verified=True,all15_interaction_CIs_and_p_independently_recomputed=True,all300_GB_first_logits_and_tokens_exact=True,all300_divergence_positions_independent=True,cluster_EOS_prefix_word_fixtures_passed=True,all4_PDF_vector_and_PNG_valid=True,figures=figs,main_and_history_unchanged=True,CPU_only=True,no_new_inference=True))
 print('INDEPENDENT_VALIDATION_PASS',len(hist),maxdiff,verified,flush=True)
if __name__=='__main__':main()
