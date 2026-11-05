"""Independent CPU verification, release audit, and artifact checksum inventory."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import pathlib,json,csv,hashlib,collections,subprocess,datetime,sys
import numpy as np
from PIL import Image
from .policy import ROOT,OUT,SEED
from receiver_cross_dataset.common import read,write,sha,hardware
from benchmarks._vendor.m4c_evaluator import EvalAIAnswerProcessor
def rows(p):return list(csv.DictReader(p.open()))
def independent_score(pred,answers,process):
 target=process(pred);norm=[process(x) for x in answers];m=sum(x==target for x in norm)
 return ((10-m)*min(1,m/3)+m*min(1,max(m-1,0)/3))/10
def holm(p,size):
 order=sorted(range(len(p)),key=lambda j:p[j]);out=[None]*len(p);running=0
 for rank,j in enumerate(order):running=max(running,min(1,p[j]*(size-rank)));out[j]=running
 return out
def main():
 av=read(OUT/'analysis_validation.json');protocol=read(OUT/'protocol.json');source=read(OUT/'execution_source_manifest.json');mr={r['question_id']:r for r in read(OUT/'sample_manifest.json')['rows']};shard=read(OUT/'shards.json');per=rows(OUT/'per_question.csv');by=collections.defaultdict(dict);processor=EvalAIAnswerProcessor();maxscore=0
 assert len(mr)==300 and len({r['image_sha256'] for r in mr.values()})==300
 assert source['protocol_sha256']==sha(OUT/'protocol.json') and all(sha(p)==v for p,v in source['files'].items())
 assert sha(OUT/'ocr_labels.csv')==protocol['OCR_labels_sha256']
 model=read(OUT/'execution_model_manifest.json')
 for name,v in model['files'].items():assert sha(pathlib.Path(model['path'])/name)==v['sha256']
 for q,r in mr.items():assert sha(r['image_path'])==r['image_sha256'] and sha(ROOT.parent.parent/'textvqa-coco-middle-transfer/vlm_research/results/textvqa_visual_reaccess_budget_exploration/input_specs'/(q+'.json'))==r['input_spec_sha256']
 assert [sum(i==w for i in shard['by_question'].values()) for w in [0,1]]==[150,150]
 assert set(shard['by_question'])==set(mr)
 expectedpilot=sorted(mr,key=lambda q:hashlib.sha256((q+'|'+mr[q]['image_sha256']).encode()).hexdigest())[:120]
 assert expectedpilot==shard['pilot120']
 new=0;negative=set();positive=set();native_first={};seen=set()
 for r in per:
  q,c=r['question_id'],r['condition'];assert (q,c) not in seen;seen.add((q,c));by[q][c]=r
  p=pathlib.Path(r['checkpoint_path']);assert sha(p)==r['checkpoint_sha256'];v=read(p);g=v['generation'];assert v['image_sha256']==mr[q]['image_sha256'] and v['input_hashes']==mr[q]['input_hashes']
  score=independent_score(r['answer'],mr[q]['answers'],processor);err=abs(score-float(r['score']));maxscore=max(maxscore,err);assert err<1e-12
  assert json.loads(r['token_ids'])==g['token_ids'] and r['answer']==g['text']
  if r['origin']=='new generation':
   new+=1;assert v['worker']==shard['by_question'][q] and v['protocol_sha256']==sha(OUT/'protocol.json') and v['source_manifest_sha256']==sha(OUT/'execution_source_manifest.json')
   family=r['family'];layer=int(r['layer']);n=g['query_layout']['prompt_length'];R=g['query_layout']['R'];prefill=[0]*32;decode=[0]*32
   if family=='All-Text':prefill[layer]=len(R)+1
   elif family in ['Ponly','Both']:prefill[layer]=1
   if family in ['All-Text','Donly','Both']:decode[layer]=g['length']-1
   elif family=='Middle-minus':
    allow=set([0]+list(range(11,21)))-{layer};decode=[0 if k in allow else g['length']-1 for k in range(32)]
   elif family=='Addback':decode=[0 if k==layer else g['length']-1 for k in range(32)]
   a=g['audit'];assert a['prefill_masked_queries_by_layer']==prefill and a['decode_masked_queries_by_layer']==decode and a['masked_edges_per_head']==576*(sum(prefill)+sum(decode))
   assert a['layer_calls']==32*g['length'] and g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1
   if float(r['delta_score_vs_reference'])< -1e-12:negative.add((q,c))
   if float(r['delta_score_vs_reference'])>1e-12:positive.add((q,c))
 assert new==av['new_checkpoints'] and len(per)==new+1200
 assert {(r['question_id'],r['condition']) for r in rows(OUT/'negative_cases.csv')}==negative
 assert {(r['question_id'],r['condition']) for r in rows(OUT/'positive_cases.csv')}==positive
 final_layer_boundary=0
 for q,cc in by.items():
  if 'All-Text-L31' in cc and 'Both-L31' in cc:
   a,b=cc['All-Text-L31'],cc['Both-L31']
   assert a['token_ids']==b['token_ids'] and a['first_logits_sha256']==b['first_logits_sha256']
   final_layer_boundary+=1
 # Recompute every reported paired CI and Holm p using a separate batching implementation.
 stats=rows(OUT/'paired_comparisons.csv');families=collections.defaultdict(list)
 for r in stats:families[r['multiplicity_family']].append(r)
 maxCI=0;maxP=0
 for family,rr in families.items():
  qs0=shard['pilot120'] if rr[0]['sample_set']=='Pilot120' else list(mr);qs=sorted(qs0,key=lambda q:mr[q]['image_sha256']);n=len(qs);ref=rr[0]['reference']
  data=np.array([[float(by[q][r['condition']]['score'])-float(by[q][ref]['score']) for r in rr] for q in qs]);rng=np.random.default_rng(SEED);bs=[]
  for j in range(0,20000,200):
   weights=rng.multinomial(n,np.full(n,1/n),size=min(200,20000-j));bs.append(100*(weights@data/n))
  ci=np.percentile(np.concatenate(bs),[2.5,97.5],axis=0)
  rng=np.random.default_rng(SEED);obs=np.abs(data.sum(0));ext=np.zeros(len(rr),dtype=np.int64)
  for j in range(0,20000,200):
   signs=rng.integers(0,2,size=(min(200,20000-j),n))*2-1;ext+=(np.abs(signs@data)>=obs[None]-1e-12).sum(0)
  p=(ext+1)/20001;adj=holm(p,int(rr[0]['family_size']))
  for j,r in enumerate(rr):
   err=max(abs(ci[0,j]-float(r['CI95_low_pp'])),abs(ci[1,j]-float(r['CI95_high_pp'])));maxCI=max(maxCI,err);assert err<1e-10
   assert abs(data[:,j].mean()*100-float(r['delta_pp']))<1e-10
   err=max(abs(p[j]-float(r['p_two_sided'])),abs(adj[j]-float(r['p_Holm'])));maxP=max(maxP,err);assert err<1e-12
 # Both workers passed identical strict gates and no own process remains.
 smokes=[read(OUT/f'gpu_smoke_worker{i}.json') for i in [0,1]]
 assert all(v['passed'] and v['n']==3 for v in smokes)
 assert [r['question_id'] for r in smokes[0]['records']]==[r['question_id'] for r in smokes[1]['records']]
 write(OUT/'gpu_smoke_validation.json',dict(passed=True,workers=2,identical3_questions=True,records=smokes))
 state=hardware();processes=read(OUT/'execution_processes.json');pids=[r['pid'] for r in processes['workers']];assert all(not pathlib.Path('/proc/'+str(pid)).exists() for pid in pids)
 release=[]
 for i,uuid in enumerate(protocol['GPU_UUIDs']):
  line=next(l for l in state['gpus'].splitlines() if uuid in l);fields=[p.strip() for p in line.split(',')]
  current=[l for l in state['processes'].splitlines() if uuid in l]
  assert not any(int(l.split(',')[0].strip()) in pids for l in current)
  release.append(dict(worker=i,uuid=uuid,physical_gpu=i+2,memory_used_MiB=int(fields[-3].split()[0]),utilization=fields[-1],no_compute_process=not current,own_pid_gone=True,external_compute_processes_now=current,interpretation='Own worker exits verified. Any later external job is recorded, never stopped.'))
 assert subprocess.check_output(['git','rev-parse','main'],cwd=ROOT.parent,text=True).strip()==protocol['main_before']
 write(OUT/'gpu_release_audit.json',dict(passed=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),released=release,own_pids=pids,snapshot=state,no_kill_or_preemption_commands_used=True,external_GPU0_GPU1_not_modified=True))
 figures=read(OUT/'figure_manifest.json');pdf_records=[]
 for r in figures['figures']:
  png=OUT/'figures'/(r['name']+'.png');pdf=OUT/'figures'/(r['name']+'.pdf')
  assert sha(png)==r['png_sha256'] and sha(pdf)==r['pdf_sha256']
  Image.open(png).verify()
  info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
  assert next(l.split(':',1)[1].strip() for l in info.splitlines() if l.startswith('Pages:'))=='1'
  text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
  assert 'TextVQA' in text or 'All-Text' in text or 'frozen' in text
  image_list=subprocess.check_output(['pdfimages','-list',str(pdf)],text=True)
  images=[l.strip() for l in image_list.splitlines() if l.split() and l.split()[0].isdigit()]
  if 'heatmap' not in r['name']:assert not images
  pdf_records.append(dict(name=r['name'],one_page=True,text_extracted=True,raster_objects=images,structure_backend='system Poppler pdfinfo/pdftotext/pdfimages',PNG_and_PDF_manifest_SHA_verified=True))
 write(OUT/'validation.json',dict(passed=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_model_files_fullSHA_reverified=True,all_images_input_specs_and_checkpoints_SHA_verified=True,source_freeze_unchanged=True,independent_official_score_formula=True,max_score_difference=maxscore,all_paired_CIs_independently_recomputed=True,max_CI_difference=maxCI,all_signflip_Holm_independently_recomputed=True,max_p_difference=maxP,complete_negative_and_positive_sets_verified=True,all_mask_budgets_independently_reconstructed=True,shards150_150_no_overlap=True,pilot_input_hash_rule_verified=True,dual_GPU_smoke=True,L31_AllText_Both_identical_tokens_and_first_logits_questions=final_layer_boundary,all_new_records=new,baselines=1200,PDFs=pdf_records,figures_require_visual_review_separately=True,main_unchanged=True,resources_released=True))
 allmandatory=av['mandatory_conditions_complete300']==139
 write(OUT/'completion.json',dict(status='COMPLETE' if allmandatory else 'PARTIAL_TIME_LIMIT' if not (OUT/'STOP.json').exists() else 'STOPPED_VALIDATION_FAILURE',complete=allmandatory,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),mandatory_conditions_complete300=av['mandatory_conditions_complete300'],mandatory_conditions_target=139,mandatory_target_records=41700,optional_conditions_complete300=av['optional_conditions_complete300'],optional_status='COMPLETE' if av['optional_conditions_complete300']==32 else 'NOT_COMPLETE_OPTIONAL',new_checkpoint_count=new,historical_reuse_count=1200,official_score_record_count=len(per),validated=True,GPU_released=True,no_background_inference=True))
 inventory={}
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and not p.name.endswith(('.tmp','.lock')) and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts:inventory[str(p.relative_to(OUT))]=dict(sha256=sha(p),bytes=p.stat().st_size)
 write(OUT/'artifact_hashes.json',dict(files=inventory,total_bytes=sum(v['bytes'] for v in inventory.values()),count=len(inventory),excluded='self, transient lock/tmp, Python bytecode'))
 print(json.dumps(dict(completion=read(OUT/'completion.json'),independent_validation=read(OUT/'validation.json'),artifacts=len(inventory)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
