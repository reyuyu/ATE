"""Prepare and audit the exact transfer study. Never starts a GPU worker."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,subprocess,shutil,hashlib,json,csv,datetime,contextlib,sys,collections,importlib.metadata
OLD=pathlib.Path('/data/lk/ATE-worktrees/textvqa-ocr-reaccess-mechanism')
W=pathlib.Path('/data/lk/ATE-worktrees/textvqa-coco-middle-transfer')
P=pathlib.Path('/data/lk/ATE-resources/experiments/textvqa_coco_middle_transfer')
BASE='51761dba1b75f0081a7be2df728392483d609c69';BRANCH='codex/textvqa-coco-middle-transfer'
def run(*a,cwd=OLD):return subprocess.check_output(list(a),cwd=cwd,text=True).strip()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rows(p):return list(csv.DictReader(p.open()))
assert run('git','rev-parse','HEAD')==BASE and not run('git','status','--porcelain')
assert not W.exists() and not P.exists(),'Do not overwrite an existing transfer study'
assert shutil.disk_usage('/data').free>25*1024**3
for d in ['/','/data','/data/lk','/data/lk/ATE-worktrees',str(OLD)]:assert not (pathlib.Path(d)/'AGENTS.md').exists(),d
assert not any(pathlib.PurePosixPath(p).name=='AGENTS.md' for p in run('git','ls-tree','-r','--name-only',BASE).splitlines())
assert subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+BRANCH],cwd=OLD).returncode==1
procs=run('ps','-eo','pid,ppid,args').splitlines()
assert not any('python -m textvqa_coco_middle_transfer' in p for p in procs),'Existing transfer worker'
main=run('git','rev-parse','main')
P.mkdir(parents=True)
subprocess.run(['git','worktree','add','--no-checkout','-b',BRANCH,str(W),BASE],cwd=OLD,check=True,capture_output=True)
gd=pathlib.Path(run('git','rev-parse','--absolute-git-dir',cwd=W));(gd/'info').mkdir(exist_ok=True)
subprocess.run(['git','sparse-checkout','init','--no-cone'],cwd=W,check=True,capture_output=True)
subprocess.run(['git','sparse-checkout','set','--no-cone','/*','!/*/','/vlm_research/','!/vlm_research/results/','/vlm_research/results/textvqa_visual_reaccess_budget_exploration/','/vlm_research/results/textvqa_ocr_reaccess_mechanism/'],cwd=W,check=True,capture_output=True)
assert run('git','rev-parse','HEAD',cwd=W)==BASE and run('git','branch','--show-current',cwd=W)==BRANCH
ROOT=W/'vlm_research';O=ROOT/'results/textvqa_coco_middle_transfer';O.mkdir()
S=ROOT/'textvqa_coco_middle_transfer';S.mkdir();(S/'__init__.py').write_text('')
for name in ['policy','engine','tests']:shutil.copy2(P.parent/('coco_transfer_'+name+'.py'),S/(name+'.py'))
sys.path.insert(0,str(ROOT))
from textvqa_coco_middle_transfer.policy import LAYER_SETS,NEW,COMPARISONS,SEED
H=ROOT/'results/textvqa_visual_reaccess_budget_exploration';OCR=ROOT/'results/textvqa_ocr_reaccess_mechanism'
manifest=read(H/'sample_manifest.json');assert len(manifest['rows'])==300
assert len({r['question_id'] for r in manifest['rows']})==300
assert len({r['image_sha256'] for r in manifest['rows']})==300
shutil.copy2(H/'sample_manifest.json',O/'sample_manifest.json')
for name in ['ocr_labels.csv','classification_evidence.csv','classification_protocol.md','classification_freeze.json']:shutil.copy2(OCR/name,O/name)
freeze=read(OCR/'classification_freeze.json');assert sha(O/'ocr_labels.csv')==freeze['labels_sha256']
labels=rows(O/'ocr_labels.csv');assert collections.Counter(r['category'] for r in labels)==dict(freeze['counts'])
hist=rows(H/'per_question.csv');assert len(hist)==5040
selected=[r for r in hist if r['condition'] in ['A','B','F']];assert len(selected)==900
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
ev=TextVQAAccuracyEvaluator();maximum=0;expected=dict(A=45.23333333333333,B=21.83333333333333,F=31.4);vals=collections.defaultdict(list);cp_audit=[]
with open(os.devnull,'w') as null,contextlib.redirect_stderr(null):
 for r in selected:
  s=ev.eval_pred_list([dict(pred_answer=r['answer'],gt_answers=json.loads(r['answers']))]);maximum=max(maximum,abs(s-float(r['score'])));assert maximum<1e-12
  cp=ROOT/r['checkpoint']
  if not cp.exists():cp=pathlib.Path('/data/lk/ATE-worktrees/textvqa-visual-reaccess-budget/vlm_research')/r['checkpoint']
  assert sha(cp)==r['checkpoint_sha256'];v=read(cp)
  assert v['generation']['token_ids']==json.loads(r['token_ids']) and v['generation']['text']==r['answer'] and v['image_sha256']==r['image_sha256']
  assert v['input_hashes']==read(H/'input_specs'/(r['question_id']+'.json'))['input_hashes']
  vals[r['condition']].append(s);cp_audit.append(dict(question_id=r['question_id'],condition=r['condition'],checkpoint_path=str(cp),sha256=sha(cp),input_hashes=v['input_hashes']))
assert all(len(vals[c])==300 and abs(100*sum(vals[c])/300-a)<1e-10 for c,a in expected.items())
for r in manifest['rows']:assert sha(pathlib.Path(r['image_path']))==r['image_sha256']
write(O/'historical_reuse_audit.json',dict(UTC=utc(),base=BASE,TextVQA_commit='65c7a9c66cd4ea3747fc11e132c4d6a01b5cd5b4',OCR_commit=BASE,baseline_records=900,official_score_max_difference=maximum,baseline_accuracy=expected,manifest_sha256=sha(H/'sample_manifest.json'),labels_sha256=sha(O/'ocr_labels.csv'),all300_image_hashes_verified=True,checkpoints=cp_audit))
for n in ['protocol.json','layer_sets.json','report_zh.md']:
 data=subprocess.check_output(['git','show',BASE+':vlm_research/results/coco_decode_visual_regions_60/'+n],cwd=W) if n!='layer_sets.json' else None
 if data is not None:
  (O/'history_context').mkdir(exist_ok=True);(O/'history_context'/('coco_'+n)).write_bytes(data)
coco=read(O/'history_context/coco_protocol.json')
assert coco['conditions']['Middle']==LAYER_SETS['C']['Decode'] and coco['conditions']['Early']==LAYER_SETS['D']['Decode'] and coco['conditions']['Late']==LAYER_SETS['E']['Decode']
write(O/'layer_sets.json',dict(UTC=utc(),index_base=0,conditions=LAYER_SETS,random_seed=SEED,random_algorithm='Python random.Random(20261105).sample(range(1,32),10), sorted with L0 prepended',new_conditions=NEW,primary_comparisons=COMPARISONS,layer_sets_frozen_before_any_new_inference=True,all_new_prefill_native=True))
snap=dict(UTC=utc(),inventory=run('nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader'),compute_processes=run('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'),disk=shutil.disk_usage('/data')._asdict(),main_before=main,processes=[p for p in procs if any(x in p for x in ['3620369','3620370','3640743','3640750'])],new_gpu_workers_started=0,model_loaded=False)
write(O/'resource_preflight.json',snap)
occupied={p.split(',')[0].strip() for p in snap['compute_processes'].splitlines() if p.strip()}
idle=[line for line in snap['inventory'].splitlines() if 'RTX 3090' in line and line.split(',')[2].strip() not in occupied and int(line.split(',')[4].split()[0])<500]
write(O/'protocol.json',dict(UTC=utc(),base_commit=BASE,branch=BRANCH,status='PREPARED_BLOCKED_NO_FREE_GPU' if not idle else 'PREPARED_REQUIRES_GPU_SMOKE',question_count=300,new_generations_required=1200,baseline_reuse=900,model='LLaVA-1.5-7B',revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',dtype='float16',attention='eager',transformers='4.49.0',do_sample=False,max_new_tokens=32,logits_to_keep=1,scorer='official MMF TextVQA ten-annotator leave-one-out consensus soft accuracy',manifest_sha256=sha(O/'sample_manifest.json'),historical_manifest_worker_field='preserved provenance only; a future single-GPU worker owns all 300 rows',labels_sha256=sha(O/'ocr_labels.csv'),layer_sets_sha256=sha(O/'layer_sets.json'),all_new_conditions='entire Prefill native including Pminus1; only true Decode query -> original Image Key hard mask before softmax; full real KV retained',gpu_policy='one truly idle RTX3090 by UUID; no sharing, preemption or waiting daemon; fresh resource check and all mandatory GPU gates before formal generation',bootstrap=dict(seed=SEED,repetitions=20000,unit='image_sha256 whole-cluster paired',CI='ordinary pointwise percentile 95%'),primary_comparisons=COMPARISONS,multiplicity='Holm across the 5 prespecified two-sided paired effects; cluster sign-flip p with plus-one. OCR subgroup comparisons descriptive ordinary CIs, no subgroup significance claims without separately declared correction.',random_rule='one preselected seed, never choose by accuracy',late_budget_difference='Late has 12 layers; Middle/Early/Random have 11',scope='functional hard masks; no physical KV deletion, kernel or speed claims',failure_policy='stop on GPU unavailability, smoke failure or OOM; no precision/data/tolerance changes'))
test=subprocess.run([sys.executable,'-m','textvqa_coco_middle_transfer.tests','-v'],cwd=ROOT,capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
(O/'cpu_tests.log').write_text(test.stdout+test.stderr)
assert test.returncode==0,(test.stdout+test.stderr)[-4000:]
write(O/'cpu_validation.json',dict(passed=True,UTC=utc(),tests=10,mask_policy_engine='historical textvqa_reaccess engine with only in-process layer table and output guard rebound',baseline_records_rescored=900,max_score_difference=maximum,all300_original_image_hashes_verified=True,all900_original_checkpoint_hashes_verified=True,frozen_OCR_labels_unchanged=True,new_prefill_masks_CPU_native_exact=True,CPU_decode_forbidden_probabilities_exact_zero=True,CUDA_not_initialized_by_tests=True,CPU_checks_do_not_replace_real_GPU_smoke=True))
write(O/'source_manifest.json',dict(UTC=utc(),files={str(p.relative_to(W)):sha(p) for p in list(S.glob('*.py'))+[ROOT/'textvqa_reaccess/engine.py',ROOT/'textvqa_reaccess/common.py']},protocol_sha256=sha(O/'protocol.json'),layer_sets_sha256=sha(O/'layer_sets.json')))
write(O/'validation.json',dict(status='BLOCKED_NO_FREE_GPU' if not idle else 'GPU_SMOKE_NOT_RUN',CPU_passed=True,GPU_smoke_passed=False,GPU_smoke_status='NOT_RUN',new_generations=0,required_new_generations=1200,experiment_complete=False,historical_reuse_verified=True,all_inputs_and_OCR_labels_preserved=True))
write(O/'completion.json',dict(status='BLOCKED_NO_FREE_GPU' if not idle else 'PENDING_GPU_SMOKE',UTC=utc(),new_generation_count=0,target_new_generation_count=1200,complete=False,no_background_waiter=True))
write(O/'resource_audit.json',dict(UTC=utc(),GPU_workers_started=0,GPU_memory_allocated_by_this_task=0,model_weights_loaded=False,other_processes_untouched=True,no_GPU_to_release=True,no_automatic_retry_or_polling=True,blocking_reason='All four RTX3090 GPUs have active compute processes' if not idle else 'Fresh real GPU smoke still required'))
print(json.dumps(dict(status=read(O/'completion.json')['status'],worktree=str(W),out=str(O),baseline_accuracy=expected,CPU_tests_passed=True,random11=LAYER_SETS['R']['Decode'],idle_GPU_count=len(idle),disk_free_GiB=snap['disk']['free']/1024**3)))
