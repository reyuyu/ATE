import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import pathlib,subprocess,shutil,json,hashlib,sys,time,datetime,ast
OLD=pathlib.Path('/data/lk/ATE-worktrees/textvqa-coco-middle-transfer');W=pathlib.Path('/data/lk/ATE-worktrees/textvqa-single-layer-visual-ablation')
P=pathlib.Path('/data/lk/ATE-resources/experiments/textvqa_single_layer_visual_ablation');BASE='c43f0f075989318c6bc061e411d4d2e703f8e9b8';BRANCH='codex/textvqa-single-layer-visual-ablation'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def git(*a,cwd=OLD):return subprocess.check_output(['git',*a],cwd=cwd,text=True).strip()
assert git('rev-parse','HEAD')==BASE and not git('status','--porcelain')
bootstrap_resume=W.exists()
if bootstrap_resume:
 assert git('rev-parse','HEAD',cwd=W)==BASE and git('branch','--show-current',cwd=W)==BRANCH
 assert not (W/'vlm_research/results/textvqa_single_layer_visual_ablation').exists(),'Only resume an uninitialized worktree, never existing scientific results'
for p in ['/','/data','/data/lk','/data/lk/ATE-worktrees',str(OLD)]:assert not (pathlib.Path(p)/'AGENTS.md').exists()
assert not any(pathlib.PurePosixPath(p).name=='AGENTS.md' for p in git('ls-tree','-r','--name-only','HEAD').splitlines())
P.mkdir(parents=True,exist_ok=bootstrap_resume)
if not bootstrap_resume:subprocess.run(['git','worktree','add','--no-checkout','-b',BRANCH,str(W),BASE],cwd=OLD,check=True,capture_output=True)
gd=pathlib.Path(git('rev-parse','--absolute-git-dir',cwd=W));(gd/'info').mkdir(exist_ok=True)
subprocess.run(['git','sparse-checkout','init','--no-cone'],cwd=W,check=True,capture_output=True)
subprocess.run(['git','sparse-checkout','set','--no-cone','/*','!/*/','/vlm_research/','!/vlm_research/results/'],cwd=W,check=True,capture_output=True)
R=W/'vlm_research';O=R/'results/textvqa_single_layer_visual_ablation';S=R/'textvqa_single_layer_visual_ablation';S.mkdir();O.mkdir(parents=True)
(S/'__init__.py').write_text('')
for n in ['policy','engine','tests','run']:shutil.copy2(P.parent/('single_layer_'+n+'.py'),S/(n+'.py'));ast.parse((S/(n+'.py')).read_text())
sys.path.insert(0,str(R))
from textvqa_single_layer_visual_ablation.policy import *
HO=OLD/'vlm_research/results/textvqa_coco_middle_transfer'
for n in ['sample_manifest.json','ocr_labels.csv','classification_freeze.json','execution_model_manifest.json','smoke_selection.json']:shutil.copy2(HO/n,O/n)
manifest=read(O/'sample_manifest.json');rows=manifest['rows'];assert len(rows)==300 and len({r['image_sha256'] for r in rows})==300
ordered=sorted(rows,key=lambda r:r['image_sha256']);assign={r['question_id']:j%2 for j,r in enumerate(ordered)}
rank=sorted(rows,key=lambda r:hashlib.sha256((r['question_id']+'|'+r['image_sha256']).encode()).hexdigest());pilot=[r['question_id'] for r in rank[:120]]
write(O/'shards.json',dict(rule='sort unique Image SHA256 lexicographically, alternating worker 0/1; all conditions for a question owned by same worker',by_question=assign,counts=[sum(v==i for v in assign.values()) for i in range(2)],pilot120=pilot,pilot_rule='lowest SHA256(Question ID + literal | + Image SHA256); input only, no Gold/score/OCR labels',pilot_counts=[sum(assign[q]==i for q in pilot) for i in range(2)]))
assert set(pilot)<=set(assign) and len(set(pilot))==120
reuse=read(HO/'historical_reuse_audit.json')['checkpoints'];reuse=[dict(r,condition={'A':'Native','F':'Decode-None','B':'Both-None'}[r['condition']]) for r in reuse]
for r in rows:
 p=HO/'checkpoints'/r['question_id']/'C.json';reuse.append(dict(question_id=r['question_id'],condition='Middle',checkpoint_path=str(p),sha256=sha(p)))
for r in reuse:assert sha(r['checkpoint_path'])==r['sha256']
for r in rows:
 assert sha(r['image_path'])==r['image_sha256'];assert sha(INPUTS/(r['question_id']+'.json'))==r['input_spec_sha256']
write(O/'historical_reuse_audit.json',dict(source_commit=BASE,checkpoints=reuse,all1200_checkpoint_hashes_verified=True,all300_image_hashes_verified=True))
model=read(O/'execution_model_manifest.json')
for n,v in model['files'].items():assert sha(pathlib.Path(model['path'])/n)==v['sha256']
# The clock began at first task/resource read; never reset after retries.
start=datetime.datetime(2026,11,5,5,17,30,tzinfo=datetime.timezone.utc).timestamp();deadline=start+6*3600
write(O/'protocol.json',dict(base_commit=BASE,branch=BRANCH,start_UTC=datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),deadline_UTC=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),start_epoch=start,deadline_epoch=deadline,gpu_deadline_epoch=deadline-1800,maximum_wall_hours=6,model_revision=model['revision'],model_manifest_sha256=sha(O/'execution_model_manifest.json'),manifest_sha256=sha(O/'sample_manifest.json'),shards_sha256=sha(O/'shards.json'),OCR_labels_sha256=sha(O/'ocr_labels.csv'),conditions=CONDITIONS,primary_conditions=PRIMARY,secondary_conditions=SECONDARY,optional_conditions=OPTIONAL,initial128=NATIVE,stage_order=['pilot120 all128 Native','300 All-Text/Both/Middle-minus','300 Ponly/Donly','optional Addback32 only after A/B complete and projected safe before deadline'],GPU_UUIDs=UUIDS,workers=2,shard_counts=[150,150],dtype='float16',attention='eager',transformers='4.49.0',greedy=True,max_new_tokens=32,logits_to_keep=1,softmax_pre_hard_mask='negative infinity for exactly query_rows x original Image Keys; full KV retained',R='after final image token through position P-2 inclusive; pre-image system tokens untouched',baseline_reuse=['Native','Middle','Decode-None','Both-None descriptive'],bootstrap=dict(repetitions=20000,seed=SEED,unit='whole Image SHA paired cluster',CI='ordinary percentile95, pointwise'),multiplicity=dict(Native128='one Holm family128 at each sample size separately; not128 selected layers',Middle11='separate Holm family11',optional32='separate Holm family32',p='two-sided whole-image paired sign-flip20000, plus-one; absent comparisons padded with p1 if partial; all exploratory'),no_noninferiority_claim=True,all_safety_failures='STOP both workers; no precision/tolerance/input changes',resource_policy='only genuinely idle physical GPU2/3 by exact UUID; never GPU0/1; one model/worker; no preemption',checkpoint_policy='atomic per image-condition JSON; protocol/source/input hashes required on resume',progress_log_minutes=30,main_before=git('rev-parse','main')))
write(O/'resource_preflight.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True),disk=shutil.disk_usage('/data')._asdict()))
test=subprocess.run([sys.executable,'-m','textvqa_single_layer_visual_ablation.tests','-v'],cwd=R,env=os.environ,capture_output=True,text=True);(O/'cpu_tests.log').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stdout+test.stderr
write(O/'cpu_validation.json',dict(passed=True,tests=7,all171_intervention_masks_and3_baselines_CPU_audited=True,CUDA_not_initialized=True,all_model_fullSHA_reverified=True,all1200_history_SHA_verified=True,all300_input_specs_and_imageSHA_verified=True,shards150_each_no_overlap=True,pilot120_input_only=True))
deps=read(HO/'execution_source_manifest.json');files={}
for n,v in deps['files'].items():
 p=R/n
 if n.startswith('results/'):p=OLD/'vlm_research'/n
 assert sha(p)==v,n
 if not n.startswith('textvqa_coco_middle_transfer/'):files[str(p)]=v
for p in S.glob('*.py'):files[str(p)]=sha(p)
for n,v in deps['external_files'].items():assert sha(n)==v;files[n]=v
write(O/'execution_source_manifest.json',dict(files=files,protocol_sha256=sha(O/'protocol.json'),model_manifest_sha256=sha(O/'execution_model_manifest.json'),analysis_excluded=True))
write(O/'completion.json',dict(status='FROZEN_PENDING_DUAL_GPU_SMOKE',complete=False,new_checkpoints=0,target_mandatory=41700,optional=9600))
for n in ['setup.py']:shutil.copy2(P.parent/'single_layer_setup.py',O/'prepare.py')
subprocess.run(['git','add','-f','--',str(S.relative_to(W)),str(O.relative_to(W))],cwd=W,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Freeze dual-GPU TextVQA single-layer visual ablation protocol and source'],cwd=W,check=True,capture_output=True)
print(json.dumps(dict(status='FROZEN',commit=git('rev-parse','HEAD',cwd=W),out=str(O),pilot_counts=read(O/'shards.json')['pilot_counts'],deadline_UTC=read(O/'protocol.json')['deadline_UTC'])),flush=True)
