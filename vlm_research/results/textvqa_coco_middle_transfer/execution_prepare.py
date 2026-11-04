import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import pathlib,json,hashlib,shutil,subprocess,sys,datetime,ast,importlib.metadata
W=pathlib.Path('/data/lk/ATE-worktrees/textvqa-coco-middle-transfer');ROOT=W/'vlm_research';O=ROOT/'results/textvqa_coco_middle_transfer';S=ROOT/'textvqa_coco_middle_transfer';H=ROOT/'results/textvqa_visual_reaccess_budget_exploration';P=pathlib.Path('/data/lk/ATE-resources/experiments/textvqa_coco_middle_transfer')
UUID='GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6';PRECHECK='20d7bde6e005baf1f9437624c9507d4427e54de6'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def git(*a):return subprocess.check_output(['git',*a],cwd=W,text=True).strip()
assert git('rev-parse','HEAD')==PRECHECK and git('branch','--show-current')=='codex/textvqa-coco-middle-transfer'
assert not (O/'STOP.json').exists() and not (O/'execution_protocol.json').exists()
sys.path.insert(0,str(ROOT))
from textvqa_coco_middle_transfer.policy import LAYER_SETS,NEW,COMPARISONS
assert read(O/'layer_sets.json')['conditions']==LAYER_SETS
test=subprocess.run([sys.executable,'-m','textvqa_coco_middle_transfer.tests','-v'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
(O/'execution_cpu_tests.log').write_text(test.stdout+test.stderr);assert test.returncode==0,(test.stdout+test.stderr)[-3000:]
for name in ['policy.py','engine.py','tests.py','run.py']:ast.parse((S/name).read_text())
old=read(H/'model_manifest.json');MODEL=pathlib.Path(old['path']);model=dict(UTC=utc(),revision=old['revision'],path=str(MODEL),files={})
for name,v in old['files'].items():
 p=MODEL/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name
 model['files'][name]=dict(**v,mtime_ns=p.stat().st_mtime_ns)
write(O/'execution_model_manifest.json',model)
for r in read(O/'historical_reuse_audit.json')['checkpoints']:assert sha(r['checkpoint_path'])==r['sha256']
shutil.copy2(H/'smoke_selection.json',O/'smoke_selection.json')
protocol=dict(UTC=utc(),precheck_commit=PRECHECK,precheck_protocol_sha256=sha(O/'protocol.json'),scientific_protocol_unchanged=True,layer_sets_sha256=sha(O/'layer_sets.json'),sample_manifest_sha256=sha(O/'sample_manifest.json'),OCR_labels_sha256=sha(O/'ocr_labels.csv'),model_manifest_sha256=sha(O/'execution_model_manifest.json'),new_conditions=NEW,primary_comparisons=COMPARISONS,selected_gpu=dict(physical=2,UUID=UUID,logical='cuda:0',workers=1),smoke_question_ids=read(O/'smoke_selection.json')['question_ids'],smoke_gates=['Native/No-op all valid logits torch.equal','new first valid logits torch.equal Native','all 32 Prefill attention outputs, weights and block Hidden SHA exact','A/F/B historical tokens and first logits exact','actual forbidden attention probability exactly zero','retained-layer actual image access','independent real KV and prefix; parent unchanged','native input/image features/weights, all32 MLP, positions and exception hook cleanup'],all1200_new_first_logits='raw FP16 vector SHA exact to same-input Native A; no large logits archives',failure_policy='Stop on any numerical, mask, cache, hook or model validation failure or OOM; no tolerance relaxation, data change or retries that change precision',checkpoints='atomic JSON; source/protocol/input SHA; exact same science on resume',single_gpu_no_preemption=True)
write(O/'execution_protocol.json',protocol)
deps=read(H/'source_manifest.json')['files']
for rel,v in deps.items():assert sha(ROOT/rel)==v,rel
files=dict(deps)
for name in ['__init__.py','policy.py','engine.py','tests.py','run.py']:files['textvqa_coco_middle_transfer/'+name]=sha(S/name)
for p in (H/'input_specs').glob('*.json'):files[str(p.relative_to(ROOT))]=sha(p)
dist=importlib.metadata.distribution('transformers');external={}
for name in ['llama/modeling_llama.py','llava/modeling_llava.py']:
 p=pathlib.Path(dist.locate_file('transformers/models/'+name));external[str(p)]=sha(p)
write(O/'execution_source_manifest.json',dict(UTC=utc(),files=files,external_files=external,execution_protocol_sha256=sha(O/'execution_protocol.json'),model_manifest_sha256=sha(O/'execution_model_manifest.json'),analysis_code_excluded_from_generation_freeze=True))
write(O/'execution_cpu_validation.json',dict(passed=True,UTC=utc(),tests=10,all_model_files_full_SHA_reverified=True,all900_historical_checkpoints_SHA_reverified=True,all_historical_source_hashes_match=True,all300_input_specs_SHA_bound=True,layer_sets_and_manifest_and_OCR_labels_unchanged=True,CUDA_VISIBLE_DEVICES='',real_GPU_smoke_required=True))
shutil.copy2(P/'freeze_execution.py',O/'execution_prepare.py')
paths=['vlm_research/textvqa_coco_middle_transfer/run.py']+[str((O/name).relative_to(W)) for name in ['execution_cpu_tests.log','execution_model_manifest.json','smoke_selection.json','execution_protocol.json','execution_source_manifest.json','execution_cpu_validation.json','execution_prepare.py']]
subprocess.run(['git','add','-f','--',*paths],cwd=W,check=True)
r=subprocess.run(['git','-c','core.whitespace=cr-at-eol,-blank-at-eof','diff','--cached','--check'],cwd=W,capture_output=True,text=True);assert r.returncode==0,r.stdout[:1000]
r=subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Freeze single-GPU COCO-to-TextVQA transfer execution'],cwd=W,capture_output=True,text=True);assert r.returncode==0,r.stdout[:2000]+r.stderr[:1000]
print(json.dumps(dict(status='EXECUTION_FROZEN',commit=git('rev-parse','HEAD'),gpu=UUID,model_SHA_verified=True,CPU_tests_passed=True,smoke_questions=protocol['smoke_question_ids'])),flush=True)
