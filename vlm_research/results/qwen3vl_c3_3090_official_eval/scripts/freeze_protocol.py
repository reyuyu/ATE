"""Freeze task authorization, Original math, request wrapper and actual hardware."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,platform,subprocess,sys
import torch,transformers,PIL
W=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval');O=W/'vlm_research/results/qwen3vl_c3_3090_official_eval';C=Path('/home/featurize/work/ate-qwen3090-control-29535')
M=Path('/home/featurize/work/ate-models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
assert not torch.cuda.is_initialized()
receipt=read(C/'model_integrity.json');assert receipt['status']=='VERIFIED'
old=read(W/'vlm_research/results/qwen3vl_c3v2_exploratory_quality_speed/model_manifest.json');files={}
for name,h in old['files'].items():
 p=M/name;state=receipt['files'][name];assert p.stat().st_size==state['bytes']
 actual=state['hash'] if name.endswith('.safetensors') else sha(p)
 assert actual==h,('Historical model differs',name)
 files[name]=dict(SHA256=h,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,verification='pinned streaming SHA256 during download' if name.endswith('.safetensors') else 'SHA256 against historical asset')
write(O/'model_manifest.json',dict(status='PASS',model=receipt['model'],revision=receipt['revision'],path=str(M),files=files,all_historical_model_SHA256_exact=True))
cpu=read(O/'cpu_validation.json');assert cpu['status']=='PASS' and cpu['all_1000_tensor_hash_bindings_exact']
sources={str(p.relative_to(W)):sha(p) for p in (O/'scripts').glob('*.py')}
sources.update(cpu['relevant_frozen_source_SHA256'])
write(O/'source_manifest.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),files=sources,original_engine_unchanged=True,wrapper_changes=['omit unused Repaired imports','parameterize per-benchmark generation cap','add fixed32 using same native incremental loop'],no_kernel_bridge_used=True))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,memory.free,driver_version,utilization.gpu','--format=csv,noheader'],text=True).strip()
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip();assert not procs
assert 'GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6' in gpu and 'RTX 3090' in gpu
write(O/'environment.json',dict(at_UTC=datetime.now(timezone.utc).isoformat(),python=sys.version,torch=torch.__version__,torch_CUDA=torch.version.cuda,transformers=transformers.__version__,Pillow=PIL.__version__,gpu_preflight=gpu,gpu_processes_before=procs,platform=platform.platform(),pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),TF32_allow_matmul=torch.backends.cuda.matmul.allow_tf32,default_dtype=str(torch.get_default_dtype()),CUDA_initialized=False))
protocol=dict(status='FROZEN_BEFORE_GPU',model=receipt['model'],revision=receipt['revision'],dtype='bfloat16',attention='eager',batch_size=1,greedy=True,logits_to_keep=1,model_path=str(M),gpu_UUID='GPU-0c68c509-f592-aff4-73ff-40a5d798ffd6',conditions=['Native','Dense-F','Physical-Original'],writer_layers=list(range(22)),visual_hidden_exit_before_layer=22,answer_attention_skip_layers=list(range(1,7)),original_visual_KV_answer_access=False,deepstack='native three injections',rotary='native mRoPE absolute positions, no renumbering',numerical_gate=dict(max_abs=.5,mean_abs=.02,top1_equal=True,full_tokens_equal=True,Native_noop_all_logits='torch.equal',strict_failure_label='NUMERICAL_EQUIVALENCE_FAILED',exploratory_continuation_only_if_all_structural_and_finite_checks_pass=True),historical_GQA1000=dict(label='HISTORICAL_GQA1000_REPRODUCTION',questions=1000,image_clusters=918,sample_manifest_SHA256='bb09f894a04f111f430841f4d074ff0d770585956a04c5401436351e235d4a88',max_new_tokens=32,new_GPU_predictions_required=True),seven_benchmarks=dict(conditions=['Native','Physical-Original'],max_new_tokens={'GQA':32,'ScienceQA-IMG':1024,'TextVQA':128,'POPE':128,'MME':128,'MMBench':1024,'OCRBench-v1':128},processor='exact fixed model default RGB Processor; no pixel/token reduction',protocol_views='ATE paper and official split/scoring retained separately; unchanged identical input requests may share newly generated predictions, never old model predictions',MMBench='author exact_matching circular; no paid LLM judge; unresolved official judge status explicit'),benchmark=dict(questions=48,regimes=['natural','fixed16','fixed32'],warmups=2,measured_repeats=5,condition_order='cyclic Latin rotation by SHA256 canonical JSON(question_id), repeat modulo 3',cache='fresh independent cache per request; fixed16=1 Prefill+15 real Decode; fixed32=1+31',timing='same shared wrapper all conditions, synchronized request wall includes H2D, vision/projector, context/cache setup, Prefill, incremental Decode, greedy/EOS and context cleanup; excludes model load, file decode, processor, text postdecode and audit extraction',aggregation='per-input repeat median; ratio of sums; paired image cluster bootstrap',measured_hooks=False),bootstrap=dict(replicates=20000,seed=20261106,confidence=.95,unit='image SHA256 cluster',question_weighted=True),source_manifest_SHA256=sha(O/'source_manifest.json'),model_manifest_SHA256=sha(O/'model_manifest.json'),historical_inputs_SHA256=sha(O/'historical_inputs.json'),no_push_authorized=True,no_training=True,no_search=True,no_repaired=True)
p=O/'protocol.json'
if p.exists():assert read(p)==protocol,'Never overwrite a changed frozen protocol'
else:write(p,protocol)
print('PROTOCOL_MODEL_HARDWARE_FREEZE_PASS',flush=True)
