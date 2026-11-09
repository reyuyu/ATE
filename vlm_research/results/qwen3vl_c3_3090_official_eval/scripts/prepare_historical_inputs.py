"""CPU regression and byte-exact binding for the original frozen 1000 inputs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,platform,subprocess,sys,unittest
import torch,transformers
from transformers import AutoProcessor
W=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval');sys.path.insert(0,str(W))
O=W/'vlm_research/results/qwen3vl_c3_3090_official_eval';O.mkdir(parents=True,exist_ok=True)
C=Path('/home/featurize/work/ate-qwen3090-control-29535')
M=Path('/home/featurize/work/ate-models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
H=W/'vlm_research/results/qwen3vl_c3v2_exploratory_quality_speed'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
torch.set_num_threads(1)
assert transformers.__version__=='4.57.6' and not torch.cuda.is_initialized()
manifest=json.loads((H/'sample_manifest.json').read_text());assert sha(H/'sample_manifest.json')=='bb09f894a04f111f430841f4d074ff0d770585956a04c5401436351e235d4a88'
source=json.loads((H/'source_manifest.json').read_text())
relevant=['vlm_research/qwen3vl_c3v2_physical/engine.py','vlm_research/qwen3vl_early_windows/experiment.py','vlm_research/qwen3vl_smoke/common.py','vlm_research/qwen_quality_speed/generate.py']
for f in relevant:assert sha(W/f)==source['files'][f],f
from vlm_research.qwen3vl_c3v2_physical import tests as oldtests
oldtests.O=W/'vlm_research/results/qwen3vl_c3v2_physical_transfer'
suite=unittest.defaultTestLoader.loadTestsFromModule(oldtests)
with (O/'cpu_regression.log').open('w') as log:r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert r.wasSuccessful(),'Original Physical CPU regression failed; see cpu_regression.log'
print('ORIGINAL_CPU_REGRESSION_PASS',r.testsRun,flush=True)
assets=json.loads((C/'historical_asset_integrity.json').read_text());assert assets['status']=='PASS' and assets['image_count']==918
processor=AutoProcessor.from_pretrained(M,local_files_only=True)
from vlm_research.qwen3vl_early_windows.experiment import prepare
rows=[]
for i,s in enumerate(manifest['samples']):
 s=dict(s);s['original_image_path']=s['image_path'];s['image_path']=assets['images'][Path(s['image_path']).name]['path']
 inputs,layout=prepare(processor,s)
 assert layout==manifest['input_bindings'][s['question_id']],('Frozen tensor binding mismatch',s['question_id'])
 rows.append(s)
 if (i+1)%100==0:print('HISTORICAL_TENSOR_BINDING_PASS',i+1,'/1000',flush=True)
 del inputs
assert len(rows)==1000 and len({s['image_sha256'] for s in rows})==918
write(O/'historical_inputs.json',dict(status='PASS',rows=rows,input_bindings=manifest['input_bindings'],smoke_question_ids=manifest['smoke_question_ids'],benchmark_question_ids=manifest['benchmark_question_ids'],original_manifest_SHA256=sha(H/'sample_manifest.json'),path_only_remap=True))
write(O/'cpu_validation.json',dict(status='PASS',tests=r.testsRun,all_1000_tensor_hash_bindings_exact=True,unique_images=918,no_CUDA_initialization=not torch.cuda.is_initialized(),relevant_frozen_source_SHA256={f:sha(W/f) for f in relevant},at_UTC=datetime.now(timezone.utc).isoformat()))
print('CPU_HISTORICAL_INPUTS_PASS',flush=True)
