"""CPU-only check of namespace paths and two existing tensor bindings; no GPU model."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
L=Path('/home/featurize/ate-local-runs/qwen3090-29535-20261110');C=L/'control';W=Path('/home/featurize/work/ATE-qwen3vl-c3-3090-official-eval');O=W/'vlm_research/results/qwen3vl_c3_3090_official_eval'
assert os.environ['CUDA_VISIBLE_DEVICES']==''
config=json.loads((C/'namespace_storage_config.json').read_text());assert os.readlink('/proc/self/ns/mnt')!=config['host_mount_namespace']
source=json.loads((O/'source_manifest.json').read_text())
for name,h in source['files'].items():assert hashlib.sha256((W/name).read_bytes()).hexdigest()==h,name
protocol=json.loads((O/'protocol.json').read_text())
for name,spec in json.loads((O/'model_manifest.json').read_text())['files'].items():
    st=(Path(protocol['model_path'])/name).stat();assert st.st_size==spec['bytes'] and st.st_mtime_ns==spec['mtime_ns'],name
manifest=json.loads((O/'input_manifest.json').read_text());paths={s['image_path'] for s in manifest['rows']}
assert all(p.startswith('/home/featurize/work/ate-qwen3090-data/') for p in paths),'Unmapped image path'
sys.path.insert(0,str(W));from transformers import AutoProcessor
from vlm_research.qwen3vl_early_windows.experiment import prepare
processor=AutoProcessor.from_pretrained(protocol['model_path'],local_files_only=True)
historical=json.loads((O/'historical_inputs.json').read_text());checked=[]
for s in historical['rows'][:2]:
    x,layout=prepare(processor,s);assert layout==historical['input_bindings'][s['question_id']],s['question_id'];checked.append(s['question_id']);del x
assert not __import__('torch').cuda.is_initialized()
parent=O/'namespace_metadata_probe';parent.mkdir(exist_ok=True);latencies=[]
for _ in range(1000):
    before=time.monotonic();parent.mkdir(exist_ok=True);latencies.append(time.monotonic()-before)
parent.rmdir()
record=dict(status='LOCAL_PARENT_NAMESPACE_CPU_CHECK_PASS',at_UTC=datetime.now(timezone.utc).isoformat(),private_namespace=os.readlink('/proc/self/ns/mnt'),host_namespace=config['host_mount_namespace'],source_files_verified=len(source['files']),model_files_size_mtime_verified=len(json.loads((O/'model_manifest.json').read_text())['files']),benchmark_image_paths_mapped=len(paths),existing_historical_input_bindings_exact=checked,GPU_initialized=False,metadata_mkdir_max_ms=max(latencies)*1000,metadata_mkdir_mean_ms=sum(latencies)/len(latencies)*1000,mounts={p:subprocess.check_output(['findmnt','-T',p,'-n','-o','SOURCE,FSTYPE,TARGET'],text=True).strip() for p in ['/home/featurize/work',*[t for _,t in config['mounts']]]})
(L/'namespace_check_receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False))
