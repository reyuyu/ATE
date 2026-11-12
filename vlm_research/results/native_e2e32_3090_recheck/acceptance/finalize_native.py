"""Scoped persistence and GPU release verification; no inference."""
from pathlib import Path
import json,hashlib,subprocess,shutil,datetime,os
O=Path('/home/featurize/work/ATE/vlm_research/results/native_e2e32_3090_recheck');R=O.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cmd(*a):return subprocess.check_output(a,text=True,timeout=25).strip()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
p=json.loads((O/'protocol.json').read_text());v=json.loads((O/'validation.json').read_text())
assert json.loads((O/'progress.json').read_text())['status']!='RUNNING'
assert not cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'),'GPU still occupied'
query=cmd('nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,power.limit','--format=csv,noheader');assert query.startswith(p['GPU_UUID']+',')
save('gpu_release.json',dict(status='PASS',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_query=query,compute_processes=[],server_not_shutdown=True,GPU_settings_unchanged=True))
(O/'GPU_final_detail.txt').write_text(cmd('nvidia-smi','-q','-d','PERFORMANCE,CLOCK,POWER,TEMPERATURE')+'\n')
for path,h in p['sources'].items():
 src=Path(path);assert sha(src)==h;dest=O/'frozen_sources'/src.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);assert sha(dest)==h
shutil.copyfile(Path(p['model_path'])/'.ate_revision.json',O/'model_integrity_receipt.json')
with (O/'report_zh.md').open('a') as f:f.write('\nGPU释放验收通过：无计算进程；`'+query+'`。服务器保持运行，未修改GPU设置。详见gpu_release.json。\n')
required=['protocol.json','environment.json','input_verification.json','smoke_validation.json','native_per_input.csv','native_session_summary.csv','drift_summary.json','gpu_telemetry.json','validation.json','report_zh.md']
assert all((O/n).is_file() for n in required)
(O/'entry_and_analysis_SHA256.txt').write_text(''.join(sha(path)+'  '+str(path)+'\n' for path in sorted((O/'code').glob('*.py'))))
files={}
for path in sorted(O.rglob('*')):
 if not path.is_file() or path.name in ['artifact_SHA256.json','run.lock'] or '__pycache__' in path.parts:continue
 assert path.resolve().is_relative_to(O.resolve())
 files[str(path.relative_to(O))]=dict(bytes=path.stat().st_size,SHA256=sha(path))
save('artifact_SHA256.json',dict(scope=str(O),files=files,model_or_image_payloads_included=False))
with (O/'artifact_SHA256.json').open('rb') as f:os.fsync(f.fileno())
print(json.dumps(dict(validation=v,GPU_release=query,artifacts=len(files)),ensure_ascii=False,indent=2))
