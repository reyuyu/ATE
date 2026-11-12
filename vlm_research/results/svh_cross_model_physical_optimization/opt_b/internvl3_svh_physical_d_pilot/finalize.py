"""CPU-only independent validation and verified persistent mirror; no GitHub push here."""
import csv,json,os,shutil,time
from pathlib import Path
from .common import *
def main():
 assert read(RUN/'execution_status.json')['status']=='COMPLETE_WITH_NUMERICAL_WAIVER'
 pid=read(RUN/'continuation_launch.json')['pid']
 assert not Path('/proc/'+str(pid)).exists(),'GPU worker must exit before finalization'
 assert read(RUN/'quality_complete.json')['generations']==360
 assert read(RUN/'efficiency_complete.json')['formal_requests']==450
 from .analyze import main as analyze
 analyze()
 m=read(RUN/'sample_manifest.json');lookup={s['question_id']:s for s in m['samples']}
 protocol_sha=sha(RUN/'protocol.json');source_sha=sha(RUN/'source_manifest.json')
 for path in (RUN/'quality_checkpoints').glob('*.json'):
  r=read(path);s=lookup[r['question_id']]
  assert r['protocol_sha256']==protocol_sha and r['source_manifest_sha256']==source_sha
  assert r['image_sha256']==s['image_sha256'] and r['gold']==s['answer']
  assert len({r['conditions'][method]['input_binding'] for method in METHODS})==1
 for path in (RUN/'efficiency_checkpoints').glob('*.json'):
  r=read(path);s=lookup[r['question_id']]
  assert r['protocol_sha256']==protocol_sha and r['source_manifest_sha256']==source_sha
  assert r['image_sha256']==s['image_sha256'] and r['question_id'] in m['efficiency_question_ids']
  assert r['input_tensor_binding']==digest(r['input_binding'])
  assert r['input_binding']['prompt']==s['prompt'] and r['input_binding']['input_ids']==s['input_ids']
  assert r['repeat'] in range(5) and r['method'] in METHODS
 # Source snapshots let an auditor resolve the first failed experiment after the current files changed.
 attempt=RUN/'engineering_attempts/attempt01_frozen_gate_failure'
 binding=[]
 for source,h in read(attempt/'source_manifest.json').items():
  snap=attempt/'source'/Path(source).name
  if snap.is_file() and sha(snap)==h:
   binding.append(dict(original_path=source,archived_path=str(snap.relative_to(RUN)),sha256=h))
  else:
   assert sha(source)==h,source
   binding.append(dict(original_path=source,unchanged_source_path=source,sha256=h))
 save(RUN/'engineering_attempts/attempt01_source_binding.json',binding)
 reference=read(RUN/'model_manifest.json')['prior_verification']
 assert sha(reference['model_verification_file'])==reference['model_verification_sha256']
 shutil.copy2(reference['model_verification_file'],RUN/'model_original_verification.json')
 telemetry=[]
 for filename in ['gpu_telemetry.csv','gpu_telemetry_continuation.csv']:
  with (RUN/filename).open() as f:
   rows=list(csv.DictReader(f));assert rows
   for row in rows:
    clean={k.strip():v.strip() for k,v in row.items()}
    assert clean['uuid']==GPU
    telemetry.append(clean)
 def values(prefix):
  key=next(k for k in telemetry[0] if k.startswith(prefix))
  return [float(r[key].split()[0]) for r in telemetry if r[key]!='[N/A]']
 save(RUN/'gpu_telemetry_summary.json',dict(records=len(telemetry),GPU_UUIDs=sorted({r['uuid'] for r in telemetry}),
  SM_clock_MHz_minmax=[min(values('clocks.current.sm')),max(values('clocks.current.sm'))],
  power_W_minmax=[min(values('power.draw')),max(values('power.draw'))],
  temperature_C_minmax=[min(values('temperature.gpu')),max(values('temperature.gpu'))],
  note='Telemetry includes idle gaps and warmup/smoke; ranges are not constant-clock claims. No dedicated performance drift threshold was preregistered for this InternVL pilot.'))
 apps=command('nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader')
 save(RUN/'gpu_final_release.json',dict(worker_pid=pid,worker_process_absent=True,
  compute_apps=apps,GPU=command('nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader'),
  own_GPU_released=True,no_other_process_killed=True,server_not_closed=True))
 # Append exact EOS/empty/truncation and measured component/call summaries to the report.
 quality=list(csv.DictReader((RUN/'per_question.csv').open()))
 smoke=read(RUN/'smoke_validation.json')
 with (RUN/'report_zh.md').open('a') as f:
  f.write('\n| 条件 | EOS /120 | 截断 /120 | 空答案 /120 | @32 Attention调用 | @32 Q/K/V/o_proj调用 |\n|---|---:|---:|---:|---:|---:|\n')
  for method,a in zip(METHODS,smoke['fixed32_audits']):
   rows=[r for r in quality if r['method']==method]
   f.write(f"| {method} | {sum(r['eos']=='True' for r in rows)} | {sum(r['truncated']=='True' for r in rows)} | {sum(r['empty']=='True' for r in rows)} | {a['attention_calls']} | {a['projection_calls']} |\n")
  f.write('\n上述调用次数在正式计时外实测于同一32步路径；Projection为Q/K/V/o_proj四者总数。组件Event时间、KV Capacity及Peak Allocated/Reserved在efficiency_summary.csv中给出。GPU全程UUID已逐请求与遥测复核。遥测含请求间空闲，不能把频率范围解释为恒频；本轮未预注册专门的5%漂移探针，不作“无漂移”证明。\n')
 # Mirror after GPU release, so NFS never holds a live model during delivery.
 OUT.mkdir(parents=True,exist_ok=True)
 count=0;bytes_total=0
 for path in sorted(RUN.rglob('*')):
  if not path.is_file() or path.name.endswith('.tmp'):continue
  target=OUT/path.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True)
  expected=sha(path)
  if not target.exists() or sha(target)!=expected:shutil.copy2(path,target)
  assert sha(target)==expected
  count+=1;bytes_total+=path.stat().st_size
 save(RUN/'persistence_verification.json',dict(status='PASS',files=count,bytes=bytes_total,source=str(RUN),destination=str(OUT),all_SHA256_equal=True))
 shutil.copy2(RUN/'persistence_verification.json',OUT/'persistence_verification.json')
 alias=Path('/home/featurize/work/ATE/vlm_research/results/internvl3_svh_physical_d_pilot_3090')
 if alias.is_symlink():assert alias.resolve()==OUT
 elif not alias.exists():alias.symlink_to(OUT,target_is_directory=True)
 else:assert alias.resolve()==OUT
 print('FINALIZATION_PASS',count,bytes_total,flush=True)
if __name__=='__main__':main()
