"""CPU-only post-run release evidence, artifact hashes and lightweight archive."""
from c55_common import *
import hashlib,zipfile
def streamsha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 v=read(N/'validation.json');assert v['execution_correctness']=='PASS' and v['formal_requests']==240;verify();workers=[]
 for session in (1,2):
  p=N/f'session{session}';s=read(p/'progress.json');r=read(p/'resource_release.json');assert not Path(f'/proc/{s["pid"]}').exists()
  workers.append(dict(session=session,pid=s['pid'],process_exited=True,pre_exit_allocated_bytes=r['pre_exit_allocated_bytes']))
 ps=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])
 resource=dict(at_ns=time.time_ns(),GPU=gpu(),observed_compute_processes=ps.splitlines() if ps else [],workers=workers,our_GPU_contexts_released_by_process_exit=True,server_left_running=True,tracked_Git_clean=True,source_integrity='PASS')
 save(N/'final_resource_status.json',resource)
 report=N/'report_zh.md';text=report.read_text();assert '## GPU释放与归档' not in text
 text+='\n## GPU释放与归档\n\n两个测量进程均已退出。最终GPU显存占用'+str(resource['GPU']['memory.used'])+' MiB；其他计算进程记录见`final_resource_status.json`。进程退出前分配量原样保留，未伪装为零。服务器保持运行。完整原始SQLite、遥测、校准和失败记录保留在本目录；轻量交付不包含模型权重或图片。\n'
 report.write_text(text)
 entries=[]
 for p in sorted(N.rglob('*')):
  if not p.is_file() or p.name=='artifact_manifest.json' or p.suffix=='.tmp':continue
  entries.append(dict(path=str(p.relative_to(N)),bytes=p.stat().st_size,sha256=streamsha(p)))
 save(N/'artifact_manifest.json',dict(at_ns=time.time_ns(),root=str(N),files=entries,file_count=len(entries),bytes=sum(x['bytes'] for x in entries),scope='Only new component efficiency result directory; no model/data rehash'))
 archive=Path('/tmp/c55_component_efficiency_core.zip');keep=[]
 for p in N.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(N)
  if len(rel.parts)==1 or rel.parts[0]=='code' or p.name in ('smoke_validation.json','environment.json','warmup_baseline.json','calibration.json','performance_warnings.json','resource_release.json','progress.json','per_request.csv','gpu_collector_summary.json','cpu_collector_summary.json','processes_collector_summary.json'):keep.append(p)
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(keep):z.write(p,str(p.relative_to(N)))
 print(json.dumps(dict(status=v['status'],formal_requests=240,manifest_files=len(entries),archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=streamsha(archive),workers_exited=True,GPU_compute_processes=resource['observed_compute_processes']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
