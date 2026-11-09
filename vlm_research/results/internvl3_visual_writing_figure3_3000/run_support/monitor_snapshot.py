import pathlib,json,subprocess
B=pathlib.Path("/home/featurize/ate-runs/internvl3-figure3-3000");D=B/"checkpoints";R=B/"runtime"
c=json.loads((B/"active_runtime.json").read_text())
def alive(pid):
 try:return bool((pathlib.Path("/proc")/str(pid)/"cmdline").read_bytes())
 except FileNotFoundError:return False
files=list(D.glob("*.json"));f=max(files,key=lambda p:p.stat().st_mtime_ns);r=json.loads(f.read_text())
requests=r["index"]*31+len(r["conditions"]);samples=r["index"]+int(r["complete"])
worker=alive(c["worker"]);mirror=alive(c["mirror"]);finalizer=alive(c["finalizer"])
status="RUNNING" if worker else ("CPU_FINALIZING" if requests==93000 else "STOPPED")
if worker and not (R/"smoke_summary.json").exists():status="SMOKE"
log=(B/"finish_resume.log").read_text()
if "ALL_AUTHORIZED_WORK_COMPLETE_REVIEW_PENDING" in log:status="COMPLETE"
elif status=="CPU_FINALIZING" and not finalizer:status="CPU_STOPPED"
if worker and not mirror:status="MIRROR_STOPPED"
compute=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_gpu_memory","--format=csv,noheader"],text=True).strip()
other=[x for x in compute.splitlines() if x.split(",")[0].strip()!=str(c["worker"])]
if other:status="GPU_CONFLICT"
uuid=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
if uuid!="GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e":status="DEVICE_MISMATCH"
print(json.dumps(dict(status=status,requests=requests,samples=samples,target=93000,GPU_UUID=uuid,
 worker_alive=worker,mirror_alive=mirror,finalizer_alive=finalizer,worker_PID=c["worker"],
 gpu_processes=compute,checkpoint_files=len(files),progress_source="direct_SSD_runtime")))
