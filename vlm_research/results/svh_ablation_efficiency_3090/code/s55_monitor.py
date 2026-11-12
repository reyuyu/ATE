"""Independent read-only GPU/CPU/process collectors. No model or torch import."""
from pathlib import Path
import os,sys,time,json,subprocess,ctypes as C,traceback
L=Path(sys.argv[1]);PID=int(sys.argv[2]);UUID=sys.argv[3];mode=sys.argv[4]
def text(p):
 try:return Path(p).read_text()
 except OSError:return None
def flag(v):(L/'STOP.flag').write_text(json.dumps(v))
if mode=='gpu':
 class Util(C.Structure):_fields_=[('gpu',C.c_uint),('memory',C.c_uint)]
 class Mem(C.Structure):_fields_=[('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]
 nv=C.CDLL('libnvidia-ml.so.1');assert nv.nvmlInit_v2()==0;h=C.c_void_p();assert nv.nvmlDeviceGetHandleByUUID(UUID.encode(),C.byref(h))==0
 def val(n,t,*args):
  v=t();rc=getattr(nv,n)(h,*args,C.byref(v));return v.value if rc==0 else 'NVML_ERROR_'+str(rc)
period={'gpu':.2,'cpu':1,'processes':2}[mode];start=time.monotonic();cycles=0;worst=0;errors=[];previous=None
with (L/(mode+'_telemetry.jsonl')).open('x') as out:
 while not (L/'monitor.stop').exists():
  if not Path('/proc/'+str(PID)).exists():break
  t=time.monotonic();ns=time.time_ns();gap=None if previous is None else (ns-previous)/1e9;previous=ns
  try:
   if mode=='gpu':
    u=Util();mem=Mem();assert nv.nvmlDeviceGetUtilizationRates(h,C.byref(u))==0;assert nv.nvmlDeviceGetMemoryInfo(h,C.byref(mem))==0
    r=dict(at_ns=ns,gap_s=gap,SM_MHz=val('nvmlDeviceGetClockInfo',C.c_uint,C.c_uint(1)),memory_MHz=val('nvmlDeviceGetClockInfo',C.c_uint,C.c_uint(2)),pstate=val('nvmlDeviceGetPerformanceState',C.c_uint),temperature_C=val('nvmlDeviceGetTemperature',C.c_uint,C.c_uint(0)),power_mW=val('nvmlDeviceGetPowerUsage',C.c_uint),power_limit_mW=val('nvmlDeviceGetEnforcedPowerLimit',C.c_uint),throttle_mask=val('nvmlDeviceGetCurrentClocksThrottleReasons',C.c_ulonglong),GPU_util=u.gpu,memory_used=mem.used)
    if isinstance(r['temperature_C'],int) and r['temperature_C']>=85:flag(dict(reason='GPU temperature>=85C',sample=r))
   elif mode=='cpu':
    cg={};raw=text('/proc/'+str(PID)+'/cgroup') or ''
    for line in raw.splitlines():
     if line.startswith('0::'):
      p=Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
      for q in [p,*p.parents]:
       if not str(q).startswith('/sys/fs/cgroup'):break
       for n in ['cpu.stat','cpu.max','cpuset.cpus.effective']:
        v=text(q/n)
        if v is not None:cg[str(q/n)]=v
    r=dict(at_ns=ns,gap_s=gap,loadavg=text('/proc/loadavg'),pressure=text('/proc/pressure/cpu'),schedstat=text('/proc/'+str(PID)+'/task/'+str(PID)+'/schedstat'),proc_stat=text('/proc/'+str(PID)+'/stat'),system_stat=text('/proc/stat'),cgroup=cg,meminfo=text('/proc/meminfo'))
    for line in (r['meminfo'] or '').splitlines():
     if line.startswith('MemAvailable:') and int(line.split()[1])<1048576:flag(dict(reason='MemAvailable<1GiB',at_ns=ns))
   else:
    v=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,timeout=3).strip();other=[s for s in v.splitlines() if s and int(s.split(',')[0])!=PID]
    r=dict(at_ns=ns,gap_s=gap,processes=v,other_processes=other)
    if other:flag(dict(reason='Other GPU compute process',sample=r))
   r['query_ms']=(time.monotonic()-t)*1000;out.write(json.dumps(r,separators=(',',':'))+'\n');out.flush();cycles+=1
  except Exception as ex:
   errors.append(dict(at_ns=ns,error=repr(ex)));out.write(json.dumps(dict(at_ns=ns,error=repr(ex)))+'\n');out.flush()
   if len(errors)>=3:flag(dict(reason='Telemetry collector errors',mode=mode,errors=errors));break
  elapsed=time.monotonic()-t;worst=max(worst,elapsed);time.sleep(max(0,period-elapsed))
(L/(mode+'_collector_summary.json')).write_text(json.dumps(dict(cycles=cycles,elapsed_s=time.monotonic()-start,process_CPU_s=time.process_time(),maximum_loop_s=worst,errors=errors),indent=2)+'\n')
