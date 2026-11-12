"""Separate-process, read-only telemetry; ctypes NVML at 5Hz, procfs at 1Hz."""
from pathlib import Path
import ctypes as C,sys,time,os,json,csv,subprocess,traceback
L=Path(sys.argv[1]);PID=int(sys.argv[2]);UUID=sys.argv[3]
class Util(C.Structure):_fields_=[('gpu',C.c_uint),('memory',C.c_uint)]
class Mem(C.Structure):_fields_=[('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]
nv=C.CDLL('libnvidia-ml.so.1');assert nv.nvmlInit_v2()==0
h=C.c_void_p();assert nv.nvmlDeviceGetHandleByUUID(UUID.encode(),C.byref(h))==0
def val(name,typ,*args):
 v=typ();code=getattr(nv,name)(h,*args,C.byref(v))
 return v.value if code==0 else 'NVML_ERROR_'+str(code)
def text(p):
 try:return Path(p).read_text()
 except (OSError,PermissionError):return None
def sched():
 base=Path('/proc')/str(PID)/'task'/str(PID);raw=text(base/'stat');status=text(base/'status')
 d=dict(stat=raw,schedstat=text(base/'schedstat'),wchan=text(base/'wchan'))
 if raw:
  v=raw[raw.rfind(')')+2:].split();d.update(state=v[0],utime_ticks=int(v[11]),stime_ticks=int(v[12]),last_cpu=int(v[36]),starttime_ticks=int(v[19]))
 if status:
  for line in status.splitlines():
   if line.startswith(('Cpus_allowed_list:','voluntary_ctxt_switches:','nonvoluntary_ctxt_switches:','Threads:')):
    k,v=line.split(':',1);d[k]=v.strip()
 return d
def cg():
 raw=text('/proc/'+str(PID)+'/cgroup');d={}
 if not raw:return d
 for line in raw.splitlines():
  if line.startswith('0::'):
   p=Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
   for q in [p,*p.parents]:
    if not str(q).startswith('/sys/fs/cgroup'):break
    for n in ['cpu.stat','cpu.max','cpuset.cpus.effective']:
     v=text(q/n)
     if v is not None:d[str(q/n)]=v
 return d
start=time.time();cpu_start=time.process_time();gpu_times=[];loop_ms=[];misses=0;errors=[];others_seen=[]
gf=(L/'gpu_telemetry.csv').open('x',newline='');cf=(L/'cpu_telemetry.csv').open('x',newline='')
gfields=['unix_time_ns','monotonic_ns','query_duration_ms','UUID','pstate','SM_MHz','memory_MHz','power_W','power_limit_W','temperature_C','GPU_util_pct','memory_util_pct','memory_used_bytes','throttle_mask','driver_version','error']
writer=csv.DictWriter(gf,fieldnames=gfields);writer.writeheader()
cwriter=csv.DictWriter(cf,fieldnames=['unix_time_ns','monotonic_ns','collector_duration_ms','loadavg','CPU_pressure','main_thread','CPU_MHz','cgroup','proc_stat_cpu','mem_available_bytes','top_processes']);cwriter.writeheader()
driver=C.create_string_buffer(96);nv.nvmlSystemGetDriverVersion(driver,96);driver=driver.value.decode()
next_cpu=next_proc=0.;deadline=time.monotonic();cycle=0
try:
 while not (L/'monitor_stop.flag').exists():
  loop0=time.perf_counter();t=time.monotonic();g=dict(unix_time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),UUID=UUID,driver_version=driver,error='')
  try:
   u=Util();m=Mem();uc=nv.nvmlDeviceGetUtilizationRates(h,C.byref(u));mc=nv.nvmlDeviceGetMemoryInfo(h,C.byref(m))
   g.update(pstate=val('nvmlDeviceGetPerformanceState',C.c_uint),SM_MHz=val('nvmlDeviceGetClockInfo',C.c_uint,C.c_uint(1)),memory_MHz=val('nvmlDeviceGetClockInfo',C.c_uint,C.c_uint(2)),power_W=val('nvmlDeviceGetPowerUsage',C.c_uint),power_limit_W=val('nvmlDeviceGetPowerManagementLimit',C.c_uint),temperature_C=val('nvmlDeviceGetTemperature',C.c_uint,C.c_uint(0)),GPU_util_pct=u.gpu if uc==0 else '',memory_util_pct=u.memory if uc==0 else '',memory_used_bytes=m.used if mc==0 else '',throttle_mask=val('nvmlDeviceGetCurrentClocksThrottleReasons',C.c_ulonglong))
   for k in ['power_W','power_limit_W']:
    if isinstance(g[k],int):g[k]/=1000
   if isinstance(g['temperature_C'],int) and g['temperature_C']>=85:(L/'STOP.flag').write_text('GPU temperature >=85C')
  except Exception as ex:g['error']=repr(ex);errors.append(g['error'])
  g['query_duration_ms']=(time.perf_counter()-loop0)*1000;writer.writerow(g);gpu_times.append(g['unix_time_ns']/1e9)
  if t>=next_cpu:
   c0=time.perf_counter();mem=text('/proc/meminfo') or '';avail=next((int(l.split()[1])*1024 for l in mem.splitlines() if l.startswith('MemAvailable:')),0)
   cpu=dict(unix_time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),loadavg=text('/proc/loadavg'),CPU_pressure=text('/proc/pressure/cpu'),main_thread=json.dumps(sched()),CPU_MHz=json.dumps([float(l.split(':')[1]) for l in (text('/proc/cpuinfo') or '').splitlines() if l.startswith('cpu MHz')]),cgroup=json.dumps(cg()),proc_stat_cpu='\n'.join(l for l in (text('/proc/stat') or '').splitlines() if l.startswith('cpu')),mem_available_bytes=avail,top_processes='')
   if cycle%25==0:
    pr=subprocess.run(['ps','-eo','pid,comm,pcpu,stat','--sort=-pcpu'],capture_output=True,text=True,timeout=3);cpu['top_processes']='\n'.join(pr.stdout.splitlines()[:10])
   if avail and avail<2**30:(L/'STOP.flag').write_text('MemAvailable <1GiB')
   cpu['collector_duration_ms']=(time.perf_counter()-c0)*1000;cwriter.writerow(cpu);next_cpu=t+1
  if t>=next_proc:
   pr=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
   if pr.returncode!=0:raise RuntimeError('GPU process query failed: '+pr.stderr)
   other=[l for l in pr.stdout.splitlines() if l.strip() and l.split(',')[0].strip()!=str(PID)]
   if other:
    others_seen.append(dict(at_ns=time.time_ns(),processes=other));(L/'STOP.flag').write_text(json.dumps(others_seen[-1]))
   next_proc=t+2
  if cycle%5==0:gf.flush();cf.flush()
  cycle+=1;loop_ms.append((time.perf_counter()-loop0)*1000);deadline+=.2
  delay=deadline-time.monotonic()
  if delay<0:misses+=1;deadline=time.monotonic()
  else:time.sleep(delay)
except BaseException as ex:
 errors.append(traceback.format_exc());(L/'STOP.flag').write_text('Telemetry failed: '+repr(ex))
finally:
 gf.flush();cf.flush();gf.close();cf.close();nv.nvmlShutdown()
 summary=dict(wall_s=time.time()-start,CPU_s=time.process_time()-cpu_start,cycles=cycle,missed_200ms_deadlines=misses,max_loop_ms=max(loop_ms) if loop_ms else None,mean_loop_ms=sum(loop_ms)/len(loop_ms) if loop_ms else None,GPU_sample_max_gap_s=max([b-a for a,b in zip(gpu_times,gpu_times[1:])],default=0),errors=errors,other_GPU_process_events=others_seen,driver=driver,nsys_not_used=True)
 (L/'telemetry_collector_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
