"""Separate, low-rate nvidia-smi telemetry process. Never changes GPU settings."""
import sys,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
out=Path(sys.argv[1]);owner=int(sys.argv[2]);stop=out/'monitor.stop'
fields=['uuid','clocks.current.sm','clocks.current.memory','power.draw','power.limit','temperature.gpu','utilization.gpu','utilization.memory','pstate','clocks_event_reasons.sw_power_cap','clocks_event_reasons.sw_thermal_slowdown']
with (out/'gpu_telemetry.jsonl').open('a',buffering=1) as f:
 while not stop.exists():
  record={'at':datetime.now(timezone.utc).isoformat(),'unix_time':time.time()}
  for key,args in [('gpu',['--query-gpu='+','.join(fields),'--format=csv,noheader,nounits']),('processes',['--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])]:
   try:record[key]=subprocess.check_output(['nvidia-smi',*args],text=True,timeout=8).strip()
   except Exception as e:record[key+'_error']=repr(e)
  record['gpu_fields']=fields
  others=[]
  for line in record.get('processes','').splitlines():
   try:
    pid=int(line.split(',')[0].strip())
    if pid!=owner:others.append(line)
   except ValueError:pass
  if others:(out/'gpu_conflict.json').write_text(json.dumps({'at':record['at'],'other_processes':others}))
  f.write(json.dumps(record)+'\n')
  for _ in range(20):
   if stop.exists():break
   time.sleep(.1)
