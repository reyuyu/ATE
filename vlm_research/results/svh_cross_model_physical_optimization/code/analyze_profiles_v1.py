from pathlib import Path
import json,csv,gzip,collections,bisect,statistics
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
def savecsv(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (A/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
summary=[];counts=[];windows=[]
for G in [p for p in A.glob('*/*') if p.name in ['profile','refine_profile']]:
 if not (G/'completion.json').exists():continue
 if not json.loads((G/'completion.json').read_text())['status'].startswith('COMPLETE'):continue
 kind=G.parent.name;campaign='refinement_E' if G.name.startswith('refine_') else 'primary'
 for p in sorted((G/'traces').glob('*.json.gz')):
  with gzip.open(p,'rt') as f:trace=json.load(f)
  es=[e for e in trace['traceEvents'] if e.get('ph')=='X']
  syncs=sorted([e for e in es if e['name']=='cudaDeviceSynchronize'],key=lambda e:e['ts'])
  if len(syncs)<3:raise AssertionError(('NO_SYNCHRONIZED_WINDOW',str(p),len(syncs)))
  scopes=sorted([e for e in es if e.get('cat')=='user_annotation' and (e['name']=='MODEL_PREFILL' or e['name'].startswith('MODEL_DECODE_'))],key=lambda e:e['ts'])
  assert len(scopes)==32,(p,len(scopes))
  pre=scopes[0];last=scopes[-1]
  start_sync=max([e for e in syncs if e['ts']+e['dur']<=pre['ts']],key=lambda e:e['ts'])
  end_sync=min([e for e in syncs if e['ts']>=last['ts']+last['dur']],key=lambda e:e['ts'])
  lo=start_sync['ts']+start_sync['dur'];hi=end_sync['ts']+end_sync['dur']
  # Includes context restoration and final sync; excludes H2D, hash, result I/O, post-timer finite/KV audits.
  current=[e for e in es if lo<=e['ts']<hi]
  method,stage=p.name[:-8].rsplit('_',1)
  method=method.replace('FastV_SVH','FastV+SVH').replace('SparseVLM_SVH','SparseVLM+SVH').replace('V2Drop_SVH','V2Drop+SVH').replace('ApET_SVH','ApET+SVH')
  window=dict(kind=kind,campaign=campaign,method=method,stage=stage,trace=str(p.relative_to(A)),start_us=lo,end_us=hi,synchronized_profiled_window_ms=(hi-lo)/1000,MODEL_forward_scopes=len(scopes),cudaDeviceSynchronize_total=len(syncs))
  windows.append(window)
  cpu=[e for e in current if e.get('cat') in ['cpu_op','cuda_runtime','user_annotation']]
  kernels=[e for e in current if e.get('cat')=='kernel']
  runtime=[e for e in current if e.get('cat')=='cuda_runtime']
  # Correlation IDs tie async GPU kernels to their CPU launch phase.
  intervals=sorted([(s['ts'],s['ts']+s['dur'],'Prefill' if s['name']=='MODEL_PREFILL' else 'Decode') for s in scopes])
  starts=[i[0] for i in intervals]
  def phase_at(t):
   i=bisect.bisect_right(starts,t)-1
   return intervals[i][2] if i>=0 and t<=intervals[i][1] else 'Outside_LanguageForward_including_Vision'
  correlations={}
  for e in current:
   if e.get('cat') not in ['cuda_runtime','cuda_driver']:continue
   corr=e.get('args',{}).get('correlation')
   if corr is not None:correlations[corr]=phase_at(e['ts'])
  groups=collections.defaultdict(list)
  for e in current:
   cat=e.get('cat','');name=e['name']
   if cat=='kernel':
    phase=correlations.get(e.get('args',{}).get('correlation'),'Unattributed_GPU')
    groups[phase,'kernel',name].append(e)
   elif cat in ['cpu_op','cuda_runtime']:
    groups[phase_at(e['ts']),cat,name].append(e)
  for (phase,cat,name),xs in groups.items():
   counts.append(dict(kind=kind,campaign=campaign,method=method,stage=stage,phase=phase,category=cat,name=name,count=len(xs),summed_duration_us=sum(x['dur'] for x in xs)))
  def n(name):return sum(e['name']==name for e in cpu)
  def duration(name):return sum(e['dur'] for e in cpu if e['name']==name)/1000
  sync=[e for e in runtime if 'Synchronize' in e['name']]
  row=dict(kind=kind,campaign=campaign,method=method,stage=stage,profile_inputs=1,generated_tokens=32,real_decode_calls=31,kernel_count=len(kernels),kernel_summed_ms=sum(e['dur'] for e in kernels)/1000,
   cuda_sync_api_count=len(sync),cuda_sync_summed_ms=sum(e['dur'] for e in sync)/1000,
   scalar_count=n('aten::item'),local_scalar_dense_count=n('aten::_local_scalar_dense'),
   isin_count=n('aten::isin'),cat_count=n('aten::cat'),index_select_count=n('aten::index_select'),
   copy_count=n('aten::copy_'),zeros_count=n('aten::zeros'),empty_count=n('aten::empty'),
   scalar_cpu_ms=duration('aten::_local_scalar_dense'),
   memcpy_event_count=sum('memcpy' in e.get('cat','').lower() for e in current),
   profile_trace=str(p.relative_to(A)),limitation='Diagnostic single fixed median input; inclusive CPU durations and GPU kernel sum are not disjoint wall-clock decomposition; no profiled latency used in formal summary')
  for phase in ['Prefill','Decode','Outside_LanguageForward_including_Vision','Unattributed_GPU']:
   row[phase+'_kernel_count']=sum(len(xs) for (ph,cat,name),xs in groups.items() if ph==phase and cat=='kernel')
   row[phase+'_sync_api_count']=sum(len(xs) for (ph,cat,name),xs in groups.items() if ph==phase and cat=='cuda_runtime' and 'Synchronize' in name)
  memory=[e for e in trace['traceEvents'] if e.get('name')=='[memory]' and e.get('args',{}).get('Device Type')==1 and lo<=e.get('ts',-1)<hi]
  memory.sort(key=lambda e:e['ts'])
  if memory:
   initial=memory[0]['args']['Total Allocated']-memory[0]['args']['Bytes'];end=memory[-1]['args']['Total Allocated'];peak=max(e['args']['Total Allocated'] for e in memory)
   row.update(GPU_allocation_events=sum(e['args']['Bytes']>0 for e in memory),GPU_free_events=sum(e['args']['Bytes']<0 for e in memory),GPU_total_allocated_MiB=sum(max(0,e['args']['Bytes']) for e in memory)/1048576,GPU_total_freed_MiB=-sum(min(0,e['args']['Bytes']) for e in memory)/1048576,profile_peak_increment_above_start_MiB=(peak-initial)/1048576,profile_peak_excess_above_end_MiB=(peak-end)/1048576,profile_start_allocated_MiB=initial/1048576,profile_end_allocated_MiB=end/1048576)
  summary.append(row)
  del trace,es,current,cpu,kernels,runtime,groups,memory
savecsv('profile_stage_counts.csv',summary);savecsv('profile_operator_counts.csv',counts);savecsv('profile_timer_windows.csv',windows)
print(json.dumps(dict(profile_stages=len(summary),operator_rows=len(counts),window_count=len(windows))))
