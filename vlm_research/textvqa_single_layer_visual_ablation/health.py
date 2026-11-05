"""CPU-only recoverable coverage log; run periodically during the active task."""
import pathlib,json,time,collections,datetime
O=pathlib.Path('/data/lk/ATE-worktrees/textvqa-single-layer-visual-ablation/vlm_research/results/textvqa_single_layer_visual_ablation')
s=json.loads((O/'shards.json').read_text());pilot=set(s['pilot120']);counts=collections.Counter();pc=collections.Counter();owned=collections.Counter();pilot_rows=0
for q in (O/'checkpoints').iterdir():
 if not q.is_dir():continue
 names={p.stem for p in q.glob('*.json')};owned[s['by_question'][q.name]]+=len(names);counts.update(names)
 if q.name in pilot:
  pc.update(names)
  if all(f'{fam}-L{k}' in names for fam in ['All-Text','Ponly','Donly','Both'] for k in range(32)):pilot_rows+=1
stages=[name for name in ['pilot120','primary300','secondary300','optional300'] if all((O/f'{name}_worker{i}.json').exists() for i in [0,1])]
now=time.time();snapshot=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_checkpoints=sum(counts.values()),by_worker=dict(owned),pilot_questions_all128_complete=pilot_rows,completed_stages=stages,conditions_complete300=sum(v==300 for v in counts.values()),STOP=(O/'STOP.json').exists())
lastpath=O/'progress_coverage_latest.json';last=json.loads(lastpath.read_text()) if lastpath.exists() else None
if last is None or now-last['epoch']>=1800 or stages!=last['completed_stages']:
 protocol=json.loads((O/'protocol.json').read_text())
 v=dict(**snapshot,epoch=now,condition_question_counts=dict(counts),condition_pilot_question_counts=dict(pc),completed_pilot_layers={f:[k for k in range(32) if pc[f'{f}-L{k}']==120] for f in ['All-Text','Ponly','Donly','Both']},completed_full300_conditions=[c for c,n in counts.items() if n==300],remaining_wall_seconds=protocol['deadline_epoch']-now,remaining_GPU_seconds=protocol['gpu_deadline_epoch']-now,interval_matrix_checkpoints_per_second=None if last is None else (snapshot['new_checkpoints']-last['new_checkpoints'])/(now-last['epoch']),note='effective matrix checkpoint throughput, not pure kernel throughput; no scores used to schedule conditions')
 tmp=lastpath.with_suffix('.json.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True)+'\n');tmp.replace(lastpath)
 with (O/'progress_coverage.jsonl').open('a') as f:f.write(json.dumps(v,ensure_ascii=False,sort_keys=True)+'\n')
completion_path=O/'completion.json';completion=json.loads(completion_path.read_text())
if not completion.get('validated',False):
 completion.update(status='GPU_COMPLETE_PENDING_CPU_VALIDATION' if all((O/f'worker_complete{i}.json').exists() for i in [0,1]) else 'RUNNING',complete=False,new_checkpoint_count=snapshot['new_checkpoints'],completed_stages=stages,last_progress_UTC=snapshot['UTC'],dual_GPU_smoke_passed=all((O/f'gpu_smoke_worker{i}.json').exists() for i in [0,1]))
 tmp=completion_path.with_suffix('.json.tmp');tmp.write_text(json.dumps(completion,ensure_ascii=False,sort_keys=True)+'\n');tmp.replace(completion_path)
print(json.dumps(snapshot))
for i in [0,1]:
 p=O/f'worker_complete{i}.json'
 if p.exists():print(p.read_text())
if (O/'STOP.json').exists():print((O/'STOP.json').read_text())
