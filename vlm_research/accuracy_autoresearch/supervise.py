"""Bounded, task-local cleanup. Never signals or terminates another process."""
import os
import time

from .common import *


def main():
    execution=read(OUT/'execution_freeze.json'); frozen=read(OUT/'final_candidate_freeze.json')
    targets=[OUT/f'worker{i}/jobs/{name}.json' for i in range(len(execution['devices']))
        for name in (frozen['final_quality_job'],frozen['final_perf_job'])]
    previous=0
    print('Task-local supervisor: wait for FINAL quality/timing; then request clean Worker exit.',flush=True)
    while not all(p.exists() for p in targets):
        if any((OUT/f'worker{i}/STOP.json').exists() for i in range(len(execution['devices']))):
            raise RuntimeError('A task worker stopped; preserve diagnostics and do not resubmit')
        if time.time()>=deadline('stop_new_gpu_work_by'):
            atomic(OUT/'SHUTDOWN.json',dict(at=now(),reason='GPU deadline; retain partial results',partial=True))
            raise RuntimeError('GPU budget reached before complete FINAL')
        if time.time()-previous>=1800:
            atomic(OUT/'progress/FINAL_supervisor.json',dict(at=now(),phase='Frozen FINAL',
                finished_job_markers=sum(p.exists() for p in targets),expected_job_markers=len(targets),
                remaining_delivery_seconds=deadline()-time.time(),candidate_freeze_sha256=digest(frozen),
                no_final_scores_read=True))
            print('FINAL status',now(),sum(p.exists() for p in targets),'/',len(targets),'job markers',flush=True)
            previous=time.time()
        time.sleep(5)
    for p in targets: assert read(p)['status']=='COMPLETE'
    atomic(OUT/'SHUTDOWN.json',dict(at=now(),reason='All frozen FINAL quality and paired performance completed',
        sources=source_hashes(),candidate_freeze_sha256=digest(frozen),no_extra_GPU_work=True))
    print('All FINAL GPU work done; clean shutdown requested',now(),flush=True)
    start=time.time()
    while not all((OUT/f'worker{i}/released.json').exists() for i in range(len(execution['devices']))):
        assert time.time()-start<90,'Worker cleanup did not finish'
        time.sleep(1)
    owned=[read(OUT/f'worker{i}/released.json')['pid'] for i in range(len(execution['devices']))]
    while True:
        state=hardware()
        if not any(f', {pid},' in state['processes'] for pid in owned): break
        assert time.time()-start<90,'Own CUDA contexts still present'
        time.sleep(1)
    atomic(OUT/'gpu_release_audit.json',dict(at=now(),owned_worker_pids=owned,devices=execution['devices'],
        final_state=state,own_CUDA_processes_absent=True,no_external_signals_or_termination=True,
        no_extra_GPU_work=True,task_clock=read(ROOT/'task_deadline.json'),
        status='GPU_COMPLETE_PENDING_CPU_RESULT_AUDIT'))
    print('GPU2/3 own CUDA processes released',owned,now(),flush=True)


if __name__=='__main__': main()
