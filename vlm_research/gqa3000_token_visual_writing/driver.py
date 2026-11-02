"""Foreground dual-worker supervisor: no main generation before both smoke gates."""
import argparse,os,subprocess,time,traceback
import numpy as np
import torch
from .common import *

def cross_gate(p,smokes,attempt):
    assert len(smokes)==2 and all(s['passed'] and s['binding']==digest(p) for s in smokes)
    for s in smokes:assert sha256(s['cross_gpu_raw']['path'])==s['cross_gpu_raw']['sha256']
    checks={}
    with np.load(smokes[0]['cross_gpu_raw']['path'],allow_pickle=False) as a,np.load(smokes[1]['cross_gpu_raw']['path'],allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        for name in a.files:
            assert torch.equal(torch.from_numpy(a[name]),torch.from_numpy(b[name])),name
            checks[name]=dict(shape=list(a[name].shape),dtype=str(a[name].dtype),strict_equal=True)
    result=dict(passed=True,binding=digest(p),attempt=attempt,checks=checks,smokes=smokes,all_main_workers_waited=True)
    atomic_json(OUT/f'attempts/{attempt}/both_gpu_smoke_gate.json',result)
    print('BOTH GPU SMOKE GATE PASS',attempt,flush=True)

def run(attempt):
    torch.set_num_threads(4);p=verify();assert_no_stop();procs={};logs={};completed={};smokes={}
    (OUT/f'attempts/{attempt}').mkdir(parents=True,exist_ok=True)
    assert not (OUT/f'attempts/{attempt}/driver_started.json').exists(),'Use a fresh attempt name for audited resume'
    atomic_json(OUT/f'attempts/{attempt}/driver_started.json',dict(pid=os.getpid(),binding=digest(p),hardware=hardware()))
    # Check both cards before either model is allocated. Each child rechecks its own UUID.
    for rank,gpu in enumerate(p['gpus']):
        done=OUT/f'shard{rank}/completion.json'
        if done.exists():
            completed[rank]=read(done);assert completed[rank]['passed'] and completed[rank]['binding']==digest(p)
            smokes[rank]=read(OUT/f"shard{rank}/attempts/{completed[rank]['attempt']}/smoke.json")
            continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'])
        before=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
        try:preflight(gpu,rank,attempt+'_driver')
        finally:
            if before is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=before
    for rank,gpu in enumerate(p['gpus']):
        if rank in completed:continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        log=(OUT/f'shard{rank}/attempts/{attempt}/stdout.txt');log.parent.mkdir(parents=True,exist_ok=True)
        logs[rank]=log.open('x',buffering=1)
        procs[rank]=subprocess.Popen([os.sys.executable,'-u','-m','vlm_research.gqa3000_token_visual_writing.worker','--rank',str(rank),'--attempt',attempt],cwd=ROOT.parent,env=env,stdout=logs[rank],stderr=subprocess.STDOUT)
        print('WORKER START',rank,'pid',procs[rank].pid,'UUID',gpu['uuid'],flush=True)
    offsets={rank:0 for rank in procs};gate=False
    try:
        while procs:
            for rank,proc in procs.items():
                logpath=OUT/f'shard{rank}/attempts/{attempt}/stdout.txt'
                with logpath.open() as f:f.seek(offsets[rank]);chunk=f.read();offsets[rank]=f.tell()
                if chunk:print(f'[GPU worker {rank}] '+chunk.rstrip(),flush=True)
                smoke=OUT/f'shard{rank}/attempts/{attempt}/smoke.json'
                if smoke.exists() and rank not in smokes:smokes[rank]=read(smoke)
                code=proc.poll()
                if code is not None:
                    assert code==0,f'Worker {rank} exited {code}'
                    complete=OUT/f'shard{rank}/completion.json';assert complete.exists();completed[rank]=read(complete)
            if not gate and len(smokes)==2:cross_gate(p,[smokes[0],smokes[1]],attempt);gate=True
            assert_no_stop()
            if all(proc.poll() is not None for proc in procs.values()):break
            time.sleep(1)
        if not procs:cross_gate(p,[smokes[0],smokes[1]],attempt)
        assert len(completed)==2 and sum(c['canonical_evaluations'] for c in completed.values())==p['unique_evaluations']
        atomic_json(OUT/f'attempts/{attempt}/driver_completion.json',dict(passed=True,binding=digest(p),workers=completed,hardware=hardware()))
        print('ALL MAIN GENERATION COMPLETE',p['unique_evaluations'],flush=True)
    except BaseException:
        failure=dict(status='FAILED',driver_pid=os.getpid(),traceback=traceback.format_exc(),hardware=hardware())
        if not (OUT/'STOP.json').exists():atomic_json(OUT/'STOP.json',failure)
        # Workers check STOP before every real forward and exit themselves; never kill unrelated jobs.
        for proc in procs.values():
            if proc.poll() is None:
                try:proc.wait(timeout=60)
                except subprocess.TimeoutExpired:print('Worker has not exited yet; preserve process for diagnosis',proc.pid,flush=True)
        raise
    finally:
        for f in logs.values():f.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='initial');a=parser.parse_args()
    assert a.attempt.replace('_','').replace('-','').isalnum();run(a.attempt)
