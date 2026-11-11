"""Bounded controller; never touches any process other than its own children."""
import subprocess
import time

from prompt_common import *


def main():
    check_sources()
    processes=[]
    logs=[]
    begin=time.monotonic()
    def wait_for(names):
        while not all((OUT/name).exists() for name in names):
            for child in processes:
                assert child.poll() is None or child.returncode==0, ('Worker failed',child.pid,child.returncode)
            assert time.monotonic()-begin<3*3600,'Own experiment deadline exceeded'
            time.sleep(3)
    try:
        for shard,uuid in enumerate(UUIDS):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
            log=(OUT/f'worker{shard}.log').open('a')
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable,str(ROOT/'run_prompts.py'),'--shard',str(shard)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        write(OUT/'own_processes.json',dict(controller=os.getpid(),workers=[p.pid for p in processes]))
        wait_for(['smoke0.json','smoke1.json'])
        write(OUT/'discovery_gate.json',dict(status='BOTH_WRAPPER_SMOKES_PASS',historical_strict_physical_failures='FAIL_RETAINED'))
        print('BOTH_SMOKES_PASS; DISCOVERY_STARTED',flush=True)
        wait_for(['discovery_complete0.json','discovery_complete1.json'])
        subprocess.run([sys.executable,str(ROOT/'analyze_prompts.py'),'select'],check=True)
        print('VALIDATION_STARTED',flush=True)
        wait_for(['validation_complete0.json','validation_complete1.json'])
        write(OUT/'performance_gate.json',dict(quality_complete=True,not_a_full_benchmark_rerun=True))
        print('QUALITY_COMPLETE; PAIRED_TIMING_STARTED',flush=True)
        wait_for(['worker_complete0.json','worker_complete1.json'])
        for child in processes:
            assert child.wait(timeout=60)==0
        subprocess.run([sys.executable,str(ROOT/'analyze_prompts.py'),'finish'],check=True)
    except BaseException as exc:
        write(OUT/'STOP.json',dict(error=repr(exc)))
        for child in processes:
            if child.poll() is None:
                child.terminate()
        for child in processes:
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        for log in logs:
            log.close()
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        write(OUT/'gpu_release_audit.json',dict(own_workers_terminated=all(p.poll() is not None for p in processes),
            child_pids=[p.pid for p in processes],gpu_uuids=UUIDS,processes=procs,
            own_gpu_processes_remain=any(str(p.pid) in procs for p in processes),wall_s=time.monotonic()-begin))


if __name__=='__main__':
    main()
