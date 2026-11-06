"""Launch or supervise exactly two owned, UUID-pinned workers."""
import argparse
import fcntl
import os
import signal
import subprocess
import time
from .common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.supervise:
        verify()
        safety()
        path = OUT / 'supervisor.log'
        with path.open('a') as log:
            process = subprocess.Popen([os.sys.executable, '-u', '-m',
                'vlm_research.seven_benchmark_quality.launch', '--supervise'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(OUT / 'supervisor_launch.json', dict(pid=process.pid, at=now(), log=str(path)))
        print('SUPERVISOR', process.pid, path, flush=True)
        return
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'STOP_FAILURE.json').exists(), 'Resolve recorded failure explicitly before restart'
    verify()
    safety()
    gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    for uuid in DEVICES:
        line = next(x for x in gpu_text.splitlines() if x.split(',')[0].strip() == uuid)
        fields = [v.strip() for v in line.split(',')]
        assert int(fields[1]) < 512 and int(fields[2]) == 0, line
        assert uuid not in apps, 'An authorized GPU already has a compute process'
    atomic(OUT / 'gpu_launch_preflight.json', dict(at=now(), gpus=gpu_text, compute_apps=apps,
           authorized_devices=DEVICES))
    children = []
    logs = []
    summarized = set()
    try:
        for rank, uuid in enumerate(DEVICES):
            folder = OUT / f'worker{rank}'
            folder.mkdir(exist_ok=True)
            log = (folder / 'run.log').open('a')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            child = subprocess.Popen([os.sys.executable, '-u', '-m',
                'vlm_research.seven_benchmark_quality.worker', '--rank', str(rank)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
        atomic(OUT / 'running.json', dict(status='RUNNING', at=now(), supervisor=os.getpid(),
            pids=[p.pid for p in children], devices=DEVICES))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('An owned worker failed; stopping the paired run')
            if (OUT / 'STOP').exists() or (OUT / 'STOP_FAILURE.json').exists():
                raise RuntimeError('Safety stop requested')
            for bench in ORDER:
                if bench not in summarized and all((OUT / f'worker{r}/{bench}_generation_complete.json').exists()
                                                   for r in range(2)):
                    subprocess.run([os.sys.executable, '-m', 'vlm_research.seven_benchmark_quality.analyze',
                                    '--benchmark', bench], cwd=ROOT, check=True)
                    summarized.add(bench)
            time.sleep(5)
        assert all(p.returncode == 0 for p in children)
        for bench in ORDER:
            if bench not in summarized:
                subprocess.run([os.sys.executable, '-m', 'vlm_research.seven_benchmark_quality.analyze',
                                '--benchmark', bench], cwd=ROOT, check=True)
        atomic(OUT / 'generation_complete.json', dict(status='GENERATED_ALL_SPLITS', at=now()))
    except BaseException as error:
        atomic(OUT / 'supervisor_failure.json', dict(status='STOPPED', error=repr(error), at=now()))
        for p in children:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in children:
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        raise
    finally:
        for log in logs:
            log.close()
        released = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                           '--format=csv,noheader,nounits'], text=True)
        atomic(OUT / 'gpu_supervisor_exit.json', dict(at=now(), compute_apps=released,
               owned_pids=[p.pid for p in children], exit_codes=[p.poll() for p in children]))


if __name__ == '__main__':
    main()
