"""Single-owner supervisor for the two explicitly authorized consecutive tasks."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback

from .common import OUT, ROOT, GPUS, check, write


def main():
    check()
    lock = (OUT / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(OUT / 'pipeline_owner.json', dict(pid=os.getpid(), started=time.time(), uuid=GPUS))
    jobs, logs = [], []
    def stop(signum, frame):
        (OUT / 'STOP_USER').touch()
        for job in jobs:
            if job.poll() is None:
                job.terminate()
        raise KeyboardInterrupt('Supervisor interrupted; checkpoints retained')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for shard, uuid in enumerate(GPUS):
            log = (OUT / f'worker{shard}.log').open('a', buffering=1)
            logs.append(log)
            job = subprocess.Popen([sys.executable, '-u', '-m', 'vlm_research.qwen3vl_overnight.run', 'worker', '--shard', str(shard)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid), stdout=log, stderr=subprocess.STDOUT)
            jobs.append(job)
        while any(job.poll() is None for job in jobs):
            write(OUT / 'supervisor_progress.json', dict(stage='NATIVE_SVH_SIXBENCH', time=time.time(),
                children=[dict(pid=j.pid, returncode=j.poll()) for j in jobs]))
            time.sleep(30)
        codes = [j.wait() for j in jobs]
        write(OUT / 'worker_exit_codes.json', codes)
        subprocess.run([sys.executable, '-m', 'vlm_research.qwen3vl_overnight.report'], cwd=ROOT, check=True)
        if any(codes) or (OUT / 'STOP_STRUCTURE.json').exists():
            raise RuntimeError('Worker structural/resource failure; see checkpoint and report, no invalid continuation')
        # This module has a separate immutable protocol and never edits the first-stage executor.
        subprocess.run([sys.executable, '-u', '-m', 'vlm_research.qwen3vl_svhr_search.run', 'pipeline'], cwd=ROOT, check=True)
        write(OUT / 'pipeline_completion.json', dict(status='COMPLETE', time=time.time()))
    except BaseException as exc:
        write(OUT / 'pipeline_failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
        for job in jobs:
            job.wait()
        for stream in logs:
            stream.close()
        from ..qwen3vl_svh_preflight48.run import resource
        released = resource()
        released['authorized_gpu_processes_remaining'] = [u for u in GPUS if u in released['processes']]
        write(OUT / 'supervisor_gpu_release.json', released)
        lock.close()


if __name__ == '__main__':
    main()
