"""Own two fixed workers; finalize and push only after complete verified output."""
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .common import ROOT, OUT, read, seal, atomic, verify, command


def main():
    p = verify()
    assert command('git', 'branch', '--show-current') == 'codex/qwen3vl-gqa3000-visual-writing-necessity'
    children, streams = [], []
    launch = time.time_ns()
    try:
        for shard, gpu in enumerate(p['gpus']):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            folder = OUT / f'shard{shard}'
            folder.mkdir(parents=True, exist_ok=True)
            stream = (folder / f'worker_{launch}.log').open('a')
            streams.append(stream)
            child = subprocess.Popen([sys.executable, '-m', 'vlm_research.qwen3vl_visual_writing_necessity.run',
                'worker', '--shard', str(shard)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            children.append(child)
        seal(OUT / f'launch_{launch}.json', dict(supervisor_pid=os.getpid(), started_unix=time.time(),
            worker_pids=[c.pid for c in children], gpu_uuids=[g['uuid'] for g in p['gpus']], protocol_sha256=__import__('hashlib').sha256((OUT/'protocol.json').read_bytes()).hexdigest()))
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None, 0) for c in children):
                raise RuntimeError(f'Worker failure: {[c.poll() for c in children]}; stopping only this experiment')
            time.sleep(3)
        assert all(c.returncode == 0 for c in children)
        assert all(read(OUT / f'shard{i}/complete.json')['generations'] == 57000 for i in range(2))
        analysis_python = '/data/lk/深度学习/.envs/ate-vlm-research/bin/python'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
        with (OUT / 'analysis.log').open('a') as log:
            subprocess.run([analysis_python, '-m', 'vlm_research.qwen3vl_visual_writing_necessity.analyze'],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert read(OUT / 'completion.json')['generations'] == 114000
        assert read(OUT / 'completion.json')['status'] == 'COMPLETE'
        processes = command('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader')
        own_pids = [str(c.pid) for c in children]
        assert all(pid not in [line.split(',')[0].strip() for line in processes.splitlines()] for pid in own_pids)
        seal(OUT / 'gpu_release_audit.json', dict(own_worker_pids=own_pids, own_workers_exited=True,
            own_compute_processes_absent=True, snapshot=processes, timestamp=time.time()))
        subprocess.run(['git', 'add', 'vlm_research/qwen3vl_visual_writing_necessity', str(OUT.relative_to(ROOT))], cwd=ROOT, check=True)
        subprocess.run(['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@localhost', 'commit', '-m',
            'Complete Qwen GQA3000 visual-writing necessity: 114000 audited generations and paper figures'], cwd=ROOT, check=True)
        commit = command('git', 'rev-parse', 'HEAD')
        subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/codex/qwen3vl-gqa3000-visual-writing-necessity'],
            cwd=ROOT, env=dict(os.environ, GIT_TERMINAL_PROMPT='0'), check=True)
        atomic(OUT / 'delivery_status.json', dict(status='PUSHED', commit=commit, timestamp=time.time()))
        print('DELIVERED', commit, flush=True)
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        atomic(OUT / f'supervisor_failure_{launch}.json', dict(status='PARTIAL', timestamp=time.time(), traceback=traceback.format_exc()))
        raise
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
