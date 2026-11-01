"""Sequential subprocesses guarantee Qwen is unloaded before InternVL starts."""
import os
import subprocess
import time

from .common import ROOT, OUT, write, read


def main():
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    started = time.time()
    stages = []
    for name, mode, executable in [
        ('qwen', 'run', '/data/lk/深度学习/.envs/ate-qwen3vl/bin/python'),
        ('internvl', 'smoke', '/data/lk/深度学习/.envs/ate-internvl3/bin/python'),
        ('internvl', 'run', '/data/lk/深度学习/.envs/ate-internvl3/bin/python'),
    ]:
        begin = time.time()
        log = OUT / f'{name}_{mode}.log'
        print('START', name, mode, log, flush=True)
        with log.open('a', encoding='utf-8') as handle:
            command = [executable, '-u', '-m', 'vlm_research.three_vlm_sensitivity.run', name, mode]
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        stage = dict(model=name, mode=mode, returncode=process.returncode, seconds=time.time()-begin, log=str(log))
        stages.append(stage)
        print('END', stage, flush=True)
        time.sleep(2)
        remaining = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid,gpu_uuid',
                                              '--format=csv,noheader'], text=True).strip()
        write(OUT / f'{name}_{mode}_release.json', dict(process_exited=True, remaining_compute_processes=remaining,
                                                     released=not bool(remaining), timestamp=time.time()))
        write(OUT / 'execution_summary.json', dict(status='running' if process.returncode == 0 else 'partial',
            stages=stages, seconds=time.time()-started, sole_physical_gpu=1))
        if process.returncode:
            raise SystemExit(process.returncode)
        if remaining:
            raise RuntimeError('GPU1 occupied after subprocess exit; do not launch next stage')
    release = dict(released=True, model_processes_exited=True, physical_gpu=1,
                   snapshot=subprocess.check_output(['nvidia-smi'], text=True), timestamp=time.time())
    write(OUT / 'gpu1_release.json', release)
    write(OUT / 'execution_summary.json', dict(status='gpu_completed_analysis_pending', stages=stages,
        seconds=time.time()-started, sole_physical_gpu=1))


if __name__ == '__main__':
    main()
