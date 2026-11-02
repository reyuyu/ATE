"""Wait for Qwen, run InternVL on the same idle GPUs, then aggregate."""
import os
import subprocess
import time

from .common import ROOT, read, write, command

OUT = ROOT/'vlm_research/results/qwen_internvl_sparse_visual_3000'
PYTHON = '/data/lk/深度学习/.envs/ate-internvl3/bin/python'
PLOT_PYTHON = '/home/likai/anaconda3/envs/AFAC/bin/python'


def main():
    while not all((OUT/'qwen'/f'gpu{g}/complete.json').exists() for g in (0, 2, 3)):
        assert not any((OUT/'qwen'/f'gpu{g}/error.json').exists() for g in (0, 2, 3)), 'Qwen failed; stop'
        time.sleep(10)
    # Complete records precede Python teardown; never allocate over a live task.
    while any(command('nvidia-smi', '-i', str(g), '--query-compute-apps=pid', '--format=csv,noheader') for g in (0, 2, 3)):
        time.sleep(5)
    processes, handles = [], []
    for gpu in (0, 2, 3):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
        log = (OUT/f'internvl_gpu{gpu}.log').open('w')
        handles.append(log)
        p = subprocess.Popen([PYTHON, '-u', '-m', 'vlm_research.visual_pathways.sparse_accuracy',
            'worker', '--model', 'internvl', '--gpu', str(gpu)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(p)
    print('INTERNVL THREE WORKERS STARTED', flush=True)
    for model in ('qwen', 'internvl'):
        if model == 'internvl':
            statuses = [p.wait() for p in processes]
            for h in handles:
                h.close()
            assert statuses == [0, 0, 0], statuses
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        subprocess.run([PYTHON, '-m', 'vlm_research.visual_pathways.sparse_accuracy_analysis', model],
            cwd=ROOT, env=env, check=True)
    subprocess.run([PLOT_PYTHON, '-m', 'vlm_research.visual_pathways.sparse_accuracy_analysis', 'report'],
        cwd=ROOT, env=env, check=True)
    write(OUT/'pipeline_complete.json', dict(passed=True, time=time.time(), models=['qwen', 'internvl'],
        questions_per_model=3000, generations_per_model=18000))


if __name__ == '__main__':
    main()
