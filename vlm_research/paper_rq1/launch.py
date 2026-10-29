"""One coordinator; four frozen-code GPU subprocesses, no other task interruption."""
import json
import os
import subprocess
import sys
import time

from .inventory import OUTPUT, ROOT
from .protocol import GPUS
from .run import preflight, verify, gate, freeze_windows


def main():
    verify()
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        for rank, uuid in enumerate(GPUS):
            os.environ['CUDA_VISIBLE_DEVICES'] = uuid
            preflight(rank)
    finally:
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous
    for stage in ('smoke', 'discovery', 'confirmation'):
        if stage == 'confirmation':
            freeze_windows()
        processes, logs = [], []
        for rank, uuid in enumerate(GPUS):
            done = OUTPUT/f'{stage}/shard{rank}/completion.json'
            if done.exists() and json.loads(done.read_text())['status'] == 'complete':
                continue
            folder = OUTPUT/f'{stage}/shard{rank}'
            folder.mkdir(parents=True, exist_ok=True)
            stamp = time.time_ns()
            log = (folder/f'console_{stamp}.txt').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       MPLCONFIGDIR='/data/lk/ATE-resources/matplotlib')
            command = [sys.executable, '-u', '-m', 'paper_rq1.run', stage, '--rank', str(rank)]
            processes.append((rank, subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
        codes = {rank: p.wait() for rank, p in processes}
        for log in logs:
            log.close()
        if any(codes.values()):
            (OUTPUT/'partial_status.json').write_text(json.dumps(dict(status='partial', failed_stage=stage,
                return_codes=codes, next_action='Inspect preserved console logs; do not bypass failed correctness/data gates.'), indent=2)+'\n')
            raise SystemExit(1)
        gate(stage)
        print('All four completed', stage, flush=True)
    with (OUTPUT/'analysis_console.txt').open('a') as log:
        code = subprocess.call([sys.executable, '-u', '-m', 'paper_rq1.analyze'], cwd=ROOT,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
