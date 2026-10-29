"""Four single-GPU processes; Smoke -> main -> exploratory generation -> CPU analysis."""
import json
import os
import shutil
import subprocess
import sys
import time

from benchmarks.dataset_registry import freeze_json
from .protocol import GPUS, OUTPUT, RESOURCES, ROOT
from .run import gate, preflight, verify


def main():
    verify()
    RESOURCES.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(RESOURCES).free < 30*1024**3:
        raise ValueError('Reserve at least30GiB for full-vocabulary replay artifacts')
    previous=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        for rank,uuid in enumerate(GPUS):
            os.environ['CUDA_VISIBLE_DEVICES']=uuid
            preflight(rank)
    finally:
        if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=previous
    for stage in ('smoke','first_token','accuracy'):
        processes=[];logs=[]
        for rank,uuid in enumerate(GPUS):
            complete=OUTPUT/f'{stage}/shard{rank}/completion.json'
            if complete.exists() and json.loads(complete.read_text())['status']=='complete':continue
            folder=OUTPUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
            log=(folder/f'console_{time.time_ns()}.txt').open('x');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                MPLCONFIGDIR=str(RESOURCES/'matplotlib'))
            command=[sys.executable,'-u','-m','paper_rq1_single_layer_gqa6.run',stage,'--rank',str(rank)]
            processes.append((rank,subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)))
        codes={rank:p.wait() for rank,p in processes}
        for log in logs:log.close()
        if any(codes.values()):
            freeze_json(OUTPUT/'failures'/f'coordinator_{time.time_ns()}.json',dict(status='partial',stage=stage,
                return_codes=codes,next_action='Inspect preserved failures; resume the same frozen code, never skip data or relax tolerance.'))
            raise SystemExit(1)
        gate(stage)
        print('All four completed',stage,flush=True)
    with (OUTPUT/'analysis_console.txt').open('a') as log:
        code=subprocess.call([sys.executable,'-u','-m','paper_rq1_single_layer_gqa6.analyze'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
    raise SystemExit(code)


if __name__=='__main__':main()
