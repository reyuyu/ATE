import json
import os
import subprocess
import sys
import time

from benchmarks.dataset_registry import freeze_json
from .protocol import GPUS, OUTPUT, ROOT, RESOURCES
from .run import gate, preflight, verify


def main():
    verify();previous=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        for rank,uuid in enumerate(GPUS):
            os.environ['CUDA_VISIBLE_DEVICES']=uuid;preflight(rank)
    finally:
        if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=previous
    for stage in ['smoke','collect']:
        children=[];logs=[]
        for rank,uuid in enumerate(GPUS):
            done=OUTPUT/f'{stage}/shard{rank}/completion.json'
            if done.exists() and json.loads(done.read_text())['status']=='complete':continue
            folder=OUTPUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
            log=(folder/f'console_{time.time_ns()}.txt').open('x');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                MPLCONFIGDIR=str(RESOURCES/'matplotlib'))
            child=subprocess.Popen([sys.executable,'-u','-m','gqa6_update_decomposition.run',stage,'--rank',str(rank)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((rank,child))
        codes={rank:p.wait() for rank,p in children}
        for log in logs:log.close()
        if any(codes.values()):
            freeze_json(OUTPUT/'failures'/f'coordinator_{time.time_ns()}.json',dict(stage=stage,codes=codes,status='partial'))
            raise SystemExit(1)
        gate(stage);print('All four completed',stage,flush=True)
    raise SystemExit(subprocess.call([sys.executable,'-u','-m','gqa6_update_decomposition.analyze'],cwd=ROOT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')))


if __name__=='__main__':main()
