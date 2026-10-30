import json
import os
import subprocess
import sys
import time
from .protocol import ROOT,OUT,RAW,GPUS
from .run import verify,gate,preflight
from benchmarks.dataset_registry import freeze_json


def stage(name):
    for rank,uuid in enumerate(GPUS):
        previous=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=uuid
        try:preflight(rank)
        finally:
            if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=previous
    children=[];files=[]
    for rank,uuid in enumerate(GPUS):
        path=OUT/f'{name}/shard{rank}/completion.json'
        if path.exists() and json.loads(path.read_text())['status']=='complete':continue
        path.parent.mkdir(parents=True,exist_ok=True);log=(path.parent/f'console_{time.time_ns()}.txt').open('x');files.append(log)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        child=subprocess.Popen([sys.executable,'-u','-m','bos_cheap.run',name,'--rank',str(rank)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);children.append((rank,child))
    codes={r:p.wait() for r,p in children}
    for f in files:f.close()
    if any(codes.values()):
        freeze_json(OUT/f'failures/coordinator_{time.time_ns()}.json',dict(stage=name,codes=codes,status='partial'));raise SystemExit(1)
    gate(name);print('Completed',name,flush=True)


def cpu(module):
    code=subprocess.call([sys.executable,'-u','-m',module],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLCONFIGDIR=str(RAW/'matplotlib')))
    if code:raise SystemExit(code)


def main():
    verify();stage('smoke');stage('old')
    if not (OUT/'predictor_freeze.json').exists():cpu('bos_cheap.predict')
    stage('new');cpu('bos_cheap.analyze')


if __name__=='__main__':main()
