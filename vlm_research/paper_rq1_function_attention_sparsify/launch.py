import os
import subprocess
import sys
import time
from .protocol import OUT,PYTHON,GPUS


def main():
    stages=sys.argv[1:] or ['smoke','stage_a']
    for stage in stages:
        children=[]
        for rank in range(3):
            folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
            log=(folder/'worker.log').open('a',buffering=1)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[rank+1],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            process=subprocess.Popen([PYTHON,'-m','paper_rq1_function_attention_sparsify.run',stage,'--rank',str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((process,log));print(stage,'GPU',rank+1,'PID',process.pid,flush=True)
        while any(p.poll() is None for p,_ in children):
            time.sleep(30)
            print(stage,'worker_status',[p.poll() for p,_ in children],flush=True)
        codes=[p.wait() for p,_ in children]
        for _,log in children:log.close()
        if any(codes):raise SystemExit('Stage failed; no subsequent GPU stage launched: '+str(codes))
        print(stage,'complete',flush=True)


if __name__=='__main__':main()
