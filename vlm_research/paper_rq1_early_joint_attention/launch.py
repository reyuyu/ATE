"""Use only the three frozen free GPUs; never terminate unrelated GPU processes."""
import os
import subprocess
import sys
import time
from .protocol import OUT,GPUS,PYTHON


def main():
    for stage in sys.argv[1:] or ['smoke','generation']:
        children=[]
        for rank in range(3):
            folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
            log=(folder/'worker.log').open('a',buffering=1)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[rank],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            p=subprocess.Popen([PYTHON,'-m','paper_rq1_early_joint_attention.run',stage,'--rank',str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((p,log));print(stage,'physical GPU',rank+1,'PID',p.pid,flush=True)
        while any(p.poll() is None for p,_ in children):time.sleep(20)
        codes=[p.wait() for p,_ in children]
        for _,log in children:log.close()
        if any(codes):raise SystemExit('Stage failed; checkpoints and failure logs preserved: '+str(codes))
        print(stage,'complete',flush=True)


if __name__=='__main__':main()
