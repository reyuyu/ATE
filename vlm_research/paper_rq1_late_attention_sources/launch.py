"""Wait for all four immutable-code workers before each central stage gate."""
import os
import subprocess
import time
from .protocol import ROOT,OUT,PYTHON,GPUS
from .run import gate,verify


def main():
    verify()
    for stage in ['smoke','full','ablation']:
        if stage=='ablation' and not (OUT/'target_freeze.json').exists():
            subprocess.run([PYTHON,'-u','-m','paper_rq1_late_attention_sources.run','select'],cwd=ROOT,check=True)
        jobs=[]
        for rank in range(4):
            if (OUT/f'{stage}/shard{rank}/completion.json').exists():continue
            folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
            stream=(folder/f'console_{time.time_ns()}.txt').open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[rank],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
            process=subprocess.Popen([PYTHON,'-u','-m','paper_rq1_late_attention_sources.run',stage,'--rank',str(rank)],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            jobs.append((process,stream,rank))
        failures=[]
        for process,stream,rank in jobs:
            status=process.wait();stream.close()
            if status:failures.append(dict(rank=rank,returncode=status))
        if failures:raise RuntimeError(f'{stage} failed: {failures}; no next stage, no silently skipped image')
        gate(stage);print('Completed',stage,flush=True)
    print('All source collection finished; all GPU workers exited. CPU analysis next.',flush=True)


if __name__=='__main__':main()
