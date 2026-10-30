"""Three physical GPU queues, four unchanged logical shards, strict stage gates."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import subprocess
import time
from .protocol import ROOT,OUT,PYTHON,GPUS
from .run import gate,verify
from .scheduling import queues


def run_queue(stage, physical_gpu, ranks):
    for rank in ranks:
        if (OUT/f'{stage}/shard{rank}/completion.json').exists():
            continue
        folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[physical_gpu],OMP_NUM_THREADS='4',
                 MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        with (folder/f'console_{time.time_ns()}.txt').open('x') as stream:
            process=subprocess.run([PYTHON,'-u','-m','paper_rq1_late_attention_sources.run',stage,'--rank',str(rank)],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode:
            return dict(rank=rank,physical_gpu=physical_gpu,returncode=process.returncode)
    return None


def main():
    lock=(OUT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify()
    for stage in ['smoke','full','ablation']:
        if stage=='ablation' and not (OUT/'target_freeze.json').exists():
            subprocess.run([PYTHON,'-u','-m','paper_rq1_late_attention_sources.run','select'],cwd=ROOT,check=True)
        with ThreadPoolExecutor(max_workers=3) as executor:
            jobs=[executor.submit(run_queue,stage,gpu,ranks) for gpu,ranks in queues().items()]
            failures=[result for job in jobs if (result:=job.result()) is not None]
        if failures:raise RuntimeError(f'{stage} failed: {failures}; no next stage, no silently skipped image')
        gate(stage);print('Completed',stage,flush=True)
    print('All source collection finished; all GPU workers exited. CPU analysis next.',flush=True)


if __name__=='__main__':main()
