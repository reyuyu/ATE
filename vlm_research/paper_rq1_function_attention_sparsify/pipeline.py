"""Local finite pipeline; resource conflicts fail closed, never wait for GPU0."""
import os
import subprocess
import time
from .protocol import OUT,ROOT,PYTHON,GPUS
from paper_rq1_single_layer_gqa6.run import atomic_json


def command(module,args=(),env=None):
    print('START',module,*args,flush=True)
    subprocess.run([PYTHON,'-m',module,*args],env=env,check=True)


def parallel(stage):
    children=[]
    for rank in range(3):
        folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
        log=(folder/'worker.log').open('a',buffering=1)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[rank+1],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        process=subprocess.Popen([PYTHON,'-m','paper_rq1_function_attention_sparsify.quality_run',stage,'--rank',str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((process,log));print(stage,'GPU',rank+1,'PID',process.pid,flush=True)
    while any(p.poll() is None for p,_ in children):
        time.sleep(30);print(stage,'worker status',[p.poll() for p,_ in children],flush=True)
    codes=[p.wait() for p,_ in children]
    for _,log in children:log.close()
    if any(codes):raise RuntimeError(stage+' failed; no later stage started: '+str(codes))


def main():
    started=time.time()
    try:
        while not all((OUT/f'stage_a/shard{i}/completion.json').exists() for i in range(3)):
            if list((OUT/'stage_a').glob('shard*/failure_*.json')):raise RuntimeError('StageA collection failed; retained partial checkpoints')
            if time.time()-started>3600:raise RuntimeError('StageA did not finish within this pipeline wait budget')
            time.sleep(30)
        for module in ['analysis','train']:
            atomic_json(OUT/'pipeline_status.json',dict(stage=module,status='running',started=started))
            command('paper_rq1_function_attention_sparsify.'+module)
        command('paper_rq1_function_attention_sparsify.quality_run',['freeze'])
        for stage in ['quality_smoke','quality','risk']:
            atomic_json(OUT/'pipeline_status.json',dict(stage=stage,status='running',started=started));parallel(stage)
        command('paper_rq1_function_attention_sparsify.summarize',['quality'])
        atomic_json(OUT/'pipeline_status.json',dict(stage='latency',status='running',started=started))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[1],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        command('paper_rq1_function_attention_sparsify.quality_run',['latency','--rank','0'],env)
        command('paper_rq1_function_attention_sparsify.summarize',['latency'])
        atomic_json(OUT/'pipeline_status.json',dict(stage='all collection/scoring/figures',status='complete',started=started,seconds=time.time()-started))
    except BaseException as error:
        atomic_json(OUT/'pipeline_status.json',dict(status='partial',error=repr(error),started=started));raise


if __name__=='__main__':main()
