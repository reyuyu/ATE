"""Offline recovery only: never mutate an active worker or completed valid result."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from benchmarks.dataset_registry import digest,freeze_json
from .protocol import OUT,CONDITIONS,PYTHON,ROOT,GPUS
from .run import verify,checkpoint_valid


def repair_incomplete_audit(folder,archive,conditions=CONDITIONS):
    if all((folder/(c+'.json')).is_file() for c in conditions) and not (folder/'audit.json').is_file():
        archive.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(folder),str(archive))
        return True
    return False


def main():
    config=verify();binding=digest(config);locks=[];actions=[];stamp=str(time.time_ns());pending=set()
    try:
        for rank in range(3):
            folder=OUT/f'generation/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
            lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        samples=json.loads((OUT/'sample_manifest.json').read_text())['samples']
        for sample in samples:
            rank=sample['rank'];folder=OUT/f'generation/shard{rank}/checkpoints'/(sample['dataset']+'__'+sample['question_id'])
            for c in CONDITIONS:
                if (folder/(c+'.json')).exists():checkpoint_valid(folder/(c+'.json'),binding,sample,c)
            archive=OUT/f'recovery_attempts/{stamp}/shard{rank}'/folder.name
            if repair_incomplete_audit(folder,archive):actions.append(dict(sample=folder.name,action='Preserve orphaned complete-condition attempt and regenerate image audit honestly',archive=str(archive)))
            complete=all((folder/(c+'.json')).is_file() for c in CONDITIONS) and (folder/'audit.json').is_file()
            if not complete:pending.add(rank)
        for rank in range(3):
            p=OUT/f'generation/shard{rank}/completion.json'
            if not p.exists():pending.add(rank)
            elif rank in pending:
                archive=OUT/f'recovery_attempts/{stamp}/shard{rank}/previous_completion.json';archive.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(p),str(archive))
        freeze_json(OUT/f'recovery_attempts/{stamp}/recovery.json',dict(actions=actions,worker_sources_unchanged=True))
    finally:
        for lock in locks:lock.close()
    children=[]
    for rank in sorted(pending):
        log=(OUT/f'generation/shard{rank}/worker.log').open('a',buffering=1)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[rank],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        child=subprocess.Popen([PYTHON,'-m','paper_rq1_decode_layerwise_quality.run','generation','--rank',str(rank)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((child,log))
    codes=[child.wait() for child,_ in children]
    for _,log in children:log.close()
    if any(codes):raise SystemExit('Recovery incomplete; all attempts preserved:'+str(codes))
    print('Recovery complete; rerun ranks',sorted(pending),flush=True)


if __name__=='__main__':main()
