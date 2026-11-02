"""One-shot batch completion: never changes conditions or restarts failed workers."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .common import OUT,ROOT,read,write,verify,sha256,hardware

BRANCH='codex/gqa3000-visual-writing-necessity-curve'


def running(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def wait_for_workers():
    workers=read(OUT/'worker_launch.json')['workers']
    while True:
        errors=list(OUT.glob('shard*/error.json'))
        if errors:raise RuntimeError('Worker failure retained: '+', '.join(str(p) for p in errors))
        complete=all((OUT/f"shard{r['shard']}/complete.json").exists() for r in workers)
        live=any(running(r['pid']) for r in workers)
        if complete and not live:return workers
        if not live:raise RuntimeError('Workers exited without both completion certificates')
        time.sleep(20)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT.parent,text=True,stderr=subprocess.STDOUT).strip()


def finish():
    lock=(OUT/'finalizer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert git('branch','--show-current')==BRANCH
    workers=wait_for_workers();p=verify()
    state=hardware();apps=state['compute_processes'].splitlines()
    assert not any(str(r['pid']) in [v.strip() for v in row.split(',')] for r in workers for row in apps)
    write(OUT/'resource_release.json',dict(passed=True,own_worker_pids_exited=True,own_GPU_contexts_absent=True,
        workers=workers,hardware=state,epoch=time.time(),other_processes_never_terminated=True))
    for name,info in p['model_files'].items():assert sha256(name)==info['sha256'],name
    images={s['image_path']:s['image_sha256'] for s in read(OUT/'sample_manifest.json')['samples']}
    for path,h in images.items():assert sha256(path)==h,path
    write(OUT/'final_hash_audit.json',dict(passed=True,full_model_file_hashes_match=True,image_files=len(images),
        all_image_file_hashes_match=True,frozen_sources_dependencies_inputs_match=True,epoch=time.time()))
    with (OUT/'analysis_execution.log').open('xb') as stream:
        subprocess.run([sys.executable,'-u','-m','vlm_research.visual_writing_necessity.analyze'],
            cwd=ROOT.parent,stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert read(OUT/'analysis_complete.json')['passed'] and read(OUT/'validation.json')['generations']==102000
    write(OUT/'completion.json',dict(status='COMPLETE',questions=3000,image_clusters=2409,conditions=34,
        free_generations=102000,Gold_NLL_evaluations=102000,full_audit_passed=True,resources_released=True,
        no_extra_conditions=True,code_commit=p['code_commit'],epoch=time.time(),analysis_source_sha256=sha256(Path(__file__).with_name('analyze.py'))))
    relative=str(OUT.relative_to(ROOT.parent))
    for path in OUT.rglob('*'):
        if path.is_file() and 'checkpoints' not in path.parts:
            assert path.stat().st_size<100*1024**2,'GitHub file limit: '+str(path)
    git('add','--','vlm_research/visual_writing_necessity',relative)
    logs=[str(path.relative_to(ROOT.parent)) for path in OUT.rglob('*.log') if 'checkpoints' not in path.parts and path.name!='finalizer_execution.log']
    if logs:git('add','-f','--',*logs)
    git('-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m','Complete GQA3000 visual writing necessity: 102000 generations and Gold NLL')
    commit=git('rev-parse','HEAD')
    push=git('push','-u','origin',BRANCH)
    assert git('ls-remote','--heads','origin',BRANCH).split()[0]==commit
    write(OUT/'delivery_status.json',dict(pushed=True,commit=commit,branch=BRANCH,push_output=push,
        report=f'https://github.com/reyuyu/ATE/blob/{commit}/{relative}/report_zh.md',epoch=time.time()))
    print('COMPLETE AND PUSHED',commit,flush=True)


if __name__=='__main__':
    try:finish()
    except BaseException:
        write(OUT/'finalization_error.json',dict(status='PARTIAL',traceback=traceback.format_exc(),epoch=time.time()))
        raise
