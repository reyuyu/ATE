"""CPU-only delivery after this exact driver exits; never starts inference."""
import datetime
import json
import os
from pathlib import Path
import select
import sqlite3
import subprocess
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/data/lk/ATE-worktrees/llava-svh-r-middle16-3090-sevenbench')
sys.path.insert(0, str(ROOT))
from vlm_research.svh_middle16.common import OUT, atomic, read, sha, rows

BRANCH = 'codex/llava-svh-r-middle16-3090-sevenbench'


def git(*args):
    return subprocess.check_output(['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@openai.com',
                                    *args], cwd=ROOT, text=True, timeout=180)


def main(pid):
    # Older Python/kernel builds may lack pidfd; also guard against PID reuse.
    if hasattr(os, 'pidfd_open'):
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            fd = None
        if fd is not None:
            try:
                select.select([fd], [], [])
            finally:
                os.close(fd)
    else:
        stat_path = Path('/proc') / str(pid) / 'stat'
        def identity():
            try:
                fields = stat_path.read_text().rsplit(')', 1)[1].split()
                return fields[19] if fields[0] != 'Z' else None
            except FileNotFoundError:
                return None
        started = identity()
        while started is not None and identity() == started:
            time.sleep(10)
    receipt = dict(driver_pid=pid, at=datetime.datetime.now().astimezone().isoformat(),
                   new_gpu_work_started=False, branch=BRANCH)
    try:
        from vlm_research.svh_middle16.analyze import main as analyze
        analyze()
        from vlm_research.svh_middle16.run import score
        samples = {s['key']: s for s in rows(OUT / 'sample_manifest.jsonl.gz')}
        checked = 0
        for path in OUT.glob('worker*/results.sqlite'):
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
                for key, payload in db.execute('SELECT key,payload FROM results'):
                    record = json.loads(payload)
                    assert score(samples[key], record['generation']['text']) == record['scoring'], key
                    checked += 1
        atomic(OUT / 'rescoring_audit.json', dict(status='PASS', raw_outputs_rescored=checked,
               method='Frozen scorer rerun from raw output; no model or additional generation'))
        failures = {p.parent.name: read(p) for p in OUT.glob('worker*/STOP.json')}
        releases = {p.parent.name: read(p) for p in OUT.glob('worker*/release.json')}
        atomic(OUT / 'failures_and_releases.json', dict(failures=failures, releases=releases))
        files = sorted(p for p in OUT.rglob('*') if p.is_file() and not p.name.endswith(('.tmp', '-wal', '-shm')))
        atomic(OUT / 'output_hashes.json', dict(files=[dict(path=str(p), bytes=p.stat().st_size,
               sha256=sha(p)) for p in files if p.name not in ['output_hashes.json', 'publication_receipt.json', 'delivery.log']]))
        assert git('branch', '--show-current').strip() == BRANCH
        assert not git('diff', '--name-only', '--', 'vlm_research/svh_middle16')
        git('add', '--', 'vlm_research/results/svh_r_middle16_sevenbench_3090')
        assert all(p.startswith('vlm_research/results/svh_r_middle16_sevenbench_3090/')
                   for p in git('diff', '--cached', '--name-only').splitlines())
        staged = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode
        if staged == 1:
            git('commit', '-m', 'Publish Middle16 verified quality outputs and explicit data blockers')
        elif staged != 0:
            raise RuntimeError('Cannot inspect staged delivery')
        head = git('rev-parse', 'HEAD').strip()
        pushed = git('push', '-u', 'origin', BRANCH)
        remote = git('ls-remote', 'origin', 'refs/heads/' + BRANCH).split()[0]
        assert remote == head
        receipt.update(status='PUSHED', commit=head, remote_commit=remote,
                       completion=read(OUT / 'completion.json')['status'], push_output=pushed)
    except BaseException:
        receipt.update(status='DELIVERY_FAILED', traceback=traceback.format_exc())
        raise
    finally:
        atomic(OUT / 'publication_receipt.json', receipt)


if __name__ == '__main__':
    main(int(sys.argv[1]))
