"""One explicit predecessor chain, not an idle-GPU polling scheduler."""
import argparse
from datetime import datetime
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

from .common import *


def prior_state():
    result = subprocess.run(['systemctl', '--user', 'show', SERVICE,
        '--property=ActiveState', '--property=SubState', '--property=ExecMainStatus',
        '--property=LoadState'], capture_output=True, text=True)
    state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode:
        # Collected transient units can disappear after a successful exit.
        # Durable shard completion plus the one-shot idle check remains required.
        assert state.get('LoadState') == 'not-found', result.stderr
        return dict(ActiveState='inactive', SubState='collected', ExecMainStatus='0')
    return state


def predecessor_complete():
    p = read(PRIOR / 'protocol.json')
    expected = p['execution_schedule']['ready_questions']
    keys = set()
    for rank in (0, 1):
        w = PRIOR / f'worker{rank}'
        assert not (w / 'STOP.json').exists(), f'Predecessor worker failed: {w}'
        done = read(w / 'completion.json')
        assert done['status'] == 'COMPLETE_SHARD' and done['completed'] == done['expected']
        db = sqlite3.connect((w / 'results.sqlite').as_uri() + '?mode=ro', uri=True)
        for (key,) in db.execute('SELECT key FROM results'):
            assert key not in keys
            keys.add(key)
        db.close()
    assert len(keys) == expected == 32882
    assert not any(k.startswith('MME:') for k in keys)
    return len(keys)


def execute(stage, uuids):
    command = [sys.executable, '-m', 'vlm_research.sparsevlm_svh.run', 'driver',
               '--stage', stage, '--gpus', *uuids]
    subprocess.run(command, cwd=ROOT, check=True)


def local_commit(label):
    files = []
    for pattern in ('*.json', '*.csv', '*.md', 'gpu_smoke/*.json', 'worker*/*.json', 'cpu_tests.log'):
        files += [p for p in OUT.glob(pattern) if p.is_file() and p.stat().st_size < 20 * 1024**2]
    files += [p for p in (OUT / 'predecessor_fastv').glob('*') if p.is_file()]
    if files:
        subprocess.run(['git', 'add', '-f', '--', *[str(p.relative_to(ROOT)) for p in files]], cwd=ROOT, check=True)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode
    if changed:
        subprocess.run(['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@openai.com',
                        'commit', '-m', label], cwd=ROOT, check=True)


def main(uuids, recover_completed_predecessor=False):
    verify()
    assert len(uuids) == 2
    initial = prior_state()
    assert initial['ActiveState'] in ('active', 'inactive') or (
        recover_completed_predecessor and initial['ActiveState'] == 'failed'), initial
    receipt = dict(status='WAITING_PREDECESSOR', service=SERVICE, gpus=uuids,
        created_at=datetime.now().astimezone().isoformat(),
        order=['FastV five benchmarks', 'FastV authorized MME', 'SparseVLM GPU smoke',
               'SparseVLM six benchmarks'], no_gpu_polling=True, no_push=True,
        recover_completed_predecessor=recover_completed_predecessor, predecessor_initial_state=initial)
    atomic(OUT / 'handoff_status.json', receipt)
    try:
        # Watch this exact task only. Never search repeatedly for free hardware.
        while True:
            state = prior_state()
            if state['ActiveState'] not in ('active', 'activating', 'deactivating'):
                break
            time.sleep(20)
        if state['ExecMainStatus'] != '0' or state['ActiveState'] == 'failed':
            assert recover_completed_predecessor and state['ExecMainStatus'] == '1', state
            assert predecessor_complete() == 32882
            from .analyze import analyze
            analyze('fastv-ready')
            recovered = PRIOR / 'recovered_five_benchmarks'
            result = read(recovered / 'completion.json')
            assert result['status'] == 'VERIFIED_FIVE_AWAITING_MME' and result['completed'] == 32882
            receipt['predecessor_recovery'] = dict(original_service_failure_preserved=state,
                independent_rescore=True, audit_sha256=sha(recovered / 'checkpoint_binding_audit.json'),
                completion_sha256=sha(recovered / 'completion.json'))
        receipt['predecessor_completed'] = predecessor_complete()
        assert shutil.disk_usage(OUT).free > 3 * 1024**3, 'Insufficient /data reserve'
        receipt['hardware_before_mme'] = check_idle(uuids)
        receipt['status'] = 'FASTV_MME_RUNNING'
        atomic(OUT / 'handoff_status.json', receipt)
        execute('mme', uuids)
        completion = read(PRIOR / 'complete_with_mme_authorized/completion.json')
        assert completion['completed'] == 35256
        evidence = OUT / 'predecessor_fastv'
        evidence.mkdir(exist_ok=True)
        for name in ('report_zh.md', 'score_summary.csv', 'task_details.csv', 'completion.json',
                     'data_acceptance.json', 'artifact_sha256.json'):
            shutil.copy2(PRIOR / 'complete_with_mme_authorized' / name, evidence / name)
        receipt['fastv_completed'] = 35256
        local_commit('Record completed FastV/SVH six-benchmark predecessor and authorized MME')
        receipt['hardware_before_sparse'] = check_idle(uuids)
        receipt['status'] = 'SPARSE_GPU_GATE'
        atomic(OUT / 'handoff_status.json', receipt)
        subprocess.run([sys.executable, '-m', 'vlm_research.sparsevlm_svh.run', 'smoke',
                        '--stage', 'sparse', '--gpus', uuids[0]], cwd=ROOT, check=True)
        receipt['status'] = 'SPARSE_FULL_RUNNING'
        atomic(OUT / 'handoff_status.json', receipt)
        execute('sparse', uuids)
        receipt['status'] = 'COMPLETE_GENERATION'
        receipt['completed_at'] = datetime.now().astimezone().isoformat()
        atomic(OUT / 'handoff_status.json', receipt)
        local_commit('Record fixed SparseVLM-prefix/SVH six-benchmark results and audits')
    except BaseException:
        receipt.update(status='STOPPED_WITH_EVIDENCE', traceback=traceback.format_exc(),
                       stopped_at=datetime.now().astimezone().isoformat())
        atomic(OUT / 'handoff_status.json', receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs=2, default=list(BASE_GPUS))
    parser.add_argument('--recover-completed-predecessor', action='store_true')
    args = parser.parse_args()
    main(args.gpus, args.recover_completed_predecessor)
