"""Task-local idle-GPU relay; never changes the frozen inference implementation."""
import argparse
import csv
import fcntl
import io
import signal
import shutil
import time
import traceback
from collections import Counter

from .common import *

RELAY = OUT / 'relay'
PYTHON = {
    'A': RES / 'author-env/bin/python',
    'B': Path('/data/lk/深度学习/.envs/ate-vlm-research/bin/python'),
}
WORKER = 'vlm_research.information_horizon_alignment.worker'


def query_gpus():
    def query(fields, kind):
        output = subprocess.check_output(
            ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
            text=True, timeout=15)
        return list(csv.reader(io.StringIO(output), skipinitialspace=True))

    gpus = {}
    for index, uuid, memory, total, utilization in query(
            'index,uuid,memory.used,memory.total,utilization.gpu', 'gpu'):
        gpus[uuid] = dict(index=int(index), uuid=uuid, memory_mib=int(memory),
                          total_mib=int(total), utilization=int(utilization), pids=[])
    for uuid, pid in query('gpu_uuid,pid', 'compute-apps'):
        assert uuid in gpus, ('Unknown compute device', uuid)
        gpus[uuid]['pids'].append(int(pid))
    assert all(gpus[u]['index'] == r for r, u in enumerate(UUIDS)), 'GPU identity changed'
    return gpus


def idle(gpu, policy):
    return (not gpu['pids'] and gpu['memory_mib'] <= policy['idle_max_memory_mib']
            and gpu['utilization'] <= policy['idle_max_utilization_percent'])


def smoke_passed(stage, rank):
    path = OUT / f'smoke/{stage}/rank{rank}/completion.json'
    return path.exists() and read(path).get('status') == 'PASSED'


def completed(stage, rank, expected, protocol):
    path = OUT / f'workers/{stage}/rank{rank}/completion.json'
    if not path.exists():
        return False
    value = read(path)
    assert value['status'] == 'COMPLETE' and value['records'] == expected, str(path)
    path = path.parent / 'results.sqlite'
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        assert db.execute('SELECT COUNT(*) FROM results').fetchone()[0] == expected
        assert db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()[0] == digest(protocol)
    finally:
        db.close()
    return True


def plan(stage, done, smokes):
    if all(done):
        return []
    if not all(smokes):
        return [(rank, True) for rank, passed in enumerate(smokes) if not passed]
    return [(rank, False) for rank, finished in enumerate(done) if not finished]


def existing_workers():
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            if WORKER.encode() in args and (path.parent / 'cwd').resolve() == ROOT:
                found.append(int(path.parent.name))
        except (OSError, PermissionError):
            continue
    return found


def archive_attempt(stage, rank):
    stamp = now().replace(':', '-').replace('+', '_')
    folder = RELAY / 'attempts' / f'{stage}-rank{rank}-{stamp}'
    folder.mkdir(parents=True, exist_ok=False)
    old = OUT / f'workers/{stage}/rank{rank}'
    for name in ('startup', 'failure', 'release', 'progress', 'completion'):
        path = old / f'{name}.json'
        if path.exists():
            shutil.copy2(path, folder / f'previous_{name}.json')
    smoke = OUT / f'smoke/{stage}/rank{rank}'
    if smoke.exists():
        shutil.copytree(smoke, folder / 'previous_smoke')
    return folder


def launch(stage, rank, smoke_only, admission):
    folder = archive_attempt(stage, rank)
    environment = dict(os.environ)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        environment.pop(key, None)
    environment.update(CUDA_VISIBLE_DEVICES=UUIDS[rank],
        HF_HUB_CACHE=str(RES / 'hf-hub'), HUGGINGFACE_HUB_CACHE=str(RES / 'hf-hub'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONUNBUFFERED='1')
    command = [str(PYTHON[stage]), '-m', WORKER, '--stage', stage, '--rank', str(rank)]
    if smoke_only:
        command.append('--smoke-only')
    atomic(folder / 'admission.json', dict(at=now(), stage=stage, rank=rank,
        smoke_only=smoke_only, gpu=admission, command=command,
        source_sha256=digest(source_hashes()), supervisor_sha256=sha(__file__)))
    with (folder / 'worker.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    atomic(folder / 'process.json', dict(pid=child.pid, at=now()))
    print('START', stage, rank, 'SMOKE' if smoke_only else 'FORMAL', child.pid, flush=True)
    return dict(process=child, folder=folder, stage=stage, rank=rank, smoke_only=smoke_only)


def stop_own(jobs):
    """Only signal children launched by this controller, never external GPU jobs."""
    for job in jobs.values():
        child = job['process']
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
    deadline = time.monotonic() + 60
    for job in jobs.values():
        child = job['process']
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def postprocess():
    command = [str(PYTHON['B']), '-m', 'vlm_research.information_horizon_alignment.analyze']
    with (RELAY / 'analysis.log').open('ab') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true', help='CPU-only preflight; no worker launch')
    args = parser.parse_args()
    RELAY.mkdir(parents=True, exist_ok=True)
    with (RELAY / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = read(OUT / 'idle_relay_policy.json')
        assert policy['gpu_uuids'] == UUIDS
        assert policy['freeze_source_sha256'] == digest(source_hashes())
        assert not existing_workers(), 'Existing worker detected; refusing duplicate launch'
        protocols = {stage: verify(stage) for stage in CONDITIONS}
        counts = Counter(s['shard'] for s in manifest())
        expected = {stage: [counts[r] * len(CONDITIONS[stage]) for r in range(4)] for stage in CONDITIONS}
        jobs, stable = {}, Counter()
        stage = 'A'
        signature = None
        self_sha = sha(__file__)

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'Controller received signal {signum}')

        signal.signal(signal.SIGTERM, interrupt)
        try:
            while True:
                assert sha(__file__) == self_sha, 'Controller source changed while active'
                assert policy == read(OUT / 'idle_relay_policy.json'), 'Relay policy changed while active'
                for label in CONDITIONS:
                    assert not (OUT / f'STOP_{label}.json').exists(), f'Frozen worker STOP_{label} present'
                if (RELAY / 'STOP').exists():
                    raise KeyboardInterrupt('Task-local relay STOP requested')
                if shutil.disk_usage('/data').free < policy['minimum_free_disk_gib'] * 1024**3:
                    raise RuntimeError('Relay disk safety floor reached; no automatic retry')
                for rank, job in list(jobs.items()):
                    code = job['process'].poll()
                    if code is None:
                        continue
                    atomic(job['folder'] / 'exit.json', dict(at=now(), returncode=code))
                    assert code == 0, f"Worker {stage}/{rank} failed; see {job['folder']}"
                    if job['smoke_only']:
                        assert smoke_passed(stage, rank)
                    else:
                        assert completed(stage, rank, expected[stage][rank], protocols[stage])
                    del jobs[rank]
                    stable[rank] = 0

                done = [completed(stage, r, expected[stage][r], protocols[stage]) for r in range(4)]
                smokes = [smoke_passed(stage, r) for r in range(4)]
                if all(done) and not jobs:
                    if args.inspect:
                        print(json.dumps(dict(stage=stage, done=done, smokes=smokes), indent=2))
                        return
                    postprocess()
                    if stage == 'A':
                        completion = read(OUT / 'completion.json')
                        assert completion['stage_A_completed'] == completion['stage_A_expected']
                        stage, stable = 'B', Counter()
                        print('STAGE_A_COMPLETE_BEGIN_B_SMOKE_GATES', now(), flush=True)
                        continue
                    atomic(RELAY / 'status.json', dict(at=now(), status='COLLECTION_COMPLETE',
                        automatic_worker_launches_finished=True, official_final_report_review_required=True))
                    print('ALL_COLLECTION_COMPLETE_REVIEW_FINAL_METRICS', now(), flush=True)
                    return

                gpus = query_gpus()
                for rank, job in jobs.items():
                    foreign = set(gpus[UUIDS[rank]]['pids']) - {job['process'].pid}
                    assert not foreign, ('External task appeared on active GPU; stopping only our work', rank, sorted(foreign))
                candidates = plan(stage, done, smokes)
                for rank in range(4):
                    stable[rank] = stable[rank] + 1 if idle(gpus[UUIDS[rank]], policy) else 0
                state = dict(at=now(), controller_pid=os.getpid(), stage=stage,
                    status='RUNNING' if jobs else 'WAITING_FOR_IDLE_GPU',
                    completed_shards=done, smoke_passed=smokes,
                    active=[dict(rank=r, pid=j['process'].pid, smoke_only=j['smoke_only']) for r, j in jobs.items()],
                    candidate_ranks=[r for r, _ in candidates], gpu_snapshot=gpus,
                    poll_seconds=policy['poll_seconds'], frozen_source_sha256=digest(source_hashes()),
                    supervisor_sha256=self_sha, policy_sha256=digest(policy))
                if args.inspect:
                    print(json.dumps(state, indent=2))
                    return
                atomic(RELAY / 'status.json', state)
                current = (stage, tuple(done), tuple(smokes), tuple(sorted(jobs)),
                           tuple((u, tuple(gpus[u]['pids'])) for u in UUIDS))
                if current != signature:
                    print(state['status'], stage, 'active', sorted(jobs), 'complete', done, now(), flush=True)
                    signature = current

                for rank, smoke_only in candidates:
                    if rank in jobs or stable[rank] < policy['consecutive_idle_polls']:
                        continue
                    # Recheck directly before launch to narrow the admission race.
                    admission = query_gpus()[UUIDS[rank]]
                    if not idle(admission, policy):
                        stable[rank] = 0
                        continue
                    verify(stage)
                    jobs[rank] = launch(stage, rank, smoke_only, admission)
                    stable[rank] = 0
                time.sleep(policy['poll_seconds'])
        except BaseException:
            atomic(RELAY / 'failure.json', dict(at=now(), status='STOPPED', stage=stage,
                traceback=traceback.format_exc(), retry_automatically=False))
            stop_own(jobs)
            atomic(RELAY / 'status.json', dict(at=now(), status='STOPPED', stage=stage,
                external_processes_untouched=True, own_children_reaped=True))
            raise


if __name__ == '__main__':
    main()
