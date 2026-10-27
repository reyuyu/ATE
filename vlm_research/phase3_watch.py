"""Local cron watchdog: wait for one GPU, run Phase 3 once, publish, stop.

Standard library only: checking occupied GPUs never imports torch or loads weights.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parent
REPOSITORY = PROJECT.parent
PYTHON = Path('/home/zsy/.conda/envs/vlm_research/bin/python')
DEFAULT_STATE = Path('/home/zsy/.local/state/ate-phase3-watch')
MARKER = '# ATE_PHASE3_GPU_WATCH'
TERMINAL = {'completed', 'failed', 'sync_failed', 'disabled'}
CUTOFFS = [0, 8, 12, 16, 20, 24, 28, 32]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def cron_update(state_dir, install):
    result = subprocess.run(['crontab', '-l'], text=True, capture_output=True)
    if result.returncode != 0 and 'no crontab' not in result.stderr.lower():
        raise RuntimeError(f'Cannot read crontab: {result.stderr}')
    lines = [line for line in result.stdout.splitlines() if not line.endswith(MARKER)]
    if install:
        command = shlex.join(['/usr/bin/python3', str(Path(__file__).resolve()), '--tick', '--state-dir', str(state_dir)])
        lines.append(f'*/15 * * * * {command} >/dev/null 2>&1 {MARKER}')
    updated = '\n'.join(lines) + ('\n' if lines else '')
    if updated != result.stdout:
        subprocess.run(['crontab', '-'], input=updated, text=True, check=True)


def read_gpus():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], text=True, capture_output=True,
                             timeout=15, check=True)
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, name, free, utilization = [v.strip() for v in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, name=name,
                         free_mib=int(free), utilization_percent=int(utilization)))
    return rows


def choose_gpu(rows, minimum=16000):
    available = [r for r in rows if r['free_mib'] >= minimum]
    return max(available, key=lambda r: (r['free_mib'], -r['utilization_percent'])) if available else None


def event(state_dir, message):
    with (state_dir / 'events.txt').open('a') as handle:
        handle.write(f'{now()} {message}\n')


def notify(message):
    # Optional local desktop notification, never email/Slack or another chat.
    if shutil.which('notify-send'):
        try:
            subprocess.run(['notify-send', 'ATE Phase 3', message], capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass


def run_logged(command, log_path, env=None, timeout=None):
    with log_path.open('a') as handle:
        result = subprocess.run(command, cwd=PROJECT, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, timeout=timeout)
    return result.returncode


def publish(run_dir, success, state_dir):
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPOSITORY, text=True).strip()
    if branch != 'main':
        raise RuntimeError(f'Expected main branch; found {branch}. Results preserved; no branch change or commit attempted')
    latest = PROJECT / 'results/phase3_knockout/LATEST.md'
    relative = run_dir.relative_to(latest.parent)
    status = 'GPU 实验和回归检查已完成' if success else '自动实验未通过，需要检查诊断'
    details = 'phase3_report.md' if success else 'watch_run_output.txt'
    latest.write_text(f'# Phase 3 最新运行\n\n- 时间：{now()}\n- 状态：{status}\n'
                      f'- [运行报告或诊断]({relative}/{details})\n'
                      f'- 完整运行目录：`{run_dir}`\n'
                      '- 自动轮询已停止；未启动其他 Knockout 类型、Decode、Pruning 或训练。\n')
    # Stage only files produced by this watchdog. Preserve unrelated work.
    paths = [str(run_dir.relative_to(REPOSITORY)), str(latest.relative_to(REPOSITORY))]
    preexisting = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=REPOSITORY, text=True)
    if preexisting.strip():
        raise RuntimeError('Git index already contains other staged changes; results preserved, no unrelated files committed')
    subprocess.run(['git', 'add', '--', *paths], cwd=REPOSITORY, check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=REPOSITORY, check=True)
    message = 'Record verified automated Phase 3 GPU knockout experiments' if success else 'Record automated Phase 3 failure diagnostics'
    subprocess.run(['git', 'commit', '-m', message], cwd=REPOSITORY, check=True, capture_output=True, text=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip()
    push_log = state_dir / 'github_sync.txt'
    commands = [['git', 'push', 'origin', 'main'],
        ['git', '-c', 'core.sshCommand=ssh -i /home/zsy/.ssh/codex_reyuyu_ATE_deploy -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o HostKeyAlias=github.com -o HostName=ssh.github.com -p 443', 'push', 'origin', 'main']]
    # No force-push, credential reads, reset, stash, or automatic conflict resolution.
    for command in commands:
        if run_logged(command, push_log) == 0:
            remote_command = command[:-3] + ['ls-remote', 'origin', 'refs/heads/main']
            remote = subprocess.check_output(remote_command, cwd=REPOSITORY, text=True, timeout=60).split()[0]
            if remote != commit:
                raise RuntimeError(f'Remote SHA differs from result commit: {remote} vs {commit}')
            return commit
    raise RuntimeError('GitHub push failed; results and local commit preserved, see github_sync.txt')


def execute_experiment(state_dir, state, gpu):
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_dir = PROJECT / 'results/phase3_knockout' / f'auto_gpu_{timestamp}'
    run_dir.mkdir(exist_ok=False)
    state.update(status='running', started_at=now(), physical_gpu=gpu,
                 run_directory=str(run_dir), worker_pid=os.getpid())
    write_json(state_dir / 'status.json', state)
    event(state_dir, f'Starting Phase 3 on physical GPU {gpu["index"]}: {run_dir}')
    notify(f'GPU {gpu["index"]} 显存满足条件，开始 Phase 3。')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu['index']), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', MPLBACKEND='Agg')
    command = [str(PYTHON), str(PROJECT / 'knockout_probe.py'), '--output-dir', str(run_dir)]
    code = run_logged(command, run_dir / 'watch_run_output.txt', env=env)
    state['experiment_exit_code'] = code
    preflight = run_dir / 'resource_preflight.json'
    if code == 2 and preflight.exists() and not json.loads(preflight.read_text())['ready']:
        # The second check caught a race or NVML/CUDA accounting difference.
        # Preserve the attempt outside Git and wait; no weights were loaded.
        archive = state_dir / 'resource_rechecks'
        archive.mkdir(exist_ok=True)
        shutil.move(str(run_dir), str(archive / run_dir.name))
        state.pop('run_directory', None)
        state.pop('worker_pid', None)
        state.update(status='waiting', last_resource_recheck=str(archive / run_dir.name))
        event(state_dir, 'CUDA preflight still insufficient; return to waiting without loading weights')
        return
    success = code == 0
    if success:
        actual = json.loads((run_dir / 'run.json').read_text())
        expected = json.loads((PROJECT / 'configs/phase3_samples.json').read_text())
        validated = json.loads((run_dir / 'samples' / actual['samples_completed'][0] / 'correctness_validation.json').read_text())
        success = (actual['completed'] and actual['cutoffs'] == CUTOFFS and validated['passed']
                   and actual['samples_completed'] == [s['id'] for s in expected['samples']])
        if success:
            tests = [str(PYTHON), '-m', 'unittest', 'test_attention_probe',
                     'test_attention_positions', 'test_knockout', 'test_phase3_watch', '-v']
            success = run_logged(tests, run_dir / 'regression_test_results.txt', env=env) == 0
    state.update(status='completed' if success else 'failed', finished_at=now(), experiment_verified=bool(success))
    write_json(run_dir / 'watch_result.json', state)
    try:
        state['commit'] = publish(run_dir, success, state_dir)
        state['github_remote_verified'] = True
    except Exception as exc:
        state.update(status='sync_failed', synchronization_error=str(exc), github_remote_verified=False)
    event(state_dir, f'Final state: {state["status"]}; commit={state.get("commit")}')
    notify('Phase 3 已完成并同步 GitHub。' if state['status'] == 'completed'
           else 'Phase 3 自动运行已停止，需要检查 status.json 和实验诊断。')


def tick(state_dir):
    state_path = state_dir / 'status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'status': 'waiting'}
    if state['status'] in TERMINAL:
        cron_update(state_dir, False)
        return state
    if state['status'] == 'running':
        # Holding the lock proves the previous worker is gone. Do not rerun a
        # partially completed experiment after a restart or lost SSH session.
        state.update(status='failed', error='Previous worker interrupted; inspect partial run before retrying')
    else:
        try:
            rows = read_gpus()
            state.update(last_checked_at=now(), gpu_snapshot=rows)
            gpu = choose_gpu(rows)
            if gpu is not None:
                execute_experiment(state_dir, state, gpu)
            else:
                state.update(status='waiting', minimum_free_mib=16000)
        except Exception as exc:
            state.update(status='failed', error=str(exc), finished_at=now())
            event(state_dir, f'Failure: {exc}')
            notify('Phase 3 轮询停止，请查看本地状态和诊断。')
    write_json(state_path, state)
    if state['status'] in TERMINAL:
        cron_update(state_dir, False)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--install-cron', action='store_true')
    actions.add_argument('--tick', action='store_true')
    actions.add_argument('--status', action='store_true')
    actions.add_argument('--disable', action='store_true')
    args = parser.parse_args()
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state_dir / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('A Phase 3 worker is already running; no duplicate started.')
            return
        path = state_dir / 'status.json'
        if args.status:
            print(path.read_text() if path.exists() else 'Not installed')
        elif args.disable:
            cron_update(state_dir, False)
            write_json(path, {'status': 'disabled', 'disabled_at': now()})
        elif args.install_cron:
            if path.exists() and json.loads(path.read_text())['status'] == 'running':
                raise RuntimeError('Previous experiment state is running; inspect it before reinstalling')
            write_json(path, {'status': 'waiting', 'installed_at': now(), 'interval_minutes': 15,
                              'minimum_free_mib': 16000})
            cron_update(state_dir, True)
            event(state_dir, 'Installed */15 cron watchdog, authorization: continue Phase 3 once a GPU is available')
            print('Installed: every 15 minutes; existing crontab entries preserved.')
        else:
            print(json.dumps(tick(state_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
