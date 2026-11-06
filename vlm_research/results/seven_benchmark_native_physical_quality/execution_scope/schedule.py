"""Control-plane subset dispatcher around the unchanged frozen runner.

This archived execution script deliberately lives outside inference sources:
only the allowed benchmark queue changes; sample/protocol/checkpoint bindings
remain the original exact ones. Its own bytes and plan are separately sealed.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from vlm_research.seven_benchmark_quality.common import OUT, atomic, digest, now, read, sha, verify

HERE = Path(__file__).resolve().parent
ALLOWED = ('POPE', 'ScienceQA-IMG', 'MMBench', 'TextVQA', 'MM-Vet')


def validate():
    plan = read(HERE / 'plan.json')
    binding = read(HERE / 'binding.json')
    assert tuple(plan['run']) == ALLOWED
    assert set(plan['deferred']) == {'GQA', 'VQAv2'}
    assert not plan['automatically_resume_deferred_benchmarks']
    assert binding['plan_sha256'] == sha(HERE / 'plan.json')
    assert binding['dispatcher_sha256'] == sha(__file__)
    assert binding['scientific_protocol_sha256'] == digest(verify())
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    plan = validate()
    if args.worker is not None:
        import torch
        from vlm_research.seven_benchmark_quality import worker
        worker.ORDER = ALLOWED
        atomic(OUT / f'worker{args.worker}/execution_scope.json',
               dict(plan=plan, binding=read(HERE / 'binding.json'), at=now(), pid=os.getpid()))
        sys.argv = [__file__, '--rank', str(args.worker)]
        with torch.inference_mode():
            worker.main()
        atomic(OUT / f'worker{args.worker}/complete.json', dict(
            status='GENERATED_REQUESTED_FIVE_SPLITS', run=ALLOWED,
            deferred=plan['deferred'], at=now()))
    elif args.supervise:
        from vlm_research.seven_benchmark_quality import launch
        original_popen = subprocess.Popen

        def scoped_popen(command, *a, **kw):
            if isinstance(command, (list, tuple)) and 'vlm_research.seven_benchmark_quality.worker' in command:
                rank = command[command.index('--rank') + 1]
                command = [command[0], '-u', str(Path(__file__).resolve()), '--worker', rank]
            return original_popen(command, *a, **kw)

        launch.ORDER = ALLOWED
        subprocess.Popen = scoped_popen
        sys.argv = [__file__, '--supervise']
        try:
            launch.main()
        finally:
            subprocess.Popen = original_popen
        atomic(OUT / 'generation_complete.json', dict(
            status='GENERATED_REQUESTED_FIVE_SPLITS', run=ALLOWED,
            deferred=plan['deferred'], at=now()))
    else:
        with (OUT / 'scope_five_supervisor.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--supervise'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(HERE / 'launch.json', dict(pid=process.pid, at=now(), run=ALLOWED,
                                        deferred=plan['deferred']))
        print('FIVE_BENCHMARK_SUPERVISOR', process.pid, flush=True)


if __name__ == '__main__':
    main()
