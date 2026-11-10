"""Archive selected historical CPU-adapter source via git show, never execute it."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKTREE = ROOT.parents[3]
COMMIT = '7e18895a6ccd57c4cb3aaedd5d512789e7467b59'
PREFIX = 'vlm_research/results/qwen3vl_c3_3090_official_eval/'
FILES = ['scripts/benchmark_inputs.py', 'scripts/runtime.py', 'scripts/freeze_benchmarks.py',
         'scripts/scorers.py', 'scripts/scorers_v2.py', 'scripts/scorers_v3.py',
         'scoring_reference/score.py', 'scoring_reference/frozen_matching_util.py',
         'scoring_reference/official_llava_m4c_evaluator.py']


def main():
    rows = []
    for path in FILES:
        data = subprocess.check_output(['git', 'show', f'{COMMIT}:{PREFIX}{path}'], cwd=WORKTREE)
        target = ROOT / 'reusable_historical_source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f'Refusing changed historical snapshot: {target}')
        target.write_bytes(data)
        rows.append(dict(commit=COMMIT, repository_path=PREFIX+path,
                         archive_path=str(target), sha256=hashlib.sha256(data).hexdigest(),
                         reuse_status='READ_ONLY_REFERENCE_NOT_EXECUTED_NOT_OFFICIAL_AUTHOR_RECIPE'))
    (ROOT / 'reusable_code_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps({'historical_source_snapshots': len(rows), 'executed': False}))


if __name__ == '__main__':
    main()
