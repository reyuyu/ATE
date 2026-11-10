"""Create hashes/completion metadata for this static audit, never for experiments."""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
WORKTREE = ROOT.parents[3]


def main():
    tests = json.loads((ROOT / 'static_validation.json').read_text())
    assert tests['static_status'] == 'PASS'
    manifest = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.name in {'artifact_sha256.json', 'SHA256SUMS', 'completion.json'}:
            continue
        data = path.read_bytes()
        manifest.append(dict(path=str(path.relative_to(ROOT)), bytes=len(data),
                             sha256=hashlib.sha256(data).hexdigest()))
    (ROOT / 'artifact_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in manifest))
    done = dict(scope='P0_STATIC_PROVENANCE_DESIGNS_ONLY',
                status='STATIC_AUDIT_COMPLETE_EVALUATION_NOT_READY',
                author_protocol='UNRESOLVED_EXCEPT_PARTIAL_MMBENCH_EVIDENCE',
                physical_status='NUMERICAL_EQUIVALENCE_FAILED_UNCHANGED',
                created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                source_audit_commit='f238d375ffda785d3062069ce6e4fd0c729cd9a6',
                branch=subprocess.check_output(['git','branch','--show-current'],cwd=WORKTREE).decode().strip(),
                static_tests=tests['tests'], gpu_requests=0, model_loads=0, new_predictions=0,
                vllm_installed=False, source_algorithm_modified=False, old_blockers_modified=False,
                pushed=False, pr_created=False,
                files=len(manifest), total_bytes=sum(r['bytes'] for r in manifest),
                hashes_file='SHA256SUMS', self_metadata_excluded_from_hashes=True)
    (ROOT / 'completion.json').write_text(json.dumps(done,indent=2)+'\n')
    print(json.dumps(done,indent=2))


if __name__ == '__main__':
    main()
