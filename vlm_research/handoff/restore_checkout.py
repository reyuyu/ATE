"""Hydrate an interrupted public checkout against its verified Git object IDs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote

import requests


COMMIT = '7970be981e3462560cb66d83e1014eaa1b1ebdae'


def blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--transport-prefix', default='')
    p.add_argument('--workers', type=int, default=6, choices=range(1, 9))
    args = p.parse_args()
    if args.audit.exists():
        p.error('Use a new audit file; completed attempts are never overwritten')
    repo = args.repo.resolve()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if head != COMMIT:
        raise ValueError('Refuse to hydrate a different commit')
    tree = json.loads(args.tree.read_text())
    if tree.get('truncated'):
        raise ValueError('Incomplete official tree')
    files = [f for f in tree['tree'] if f['type'] == 'blob']
    files.sort(key=lambda f: (not (f['path'].startswith('vlm_research/') and f['path'].count('/') == 1), f['size']))
    raw = subprocess.check_output(['git', 'ls-tree', '-r', '-z', 'HEAD'], cwd=repo)
    expected = {}
    for row in raw.split(b'\0'):
        if row:
            meta, name = row.split(b'\t', 1)
            mode, kind, digest = meta.decode().split()
            if mode != '100644' or kind != 'blob':
                raise ValueError('Unexpected symlink/executable/submodule; inspect explicitly')
            expected[name.decode()] = digest
    if expected != {f['path']: f['sha'] for f in files}:
        raise ValueError('API tree disagrees with local commit tree')
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    state = dict(commit=head, started=datetime.now(timezone.utc).isoformat(),
                 transport_prefix=args.transport_prefix, expected_files=len(files),
                 tree_sha256=hashlib.sha256(args.tree.read_bytes()).hexdigest(),
                 status='running', files=[], failures=[])

    def save():
        temp = args.audit.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(args.audit)

    save()

    def restore(item):
        name = item['path']
        path = repo / name
        if not path.resolve().is_relative_to(repo) or path.is_symlink():
            raise ValueError('Unsafe path: ' + name)
        if path.exists():
            data = path.read_bytes()
            if blob_sha(data) != item['sha']:
                raise ValueError('Preserve and investigate changed local file: ' + name)
            source = 'existing_exact'
        else:
            source = 'downloaded_verified'
            url = args.transport_prefix + 'https://raw.githubusercontent.com/reyuyu/ATE/' + COMMIT + '/' + quote(name)
            with requests.Session() as session:
                session.trust_env = False
                for attempt in range(4):
                    try:
                        response = session.get(url, timeout=(15, 90))
                        if response.status_code == 429:
                            wait = response.headers.get('Retry-After', '')
                            time.sleep(int(wait) if wait.isdigit() else 30 * (attempt + 1))
                            continue
                        response.raise_for_status()
                        data = response.content
                        if len(data) != item['size'] or blob_sha(data) != item['sha']:
                            raise ValueError('Downloaded object hash/size mismatch: ' + name)
                        break
                    except (requests.RequestException, ValueError):
                        if attempt == 3:
                            raise
                        time.sleep(2 ** (attempt + 1))
                else:
                    raise RuntimeError('Repeated rate limit: ' + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation preserves any concurrently introduced user file.
            with path.open('xb') as target:
                target.write(data)
        written = subprocess.check_output(['git', 'hash-object', '--no-filters', '-w', '--stdin'],
                                          input=data, cwd=repo).decode().strip()
        if written != item['sha']:
            raise ValueError('Git object write mismatch: ' + name)
        return dict(path=name, bytes=len(data), git_blob_sha1=written, source=source)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = {pool.submit(restore, f): f['path'] for f in files}
            for task in as_completed(tasks):
                try:
                    state['files'].append(task.result())
                except Exception as exc:
                    state['failures'].append(dict(path=tasks[task], error=repr(exc)))
                if (len(state['files']) + len(state['failures'])) % 20 == 0:
                    save()
                    print(json.dumps(dict(verified=len(state['files']), failed=len(state['failures']),
                                          total=len(files))), flush=True)
        if state['failures']:
            raise RuntimeError('Checkout incomplete; see per-file failures in audit')
        subprocess.run(['git', 'read-tree', 'HEAD'], cwd=repo, check=True)
        diff = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--'], cwd=repo, text=True)
        if diff.strip():
            raise ValueError('Tracked checkout differs from commit: ' + diff)
        state['status'] = 'complete_all_tracked_files_exact'
    except BaseException as exc:
        state.update(status='incomplete', error=repr(exc))
        raise
    finally:
        state['finished'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
