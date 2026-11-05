"""Bounded, revision-pinned annotation acquisition; never downloads image archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import requests

REPOS = {
    'information_flow': ('FightingFighting/cross-modal-information-flow-in-MLLM', 'c9a1111dac4baa175f67716fab473789320187c0', [
        'datasets/GQA_val_correct_question_with_choose_ChooseAttr.csv',
        'datasets/GQA_val_correct_question_with_positionQuery_QueryAttr.csv',
        'datasets/GQA_val_correct_question_with_existThatOr_LogicalObj.csv', 'README.md']),
    'pope': ('AoiDragon/POPE', '08d957b917e5a378a2f99d35b6293c536a66298b', [
        'output/coco/coco_pope_random.json', 'output/coco/coco_pope_popular.json',
        'output/coco/coco_pope_adversarial.json', 'evaluate.py', 'README.md']),
    'scienceqa': ('lupantech/ScienceQA', '2cbf8318e07b9ece895bb2ae605e71e38d623264', [
        'data/scienceqa/problems.json', 'data/scienceqa/pid_splits.json', 'README.md']),
    'llava': ('haotian-liu/LLaVA', 'c121f0432da27facab705978f83c4ada465e46fd', [
        'docs/Evaluation.md', 'llava/eval/eval_pope.py', 'llava/eval/eval_science_qa.py',
        'llava/eval/eval_textvqa.py', 'llava/eval/m4c_evaluator.py',
        'llava/eval/model_vqa_loader.py', 'llava/eval/model_vqa_science.py',
        'scripts/convert_gqa_for_eval.py', 'scripts/convert_sqa_to_llava.py', 'LICENSE']),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def download(session, url, path, cap=80 * 1024**2, headers=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {'url': url, 'path': str(path), 'max_bytes': cap}
    if path.is_file():
        return dict(record, status='present', bytes=path.stat().st_size, sha256=sha(path))
    tmp = path.with_suffix(path.suffix + '.partial')
    try:
        with session.get(url, stream=True, timeout=(5, 30), headers=headers) as r:
            record['http_status'] = r.status_code
            r.raise_for_status()
            record.update(etag=r.headers.get('ETag'), last_modified=r.headers.get('Last-Modified'),
                          final_url=r.url.split('?')[0])
            if int(r.headers.get('content-length', 0)) > cap:
                raise ValueError('Response exceeds explicit download cap')
            n = 0
            with tmp.open('wb') as f:
                for chunk in r.iter_content(1024**2):
                    n += len(chunk)
                    if n > cap:
                        raise ValueError('Stream exceeds explicit download cap')
                    if shutil.disk_usage(path.parent).free < 50 * 1024**3:
                        raise ValueError('Less than 50GiB safety headroom')
                    f.write(chunk)
            tmp.replace(path)
        return dict(record, status='downloaded', bytes=n, sha256=sha(path))
    except Exception as e:
        return dict(record, status='failed', error=f'{type(e).__name__}: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.trust_env = False
    records = []
    for name, (repo, revision, files) in REPOS.items():
        for file in files:
            url = f'https://api.github.com/repos/{repo}/contents/{file}?ref={revision}'
            result = download(s, url, args.root/name/file, headers={'Accept': 'application/vnd.github.raw+json'})
            result.update(repository=repo, revision=revision, source_file=file)
            records.append(result)
            print(name, file, result['status'], result.get('bytes', result.get('error')), flush=True)
    for name, url in {
        'gqa_download.html': 'https://cs.stanford.edu/people/dorarad/gqa/download.html',
        'gqa_evaluate.html': 'https://cs.stanford.edu/people/dorarad/gqa/evaluate.html',
        'textvqa_dataset.html': 'https://textvqa.org/dataset/',
        'textvqa_hf.json': 'https://hf-mirror.com/api/datasets/lmms-lab-encoder/textvqa',
        'scienceqa_hf.json': 'https://hf-mirror.com/api/datasets/derek-thomas/ScienceQA',
    }.items():
        result = download(s, url, args.root/'preflight'/name)
        records.append(result)
        print(name, result['status'], result.get('bytes', result.get('error')), flush=True)
    # The mirror builder is required to recover official ScienceQA PID mappings.
    for file in ('tutorial/ScienceQA.py', 'README.md'):
        revision = 'f18b0a70359ebfb41f658fd564208d0355b013f4'
        url = f'https://hf-mirror.com/datasets/derek-thomas/ScienceQA/resolve/{revision}/{file}'
        result = download(s, url, args.root/'scienceqa_hf'/file)
        result.update(repository='derek-thomas/ScienceQA', revision=revision)
        records.append(result)
    audit = args.root/'acquisition.json'
    previous = json.loads(audit.read_text()) if audit.exists() else []
    previous.append({'time_unix': time.time(), 'records': records})
    audit.write_text(json.dumps(previous, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
