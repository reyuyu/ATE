"""Bounded source availability checks and local inventory; no GPU or sample selection."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import requests

from benchmarks.acquire import download
from benchmarks.dataset_registry import sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/paper_rq1_attention_time'
RESOURCES = Path('/data/lk/ATE-resources/paper_rq1')
OLD = Path('/data/lk/ATE-resources/benchmarks/phase4_0')
URLS = {
    'gqa_download.html': 'https://cs.stanford.edu/people/dorarad/gqa/download.html',
    'gqa_evaluate.html': 'https://cs.stanford.edu/people/dorarad/gqa/evaluate.html',
    'gqa_mirror.json': 'https://hf-mirror.com/api/datasets/lmms-lab/GQA',
    'textcaps_val.json': 'https://dl.fbaipublicfiles.com/textcaps/TextCaps_0.1_val.json',
    'textcaps_dataset.html': 'https://textvqa.org/textcaps/dataset/',
    'coco_caption_readme.md': 'https://raw.githubusercontent.com/tylin/coco-caption/master/README.md',
    'gqa_mirror_official.json': 'https://huggingface.co/api/datasets/lmms-lab/GQA',
}


def fetch(item):
    name, url = item
    session = requests.Session()
    session.trust_env = False
    return dict(name=name, **download(session, url, RESOURCES/'sources'/name, cap=80*1024**2))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = list(ThreadPoolExecutor(4).map(fetch, URLS.items()))
    for r in records:
        print(r['name'], r['status'], r.get('bytes', r.get('error')), flush=True)
    pope = [json.loads(x) for x in (OLD/'sources/pope/output/coco/coco_pope_adversarial.json').read_text().splitlines() if x]
    science = json.loads((OLD/'sources/scienceqa/data/scienceqa/problems.json').read_text())
    test_ids = json.loads((OLD/'sources/scienceqa/data/scienceqa/pid_splits.json').read_text())['test']
    text = json.loads((OLD/'sources/textvqa/TextVQA_0.5.1_val.json').read_text())['data']
    counts = dict(pope_adversarial=dict(questions=len(pope), images=len({r['image'] for r in pope})),
                  scienceqa_img=dict(questions=sum(bool(science[q]['image']) for q in test_ids)),
                  textvqa=dict(questions=len(text), images=len({r['image_id'] for r in text})),
                  gqa_balanced=dict(local_official_annotation=False),
                  coco_caption=dict(local_official_annotation=False), textcaps=dict(local_official_annotation=False))
    report = dict(status='preflight_not_frozen', time_unix=time.time(), source_checks=records, local_annotation_counts=counts,
                  existing_scienceqa_parquet=str(OLD/'sources/scienceqa_hf/test-00000-of-00001-f0e719df791966ff.parquet'),
                  safety='No GPU work, no selection, no substitution of correct_question GQA. Availability is not a frozen split.')
    (OUTPUT/'initial_inventory.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
