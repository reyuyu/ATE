"""Freeze all official TextVQA_VAL requests on CPU, before any predictions."""
import ast
import base64
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import time
from pathlib import Path

from .common import *


def main():
    import torch
    from PIL import Image
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    assert not (OUT/'sample_manifest.jsonl.gz').exists(), 'Do not replace frozen requests'
    data = OUT/'server_only/TextVQA_VAL.tsv'
    md5 = hashlib.md5()
    with data.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            md5.update(block)
    assert md5.hexdigest() == DATASET_MD5
    scope = upstream()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False, local_files_only=True)
    old = json.loads(Path('/data/lk/ATE-resources/benchmarks/phase4_0/sources/textvqa/TextVQA_0.5.1_val.json').read_text())
    original = {str(r['question_id']): r for r in old['data']}
    assert len(original) == 5000
    images = OUT/'server_only/images'
    images.mkdir(parents=True, exist_ok=True)
    metadata, visual, rows = {}, {}, []
    csv.field_size_limit(32*1024**2)
    start = time.monotonic()
    with data.open() as stream:
        for number, row in enumerate(csv.DictReader(stream, delimiter='\t')):
            key = str(row['index'])
            gold = ast.literal_eval(row['answer'])
            assert row['question'] == original[key]['question'] and gold == original[key]['answers']
            image = row['image']
            if len(image) > 64:
                source_sha = hashlib.sha256(image.encode()).hexdigest()
                if source_sha not in visual:
                    # Fresh official dump_image uses PIL save(.jpg), not the original JPEG bytes.
                    decoded = scope['decode_base64_to_image'](image)
                    buffer = io.BytesIO()
                    decoded.save(buffer, format='JPEG')
                    encoded = buffer.getvalue()
                    image_sha = hashlib.sha256(encoded).hexdigest()
                    path = images/(image_sha+'.jpg')
                    old_image = Path('/data/lk/ATE-benchmarks-20261105/datasets/TextVQA/images')/(original[key]['image_id']+'.jpg')
                    if old_image.exists() and sha(old_image) == image_sha:
                        path = old_image
                    elif path.exists():
                        assert sha(path) == image_sha
                    else:
                        path.write_bytes(encoded)
                    pixels = scope['load_image'](str(path), max_num=6).to(torch.bfloat16)
                    assert pixels.shape[1:] == (3,448,448) and 1 <= pixels.shape[0] <= 7
                    assert torch.isfinite(pixels).all()
                    visual[source_sha] = dict(image_path=str(path), image_sha256=image_sha,
                        source_image_base64_sha256=source_sha, size=list(decoded.size), tiles=int(pixels.shape[0]),
                        pixels_bf16_sha256=tensor_sha(pixels), pixels_shape=list(pixels.shape))
                    del pixels
                metadata[key] = visual[source_sha]
                ref = key
            else:
                ref = image
            rows.append(dict(question_id=key, image_ref=ref, question=row['question'], gold=gold,
                original_image_id=original[key]['image_id']))
            if (number+1) % 250 == 0:
                print('OFFICIAL_CPU_INPUTS', number+1, 'unique_images', len(visual), flush=True)
    assert len(rows) == len({r['question_id'] for r in rows}) == 5000
    for row in rows:
        row.update(metadata[row.pop('image_ref')])
        request = official_request(scope, row['question'], row['image_path'])
        prompt, ids, roles = render(tok, request, row['tiles'])
        row.update(request=request, prompt=prompt, input_ids=ids[0].tolist(), roles=roles,
                   input_tokens=int(ids.shape[1]), image_tokens=len(roles['image']),
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), input_ids_sha256=tensor_sha(ids),
                   generation=GENERATION)
        row['input_binding_sha256'] = digest({k: row[k] for k in ['question_id','gold','image_sha256','prompt_sha256',
            'pixels_bf16_sha256','input_ids_sha256','image_tokens','generation']})
    rows.sort(key=lambda r: int(r['question_id']))
    path = OUT/'sample_manifest.jsonl.gz'
    with gzip.open(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True)+'\n')
    ordered = sorted(rows, key=lambda r: (r['image_tokens'],r['input_tokens'],r['question_id']))
    smoke = [ordered[0]['question_id'], ordered[len(ordered)//2]['question_id'], ordered[-1]['question_id']]
    write(OUT/'dataset_validation.json', dict(status='CPU_INPUTS_FROZEN_NOT_GPU_VALIDATED', questions=5000,
        unique_image_sha256=len({r['image_sha256'] for r in rows}), matched_original_questions_and_ten_gold=5000,
        official_tsv_md5=md5.hexdigest(), official_tsv_sha256=sha(data),
        dataset_url='https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv',
        verified_tls_mirror='https://opencompass.oss-cn-shanghai.aliyuncs.com/utils/VLMEval/TextVQA_VAL.tsv',
        mirror_note='Identical Content-Length/ETag; dataset identity checked against upstream pinned MD5. No TLS verification disabled for the download.',
        tiles_histogram=dict(Counter(r['tiles'] for r in rows)), max_visual_tokens=max(r['image_tokens'] for r in rows),
        manifest_sha256=sha(path), smoke_ids=smoke, cpu_elapsed_s=time.monotonic()-start,
        cuda_initialized=torch.cuda.is_initialized(), image_conversion='Pinned fresh VLMEvalKit PIL JPEG dump semantics; never old 1-tile pixels.',
        image_deduplication='Deduplicated identical official payload/decoded bytes only; all 5000 questions retained.'))
    write(OUT/'prompt_examples.json', [{k:r[k] for k in ['question_id','question','request','prompt','input_ids','tiles','image_tokens',
        'input_tokens','prompt_sha256','pixels_bf16_sha256','generation']} for r in ordered[::1000]] +
        [{k:ordered[-1][k] for k in ['question_id','request','prompt','input_ids','tiles','image_tokens','generation']}])
    print('CPU_COMPLETE',len(rows),'requests; smoke',smoke,flush=True)


if __name__ == '__main__':
    main()
