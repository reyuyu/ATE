"""Bind every official validation row, local image and native prompt before inference."""
import collections
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

from .common import *
from ..benchmarks.protocols import SHORT
from ..fastp2_bypass_writer.common import hardware


def main():
    assert not (OUT / 'execution_freeze.json').exists(), 'Do not overwrite frozen experiment'
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, '/data/lk/ATE-resources/benchmarks/phase4_0/tools')
    import pyarrow.parquet as pq
    source = pq.read_table(ANNOTATION, columns=['id','imageId','question','answer','isBalanced','types']).to_pylist()
    raw = {s['id']: s for s in source}
    assert len(raw) == len(source) == 132062 and all(s['isBalanced'] for s in source)
    del source
    rows = [json.loads(line) for line in (DATA / 'inputs.jsonl').open()]
    answers = read(DATA / 'answers.json')
    assert len(rows) == len(answers) == len(raw) and {r['question_id'] for r in rows} == set(raw)
    inventory = {r['image_id']: r for r in read(AUDIT / 'GQA_image_validation.json')}
    assert len(inventory) == 10234
    for i, r in enumerate(inventory.values()):
        assert r['readable'] and sha(r['path']) == r['sha256']
        with Image.open(r['path']) as image:
            image.verify()
        if (i+1) % 2000 == 0:
            print('IMAGE_AUDIT', i+1, flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    frozen = []
    for start in range(0, len(rows), 512):
        batch = rows[start:start+512]
        chats = [processor.apply_chat_template([dict(role='user', content=[dict(type='image'),
            dict(type='text', text=r['question']+'\n'+SHORT)])], add_generation_prompt=True) for r in batch]
        encoded = processor.tokenizer(chats, return_offsets_mapping=True, padding=False)
        for r, chat, compact, offsets in zip(batch, chats, encoded['input_ids'], encoded['offset_mapping']):
            qid = r['question_id']; original = raw[qid]
            assert r['split'] == 'val_balanced'
            assert (r['image_id'],r['question'],answers[qid]) == (original['imageId'],original['question'],original['answer'])
            assert r['image_path'] == inventory[r['image_id']]['path']
            assert compact.count(32000) == 1
            image_index = compact.index(32000)
            ids = compact[:image_index]+[32000]*576+compact[image_index+1:]
            qstart = chat.index(r['question']); qend = qstart+len(r['question'])
            q = [i+(575 if i>image_index else 0) for i,(a,b) in enumerate(offsets) if b>qstart and a<qend]
            assert q and min(q)>image_index+575 and max(q)<len(ids)-1
            frozen.append(dict(r, dataset='GQA', task=original['types']['structural'], types=original['types'],
                answer=answers[qid], image_sha256=inventory[r['image_id']]['sha256'],
                prompt_text=r['question']+'\n'+SHORT, compact_prompt_ids=compact,
                question_positions=q, expanded_prompt_ids_sha256=digest(ids),
                rendered_prompt_sha256=digest(chat)))
        if start % 10240 == 0:
            print('PROMPT_FREEZE',len(frozen),flush=True)
    groups = collections.defaultdict(list)
    for s in frozen:
        groups[s['image_sha256']].append(s['question_id'])
    loads = [0,0]; shards = {}
    for h, keys in sorted(groups.items(), key=lambda x:(-len(x[1]), digest([SEED,x[0]]))):
        rank = min(range(2), key=lambda i:(loads[i],i))
        shards[h] = rank; loads[rank] += len(keys)
    smoke = []
    for task in ('query','verify','choose'):
        pool = [s for s in frozen if s['task']==task]
        smoke.append(min(pool,key=lambda s:digest([SEED,'smoke',s['question_id']]))['question_id'])
    with gzip.open(OUT / 'sample_manifest.jsonl.gz', 'wt', compresslevel=6) as stream:
        for s in sorted(frozen,key=lambda s:s['question_id']):
            stream.write(json.dumps(s, ensure_ascii=False, separators=(',',':'))+'\n')
    atomic(OUT / 'partition.json',dict(image_shard=shards,question_counts=loads,smoke=smoke))
    annotations = {str(p):sha(p) for p in (DATA/'inputs.jsonl',DATA/'answers.json',ANNOTATION,
        AUDIT/'GQA_validated.json',AUDIT/'GQA_image_validation.json',
        Path('/data/lk/ATE-resources/paper_rq1/annotations/gqa_eval.py'),
        Path('/data/lk/ATE-resources/benchmarks/phase4_0/sources/llava/scripts/convert_gqa_for_eval.py'))}
    weights = {str(p):sha(p) for p in sorted(MODEL.glob('*.safetensors'))}
    assert weights
    atomic(OUT / 'protocol.json',dict(created_at=now(),seed=SEED,revision=REVISION,
        model='llava-hf/llava-1.5-7b-hf',dtype='FP16',attention='eager',transformers='4.49.0',
        questions=len(frozen),image_ids=len(inventory),image_sha_clusters=len(groups),split='val_balanced',
        hidden_official_test=False,labels=LABELS,answer_image_layers=KEEP,visual_hidden_exit=CUTOFF,
        receiver_image_layers=list(range(20)),answer_attention_bypass=[2,3,4,5],
        max_new_tokens=32,logits_to_keep=1,greedy=True,independent_KV=True,
        R='All post-image text queries before original P-1; no new R writing above L19.',
        scoring='Pinned LLaVA conversion text.rstrip(\'.\').lower(), then exact equality to official short answer; question weighted.',
        image_caveat='Complete local MMEB GQA images; original-resolution identity to Stanford archive not established. Not a testdev leaderboard score.',
        comparison='C3-V2+L11-L24 minus C3-V2; paired image-SHA cluster bootstrap 20000, seed fixed; exploratory.',
        tolerances=TOLERANCES['llava'],devices=DEVICES,performance_benchmark=False,
        no_native_full_accuracy_run=True,all_264124_generations_new=True,
        vision_reuse='At most one current-image feature tensor; exact equality validated in smoke. No cross-condition language KV reuse.',
        disk_safety_gib=10,annotations=annotations))
    atomic(OUT / 'data_audit.json',dict(at=now(),questions=len(frozen),unique_ids=len(raw),
        image_ids=len(inventory),image_clusters=len(groups),all_images_readable_and_rehashed=True,
        rows_exactly_match_upstream_parquet=True,types=dict(collections.Counter(s['task'] for s in frozen)),
        no_output_filtering=True,annotations=annotations))
    atomic(OUT / 'execution_freeze.json',dict(at=now(),sources=sources(),weights=weights,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        artifacts={n:sha(OUT/n) for n in ('sample_manifest.jsonl.gz','partition.json','protocol.json','data_audit.json','cpu_tests.log')}))
    atomic(OUT / 'initial_resources.json',dict(at=now(),**hardware()))
    print('FROZEN',len(frozen),'questions',len(groups),'image clusters',loads,flush=True)


if __name__ == '__main__':
    main()
