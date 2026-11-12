"""CPU-only full dataset, provenance, image SHA and sharding freeze."""
import collections
import shutil
from PIL import Image
from .common import *

def main():
    assert not (OUT/'sample_manifest.json').exists(), 'Do not overwrite frozen inputs'
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    audit=Path('/data/lk/ATE-benchmark-audit-20261105-2230')
    safety=read(audit/'dataset_safety_final.json')
    for bench in ('MME','SEED-Bench'):
        assert sha(DATA/bench/'inputs.jsonl')==safety['checks'][bench]['inputs_sha256']
    copies=['MME_validated.json','SEED-Bench_validated.json','MME_official_eval_alignment.json',
        'MME_subtask_coverage.json','download_sources.json','upstream_integrity_checks.json']
    for n in copies:
        p=OUT/'download_audit'/n;p.parent.mkdir(exist_ok=True);shutil.copy2(audit/n,p)
    tool=DATA/'MME-official-eval-tool/eval_tool/calculation.py'
    shutil.copy2(tool,OUT/'official_reference/MME_calculation.py')
    image_meta={}
    def image_info(p):
        p=str(p)
        if p not in image_meta:
            assert Path(p).is_file(),p
            with Image.open(p) as im:
                im.load();w,h=im.size
            image_meta[p]=dict(sha256=sha(p),width=w,height=h,bytes=Path(p).stat().st_size)
        return image_meta[p]
    samples=[];counts={}
    anns=read(DATA/'MME/answers.json')
    mme=[json.loads(l) for l in (DATA/'MME/inputs.jsonl').open()]
    for i,q in enumerate(mme):
        info=image_info(q['image_path'])
        s=dict(key=f'MME__{i:05d}',benchmark='MME',record_index=i,source_question_id=q['question_id'],
               question=q['question'],answer=anns[q['question_id']],task=q['subtask'],
               image_path=q['image_path'],image_sha256=info['sha256'],width=info['width'],height=info['height'],
               pair_id=q['image_id'])
        assert s['answer'] in ('yes','no')
        # Align the exact source question and gold with its native image-side txt.
        raw=Path(q['image_path']).with_suffix('.txt').read_text().splitlines()
        assert any(parts[0]==s['question'] and parts[1].lower()==s['answer']
                   for parts in (line.split('\t') for line in raw) if len(parts)==2),s['key']
        samples.append(s)
    assert len(mme)==2374 and len({s['image_path'] for s in samples})==1187
    assert set(q['subtask'] for q in mme)==set(sum(TASKS.values(),[]))
    pair=collections.Counter(s['pair_id'] for s in samples);assert set(pair.values())=={2}
    raw=read(DATA/'SEED-Bench/SEED-Bench.json');type_names={v:k for k,v in raw['question_type'].items()}
    seed=[]
    for index,q in enumerate(raw['questions']):
        if q['data_type']!='image':continue
        p=DATA/'SEED-Bench/SEED-Bench-image'/q['data_id'];info=image_info(p)
        seed.append(dict(key=f'SEED__{index:05d}',benchmark='SEED',record_index=index,
            source_question_id=str(q['question_id']),question=q['question'],answer=q['answer'],
            choices=[q['choice_'+c] for c in 'abcd'],task=type_names[q['question_type_id']],
            question_type_id=q['question_type_id'],image_path=str(p),image_sha256=info['sha256'],
            width=info['width'],height=info['height']))
        assert q['answer'] in 'ABCD'
    assert len(seed)==14233 and len({s['image_path'] for s in seed})==5624
    assert len({s['source_question_id'] for s in seed})==14232
    samples+=seed
    clusters=collections.defaultdict(list)
    for s in samples:clusters[s['image_sha256']].append(s)
    loads=[0,0];assign={}
    for h in sorted(clusters,key=lambda h:(-len(clusters[h]),digest(['image-shard',20261106,h]))):
        rank=min(range(2),key=lambda r:(loads[r],r));assign[h]=rank;loads[rank]+=len(clusters[h])
    for s in samples:s['shard']=assign[s['image_sha256']]
    assert len({s['key'] for s in samples})==len(samples)
    atomic(OUT/'sample_manifest.json',samples)
    atomic(OUT/'image_manifest.json',image_meta)
    source={str(p):sha(p) for p in [DATA/'MME/inputs.jsonl',DATA/'MME/answers.json',
        DATA/'SEED-Bench/SEED-Bench.json',DATA/'SEED-Bench/inputs.jsonl',tool]}
    atomic(OUT/'data_validation.json',dict(status='PASSED',at=now(),n=len(samples),
        mme=2374,seed=14233,unique_content_sha=len(clusters),unique_paths=len(image_meta),
        duplicate_official_seed_ids_preserved=1,shard_questions=loads,source_hashes=source,
        image_manifest_sha256=sha(OUT/'image_manifest.json'),all_images_decoded=True,
        mme_raw_annotation_alignment=True,no_image_shard_overlap=True))
    print('DATA FROZEN',len(samples),loads,flush=True)

if __name__=='__main__':main()
