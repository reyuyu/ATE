import csv
import json

from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1_single_layer_gqa6 import run as prior
from .protocol import OLD, OUTPUT, PROTOCOL, SEED, TASKS


def select(samples):
    chosen=[]
    for task in TASKS:
        pool=[s for s in samples if s['task']==task]
        if len(pool)!=500:raise ValueError('Original task does not contain exactly500')
        def key(s):
            return digest([SEED,'native-updates-A-v1',task,s['question_id'],s['image_sha256'],s['input_mapping']['prompt_ids']])
        chosen.extend(sorted(pool,key=key)[:100])
    if len({s['question_id'] for s in chosen})!=600:raise ValueError('Duplicate selected question')
    return chosen


def freeze():
    prior.verify()
    if (OUTPUT/'execution_manifest.json').exists():raise ValueError('Already execution-frozen')
    chosen=select(json.loads((OLD/'sample_manifest.json').read_text())['samples'])
    keys={(s['task'],s['question_id']) for s in chosen}
    historical={}
    for filename,field in [('bos_mass_per_question.csv','bos_mass'),('first_token_per_question.csv','delta_logp')]:
        seen=set()
        with (OLD/filename).open() as f:
            for row in csv.DictReader(f):
                key=(row['task'],row['question_id'])
                if key not in keys:continue
                identity=(*key,int(row['layer']))
                if identity in seen:raise ValueError('Duplicate historical join')
                seen.add(identity)
                historical.setdefault('__'.join(key),{}).setdefault(row['layer'],{})[field]=float(row[field])
        if len(seen)!=19200:raise ValueError('Incomplete historical CSV join')
    references={}
    for s in chosen:
        name=s['task']+'__'+s['question_id']
        checkpoint=OLD/f"first_token/shard{s['rank']}/checkpoints/{name}.json"
        old=json.loads(checkpoint.read_text())
        if old['sample_sha256']!=digest(s):raise ValueError('Original sample binding changed')
        result=old['result']
        if sha256(result['logits_path'])!=result['logits_sha256'] or sha256(s['image_path'])!=s['image_sha256']:
            raise ValueError('Missing/changed original image or raw Full logits')
        for i in range(32):
            row=historical[name][str(i)]
            if row!={k:result[k2][i][k] for k,k2 in [('bos_mass','bos'),('delta_logp','metrics')]}:
                raise ValueError('Historical CSV/checkpoint mismatch')
            if result['bos'][i]['query']!=s['input_mapping']['p_minus_one']:
                raise ValueError('Historical query mapping differs')
        references[name]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            logits_path=result['logits_path'],logits_sha256=result['logits_sha256'],rows=historical[name])
    sources={str(OLD/f):sha256(OLD/f) for f in ['sample_manifest.json','completion.json',
        'execution_manifest.json','bos_mass_per_question.csv','first_token_per_question.csv','first_token_layer_summary.csv']}
    freeze_json(OUTPUT/'sample_manifest.json',dict(protocol=PROTOCOL,samples=chosen))
    freeze_json(OUTPUT/'historical_bindings.json',references)
    for rank in range(4):freeze_json(OUTPUT/f'manifests/shard{rank}.json',dict(samples=[s for s in chosen if s['rank']==rank]))
    freeze_json(OUTPUT/'data_manifest.json',dict(questions=600,unique_images=len({s['image_id'] for s in chosen}),
        per_task={t:dict(questions=100,images=len({s['image_id'] for s in chosen if s['task']==t})) for t in TASKS},
        per_gpu={str(i):sum(s['rank']==i for s in chosen) for i in range(4)},
        source_sha256=sources,sample_manifest_sha256=sha256(OUTPUT/'sample_manifest.json'),
        historical_bindings_sha256=sha256(OUTPUT/'historical_bindings.json'),
        shard_sha256={str(i):sha256(OUTPUT/f'manifests/shard{i}.json') for i in range(4)},
        selection_before_new_outputs=True,replacements=0))
    print(json.dumps(json.loads((OUTPUT/'data_manifest.json').read_text()),indent=2))


if __name__=='__main__':freeze()
