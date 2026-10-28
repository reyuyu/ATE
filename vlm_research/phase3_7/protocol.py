"""CPU preparation, provenance, deterministic stratification and immutable binding."""
import ast
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'e89c8acce6e674ecfbd2c8acad57faf71bf4d547'
HISTORY = ROOT / 'results/phase3_2_fixed_k_exit'
SOURCE = ROOT / 'configs/phase3_1_data/GQA_val_correct_question_with_relChooser_ChooseRel.csv'
OUTPUT = ROOT / 'results/phase3_7_l0_l1_pathways'
CONDITIONS = ['B0', 'B1', 'B2', 'B3', 'B4']
PINS = {'torch': '2.6.0+cu124', 'torchvision': '0.21.0+cu124', 'transformers': '4.49.0',
        'accelerate': '1.4.0', 'huggingface_hub': '0.29.3', 'safetensors': '0.5.3',
        'pillow': '11.1.0', 'numpy': '1.26.4', 'matplotlib': '3.10.1', 'sentencepiece': '0.2.0'}
RULES = dict(layers=[0, 1], conditions=CONDITIONS, images=16,
 selection='Within each gold left/right x historical Full correct/wrong stratum, sort abs(Full Margin),qid; split into four contiguous rank bins (numpy.array_split). In each bin prefer a geometry-valid same-length same-P-1-token real pair, then any valid pair, then nearest bin median abs Margin, then qid. One image per bin. No current model outcome used.',
 pair='Original frozen CVPR-author GQA ChooseRel CSV rows only; same image, distinct question and subject object2 ID, positive visible target ROIs with distinct patch sets. Choose minimum prompt-length difference, then qid. No rewritten questions, labels, boxes, padding or resampling.',
 source='Fixed CVPR 2025 author CSV at c9a1111dac4baa175f67716fab473789320187c0, with original GQA IDs/answers/object boxes; not independently verified against original Stanford annotation JSON. Official download page connection timed out during preparation.',
 score='Original full conditional candidate sequence logP, no EOS; fixed left/right each verified one token; original teacher-forced candidate inputs and prediction P-1, margin>0 correct, tie wrong. A is prompt-only Native Full, B uses original candidate-prefill protocol; do not compare them bitwise across lengths.',
 spatial='Per real query and head normalize visual probabilities, then average queries within Question, then heads. P-1 is one query. JS is sqrt(Jensen-Shannon divergence), natural log, no pseudocount; cosine compares pooled spatial maps. Raw visual Mass is separately head/query averaged.',
 hidden='B: compare each token to its own Full then average Question token cosine distances and change norms. A: mean-pool real Question token hidden states separately per question, then compare pooled vectors; never align different questions by absolute token index. P-1 is a single vector. Capture decoder block outputs (after residual/MLP).',
 l0_check='For same image/P-1 token/absolute position matched pairs require exact L0 P-1 input, visual inputs and visual QK scores; compare visual-only FP32 softmax exactly. Raw FP16 attention conditional maps can differ through denominator-dependent rounding. Do not call raw Mass differences spatial selectivity.',
 audit='Reuse original additive mask clone / -inf knockout. Only L0/L1 targets differ; local same-input Native Attention replay checks non-target query rows bitwise. Later layer masks, not their attention values after changed hidden inputs, must remain native. No requirement that unblocked probabilities within a masked query stay equal.',
 interpretation='Descriptive stratified mechanism pilot, not unbiased population estimate or statistical significance. No layer scan, training, decode, cache, physical exit, additional mask or dataset.')


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    from fixed_k_exit import atomic_json
    atomic_json(path, value)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(list(ROOT.glob('*.py')) + list((ROOT/'phase3_7').glob('*.py')))}


def packages():
    found = {k: importlib.metadata.version(k) for k in PINS}
    if found != PINS or sys.version_info[:2] != (3, 11):
        raise ValueError(f'Fixed package gate failed: {found}')
    return found


def select(strata):
    import numpy as np
    result = []
    for label in ('left', 'right'):
        for correct in (False, True):
            pool = sorted([s for s in strata if s['answer'] == label and s['historical_correct'] == correct],
                          key=lambda s: (abs(s['historical_full_margin']), s['question_id']))
            if len(pool) < 4:
                raise ValueError('Insufficient frozen historical stratum')
            for bin_index, indices in enumerate(np.array_split(np.arange(len(pool)), 4)):
                bin_rows = [pool[int(i)] for i in indices]
                center = float(np.median([abs(s['historical_full_margin']) for s in bin_rows]))
                chosen = min(bin_rows, key=lambda s: (not s['matched_pair'], not bool(s['pair']),
                             abs(abs(s['historical_full_margin'])-center), s['question_id']))
                result.append(dict(chosen, selection_stratum=f'{label}_{correct}', selection_rank_bin=bin_index,
                                   selection_bin_question_ids=[s['question_id'] for s in bin_rows]))
    if len({s['img_id'] for s in result}) != 16:
        raise ValueError('Pilot requires 16 distinct frozen images')
    return result


def cpu_input(processor, row, image_path, image_token_id=32000):
    from PIL import Image
    from attention_position_probe import render_prompt, locate_question_tokens
    from physical_exit_pilot import candidate_inputs
    with Image.open(image_path) as image:
        data = processor(images=image.convert('RGB'), text=render_prompt(processor, row['question']), return_tensors='pt')
    ids = data['input_ids'][0].tolist()
    q, audit = locate_question_tokens(processor, row['question'], ids, image_token_id)
    candidates = candidate_inputs(processor, data, ['left', 'right'])
    records = [r for _, r in candidates]
    if any(len(r) != 1 or r[0]['query_position'] != len(ids)-1 or r[0]['target_token_position'] != len(ids) for r in records):
        raise ValueError('Unexpected original candidate mapping')
    return dict(prompt_ids=ids, question_mapping=q, question_audit=audit, answer_query=len(ids)-1,
                candidate_mapping=[dict(label=label,input_ids=x['input_ids'][0].tolist(),answer_records=r)
                                   for label,(x,r) in zip(['left','right'],candidates)],
                visual_indices=[i for i,t in enumerate(ids) if t==image_token_id])


def prepare(model_path, image_root):
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    from attention_position_probe import render_prompt
    from roi_sink_common import geometry, map_box
    from prepare_fixed_k_data import digest as source_row_digest
    packages()
    if OUTPUT.exists():
        raise FileExistsError('Never replace a frozen experiment directory')
    if git('diff', 'HEAD', '--', '*.py'):
        raise ValueError('Commit implementation before freezing inputs')
    subprocess.run(['git','merge-base','--is-ancestor',BASELINE,'HEAD'],cwd=ROOT,check=True)
    processor = AutoProcessor.from_pretrained(str(model_path),local_files_only=True)
    image_token_id=read(model_path/'config.json')['image_token_index']
    rows = list(csv.DictReader(SOURCE.open()))
    lookup = {r['question_id']:r for r in rows}
    history = {r['question_id']:r for r in csv.DictReader((HISTORY/'per_sample_results.csv').open()) if r['cutoff']=='24'}
    historical_manifest = read(HISTORY/'manifest.json')
    if sha(SOURCE) != historical_manifest['source_csv_sha256']:
        raise ValueError('Historical source CSV changed')
    by_image = {}
    for r in rows:
        by_image.setdefault(r['img_id'],[]).append(r)
    object_lookup = {}
    def annotation(r,g):
        if r['answer'] not in ('left','right') or r['answer'] != r['true option']:
            raise ValueError('Invalid source answer')
        operations = ast.literal_eval(r['operations'])
        if f"({r['object2 id']})" not in operations[-1]['argument'] or operations[-1]['operation'] != 'choose rel':
            raise ValueError('Source subject/object mapping is not supported')
        objects=[]
        for j in (1,2):
            box=[float(r[f'object{j} {k}']) for k in ('x','y','w','h')]
            key=(r['img_id'],r[f'object{j} id'])
            if key in object_lookup and object_lookup[key]!=box:
                raise ValueError('Conflicting same-image object boxes in source')
            object_lookup[key]=box
            objects.append(dict(map_box(box,g),object_id=r[f'object{j} id'],name=r[f'object{j} name'],role='anchor' if j==1 else 'subject'))
        return dict(question_id=r['question_id'],img_id=r['img_id'],question=r['question'],answer=r['answer'],
                    source_row_sha256=source_row_digest(r),source_row=r,objects=objects)
    def prompt_ids(r):
        return processor.tokenizer(render_prompt(processor,r['question']))['input_ids']
    pool=[];pair_available=0
    for s in sorted(historical_manifest['samples'],key=lambda s:s['question_id']):
        original=lookup[s['question_id']]
        if source_row_digest(original)!=s['source_row_sha256']:
            raise ValueError('Original source row mismatch')
        path=image_root/f"image_{s['img_id']}.jpg"
        if sha(path)!=s['image_sha256']:
            raise ValueError('Original image hash mismatch')
        with Image.open(path) as im:
            im=im.convert('RGB')
            if list(im.size)!=s['source_image_size']:
                raise ValueError('Source image dimensions mismatch')
            g,_=geometry(im,processor.image_processor)
        primary=annotation(original,g);first_ids=prompt_ids(original)
        pairs=[]
        for r in by_image[s['img_id']]:
            if r['question']==s['question'] or r['object2 id']==original['object2 id']:
                continue
            other=annotation(r,g)
            a,b=primary['objects'][1],other['objects'][1]
            if min(a['patch_count'],b['patch_count'])==0 or a['patch_indices']==b['patch_indices']:
                continue
            ids=prompt_ids(r)
            pairs.append((abs(len(first_ids)-len(ids)),r['question_id'],other))
        pairs.sort(key=lambda t:t[:2]);pair=pairs[0][2] if pairs else None
        matched=bool(pair and len(first_ids)==len(prompt_ids(pair)) and first_ids[-1]==prompt_ids(pair)[-1])
        pair_available+=bool(pair)
        margin=float(history[s['question_id']]['full_margin'])
        pool.append(dict(primary,original_sample=s,image_path=str(path),image_sha256=s['image_sha256'],geometry=g,
                         historical_full_margin=margin,historical_correct=margin>0,pair=pair,matched_pair=matched))
    chosen=select(pool)
    for s in chosen:
        s['input_mapping']=cpu_input(processor,s,s['image_path'],image_token_id)
        old=read(HISTORY/'checkpoints/run'/f"{s['question_id']}.json")['body']['data']
        if s['input_mapping']['prompt_ids']!=old['prompt_ids'] or s['input_mapping']['question_mapping']!=old['question_mapping']:
            raise ValueError('Historical prompt/question mapping mismatch')
        if any(a['input_ids']!=b['input_ids'] or a['answer_records']!=b['answer_records'] for a,b in zip(s['input_mapping']['candidate_mapping'],old['candidate_mapping'])):
            raise ValueError('Historical candidate mapping mismatch')
        if s['pair']:
            s['pair']['input_mapping']=cpu_input(processor,s['pair'],s['image_path'],image_token_id)
    model_files={}
    for item in read(ROOT/'configs/model.json')['files']:
        p=model_path/item['name'];h=sha(p)
        if p.stat().st_size!=item['bytes'] or (item.get('lfs_sha256') and h!=item['lfs_sha256']):
            raise ValueError('Fixed model file mismatch')
        model_files[item['name']]=h
    inputs=[SOURCE,ROOT/'configs/phase3_1_data/SOURCES.md',HISTORY/'manifest.json',HISTORY/'per_sample_results.csv',ROOT/'configs/model.json',ROOT/'requirements.txt']
    inputs += [HISTORY/'checkpoints/run'/f"{s['question_id']}.json" for s in chosen]
    config=dict(rules=RULES,baseline=BASELINE,code_commit=git('rev-parse','HEAD'),sources=sources(),packages=PINS,
                original_inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},model_path=str(model_path),
                model_files_sha256=model_files,image_root=str(image_root),samples=chosen,
                source_available_pair_images=pair_available,experiment_a_pairs=sum(bool(s['pair']) for s in chosen),
                experiment_a_matched_pairs=sum(s['matched_pair'] for s in chosen),smoke_ids=[chosen[0]['question_id'],next(s['question_id'] for s in chosen if s['pair'])])
    OUTPUT.mkdir(parents=True)
    atomic(OUTPUT/'manifest.json',dict(binding=digest(config),config=config))
    print('FROZEN',digest(config),'B_images',16,'A_pairs',config['experiment_a_pairs'],'matched',config['experiment_a_matched_pairs'],flush=True)


def protocol():
    envelope=read(OUTPUT/'manifest.json');c=envelope['config']
    if digest(c)!=envelope['binding'] or c['rules']!=RULES or sources()!=c['sources'] or git('rev-parse','HEAD')!=c['code_commit']:
        raise ValueError('Frozen protocol/code mismatch')
    for p,h in c['original_inputs_sha256'].items():
        if sha(ROOT/p)!=h:
            raise ValueError('Original source/history changed')
    return c,envelope['binding']
