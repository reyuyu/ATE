import collections
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

from ..accuracy_autoresearch.common import config, source_hashes as inherited_sources
from ..physical_visual_handoff.common import atomic, digest, read, sha, write_csv, TOLERANCES
from ..benchmarks.protocols import MODEL_PATH, SHORT, MC
from ..benchmarks.scoring import generated_score, textvqa_score
from ..ate_fulljoint_vqav2.common import score as vqa_score
from ..closedqa_transfer.common import scored as closed_score, offsets_for

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/c3v2_reverse_writer_pruning'
OLD = Path('/data/lk/ATE-worktrees/ate-closedqa-frozen-transfer/vlm_research/results/ate_closedqa_frozen_transfer')
AUTO = Path('/data/lk/ATE-worktrees/ate-accuracy-autoresearch-6h/vlm_research/results/accuracy_autoresearch_6h')
PHASE = '7ccb44941cb48e901e6e23e96bae6da173fe2ce5'
SEED = 20261105
LABELS = ('C3', 'E1', 'E2-P1', 'E2-P2', 'E2-P3', 'E2-P4', 'E2-S2')
CORE = ('C3', 'E1', 'E2-P2')
ASSISTANT_IDS = [319, 1799, 9047, 13566]
CFG = config('all', 'both', head=1)
DEVICES = [dict(index=0, uuid='GPU-a63c6362-dcfd-4a26-c44b-014b40127c89'),
           dict(index=1, uuid='GPU-b08489af-0a8a-eb48-55ba-8346733af417')]
TASKS = ('scienceqa_img', 'pope_random', 'pope_popular', 'pope_adversarial', 'VQAv2', 'textvqa', 'GQA')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def source_hashes():
    files = [p for p in Path(__file__).parent.glob('*.py') if p.name != 'analyze.py']
    files += [ROOT/'vlm_research'/p for p in ('closedqa_transfer/engine.py', 'closedqa_transfer/common.py',
        'benchmarks/scoring.py', 'benchmarks/protocols.py', 'ate_fulljoint_vqav2/common.py')]
    return dict(inherited_sources(), **{str(p.relative_to(ROOT)): sha(p) for p in files})


def writer_set(mapping, label):
    assert label in LABELS
    blocked = set() if label == 'C3' else set(mapping['parts']['Instruction'])
    a = mapping['parts']['ASSISTANT']
    if label.startswith('E2-P'):
        blocked.update(a[:int(label[-1])])
    elif label == 'E2-S2':
        blocked.update(a[-2:])
    assert blocked <= set(mapping['R'])
    assert blocked.isdisjoint(mapping['parts']['Q'] + mapping['parts']['Hint'] + mapping['parts']['Options'])
    assert mapping['Pminus1'] not in blocked
    return sorted(set(mapping['R']) - blocked)


def localize(processor, s):
    payload = s['prompt_text']
    instruction = MC if s['dataset'] == 'scienceqa_img' else SHORT
    assert payload.endswith(instruction)
    chat = processor.apply_chat_template([dict(role='user', content=[dict(type='image'), dict(type='text', text=payload)])], add_generation_prompt=True)
    text = chat.replace('<image>', '<image>'*576)
    e = processor.tokenizer(text, return_offsets_mapping=True)
    assert e['input_ids'] == s['input_mapping']['prompt_ids'], 'Historical input token mismatch'
    start = text.index(payload)
    qstart = s['input_mapping'].get('Q_char_span', [start])[0]
    assert text[qstart:qstart+len(s['question'])] == s['question']
    istart = start + len(payload) - len(instruction)
    astart = text.rindex('ASSISTANT')
    spans = dict(Q=[qstart, qstart+len(s['question'])], Instruction=[istart, istart+len(instruction)],
                 ASSISTANT=[astart, astart+len('ASSISTANT')])
    parts = {name: offsets_for(e, lo, hi, text) for name, (lo, hi) in spans.items()}
    for name, (lo, hi) in spans.items():
        for c in range(lo, hi):
            assert text[c].isspace() or any(e['offset_mapping'][i][0] <= c < e['offset_mapping'][i][1] for i in parts[name]), (name, c)
    assert [e['input_ids'][i] for i in parts['ASSISTANT']] == ASSISTANT_IDS
    for name in ('Hint', 'Options'):
        parts[name] = s['input_mapping'].get('parts', {}).get(name, [])
    image = [i for i, v in enumerate(e['input_ids']) if v == 32000]
    r = list(range(image[-1]+1, len(e['input_ids'])-1))
    assert len(image) == 576 and set(sum(parts.values(), [])) <= set(r)
    assert set(parts['Instruction']+parts['ASSISTANT']).isdisjoint(parts['Q']+parts['Hint']+parts['Options'])
    return dict(prompt_ids=e['input_ids'], image=image, R=r, Pminus1=len(e['input_ids'])-1,
        parts=parts, char_spans=spans, offsets=e['offset_mapping'], rendered_prompt=chat,
        prompt_text=payload, expanded_prompt_sha256=hashlib.sha256(text.encode()).hexdigest())


def freeze():
    from PIL import Image
    from transformers import AutoProcessor
    assert not (OUT/'execution_freeze.json').exists(), 'Do not overwrite frozen protocol'
    p = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    auto = read(AUTO/'sample_manifest.json')
    closed = read(OLD/'sample_manifest.json')
    samples = {}; used = set()
    for task in sorted({s['task'] for s in auto.values() if s['dataset']=='GQA'}):
        options = sorted(((k,s) for k,s in auto.items() if s['dataset']=='GQA' and s['task']==task),
                         key=lambda x: digest([SEED, 'GQA600', x[1]['image_sha256'], x[1]['question_id']]))
        chosen = []
        for k,s in options:
            if s['image_sha256'] not in used:
                chosen.append((k,s)); used.add(s['image_sha256'])
            if len(chosen)==100: break
        assert len(chosen)==100
        samples.update(chosen)
    samples.update({k:s for k,s in auto.items() if s['dataset']=='VQAv2'})
    samples.update(closed)
    raw = subprocess.check_output(['git','show',PHASE+':vlm_research/results/phase4_0_dataset_suite/manifests/textvqa.json'], cwd=ROOT)
    textm = json.loads(raw)
    assert len(textm['samples'])==500, 'Do not silently substitute missing assets'
    for s in textm['samples']:
        previous = s['metadata']['input_mapping']
        assert len(s['answer']) == 10
        samples['textvqa__'+s['question_id']] = dict(s, task='textvqa', prompt_text=previous['text_payload'], input_mapping=previous)
    assert len(samples)==3000
    images = {}; audits = []; inputs = []
    for k, original in list(samples.items()):
        s = dict(original)
        if s['image_path'] not in images:
            assert str(Path(s['image_path']).resolve()).startswith('/data/')
            images[s['image_path']] = sha(s['image_path'])
            with Image.open(s['image_path']) as im: im.verify()
        assert images[s['image_path']] == s['image_sha256']
        m = localize(p,s)
        if 'historical_input_hashes' in s: inputs.append(k)
        # Keep only needed annotation and binding fields; no historical arrays/model copies.
        keep = ('dataset','task','question_id','image_id','image_path','image_sha256','question','answer','answers',
                'answer_type','question_type','candidates','gold_index','subject','choice_count','source_row_sha256')
        s = {f:s[f] for f in keep if f in s}
        s.update(prompt_text=m['prompt_text'], input_mapping=m,
                 history_sample_sha256=digest(original), history_pool='Phase4 TextVQA' if s['dataset']=='textvqa' else '801d closedqa' if k in closed else '60d autoresearch')
        s['prompt_sha256'] = digest(m['prompt_ids'])
        s['answer_type'] = s.get('answer_type', s.get('subject') or s['task'])
        samples[k] = s
        for label in LABELS:
            w = writer_set(m,label)
            audits.append(dict(sample=k,task=s['task'],condition=label,question_id=s['question_id'],
                instruction_positions=m['parts']['Instruction'],instruction_ids=[m['prompt_ids'][i] for i in m['parts']['Instruction']],
                instruction_tokens=len(m['parts']['Instruction']),assistant_positions=m['parts']['ASSISTANT'],assistant_ids=ASSISTANT_IDS,
                assistant_retained=sum(i in w for i in m['parts']['ASSISTANT']),writers=len(w),R=len(m['R']),writer_ratio=len(w)/len(m['R']),
                blocked_R=sorted(set(m['R'])-set(w)),Q_preserved=set(m['parts']['Q'])<=set(w),
                hint_options_preserved=set(m['parts']['Hint']+m['parts']['Options'])<=set(w),Pminus1=m['Pminus1'],prompt_sha256=s['prompt_sha256']))
    groups = collections.defaultdict(list)
    for k,s in samples.items(): groups[s['image_sha256']].append(k)
    shard = {}; loads = [0,0]
    for h,keys in sorted(groups.items(), key=lambda kv:(-len(kv[1]),digest([SEED,'shard',kv[0]]))):
        rank = min(range(2),key=lambda i:(loads[i],i)); shard[h]=rank; loads[rank]+=len(keys)
    canonical = {}; aliases = {}
    for k,s in sorted(samples.items()):
        identity = digest([s['image_sha256'],s['input_mapping']['prompt_ids'],s['input_mapping']['parts']])
        aliases[k]=canonical.setdefault(identity,k)
    ordered = sorted(samples,key=lambda k:digest([SEED,'smoke',samples[k]['image_sha256'],k]))
    smoke = []
    for dataset in ('scienceqa_img','pope','VQAv2','textvqa','GQA'):
        candidates = [k for k in ordered if samples[k]['dataset']==dataset]
        if dataset=='scienceqa_img':
            candidates.sort(key=lambda k: (-bool(samples[k]['input_mapping']['parts']['Hint']), -samples[k].get('choice_count',0)))
        smoke.append(candidates[0])
    explore=[]
    strata=sorted({s['task'] for s in samples.values()})
    targets={t:5 if t not in TASKS else 20 if t in ('scienceqa_img','VQAv2','textvqa') else 10 for t in strata}
    for t,n in targets.items():
        ks=sorted((k for k,s in samples.items() if s['task']==t),key=lambda k:digest([SEED,'explore120',k]))
        explore+=ks[:n]
    assert len(explore)==120
    weights = read(AUTO/'execution_freeze.json')['weights']
    assert all(sha(path)==h for path,h in weights.items())
    atomic(OUT/'sample_manifest.json',samples)
    atomic(OUT/'partition.json',dict(image_shard=shard,loads=loads,canonical_input=aliases,smoke=smoke,explore120=explore))
    write_csv(OUT/'writer_mapping_audit.csv',audits)
    atomic(OUT/'protocol.json',dict(created_at=now(),started_at='2026-11-05T15:47:43+08:00',
        deliver_by='2026-11-05T21:47:43+08:00',stop_new_gpu_work_by='2026-11-05T21:17:43+08:00',
        devices=DEVICES,authorization_change='User explicitly authorized GPU0 and GPU1 after GPU2/3 were found occupied.',
        labels=LABELS,config=CFG,seed=SEED,max_new_tokens=32,logits_to_keep=1,
        instruction=dict(short=SHORT,scienceqa=MC),assistant_ids=ASSISTANT_IDS,
        mask='Only removed R -> original Image connections in Prefill L0-L19. No token deletion; dense attention still executes.',
        inherited='C3-V2: Image rows retained before L20; physical exit before L20; P1+Decode no image; physical answer-only Attention bypass L2-L5.',
        selection='GQA: stable input/image hash, 100/task and image unique; all other cohorts unchanged; no outcome filtering.',
        primary_family=dict(comparisons=['E1-C3','E2-P2-E1'],datasets=TASKS,size=14,
            method='Bonferroni14 two-sided cluster bootstrap intervals, primary Accuracy; ordinary 95% intervals also shown'),
        bootstrap=dict(replicates=20000,cluster='Image SHA256',weighting='Equal questions, not equal clusters'),
        fallback='Core C3/E1/E2-P2 all datasets first, then other conditions on frozen 120, then full cohort. Stop new GPU work 30 minutes before deadline.',
        exploratory=True,no_new_physical_acceleration=True,historical_images_previously_studied=True))
    atomic(OUT/'history_binding_audit.json',dict(auto_manifest_sha256=sha(AUTO/'sample_manifest.json'),
        closedqa_manifest_sha256=sha(OLD/'sample_manifest.json'),textvqa_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        model_weights_rehashed=True,image_paths=len(images),image_clusters=len(groups),samples=len(samples),
        counts=dict(collections.Counter(s['dataset'] for s in samples.values())),all_selected_questions_previously_frozen=True,
        history_overlap='All 3000 selected questions and their images are from previously studied cohorts; not an independent external test.'))
    atomic(OUT/'execution_freeze.json',dict(at=now(),sources=source_hashes(),weights=weights,devices=DEVICES,
        sample_manifest_sha256=sha(OUT/'sample_manifest.json'),partition_sha256=sha(OUT/'partition.json'),
        protocol_sha256=sha(OUT/'protocol.json'),tolerances=TOLERANCES))
    print('FROZEN',len(samples),'questions',len(groups),'image clusters',loads,flush=True)


def scored(s,text):
    if s['dataset']=='GQA':
        return dict(score=float(text.rstrip('.').lower()==s['answer']),parsed=text,parse_failed=False)
    if s['dataset']=='VQAv2':
        return dict(score=float(vqa_score(text,s['answers'])),parsed=text,parse_failed=False)
    if s['dataset']=='textvqa':
        return dict(score=float(textvqa_score(text,s['answer'])),parsed=text,parse_failed=False)
    return closed_score(s,text)


def checkpoint(key,label):
    return OUT/'checkpoints'/key/(label+'.json')


def record(key,label,sample):
    p=checkpoint(key,label)
    if not p.exists(): return None
    b=read(p); r=b['payload']
    assert b['sha256']==digest(r) and r['sample_sha256']==digest(sample)
    assert r['sources']==source_hashes() and r['condition']==label and r['config']==CFG
    assert scored(sample,r['generation']['text'])==r['scoring']
    return r
