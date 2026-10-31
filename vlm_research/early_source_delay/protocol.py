"""Outcome-blind hash sampling, true Question spans, and immutable reuse bindings."""
import hashlib
import subprocess
from pathlib import Path

from transformers import AutoProcessor
from attention_position_probe import locate_question_tokens
from benchmarks.dataset_registry import digest, freeze_json, sha256
from l0l1_bos.protocol import ROOT, old, read, prior, identity
from paper_rq1_single_layer_gqa6.run import atomic_json

OUT = ROOT / 'results/early_source_delay_240'
SEED = 20261031
GPU = 'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6'
BASE = 'c67bc7d1a9d0603927b1f18b52777d790f4d1d85'
REFERENCES = ['f1e661fd228ec600e5a63b3016d68bd2fde64940', BASE,
    'e495adf4fd93337a3638981bbb66c2bbac81242d']
CONDITIONS = {
    'A1': dict(layer=0, query='last', source='image'),
    'A2': dict(layer=0, query='last', source='question'),
    'A3': dict(layer=0, query='question', source='image'),
    'A4': dict(layer=1, query='last', source='image'),
    'A5': dict(layer=1, query='last', source='question'),
    'A6': dict(layer=1, query='question', source='image'),
    'EarlyOFF': dict(early_off=True),
    'L2Image': dict(layer=2, query='last', source='image'),
    'EarlyOFF_L2Image': dict(early_off=True, layer=2, query='last', source='image'),
    'L2Question': dict(layer=2, query='last', source='question'),
    'EarlyOFF_L2Question': dict(early_off=True, layer=2, query='last', source='question'),
}
KEYS = ('Full', *CONDITIONS)


def selection_hash(sample):
    key = f'early_source_delay_240|{SEED}|{sample["task"]}|{sample["question_id"]}|{sample["image_sha256"]}'
    return hashlib.sha256(key.encode('ascii')).hexdigest()


def select(samples):
    result = []
    for task in old.TASKS:
        pool = sorted((s for s in samples if s['task'] == task), key=lambda s: (selection_hash(s), s['question_id']))
        assert len(pool) == 500
        result += pool[:40]
    return result


def true_mapping(processor, sample):
    text, question = sample['prompt_text'], sample['question']
    assert text.count(question) == 1, 'Ambiguous original Question substring'
    records, audit = locate_question_tokens(processor, text, sample['input_mapping']['prompt_ids'], 32000)
    start = text.index(question); end = start + len(question)
    selected = []
    for row in records:
        left, right = row['question_char_span']
        if left < end and right > start:
            assert start <= left < right <= end, 'Question/template token crosses boundary'
            selected.append(row)
    q = [r['sequence_index'] for r in selected]
    assert processor.tokenizer.decode([r['token_id'] for r in selected], clean_up_tokenization_spaces=False) == question
    ids = sample['input_mapping']['prompt_ids']; n = len(ids)
    image = [i for i, t in enumerate(ids) if t == 32000]
    bos = [i for i, t in enumerate(ids) if t == processor.tokenizer.bos_token_id]
    assert len(image) == 576 and bos == [0] and q and max(q) < n-1
    assert not any(ids[i] in processor.tokenizer.all_special_ids for i in q)
    template = sorted(set(range(n)) - set(image) - set(bos) - set(q))
    assert sum(map(len, [image, bos, q, template])) == n
    return dict(bos=bos, image=image, question=q, template=template,
        query=n-1, question_char_span_in_slot=[start, end], question_token_records=selected,
        exact_question_reconstruction=True, expanded_ids_exact=audit['expanded_ids_exactly_match_processor'])


def historical(sample):
    previous = prior(sample)
    path = ROOT / 'results/gpu2_l0l1_bos_mechanism/checkpoints' / f'{identity(sample)}.json'
    record = read(path)
    assert record['sample_sha256'] == digest(sample) and record['result_sha256'] == digest(record['generation'])
    assert record['condition'] == 'L0L1_OFF'
    full_path = old.baseline_path(sample)
    return previous, record['generation'], dict(full_path=str(full_path.relative_to(ROOT)),
        full_sha256=sha256(full_path), early_p1_decode_path=str(path.relative_to(ROOT)), early_p1_decode_sha256=sha256(path))


def sources():
    paths = list((ROOT/'early_source_delay').glob('*.py'))
    names = ['test_early_source_delay.py', 'attention_position_probe.py', 'directed_suffix.py',
        'knockout_probe.py', 'l0l1_bos/capture.py', 'l0l1_bos/protocol.py', 'l0l1_bos/run.py',
        'paper_rq1/engine.py', 'paper_rq1/scorers.py', 'paper_rq1_early_joint_attention/engine.py',
        'paper_rq1_early_joint_attention/statistics.py', 'paper_rq1_single_layer_gqa6/run.py',
        'first_answer_attention/engine.py', 'first_answer_attention/run.py', 'vlm_common.py']
    paths += [ROOT/n for n in names]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def prepare():
    samples = old.cohort()
    chosen = select(samples)
    # Selection is complete before inspecting historical outcomes.
    processor = AutoProcessor.from_pretrained(old.MODEL_PATH, local_files_only=True)
    manifest = []
    for s in chosen:
        assert sha256(s['image_path']) == s['image_sha256']
        previous, early, reuse = historical(s)
        manifest.append(dict(sample=s, sample_sha256=digest(s), selection_hash=selection_hash(s),
            mapping=true_mapping(processor, s), reuse=reuse))
    freeze_json(OUT/'sample_manifest.json', manifest)
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in
        [old.OUT/'execution_manifest.json', old.OUT/'per_question.csv',
         ROOT/'results/gpu2_l0l1_bos_mechanism/result.json',
         old.previous.ORIGINAL/'sample_manifest.json']}
    reviewed = {}
    for name in ['phase3_7/pathways.py', 'directed_suffix.py', 'knockout_probe.py', 'attention_position_probe.py']:
        raw = subprocess.check_output(['git','show',REFERENCES[0]+':vlm_research/'+name],cwd=ROOT)
        reviewed[name] = hashlib.sha256(raw).hexdigest()
        if not name.startswith('phase3_7/'):
            assert reviewed[name] == sha256(ROOT/name), 'Shared historical mask/mapping primitive changed'
    config = dict(status='frozen', base_commit=BASE, reviewed_commits=REFERENCES,
        phase37_reviewed_sources=reviewed, questions=240, original_questions=3000,
        per_task={t:40 for t in old.TASKS}, images=len({r['sample']['image_sha256'] for r in manifest}),
        seed=SEED, sample_manifest_sha256=sha256(OUT/'sample_manifest.json'), protected_sha256=protected,
        selection='SHA256(early_source_delay_240|seed|task|question_id|image_sha256); lowest40/task; no outcome filter',
        cases=[identity(next(r['sample'] for r in manifest if r['sample']['task']==t)) for t in old.TASKS],
        conditions=CONDITIONS, gpu_uuid=GPU, model_revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
        generation=dict(greedy=True,max_new_tokens=32,eos_token_id=2,independent_kv=True),
        temporal_scope='All interventions original Prefill only. Last=P-1; Question=multiple real Question rows. All Decode normal.',
        mass='P-1 only, layers0..5, disjoint BOS/Image576/trueQuestion/Template including answer instruction',
        full_reuse='240 historical complete answers; native cached Prefill permitted only for row/hidden measurement and exact binding check',
        early_reuse='Historical P-1+Decode OFF first logits only; NEVER reuse its complete answers as P-1-only',
        new_generations_max=2640, bootstrap_repetitions=200000,
        interactions=dict(formula='D_s=(EarlyOFF-EarlyOFF_L2s)-(Full-L2s)',
            direction='Predeclared positive D means increased conditional functional dependence; not presumed positive',
            family=2, intervals='two-sided ordinary95 and Bonferroni family95 (97.5% marginal)',
            unit='whole image SHA256 clusters, question-weighted paired ratio',
            secondary='Same contrasts on first Top1 flip and native-token LogP Loss; descriptive only'),
        exploratory='Six A comparisons, report ordinary95 and within-six Bonferroni95; taskwise descriptive. Degenerate zero flip bootstrap is not equivalence.',
        boundary='Mass redistribution, path causal contribution, functional compensation and visual semantic integration are distinct claims. No H3 mediation/path-composition test.')
    freeze_json(OUT/'protocol.json', config)
    print('Frozen outcome-blind240; true Question offsets and240 original image/full bindings verified',flush=True)


def verify():
    old.verify()
    p = read(OUT/'protocol.json')
    assert p['conditions'] == CONDITIONS and p['sample_manifest_sha256'] == sha256(OUT/'sample_manifest.json')
    for name,h in p['protected_sha256'].items(): assert sha256(ROOT/name)==h,name
    return p


def entries():
    verify()
    return read(OUT/'sample_manifest.json')


def check_reuse(entry):
    s=entry['sample']; assert digest(s)==entry['sample_sha256']
    previous,early,reuse=historical(s)
    assert reuse==entry['reuse']
    return previous,early


def freeze():
    verify(); cpu=read(OUT/'cpu_validation.json')
    assert cpu['passed'] and cpu['source_sha256']==sources()
    assert not subprocess.check_output(['git','diff','HEAD','--','early_source_delay','test_early_source_delay.py'],cwd=ROOT,text=True).strip()
    freeze_json(OUT/'execution_manifest.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=sources(), protocol_sha256=sha256(OUT/'protocol.json'),cpu_sha256=sha256(OUT/'cpu_validation.json')))


def execution():
    verify(); m=read(OUT/'execution_manifest.json')
    assert m['source_sha256']==sources() and m['protocol_sha256']==sha256(OUT/'protocol.json')
    assert m['cpu_sha256']==sha256(OUT/'cpu_validation.json')
    return m
