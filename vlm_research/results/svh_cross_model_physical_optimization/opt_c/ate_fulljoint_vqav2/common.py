import hashlib
import re
import shutil
import subprocess
import zipfile
import json
from pathlib import Path

from ..physical_visual_handoff.common import ROOT, MODELS, TOLERANCES, atomic, digest, read, sha, write_csv
from ..fastp2_bypass_writer.common import hardware
from ..receiver_cross_dataset.official_vqa_eval import VQAEval
from .engine import CONDITIONS

OUT = ROOT / 'vlm_research/results/ate_fulljoint_vqav2_quick'
HISTORY = Path('/data/lk/ATE-worktrees/single-gpu-receiver-cross-dataset-validation')
DATA = HISTORY / 'vlm_research/results/receiver_cross_dataset_validation/vqa_v2'
RULE_COMMIT = '644834d3e22194b4982758dabd59d3e479ef8ce3'
SEED = 20261105
EXECUTION_FILES = ('__init__.py', 'common.py', 'engine.py', 'run.py', 'tests.py')


class EmptyVQA:
    def getQuesIds(self):
        return []


EVALUATOR = VQAEval(EmptyVQA(), EmptyVQA())


def normalize(text):
    return EVALUATOR.processDigitArticle(EVALUATOR.processPunctuation(
        text.replace('\n', ' ').replace('\t', ' ').strip()))


def score(text, answers):
    assert len(answers) == 10
    pred = normalize(text)
    answers = [normalize(a) for a in answers]
    return sum(min(1., sum(pred == a for j, a in enumerate(answers) if j != i) / 3.)
               for i in range(10)) / 10


def conditional_writers(question, question_positions, assistant):
    words = re.findall('[a-z]+', question.lower())
    triggered = (words[0] if words else '') in ('what', 'which') or len(question_positions) >= 15
    return sorted(set(question_positions) | (set(assistant) if triggered else set())), triggered


def sources():
    files = [Path(__file__).parent / n for n in EXECUTION_FILES]
    files += [ROOT / 'vlm_research' / rel for rel in (
        'fastp2_bypass_writer/engine.py', 'fastp2_bypass_writer/run.py',
        'physical_handoff_fastpath/engine.py', 'physical_handoff_fastpath/ledger.py',
        'physical_visual_handoff/engine.py', 'physical_visual_handoff/common.py',
        'p1_layerwise_accuracy/engine.py', 'semantic_visual_writers/groups.py',
        'receiver_cross_dataset/official_vqa_eval.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def verify():
    p = read(OUT / 'protocol.json')
    assert sources() == p['sources'], 'Frozen source changed'
    assert sha(OUT / 'sample_manifest.json') == p['manifest_sha256']
    assert sha(OUT / 'input_specs.json') == p['specs_sha256']
    return p


def git_binding(commit, relative, path=None):
    data = subprocess.check_output(['git', 'show', commit + ':' + relative], cwd=ROOT)
    h = hashlib.sha256(data).hexdigest()
    if path is not None:
        assert h == sha(path), relative
    return dict(commit=commit, path=relative, sha256=h)


def freeze():
    import torch
    from transformers import AutoProcessor
    from PIL import Image
    from ..semantic_visual_writers.groups import token_groups, intact
    from ..p1_layerwise_accuracy.engine import tensor_hash

    assert not torch.cuda.is_initialized() and not (OUT / 'protocol.json').exists()
    assert str(ROOT).startswith('/data/') and shutil.disk_usage('/data').free > 20 * 1024**3
    relative = 'vlm_research/results/receiver_cross_dataset_validation/vqa_v2/sample_manifest.json'
    bindings = [git_binding('b483cdc524', relative, DATA / 'sample_manifest.json'),
                git_binding(RULE_COMMIT, 'vlm_research/semantic_qrelay_search3h/refinements.py'),
                git_binding(RULE_COMMIT, 'vlm_research/results/semantic_single_gpu_search3h/final_report_zh.md'),
                git_binding('b483cdc524', 'vlm_research/receiver_cross_dataset/official_vqa_eval.py',
                            ROOT / 'vlm_research/receiver_cross_dataset/official_vqa_eval.py')]
    winner = json.loads(subprocess.check_output(['git', 'show', RULE_COMMIT +
        ':vlm_research/results/semantic_single_gpu_search3h/phase6_fixed_compositions/winner_freeze.json'], cwd=ROOT))
    assert winner['config'] == dict(base='Q', hit='QSA', rule='whatwhich|long15')
    manifest = read(DATA / 'sample_manifest.json')
    assert manifest['status'] == 'FROZEN' and len(manifest['rows']) == 500
    pool = sorted(manifest['rows'], key=lambda s: digest([SEED, 'vqav2-fulljoint60', s['question_id'], s['image_id']]))
    chosen, seen = [], set()
    for s in pool:
        if s['image_sha256'] in seen:
            continue
        seen.add(s['image_sha256'])
        chosen.append(s)
        if len(chosen) == 60:
            break
    assert len(chosen) == len(seen) == 60
    archive = Path('/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation/sources/vqa_annotations.zip')
    with zipfile.ZipFile(archive) as z:
        annotations = json.loads(z.read('v2_mscoco_val2014_annotations.json'))['annotations']
    qids = {s['question_id'] for s in chosen}
    annotations = {str(r['question_id']): r for r in annotations if str(r['question_id']) in qids}
    model = Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf') / MODELS['llava']['revision']
    proc = AutoProcessor.from_pretrained(model, local_files_only=True)
    proc.patch_size, proc.num_additional_image_tokens = 14, 1
    proc.vision_feature_select_strategy, proc.tokenizer.padding_side = 'default', 'left'
    samples, specs = {}, {}
    for row in chosen:
        s = dict(row)
        q = s['question_id']
        annotation = annotations[q]
        assert str(annotation['image_id']) == s['image_id']
        assert [a['answer'] for a in annotation['answers']] == s['answers']
        assert sha(s['image_path']) == s['image_sha256']
        assert str(Path(s['image_path']).resolve()).startswith('/data/')
        s.update(dataset='VQAv2-val2014', task='VQAv2', answer=annotation['multiple_choice_answer'],
                 prompt_text=s['question'] + '\nAnswer the question using a single word or phrase.')
        chat = proc.apply_chat_template([dict(role='user', content=[dict(type='image'),
            dict(type='text', text=s['prompt_text'])])], add_generation_prompt=True)
        with Image.open(s['image_path']) as im:
            inputs = proc(images=im.convert('RGB'), text=chat, return_tensors='pt')
        inputs['pixel_values'] = inputs['pixel_values'].half()
        ids = inputs['input_ids'][0].tolist()
        image = [i for i, v in enumerate(ids) if v == 32000]
        relay = list(range(image[-1] + 1, len(ids) - 1))
        groups, question, _ = token_groups(proc.tokenizer, s['question'], chat, ids, image, relay)
        assistant = groups[-1]['positions']
        assert groups[-1]['words'] == ['ASSISTANT'] and len(assistant) == 4
        assert [ids[i] for i in assistant] == [319, 1799, 9047, 13566]
        writers, triggered = conditional_writers(s['question'], question, assistant)
        assert intact(writers, groups) and len(image) == 576 and ids[-1] == 29901
        old = read(DATA / 'input_specs' / (q + '.json'))
        assert old['prompt_ids'] == ids and old['Q'] == question and old['ASSISTANT'] == assistant
        assert old['writers']['Threshold-15'] == writers and old['prompt_text'] == chat
        input_hashes = {k: tensor_hash(v) for k, v in inputs.items()}
        assert input_hashes == old['input_hashes']
        s['input_mapping'] = dict(prompt_ids=ids)
        specs[q] = dict(Q=question, ASSISTANT=assistant, R=relay, I=image, writers=writers,
                        triggered=triggered, prompt_chat=chat, input_hashes=input_hashes,
                        historical_spec_sha256=sha(DATA / 'input_specs' / (q + '.json')))
        samples[q] = s
    state = hardware()
    devices = []
    for line in state['gpus'].splitlines():
        index, uuid, name, used, free, util = [v.strip() for v in line.split(',')]
        if int(index) in (2, 3) and '3090' in name and uuid not in state['processes'] and int(used) < 200 and int(free) > 22000 and int(util) == 0:
            devices.append(dict(index=int(index), uuid=uuid, name=name))
    assert devices, 'No idle RTX3090; do not poll or preempt'
    performance = sorted(samples, key=lambda q: digest([SEED, 'performance20', q]))[:20]
    smoke = []
    for predicate in (lambda q: specs[q]['triggered'], lambda q: not specs[q]['triggered'], lambda q: len(specs[q]['Q']) >= 15):
        options = [q for q in sorted(samples) if predicate(q) and q not in smoke]
        if options:
            smoke.append(options[0])
    smoke += [q for q in sorted(samples) if q not in smoke][:3 - len(smoke)]
    weights = read(ROOT / 'vlm_research/results/fastp2_attention_bypass_sparse_writer/protocol.json')['weights']
    assert all(sha(path) == expected for path, expected in weights.items())
    atomic(OUT / 'sample_manifest.json', samples)
    atomic(OUT / 'input_specs.json', specs)
    atomic(OUT / 'gpu_preflight.json', state)
    atomic(OUT / 'protocol.json', dict(seed=SEED, model=MODELS['llava'], sources=sources(), weights=weights,
        manifest_sha256=sha(OUT / 'sample_manifest.json'), specs_sha256=sha(OUT / 'input_specs.json'),
        history=bindings, annotations_sha256=sha(archive), conditions=CONDITIONS, device=devices[0],
        writer_rule='Fixed-Q-Conditional-Assistant-15', handoff_layer=20, A2_layers=[2, 3, 4, 5],
        question_count=60, unique_images=60, performance_ids=performance, smoke_ids=smoke,
        sampling='Stable input-only hash from historical frozen 500 local VQAv2 validation questions; one SHA per image; not sampled from all VQAv2',
        greedy=True, max_new_tokens=32, logits_to_keep=0, warmups=3, repetitions=5,
        numerical_tolerances=TOLERANCES['llava'], bootstrap_seed=SEED, bootstrap_replicates=20000,
        exploratory=True, scoring='Pinned official VQAEval normalization + exact 10-annotator leave-one-out consensus',
        timing='Frozen Fast-P2 CUDA phases; synchronized cold-request wall E2E, no visual reuse; I/O and Processor excluded for all conditions',
        dense_reference='Full 32-layer full sequence computation and full visual KV with D1 source masks plus zero-output I+A reference at L2-L5; smoke only',
        no_fixed_length_benchmark=True, no_new_dataset_download=True))
    assert not torch.cuda.is_initialized()
    print('FROZEN', len(samples), 'quality', len(performance), 'performance', devices[0], flush=True)


if __name__ == '__main__':
    freeze()
