import collections
import importlib.util
import subprocess
from pathlib import Path

from ..physical_visual_handoff.common import ROOT, MODELS, atomic, digest, read, sha, key, TOLERANCES
from ..physical_handoff_fastpath.engine import Layout
from ..semantic_visual_writers.groups import token_groups, select, RANDOM_SEEDS

OUT = ROOT / 'vlm_research/results/fastp2_attention_bypass_sparse_writer'
BASE = '78a7436d39480d0400351d3becda63992409ee17'
WORKTREES = Path('/data/lk/ATE-worktrees')
TEXT_ROOT = WORKTREES / 'single-gpu-receiver-cross-dataset-validation/vlm_research/results/receiver_cross_dataset_validation/textvqa'
HISTORY = {
    'ia': ('8019c6009b', WORKTREES / 'query-role-ia-joint-240', 'vlm_research/results/query_role_ia_joint_240/sample_manifest.json'),
    'textvqa': ('b483cdc524', WORKTREES / 'single-gpu-receiver-cross-dataset-validation', 'vlm_research/results/receiver_cross_dataset_validation/textvqa/sample_manifest.json'),
    'semantic': ('01c82b1d39360fc73bca3d10d12886672ffee370', WORKTREES / 'gqa3000-space-nl-visual-writers', 'vlm_research/results/gqa3000_token_visual_writing/protocol.json'),
    'residual': ('fa959891fb', WORKTREES / 'residual-only-pilot', 'vlm_research/results/l2_l4_residual_only_pilot/report.md'),
    'frontier': ('2a45dcdfb6', WORKTREES / 'sparse-visual-writing-frontier-240', 'vlm_research/results/sparse_visual_writing_frontier_240/report_zh.md'),
}


def sources():
    files = [Path(__file__).parent/n for n in ('__init__.py','engine.py','common.py','run.py','tests.py')]
    files += [ROOT / 'vlm_research' / rel for rel in (
        'physical_handoff_fastpath/engine.py', 'physical_handoff_fastpath/ledger.py',
        'semantic_visual_writers/groups.py', 'p1_layerwise_accuracy/engine.py',
        'physical_visual_handoff/common.py', 'benchmarks/_vendor/m4c_evaluator.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def hardware():
    return dict(gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu',
                                             '--format=csv,noheader,nounits'], text=True),
                processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                                   '--format=csv,noheader,nounits'], text=True),
                disk=subprocess.check_output(['df', '-h', '/data'], text=True))


def verify():
    p = read(OUT / 'protocol.json')
    assert p['sources'] == sources(), 'Frozen source changed'
    assert p['manifest_sha256'] == sha(OUT/'sample_manifest.json')
    assert p['specs_sha256'] == sha(OUT/'input_specs.json')
    return p


def selection(processor, sample, ids):
    chat = processor.apply_chat_template([dict(role='user', content=[dict(type='image'),
            dict(type='text', text=sample['prompt_text'])])], add_generation_prompt=True)
    image = [i for i, v in enumerate(ids) if v == 32000]
    assert len(image) == 576
    relay = list(range(image[-1]+1, len(ids)-1))
    groups, qpos, _ = token_groups(processor.tokenizer, sample['question'], chat, ids, image, relay)
    return select(groups, qpos, relay, sample['question'])


def score(sample, text):
    if sample['dataset'] != 'textvqa300':
        return float(text.strip().rstrip('.').lower() == sample['answer'].strip().lower())
    path = ROOT/'vlm_research/benchmarks/_vendor/m4c_evaluator.py'
    assert sha(path) == '8b49e1033e29d551c0bdd86d9149e2b54370e4cd17e52176ac9061b3521f0774'
    spec = importlib.util.spec_from_file_location('frozen_m4c', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator = module.TextVQAAccuracyEvaluator()
    return float(evaluator._compute_answer_scores(sample['answers']).get(evaluator.answer_processor(text), 0.))


def freeze():
    import os
    import shutil
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    assert str(ROOT).startswith('/data/') and shutil.disk_usage('/data').free > 20*1024**3
    assert not (OUT/'protocol.json').exists(), 'Already frozen; do not replace'
    OUT.mkdir(parents=True, exist_ok=True)
    historical = {}
    for label, (commit, tree, rel) in HISTORY.items():
        blob = subprocess.check_output(['git', 'show', commit+':'+rel], cwd=ROOT)
        import hashlib
        assert hashlib.sha256(blob).hexdigest() == sha(tree/rel), label
        historical[label] = dict(commit=subprocess.check_output(['git', 'rev-parse', commit], cwd=ROOT, text=True).strip(),
                                 path=rel, sha256=sha(tree/rel))
    # The exact historical selector is shared, not redesigned for this experiment.
    assert sha(ROOT/'vlm_research/semantic_visual_writers/groups.py') == sha(
        HISTORY['semantic'][1]/'vlm_research/semantic_visual_writers/groups.py')
    pilot = read(ROOT/'vlm_research/results/physical_handoff_fastpath_tflops/llava/sample_manifest.json')
    gqa = read(HISTORY['ia'][1]/HISTORY['ia'][2])['samples']
    text = read(TEXT_ROOT/'sample_manifest.json')['rows']
    cohorts = dict(pilot60=pilot['samples'], gqa240=gqa, textvqa300=[])
    for row in text:
        s = dict(row, task='TextVQA', answer='',
                 prompt_text=row['question']+'\nAnswer the question using a single word or phrase.')
        old_spec = read(TEXT_ROOT/'input_specs'/(row['question_id']+'.json'))
        s['input_mapping'] = dict(prompt_ids=old_spec['prompt_ids'])
        cohorts['textvqa300'].append(s)
    assert [len(cohorts[k]) for k in cohorts] == [60, 240, 300]
    assert all(v == 40 for v in collections.Counter(s['task'] for s in gqa).values())
    model = Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf')/MODELS['llava']['revision']
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    processor.patch_size = 14
    processor.num_additional_image_tokens = 1
    processor.vision_feature_select_strategy = 'default'
    processor.tokenizer.padding_side = 'left'
    specs, all_samples = {}, {}
    for dataset, cohort in cohorts.items():
        for original in cohort:
            s = dict(original, dataset=dataset)
            sample_key = dataset+'/'+key(s)
            assert sha(s['image_path']) == s['image_sha256'] and str(Path(s['image_path']).resolve()).startswith('/data/')
            chat = processor.apply_chat_template([dict(role='user', content=[dict(type='image'),
                    dict(type='text', text=s['prompt_text'])])], add_generation_prompt=True)
            with Image.open(s['image_path']) as im:
                inputs = processor(images=im.convert('RGB'), text=chat, return_tensors='pt')
            inputs['pixel_values'] = inputs['pixel_values'].half()
            ids = inputs['input_ids'][0].tolist()
            assert ids == s['input_mapping']['prompt_ids'], sample_key
            if dataset == 'textvqa300':
                assert chat == read(TEXT_ROOT/'input_specs'/(s['question_id']+'.json'))['prompt_text']
            from ..p1_layerwise_accuracy.engine import tensor_hash
            spec = dict(selection=selection(processor, s, ids),
                        input_hashes={k: tensor_hash(v) for k, v in inputs.items()})
            token_manifest = HISTORY['semantic'][1]/'vlm_research/results/gqa3000_token_visual_writing/token_manifests'/(key(s)+'.json')
            if dataset != 'textvqa300' and token_manifest.exists():
                old = read(token_manifest)
                assert old['natural'] == spec['selection']
                spec['historical_token_manifest_sha256'] = sha(token_manifest)
            specs[sample_key], all_samples[sample_key] = spec, s
        print('INPUT_FREEZE', dataset, len(cohort), flush=True)
    state = hardware()
    candidates = []
    for line in state['gpus'].splitlines():
        i, uuid, name, used, free, util = [x.strip() for x in line.split(',')]
        if uuid not in state['processes'] and int(used) < 200 and int(free) > 22000 and int(util) == 0:
            candidates.append(dict(index=int(i), uuid=uuid, name=name))
    assert candidates, 'No idle GPU; stop without polling'
    devices = candidates[:2]
    image_groups = collections.defaultdict(list)
    for k, s in all_samples.items():
        image_groups[s['image_sha256']].append(k)
    totals, assignment = [0]*len(devices), {}
    for im, ks in sorted(image_groups.items(), key=lambda x: (-len(x[1]), x[0])):
        rank = min(range(len(devices)), key=lambda j: (totals[j], j))
        assignment[im] = rank
        totals[rank] += len(ks)
    atomic(OUT/'sample_manifest.json', all_samples)
    atomic(OUT/'input_specs.json', specs)
    atomic(OUT/'gpu_preflight.json', state)
    weights = {}
    old_weights = read(ROOT/'vlm_research/results/physical_handoff_fastpath_tflops/llava/protocol.json')['old_weights_manifest']
    for file, info in old_weights.items():
        actual = sha(file)
        expected = info['sha256'] if isinstance(info, dict) else info
        assert actual == expected
        weights[file] = actual
    atomic(OUT/'protocol.json', dict(base=BASE, sources=sources(), historical=historical, weights=weights,
        devices=devices, assignment=assignment, shard_sample_counts=totals,
        manifest_sha256=sha(OUT/'sample_manifest.json'), specs_sha256=sha(OUT/'input_specs.json'),
        model=MODELS['llava'], handoff_layer=20, bypass_layers=[2,3,4,5], conditions=list(__import__(
            'vlm_research.fastp2_bypass_writer.engine', fromlist=['CONDITIONS']).CONDITIONS),
        random_seeds=RANDOM_SEEDS, warmups=5, repetitions=20, max_new_tokens=32,
        fixed_predictions=[16,64], fixed_input=key(pilot['smoke'][0]), smoke=pilot['smoke'][:3],
        numerical_tolerances=TOLERANCES['llava'], numerical_gate='Historical allclose + hidden RMSE/cosine; unchanged thresholds; exact Native/Noop and A0 baseline',
        quality_bootstrap=20000, bootstrap_seed=20261104, exploratory=True,
        timing='same frozen Fast-P2 cold request setup, CUDA event phases and synchronized wall E2E; no visual replay; profiler outside timing',
        B3='Eager two query groups: image-access queries attend full KV; remaining R and P-1 attend nonimage KV; dense QKV/O and MLP retained',
        hardware_counters='Unavailable unless new ncu probe succeeds; never substitute NVML or shape census'))
    assert not torch.cuda.is_initialized()
    print('FROZEN', totals, devices, flush=True)


if __name__ == '__main__':
    freeze()
