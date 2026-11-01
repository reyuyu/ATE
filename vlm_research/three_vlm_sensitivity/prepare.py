"""CPU-only shared data and per-architecture official input freezing."""
import argparse
import importlib.util
import importlib.metadata
import sys
from collections import Counter

from .common import (OUT, LLAVA, QWEN, INTERN, MODELS, TASKS, ROOT, model_path, read, write,
                     sha256, digest, source_hashes)


def shared():
    assert not (OUT / 'shared_3000_manifest.json').exists()
    source = LLAVA / 'sample_manifest.json'
    original = read(source)['samples']
    assert len(original) == 3000
    assert Counter(s['task'] for s in original) == Counter({t: 500 for t in TASKS})
    assert len({(s['task'], s['question_id']) for s in original}) == 3000
    samples = []
    for s in sorted(original, key=lambda s: (TASKS.index(s['task']), s['question_id'])):
        assert sha256(s['image_path']) == s['image_sha256']
        item = {k: s[k] for k in ('task', 'question_id', 'image_id', 'image_sha256', 'image_path', 'question', 'answer')}
        item['source_sample_sha256'] = digest(s)
        samples.append(item)
    write(OUT / 'shared_3000_manifest.json', dict(source=str(source), source_sha256=sha256(source),
        source_commit='221cdde56dde53908aaa953fe26672fd33334f84', samples=samples, count=3000,
        unique_images=len({s['image_sha256'] for s in samples}), no_resampling=True))


def prepare(name):
    from transformers import AutoProcessor, AutoTokenizer
    assert importlib.metadata.version('transformers') == MODELS[name]['transformers']
    target = OUT / f'{name}_input_manifest.json'
    assert not target.exists()
    directory = model_path(name)
    if name == 'qwen':
        from ..qwen3vl_early_windows.experiment import prepare as official_prepare
        processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        old_manifest = read(QWEN / 'model_manifest.json')
        old_samples = {(s['task'], s['question_id']): s for s in read(QWEN / 'sample_manifest.json')['samples']}
    else:
        from ..internvl3_smoke.engine import partitions
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=False)
        fast = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True)
        spec = importlib.util.spec_from_file_location('frozen_internvl_conversation', directory / 'conversation.py')
        conversation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conversation)
        template = conversation.get_conv_template('internvl2_5')
        old_manifest = read(INTERN / 'protocol.json')['model_files_manifest']
        old_samples = {(s['task'], s['question_id']): s for s in read(INTERN / 'sample_manifest.json')['samples']}
    for f in old_manifest['files']:
        assert sha256(directory / f['path']) == f['sha256'], f['path']
    samples = []
    for number, sample in enumerate(read(OUT / 'shared_3000_manifest.json')['samples'], 1):
        if name == 'qwen':
            _, layout = official_prepare(processor, sample)
            item = dict(layout=layout, input_binding=digest(layout))
        else:
            request = f'<image>\n{sample["question"]}\nAnswer using a single word or short phrase.'
            chat = template.copy()
            chat.append_message(chat.roles[0], request)
            chat.append_message(chat.roles[1], None)
            prompt = chat.get_prompt().replace('<image>', '<img>' + '<IMG_CONTEXT>' * 256 + '</img>', 1)
            ids = tokenizer(prompt)['input_ids']
            layout = partitions(tokenizer, fast, prompt, sample['question'], ids,
                                tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>'))
            assert layout['image_tokens'] == 256
            item = dict(layout=layout, input_ids=ids, prompt=prompt, request=request, max_num=1)
            item['input_binding'] = digest(item)
        old = old_samples.get((sample['task'], sample['question_id']))
        if old:
            assert all(old[k] == sample[k] for k in ('image_sha256', 'image_id', 'question', 'answer'))
            assert old['layout'] == layout
            if name == 'qwen':
                assert old['input_binding'] == item['input_binding']
            else:
                assert old['input_ids'] == item['input_ids'] and old['prompt'] == item['prompt']
        samples.append(dict(task=sample['task'], question_id=sample['question_id'], **item))
        if number % 250 == 0:
            print(name, 'FREEZE', number, '/3000', flush=True)
    write(target, dict(model=MODELS[name], model_path=str(directory), samples=samples,
        history_manifest=old_manifest, historical_inputs_exact=len(old_samples),
        environment=dict(executable=sys.executable, python=sys.version,
            packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()})))


def freeze():
    assert not (OUT / 'protocol.json').exists()
    bindings = {name: dict(config=MODELS[name], input_manifest_sha256=sha256(OUT / f'{name}_input_manifest.json'),
        environment=read(OUT / f'{name}_input_manifest.json')['environment'],
        verified_model_files=read(OUT / f'{name}_input_manifest.json')['history_manifest']) for name in MODELS}
    write(OUT / 'model_and_environment_bindings.json', bindings)
    write(OUT / 'protocol.json', dict(scope='Same original 3000 GQA, single-layer original P-1 o_proj OFF only; no decode/generation/source mask',
        seed=20261101, bootstrap_repetitions=20000, bootstrap_cluster='Image SHA256; question weighted',
        physical_gpu=1, visible_devices='1', process_device='cuda:0', batch_size=1,
        models=MODELS, shared_manifest_sha256=sha256(OUT / 'shared_3000_manifest.json'),
        model_bindings_sha256=sha256(OUT / 'model_and_environment_bindings.json'),
        code_sha256=source_hashes(), input_manifests={name: sha256(OUT / f'{name}_input_manifest.json') for name in MODELS},
        signed_loss='FP32 log_softmax(z_full)[argmax(z_full)] - FP32 log_softmax(z_off)[argmax(z_full)]',
        absolute_change='Per-question abs(signed_loss), then mean; NOT abs(mean(signed_loss))',
        noop_tolerance=0, legacy_replay_tolerance=0, llava_historical_tolerances=read(LLAVA / 'sample_manifest.json')['protocol']['tolerances'],
        visual_replay='Only current-sample complete official visual outputs, including all Qwen DeepStack features; verify bit-exact against native. No batches cached.',
        inferential_scope='Descriptive exploration on same 3000; no noninferiority or speedup claims; unequal official templates/precision/token budgets retained'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['shared', 'qwen', 'internvl', 'freeze'])
    mode = parser.parse_args().mode
    shared() if mode == 'shared' else freeze() if mode == 'freeze' else prepare(mode)
