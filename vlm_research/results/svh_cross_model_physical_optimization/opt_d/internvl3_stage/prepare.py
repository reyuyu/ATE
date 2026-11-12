"""Freeze actual prompts/partitions on CPU before any model scoring."""
import ast
import importlib.util
import importlib.metadata
import sys

from transformers import AutoTokenizer

from .common import (OUT, ROOT, SOURCE, MODEL, REVISION, SEED, TASKS, CONDITIONS,
                     HISTORY, read, write, sha256, sources, select, prediction)
from vlm_research.internvl3_smoke.engine import partitions, MASS_TOLERANCE


def main():
    assert not (OUT/'protocol.json').exists(), 'Never silently resample/refreeze'
    old = ROOT/'vlm_research/results/internvl3_8b_single_gpu_smoke'
    assert read(old/'download_validation.json')['passed']
    assert read(old/'post_run_validation.json')['passed']
    manifest = read(old/'model_files_sha256.json')
    for entry in manifest['files']:
        file = MODEL/entry['path']
        assert sha256(file) == entry['sha256'], str(file)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    fast = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    spec = importlib.util.spec_from_file_location('pinned_internvl_conversation', MODEL/'conversation.py')
    conversation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conversation)
    template = conversation.get_conv_template('internvl2_5')
    rows = read(SOURCE)['samples']
    assert len(rows) == 3000 and all(sum(r['task']==t for r in rows)==500 for t in TASKS)
    samples = []
    for row in select(rows):
        sample = {k: row[k] for k in ('task','question_id','image_id','image_path','image_sha256','question','answer')}
        assert sha256(sample['image_path']) == sample['image_sha256']
        sample['request'] = f'<image>\n{sample["question"]}\nAnswer using a single word or short phrase.'
        chat = template.copy()
        chat.append_message(chat.roles[0], sample['request'])
        chat.append_message(chat.roles[1], None)
        prompt = chat.get_prompt().replace('<image>','<img>'+'<IMG_CONTEXT>'*256+'</img>',1)
        ids = tokenizer(prompt)['input_ids']
        layout = partitions(tokenizer, fast, prompt, sample['question'], ids, tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>'))
        assert layout['image_tokens'] == 256 and not layout['actual_bos_positions']
        sample.update(prompt=prompt, input_ids=ids, layout=layout, max_num=1)
        samples.append(sample)
    tree = ast.parse(HISTORY['scorer'].read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gqa_prediction')
    scope = {}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(HISTORY['scorer']),'exec'),scope)
    assert all(scope['gqa_prediction'](text)==prediction(text) for text in ['Blue','cat.','yes...','red!',''])
    write(OUT/'sample_manifest.json', dict(seed=SEED, source=str(SOURCE), source_sha256=sha256(SOURCE),
        source_commit='221cdde56dde53908aaa953fe26672fd33334f84', count=120,
        unique_images=len({r['image_sha256'] for r in samples}),
        selection='20 task-rounds; min SHA256(JSON[seed,task,qid,imageSHA]), prefer globally unused image, no model outputs used',samples=samples))
    ordered = sorted(samples,key=lambda r:(r['layout']['input_length'],r['task'],r['question_id']))
    smoke = [(s['task'],s['question_id']) for s in (ordered[0],ordered[len(ordered)//2],ordered[-1])]
    write(OUT/'environment.json',dict(executable=sys.executable, python=sys.version,
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        model_environment_unchanged_from_smoke=True, analysis_environment='/data/lk/深度学习/.envs/ate-vlm-research/bin/python (CPU only; no package modifications)'))
    write(OUT/'protocol.json',dict(model_repo='OpenGVLab/InternVL3-8B',revision=REVISION,model_path=str(MODEL),
        physical_gpu=1,visible_devices='1',dtype='bfloat16',attention='eager',max_num=1,image_tokens=256,
        conditions=CONDITIONS,language_layers=28,max_new_tokens=32,do_sample=False,use_cache=True,
        sample_manifest_sha256=sha256(OUT/'sample_manifest.json'),code_sha256=sources(),
        history_sha256={str(p):sha256(p) for p in HISTORY.values()},model_files_manifest=manifest,
        smoke_samples=smoke,noop_tolerance=0.0,mass_row_sum_tolerance=MASS_TOLERANCE,
        bootstrap=dict(seed=SEED,repetitions=20000,cluster='Image SHA256',weighting='Question-weighted',quantiles=[.025,.975]),
        intervention='o_proj output zero; single-layer original P-1 only, window original P-1 plus every actual decode query; other prefill/QKV/cache/MLP/residual/vision unchanged',
        full='Official chat greedy full generation reused for native first logits and immutable vision-injected embeddings; independent fresh caches for all replays/conditions',
        scope='Exploratory 120 questions, 3360 single-layer prefill interventions, 840 full/window generations; no additional combinations or source masks',
        cross_model='LLaVA 3000 historical questions/32 layers/576 visual tokens/BOS vs InternVL 120/28/256/Start Anchor; not paired cross-model or causal stage proof'))
    write(OUT/'source_partition_audit.json',dict(passed=True,samples=120,unique_images=len({r['image_sha256'] for r in samples}),
        image_tokens=256,exclusive_exhaustive=True,fast_slow_token_ids_exact=True,bos_absent=True,
        input_lengths=[r['layout']['input_length'] for r in ordered],smoke_short_typical_long=smoke))
    print('FROZEN',len(samples),'images',len({r['image_sha256'] for r in samples}),'lengths',ordered[0]['layout']['input_length'],ordered[-1]['layout']['input_length'],flush=True)


if __name__ == '__main__':
    main()
