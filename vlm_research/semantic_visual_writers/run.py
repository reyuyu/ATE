"""CPU audit and freeze, strict Smoke and image-independence gate, serial matrix."""
import argparse
import fcntl
import io
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from paper_rq1_single_layer_gqa6.run import prepared, tensor_hashes
from query_role_attention.run import save_raw
from query_role_smoke_fix.run import Recorder
from vlm_common import load_model, gpu_memory

from .common import *
from .groups import token_groups, select, STOP
from .engine import generate, WriterMask


def cpu():
    stream = io.StringIO()
    suite = unittest.TestSuite()
    for module in ['query_role_attention.tests', 'query_role_smoke_fix.tests', 'query_role_formal.tests',
                   'relay_source_causality.tests', 'visual_write_window.tests', 'semantic_visual_writers.tests']:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('vlm_research.'+module))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    write(OUT/'cpu_validation.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        output=stream.getvalue(), sources=sources(), cuda_initialized=torch.cuda.is_initialized()))
    print(stream.getvalue(), flush=True)
    assert result.wasSuccessful() and not torch.cuda.is_initialized()
    write(OUT/'all_writer_reuse_audit.json', audit_all_writer())
    print('Audited 240 historical All-Writer checkpoints and original logits', flush=True)


def make_spec(processor, s):
    m = s['input_mapping']
    chat = processor.apply_chat_template([{'role':'user','content':[
        {'type':'image'}, {'type':'text','text':s['prompt_text']}]}], tokenize=False, add_generation_prompt=True)
    assert digest(chat) == m['chat_prompt_sha256']
    ids, image = m['prompt_ids'], m['visual_positions']
    relay = list(range(image[-1]+1, len(ids)-1))
    groups, qpos, words = token_groups(processor.tokenizer, s['question'], chat, ids, image, relay)
    selection = select(groups, qpos, relay, s['question'])
    assert selection['k'] > 0, 'No nonempty whole-word matched design; stop before GPU'
    return dict(question=s['question'], prompt_ids_sha256=digest(ids), relay_positions=relay,
        selection=selection, words=words, selection_binding=digest(selection),
        writers_decoded={c: processor.tokenizer.decode([ids[q] for q in positions])
                         for c, positions in selection['writers'].items()})


def freeze():
    assert read(OUT/'cpu_validation.json')['passed']
    assert read(OUT/'cpu_validation.json')['sources'] == sources()
    assert not (OUT/'protocol.json').exists()
    code_commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT.parent, text=True).strip()
    paths = ['vlm_research/'+name for name in sources()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths], cwd=ROOT.parent, text=True).strip()
    assert set(subprocess.check_output(['git','ls-files','--',*paths], cwd=ROOT.parent, text=True).splitlines()) == set(paths)
    old = read(HISTORY/'protocol.json')
    for path, info in old['model_files'].items():
        assert sha256(path) == info['sha256'], path
    for name in ['sample_manifest.json','input_mappings.json','image_swap_manifest.json','image_swap_inputs.json']:
        shutil.copyfile(HISTORY/name, OUT/name)
    assert sha256(OUT/'sample_manifest.json') == MANIFEST_SHA
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    samples = read(OUT/'sample_manifest.json')['samples']
    assert len(samples) == len({s['image_sha256'] for s in samples}) == 240
    assert all(sum(s['task'] == task for s in samples) == 40 for task in TASKS)
    main = {key(s): make_spec(processor, s) for s in samples}
    swaps = {}
    pairs = read(OUT/'image_swap_manifest.json')['pairs']
    assert len(pairs) == 40
    for pair in pairs:
        a,b = pair['samples']
        assert a['question'] == b['question'] and a['answer'] != b['answer'] and a['image_sha256'] != b['image_sha256']
        assert a['input_mapping']['prompt_ids'] == b['input_mapping']['prompt_ids']
        for side,s in enumerate(pair['samples']):
            assert sha256(s['image_path']) == s['image_sha256']
            swaps[f"{pair['pair_id']}__{side}"] = make_spec(processor, s)
        assert swaps[pair['pair_id']+'__0'] == swaps[pair['pair_id']+'__1']
    write(OUT/'token_group_manifest.json', dict(main=main, swaps=swaps))
    specs = list(main.values())
    write(OUT/'token_budget_audit.json', dict(passed=True, questions=240,
        k_min=min(s['selection']['k'] for s in specs), k_max=max(s['selection']['k'] for s in specs),
        k_mean=float(np.mean([s['selection']['k'] for s in specs])),
        reduced_for_whole_word_suffix=sum(s['selection']['budget_reduction_for_whole_word_suffix']>0 for s in specs),
        scaffold_deficits=sum(s['selection']['scaffold_budget_deficit']>0 for s in specs),
        all_matched_controls_exact=True, all_selections_whole_word=True, same_text_swap_selections_identical=True))
    gpu = select_gpu()
    names = ['cpu_validation.json','all_writer_reuse_audit.json','sample_manifest.json','input_mappings.json',
             'image_swap_manifest.json','image_swap_inputs.json','token_group_manifest.json','token_budget_audit.json','gpu_selection.json']
    protocol = dict(base_commit=BASE, source_mask_reference='e139c76e32edd3eacdf44d7172069bc37b230b60',
        query_role_reference='8d12f8c05f2b246d029ec4d9ffe45d1f857fa947', code_commit=code_commit,
        sources=sources(), bindings={name:sha256(OUT/name) for name in names},
        model_files=old['model_files'], model_revision=REVISION, model_path=str(MODEL),
        transformers_sources=old['transformers_sources'], scorer_sources=old['scorer_sources'],
        generation=old['generation'], dtype='float16', attention='eager', transformers='4.49.0',
        dependencies=subprocess.check_output([os.sys.executable,'-m','pip','freeze'], text=True).splitlines(),
        conditions=CONDITIONS, main_records=2400, reused_main_generations=240, new_main_generations=2160,
        swap_conditions=SWAP_CONDITIONS, swap_pairs=40, swap_generations=400, smoke_ids=SMOKE_IDS,
        gpu=gpu, layers=list(range(32)),
        answer_rule='P-1 and every real decode query (including EOS prediction) blocked from original image keys at all 32 layers in all ten conditions',
        relay_rule='Only selected original R query rows can read original image keys, at every layer; all non-image sources and all other computations native',
        lexical_rule=dict(method='Frozen closed functional stoplist; all remaining question whole words are core candidates, including wh, relations, logical operators, numbers and referential anchors',
            stopwords=sorted(STOP), external_tagger=False, no_gold_or_outputs=True,
            budget='Largest attainable core-group budget compatible with exact whole-word Question-Last-k and R-Last-k; left-to-right semantic subset tie break',
            drop_semantic='Complement of the final frozen Semantic-k set, not of every unselected lexical candidate; necessary for unambiguous same-set necessity test',
            scaffold='Left-to-right noncore groups, largest attainable budget no larger than k; report deficits without deleting questions',
            random='Seeded random permutation of whole R groups followed by first feasible exact-k subset DP, not uniform over independent subwords',
            random_seeds=RANDOM_SEEDS, random_key='seed plus question text only, never image, gold or model output'),
        inference=dict(bootstrap=20000, seed=SEED, cluster='Image SHA256, paired across all conditions',
            random_average='Per-question arithmetic mean of three seed correctness indicators, not 720 independent samples',
            primary_family=['Semantic minus random mean', 'Semantic minus Question-Last', 'Semantic minus R-Last',
                'Semantic minus Scaffold', 'Semantic minus All', 'Semantic minus No', 'Drop-Semantic minus All'],
            ordinary_CI='Exploratory percentile 95%', adjusted_CI='Also Bonferroni over the seven declared primary contrasts',
            quality_plot_strata='Input-only k ranges: 1-6, 7-10, >=11; no new budget policies or generations'),
        interpretation='Writer denotes permission to directly read image keys, not unique information storage or actual speedup',
        stop='No more layers, activation patching, router, training or kernel experiments')
    write(OUT/'protocol.json', protocol)
    print('FROZEN', sha256(OUT/'protocol.json'), 'GPU', gpu, flush=True)


def smoke(model, processor, samples, mappings, specs):
    cases = []
    for qid in SMOKE_IDS:
        i = next(i for i,s in enumerate(samples) if s['question_id'] == qid)
        s = samples[i]
        inputs = prepared(model, processor, s)
        assert tensor_hashes(inputs) == mappings[i]['input_hashes']
        weights = {n:p._version for n,p in model.named_parameters()}
        footprint = hook_footprint(model)
        selection = specs[key(s)]['selection']
        with torch.inference_mode(), Recorder(model) as native_rec:
            native = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=32,
                use_cache=True, logits_to_keep=1, eos_token_id=2)
        native_ids = native[0, inputs['input_ids'].shape[1]:].tolist()
        parent_result, _, parent_trace, parent = generate(model, processor.tokenizer, inputs,
            selection['writers']['All-Writer'], noop=True, capture=True, retain_cache=True)
        assert parent_result['token_ids'] == native_ids
        assert len(parent_trace) == len(native_rec.logits)
        assert all(torch.equal(a, b[-1].float()) for a,b in zip(parent_trace,native_rec.logits))
        parent_hash = cache_fingerprint(parent)
        noops = []
        for condition in ['No-Writer','Semantic-k','R-Last-k']:
            g, _, trace, _ = generate(model, processor.tokenizer, inputs, selection['writers'][condition], noop=True, capture=True)
            assert g['token_ids'] == native_ids and len(trace) == len(parent_trace)
            assert all(torch.equal(a,b) for a,b in zip(trace,parent_trace))
            noops.append(condition)
        checks = []
        for condition in CONDITIONS:
            counts, handles = [0]*32, []
            try:
                for k, layer in enumerate(model_layers(model)):
                    def mlp(module,args,out,k=k):
                        assert torch.isfinite(out).all()
                        counts[k] += 1
                    handles.append(layer.mlp.register_forward_hook(mlp))
                g, first, _, own = generate(model, processor.tokenizer, inputs,
                    selection['writers'][condition], retain_cache=True)
            finally:
                for h in handles:
                    h.remove()
            assert counts == [g['length']]*32
            assert cache_fingerprint(parent) == parent_hash
            assert all(x.data_ptr() != y.data_ptr() for a,b in zip(parent.to_legacy_cache(),own.to_legacy_cache()) for x,y in zip(a,b))
            if condition in ['All-Writer','No-Writer']:
                reference = read(HISTORY/'checkpoints'/key(s)/(condition+'.json'))
                assert torch.equal(first, first_logits(reference)) and g['token_ids'] == reference['generation']['token_ids']
            checks.append(dict(condition=condition, writer_count=g['writer_count'],
                correct_mask_probabilities=True, all_MLP_calls_native=True, independent_cache=True,
                max_probability_row_error=max(e['row_sum_max_error'] for e in g['source_events'])))
            del own
        try:
            with WriterMask(model, inputs, []):
                raise RuntimeError('intentional cleanup')
        except RuntimeError as error:
            assert str(error) == 'intentional cleanup'
        assert hook_footprint(model) == footprint and tensor_hashes(inputs) == mappings[i]['input_hashes']
        assert weights == {n:p._version for n,p in model.named_parameters()}
        cases.append(dict(question_id=qid, passed=True, native_HF_noop_all_valid_logits_equal=True,
            noop_conditions=noops, logits_to_keep=1, historical_All_No_exact=True,
            hook_cleanup_normal_exception=True, independent_cache_storage=True, checks=checks))
        del parent, parent_result, parent_trace, native_rec
        print('SMOKE PASS', qid, '10 conditions', flush=True)
    return dict(passed=True, cases=cases, strict_torch_equal=True, no_tolerance_relaxation=True)


def save_trace(path, trace):
    array = torch.stack(trace).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            assert np.array_equal(data['valid_logits'], array)
    else:
        temp = path.with_suffix(f'.{os.getpid()}.tmp')
        with temp.open('xb') as handle:
            np.savez_compressed(handle, valid_logits=array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp,path)
    return dict(path=str(path), sha256=sha256(path))


def record(model, processor, s, inputs, condition, spec, protocol, stem, swap=False, capture=False):
    directory = 'swap_checkpoints' if swap else 'checkpoints'
    path = OUT/directory/stem/(condition+'.json')
    if path.exists():
        r = read(path)
        assert r['binding'] == digest(protocol) and r['sample_sha256'] == digest(s)
        assert r['selection_binding'] == spec['selection_binding']
        assert r['result_sha256'] == digest(r['generation'])
        first_logits(r)
        return r
    before, footprint = tensor_hashes(inputs), hook_footprint(model)
    weights = {n:p._version for n,p in model.named_parameters()}
    writers = spec['selection']['writers'][condition]
    if condition == 'All-Writer' and not swap:
        original = HISTORY/'checkpoints'/key(s)/'All-Writer.json'
        old = read(original)
        assert old['input_hashes'] == before
        g, raw, trace_raw = old['generation'], old['raw'], None
        provenance = dict(reused=True, checkpoint=str(original.relative_to(ROOT)), sha256=sha256(original),
            binding=old['binding'], code_commit=old['code_commit'])
    else:
        g, first, trace, _ = generate(model, processor.tokenizer, inputs, writers, capture=capture)
        g.update(correct=int(g['prediction'] == s['answer']), first_logits_sha256=digest(first.tolist()))
        raw_dir = RAW/('swap' if swap else 'main')/stem
        raw = save_raw(raw_dir/(condition+'.npz'), first)
        trace_raw = save_trace(raw_dir/(condition+'__trace.npz'), trace) if capture else None
        provenance = dict(reused=False)
    assert tensor_hashes(inputs) == before and hook_footprint(model) == footprint
    assert weights == {n:p._version for n,p in model.named_parameters()}
    row = dict(binding=digest(protocol), code_commit=protocol['code_commit'], sample_sha256=digest(s),
        question_id=s['question_id'], task=s['task'], gold=s['answer'], image_sha256=s['image_sha256'],
        condition=condition, input_hashes=before, selection_binding=spec['selection_binding'],
        writer_positions=writers, writer_count=len(writers), relay_count=len(spec['relay_positions']),
        generation=g, result_sha256=digest(g), raw=raw, valid_trace_raw=trace_raw, provenance=provenance)
    write(path,row)
    return row


def gpu():
    start = time.perf_counter()
    p = verify()
    assert transformers.__version__ == '4.49.0'
    preflight(p['gpu'])
    model, processor, env = load_model(MODEL)
    assert torch.cuda.device_count() == 1 and env['parameter_dtypes'] == ['torch.float16']
    env.update(pid=os.getpid(), gpu=p['gpu'], python_executable=os.sys.executable,
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip())
    write(OUT/f'environment_{os.getpid()}.json', env)
    samples = read(OUT/'sample_manifest.json')['samples']
    mappings = read(OUT/'input_mappings.json')
    specs = read(OUT/'token_group_manifest.json')
    if not (OUT/'gpu_smoke_validation.json').exists():
        write(OUT/'gpu_smoke_validation.json', smoke(model,processor,samples,mappings,specs['main']))
    assert read(OUT/'gpu_smoke_validation.json')['passed']
    swap_pairs = read(OUT/'image_swap_manifest.json')['pairs']
    swap_inputs = read(OUT/'image_swap_inputs.json')
    controls = []
    # Strict all-step image independence is a gate, not a post-hoc observation.
    for pair in swap_pairs:
        values = []
        for side,s in enumerate(pair['samples']):
            stem = f"{pair['pair_id']}__{side}"
            inputs = prepared(model,processor,s)
            assert tensor_hashes(inputs) == swap_inputs[stem]['input_hashes']
            r = record(model,processor,s,inputs,'No-Writer',specs['swaps'][stem],p,stem,swap=True,capture=True)
            assert sha256(r['valid_trace_raw']['path']) == r['valid_trace_raw']['sha256']
            with np.load(r['valid_trace_raw']['path'],allow_pickle=False) as data:
                values.append(torch.from_numpy(data['valid_logits'].copy()))
        assert torch.equal(values[0],values[1]), 'No-Writer image independence failed; stop and inspect paths'
        controls.append(dict(pair_id=pair['pair_id'], every_valid_logit_exact=True, max_absolute_difference=0.))
    write(OUT/'image_independence_validation.json', dict(passed=True, pairs=controls, strict_torch_equal=True))
    print('GATE PASS: Smoke and all 40 No-Writer paired controls exact',flush=True)
    for pair in swap_pairs:
        for side,s in enumerate(pair['samples']):
            stem = f"{pair['pair_id']}__{side}"
            inputs = prepared(model,processor,s)
            assert tensor_hashes(inputs) == swap_inputs[stem]['input_hashes']
            for condition in SWAP_CONDITIONS:
                record(model,processor,s,inputs,condition,specs['swaps'][stem],p,stem,swap=True,capture=condition=='No-Writer')
    print('SWAPS COMPLETE 400/400 (separate from main240)',flush=True)
    formal_start = time.perf_counter()
    for i,s in enumerate(samples):
        inputs = prepared(model,processor,s)
        assert tensor_hashes(inputs) == mappings[i]['input_hashes']
        spec = specs['main'][key(s)]
        for condition in CONDITIONS:
            record(model,processor,s,inputs,condition,spec,p,key(s))
        if (i+1)%20 == 0:
            print(f'MAIN {(i+1)*10}/2400 records ({(i+1)*9}/2160 new), elapsed={time.perf_counter()-formal_start:.1f}s',flush=True)
    verify()
    for path, info in p['model_files'].items():
        assert sha256(path) == info['sha256'], path
    assert len(list((OUT/'checkpoints').glob('*/*.json'))) == 2400
    assert len(list((OUT/'swap_checkpoints').glob('*/*.json'))) == 400
    write(OUT/'gpu_completion.json', dict(passed=True, binding=digest(p), pid=os.getpid(),
        main_records=2400, new_main_generations=2160, reused_main_generations=240, swap_generations=400,
        model_replicas=1, seconds=time.perf_counter()-start, formal_seconds=time.perf_counter()-formal_start,
        model_full_sha256_rechecked=True, memory=gpu_memory()))
    print('COLLECTION COMPLETE: 2400 main records / 400 swap; model process exits',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['cpu','freeze','gpu'])
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        with (OUT/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            globals()[args.stage]()
    except BaseException:
        write(OUT/'failures'/f'{args.stage}_{time.time_ns()}.json', dict(status='PARTIAL',stage=args.stage,
            traceback=traceback.format_exc(), hardware=hardware()))
        raise
