"""Immutable 240-question, four-condition runner on physical GPU1-GPU3."""
import argparse
from collections import Counter, defaultdict
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch

from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers, tensor_hash
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.run import prepared, atomic_json, tensor_hashes
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_early_joint_attention.protocol import ROOT, MODEL_PATH
from figure3_decode_windows.run import rank_for, OUT as FIGURE, cohort
from rq2_source_knockout import run as old
from rq2_source_knockout.engine import forward as old_forward
from vlm_common import load_model, gpu_memory
from .engine import CONDITIONS, LABELS, MultiImageKnockout, generate

BASE = '3a7c498c722dab7344419490372c2af5dbb67a4d'
OUT = ROOT / 'results/rq2_two_layer_visual_sufficiency_240'
SEED = 20261031
CONTRASTS = [('A','Full'), ('B','Full'), ('C','Full'), ('D','Full'), ('C','A'), ('C','B')]
GPUS = ['GPU-b08489af-0a8a-eb48-55ba-8346733af417', *old.GPUS]


def assign(ss):
    groups = defaultdict(list)
    for s in ss:
        groups[s['image_sha256']].append(s)
    loads, assignments = [0]*3, {}
    for image in sorted(groups, key=lambda h: (-len(groups[h]),digest(['rq2.2',SEED,h]))):
        rank = min(range(3), key=lambda k: (loads[k],k))
        for s in groups[image]:
            assignments[s['question_id']] = rank
        loads[rank] += len(groups[image])
    return assignments


def preflight(rank):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPUS[rank]
    table = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True)
    row = next(line.split(', ') for line in table.splitlines() if GPUS[rank] in line)
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader'],text=True)
    if int(row[0]) != rank+1 or int(row[3]) < 18000 or any(line.startswith(GPUS[rank]) for line in apps.splitlines()):
        raise ValueError('GPU occupied; never stop other processes')
    return dict(physical_gpu=rank+1,uuid=GPUS[rank],snapshot=table,processes=apps)


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    names = set(old.sources()) | {'rq2_visual_sufficiency/' + n for n in
        ('__init__.py', 'engine.py', 'run.py', 'analyze.py')}
    names |= {'test_rq2_visual_sufficiency.py', 'paper_rq1/engine.py',
              'paper_rq1/scorers.py', 'paper_rq1_early_joint_attention/engine.py'}
    return {n: sha256(ROOT/n) for n in sorted(names)}


def samples():
    return read(OUT/'sample_manifest.json')['samples']


def full_path(sample):
    return FIGURE / f"shard{rank_for(sample)}/checkpoints/{sample['task']}__{sample['question_id']}/Full.json"


def source_path(sample):
    rank = read(old.OUT/'sample_manifest.json')['assignments'][sample['question_id']]
    return old.OUT / f"shard{rank}/checkpoints/{sample['task']}__{sample['question_id']}.json"


def historical_logits(sample):
    record = read(source_path(sample))['result']
    path = old.OUT/record['full_logits_file']
    assert sha256(path) == record['full_logits_file_sha256']
    with np.load(path) as data:
        return torch.from_numpy(data['full_logits'].copy())


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(old.OUT/'sample_manifest.json')
    ss = manifest['samples']
    assert len(ss) == 240 and manifest['images'] == 235
    assert Counter(s['task'] for s in ss) == Counter({t:40 for t in old.TASKS})
    original = {s['question_id']:s for s in cohort()}
    figure = read(FIGURE/'execution_manifest.json')
    assert figure['model_revision'] == old.REVISION
    assert figure['generation']['max_new_tokens'] == 32 and figure['generation']['use_cache']
    bindings = []
    for sample in ss:
        assert sample == original[sample['question_id']]
        assert sha256(sample['image_path']) == sample['image_sha256']
        full = read(full_path(sample))
        assert full['binding'] == digest(figure) and full['sample_sha256'] == digest(sample)
        gen = full['generation']
        assert full['result_sha256'] == digest(gen) and gen['edits'] == 0
        assert gen['score'] == int(gqa_prediction(gen['text']) == sample['answer'])
        logits = historical_logits(sample)
        assert int(logits.argmax()) == gen['token_ids'][0]
        rec = read(source_path(sample))
        assert rec['sample_sha256'] == digest(sample) and rec['result_sha256'] == digest(rec['result'])
        bindings.append(dict(question_id=sample['question_id'], sample_sha256=digest(sample),
            full_file=str(full_path(sample).relative_to(ROOT)), full_sha256=sha256(full_path(sample)),
            source_file=str(source_path(sample).relative_to(ROOT)), source_sha256=sha256(source_path(sample)),
            first_logits_sha256=tensor_hash(logits)))
    freeze_json(OUT/'sample_manifest.json', manifest)
    freeze_json(OUT/'assignments.json',assign(ss))
    freeze_json(OUT/'history_bindings.json', dict(records=bindings,
        original_manifest_sha256=sha256(old.OUT/'sample_manifest.json'),
        figure_manifest_sha256=sha256(FIGURE/'execution_manifest.json'),
        old_protocol_sha256=sha256(old.OUT/'protocol.json'),
        original_scorer_sha256=sha256(old.ORIGINAL/'scorer_validation.json')))
    print('Bound240 questions/235 images; exact original samples, Full and first-token histories', flush=True)


def cpu():
    stream = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
                              ['test_rq2_source_knockout', 'test_rq2_visual_sufficiency'])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    if not result.wasSuccessful():
        raise AssertionError('CPU tests failed')
    freeze_json(OUT/'cpu_validation.json', dict(passed=True, tests=result.testsRun,
                source=sources(), output=stream.getvalue()))


def code_check():
    paths = ['vlm_research/'+n for n in sources()]
    assert set(subprocess.check_output(['git','ls-files','--',*paths], cwd=ROOT.parent, text=True).splitlines()) == set(paths)
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert read(OUT/'cpu_validation.json')['source'] == sources()


def smoke(rank):
    code_check()
    pre = preflight(rank)
    started = time.perf_counter()
    model, processor, environment = load_model(MODEL_PATH)
    footprint = hook_footprint(model)
    versions = {n:p._version for n,p in model.named_parameters()}
    checks = []
    for task in ['ChooseRel','QueryAttr']:
        sample = next(s for s in samples() if s['task'] == task)
        prompt = prepared(model, processor, sample)
        before = tensor_hashes(prompt)
        bos = processor.tokenizer.bos_token_id
        full, first, trace, debug = generate(model,prompt,bos,[],audit=True,collect=True)
        n = prompt['input_ids'].shape[1]
        with torch.inference_mode():
            native = model.generate(**prompt,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,eos_token_id=2)[0,n:].tolist()
        assert native == full['token_ids'] == read(full_path(sample))['generation']['token_ids']
        previous = historical_logits(sample)
        assert torch.equal(first, previous)
        noop, no_first, no_trace, no_debug = generate(model,prompt,bos,CONDITIONS['A'],noop=True,audit=True,collect=True)
        assert noop['token_ids'] == full['token_ids'] and torch.equal(first,no_first)
        assert all(torch.equal(a,b) for a,b in zip(trace,no_trace)) and len(trace) == len(no_trace)
        assert torch.equal(debug['prefill_logits'],no_debug['prefill_logits'])
        single, single_first, _, _ = generate(model,prompt,bos,[14])
        legacy, _, _ = old_forward(model,prompt,bos,14,'Image')
        assert torch.equal(single_first,legacy)
        old14 = next(r for r in read(source_path(sample))['result']['metrics'] if r['layer']==14 and r['source']=='Image')
        assert tensor_hash(single_first) == old14['logits_sha256']
        conditions = []
        for key, blocked in CONDITIONS.items():
            result, current, _, audit = generate(model,prompt,bos,blocked,audit=True)
            assert torch.equal(audit['prefill_logits'][0,:n-1],debug['prefill_logits'][0,:n-1])
            assert audit['observer'].states == debug['observer'].states
            assert all(a['other_query_attention_sha256'] == b['other_query_attention_sha256']
                       for a,b in zip(audit['observer'].rows,debug['observer'].rows))
            for row in audit['observer'].rows:
                if row['layer'] in blocked:
                    assert row['Image_max'] == 0
                else:
                    assert row['Image'] > 0
            # Running the same condition twice starts from a fresh cache and is deterministic.
            again, again_first, _, _ = generate(model,prompt,bos,blocked)
            assert again == result and torch.equal(current,again_first)
            assert hook_footprint(model) == footprint
            conditions.append(dict(condition=key,masked_layers=blocked,other_queries_exact=True,
                masked_probabilities_exact_zero=True,kept_layers_image_mass_positive=True,
                independent_repeat_exact=True,mask_removed_before_decode=True,
                first_logits_sha256=tensor_hash(current),token_ids=result['token_ids']))
        try:
            with MultiImageKnockout(model,prompt,bos,CONDITIONS['A']):
                raise RuntimeError('deliberate exception cleanup')
        except RuntimeError:
            pass
        assert footprint == hook_footprint(model) and before == tensor_hashes(prompt)
        assert versions == {n:p._version for n,p in model.named_parameters()}
        checks.append(dict(task=task,question_id=sample['question_id'],visual_tokens=576,
            query=n-1,full_history_logits_exact=True,full_history_max_abs_difference=float((first-previous).abs().max()),
            native_full_generation_exact=True,noop_all_valid_logits_and_generation_exact=True,
            historical_L14_logits_exact=True,inputs_and_parameters_unchanged=True,
            exception_and_normal_hook_cleanup=True,conditions=conditions))
    atomic_json(OUT/f'shard{rank}/smoke.json', dict(passed=True,source=sources(),environment=environment,
        preflight=pre,checks=checks,seconds=time.perf_counter()-started))
    print('GPU smoke passed',rank+1,flush=True)


def freeze():
    code_check()
    for rank in range(3):
        s = read(OUT/f'shard{rank}/smoke.json')
        assert s['passed'] and s['source'] == sources()
    prior = read(old.OUT/'protocol.json')
    for path, record in prior['model_files'].items():
        assert sha256(path) == record['sha256']
    for path, h in prior['transformers_source'].items():
        assert sha256(path) == h
    names = ['sample_manifest.json','assignments.json','history_bindings.json','cpu_validation.json',
             *[f'shard{k}/smoke.json' for k in range(3)]]
    freeze_json(OUT/'protocol.json',dict(status='frozen',base_commit=BASE,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source=sources(),bindings={n:sha256(OUT/n) for n in names},model_revision=old.REVISION,
        model_files=prior['model_files'],transformers_source=prior['transformers_source'],
        transformers='4.49.0',dtype='FP16',attention='eager',gpu_uuids=GPUS,
        samples=240,images=235,new_conditions=CONDITIONS,labels=LABELS,new_generations=960,
        generation=dict(do_sample=False,num_beams=1,max_new_tokens=32,eos_token_id=2,use_cache=True),
        intervention='Pre-softmax -inf, original Prefill P-1 ->576 Image keys only; mask removed before native Decode. MLP, other Queries and positions unchanged.',
        full_reuse='Figure3 native Full complete answers; RQ2.1 same-input native Full first-token logits. Strict binding and all three GPU exact smoke required.',
        statistics=dict(seed=SEED,repetitions=200000,unit='235 whole image SHA256 clusters; paired question-weighted accuracy',
            family=CONTRASTS,simultaneous='Bonferroni6 percentile bootstrap; ordinary95 and familywise95',
            tasks='Descriptive only; no taskwise confirmatory claims',recovery_ratio='Omitted if denominator unstable; not needed for primary contrasts'),
        limits='Exploration on the same240 that selected L14/L19; no equivalence claim, no physical acceleration, no new layer combinations or2760-question extension.'))
    print('Protocol frozen after all three GPU smokes',flush=True)


def verify():
    c = read(OUT/'protocol.json')
    assert c['source'] == sources() and c['new_conditions'] == CONDITIONS
    for n,h in c['bindings'].items():
        assert sha256(OUT/n) == h
    for p,h in c['transformers_source'].items():
        assert sha256(p) == h
    for p,b in c['model_files'].items():
        stat = Path(p).stat()
        assert stat.st_size == b['bytes'] and stat.st_mtime_ns == b['mtime_ns']
    return c


def worker(rank):
    c = verify(); binding = digest(c)
    folder = OUT/f'shard{rank}'
    folder.mkdir(exist_ok=True)
    lock = (folder/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'completion.json').exists():
        assert read(folder/'completion.json')['binding'] == binding
        print('Already complete; no GPU loaded',flush=True); return
    pre = preflight(rank)
    started_epoch = time.time(); started = time.perf_counter()
    model,processor,environment = load_model(MODEL_PATH)
    assert environment['transformers'] == '4.49.0' and environment['parameter_dtypes'] == ['torch.float16']
    assert environment['model_revision'] == old.REVISION and not model.training
    versions = {n:p._version for n,p in model.named_parameters()}; footprint = hook_footprint(model)
    assignments = read(OUT/'assignments.json')
    selected = [s for s in samples() if assignments[s['question_id']] == rank]
    resumed = 0
    for i,sample in enumerate(selected):
        prompt = None
        for key,blocked in CONDITIONS.items():
            path = folder/f"checkpoints/{sample['task']}__{sample['question_id']}/{key}.json"
            if path.exists():
                rec = read(path)
                assert rec['binding'] == binding and rec['sample_sha256'] == digest(sample) and rec['condition'] == key
                assert rec['result_sha256'] == digest(rec['generation']) and rec['metric_sha256'] == digest(rec['first_token'])
                assert sha256(OUT/rec['logits_file']) == rec['logits_sha256']
                resumed += 1; continue
            if prompt is None:
                prompt = prepared(model,processor,sample); before = tensor_hashes(prompt)
                full = historical_logits(sample)
            tick = time.perf_counter()
            generation,first,_,_ = generate(model,prompt,processor.tokenizer.bos_token_id,blocked)
            text = processor.tokenizer.decode(generation['token_ids'],skip_special_tokens=True).strip()
            generation.update(text=text,prediction=gqa_prediction(text),score=int(gqa_prediction(text)==sample['answer']))
            metric = metric_rows(full,first[None])[0]
            metric['logp_loss'] = -metric['delta_logp']
            logits_file = path.with_suffix('.npz'); logits_file.parent.mkdir(parents=True,exist_ok=True)
            temp = logits_file.with_suffix('.npz.tmp')
            with temp.open('wb') as f:
                np.savez_compressed(f,first_logits=first.numpy())
            temp.replace(logits_file)
            assert before == tensor_hashes(prompt) and footprint == hook_footprint(model)
            assert versions == {n:p._version for n,p in model.named_parameters()}
            atomic_json(path,dict(binding=binding,sample_sha256=digest(sample),task=sample['task'],
                question_id=sample['question_id'],image_id=sample['image_id'],image_sha256=sample['image_sha256'],
                condition=key,generation=generation,result_sha256=digest(generation),first_token=metric,metric_sha256=digest(metric),
                logits_file=str(logits_file.relative_to(OUT)),logits_sha256=sha256(logits_file),seconds=time.perf_counter()-tick,
                hooks_clean=True,inputs_unchanged=True,parameter_versions_unchanged=True))
        if (i+1)%10 == 0 or i+1 == len(selected):
            print(json.dumps(dict(gpu=rank+1,questions=i+1,total=len(selected),generations=(i+1)*4,seconds=time.perf_counter()-started)),flush=True)
    verify()
    atomic_json(folder/'completion.json',dict(status='complete',binding=binding,questions=len(selected),generations=len(selected)*4,
        resumed=resumed,started_epoch=started_epoch,finished_epoch=time.time(),seconds=time.perf_counter()-started,
        environment=environment,preflight=pre,memory=gpu_memory(),hooks_clean=footprint==hook_footprint(model),
        parameter_versions_unchanged=versions=={n:p._version for n,p in model.named_parameters()}))
    print('Complete GPU',rank+1,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','cpu','smoke','freeze','worker'])
    parser.add_argument('--rank',type=int,choices=[0,1,2],default=0)
    args = parser.parse_args()
    try:
        if args.stage in ('smoke','worker'):
            globals()[args.stage](args.rank)
        else:
            globals()[args.stage]()
    except BaseException:
        atomic_json(OUT/f'failures/{args.stage}_shard{args.rank}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
