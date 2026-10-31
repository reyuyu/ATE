"""Small immutable two-GPU runner using existing inputs, masks and FP32 metrics."""
import argparse
from collections import Counter, defaultdict
import fcntl
import inspect
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
import transformers.models.llama.modeling_llama as llama

from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from figure3_decode_windows.run import cohort, TASKS, ORIGINAL
from knockout_probe import model_layers, tensor_hash
from paper_rq1_single_layer_gqa6.run import prepared, atomic_json, tensor_hashes
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_early_joint_attention.protocol import ROOT, MODEL_PATH
from vlm_common import load_model, gpu_memory
from .engine import SOURCES, SourceKnockout, forward, partition

OUT = ROOT / 'results/rq2_single_layer_source_240'
MASS = ROOT / 'results/first_answer_attention_mass'
SEED = 20261031
BASE = '2b10df01904a7fc8090756a90de72c75f91e0f47'
ORIGINAL_COMMIT = '221cdde56dde53908aaa953fe26672fd33334f84'
REVISION = 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
GPUS = ['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6', 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    names = ['rq2_source_knockout/' + n for n in ['__init__.py', 'engine.py', 'run.py', 'analyze.py']]
    names += ['test_rq2_source_knockout.py', 'knockout_probe.py', 'vlm_common.py',
        'figure3_decode_windows/run.py', 'paper_rq1_single_layer_gqa6/run.py',
        'paper_rq1_single_layer_gqa6/engine.py', 'benchmarks/dataset_registry.py',
        'paper_rq1_early_joint_attention/statistics.py', 'paper_rq1_decode_layerwise_quality/statistics.py']
    return {n: sha256(ROOT / n) for n in names}


def selected_samples():
    return sorted([s for task in TASKS for s in sorted(
        (s for s in cohort() if s['task'] == task),
        key=lambda s: digest(['rq2-source240', SEED, task, s['question_id']]))[:40]],
        key=lambda s: (s['task'], s['question_id']))


def assign(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample['image_sha256']].append(sample)
    loads, ranks = [0, 0], {}
    for image in sorted(groups, key=lambda h: (-len(groups[h]), digest(['shard', SEED, h]))):
        rank = min(range(2), key=lambda k: (loads[k], k))
        for sample in groups[image]:
            ranks[sample['question_id']] = rank
        loads[rank] += len(groups[image])
    return ranks


def old_record(sample):
    path = ORIGINAL / f"first_token/shard{sample['rank']}/checkpoints/{sample['task']}__{sample['question_id']}.json"
    value = read(path)
    assert value['sample_sha256'] == digest(sample)
    return value['result'], path


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = selected_samples()
    assert len(samples) == 240 and Counter(s['task'] for s in samples) == Counter({t:40 for t in TASKS})
    original = 'vlm_research/results/paper_rq1_single_layer_gqa6/sample_manifest.json'
    blob = subprocess.check_output(['git', 'rev-parse', f'{ORIGINAL_COMMIT}:{original}'], cwd=ROOT, text=True).strip()
    assert blob == subprocess.check_output(['git', 'hash-object', ORIGINAL/'sample_manifest.json'], text=True).strip()
    ranks, history = assign(samples), []
    for sample in samples:
        assert sha256(sample['image_path']) == sample['image_sha256']
        old, path = old_record(sample)
        assert sha256(old['logits_path']) == old['logits_sha256']
        mass = MASS / f"checkpoints/{sample['task']}__{sample['question_id']}.json"
        assert read(mass)['sample_sha256'] == digest(sample)
        history.append(dict(question_id=sample['question_id'], first_token_checkpoint=str(path),
            checkpoint_sha256=sha256(path), raw_logits=old['logits_path'], raw_sha256=old['logits_sha256'],
            native_mass_checkpoint=str(mass), native_mass_sha256=sha256(mass)))
    freeze_json(OUT/'sample_manifest.json', dict(seed=SEED, original_manifest_sha256=sha256(ORIGINAL/'sample_manifest.json'),
        original_git_blob=blob, selection="Per task take40 smallest SHA256(canonical JSON ['rq2-source240',20261031,task,question_id]); independent of outputs.",
        samples=samples, sample_sha256=[digest(s) for s in samples], assignments=ranks,
        shard_questions=dict(Counter(str(r) for r in ranks.values())), images=len({s['image_sha256'] for s in samples})))
    freeze_json(OUT/'history_bindings.json', history)
    print('Frozen240:', Counter(ranks.values()), 'images', len({s['image_sha256'] for s in samples}), flush=True)


def cpu():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_rq2_source_knockout'))
    stamp = time.time_ns()
    record = dict(passed=result.wasSuccessful(), tests=result.testsRun, source=sources(), log=stream.getvalue())
    atomic_json(OUT/f'attempts/cpu_{stamp}.json', record)
    if not result.wasSuccessful():
        print(stream.getvalue(), flush=True)
        raise AssertionError('CPU tests failed')
    freeze_json(OUT/'cpu_validation.json', record)
    print(stream.getvalue(), flush=True)


def code_check():
    paths = ['vlm_research/'+n for n in sources()]
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines()) == set(paths)
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert read(OUT/'cpu_validation.json')['source'] == sources()


def preflight(rank):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPUS[rank]:
        raise ValueError('Only physical GPU2/3 with their frozen UUIDs')
    table = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows = [line.split(', ') for line in table.splitlines()]
    chosen = next(row for row in rows if row[1] == GPUS[rank])
    assert int(chosen[0]) == rank+2
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    if any(line.startswith(GPUS[rank]) for line in processes.splitlines()) or int(chosen[3]) < 18000:
        raise ValueError('GPU occupied; never stop another process')
    return dict(physical_gpu=rank+2,uuid=GPUS[rank],snapshot=table,processes=processes)


def history_compare(sample, logits, rows):
    old, _ = old_record(sample)
    with np.load(old['logits_path']) as raw:
        previous = torch.from_numpy(raw['full_logits'])
    delta = float((previous-logits).abs().max())
    historical = read(MASS/f"checkpoints/{sample['task']}__{sample['question_id']}.json")['result']['rows']
    masses = []
    for current, prior in zip(rows, historical):
        masses.extend([abs(current['Image']-prior['image_mass']), abs(current['BOS']-prior['bos_mass']),
            abs(current['Prompt']+current['Self']-prior['prompt_mass'])])
    return dict(historical_full_exact=torch.equal(previous,logits), historical_full_max_abs_difference=delta,
        historical_full_top1_same=int(previous.argmax())==int(logits.argmax()),
        historical_mass_max_abs_difference=max(masses), baseline='Current same-GPU Native Full for every paired comparison; historical values only provenance/check.')


def smoke(rank):
    code_check()
    pre = preflight(rank)
    model, processor, environment = load_model(MODEL_PATH)
    assert environment['transformers']=='4.49.0' and environment['parameter_dtypes']==['torch.float16']
    assert environment['model_revision']==REVISION and not model.training
    footprint = hook_footprint(model)
    versions = {n:p._version for n,p in model.named_parameters()}
    chosen = [next(s for s in selected_samples() if s['task']==t) for t in ['ChooseRel','QueryAttr']]
    checks=[]
    for sample in chosen:
        inputs=prepared(model,processor,sample)
        before=tensor_hashes(inputs)
        bos=processor.tokenizer.bos_token_id
        full, observed, _=forward(model,inputs,bos,observe=True,audit=True)
        query=sample['input_mapping']['p_minus_one']
        noop, no, _=forward(model,inputs,bos,layer=15,source='Prompt',noop=True,observe=True,audit=True)
        assert torch.equal(noop,full) and no.states==observed.states and no.rows==observed.rows
        plain,_,_=forward(model,inputs,bos)
        assert torch.equal(plain,full[0,query])
        interventions=[]
        for layer,source in [(0,'Image'),(15,'Prompt'),(31,'BOS')]:
            changed,trace,mask=forward(model,inputs,bos,layer,source,observe=True,audit=True)
            assert torch.equal(changed[0,:query],full[0,:query])
            assert trace.states==observed.states
            assert all(a['other_query_attention_sha256']==b['other_query_attention_sha256']
                for a,b in zip(trace.rows,observed.rows))
            assert trace.rows[layer][source+'_max']==0
            assert all(trace.rows[k]==observed.rows[k] for k in range(layer))
            interventions.append(dict(layer=layer,source=source,target_attention_exact_zero=True,
                all_other_query_logits_states_and_attention_exact=True,mask_records=mask.records))
        try:
            with SourceKnockout(model_layers(model),inputs,model.config.image_token_index,bos,15,'Prompt'):
                raise RuntimeError('deliberate cleanup smoke')
        except RuntimeError:
            pass
        restored,_,_=forward(model,inputs,bos)
        assert torch.equal(restored,plain) and before==tensor_hashes(inputs)
        assert footprint==hook_footprint(model) and versions=={n:p._version for n,p in model.named_parameters()}
        checks.append(dict(question_id=sample['question_id'],query=query,prompt_ids_sha256=digest(sample['input_mapping']['prompt_ids']),
            full_logits_sha256=tensor_hash(plain),noop_all_valid_logits_exact=True,plain_observed_logits_exact=True,
            all32_layers_collected=True,exception_cleanup=True,normal_cleanup=True,inputs_and_parameter_versions_unchanged=True,
            history=history_compare(sample,plain,observed.rows),conditions=interventions))
    atomic_json(OUT/f'shard{rank}/smoke.json',dict(passed=True,source=sources(),sample_manifest_sha256=sha256(OUT/'sample_manifest.json'),
        history_sha256=sha256(OUT/'history_bindings.json'),environment=environment,preflight=pre,checks=checks))
    print('Smoke passed GPU',rank+2,flush=True)


def freeze():
    code_check()
    smokes=[read(OUT/f'shard{k}/smoke.json') for k in range(2)]
    for s in smokes:
        assert s['passed'] and s['source']==sources() and s['sample_manifest_sha256']==sha256(OUT/'sample_manifest.json')
    old=read(ORIGINAL/'execution_manifest.json')
    for p,b in old['model_files'].items():
        assert sha256(p)==b['sha256']
    package=inspect.getfile(llama)
    freeze_json(OUT/'protocol.json',dict(status='frozen',base_commit=BASE,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source=sources(),
        model_revision=REVISION,model_files=old['model_files'],transformers='4.49.0',dtype='float16',attention='eager',
        transformers_source={package:sha256(package)},samples=240,conditions=96,layers=list(range(32)),sources=list(SOURCES),
        interventions='Original P-1 query only; pre-softmax -inf mask, one layer, one source; no decode/cache/generation.',
        prompt_definition='All valid text keys except BOS, image, and P-1 itself. Self is always retained.',
        historical_mass_definition='Old prompt_mass includes P-1. New blocked Prompt mass excludes self; record both and validate native old mass.',
        metrics='FP32 whole-vocabulary log_softmax; signed Full-top1 LogP loss, top1 flip, natural-log JS. Not gold loss or accuracy.',
        baseline='One paired same-GPU native Full per question, needed for exact distribution and self-mass audit. No full answer generation.',
        bindings={n:sha256(OUT/n) for n in ['sample_manifest.json','history_bindings.json','cpu_validation.json','shard0/smoke.json','shard1/smoke.json']},
        gpu_uuids=GPUS,bootstrap=dict(seed=SEED,repetitions=20000,unit='Whole image SHA256 cluster resampling; question-weighted ratio',
            intervals='Ordinary pointwise95%; exploratory96-condition/task scan, no uncorrected confirmatory significance claims'),
        limits='Only240 questions,32 layers,3sources,first-answer query; no accuracy, new interventions, training or expanded sample.'))
    print('Protocol frozen after both GPU smokes',flush=True)


def verify():
    c=read(OUT/'protocol.json')
    assert c['source']==sources() and c['gpu_uuids']==GPUS
    for name,h in c['bindings'].items(): assert sha256(OUT/name)==h
    for p,h in c['transformers_source'].items(): assert sha256(p)==h
    for p,b in c['model_files'].items():
        st=Path(p).stat()
        assert st.st_size==b['bytes'] and st.st_mtime_ns==b['mtime_ns']
    return c


def worker(rank):
    c=verify();binding=digest(c)
    manifest=read(OUT/'sample_manifest.json')
    samples=[s for s in manifest['samples'] if manifest['assignments'][s['question_id']]==rank]
    folder=OUT/f'shard{rank}';folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'completion.json').exists():
        assert read(folder/'completion.json')['binding']==binding
        print('Already complete; no GPU allocated',flush=True);return
    pre=preflight(rank)
    model,processor,environment=load_model(MODEL_PATH)
    assert environment['transformers']=='4.49.0' and environment['parameter_dtypes']==['torch.float16']
    assert environment['model_revision']==REVISION and not model.training
    versions={n:p._version for n,p in model.named_parameters()};footprint=hook_footprint(model)
    started=time.perf_counter();resumed=0
    for i,sample in enumerate(samples):
        path=folder/f"checkpoints/{sample['task']}__{sample['question_id']}.json"
        if path.exists():
            record=read(path)
            assert record['binding']==binding and record['sample_sha256']==digest(sample)
            assert record['result_sha256']==digest(record['result']) and len(record['result']['metrics'])==96
            resumed+=1;continue
        tick=time.perf_counter();inputs=prepared(model,processor,sample);before=tensor_hashes(inputs)
        bos=processor.tokenizer.bos_token_id
        full,observed,_=forward(model,inputs,bos,observe=True)
        full_path=folder/f"full_logits/{sample['task']}__{sample['question_id']}.npz"
        full_path.parent.mkdir(exist_ok=True)
        with full_path.open('wb') as f: np.savez_compressed(f,full_logits=full.numpy())
        changed=[];audits=[]
        for layer in range(32):
            for source in SOURCES:
                logits,_,mask=forward(model,inputs,bos,layer,source)
                changed.append(logits)
                audits.append(dict(layer=layer,source=source,logits_sha256=tensor_hash(logits),
                    masked_connections=next(r['blocked_connections_per_head'] for r in mask.records if r['applied'])))
        rows=[]
        oldmass=read(MASS/f"checkpoints/{sample['task']}__{sample['question_id']}.json")['result']['rows']
        for metric,audit in zip(metric_rows(full,torch.stack(changed)),audits):
            layer,source=audit['layer'],audit['source'];mass=observed.rows[layer]
            previous=oldmass[layer]
            rows.append(dict(**audit,**metric,logp_loss=-metric['delta_logp'],full_source_mass=mass[source],
                native_self_mass=mass['Self'],native_prompt_mass_including_self=mass['Prompt']+mass['Self'],
                historical_source_mass=previous[{'Image':'image_mass','BOS':'bos_mass','Prompt':'prompt_mass'}[source]],
                full_logits_sha256=tensor_hash(full)))
        assert before==tensor_hashes(inputs) and footprint==hook_footprint(model)
        assert versions=={n:p._version for n,p in model.named_parameters()}
        result=dict(metrics=rows,native_rows=observed.rows,history=history_compare(sample,full,observed.rows),
            full_logits_file=str(full_path.relative_to(OUT)),full_logits_file_sha256=sha256(full_path),
            seconds=time.perf_counter()-tick,hooks_clean=True,inputs_unchanged=True,parameters_versions_unchanged=True)
        atomic_json(path,dict(binding=binding,sample_sha256=digest(sample),task=sample['task'],question_id=sample['question_id'],
            image_id=sample['image_id'],image_sha256=sample['image_sha256'],physical_gpu=rank+2,result=result,result_sha256=digest(result)))
        if (i+1)%5==0 or i+1==len(samples):
            print(json.dumps(dict(gpu=rank+2,questions=i+1,total=len(samples),conditions=(i+1)*96,seconds=time.perf_counter()-started,
                last_question_seconds=result['seconds'])),flush=True)
    verify()
    atomic_json(folder/'completion.json',dict(status='complete',binding=binding,questions=len(samples),conditions=len(samples)*96,
        seconds=time.perf_counter()-started,resumed=resumed,environment=environment,preflight=pre,memory=gpu_memory(),
        hooks_clean=footprint==hook_footprint(model),parameters_unchanged=versions=={n:p._version for n,p in model.named_parameters()}))
    print('Complete GPU',rank+2,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','cpu','smoke','freeze','worker'])
    parser.add_argument('--rank',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    try:
        if args.stage in ['smoke','worker']: globals()[args.stage](args.rank)
        else: globals()[args.stage]()
    except BaseException:
        atomic_json(OUT/f'failures/{args.stage}_rank{args.rank}_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
