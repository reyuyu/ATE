"""RQ2.7: reuse audited P-1 and incremental visual masks on 60 frozen captions."""
import argparse
import csv
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

from benchmarks.dataset_registry import digest, sha256, freeze_json
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers, tensor_hash
from paper_rq1.engine import step_inputs
from paper_rq1_single_layer_gqa6.run import atomic_json, tensor_hashes
from rq2_decode_visual.engine import DecodeImageKnockout, DecodeProbabilities, BLOCKED
from rq2_source_knockout.engine import Observer
from rq2_visual_sufficiency.engine import MultiImageKnockout
from rq2_visual_handoff import run as history
from vlm_common import load_model, gpu_memory

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/rq2_sparse_visual_caption_60'
BASE = '4194b798339b7784209dc1f46360c8eb5e9a520d'
GPU = 'GPU-a63c6362-dcfd-4a26-c44b-014b40127c89'
KEYS = ['A', 'B', 'S', 'DS', 'DN']
NEW = ['S', 'DS', 'DN']
read = history.read


def layers(key, count):
    all_layers = list(range(count))
    sparse = [k for k in all_layers if k not in (14, 19)]
    return {'A': ([], []), 'B': (all_layers, all_layers), 'S': (sparse, sparse),
            'DS': ([], sparse), 'DN': ([], all_layers)}[key]


@torch.inference_mode()
def generate(model, inputs, bos, key, limit=96, noop=False, audit=False,
             collect=False, expected_visual=576):
    pre, dec = layers(key, len(model_layers(model)))
    n = inputs['input_ids'].shape[1]
    if limit < 1 or not bool(inputs['attention_mask'].all()):
        raise ValueError('Require a positive limit and one unpadded prompt')
    tokens, cache, rows, trace, debug = [], None, [], [], {}
    while len(tokens) < limit and (not tokens or tokens[-1] != 2):
        step = len(tokens)
        mask = (DecodeImageKnockout(model, inputs, dec, step, noop, expected_visual) if step
                else MultiImageKnockout(model, inputs, bos, pre, noop, expected_visual))
        obs = (DecodeProbabilities(model, mask) if step else
               Observer(model, inputs, bos, audit=audit, expected_visual=expected_visual))
        with mask, obs:
            output = model(**step_inputs(inputs, tokens), past_key_values=cache,
                           use_cache=True, output_attentions=True, return_dict=True)
        mask.assert_one_forward()
        assert not mask.handles and not obs.handles
        actual = [r['layer'] for r in mask.records if r['applied']]
        assert actual == ([] if noop else dec if step else pre)
        cache = output.past_key_values
        assert all(cache.get_seq_length(k) == n+step for k in range(len(model_layers(model))))
        for row in obs.rows:
            maximum = row['image_probability_max'] if step else row['Image_max']
            blocked = row['layer'] in actual
            assert maximum == 0. if blocked else maximum > 0.
            rows.append(dict(step=step, layer=row['layer'], blocked=blocked,
                image_probability_max=maximum,
                image_mass=row['image_mass'] if step else row['Image'],
                row_error=row['row_sum_max_error'] if step else row['row_error']))
        logits = output.logits[0, -1].detach().float().cpu().clone()
        assert torch.isfinite(logits).all()
        if collect:
            trace.append(logits)
        if not step:
            first = logits
            debug.update(prefill_rows=obs.rows, states=obs.states)
            if audit:
                debug['prefill_logits'] = output.logits.detach().float().cpu().clone()
        tokens.append(int(logits.argmax()))
    return dict(token_ids=tokens, eos=tokens[-1] == 2,
        truncated=len(tokens)==limit and tokens[-1]!=2, decode_steps=len(tokens)-1,
        own_prefix=True, own_cache=True, prefill_blocked=[] if noop else pre,
        decode_blocked=[] if noop else dec), first, rows, trace, debug


def sources():
    return dict(history.sources(), **{n: sha256(ROOT/n) for n in
        ['rq2_sparse_caption.py', 'rq2_sparse_caption_analysis.py', 'test_rq2_sparse_caption.py']})


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    old = history.verify()
    cm = read(history.OUT/'caption_manifest.json')
    samples = cm['samples']
    assert len(samples)==len({s['image_sha256'] for s in samples})==60
    assignments = read(history.OUT/'sample_manifest.json')['assignments']['caption']
    bindings = {}
    for s in samples:
        q = s['question_id']; bindings[q] = {}
        assert sha256(s['image_path'])==s['image_sha256']
        for key in ['A', 'B']:
            path = history.OUT/f"shard{assignments[q]}/caption/{q.replace(':','_')}/{key}.json"
            record = history.valid_checkpoint(path, digest(old), s, key)
            assert record['result']['generation']['eos']
            bindings[q][key] = dict(path=str(path.relative_to(ROOT)), sha256=sha256(path),
                raw_file=str((history.OUT/record['raw_file']).relative_to(ROOT)),
                raw_sha256=record['raw_sha256'], sample_sha256=digest(s))
    freeze_json(OUT/'sample_manifest.json', dict(samples=samples,
        historical_manifest_sha256=sha256(history.OUT/'caption_manifest.json')))
    freeze_json(OUT/'history_bindings.json', bindings)
    suite = unittest.defaultTestLoader.loadTestsFromName('test_rq2_sparse_caption')
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    atomic_json(OUT/'cpu_validation.json', dict(passed=result.wasSuccessful(),
        tests=result.testsRun, output=stream.getvalue()))
    assert result.wasSuccessful(), stream.getvalue()
    for p,b in old['model_files'].items():
        assert sha256(p)==b['sha256']
    freeze_json(OUT/'protocol.json', dict(base=BASE,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source=sources(), model_revision=old['model_revision'], model_files=old['model_files'],
        transformers_source=old['transformers_source'], scorer_source=cm['scorer_source_sha256'],
        bindings={n:sha256(OUT/n) for n in ['sample_manifest.json','history_bindings.json','cpu_validation.json']},
        gpu=dict(physical_gpu=0, uuid=GPU), conditions={k:dict(zip(['prefill_blocked','decode_blocked'],layers(k,32))) for k in KEYS},
        generation=dict(max_new_tokens=96,do_sample=False,eos_token_id=2,use_cache=True),
        dtype='float16', attention='eager', transformers='4.49.0', new_generations=180,
        comparisons=[['DS','A'],['DS','DN'],['S','B'],['DN','A']],
        bootstrap=dict(unit='image_sha256', repetitions=100000, seed=20261101,
            interval='ordinary descriptive paired percentile 95%; no NI/equivalence claim'),
        scores='Historical audited official CIDEr/SPICE, raw scale; BLEU omitted because not in frozen scorer.',
        recorded_start_epoch=start, worktree_created_epoch=(ROOT.parent/'.git').stat().st_mtime,
        clock_note='Host epoch timing; client task date 2026-11-01.',
        stop='Only 60 frozen captions and 3 added conditions; no search/pruning/training/speed claims.'))
    print('Frozen: 60 images, 180 new generations; CPU tests:', result.testsRun, flush=True)


def verify():
    p = read(OUT/'protocol.json'); assert p['source']==sources()
    for n,h in p['bindings'].items(): assert sha256(OUT/n)==h
    for n,h in dict(p['transformers_source'], **p['scorer_source']).items(): assert sha256(n)==h
    for n,b in p['model_files'].items():
        st=Path(n).stat(); assert st.st_size==b['bytes'] and st.st_mtime_ns==b['mtime_ns']
    return p


def historical(q, key):
    binding = read(OUT/'history_bindings.json')[q][key]
    assert sha256(ROOT/binding['path'])==binding['sha256']
    assert sha256(ROOT/binding['raw_file'])==binding['raw_sha256']
    r=read(ROOT/binding['path'])
    assert r['sample_sha256']==binding['sample_sha256']
    assert digest(r['result'])==r['result_sha256']
    with np.load(ROOT/binding['raw_file']) as raw: first=torch.from_numpy(raw['first_logits'].copy())
    return r['result']['generation'], first


def smoke(model, processor, sample):
    data=history.prompt(model,processor,sample); bos=processor.tokenizer.bos_token_id
    foot=hook_footprint(model); versions={n:p._version for n,p in model.named_parameters()}; before=tensor_hashes(data)
    a,first,_,trace,debug=generate(model,data,bos,'A',collect=True,audit=True)
    ha,hf=historical(sample['question_id'],'A')
    assert torch.equal(first,hf) and a['token_ids']==ha['token_ids']
    noop,nf,_,nt,nd=generate(model,data,bos,'S',noop=True,collect=True,audit=True)
    assert torch.equal(debug['prefill_logits'],nd['prefill_logits'])
    assert noop['token_ids']==a['token_ids'] and all(torch.equal(x,y) for x,y in zip(trace,nt))
    with torch.inference_mode():
        native=model.generate(**data, do_sample=False, max_new_tokens=96,use_cache=True)
    assert native[0,data['input_ids'].shape[1]:].tolist()==a['token_ids']
    b,bf,_,_,_=generate(model,data,bos,'B')
    hb,hbf=historical(sample['question_id'],'B')
    assert torch.equal(bf,hbf) and b['token_ids']==hb['token_ids']
    checks={}
    for key in NEW:
        g,f,rows,_,d=generate(model,data,bos,key,limit=4,audit=True)
        assert torch.equal(debug['prefill_logits'][:,:-1],d['prefill_logits'][:,:-1])
        assert debug['states']==d['states']
        assert [r['other_query_attention_sha256'] for r in debug['prefill_rows']]==[r['other_query_attention_sha256'] for r in d['prefill_rows']]
        if key in ['DS','DN']: assert torch.equal(f,first)
        assert g['decode_steps']>=1
        checks[key]=dict(first_logits_exact=bool(torch.equal(f,first)),
            non_p1_prefill_logits_states_attention_exact=True, probability_rows=rows)
    history.guard(model,data,foot,versions,before)
    atomic_json(OUT/'smoke.json',dict(passed=True,protocol_sha256=sha256(OUT/'protocol.json'),
        native_noop_all_logits_exact=True,historical_AB_full_generation_exact=True,
        native_HF_tokens_exact=True,checks=checks,hooks_clean=True,formal_generations=0,
        smoke_complete_generations=4,smoke_short_generations=3))


def checkpoint(path, binding, sample, key):
    r=read(path); assert r['binding']==binding and r['sample_sha256']==digest(sample) and r['condition']==key
    assert digest(r['generation'])==r['generation_sha256'] and sha256(OUT/r['raw_file'])==r['raw_sha256']
    return r


def worker():
    p=verify(); binding=digest(p)
    lock=(OUT/'worker.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'completion.json').exists():
        assert read(OUT/'completion.json')['binding']==binding
        print('Already complete; model not loaded'); return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0',GPU]
    table,apps=history.snapshot(); fields=next(r for r in table.splitlines() if r.startswith('0, ')).split(', ')
    assert fields[1]==GPU and int(fields[4])>=18000 and int(fields[5])==0
    assert not any(r.startswith(GPU) for r in apps.splitlines())
    load_start=time.time();model,processor,env=load_model(history.MODEL_PATH)
    assert env['model_revision']==p['model_revision'] and env['transformers']=='4.49.0'
    assert env['parameter_dtypes']==['torch.float16'] and not model.training
    samples=read(OUT/'sample_manifest.json')['samples']
    smoke_start=time.time(); smoke(model,processor,samples[0]); start=time.time()
    foot=hook_footprint(model); versions={n:p._version for n,p in model.named_parameters()}
    done=resumed=0
    for s in samples:
        inputs=history.prompt(model,processor,s); before=tensor_hashes(inputs)
        _,hf=historical(s['question_id'],'A')
        for key in NEW:
            path=OUT/f"checkpoints/{s['question_id'].replace(':','_')}/{key}.json"
            if path.exists(): checkpoint(path,binding,s,key); resumed+=1; done+=1; continue
            g,first,rows,_,_=generate(model,inputs,processor.tokenizer.bos_token_id,key)
            if key in ['DS','DN']: assert torch.equal(first,hf),'Historical Full mismatch; stop'
            g['text']=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()
            history.guard(model,inputs,foot,versions,before)
            raw=path.with_suffix('.npz');history.raw_save(raw,dict(first_logits=first.numpy()))
            atomic_json(path,dict(binding=binding,sample_sha256=digest(s),condition=key,
                generation=g,generation_sha256=digest(g),probability_rows=rows,
                raw_file=str(raw.relative_to(OUT)),raw_sha256=sha256(raw),
                first_logits_exact_historical_Full=bool(torch.equal(first,hf)),
                inputs_unchanged=True,parameters_unchanged=True,hooks_clean=True))
            done+=1
        if done%15==0: print(f'{done}/180 formal generations; {time.time()-start:.1f}s',flush=True)
    verify()
    atomic_json(OUT/'completion.json',dict(binding=binding,status='complete',generations=done,resumed=resumed,
        formal_seconds=time.time()-start,model_load_seconds=smoke_start-load_start,
        smoke_seconds=start-smoke_start,worker_seconds=time.time()-load_start,
        finished_epoch=time.time(),environment=env,memory=gpu_memory(),preflight=table,
        gpu_uuid=GPU,hooks_clean=foot==hook_footprint(model)))
    print('Completed; worker exiting and releasing GPU',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','worker']);args=parser.parse_args()
    try: globals()[args.stage]()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        atomic_json(OUT/f'failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
