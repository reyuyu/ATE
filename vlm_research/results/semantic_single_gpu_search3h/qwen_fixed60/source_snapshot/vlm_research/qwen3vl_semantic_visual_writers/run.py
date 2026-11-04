"""CPU audit, immutable freeze, strict Smoke and one native-model GPU worker."""
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .common import *
from .engine import roles, WriterMask, generate
from ..semantic_visual_writers.groups import token_groups,select,STOP
from ..qwen3vl_early_windows.experiment import prepare,hook_counts
from ..qwen3vl_query_role.engine import generate as native_generate,feature_hashes
from ..qwen3vl_query_role.run import environment,check_models


def layout_comparison(a,b):
    fields={k:a['layout'][k]==b['layout'][k] for k in ['input_ids','image_positions','image_grid_thw']}
    fields['mrope']=a['mrope']==b['mrope']
    return dict(same_layout=all(fields.values()),differing_fields=[k for k,v in fields.items() if not v])


def make_spec(processor,config,question,layout):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    mapping=roles(layout,processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'))
    expanded=layout['prompt'].replace(processor.image_token,processor.image_token*len(mapping['I']))
    assert '<image>' not in expanded
    groups,qpos,words=token_groups(processor.tokenizer,question,expanded,layout['input_ids'],mapping['I'],mapping['R'])
    assert qpos==layout['question_positions']
    selection=select(groups,qpos,mapping['R'],question)
    ids=torch.tensor([layout['input_ids']])
    positions,delta=Qwen3VLModel.get_rope_index(SimpleNamespace(config=config),ids,
        torch.tensor([layout['image_grid_thw']]),attention_mask=torch.ones_like(ids))
    spec=dict(layout=layout,roles=mapping,selection=selection,words=words,
        mrope=dict(position_ids_sha256=tensor_digest(positions),delta=int(delta.item())),
        writers_decoded={c:processor.tokenizer.decode([layout['input_ids'][p] for p in selected])
            for c,selected in selection['writers'].items()})
    spec['binding']=digest(spec)
    return spec


def cpu():
    from . import tests
    from ..qwen3vl_query_role import tests as historical_tests
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),
        unittest.defaultTestLoader.loadTestsFromModule(historical_tests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and not torch.cuda.is_initialized()
    path=ROOT/'vlm_research/semantic_visual_writers/groups.py'
    original=command('git','show',LLAVA_COMMIT+':vlm_research/semantic_visual_writers/groups.py')+'\n'
    assert sha256(path)==SELECTOR_SHA and path.read_text()==original
    write(OUT/'cpu_tests.json',dict(passed=True,tests=result.testsRun,sources=sources(),
        selector_byte_identical_to_llava=True,selector_sha256=sha256(path),gpu_allocations=0,cuda_initialized=False))


def freeze():
    from transformers import AutoProcessor,AutoConfig
    import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
    import transformers.masking_utils as masks
    import transformers.generation.utils as generation
    assert not (OUT/'protocol.json').exists()
    assert read(OUT/'cpu_tests.json')['passed'] and read(OUT/'cpu_tests.json')['sources']==sources()
    assert sha256(LLAVA/'sample_manifest.json')==MANIFEST_SHA
    copies={'sample_manifest.json':LLAVA/'sample_manifest.json',
        'image_swap_manifest.json':LLAVA/'image_swap_manifest.json',
        'model_manifest.json':HISTORY/'model_manifest.json',
        'llava_per_question.csv':LLAVA/'per_question.csv',
        'llava_accuracy_summary.csv':LLAVA/'accuracy_summary.csv',
        'llava_paired_comparisons.csv':LLAVA/'paired_comparisons.csv',
        'llava_image_swap_summary.csv':LLAVA/'image_swap_summary.csv',
        'llava_token_group_manifest.json':LLAVA/'token_group_manifest.json'}
    for name,path in copies.items():
        if path.parent==LLAVA:
            blob=__import__('subprocess').check_output(['git','show',LLAVA_COMMIT+':vlm_research/results/semantic_visual_writers_240/'+path.name])
            assert hashlib.sha256(blob).hexdigest()==sha256(path),path
        shutil.copyfile(path,OUT/name)
    samples=read(OUT/'sample_manifest.json')['samples']
    assert len(samples)==240 and len({s['image_sha256'] for s in samples})==240
    assert sorted(Counter(s['task'] for s in samples).values())==[40]*6
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    config=AutoConfig.from_pretrained(MODEL,local_files_only=True)
    old={key(s):s for s in read(HISTORY/'qwen_input_manifest.json')['samples']}
    main={}
    for i,s in enumerate(samples):
        _,layout=prepare(processor,s)
        assert layout==old[key(s)]['layout'], 'Historical native input changed'
        main[key(s)]=make_spec(processor,config,s['question'],layout)
        if (i+1)%40==0:
            print('PREPARED',i+1,'/240',flush=True)
    pairs=read(OUT/'image_swap_manifest.json')['pairs']
    assert len(pairs)==40
    swaps,comparison={},[]
    for pair in pairs:
        a,b=pair['samples']
        assert a['question']==b['question'] and a['answer']!=b['answer'] and a['image_sha256']!=b['image_sha256']
        specs=[]
        for side,s in enumerate(pair['samples']):
            _,layout=prepare(processor,s)
            spec=make_spec(processor,config,s['question'],layout)
            swaps[f"{pair['pair_id']}__{side}"]=spec
            specs.append(spec)
        for c in CONDITIONS:
            ids=[[sp['layout']['input_ids'][p] for p in sp['selection']['writers'][c]] for sp in specs]
            assert ids[0]==ids[1], 'Same text must select same token groups, even with different image layouts'
        comparison.append(dict(pair_id=pair['pair_id'],**layout_comparison(*specs)))
    write(OUT/'token_group_manifest.json',dict(main=main,swaps=swaps))
    write(OUT/'swap_layout_manifest.json',dict(pairs=comparison,
        same_layout=sum(r['same_layout'] for r in comparison),different_layout=sum(not r['same_layout'] for r in comparison),
        rule='Strict No-Writer all-step equality only if original input IDs, image positions, grid and native mRoPE all match'))
    budgets=[sp['selection'] for sp in main.values()]
    write(OUT/'token_budget_audit.json',dict(passed=True,questions=240,
        k_min=min(s['k'] for s in budgets),k_max=max(s['k'] for s in budgets),k_mean=float(np.mean([s['k'] for s in budgets])),
        reduced_for_whole_word_suffix=sum(s['budget_reduction_for_whole_word_suffix']>0 for s in budgets),
        scaffold_deficits=sum(s['scaffold_budget_deficit']>0 for s in budgets),
        image_tokens_min=min(len(s['roles']['I']) for s in main.values()),
        image_tokens_max=max(len(s['roles']['I']) for s in main.values()),
        selector_sha256=SELECTOR_SHA,no_gold_or_model_outputs=True,all_matched_controls_whole_word=True))
    check_models(read(OUT/'model_manifest.json'))
    env=environment()
    assert env==read(HISTORY/'environment.json')
    write(OUT/'environment.json',env)
    write(OUT/'dependencies.json',dict(pip_freeze=command(sys.executable,'-m','pip','freeze')))
    paths=[inspect.getfile(m) for m in [modeling,masks,generation]]
    artifacts=['cpu_tests.json',*copies,'token_group_manifest.json','swap_layout_manifest.json',
        'token_budget_audit.json','environment.json','dependencies.json']
    gpu=select_gpu()
    ordered=sorted(main,key=lambda k:(main[k]['layout']['input_length'],k))
    protocol=dict(title='Qwen3-VL Semantic Visual Writers / fixed 240',base_commit=BASE,llava_commit=LLAVA_COMMIT,
        code_commit=command('git','rev-parse','HEAD'),sources=sources(),artifacts={n:sha256(OUT/n) for n in artifacts},
        official_sources={p:sha256(p) for p in paths},model_revision=REVISION,model_path=str(MODEL),
        dtype='bfloat16',attention='eager',transformers='4.57.6',model_layers=36,vocab_size=config.text_config.vocab_size,
        generation=dict(greedy=True,max_new_tokens=32,logits_to_keep=1,independent_real_KV=True),
        conditions=CONDITIONS,main_records=2400,new_main_generations=2400,swap_conditions=SWAP_CONDITIONS,swap_generations=400,
        gpu=gpu,smoke_keys=[ordered[0],ordered[len(ordered)//2],ordered[-1]],
        role_definition='I = native image-pad positions, variable length; R = every original position after final I and before P-1, INCLUDING vision_end and native template special tokens',
        historical_role_boundary_note='Earlier o_proj query-role R excluded vision_end. Source-writer masks include it to avoid an uncontrolled post-image visual relay; no old All-Writer result is reused.',
        mask='All36 layers: block P-1 and all real decode queries from original I; block non-selected R -> I. Only pre-softmax hard mask. Native causal/pad mask and all other paths retained.',
        selector=dict(sha256=SELECTOR_SHA,stopwords=sorted(STOP),random_seeds=RANDOM_SEEDS,
            unchanged_algorithm=True,native_token_offsets=True,
            budget='Largest feasible whole-word semantic budget matching exact Question/R tail and seeded random controls; Scaffold deficit retained; Drop = selected Semantic complement',
            random='Question-text keyed permutation of whole groups plus unchanged exact-budget DP; not independent uniformly sampled subwords'),
        no_op='All returned valid logits torch.equal, native logits_to_keep=1, never relaxed',
        probability_audit='Prohibited probabilities exactly zero; FP32 softmax cast to native BF16, row-sum bound fixed at torch.finfo(bfloat16).eps, not a logit tolerance',
        swaps='Frozen40 pairs; same-layout No-Writer strict all-step torch.equal gate; different-grid/mRoPE pairs report position confounding separately and are never called visual leakage solely from non-equality',
        inference=dict(seed=SEED,repetitions=20000,cluster='Image SHA256',primary=PRIMARY,
            correction='Ordinary exploratory95% and Bonferroni7 intervals for seven prespecified contrasts',
            random_mean='Average three correctness indicators within each question before paired cluster resampling',
            cross_model='Descriptive same-question paired comparison; separate original tokenizer, prompt and BF16/FP16 environments, not pure architecture identification'),
        stop='No training, layer/token search, router, activation patching or speedup implementation')
    write(OUT/'protocol.json',protocol)
    print('FROZEN',sha256(OUT/'protocol.json'),'GPU',gpu,flush=True)


def checked_inputs(processor,s,spec):
    inputs,layout=prepare(processor,s)
    assert layout==spec['layout']
    return {k:v.to('cuda:0') for k,v in inputs.items()}


def cache_fingerprint(cache):
    return digest([[tensor_digest(layer.keys),tensor_digest(layer.values)] for layer in cache.layers])


def smoke(model,processor,samples,specs,p):
    by={key(s):s for s in samples}
    cases=[]
    for identifier in p['smoke_keys']:
        s,spec=by[identifier],specs[identifier]
        inputs=checked_inputs(processor,s,spec)
        before={k:tensor_digest(v) for k,v in inputs.items()}
        weights=[(n,v._version,v.data_ptr()) for n,v in model.named_parameters()]
        original,steps,observer,parent=native_generate(model,processor,inputs,spec['roles'],native=True,audit=True,trace=True)
        observed_features=feature_hashes(observer)
        old=read(HISTORY/'checkpoints'/identifier/'Full.json')
        assert original['token_ids']==old['token_ids']
        assert sha256(old['raw_first_logits_path'])==old['raw_first_logits_sha256']
        assert np.array_equal(steps[0].numpy(),np.load(old['raw_first_logits_path'],allow_pickle=False))
        parent_hash=cache_fingerprint(parent)
        checks=[]
        for c in ['All-Writer','No-Writer','Semantic-k']:
            g,scores,trace,own=generate(model,processor,inputs,spec,c,noop=True,audit=True,trace=True)
            assert g['token_ids']==original['token_ids'] and len(scores)==len(steps)
            assert all(torch.equal(a,b) for a,b in zip(scores,steps))
            assert feature_hashes(trace)==observed_features
            assert parent is not own and all(a.keys.data_ptr()!=b.keys.data_ptr() and a.values.data_ptr()!=b.values.data_ptr() for a,b in zip(parent.layers,own.layers))
            assert cache_fingerprint(parent)==parent_hash
            del own,trace,scores
        for c in CONDITIONS:
            g,scores,trace,own=generate(model,processor,inputs,spec,c,audit=True,trace=True)
            assert feature_hashes(trace)==observed_features
            assert trace.validate() and trace.mlp_calls==[len(g['token_ids'])]*36
            assert cache_fingerprint(parent)==parent_hash
            assert parent is not own and all(a.keys.data_ptr()!=b.keys.data_ptr() and a.values.data_ptr()!=b.values.data_ptr() for a,b in zip(parent.layers,own.layers))
            checks.append(dict(condition=c,mask_zero=True,cache_independent=True,native_deepstack_mrope=True,
                row_sum_max_error=max(e['row_sum_max_error'] for e in g['source_events'])))
            del own,trace,scores
        counts=hook_counts(model)
        try:
            with WriterMask(model,spec,[]):
                raise RuntimeError('intentional cleanup')
        except RuntimeError as error:
            assert str(error)=='intentional cleanup'
        assert hook_counts(model)==counts
        assert before=={k:tensor_digest(v) for k,v in inputs.items()}
        assert weights==[(n,v._version,v.data_ptr()) for n,v in model.named_parameters()]
        cases.append(dict(key=identifier,passed=True,native_noop_all_valid_logits_exact=True,
            historical_native_full_exact=True,independent_cache_storage_parent_unchanged=True,
            DeepStack_MLP_residual_validated=True,native_mrope_all_steps=True,exception_cleanup=True,checks=checks))
        del parent,steps,observer,inputs
        gc.collect()
        torch.cuda.empty_cache()
        print('SMOKE PASS',identifier,'10 conditions',flush=True)
    return dict(passed=True,cases=cases,no_tolerance_relaxation=True)


def record(model,processor,s,spec,condition,p,stem,swap=False,inputs=None):
    path=checkpoint(stem,condition,swap)
    if path.exists():
        old=read(path)
        assert old['binding']==digest(p) and old['sample_sha256']==digest(s) and old['input_binding']==spec['binding']
        assert digest(old['generation'])==old['result_sha256']
        raw(old)
        return old
    if inputs is None:
        inputs=checked_inputs(processor,s,spec)
    before={k:tensor_digest(v) for k,v in inputs.items()}
    weights=[(n,v._version,v.data_ptr()) for n,v in model.named_parameters()]
    capture=swap and condition=='No-Writer'
    g,scores,_,cache=generate(model,processor,inputs,spec,condition,audit=capture)
    assert before=={k:tensor_digest(v) for k,v in inputs.items()}
    assert weights==[(n,v._version,v.data_ptr()) for n,v in model.named_parameters()]
    g.update(correct=int(g['prediction']==s['answer'].lower()),first_logits_sha256=digest(scores[0].tolist()))
    row=dict(binding=digest(p),sample_sha256=digest(s),input_binding=spec['binding'],condition=condition,
        question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],gold=s['answer'],
        generation=g,result_sha256=digest(g),raw=save_raw(stem,condition,scores[0],scores if capture else None,swap))
    write(path,row)
    del cache
    return row


def execute():
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    p=verify()
    assert environment()==read(OUT/'environment.json')
    assert command('git','status','--porcelain','--','vlm_research/qwen3vl_semantic_visual_writers','vlm_research/semantic_visual_writers')==''
    preflight(p['gpu'])
    check_models(read(OUT/'model_manifest.json'))
    started=time.perf_counter()
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
    assert torch.cuda.device_count()==1 and len(model.model.language_model.layers)==36
    assert {v.dtype for v in model.parameters()}=={torch.bfloat16}
    assert {str(v.device) for v in model.parameters()}=={'cuda:0'}
    assert model.config.text_config._attn_implementation==model.config.vision_config._attn_implementation=='eager'
    write(OUT/f'execution_{os.getpid()}.json',dict(pid=os.getpid(),gpu=p['gpu'],code_commit=command('git','rev-parse','HEAD'),binding=digest(p)))
    samples=read(OUT/'sample_manifest.json')['samples']
    specs=read(OUT/'token_group_manifest.json')
    if not (OUT/'gpu_smoke_validation.json').exists():
        write(OUT/'gpu_smoke_validation.json',smoke(model,processor,samples,specs['main'],p))
    assert read(OUT/'gpu_smoke_validation.json')['passed']
    pairs=read(OUT/'image_swap_manifest.json')['pairs']
    layouts={r['pair_id']:r for r in read(OUT/'swap_layout_manifest.json')['pairs']}
    independence=[]
    for pair in pairs:
        values=[]
        for side,s in enumerate(pair['samples']):
            stem=f"{pair['pair_id']}__{side}"
            r=record(model,processor,s,specs['swaps'][stem],'No-Writer',p,stem,True)
            values.append(raw(r,True))
        exact=np.array_equal(*values)
        same=layouts[pair['pair_id']]['same_layout']
        if same:
            assert torch.equal(torch.from_numpy(values[0]),torch.from_numpy(values[1])), 'Matched-layout No-Writer visual invariance failed; stop'
        independence.append(dict(pair_id=pair['pair_id'],same_layout=same,all_valid_logits_exact=exact,
            positional_confound=not same,strict_check_required=same))
    write(OUT/'image_independence_validation.json',dict(passed=True,pairs=independence,
        same_layout_strict_pass=sum(r['same_layout'] and r['all_valid_logits_exact'] for r in independence),
        confounded_pairs=sum(r['positional_confound'] for r in independence)))
    print('GATE PASS: Smoke and layout-aware No-Writer controls',flush=True)
    for pair in pairs:
        for side,s in enumerate(pair['samples']):
            stem=f"{pair['pair_id']}__{side}"
            inputs=checked_inputs(processor,s,specs['swaps'][stem])
            for c in SWAP_CONDITIONS:
                record(model,processor,s,specs['swaps'][stem],c,p,stem,True,inputs)
            del inputs
    print('SWAPS COMPLETE 400/400',flush=True)
    formal=time.perf_counter()
    for i,s in enumerate(samples):
        inputs=checked_inputs(processor,s,specs['main'][key(s)])
        for c in CONDITIONS:
            record(model,processor,s,specs['main'][key(s)],c,p,key(s),inputs=inputs)
        del inputs
        if (i+1)%20==0:
            print(f'MAIN {(i+1)*10}/2400 elapsed={time.perf_counter()-formal:.1f}s',flush=True)
    verify()
    check_models(read(OUT/'model_manifest.json'))
    assert len(list((OUT/'checkpoints').glob('*/*.json')))==2400
    assert len(list((OUT/'swap_checkpoints').glob('*/*.json')))==400
    write(OUT/'generation_completion.json',dict(passed=True,binding=digest(p),pid=os.getpid(),
        main_records=2400,new_main_generations=2400,swap_generations=400,
        code_commit=command('git','rev-parse','HEAD'),seconds=time.perf_counter()-started,
        formal_seconds=time.perf_counter()-formal,model_full_hash_rechecked=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    print('COLLECTION COMPLETE 2400 main /400 swaps; model exits normally',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['cpu','freeze','execute'])
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        with (OUT/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            globals()[args.stage]()
    except BaseException:
        write(OUT/'failures'/f'{args.stage}_{time.time_ns()}.json',dict(status='PARTIAL',stage=args.stage,traceback=traceback.format_exc()))
        raise
