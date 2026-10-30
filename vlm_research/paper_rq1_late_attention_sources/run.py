"""Frozen four-stage collection with strict native-prefix replay and checkpoints."""
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch
import transformers
import transformers.models.llama.modeling_llama as llama
from transformers import AutoTokenizer
from benchmarks.dataset_registry import digest,freeze_json,sha256
from benchmarks.smoke import hook_footprint
from paper_rq1.run import prepared,preflight
from paper_rq1_single_layer_gqa6.run import atomic_json,tensor_hashes
from paper_rq1_token_role_decode.run import sources as prior_sources, raw_save
from paper_rq1_token_role_decode.labels import align,select
from paper_rq1.analyze import write_csv
from vlm_common import load_model,gpu_memory
from .protocol import ROOT,OUT,OLD,RAW,MODEL_PATH,PROTOCOL,LAYERS,TYPES,conditions
from .engine import step,generate,metrics,partition,step_inputs,cache_fingerprint
from .scheduling import SHARDS, SHARD_GPU, configuration


def sources():
    result=prior_sources()
    names=['paper_rq1_late_attention_sources/'+n+'.py' for n in ['__init__','protocol','data','engine','run','launch','scheduling']]
    names+=['test_late_attention_sources.py','test_late_source_scheduling.py','paper_rq1_token_role_decode/README.md']
    result.update({n:sha256(ROOT/n) for n in names})
    # prior run/labels/protocol/engine are reused directly and already bound above.
    return result


def cpu():
    if (OUT/'execution_manifest.json').exists():raise ValueError('CPU acceptance already GPU-frozen')
    if (OUT/'cpu_validation.json').exists():
        archive=OUT/'audit';archive.mkdir(exist_ok=True);stamp=str(time.time_ns())
        for n in ['cpu_validation.json','cpu_tests.txt']:os.replace(OUT/n,archive/(stamp+'_'+n))
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_late_attention_sources','test_token_role_decode','test_late_source_scheduling'])
    with (OUT/'cpu_tests.txt').open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not result.wasSuccessful():raise ValueError('CPU tests failed; inspect cpu_tests.txt')
    freeze_json(OUT/'cpu_validation.json',dict(passed=True,tests=result.testsRun,source_sha256=sources(),log_sha256=sha256(OUT/'cpu_tests.txt')))


def freeze():
    cpu_result=json.loads((OUT/'cpu_validation.json').read_text())
    if not cpu_result['passed'] or cpu_result['source_sha256']!=sources():raise ValueError('CPU acceptance/source mismatch')
    names=['vlm_research/'+n for n in sources()]
    tracked=subprocess.check_output(['git','ls-files','--',*names],cwd=ROOT.parent,text=True).splitlines()
    changed=subprocess.check_output(['git','diff','HEAD','--name-only','--',*names],cwd=ROOT.parent,text=True)
    if set(tracked)!=set(names) or changed.strip():raise ValueError('Commit exact tested collector before freeze')
    model=json.loads((OLD/'execution_manifest.json').read_text())['model_files']
    for name,record in model.items():
        if sha256(name)!=record['sha256']:raise ValueError('Changed pinned model artifact')
    freeze_json(OUT/'execution_manifest.json',dict(protocol=PROTOCOL,scheduling=configuration(),source_sha256=sources(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        data_sha256=sha256(OUT/'data_manifest.json'),sample_sha256=sha256(OUT/'sample_manifest.json'),
        cpu_sha256=sha256(OUT/'cpu_validation.json'),model_files=model,
        transformers_source_sha256={inspect.getfile(llama):sha256(inspect.getfile(llama))}))


def verify():
    config=json.loads((OUT/'execution_manifest.json').read_text())
    if config['source_sha256']!=sources() or config['protocol']!=PROTOCOL or config['scheduling']!=configuration():raise ValueError('Frozen source/protocol/scheduling changed')
    for key,name in [('data_sha256','data_manifest.json'),('sample_sha256','sample_manifest.json'),('cpu_sha256','cpu_validation.json')]:
        if sha256(OUT/name)!=config[key]:raise ValueError('Frozen input changed')
    for name,h in config['transformers_source_sha256'].items():
        if sha256(name)!=h:raise ValueError('Installed Transformers changed')
    for name,binding in config['model_files'].items():
        stat=Path(name).stat()
        if stat.st_size!=binding['bytes'] or stat.st_mtime_ns!=binding['mtime_ns']:raise ValueError('Pinned model artifact changed')
    return config


def gate(stage):
    config=verify();results=[]
    for rank in SHARDS:
        r=json.loads((OUT/f'{stage}/shard{rank}/completion.json').read_text())
        if (r['status']!='complete' or r['binding']!=digest(config) or r['rank']!=rank
                or r['physical_gpu']!=SHARD_GPU[rank]):raise ValueError('Need four same-code logical shards on authorized three GPUs: '+stage)
        results.append(r)
    return results


def groups_for(model,prompt,t):
    return partition(prompt,t,model.config.image_token_index,model.config.text_config.bos_token_id)


def smoke(model,processor,sample,rank):
    prompt=prepared(model,processor,sample);inputs_before=tensor_hashes(prompt)
    result,native_logits=generate(model,prompt)
    with torch.inference_mode():reference=model.generate(**prompt,do_sample=False,use_cache=True,max_new_tokens=64)
    if reference[0,prompt['input_ids'].shape[-1]:].tolist()!=result['token_ids']:raise AssertionError('Native generate tokens differ')
    history=None;prefix=[];checks=[];timing=[]
    for t,token in enumerate(result['token_ids']):
        inputs=step_inputs(prompt,prefix);native,future,_=step(model,inputs,history)
        np.testing.assert_array_equal(native.numpy(),native_logits[t])
        if t in [1,4]:
            groups=groups_for(model,prompt,t);before=cache_fingerprint(history)
            observed,_,trace=step(model,inputs,history,groups,observe=True)
            np.testing.assert_array_equal(native.numpy(),observed.numpy())
            checks.append(dict(t=t,mode='Native-observer',exact=True,groups=groups,audits=trace['audits'],rows=trace['rows']))
            for layer in LAYERS:
                for mode in conditions(layer):
                    noop,_,_=step(model,inputs,history,groups,mode,layer,noop=True)
                    np.testing.assert_array_equal(noop.numpy(),native.numpy())
                    started=time.perf_counter()
                    changed,altered_cache,record=step(model,inputs,history,groups,mode,layer)
                    del altered_cache
                    timing.append(dict(t=t,mode=mode,layer=layer,seconds=time.perf_counter()-started))
                    if cache_fingerprint(history)!=before:raise AssertionError('Smoke cache contaminated')
                    checks.append(dict(t=t,mode=mode,layer=layer,noop_exact=True,parent_cache_exact=True,
                        edits=record['edits'],**metrics(native,changed)))
            after,_,_=step(model,inputs,history)
            np.testing.assert_array_equal(after.numpy(),native.numpy())
        prefix.append(token);history=future
    if {r['t'] for r in checks}!={1,4}:raise AssertionError('Disjoint Smoke must reach both real Decode steps')
    if tensor_hashes(prompt)!=inputs_before:raise AssertionError('Smoke input mutated')
    return dict(passed=True,native_generate_tokens_exact=True,replay_all_steps_exact=True,generation=result,checks=checks,timings=timing,
        memory=gpu_memory(),**raw_save(RAW/f'smoke/shard{rank}/native.npz',full_logits=native_logits))


def ablate(model,processor,sample,generation,selected):
    prompt=prepared(model,processor,sample);input_hashes=tensor_hashes(prompt)
    if sha256(generation['raw_path'])!=generation['raw_sha256']:raise ValueError('Native logits changed')
    with np.load(generation['raw_path']) as raw:expected=raw['full_logits']
    targets={r['generation_step']:r for r in selected}
    if len(targets)!=len(selected) or any(t<1 for t in targets):raise ValueError('Targets must be unique real Decode steps')
    history=None;prefix=[];rows=[];source_rows=[];audits=[];raw_logits=[]
    for t,token in enumerate(generation['token_ids']):
        inputs=step_inputs(prompt,prefix);native,future,_=step(model,inputs,history)
        np.testing.assert_array_equal(native.numpy(),expected[t])
        if int(native.argmax())!=token:raise AssertionError('Full replay token differs')
        if t in targets:
            target=targets[t];n=prompt['input_ids'].shape[1]
            if target['token_id']!=token or int(inputs['input_ids'][0,0])!=generation['token_ids'][t-1] or history.get_seq_length()!=n+t-1:
                raise AssertionError('Target/consumed token or cache position mismatch')
            groups=groups_for(model,prompt,t)
            if inputs['attention_mask'].shape[-1]!=n+t or int(inputs['cache_position'])!=n+t-1:
                raise AssertionError('Visible key/cache mapping mismatch')
            before=cache_fingerprint(history)
            observed,observer_cache,trace=step(model,inputs,history,groups,observe=True)
            del observer_cache
            np.testing.assert_array_equal(native.numpy(),observed.numpy())
            shared=dict(image_id=sample['image_identity'],image_sha256=sample['image_sha256'],word=target['word'],word_type=target['word_type'],
                generation_step=t,relative_generation_position=target['relative_generation_position'],token_length=target['token_length'],
                target_token_id=token,consumed_token_id=prefix[-1],caption_length=generation['length'],prompt_length=n,
                char_start=target['char_start'],char_end=target['char_end'])
            base=metrics(native,native)
            source_rows.extend(dict(shared,full_token_logp=base['full_token_logp'],full_token_entropy=base['full_token_entropy'],**r) for r in trace['rows'])
            for layer in LAYERS:
                rows.append(dict(shared,mode='Full',layer=layer,**base))
                for mode in conditions(layer):
                    started=time.perf_counter()
                    changed,changed_cache,_=step(model,inputs,history,groups,mode,layer)
                    del changed_cache
                    rows.append(dict(shared,mode=mode,layer=layer,seconds=time.perf_counter()-started,**metrics(native,changed)))
                    raw_logits.append(changed.numpy())
            after=cache_fingerprint(history)
            if after!=before:raise AssertionError('Formal sources contaminated Full history')
            audits.append(dict(t=t,consumed=prefix[-1],predicted=token,history_length=history.get_seq_length(),groups=groups,
                parent_sha256_before=before,parent_sha256_after=after,replay_logits_exact=True,observer_logits_exact=True,
                observer_numerical=trace['audits'],intervention_count=46))
        prefix.append(token);history=future
    if tensor_hashes(prompt)!=input_hashes:raise AssertionError('Input mutated by intervention')
    if len(audits)!=len(selected) or len(rows)!=56*len(selected) or len(source_rows)!=44*len(selected):raise AssertionError('Incomplete formal target')
    return dict(rows=rows,source_rows=source_rows,audits=audits,positions=len(selected)),np.stack(raw_logits) if raw_logits else np.empty((0,expected.shape[-1]),dtype=np.float32)


def verify_targets():
    target=json.loads((OUT/'target_freeze.json').read_text())
    if sha256(OUT/'selected_token_manifest.json')!=target['selected_sha256'] or target['binding']!=digest(verify()):raise ValueError('Frozen targets changed')
    return target


def worker(stage,rank):
    config=verify();binding=digest(config)
    if stage=='full':gate('smoke')
    if stage=='ablation':
        full_stage=gate('full');target=verify_targets()
        chosen=json.loads((OUT/'selected_token_manifest.json').read_text())
        if chosen['binding']!=binding or chosen['full_stage_digest']!=digest(full_stage):raise ValueError('Different native target trajectory')
    folder=OUT/f'{stage}/shard{rank}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    physical_gpu=SHARD_GPU[rank]
    pre=preflight(physical_gpu)
    if transformers.__version__!='4.49.0':raise ValueError('Wrong Transformers environment')
    model,processor,env=load_model(MODEL_PATH)
    env.update(binding=binding,code_commit=config['code_commit'],preflight=pre,logical_shard=rank,physical_gpu=physical_gpu)
    if (folder/'environment.json').exists():
        prior=json.loads((folder/'environment.json').read_text())
        if any(prior[k]!=env[k] for k in ['binding','code_commit','torch','transformers','model_revision','physical_gpu','cuda_visible_devices']):raise ValueError('Resume environment changed')
        freeze_json(folder/f'resume_{time.time_ns()}.json',env)
    else:freeze_json(folder/'environment.json',env)
    original=llama.eager_attention_forward;hooks=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    manifest=json.loads((OUT/'sample_manifest.json').read_text())
    samples=[manifest['smoke']] if stage=='smoke' else [r for r in manifest['samples'] if r['rank']==rank]
    started=time.perf_counter();complete=0
    for sample in samples:
        name=sample['image_identity'].replace(':','_');path=folder/f'checkpoints/{name}.json'
        if path.exists():
            r=json.loads(path.read_text())
            if r['binding']!=binding or r['sample_sha256']!=digest(sample) or sha256(r['raw_path'])!=r['raw_sha256']:raise ValueError('Invalid resume checkpoint')
            if stage=='ablation' and r['selected_sha256']!=target['selected_sha256']:raise ValueError('Resume target changed')
            complete+=1;continue
        try:
            if stage=='smoke':result=smoke(model,processor,sample,rank)
            elif stage=='full':
                prompt=prepared(model,processor,sample);input_before=tensor_hashes(prompt)
                result,logits=generate(model,prompt)
                if tensor_hashes(prompt)!=input_before:raise AssertionError('Full input mutated')
                result.update(caption=processor.tokenizer.decode(result['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False),
                    **raw_save(RAW/f'full/shard{rank}/{name}.npz',full_logits=logits))
            else:
                full_path=OUT/f'full/shard{rank}/checkpoints/{name}.json'
                if sha256(full_path)!=chosen['images'][sample['image_identity']]['full_checkpoint_sha256']:raise ValueError('Native checkpoint changed')
                generation=json.loads(full_path.read_text());selected=chosen['images'][sample['image_identity']]['selected']
                result,logits=ablate(model,processor,sample,generation,selected)
                result.update(selected_sha256=target['selected_sha256'],**raw_save(RAW/f'ablation/shard{rank}/{name}.npz',altered_logits=logits))
            if hook_footprint(model)!=hooks or llama.eager_attention_forward is not original:raise AssertionError('Leaked hooks/eager wrapper')
            result.update(binding=binding,sample_sha256=digest(sample),image_id=sample['image_identity'])
            atomic_json(path,result);complete+=1
            print(stage,rank,complete,'/',len(samples),name,flush=True)
        except BaseException as exc:
            atomic_json(folder/f'failures/{name}_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),binding=binding,skipped=False));raise
    if versions!={n:p._version for n,p in model.named_parameters()} or hooks!=hook_footprint(model):raise AssertionError('Weights/hooks changed')
    verify()
    if stage=='ablation':verify_targets()
    atomic_json(folder/'completion.json',dict(status='complete',binding=binding,rank=rank,physical_gpu=physical_gpu,expected=len(samples),completed=complete,
        wall_seconds=time.perf_counter()-started,weights_versions_unchanged=True,hooks_clean=True,memory=gpu_memory()))


def choose_targets():
    full=gate('full');config=verify()
    if any((OUT/f'ablation/shard{i}').exists() for i in range(4)):raise ValueError('Formal ablation preceded target freeze')
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,use_fast=True)
    samples=json.loads((OUT/'sample_manifest.json').read_text())['samples'];images={};captions=[];tokens=[]
    for sample in samples:
        path=OUT/f"full/shard{sample['rank']}/checkpoints/{sample['image_identity'].replace(':','_')}.json"
        generation=json.loads(path.read_text())
        if sha256(generation['raw_path'])!=generation['raw_sha256']:raise ValueError('Changed native raw logits')
        annotation=align(tokenizer,generation['token_ids']);selected=select(sample['image_sha256'],annotation,generation['length'])
        images[sample['image_identity']]=dict(image_sha256=sample['image_sha256'],full_checkpoint_sha256=sha256(path),
            caption=generation['caption'],annotation=annotation,selected=selected,
            missing_types=[role for role in TYPES if role not in [r['word_type'] for r in selected]])
        with np.load(generation['raw_path']) as f:lp=torch.from_numpy(f['full_logits']).float().log_softmax(-1)
        per_lp=lp.gather(-1,torch.tensor(generation['token_ids'])[:,None]).squeeze(-1).tolist()
        entropy=(-(lp.exp()*lp).sum(-1)).tolist()
        captions.append(dict(image_id=sample['image_identity'],rank=sample['rank'],image_sha256=sample['image_sha256'],caption=generation['caption'],
            token_ids=json.dumps(generation['token_ids']),token_logp=json.dumps(per_lp),token_entropy=json.dumps(entropy),length=generation['length'],
            eos=generation['eos'],truncated=generation['truncated'],generation_seconds=generation['seconds'],alignment_exact=annotation['roundtrip_exact']))
        tokens.extend(dict(image_id=sample['image_identity'],**r,full_logp=per_lp[r['generation_step']],entropy=entropy[r['generation_step']]) for r in annotation['tokens'])
    freeze_json(OUT/'selected_token_manifest.json',dict(binding=digest(config),full_stage_digest=digest(full),images=images,
        tokenizer_class=type(tokenizer).__name__,tokenizer_fast=tokenizer.is_fast,
        selection_rule=PROTOCOL['targets'],formal_ablation_outputs_present=False))
    write_csv(OUT/'full_caption_generations.csv',captions);write_csv(OUT/'full_token_annotations.csv',tokens)
    counts={role:sum(r['word_type']==role for value in images.values() for r in value['selected']) for role in TYPES}
    freeze_json(OUT/'target_freeze.json',dict(images=200,positions=sum(counts.values()),counts=counts,
        alignment_failures=sum(not value['annotation']['roundtrip_exact'] for value in images.values()),
        selected_sha256=sha256(OUT/'selected_token_manifest.json'),binding=digest(config),
        full_stage_digest=digest(full),frozen_before_formal_ablation=True))
    print('Frozen source targets',counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','select','smoke','full','ablation'])
    parser.add_argument('--rank',type=int,choices=range(4));args=parser.parse_args()
    if args.stage=='cpu':cpu()
    elif args.stage=='freeze':freeze()
    elif args.stage=='select':choose_targets()
    elif args.rank is None:parser.error('--rank required for GPU workers')
    else:worker(args.stage,args.rank)
