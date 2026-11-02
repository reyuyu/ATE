"""One idle GPU, one model, strict smoke gate, exactly 240 new joint answers."""
import argparse
import fcntl
import io
import os
import shutil
import subprocess
import time
import traceback
import unittest
import numpy as np
import torch
import transformers
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from query_role_attention.engine import generate as historical_generate
from query_role_attention.run import save_raw
from query_role_formal.common import divergence,repeated_trigram
import query_role_smoke_fix.run as repaired
from vlm_common import load_model,gpu_memory
from .common import *
from .engine import generate,roles,RoleBypass

def cpu():
    assert not torch.cuda.is_initialized()
    stream=io.StringIO();suite=unittest.TestSuite()
    for name in ['query_role_attention.tests','query_role_smoke_fix.tests','query_role_formal.tests','query_role_ia_joint.tests']:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('vlm_research.'+name))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,output=stream.getvalue(),sources=sources(),cuda_initialized=torch.cuda.is_initialized()))
    print(stream.getvalue(),flush=True)
    assert result.wasSuccessful() and not torch.cuda.is_initialized()
    audit=audit_history();write(OUT/'historical_baseline_validation.json',audit)
    print('Historical 720 baseline records, raw arrays and all delivery hashes verified',flush=True)

def freeze():
    acceptance=read(OUT/'cpu_validation.json');assert acceptance['passed'] and acceptance['sources']==sources()
    assert read(OUT/'historical_baseline_validation.json')['passed']
    paths=['vlm_research/'+n for n in sources()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    old=read(OLD/'protocol.json')
    for path,info in old['model_files'].items():assert sha256(path)==info['sha256'],path
    for name in ['sample_manifest.json','input_mappings.json']:
        assert not (OUT/name).exists();shutil.copyfile(OLD/name,OUT/name)
    gpu=select_gpu()
    protocol=dict(base_commit=BASE,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        sources=sources(),history_artifacts=history_hashes(),historical_protocol_sha256=sha256(OLD/'protocol.json'),
        bindings={n:sha256(OUT/n) for n in ['sample_manifest.json','input_mappings.json','cpu_validation.json','historical_baseline_validation.json','gpu_selection.json']},
        model_files=old['model_files'],model_revision=REVISION,model_path=str(MODEL),transformers_sources=old['transformers_sources'],scorer_sources=old['scorer_sources'],
        dtype='float16',attention='eager',transformers='4.49.0',processor='Same frozen local processor/tokenizer files, patch_size and num_additional_image_tokens=1, vision_feature_select_strategy=default, left padding; input tensors must match historical hashes',
        generation=old['generation'],conditions=CONDITIONS,new_condition=JOINT,new_formal_generations=240,reused_baseline_generations=720,layers=LAYERS,
        intervention='Only self_attn.o_proj OUTPUT rows: same prefill union I and original P-1; every actual decode query including EOS prediction. Other queries and R not directly edited. No attention mask, QKV, KV, MLP, residual or parameter edits.',
        retained_R_boundary='No direct zero at R; downstream R hidden states and KV can change indirectly. No assertion of equality to Full.',
        comparisons=['I+A - Full','I+A - I','I+A - A','interaction: joint-I-A+Full'],
        unique_harm='Full, I and A all correct, joint wrong',bootstrap=dict(seed=SEED,repetitions=20000,cluster='Image SHA256',same_draws_all_contrasts=True,interval='Exploratory two-sided ordinary percentile 95%'),
        smoke_question_ids=['00150746','11697551','11127723'],strict_equality='torch.equal, same LM-head shape logits_to_keep=1; no tolerances',
        manifest_note=old['manifest_note'],gpu=gpu,scoring=old['scoring'],exploratory=True,noninferiority_claim=False,no_physical_skipping=True,
        other_joint_experiment='e005bb5c5cb5e14ca6eff46181c5d238b2ae879c is A-Skip + Text-Only source mask, not this I+A output ablation')
    write(OUT/'protocol.json',protocol);verify();print('Frozen protocol, source commit, model, inputs, historical bindings and GPU UUID',flush=True)

def custom(model,processor,inputs,c,**kw):
    with repaired.Recorder(model) as rec:
        result=generate(model,processor.tokenizer,inputs,c,logits_to_keep=1,**kw)
    assert all(call['logits_to_keep']==1 for call in rec.calls)
    return result,rec

@torch.inference_mode()
def smoke(model,processor,manifest,mapping,binding):
    repaired.OUT=OUT;rows=[]
    for i in manifest['smoke_indices']:
        s=manifest['samples'][i];qid=s['question_id'];inputs=prepared(model,processor,s)
        before=tensor_hashes(inputs);footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
        assert before==mapping[i]['input_hashes'] and roles(inputs)==mapping[i]['role_layout']
        ids,native=repaired.native(model,inputs)
        full,frec=custom(model,processor,inputs,'Full',retain_cache=True)
        repaired.compare('native_full',qid,native,frec,binding)
        assert ids==full[0]['token_ids']
        noop,nrec=custom(model,processor,inputs,JOINT,noop=True)
        repaired.compare('joint_noop_full',qid,frec,nrec,binding)
        assert noop[0]['token_ids']==ids
        parent=full[3];parent_hash=cache_fingerprint(parent)
        parity=[]
        for c in BASELINES:
            with repaired.Recorder(model) as oldrec:
                old=historical_generate(model,processor.tokenizer,inputs,c,logits_to_keep=1)
            new,nr=custom(model,processor,inputs,c)
            repaired.compare('historical_path_'+c,qid,oldrec,nr,binding)
            history=read(checkpoint(s,c))
            with np.load(history['raw']['path']) as data:raw=torch.from_numpy(data['first_logits'].copy())
            exact=torch.equal(new[1],raw) and new[0]['token_ids']==history['generation']['token_ids']
            write(OUT/'comparisons'/f'{qid}__historical_saved_{c}.json',dict(exact=exact,current_sha256=digest(new[1].tolist()),historical_sha256=history['generation']['first_logits_sha256'],raw_path=history['raw']['path'],raw_sha256=sha256(history['raw']['path'])))
            assert exact,'Saved historical baseline mismatch; stop before formal collection'
            parity.append(dict(condition=c,all_step_same_shape_logits_exact=True,saved_first_logits_and_tokens_exact=True))
        counts=[0]*32;handles=[]
        try:
            for k,layer in enumerate(model_layers(model)):
                def mlp(module,args,out,k=k):
                    assert torch.isfinite(out).all();counts[k]+=1
                handles.append(layer.mlp.register_forward_hook(mlp))
            joint,jrec=custom(model,processor,inputs,JOINT,retain_cache=True)
        finally:
            for handle in handles:handle.remove()
        g=joint[0];owned=joint[3]
        assert counts==[g['length']]*32
        assert cache_fingerprint(parent)==parent_hash
        assert all(x.data_ptr()!=y.data_ptr() for a,b in zip(parent.to_legacy_cache(),owned.to_legacy_cache()) for x,y in zip(a,b))
        assert len(g['hook_events'])==4*g['length']
        assert all(e['target_count']==577 and e['phase']=='prefill' for e in g['hook_events'][:4])
        assert all(e['target_count']==1 and e['phase']=='decode' for e in g['hook_events'][4:])
        assert all(e['target_exact_zero'] and e['non_target_rows_exact'] and e['layer'] in LAYERS for e in g['hook_events'])
        try:
            with RoleBypass(model,roles(inputs),JOINT):raise RuntimeError('intentional cleanup smoke')
        except RuntimeError as e:assert str(e)=='intentional cleanup smoke'
        assert hook_footprint(model)==footprint and tensor_hashes(inputs)==before
        assert versions=={n:p._version for n,p in model.named_parameters()}
        row=dict(question_id=qid,passed=True,parity=parity,full_native_noop_all_valid_logits_exact=True,joint_generation=g,
            independent_cache=True,parent_cache_unchanged=True,all_32_mlp_calls=counts,R_not_directly_modified=True,
            other_layers_not_directly_modified=True,inputs_parameters_unchanged=True,hooks_clean=True,exception_cleanup=True,no_nan_inf_oom=True)
        write(OUT/'smoke_checkpoints'/f'{qid}.json',row);rows.append(row)
        del owned,parent,full,joint,jrec
        print('STRICT GPU SMOKE PASS',qid,flush=True)
    return dict(passed=True,questions=3,rows=rows,binding=binding,no_tolerance_relaxation=True,formal_generations=0)

def gpu():
    started=time.perf_counter();p=verify();binding=digest(p)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight(p['gpu']);assert transformers.__version__=='4.49.0'
    model,processor,environment=load_model(MODEL)
    assert torch.cuda.device_count()==1 and environment['parameter_dtypes']==['torch.float16']
    apps=hardware()['compute_processes']
    assert any(x.split(',')[0].strip()==p['gpu']['uuid'] and int(x.split(',')[1])==os.getpid() for x in apps.splitlines())
    environment.update(pid=os.getpid(),gpu=p['gpu'],python_executable=os.sys.executable,
        dependencies=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines())
    write(OUT/'environment.json',environment)
    manifest=read(OUT/'sample_manifest.json');mapping=read(OUT/'input_mappings.json')
    versions={n:v._version for n,v in model.named_parameters()};footprint=hook_footprint(model)
    acceptance=smoke(model,processor,manifest,mapping,binding)
    write(OUT/'gpu_smoke_validation.json',acceptance)
    verify();formal_started=time.perf_counter();computed=resumed=0
    print('GATE PASS: starting exactly 240 new I+A formal generations',flush=True)
    for i,s in enumerate(manifest['samples']):
        path=OUT/f"checkpoints/{s['task']}__{s['question_id']}/{JOINT}.json"
        if path.exists():
            row=read(path);assert row['binding']==binding and row['result_sha256']==digest(row['generation'])
            assert sha256(row['raw']['path'])==row['raw']['sha256'];resumed+=1;continue
        prompt=prepared(model,processor,s)
        assert tensor_hashes(prompt)==mapping[i]['input_hashes'] and roles(prompt)==mapping[i]['role_layout']
        full=read(checkpoint(s,'Full'))
        with np.load(full['raw']['path']) as raw:full_logits=torch.from_numpy(raw['first_logits'].copy())
        assert sha256(full['raw']['path'])==full['raw']['sha256']
        tick=time.perf_counter();g,first,_,_=generate(model,processor.tokenizer,prompt,JOINT,logits_to_keep=1)
        g.update(first_metrics(full_logits,first),correct=int(g['prediction']==s['answer']),
            full_correct_to_wrong=int(full['generation']['correct'] and g['prediction']!=s['answer']),
            full_wrong_to_correct=int(not full['generation']['correct'] and g['prediction']==s['answer']),
            sequence_changed=g['token_ids']!=full['generation']['token_ids'],first_divergence=divergence(full['generation']['token_ids'],g['token_ids']),
            repeated_trigram=repeated_trigram(g['token_ids']),full_first_logits_sha256=digest(full_logits.tolist()),first_logits_sha256=digest(first.tolist()))
        raw=save_raw(RAW/f"{s['task']}__{s['question_id']}/{JOINT}.npz",first)
        assert tensor_hashes(prompt)==mapping[i]['input_hashes'] and hook_footprint(model)==footprint
        assert versions=={n:v._version for n,v in model.named_parameters()}
        write(path,dict(binding=binding,code_commit=p['code_commit'],sample_sha256=digest(s),question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],gold=s['answer'],
            input_hashes=mapping[i]['input_hashes'],prompt_sha256=s['input_mapping']['chat_prompt_sha256'],condition=JOINT,generation=g,result_sha256=digest(g),raw=raw,seconds=time.perf_counter()-tick))
        computed+=1
        if (i+1)%20==0:print('PROGRESS',i+1,'/240',round(time.perf_counter()-formal_started,2),'seconds',flush=True)
    verify();assert len(list((OUT/'checkpoints').glob('*/*.json')))==240
    write(OUT/'gpu_completion.json',dict(passed=True,binding=binding,pid=os.getpid(),new_condition=JOINT,formal_generations=240,computed=computed,resumed=resumed,model_replicas=1,
        formal_seconds=time.perf_counter()-formal_started,total_seconds=time.perf_counter()-started,memory=gpu_memory(),gpu=p['gpu'],hooks_clean=hook_footprint(model)==footprint,parameters_unchanged=True))
    print('240/240 collected; GPU process exiting',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','gpu']);args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        write(OUT/'failures'/f'{args.stage}_{time.time_ns()}.json',dict(status='PARTIAL',stage=args.stage,traceback=traceback.format_exc(),hardware=hardware(),checkpoints=len(list((OUT/'checkpoints').glob('*/*.json')))))
        write(OUT/'completion.json',dict(status='PARTIAL',complete=False,failed_stage=args.stage))
        raise
