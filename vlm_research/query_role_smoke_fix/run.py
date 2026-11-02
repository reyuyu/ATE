"""No formal collection entrypoint: diagnose one question, then three frozen smokes."""
import argparse
from contextlib import AbstractContextManager
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import torch
import transformers
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from paper_rq1_single_layer_gqa6.run import prepared, tensor_hashes
from query_role_attention.common import (ROOT, OUT as OLD, MODEL, UUID, REVISION,
    read, write, sha256, digest, sources as role_sources, preflight)
from query_role_attention.engine import generate, roles, RoleBypass
from vlm_common import load_model, gpu_memory

OUT = ROOT/'results/query_role_smoke_logits_fix'
BASE = '094e1d7916d1e18351bffaf950c68280ba29bc85'


def sources():
    return {**role_sources(), **{str(p.relative_to(ROOT)): sha256(p)
        for p in sorted(Path(__file__).parent.glob('*.py'))}}


def historical_files():
    return {str(p.relative_to(OLD)): sha256(p) for p in sorted(OLD.rglob('*')) if p.is_file()}


def summary(a, b):
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    same = a.shape == b.shape
    delta = (a.double()-b.double()).abs() if same and finite else None
    return dict(left_shape=list(a.shape), right_shape=list(b.shape),
        left_dtype=str(a.dtype), right_dtype=str(b.dtype), finite=finite,
        exact=bool(torch.equal(a, b)),
        unequal_elements=int(torch.count_nonzero(a != b)) if same else None,
        max_absolute_error=float(delta.max()) if delta is not None and delta.numel() else None,
        mean_absolute_error=float(delta.mean()) if delta is not None and delta.numel() else None,
        left_top1=int(a.reshape(-1, a.shape[-1])[-1].argmax()),
        right_top1=int(b.reshape(-1, b.shape[-1])[-1].argmax()),
        top1_flip=int(a.reshape(-1,a.shape[-1])[-1].argmax()) != int(b.reshape(-1,b.shape[-1])[-1].argmax()))


class Recorder(AbstractContextManager):
    """Read-only model arguments, actual LM-head inputs, and raw output logits."""
    def __init__(self, model):
        self.model, self.handles, self.calls, self.hidden, self.logits = model, [], [], [], []

    def __enter__(self):
        def before(module, args, kw):
            values = {}
            for k,v in kw.items():
                if torch.is_tensor(v):
                    values[k] = dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_hashes({k:v})[k])
                    if k in ('cache_position','position_ids') or v.numel() < 16:
                        values[k]['values'] = v.detach().cpu().tolist()
                elif k == 'past_key_values':
                    values[k] = dict(type=type(v).__name__, length=v.get_seq_length() if v is not None else 0)
                elif v is None or isinstance(v,(bool,int,float,str)):
                    values[k] = v
                else:
                    values[k] = dict(type=type(v).__name__)
            self.calls.append(values)
        def head(module, args):
            self.hidden.append(args[0].detach().cpu().clone())
        def after(module, args, output):
            assert torch.isfinite(output.logits).all(), 'Nonfinite raw logits'
            self.logits.append(output.logits[0].detach().cpu().clone())
        try:
            self.handles.append(self.model.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(self.model.language_model.lm_head.register_forward_pre_hook(head))
            self.handles.append(self.model.register_forward_hook(after))
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*exc):
        for h in self.handles: h.remove()
        self.handles = []


def compare(label, qid, left, right, binding, require=True, effective=False):
    a,b = left.logits,right.logits
    rows = []
    for t,(x,y) in enumerate(zip(a,b)):
        if effective: x,y=x[-1],y[-1]
        row = dict(question_id=qid,t=t,phase='prefill' if t==0 else 'decode',
            left_forward=left.calls[t],right_forward=right.calls[t],
            lm_head_input_comparison=summary(left.hidden[t][:,-1],right.hidden[t][:,-1]),
            **summary(x,y))
        rows.append(row)
    result = dict(label=label,binding=binding,left_steps=len(a),right_steps=len(b),
        all_exact=len(a)==len(b) and all(r['exact'] and r['finite'] for r in rows),
        first_difference=next((r for r in rows if not r['exact']),None),steps=rows)
    path=OUT/'comparisons'/f'{qid}__{label}.json'
    write(path,result)
    if not result['all_exact']:
        # Small prediction vectors only, never complete Attention matrices.
        for t,(x,y) in enumerate(zip(a,b)):
            if not torch.equal(x[-1],y[-1]):
                array=OUT/'diagnostic_arrays'/f'{qid}__{label}__t{t}.pt'
                array.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(left=x[-1],right=y[-1]),array)
    if require:
        assert result['all_exact'], f'Strict logits mismatch; saved {path}'
    return result


@torch.inference_mode()
def native(model, prompt):
    with Recorder(model) as rec:
        ids=model.generate(**prompt,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True)
    return ids[0,prompt['input_ids'].shape[1]:].tolist(),rec


def custom(model,processor,prompt,condition='Full',noop=False,keep=1,cache=False):
    with Recorder(model) as rec:
        result=generate(model,processor.tokenizer,prompt,condition,noop=noop,
                        retain_cache=cache,logits_to_keep=keep)
    return result,rec


def cpu():
    assert not torch.cuda.is_initialized()
    stream=io.StringIO()
    suite=unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromName('vlm_research.query_role_attention.tests'),
        unittest.defaultTestLoader.loadTestsFromName('vlm_research.query_role_smoke_fix.tests')])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        output=stream.getvalue(),cuda_initialized=torch.cuda.is_initialized(),sources=sources()))
    print(stream.getvalue(),flush=True)
    assert result.wasSuccessful()


def freeze():
    acceptance=read(OUT/'cpu_validation.json')
    assert acceptance['passed'] and acceptance['sources']==sources()
    paths=['vlm_research/'+p for p in sources()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    tracked=subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines()
    assert set(paths)==set(tracked)
    old=read(OLD/'protocol.json')
    for path,info in old['model_files'].items():
        assert sha256(path)==info['sha256'],path
    for path,h in {**old['transformers_sources'],**old['scorer_sources']}.items():
        assert sha256(path)==h,path
    manifest=read(OLD/'sample_manifest.json')
    qids=[manifest['samples'][i]['question_id'] for i in manifest['smoke_indices']]
    assert qids==['00150746','11697551','11127723']
    write(OUT/'protocol.json',dict(base=BASE,code_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),sources=sources(),
        historical_files=historical_files(),historical_manifest_sha256=sha256(OLD/'sample_manifest.json'),
        model_revision=REVISION,model_files=old['model_files'],
        transformers_sources=old['transformers_sources'],scorer_sources=old['scorer_sources'],
        environment=dict(dtype='float16',attention='eager',transformers='4.49.0'),
        smoke_question_ids=qids,diagnosis_question='00150746',
        formal_generations_authorized=0,strict_equality='torch.equal; no tolerances',
        repair='Slice hidden states before lm_head using logits_to_keep=1, matching HF generate',
        scientific_intervention_unchanged=True,gpu=dict(physical=1,uuid=UUID,logical='cuda:0')))
    print('Frozen new repair sources/protocol; old artifacts unchanged',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    assert p['sources']==sources()
    assert p['historical_files']==historical_files()
    for path,h in {**p['transformers_sources'],**p['scorer_sources']}.items():
        assert sha256(path)==h
    return p


@torch.inference_mode()
def diagnose(model,processor,s,mapping,binding):
    qid=s['question_id']; assert qid=='00150746'
    prompt=prepared(model,processor,s)
    assert tensor_hashes(prompt)==mapping['input_hashes']
    binding=dict(protocol_sha256=binding,question_sha256=digest(s),input_hashes=tensor_hashes(prompt),
        image_sha256=s['image_sha256'],model_revision=REVISION)
    print('DIAGNOSE native HF',qid,flush=True)
    ids,hf=native(model,prompt)
    print('DIAGNOSE old all-position LM head',flush=True)
    old,allrows=custom(model,processor,prompt,keep=0)
    observed=compare('native_vs_all_positions',qid,hf,allrows,binding,require=False,effective=True)
    assert ids==old[0]['token_ids']
    print('DIAGNOSE matched last-query LM head',flush=True)
    fixed,last=custom(model,processor,prompt)
    aligned=compare('native_vs_fixed',qid,hf,last,binding,effective=True)
    assert ids==fixed[0]['token_ids']
    # Isolate LM-head matrix shape using the exact same captured hidden tensor.
    hidden=allrows.hidden[0].to(device='cuda:0')
    assert torch.equal(hidden[:,-1:].cpu(),hf.hidden[0])
    projected_all=model.language_model.lm_head(hidden)[0].cpu()
    projected_last=model.language_model.lm_head(hidden[:,-1:])[0].cpu()
    replay=dict(all_projection_matches_observed=summary(projected_all,allrows.logits[0]),
        last_projection_matches_native=summary(projected_last,hf.logits[0]),
        cross_shape_projection=summary(projected_all[-1],projected_last[-1]),
        identical_last_hidden=summary(allrows.hidden[0][:,-1:],hf.hidden[0]))
    write(OUT/'lm_head_isolation.json',replay)
    assert replay['all_projection_matches_observed']['exact']
    assert replay['last_projection_matches_native']['exact']
    assert replay['identical_last_hidden']['exact']
    # Every same-shape control is strict, including all valid prefill rows at keep=0.
    controls=[]
    for keep,reference in [(0,allrows),(1,last)]:
        for label,condition,noop in [('repeat','Full',False),('noop','A-Bypass',True)]:
            result,rec=custom(model,processor,prompt,condition,noop=noop,keep=keep)
            controls.append(compare(f'keep{keep}_{label}',qid,reference,rec,binding))
            assert result[0]['token_ids']==ids
    result=dict(root_cause_confirmed=not observed['all_exact'] and aligned['all_exact'] and
        not replay['cross_shape_projection']['exact'],native_vs_all_positions=observed,
        native_vs_fixed=aligned,lm_head_isolation=replay,controls=controls,
        same_shape_strict_equal=True,full_generation_tokens_exact=True,binding=binding)
    write(OUT/'logits_comparison.json',result)
    assert result['root_cause_confirmed'], 'Candidate cause not established; stop before three-question smoke'
    print('DIAGNOSIS PASS: identical last hidden; LM-head shape alone reproduces difference',flush=True)
    return result


@torch.inference_mode()
def smoke(model,processor,manifest,mappings,binding):
    rows=[]
    for i in manifest['smoke_indices']:
        s=manifest['samples'][i];qid=s['question_id']
        print('SMOKE BEGIN',qid,flush=True)
        inputs=prepared(model,processor,s)
        before=tensor_hashes(inputs); footprint=hook_footprint(model)
        assert before==mappings[i]['input_hashes'] and roles(inputs)==mappings[i]['role_layout']
        versions={n:p._version for n,p in model.named_parameters()}
        ids,hf=native(model,inputs)
        full,fullrec=custom(model,processor,inputs,cache=True)
        compare('smoke_native_fixed',qid,hf,fullrec,binding,effective=True)
        assert full[0]['token_ids']==ids
        parent=full[3];parent_hash=cache_fingerprint(parent)
        replay,reprec=custom(model,processor,inputs)
        compare('smoke_full_replay',qid,fullrec,reprec,binding)
        assert replay[0]['token_ids']==ids
        audits={}
        for condition in ('I-Bypass','R-Bypass','A-Bypass'):
            noop,nrec=custom(model,processor,inputs,condition,noop=True)
            compare('smoke_'+condition+'_noop',qid,fullrec,nrec,binding)
            assert noop[0]['token_ids']==ids
            counts=[0]*32;handles=[]
            try:
                for k,layer in enumerate(model_layers(model)):
                    def mlp(module,args,out,k=k):
                        assert torch.isfinite(out).all()
                        counts[k]+=1
                    handles.append(layer.mlp.register_forward_hook(mlp))
                changed,crec=custom(model,processor,inputs,condition,cache=True)
            finally:
                for h in handles:h.remove()
            generation,_,_,owned=changed
            assert counts==[generation['length']]*32
            assert cache_fingerprint(parent)==parent_hash
            assert all(x.data_ptr()!=y.data_ptr() for a,b in zip(parent.to_legacy_cache(),owned.to_legacy_cache())
                       for x,y in zip(a,b))
            assert hook_footprint(model)==footprint
            audits[condition]=dict(noop_all_returned_valid_logits_exact=True,
                target_exact_zero=True,non_target_rows_exact=True,
                independent_cache=True,parent_cache_unchanged=True,mlp_calls_per_layer=counts,
                cache_lengths=generation['cache_lengths'],events=generation['hook_events'],
                finite=True,eos=generation['eos'],length=generation['length'])
            del owned,changed,crec
        # Actual GPU exception-path cleanup without any extra forward.
        try:
            with RoleBypass(model,roles(inputs),'A-Bypass'):
                raise RuntimeError('intentional cleanup smoke')
        except RuntimeError as e:
            assert str(e)=='intentional cleanup smoke'
        assert hook_footprint(model)==footprint
        assert tensor_hashes(inputs)==before and versions=={n:p._version for n,p in model.named_parameters()}
        row=dict(question_id=qid,task=s['task'],passed=True,input_hashes=before,
            native_full_tokens_exact=True,native_full_logits_exact=True,
            full_repeat_exact=True,roles=audits,hooks_clean=True,exception_cleanup=True,
            inputs_parameters_unchanged=True,full_token_ids=ids,query_layout=roles(inputs))
        write(OUT/'smoke_checkpoints'/f'{qid}.json',row)
        rows.append(row)
        del parent,full,fullrec,hf,reprec,noop,nrec
        print('GPU SMOKE PASS',qid,flush=True)
    return rows


def gpu():
    p=verify();assert read(OUT/'cpu_validation.json')['passed']
    started=time.perf_counter()
    check=preflight();write(OUT/'gpu_preflight.json',check)
    model,processor,environment=load_model(MODEL)
    assert transformers.__version__=='4.49.0' and torch.cuda.device_count()==1
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader'],text=True)
    assert any(line.split(',')[0].strip()==UUID and int(line.split(',')[1])==os.getpid()
               for line in apps.splitlines())
    environment.update(pid=os.getpid(),physical_gpu=1,uuid=UUID,logical='cuda:0',
        process_mapping=apps,python_executable=os.sys.executable,
        forward_signature=str(inspect.signature(model.forward)),
        prepare_inputs_signature=str(inspect.signature(model.prepare_inputs_for_generation)))
    write(OUT/'environment.json',environment)
    manifest=read(OLD/'sample_manifest.json');mappings=read(OLD/'input_mappings.json')
    baseline=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    binding=digest(p)
    i=manifest['smoke_indices'][0]
    diagnose(model,processor,manifest['samples'][i],mappings[i],binding)
    assert hook_footprint(model)==baseline
    rows=smoke(model,processor,manifest,mappings,binding)
    assert versions=={n:p._version for n,p in model.named_parameters()} and hook_footprint(model)==baseline
    verify()
    write(OUT/'gpu_smoke_validation.json',dict(passed=True,questions=len(rows),rows=rows,
        seconds_including_load=time.perf_counter()-started,memory=gpu_memory(),
        no_nan_inf_oom=True,hooks_clean=True,parameters_unchanged=True,
        formal_generations=0,stopped_after_three_smokes=True,binding=binding))
    print('COMPLETE: 3/3 strict smokes; formal collection intentionally NOT started',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['cpu','freeze','gpu'])
    args=parser.parse_args()
    try:
        globals()[args.stage]()
    except BaseException:
        write(OUT/'failures'/f'{args.stage}_{time.time_ns()}.json',dict(
            stage=args.stage,traceback=traceback.format_exc(),formal_generations=0))
        raise
