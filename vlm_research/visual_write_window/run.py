"""Freeze, strict validation gate, 4320 serial generations and separate swaps."""
import argparse,fcntl,io,os,shutil,subprocess,time,traceback,unittest
import numpy as np
import torch,transformers
from transformers import AutoProcessor
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from query_role_attention.run import save_raw
from first_answer_attention.engine import first_metrics
from relay_source_causality.run import frozen_inputs
from vlm_common import load_model,gpu_memory
import query_role_smoke_fix.run as repaired
from .common import *
from .engine import generate,WindowMask

def cpu():
    stream=io.StringIO();suite=unittest.TestSuite()
    for name in ['query_role_attention.tests','query_role_smoke_fix.tests','query_role_formal.tests','relay_source_causality.tests','visual_write_window.tests']:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('vlm_research.'+name))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,output=stream.getvalue(),sources=sources(),cuda_initialized=torch.cuda.is_initialized()))
    print(stream.getvalue(),flush=True);assert result.wasSuccessful() and not torch.cuda.is_initialized()
    audit=audit_history();write(OUT/'historical_native_audit.json',audit)
    a=read(OLD/'sample_manifest.json')['samples'];h=ROOT/'results/rq2_visual_text_handoff_240';b=read(h/'sample_manifest.json')['samples']
    overlap=len({s['question_id'] for s in a}&{s['question_id'] for s in b})
    write(OUT/'historical_reuse_audit.json',dict(passed=True,reused_main_generations=0,candidate=str(h),question_overlap=overlap,
        condition='Handoff B has matching relay/answer source semantics',rejection='Only 20/240 questions overlap; old generation does not explicitly set logits_to_keep=1. No same-240 same-path All-Writer artifact found.',
        inspected_reports={str(p.relative_to(ROOT)):sha256(p) for p in [h/'report_zh.md',ROOT/'results/relay_source_causality_240/report_zh.md',ROOT/'results/query_role_ia_joint_240/report_zh.md']},
        native_Full_not_equivalent='Native Full permits P-1 and decode image access. Not an All-Writer baseline.',new_main_generations=4320))

def freeze():
    assert read(OUT/'cpu_validation.json')['passed'] and read(OUT/'cpu_validation.json')['sources']==sources()
    paths=['vlm_research/'+n for n in sources()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    old=read(OLD/'protocol.json')
    for name,info in old['model_files'].items():assert sha256(name)==info['sha256'],name
    for name in ['sample_manifest.json','input_mappings.json']:
        assert not (OUT/name).exists();shutil.copyfile(OLD/name,OUT/name)
    assert sha256(OUT/'sample_manifest.json')==MANIFEST_SHA
    shutil.copyfile(SWAP,OUT/'image_swap_manifest.json');pairs=read(SWAP)['pairs'];assert len(pairs)==40
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    processor.patch_size=14;processor.num_additional_image_tokens=1;processor.vision_feature_select_strategy='default';processor.tokenizer.padding_side='left'
    swapmap={}
    for pair in pairs:
        a,b=pair['samples'];assert a['question']==b['question'] and a['answer']!=b['answer'] and a['image_sha256']!=b['image_sha256']
        assert a['input_mapping']['prompt_ids']==b['input_mapping']['prompt_ids']
        for side,s in enumerate(pair['samples']):
            v=frozen_inputs(processor,s);assert v['image_sha256']==s['image_sha256'];swapmap[f"{pair['pair_id']}__{side}"]=v
    write(OUT/'image_swap_inputs.json',swapmap);gpu=select_gpu()
    names=['cpu_validation.json','historical_native_audit.json','historical_reuse_audit.json','sample_manifest.json','input_mappings.json','image_swap_manifest.json','image_swap_inputs.json','gpu_selection.json']
    p=dict(base_commit=BASE,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),sources=sources(),
        bindings={n:sha256(OUT/n) for n in names},model_files=old['model_files'],model_revision=REVISION,model_path=str(MODEL),
        transformers_sources=old['transformers_sources'],scorer_sources=old['scorer_sources'],generation=old['generation'],dtype='float16',attention='eager',transformers='4.49.0',
        dependencies=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines(),conditions={c:writer_layers(c) for c in CONDITIONS},windows=WINDOWS,
        answer_rule='Every condition: P-1 and every real decode query, including EOS prediction, cannot read original image keys at all 32 layers.',
        relay_rule='R = every original nonvisual query after final image token and strictly before P-1. Only frozen writer layers allow R->original image keys.',
        mask='Clone native causal additive mask; only legal selected query->original 576 image key entries become -inf. No o_proj ablation. Other queries, keys, MLP, residual, RoPE and KV native.',
        actual_probabilities='Observe actual eager returned post-softmax matrices in each layer and real forward; forbidden max exactly zero; existing ROW_ATOL=0.002. No tolerance relaxed.',
        main=dict(questions=240,new_generations=4320,reused_generations=0,independent_cache=True,conditions_serial=True),
        swap=dict(pairs=40,conditions=['No-Writer'],new_generations=80,independent_main_statistics=True,
            acceptance='Same text IDs, different image SHA/gold. Complete generated tokens and every effective logits vector must be torch.equal; maximum difference zero. Failure stops main collection.'),
        smoke=dict(question_ids=['00150746','11697551','11127723'],all_18_conditions=True,Full_Noop_HF='Same logits_to_keep=1 shape, strict torch.equal on all effective logits'),
        statistics=dict(seed=20261102,repetitions=20000,cluster='Image SHA256',interval='Exploratory two-sided percentile 95%',same_draws=True,
            primary_sufficiency=[f'Only-W{k} - No-Writer' for k in range(8)],primary_necessity=[f'All-Writer - Drop-W{k}' for k in range(8)],
            descriptive='Each condition vs All-Writer and vs No-Writer; six tasks; no confirmatory multiplicity, equivalence or NI claim'),
        manifest_note=old['manifest_note'],gpu=gpu,exploratory=True,stop='No patching, new model, router, kernel, KV pruning or extra conditions')
    write(OUT/'protocol.json',p);verify();print('FROZEN',p['code_commit'],'4320 main + 80 independent No-Writer swap',flush=True)

def custom(model,processor,inputs,c,**kw):
    with repaired.Recorder(model) as rec:r=generate(model,processor.tokenizer,inputs,c,**kw)
    assert all(call['logits_to_keep']==1 for call in rec.calls)
    return r,rec

@torch.inference_mode()
def smoke(model,processor,manifest,mapping,binding):
    repaired.OUT=OUT;rows=[]
    for qid in ['00150746','11697551','11127723']:
        i=next(i for i,s in enumerate(manifest['samples']) if s['question_id']==qid);s=manifest['samples'][i]
        inputs=prepared(model,processor,s);before=tensor_hashes(inputs);assert before==mapping[i]['input_hashes']
        footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
        ids,hf=repaired.native(model,inputs);native,nrec=custom(model,processor,inputs,'Native',retain_cache=True)
        repaired.compare('HF_Native',qid,hf,nrec,binding);assert ids==native[0]['token_ids']
        parent=native[3];parent_hash=cache_fingerprint(parent)
        for c in ['All-Writer','No-Writer','Only-W0','Only-W7','Drop-W0','Drop-W7']:
            noop,rec=custom(model,processor,inputs,c,noop=True)
            repaired.compare(c+'_noop',qid,nrec,rec,binding);assert noop[0]['token_ids']==ids
        saved=read(OLD/f"checkpoints/{s['task']}__{qid}/Full.json")
        with np.load(saved['raw']['path']) as a:raw=torch.from_numpy(a['first_logits'].copy())
        assert torch.equal(native[1],raw) and native[0]['token_ids']==saved['generation']['token_ids']
        checks={}
        for c in CONDITIONS:
            counts=[0]*32;handles=[]
            try:
                for k,layer in enumerate(model_layers(model)):
                    def mlp(module,args,out,k=k):assert torch.isfinite(out).all();counts[k]+=1
                    handles.append(layer.mlp.register_forward_hook(mlp))
                changed,_=custom(model,processor,inputs,c,retain_cache=True)
            finally:
                for h in handles:h.remove()
            g=changed[0];owned=changed[3];assert counts==[g['length']]*32
            assert cache_fingerprint(parent)==parent_hash
            assert all(x.data_ptr()!=y.data_ptr() for a,b in zip(parent.to_legacy_cache(),owned.to_legacy_cache()) for x,y in zip(a,b))
            assert all(e['blocked_probability_max']==0 and e['answer_image_masked'] for e in g['source_events'])
            assert all(e['R_image_masked']==(e['layer'] not in writer_layers(c)) for e in g['source_events'][:32])
            assert hook_footprint(model)==footprint and tensor_hashes(inputs)==before
            checks[c]=dict(generation=g,all_32_mlp_calls=counts,cache_independent=True,parent_cache_unchanged=True)
            del owned,changed
        try:
            with WindowMask(model,inputs,'No-Writer'):raise RuntimeError('intentional cleanup')
        except RuntimeError as e:assert str(e)=='intentional cleanup'
        assert hook_footprint(model)==footprint and versions=={n:p._version for n,p in model.named_parameters()}
        row=dict(question_id=qid,passed=True,strict_HF_Full_noop_equal=True,historical_native_exact=True,all_conditions=checks,hook_cleanup_normal_exception=True,input_weights_unchanged=True)
        write(OUT/'smoke_checkpoints'/f'{qid}.json',row);rows.append(dict(question_id=qid,passed=True,conditions=18))
        print('STRICT SMOKE PASS',qid,'18 conditions',flush=True)
    return dict(passed=True,rows=rows,binding=binding,strict_torch_equal=True)

def record(model,processor,s,inputs,c,p,binding,path,rawpath,reference=None,capture=False):
    assert not path.exists(),'No overwriting or automatic repeated generation'
    before=tensor_hashes(inputs);footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    g,first,trace,_=generate(model,processor.tokenizer,inputs,c,capture=capture)
    g.update(correct=int(g['prediction']==s['answer']),first_logits_sha256=digest(first.tolist()))
    if reference is not None:g.update(first_metrics(reference,first))
    raw=save_raw(rawpath,first)
    assert tensor_hashes(inputs)==before and hook_footprint(model)==footprint and versions=={n:p._version for n,p in model.named_parameters()}
    row=dict(binding=binding,code_commit=p['code_commit'],sample_sha256=digest(s),question_id=s['question_id'],task=s['task'],gold=s['answer'],image_sha256=s['image_sha256'],
        input_hashes=before,condition=c,generation=g,result_sha256=digest(g),raw=raw)
    write(path,row);return row,first,trace

def gpu():
    start=time.perf_counter();p=verify();binding=digest(p)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight(p['gpu']);assert transformers.__version__=='4.49.0'
    model,processor,env=load_model(MODEL);assert torch.cuda.device_count()==1
    env.update(pid=os.getpid(),gpu=p['gpu'],python_executable=os.sys.executable);write(OUT/'environment.json',env)
    m=read(OUT/'sample_manifest.json');mapping=read(OUT/'input_mappings.json')
    write(OUT/'gpu_smoke_validation.json',smoke(model,processor,m,mapping,binding));verify()
    # Run the stricter image-independence control before the costly matrix.
    sm=read(OUT/'image_swap_inputs.json');pairs=[]
    for pair in read(OUT/'image_swap_manifest.json')['pairs']:
        records=[];traces=[]
        for side,s in enumerate(pair['samples']):
            key=f"{pair['pair_id']}__{side}";inputs=prepared(model,processor,s);assert tensor_hashes(inputs)==sm[key]['input_hashes']
            row,_,trace=record(model,processor,s,inputs,'No-Writer',p,binding,OUT/f'swap_checkpoints/{key}/No-Writer.json',RAW/f'swap/{key}.npz',capture=True)
            records.append(row);traces.append(trace)
        exact=len(traces[0])==len(traces[1]) and all(torch.equal(a,b) for a,b in zip(*traces))
        assert exact and records[0]['generation']['token_ids']==records[1]['generation']['token_ids'],'No-Writer image-independence failed; stop'
        pairs.append(dict(pair_id=pair['pair_id'],passed=True,every_effective_logits_equal=True,max_absolute_difference=0.,same_generated_tokens=True))
    write(OUT/'image_independence_validation.json',dict(passed=True,pairs=pairs,binding=binding,strict_torch_equal=True))
    print('GATE PASS: 3 smoke / all18 conditions and No-Writer 40 swap pairs exact; main starts',flush=True)
    for i,s in enumerate(m['samples']):
        inputs=prepared(model,processor,s);assert tensor_hashes(inputs)==mapping[i]['input_hashes'];ref=None
        for c in CONDITIONS:
            stem=f"{s['task']}__{s['question_id']}/{c}"
            _,v,_=record(model,processor,s,inputs,c,p,binding,OUT/f'checkpoints/{stem}.json',RAW/f'main/{stem}.npz',ref)
            if c=='All-Writer':ref=v
        if (i+1)%10==0:print('MAIN PROGRESS',(i+1)*18,'/4320','seconds',round(time.perf_counter()-start,1),flush=True)
    verify();assert len(list((OUT/'checkpoints').glob('*/*.json')))==4320 and len(list((OUT/'swap_checkpoints').glob('*/*.json')))==80
    write(OUT/'gpu_completion.json',dict(passed=True,binding=binding,pid=os.getpid(),new_main_generations=4320,reused_main_generations=0,swap_generations=80,model_replicas=1,seconds=time.perf_counter()-start,memory=gpu_memory()))
    print('COLLECTION COMPLETE; process exits',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['cpu','freeze','gpu']);args=a.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        write(OUT/'failures'/f'{args.stage}_{time.time_ns()}.json',dict(status='PARTIAL',stage=args.stage,traceback=traceback.format_exc(),hardware=hardware()))
        write(OUT/'completion.json',dict(status='PARTIAL',complete=False,failed_stage=args.stage));raise
