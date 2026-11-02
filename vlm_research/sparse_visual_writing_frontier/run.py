"""CPU audit, freeze, strict compatibility smoke, then 1440 new answers."""
import argparse,fcntl,io,os,shutil,subprocess,time,traceback,unittest
import numpy as np
import torch,transformers
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from query_role_attention.run import save_raw
from vlm_common import load_model,gpu_memory
from visual_write_window.engine import generate as historical_generate
import query_role_smoke_fix.run as repaired
from .common import *
from .engine import generate,WindowMask

def cpu():
    stream=io.StringIO();suite=unittest.TestSuite()
    for name in ['query_role_attention.tests','query_role_smoke_fix.tests','query_role_formal.tests','relay_source_causality.tests','visual_write_window.tests','sparse_visual_writing_frontier.tests']:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('vlm_research.'+name))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,output=stream.getvalue(),sources=sources(),cuda_initialized=torch.cuda.is_initialized()))
    print(stream.getvalue(),flush=True);assert result.wasSuccessful() and not torch.cuda.is_initialized()
    write(OUT/'historical_reuse_audit.json',audit_history());print('Audited 960 reusable frontier baselines and separate native Full',flush=True)

def freeze():
    a=read(OUT/'cpu_validation.json');assert a['passed'] and a['sources']==sources()
    assert read(OUT/'historical_reuse_audit.json')['passed']
    paths=['vlm_research/'+n for n in sources()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    old=read(OLD/'protocol.json')
    for name,info in old['model_files'].items():assert sha256(name)==info['sha256'],name
    for name in ['sample_manifest.json','input_mappings.json']:
        assert not (OUT/name).exists();shutil.copyfile(OLD/name,OUT/name)
    assert sha256(OUT/'sample_manifest.json')==MANIFEST_SHA
    gpu=select_gpu();names=['cpu_validation.json','historical_reuse_audit.json','sample_manifest.json','input_mappings.json','gpu_selection.json']
    p=dict(base_commit=BASE,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),sources=sources(),
        bindings={n:sha256(OUT/n) for n in names},historical_protocol_sha256=sha256(OLD/'protocol.json'),historical_completion_sha256=sha256(OLD/'completion.json'),
        model_files=old['model_files'],model_revision=REVISION,model_path=str(MODEL),dtype='float16',attention='eager',transformers='4.49.0',
        transformers_sources=old['transformers_sources'],scorer_sources=old['scorer_sources'],generation=old['generation'],
        dependencies=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines(),conditions=KEEP,new_conditions=NEW,reused_conditions=BASELINES,
        main=dict(questions=240,new_generations=1440,reused_generations=960,independent_cache=True,conditions_serial=True),
        answer_rule='P-1 and every real decode query including EOS prediction blocked from original Image Keys at all 32 layers in every frontier condition.',
        relay_rule='All original nonimage queries strictly after final image token and before P-1. R->Image open only at the explicit kept layers.',
        hard_mask_implementation='Call frozen visual_write_window.engine.apply_mask at each layer, selecting historical All-Writer on kept layers and No-Writer elsewhere. No historical source edits or output ablation.',
        probability_validation='Actual eager post-softmax forbidden maximum exactly zero in every layer/forward; existing row-sum bound 0.002; BOS/Self and non-target native mask retained.',
        comparisons=[dict(a=c,b='All-Writer',role='new_vs_all' if c in NEW else 'historical_reference') for c in CONDITIONS if c!='All-Writer']+
                    [dict(a=a,b=b,role='same_layer_count') for a,b in SAME_COUNT],
        statistics=dict(seed=20261102,repetitions=20000,cluster='Image SHA256',same_draws=True,interval='Exploratory two-sided percentile 95%',no_confirmatory_multiplicity=True),
        quality_bounds=dict(minus_one_pp=-1.,minus_two_pp=-2.,rule='Candidate iff observed accuracy difference versus All-Writer is at least bound. Point-estimate screen only, not non-inferiority; CI crossing zero proves nothing.'),
        frontier='Point Pareto frontier across ten conditions using kept-layer count and full-answer accuracy. Ties at identical count and accuracy retained; uncertainty not removed. Native Full excluded.',
        native_full_reference=dict(reused_records=240,correct=206,separate_protocol=True,not_pooled=True,reason='Native answer queries can read image keys; All-Writer cannot.'),
        smoke=dict(question_ids=['00150746','11697551','11127723'],all_six_new_conditions=True,all_four_reused_baselines=True,
            strict='HF/Native/all six no-ops match all effective logits by torch.equal with logits_to_keep=1. Old/new engines match all effective baseline logits; saved baseline first vectors and tokens exact.'),
        manifest_note=old['manifest_note'],gpu=gpu,exploratory_historical_cohort=True,no_actual_acceleration=True,
        stop='No extra search, joint token ablation, router, kernel, new model or dataset.')
    write(OUT/'protocol.json',p);verify();print('FROZEN',p['code_commit'],'1440 new / 960 reused / native reference separate',flush=True)

def custom(model,processor,inputs,c,**kw):
    with repaired.Recorder(model) as rec:result=generate(model,processor.tokenizer,inputs,c,**kw)
    assert all(v['logits_to_keep']==1 for v in rec.calls)
    return result,rec

@torch.inference_mode()
def smoke(model,processor,manifest,mapping,binding):
    repaired.OUT=OUT;rows=[]
    for qid in ['00150746','11697551','11127723']:
        i=next(i for i,s in enumerate(manifest['samples']) if s['question_id']==qid);s=manifest['samples'][i]
        inputs=prepared(model,processor,s);before=tensor_hashes(inputs);assert before==mapping[i]['input_hashes']
        footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
        ids,hf=repaired.native(model,inputs);native,nrec=custom(model,processor,inputs,'Native',retain_cache=True)
        repaired.compare('HF_Native',qid,hf,nrec,binding);assert native[0]['token_ids']==ids
        saved=read(NATIVE/f"checkpoints/{s['task']}__{qid}/Full.json")
        with np.load(saved['raw']['path']) as z:v=torch.from_numpy(z['first_logits'].copy())
        assert torch.equal(v,native[1]) and saved['generation']['token_ids']==ids
        parent=native[3];parent_hash=cache_fingerprint(parent)
        for c in NEW:
            noop,rec=custom(model,processor,inputs,c,noop=True)
            repaired.compare(c+'_noop',qid,nrec,rec,binding);assert noop[0]['token_ids']==ids
        history={}
        for c in BASELINES:
            with repaired.Recorder(model) as oldrec:old=historical_generate(model,processor.tokenizer,inputs,c)
            changed,rec=custom(model,processor,inputs,c);repaired.compare(c+'_old_new',qid,oldrec,rec,binding)
            saved=read(checkpoint(s,c))
            with np.load(saved['raw']['path']) as z:v=torch.from_numpy(z['first_logits'].copy())
            assert torch.equal(changed[1],v) and changed[0]['token_ids']==old[0]['token_ids']==saved['generation']['token_ids']
            history[c]=dict(saved_first_and_tokens_exact=True,old_new_all_effective_logits_exact=True)
        checks={}
        for c in NEW:
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
            assert all(e['R_image_masked']==(e['layer'] not in KEEP[c]) for e in g['source_events'][:32])
            assert hook_footprint(model)==footprint and tensor_hashes(inputs)==before
            checks[c]=dict(generation=g,all_32_mlp_calls=counts,cache_independent=True,parent_cache_unchanged=True)
            del owned,changed
        try:
            with WindowMask(model,inputs,NEW[0]):raise RuntimeError('intentional cleanup')
        except RuntimeError as e:assert str(e)=='intentional cleanup'
        assert hook_footprint(model)==footprint and versions=={n:p._version for n,p in model.named_parameters()}
        write(OUT/'smoke_checkpoints'/f'{qid}.json',dict(passed=True,question_id=qid,all_new_conditions=checks,history=history,
            HF_Native_Noop_exact=True,cache_independence=True,input_weights_unchanged=True,hook_cleanup_normal_exception=True))
        rows.append(dict(question_id=qid,passed=True,new_conditions=6,reused_baselines=4));print('STRICT SMOKE PASS',qid,flush=True)
    return dict(passed=True,binding=binding,rows=rows,strict_torch_equal=True,smoke_generations=66)

def record(model,processor,s,inputs,c,p,binding,reference):
    path=OUT/f"checkpoints/{s['task']}__{s['question_id']}/{c}.json";assert not path.exists()
    before=tensor_hashes(inputs);footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    g,first,_,_=generate(model,processor.tokenizer,inputs,c)
    g.update(correct=int(g['prediction']==s['answer']),first_logits_sha256=digest(first.tolist()),**first_metrics(reference,first))
    raw=save_raw(RAW/f"main/{s['task']}__{s['question_id']}/{c}.npz",first)
    assert tensor_hashes(inputs)==before and hook_footprint(model)==footprint and versions=={n:p._version for n,p in model.named_parameters()}
    write(path,dict(binding=binding,code_commit=p['code_commit'],sample_sha256=digest(s),question_id=s['question_id'],task=s['task'],gold=s['answer'],image_sha256=s['image_sha256'],
        input_hashes=before,condition=c,generation=g,result_sha256=digest(g),raw=raw))

def gpu():
    start=time.perf_counter();p=verify();binding=digest(p)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight(p['gpu']);assert transformers.__version__=='4.49.0'
    model,processor,env=load_model(MODEL);assert torch.cuda.device_count()==1
    env.update(pid=os.getpid(),gpu=p['gpu'],python_executable=os.sys.executable);write(OUT/'environment.json',env)
    manifest=read(OUT/'sample_manifest.json');mapping=read(OUT/'input_mappings.json')
    write(OUT/'gpu_smoke_validation.json',smoke(model,processor,manifest,mapping,binding));verify()
    print('GATE PASS: all smoke and historical compatibility checks; 1440 new answers start',flush=True)
    for i,s in enumerate(manifest['samples']):
        inputs=prepared(model,processor,s);assert tensor_hashes(inputs)==mapping[i]['input_hashes']
        ref=read(checkpoint(s,'All-Writer'))
        with np.load(ref['raw']['path']) as z:reference=torch.from_numpy(z['first_logits'].copy())
        for c in NEW:record(model,processor,s,inputs,c,p,binding,reference)
        if (i+1)%10==0:print('MAIN PROGRESS',6*(i+1),'/1440','seconds',round(time.perf_counter()-start,1),flush=True)
    verify();assert len(list((OUT/'checkpoints').glob('*/*.json')))==1440
    write(OUT/'gpu_completion.json',dict(passed=True,binding=binding,pid=os.getpid(),new_main_generations=1440,reused_main_records=960,native_full_reference_records=240,
        model_replicas=1,seconds=time.perf_counter()-start,memory=gpu_memory()))
    print('COLLECTION COMPLETE; model process exits',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','gpu']);args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        write(OUT/'failures'/f'{args.stage}_{time.time_ns()}.json',dict(status='PARTIAL',stage=args.stage,traceback=traceback.format_exc(),hardware=hardware()))
        write(OUT/'completion.json',dict(status='PARTIAL',complete=False,failed_stage=args.stage));raise
