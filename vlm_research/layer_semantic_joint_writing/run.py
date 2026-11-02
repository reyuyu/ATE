"""CPU audit and immutable freeze, strict GPU gate, then exactly 720 new answers."""
import argparse,fcntl,io,os,shutil,subprocess,time,traceback,unittest
import numpy as np
import torch,transformers
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from query_role_attention.run import save_raw
from vlm_common import load_model,gpu_memory
from semantic_visual_writers.engine import generate as semantic_generate
from sparse_visual_writing_frontier.engine import generate as layer_generate
from visual_write_window.engine import generate as all_generate
from first_answer_attention.engine import first_metrics
import query_role_smoke_fix.run as repaired
from .common import *
from .engine import generate,JointMask

def cpu():
    stream=io.StringIO();suite=unittest.TestSuite()
    for name in ['query_role_attention.tests','query_role_smoke_fix.tests','query_role_formal.tests','relay_source_causality.tests','visual_write_window.tests','semantic_visual_writers.tests','sparse_visual_writing_frontier.tests','layer_semantic_joint_writing.tests']:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('vlm_research.'+name))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,output=stream.getvalue(),sources=sources(),cuda_initialized=torch.cuda.is_initialized()))
    print(stream.getvalue(),flush=True);assert result.wasSuccessful() and not torch.cuda.is_initialized()
    write(OUT/'historical_reuse_audit.json',audit_history());print('AUDIT PASS: 1200 reusable records and immutable Semantic-k sets',flush=True)

def freeze():
    a=read(OUT/'cpu_validation.json');assert a['passed'] and a['sources']==sources()
    audit=read(OUT/'historical_reuse_audit.json');assert audit['passed']
    paths=['vlm_research/'+n for n in sources()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    old=read(LAYER/'protocol.json')
    for name,info in old['model_files'].items():assert sha256(name)==info['sha256'],name
    for name,src in [('sample_manifest.json',ALL),('input_mappings.json',ALL),('token_group_manifest.json',SEM)]:
        assert not (OUT/name).exists();shutil.copyfile(src/name,OUT/name)
    assert sha256(OUT/'sample_manifest.json')==MANIFEST_SHA
    assert sha256(OUT/'token_group_manifest.json')==audit['semantic_selection_sha256']
    for name,e in audit['records'].items():
        dst=OUT/'reused_checkpoints'/name;assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(e['path'],dst);assert sha256(dst)==e['sha256']
    gpu=select_gpu();names=['cpu_validation.json','historical_reuse_audit.json','sample_manifest.json','input_mappings.json','token_group_manifest.json','gpu_selection.json']
    p=dict(base_commit=BASE,semantic_reference_commit=SEM_COMMIT,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),sources=sources(),
        bindings={n:sha256(OUT/n) for n in names},model_files=old['model_files'],model_revision=REVISION,model_path=str(MODEL),dtype='float16',attention='eager',transformers='4.49.0',
        transformers_sources=old['transformers_sources'],scorer_sources=old['scorer_sources'],generation=old['generation'],
        dependencies=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines(),conditions=KEEP,new_conditions=NEW,reused_conditions=BASELINES,
        main=dict(questions=240,new_generations=720,reused_generations=1200,independent_cache=True,conditions_serial=True),
        answer_rule='P-1 and all real decode queries including EOS prediction blocked from original 576 image keys at every layer.',
        relay_rule='Only frozen Semantic-k R rows can read image keys, only in Joint kept layers. All unselected R rows blocked at all32 layers. Image queries, other edges and computations native.',
        selection_rule='Byte-identical historical token_group_manifest.json copied; never call token selection or modify per-question sets.',
        hard_mask='Direct call to frozen semantic_visual_writers.engine.apply_mask with historical Semantic-k writers on kept layers and an empty set elsewhere; no o_proj ablation.',
        comparisons=[dict(a=c,b=b,role='joint_vs_all' if b=='All-Writer' else 'joint_vs_semantic' if b=='Semantic-k' else 'joint_vs_layer') for c in NEW for b in ['All-Writer',MATCH[c],'Semantic-k']],
        interactions=[dict(joint=c,layer=MATCH[c],semantic='Semantic-k',all='All-Writer',formula='Joint - Layer - Semantic + All',negative='Extra accuracy damage beyond additive single-condition effects') for c in NEW],
        unique_harm='All-Writer, matching Layer-only and Semantic-only correct; Joint wrong. Also report both singles correct but Joint wrong regardless of All.',
        statistics=dict(seed=SEED,repetitions=20000,cluster='Image SHA256',same_draws=True,interval='Exploratory two-sided percentile 95%',multiplicity='Unadjusted exploratory, not confirmatory'),
        quality_bounds=dict(minus_one_pp=-1.,minus_two_pp=-2.,rule='Observed Joint minus All accuracy at least bound; point screen only, not non-inferiority.'),
        budget=dict(per_question='kept_layers * selected_R_queries * 576 / (32 * all_R_queries * 576)',
            pooled='sum allowed R-image edges over samples / sum All-Writer R-image edges; primary actual connection-count ratio',
            macro='Unweighted mean of per-question ratios, separately reported',scope='R->original Image only, no heads factor; all query-key matmuls and full KV retained'),
        smoke=dict(question_ids=SMOKE_IDS,all_three_new=True,all_five_reused=True,strict='HF/Native/three noops all valid logits torch.equal; old/new/saved baselines exact. Every actual prohibited softmax probability exactly0; row bound .002 unchanged.'),
        gpu=gpu,exploratory_historical_cohort=True,no_actual_acceleration=True,
        stop='No new token policy, physical image exit, KV pruning, kernel, router, extra model or dataset.')
    write(OUT/'protocol.json',p);verify();print('FROZEN',p['code_commit'],'720 new / 1200 reused',flush=True)

def custom(model,processor,inputs,c,spec,**kw):
    with repaired.Recorder(model) as rec:result=generate(model,processor.tokenizer,inputs,c,spec,**kw)
    assert all(v['logits_to_keep']==1 for v in rec.calls);return result,rec

@torch.inference_mode()
def smoke(model,processor,samples,mapping,specs,binding):
    repaired.OUT=OUT;rows=[]
    for qid in SMOKE_IDS:
        i=next(i for i,s in enumerate(samples) if s['question_id']==qid);s=samples[i];spec=specs['main'][key(s)]
        inputs=prepared(model,processor,s);before=tensor_hashes(inputs);assert before==mapping[i]['input_hashes']
        footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
        ids,hf=repaired.native(model,inputs);native,nrec=custom(model,processor,inputs,'Native',spec,retain_cache=True)
        repaired.compare('HF_Native',qid,hf,nrec,binding);assert native[0]['token_ids']==ids
        parent=native[3];parent_hash=cache_fingerprint(parent)
        for c in NEW:
            noop,rec=custom(model,processor,inputs,c,spec,noop=True)
            repaired.compare(c+'_noop',qid,nrec,rec,binding);assert noop[0]['token_ids']==ids
        history={}
        for c in BASELINES:
            with repaired.Recorder(model) as oldrec:
                if c=='Semantic-k':old=semantic_generate(model,processor.tokenizer,inputs,spec['selection']['writers'][c])
                elif c=='All-Writer':old=all_generate(model,processor.tokenizer,inputs,c)
                else:old=layer_generate(model,processor.tokenizer,inputs,c)
            changed,rec=custom(model,processor,inputs,c,spec);repaired.compare(c+'_old_new',qid,oldrec,rec,binding)
            saved=read(reused_path(s,c));v=raw(saved)
            assert torch.equal(changed[1],v) and changed[0]['token_ids']==old[0]['token_ids']==saved['generation']['token_ids']
            history[c]=dict(saved_first_and_tokens_exact=True,old_new_all_effective_logits_exact=True)
        checks={}
        for c in NEW:
            counts=[0]*32;handles=[]
            try:
                for k,layer in enumerate(model_layers(model)):
                    def mlp(module,args,out,k=k):assert torch.isfinite(out).all();counts[k]+=1
                    handles.append(layer.mlp.register_forward_hook(mlp))
                changed,_=custom(model,processor,inputs,c,spec,retain_cache=True)
            finally:
                for h in handles:h.remove()
            g=changed[0];owned=changed[3];assert counts==[g['length']]*32
            assert cache_fingerprint(parent)==parent_hash
            assert all(x.data_ptr()!=y.data_ptr() for a,b in zip(parent.to_legacy_cache(),owned.to_legacy_cache()) for x,y in zip(a,b))
            assert all(e['blocked_probability_max']==0 and e['answer_image_masked'] for e in g['source_events'])
            assert all(e['permitted_writer_queries']==(spec['selection']['k'] if e['layer'] in KEEP[c] else 0) for e in g['source_events'])
            assert hook_footprint(model)==footprint and tensor_hashes(inputs)==before
            checks[c]=dict(generation=g,all_32_mlp_calls=counts,cache_independent=True,parent_cache_unchanged=True);del owned,changed
        try:
            with JointMask(model,inputs,NEW[0],spec):raise RuntimeError('intentional cleanup')
        except RuntimeError as e:assert str(e)=='intentional cleanup'
        assert hook_footprint(model)==footprint and versions=={n:p._version for n,p in model.named_parameters()}
        write(OUT/'smoke_checkpoints'/f'{qid}.json',dict(passed=True,question_id=qid,all_new_conditions=checks,history=history,
            HF_Native_Noop_exact=True,cache_independence=True,input_weights_unchanged=True,hook_cleanup_normal_exception=True))
        rows.append(dict(question_id=qid,passed=True,new_conditions=3,reused_baselines=5));print('STRICT SMOKE PASS',qid,flush=True)
    return dict(passed=True,binding=binding,rows=rows,strict_torch_equal=True,smoke_generations=54)

def record(model,processor,s,inputs,c,spec,p,binding,reference):
    path=OUT/'checkpoints'/key(s)/(c+'.json');assert not path.exists()
    before=tensor_hashes(inputs);footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    g,first,_,_=generate(model,processor.tokenizer,inputs,c,spec)
    g.update(correct=int(g['prediction']==s['answer']),first_logits_sha256=digest(first.tolist()),**first_metrics(reference,first))
    rr=save_raw(RAW/'main'/key(s)/(c+'.npz'),first)
    assert tensor_hashes(inputs)==before and hook_footprint(model)==footprint and versions=={n:p._version for n,p in model.named_parameters()}
    write(path,dict(binding=binding,code_commit=p['code_commit'],sample_sha256=digest(s),question_id=s['question_id'],task=s['task'],gold=s['answer'],image_sha256=s['image_sha256'],
        input_hashes=before,condition=c,selection_binding=spec['selection_binding'],generation=g,result_sha256=digest(g),raw=rr))

def gpu():
    start=time.perf_counter();p=verify();binding=digest(p)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight(p['gpu']);assert transformers.__version__=='4.49.0'
    model,processor,env=load_model(MODEL);assert torch.cuda.device_count()==1
    env.update(pid=os.getpid(),gpu=p['gpu'],python_executable=os.sys.executable);write(OUT/'environment.json',env)
    samples=read(OUT/'sample_manifest.json')['samples'];mapping=read(OUT/'input_mappings.json');specs=read(OUT/'token_group_manifest.json')
    write(OUT/'gpu_smoke_validation.json',smoke(model,processor,samples,mapping,specs,binding));verify()
    print('GATE PASS: all strict checks; 720 new answers start',flush=True)
    for i,s in enumerate(samples):
        inputs=prepared(model,processor,s);assert tensor_hashes(inputs)==mapping[i]['input_hashes'];reference=raw(read(reused_path(s,'All-Writer')))
        for c in NEW:record(model,processor,s,inputs,c,specs['main'][key(s)],p,binding,reference)
        if (i+1)%10==0:print('MAIN PROGRESS',3*(i+1),'/720','seconds',round(time.perf_counter()-start,1),flush=True)
    verify();assert len(list((OUT/'checkpoints').glob('*/*.json')))==720
    write(OUT/'gpu_completion.json',dict(passed=True,binding=binding,pid=os.getpid(),new_main_generations=720,reused_main_records=1200,
        model_replicas=1,seconds=time.perf_counter()-start,memory=gpu_memory()))
    print('COLLECTION COMPLETE; model process exits',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','gpu']);args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        write(OUT/'failures'/f'{args.stage}_{time.time_ns()}.json',dict(status='PARTIAL',stage=args.stage,traceback=traceback.format_exc(),hardware=hardware()))
        write(OUT/'completion.json',dict(status='PARTIAL',complete=False,failed_stage=args.stage));raise
