"""Freeze sources and controls, strict smoke, then 480 main + 240 swap generations."""
import argparse,fcntl,io,os,shutil,subprocess,time,traceback,unittest
from PIL import Image
import numpy as np
import torch
import transformers
from transformers import AutoProcessor
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from query_role_attention.engine import generate as old_generate
from query_role_attention.run import save_raw
import query_role_smoke_fix.run as repaired
from vlm_common import load_model,gpu_memory
from .common import *
from .engine import generate,SourceMask,text_generate

def cpu():
    stream=io.StringIO();suite=unittest.TestSuite()
    for name in ['query_role_attention.tests','query_role_smoke_fix.tests','query_role_formal.tests','relay_source_causality.tests']:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('vlm_research.'+name))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(OUT/'cpu_validation.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,output=stream.getvalue(),sources=sources(),cuda_initialized=torch.cuda.is_initialized()))
    print(stream.getvalue(),flush=True);assert result.wasSuccessful() and not torch.cuda.is_initialized()
    write(OUT/'historical_baseline_validation.json',audit_history())
    print('480 baseline records, actual raw arrays and historical artifacts verified',flush=True)

def frozen_inputs(processor,s):
    chat=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
    with Image.open(s['image_path']) as im:inputs=processor(images=im.convert('RGB'),text=chat,return_tensors='pt').to(dtype=torch.float16)
    assert inputs['input_ids'][0].tolist()==s['input_mapping']['prompt_ids']
    return dict(input_hashes=tensor_hashes(inputs),sample_sha256=digest(s),image_sha256=sha256(s['image_path']))

def freeze():
    a=read(OUT/'cpu_validation.json');assert a['passed'] and a['sources']==sources()
    assert read(OUT/'historical_baseline_validation.json')['passed']
    paths=['vlm_research/'+n for n in sources()]
    assert not subprocess.check_output(['git','diff','HEAD','--',*paths],cwd=ROOT.parent,text=True).strip()
    assert set(subprocess.check_output(['git','ls-files','--',*paths],cwd=ROOT.parent,text=True).splitlines())==set(paths)
    old=read(OLD/'protocol.json')
    for path,info in old['model_files'].items():assert sha256(path)==info['sha256'],path
    for name in ['sample_manifest.json','input_mappings.json']:
        assert not (OUT/name).exists();shutil.copyfile(OLD/name,OUT/name)
    shutil.copyfile(SWAP,OUT/'image_swap_manifest.json');pairs=read(SWAP)['pairs'];assert len(pairs)==40
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    processor.patch_size=14;processor.num_additional_image_tokens=1;processor.vision_feature_select_strategy='default';processor.tokenizer.padding_side='left'
    swap_inputs={}
    for pair in pairs:
        left,right=pair['samples'];assert left['question']==right['question'] and left['answer']!=right['answer'] and left['image_sha256']!=right['image_sha256']
        assert left['input_mapping']['prompt_ids']==right['input_mapping']['prompt_ids']
        for side,s in enumerate(pair['samples']):
            v=frozen_inputs(processor,s);assert v['image_sha256']==s['image_sha256'];swap_inputs[f"{pair['pair_id']}__{side}"]=v
    write(OUT/'image_swap_inputs.json',swap_inputs)
    samples=read(OUT/'sample_manifest.json')['samples'];diagnostics=[]
    for task in TASKS:
        for s in [x for x in samples if x['task']==task][:2]:
            chat=processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':s['prompt_text']}]}],tokenize=False,add_generation_prompt=True)
            inputs=processor.tokenizer(chat,return_tensors='pt');assert not (inputs['input_ids']==32000).any()
            diagnostics.append(dict(question_id=s['question_id'],task=task,text=chat,prompt_ids=inputs['input_ids'][0].tolist(),input_hashes=tensor_hashes(inputs),gold=None,VQA_accuracy_valid=False))
    write(OUT/'text_diagnostic_manifest.json',dict(samples=diagnostics,questions=12,selection='First two frozen GQA questions per task, without model-output filtering',
        limitation='No existing audited text QA dataset located; Question-only output-sensitivity diagnostic, no gold or VQA accuracy',scope='Every original non-BOS non-final text query; P-1 and decode native'))
    gpu=select_gpu()
    names=['sample_manifest.json','input_mappings.json','cpu_validation.json','historical_baseline_validation.json','image_swap_manifest.json','image_swap_inputs.json','text_diagnostic_manifest.json','gpu_selection.json']
    p=dict(base_commit=BASE,historical_formal_commit='8d12f8c05f2b246d029ec4d9ffe45d1f857fa947',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        sources=sources(),history_artifacts=history_hashes(),bindings={n:sha256(OUT/n) for n in names},model_files=old['model_files'],model_revision=REVISION,model_path=str(MODEL),
        transformers_sources=old['transformers_sources'],scorer_sources=old['scorer_sources'],generation=old['generation'],dtype='float16',attention='eager',transformers='4.49.0',
        dependencies=subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines(),conditions=CONDITIONS,layers=LAYERS,new_main_generations=480,reused_main_generations=480,
        source_rules=dict(Image='Original 576 image keys, only causal visible R edges',Text='k<q, non-image, excluding unique BOS; Self k=q preserved',BOS='Unique actual native BOS key preserved',Self='Current query key preserved'),
        intervention='Clone native additive causal mask; set only target R->source legal edges to -inf before softmax at L2-L5. Renormalize. All other queries including P-1/decode and other layers native.',
        actual_probability_audit='Target-layer output_attentions=True via prehook; inspect real post-softmax weights, forbidden max==0, row sums within historical ROW_ATOL=0.002; release matrices immediately',
        swap=dict(pairs=40,new_generations=240,conditions=['Full']+NEW,full_reused=False,reason='Historical handoff generation lacks explicit logits_to_keep=1. Recompute pair Full in current matching path; no failed controls pooled.'),
        text_diagnostic=dict(questions=12,new_generations=24,conditions=['Native','Text-Prefill-Bypass'],VQA_accuracy_valid=False),
        comparisons=['R-All - Full','R-Image-OFF - Full','R-Text-OFF - Full','R-Image-OFF - R-Text-OFF'],bootstrap=dict(seed=SEED,repetitions=20000,cluster='Image SHA256',same_draws=True,interval='Exploratory two-sided percentile 95%'),
        manifest_note=old['manifest_note'],gpu=gpu,exploratory=True,no_equivalence_or_NI=True,no_information_fraction_claim=True,
        historical_equivalence='Handoff: all 32 layers and answer-path factors; source-knockout: P-1 queries; role experiments: whole o_proj output. No equivalent R-source/L2-L5 condition found.',
        background_reports={str(f.relative_to(ROOT)):sha256(f) for f in [ROOT/'results/query_role_ia_joint_240/report_zh.md',ROOT/'results/rq2_visual_text_handoff_240/report_zh.md']})
    write(OUT/'protocol.json',p);verify();print('Frozen model, source, 240 main questions, 40 swap pairs and 12 diagnostic prompts',flush=True)

def custom(model,processor,prompt,c,**kw):
    with repaired.Recorder(model) as rec:result=generate(model,processor.tokenizer,prompt,c,**kw)
    assert all(call['logits_to_keep']==1 for call in rec.calls)
    return result,rec

@torch.inference_mode()
def smoke(model,processor,manifest,mapping,binding):
    repaired.OUT=OUT;rows=[]
    for i in manifest['smoke_indices']:
        s=manifest['samples'][i];qid=s['question_id'];inputs=prepared(model,processor,s);before=tensor_hashes(inputs)
        footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
        assert before==mapping[i]['input_hashes']
        ids,hf=repaired.native(model,inputs);full,frec=custom(model,processor,inputs,'Full',retain_cache=True)
        repaired.compare('native_full',qid,hf,frec,binding);assert ids==full[0]['token_ids']
        parent=full[3];parent_hash=cache_fingerprint(parent)
        for c in BASELINES:
            with repaired.Recorder(model) as oldrec:old=old_generate(model,processor.tokenizer,inputs,c,logits_to_keep=1)
            if c=='Full':repaired.compare('historical_full',qid,oldrec,frec,binding)
            saved=read(checkpoint(s,c))
            with np.load(saved['raw']['path']) as data:raw=torch.from_numpy(data['first_logits'].copy())
            exact=torch.equal(old[1],raw) and old[0]['token_ids']==saved['generation']['token_ids']
            write(OUT/'comparisons'/f'{qid}__saved_{c}.json',dict(exact=exact,raw_path=saved['raw']['path'],raw_sha256=sha256(saved['raw']['path']),first_shape=list(raw.shape)))
            assert exact,'Historical saved baseline mismatch; stop'
        checks={}
        for c in NEW:
            noop,nrec=custom(model,processor,inputs,c,noop=True)
            repaired.compare(c+'_noop',qid,frec,nrec,binding);assert noop[0]['token_ids']==ids
            counts=[0]*32;handles=[]
            try:
                for k,layer in enumerate(model_layers(model)):
                    def mlp(module,args,out,k=k):assert torch.isfinite(out).all();counts[k]+=1
                    handles.append(layer.mlp.register_forward_hook(mlp))
                changed,crec=custom(model,processor,inputs,c,retain_cache=True)
            finally:
                for h in handles:h.remove()
            g=changed[0];owned=changed[3];assert counts==[g['length']]*32
            assert cache_fingerprint(parent)==parent_hash
            assert all(x.data_ptr()!=y.data_ptr() for a,b in zip(parent.to_legacy_cache(),owned.to_legacy_cache()) for x,y in zip(a,b))
            assert all(e['blocked_probability_max']==0 and e['applied'] for e in g['source_events'][:4])
            assert all(not e['applied'] for e in g['source_events'][4:])
            assert hook_footprint(model)==footprint and tensor_hashes(inputs)==before
            checks[c]=dict(generation=g,all_32_mlp_calls=counts,independent_cache=True,parent_cache_unchanged=True,actual_forbidden_probabilities_zero=True)
            del owned,changed,crec
        try:
            with SourceMask(model,inputs,NEW[0]):raise RuntimeError('intentional cleanup')
        except RuntimeError as e:assert str(e)=='intentional cleanup'
        assert hook_footprint(model)==footprint and versions=={n:p._version for n,p in model.named_parameters()}
        row=dict(question_id=qid,passed=True,full_noop_native_logits_exact=True,history_Full_R_saved_exact=True,source_checks=checks,
            inputs_weights_unchanged=True,normal_exception_cleanup=True,no_nan_inf_oom=True)
        write(OUT/'smoke_checkpoints'/f'{qid}.json',row);rows.append(row)
        print('STRICT GPU SMOKE PASS',qid,flush=True)
    return dict(passed=True,questions=3,rows=rows,binding=binding,strict_torch_equal=True)

def record(model,processor,s,prompt,c,p,binding,path,rawpath,full_logits=None,input_hashes=None):
    if path.exists():
        row=read(path);assert row['binding']==binding and row['sample_sha256']==digest(s)
        assert row['result_sha256']==digest(row['generation']) and sha256(row['raw']['path'])==row['raw']['sha256'];return row
    before=tensor_hashes(prompt);assert input_hashes is None or before==input_hashes
    footprint=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
    g,first,_,_=generate(model,processor.tokenizer,prompt,c)
    g.update(correct=int(g['prediction']==s['answer']),first_logits_sha256=digest(first.tolist()))
    if full_logits is not None:g.update(first_metrics(full_logits,first),full_first_logits_sha256=digest(full_logits.tolist()))
    raw=save_raw(rawpath,first)
    assert tensor_hashes(prompt)==before and hook_footprint(model)==footprint and versions=={n:p._version for n,p in model.named_parameters()}
    row=dict(binding=binding,code_commit=p['code_commit'],sample_sha256=digest(s),question_id=s['question_id'],task=s['task'],gold=s['answer'],image_sha256=s['image_sha256'],
        input_hashes=before,condition=c,generation=g,result_sha256=digest(g),raw=raw)
    write(path,row);return row

def gpu():
    started=time.perf_counter();p=verify();binding=digest(p)
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight(p['gpu']);assert transformers.__version__=='4.49.0'
    model,processor,env=load_model(MODEL);assert torch.cuda.device_count()==1
    env.update(pid=os.getpid(),gpu=p['gpu'],python_executable=os.sys.executable);write(OUT/'environment.json',env)
    manifest=read(OUT/'sample_manifest.json');mapping=read(OUT/'input_mappings.json')
    write(OUT/'gpu_smoke_validation.json',smoke(model,processor,manifest,mapping,binding));verify()
    print('GATE PASS: 480 main generations, swaps and diagnostics kept separate',flush=True)
    for i,s in enumerate(manifest['samples']):
        inputs=prepared(model,processor,s);assert tensor_hashes(inputs)==mapping[i]['input_hashes']
        full=read(checkpoint(s,'Full'))
        with np.load(full['raw']['path']) as data:fl=torch.from_numpy(data['first_logits'].copy())
        for c in NEW:
            stem=f"{s['task']}__{s['question_id']}/{c}"
            record(model,processor,s,inputs,c,p,binding,OUT/f'checkpoints/{stem}.json',RAW/f'main/{stem}.npz',fl,mapping[i]['input_hashes'])
        if (i+1)%20==0:print('MAIN PROGRESS',2*(i+1),'/480',flush=True)
    swapmap=read(OUT/'image_swap_inputs.json')
    for i,pair in enumerate(read(OUT/'image_swap_manifest.json')['pairs']):
        for side,s in enumerate(pair['samples']):
            inputs=prepared(model,processor,s);key=f"{pair['pair_id']}__{side}";assert tensor_hashes(inputs)==swapmap[key]['input_hashes']
            fl=None
            for c in ['Full']+NEW:
                stem=f'{key}/{c}';r=record(model,processor,s,inputs,c,p,binding,OUT/f'swap_checkpoints/{stem}.json',RAW/f'swap/{stem}.npz',fl,swapmap[key]['input_hashes'])
                if c=='Full':
                    with np.load(r['raw']['path']) as data:fl=torch.from_numpy(data['first_logits'].copy())
        if (i+1)%10==0:print('SWAP PROGRESS',i+1,'/40 pairs',flush=True)
    for s in read(OUT/'text_diagnostic_manifest.json')['samples']:
        inputs=processor.tokenizer(s['text'],return_tensors='pt').to('cuda:0');assert tensor_hashes(inputs)==s['input_hashes']
        before=hook_footprint(model);versions={n:p._version for n,p in model.named_parameters()}
        for bypass in [False,True]:
            label='Bypass' if bypass else 'Native';g,first=text_generate(model,processor.tokenizer,inputs,bypass)
            raw=save_raw(RAW/f"text/{s['question_id']}/{label}.npz",first)
            write(OUT/f"text_checkpoints/{s['question_id']}/{label}.json",dict(binding=binding,sample_sha256=digest(s),input_hashes=tensor_hashes(inputs),generation=g,result_sha256=digest(g),raw=raw))
        assert before==hook_footprint(model) and versions=={n:p._version for n,p in model.named_parameters()} and tensor_hashes(inputs)==s['input_hashes']
    verify();assert len(list((OUT/'checkpoints').glob('*/*.json')))==480 and len(list((OUT/'swap_checkpoints').glob('*/*.json')))==240
    write(OUT/'gpu_completion.json',dict(passed=True,binding=binding,pid=os.getpid(),new_main_generations=480,swap_generations=240,text_diagnostic_generations=24,model_replicas=1,seconds=time.perf_counter()-started,memory=gpu_memory()))
    print('COMPLETE collection: main480 / swap240 / diagnostic24; model process exits',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cpu','freeze','gpu']);args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException:
        write(OUT/'failures'/f'{args.stage}_{time.time_ns()}.json',dict(status='PARTIAL',stage=args.stage,traceback=traceback.format_exc(),hardware=hardware()))
        write(OUT/'completion.json',dict(status='PARTIAL',complete=False,failed_stage=args.stage));raise
