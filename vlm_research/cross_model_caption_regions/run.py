import argparse
import fcntl
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from types import SimpleNamespace

import torch

from .common import *
from .engine import prepare, generate, qwen_inputs, intern_layout
from ..internvl3_smoke.engine import hooks_count, tensor_hash, official_preprocessor


def layouts(kind):
    from transformers import AutoProcessor, AutoTokenizer
    samples = read(MANIFEST)['samples']
    if kind == 'qwen':
        processor = AutoProcessor.from_pretrained(QMODEL,local_files_only=True)
        rows = {s['question_id']:qwen_inputs(processor,s)[1] for s in samples}
    else:
        tokenizer=AutoTokenizer.from_pretrained(IMODEL,use_fast=False,local_files_only=True)
        fast=AutoTokenizer.from_pretrained(IMODEL,use_fast=True,local_files_only=True)
        spec=importlib.util.spec_from_file_location('frozen_intern_conversation',IMODEL/'conversation.py')
        module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
        conv=module.get_conv_template(read(IMODEL/'config.json')['template'])
        dummy=SimpleNamespace(conv_template=conv,system_message=conv.system_message)
        rows={s['question_id']:intern_layout(dummy,tokenizer,fast,s)[1] for s in samples}
    write(OUT/f'{kind}_input_layouts.json',rows)
    print('CPU input layouts frozen',kind,len(rows),flush=True)


def freeze():
    assert not (OUT/'protocol.json').exists()
    inventory=preflight()
    manifest=read(MANIFEST)
    samples=manifest['samples']
    assert len(samples)==len({s['image_sha256'] for s in samples})==60
    for s in samples:
        assert s['question']==s['prompt_text']==PROMPT and len(s['answer'])==5
        assert sha(s['image_path'])==s['image_sha256']
    write(OUT/'sample_manifest.json',manifest)
    manifests={'qwen':ROOT/'vlm_research/results/qwen3vl_image_source_240/model_manifest.json',
               'internvl':ROOT/'vlm_research/results/visual_pathways_seven_hour/internvl_model_manifest.json'}
    model_files={}
    for kind,path in manifests.items():
        records=[]
        for e in read(path)['files']:
            f=MODELS[kind]/e['path']; assert sha(f)==e['sha256'],str(f)
            st=f.stat(); records.append(dict(path=e['path'],sha256=e['sha256'],bytes=st.st_size,mtime_ns=st.st_mtime_ns))
        model_files[kind]=records
    from . import tests
    import unittest,io
    output=io.StringIO(); suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    (OUT/'cpu_tests.txt').write_text(output.getvalue())
    assert result.wasSuccessful()
    write(OUT/'cpu_validation.json',dict(passed=True,tests=result.testsRun))
    scorefiles=list((Path('/data/lk/ATE-resources/paper_rq1/metric_deps')/'pycocoevalcap').rglob('*.py'))
    scorefiles+=list((Path('/data/lk/ATE-resources/paper_rq1/metric_deps')/'pycocoevalcap').rglob('*.jar'))
    historyfiles=[MANIFEST,LLAVA/'protocol.json',LLAVA/'sample_manifest.json',LLAVA/'per_image.csv',
                  LLAVA/'condition_summary.csv',HISTORY/'paper_rq1/scorers.py',HISTORY/'paper_rq1/inventory.py']
    historyfiles+=list((LLAVA/'scores').glob('*.json'))
    bindings={str(f):sha(f) for f in scorefiles+historyfiles+list(manifests.values())}
    bindings.update({str(OUT/f):sha(OUT/f) for f in ['sample_manifest.json','qwen_input_layouts.json',
                    'internvl_input_layouts.json','cpu_validation.json','cpu_tests.txt']})
    write(OUT/'protocol.json',dict(base_commit=command('git','rev-parse','HEAD'),source_sha256=sources(),
        conditions={m:conditions(m) for m in MODELS},bindings=bindings,model_files=model_files,
        model_revisions={m:p.name for m,p in MODELS.items()},physical_gpu=1,uuid=UUID,preflight=inventory,
        prompt=PROMPT,dtype='bfloat16',attention='eager',generation=dict(greedy=True,max_new_tokens=96,cache='independent native DynamicCache',prefill='Entirely native, including P-1'),
        intervention='Reused GenerationMask; noop throughout Prefill, original softmax-before -inf Image mask only for actual incremental Decode queries',
        internvl=dict(max_num=1,logits_to_keep=1),qwen_deepstack='Native unchanged',
        bootstrap=dict(repetitions=20000,seed=20261102,unit='Image SHA256 paired whole cluster',
                       metric='Official per-image scores, CIDEr fixed 60-image corpus IDF'),
        comparisons='Each condition-Full; Middle-Early; Middle-Late; Middle-each equal-count Random; Middle-mean equal-count Random',
        count_caveat='Qwen Early=12 total, Middle/Late=13; only Random12 sets match Middle count. Random seeds are not image replicates.',
        random='Python random.Random(seed).sample(range(1,n),k); sorted; prepend L0. Fixed across images/steps.',
        qualitative='Posthoc descriptive: 3 strongest Middle-vs-mean(Early,Late) SPICE gains and 3 strongest counterexamples across models, 6 unique image SHA256; fixed SHA tie-break. Never filters metrics.',
        scope='Exploratory same 60 historical images; no independent confirmation, equivalence, necessity, or speed claims',
        smoke_ids=[samples[0]['question_id'],max(samples,key=lambda s:s['raw_row']['height']*s['raw_row']['width'])['question_id']]))
    print('FROZEN: 60 images x 8 conditions x 2 models; 960 new generations',flush=True)


@torch.inference_mode()
def smoke(kind,model,processor,sample,fast,loader):
    inputs,layout,pixels=prepare(kind,model,processor,sample,fast,loader)
    assert layout==read(OUT/f'{kind}_input_layouts.json')[sample['question_id']]
    tokenizer=processor.tokenizer if kind=='qwen' else processor
    nlayer=36 if kind=='qwen' else 28
    base,first,_=generate(kind,model,tokenizer,inputs,layout,list(range(nlayer)),native=True)
    full,ff,ref=generate(kind,model,tokenizer,inputs,layout,list(range(nlayer)),capture=True)
    noop,nf,_=generate(kind,model,tokenizer,inputs,layout,[],noop=True,capture=True)
    assert torch.equal(first,ff) and torch.equal(ff,nf)
    assert base['token_ids']==full['token_ids']==noop['token_ids']
    assert base['logits_sha256']==full['logits_sha256']==noop['logits_sha256']
    assert len(base['token_ids'])>1
    if kind=='internvl':
        from ..internvl3_smoke.run import ChatCapture
        with ChatCapture(model) as official:
            text=model.chat(processor,pixels,layout['request'],dict(do_sample=False,num_beams=1,max_new_tokens=96))
        assert official.input_ids[0].tolist()==layout['input_ids']
        assert official.generated_ids==base['token_ids'] and torch.equal(official.first_logits,first)
        assert torch.equal(official.saved['inputs_embeds'],inputs['inputs_embeds'].cpu())
    checks={}
    for key,keep in conditions(kind).items():
        value,f,trace=generate(kind,model,tokenizer,inputs,layout,keep,capture=True,limit=3)
        assert torch.equal(f,first) and value['token_ids'][0]==base['token_ids'][0]
        assert trace['full_prefill']==ref['full_prefill']
        assert trace['deepstack']==ref['deepstack']
        if kind=='qwen': assert len(trace['deepstack'])==3
        assert value['decode_steps']>=1
        checks[key]=dict(entire_prefill_hidden_exact=True,first_logits_exact=True,
                        deepstack_exact=True,rows=trace['rows'],audit=value['audit'])
    return dict(passed=True,question_id=sample['question_id'],input_length=layout['input_length'],
                image_tokens=layout['image_tokens'],official_native_generation_exact=True,
                noop_all_step_logits_exact=True,checks=checks)


@torch.inference_mode()
def worker(kind):
    protocol=verify(); inventory=preflight()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    folder=OUT/kind;folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'complete.json').exists():
        assert read(folder/'complete.json')['protocol_sha256']==sha(OUT/'protocol.json')
        print('Already complete',kind);return
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==1
    props=torch.cuda.get_device_properties(0)
    assert uuid.UUID(str(props.uuid).removeprefix('GPU-')) == uuid.UUID(UUID.removeprefix('GPU-'))
    started=time.perf_counter()
    if kind=='qwen':
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        processor=AutoProcessor.from_pretrained(QMODEL,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(QMODEL,local_files_only=True,dtype=torch.bfloat16,
                    attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True).eval()
        assert model.config.text_config._attn_implementation==model.config.vision_config._attn_implementation=='eager'
        fast=loader=None
    else:
        from ..visual_pathways.scan import load_intern
        model,processor,fast=load_intern();loader,_=official_preprocessor(IMODEL)
    assert {str(p.device) for p in model.parameters()}=={'cuda:0'}
    assert {p.dtype for p in model.parameters()}=={torch.bfloat16}
    versions=[(p._version,p.data_ptr()) for p in model.parameters()];footprint=hooks_count(model)
    env=dict(executable=sys.executable,versions={n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','Pillow']},
             physical_gpu=1,uuid=UUID,logical_device=0,pid=os.getpid(),inventory=inventory,code_commit=command('git','rev-parse','HEAD'))
    write(folder/'environment.json',env)
    samples=read(OUT/'sample_manifest.json')['samples']
    byid={s['question_id']:s for s in samples}
    validations=[smoke(kind,model,processor,byid[q],fast,loader) for q in dict.fromkeys(protocol['smoke_ids'])]
    write(folder/'smoke.json',dict(passed=True,images=validations,seconds=time.perf_counter()-started))
    print('SMOKE PASS',kind,flush=True)
    tokenizer=processor.tokenizer if kind=='qwen' else processor
    formal=time.perf_counter();count=0
    for index,sample in enumerate(samples,1):
        inputs,layout,_=prepare(kind,model,processor,sample,fast,loader)
        assert layout==read(OUT/f'{kind}_input_layouts.json')[sample['question_id']]
        first=None;baseline=None
        for key,keep in conditions(kind).items():
            path=folder/'checkpoints'/sample['question_id'].replace(':','_')/(key+'.json')
            if path.exists():
                r=read(path);assert r['protocol_sha256']==sha(OUT/'protocol.json') and r['sample_sha256']==digest(sample)
                assert r['generation_sha256']==digest(r['generation'])
                f=torch.load(path.with_suffix('.pt'),weights_only=True)
                assert sha(path.with_suffix('.pt'))==r['first_logits_file_sha256']
                value=r['generation']
            else:
                tick=time.perf_counter()
                value,f,_=generate(kind,model,tokenizer,inputs,layout,keep)
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix('.pt.tmp');torch.save(f,tmp);os.replace(tmp,path.with_suffix('.pt'))
                r=dict(model=kind,condition=key,question_id=sample['question_id'],image_sha256=sample['image_sha256'],
                    sample_sha256=digest(sample),input_binding=digest(layout),protocol_sha256=sha(OUT/'protocol.json'),
                    generation=value,generation_sha256=digest(value),first_logits_file_sha256=sha(path.with_suffix('.pt')),
                    seconds=time.perf_counter()-tick,physical_gpu=1,code_commit=env['code_commit'])
                write(path,r)
            if first is None: first,baseline=f,value
            assert torch.equal(first,f) and value['token_ids'][0]==baseline['token_ids'][0]
            assert versions==[(p._version,p.data_ptr()) for p in model.parameters()] and hooks_count(model)==footprint
            count+=1
        write(folder/'progress.json',dict(images=index,generations=count,total=480,seconds=time.perf_counter()-formal))
        print(f'{kind} {index}/60; {count}/480; {(time.perf_counter()-formal):.1f}s',flush=True)
    write(folder/'complete.json',dict(complete=True,generations=count,protocol_sha256=sha(OUT/'protocol.json'),
        seconds_including_load_smoke=time.perf_counter()-started,formal_seconds=time.perf_counter()-formal,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        weights_hooks_unchanged=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['layouts','freeze','worker']);parser.add_argument('--model',choices=list(MODELS))
    args=parser.parse_args()
    try:
        if args.action=='layouts':layouts(args.model)
        elif args.action=='freeze':freeze()
        else:worker(args.model)
    except BaseException as exc:
        write(OUT/f'error_{args.action}_{args.model}_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
