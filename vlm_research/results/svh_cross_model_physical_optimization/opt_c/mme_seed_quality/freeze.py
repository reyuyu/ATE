"""Freeze the executable protocol and CPU token boundaries, before GPU use."""
import collections
from transformers import AutoProcessor
from .common import *

def main():
    assert not (OUT/'protocol.json').exists()
    rows=read(OUT/'sample_manifest.json')
    processor=AutoProcessor.from_pretrained(MODELS['llava'],local_files_only=True)
    bindings={}
    for s in rows:
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':s['question']}]}]
        prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        expanded=prompt.replace(processor.image_token,processor.image_token*576)
        ids=processor.tokenizer(expanded)['input_ids']
        image_id=processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        assert ids.count(image_id)==576
        b=dict(prompt=prompt,input_ids=ids)
        if s['benchmark']=='SEED':
            cands=[]
            for answer in s['choices']:
                full=processor.tokenizer(expanded+' '+answer+processor.tokenizer.eos_token)['input_ids']
                assert full[:len(ids)]==ids,s['key']
                tail=full[len(ids):];assert tail and tail[-1]==processor.tokenizer.eos_token_id
                cands.append(tail)
            b['candidate_token_ids']=cands
        bindings[s['key']]=b
    import gzip
    (OUT/'llava_input_bindings.json.gz').write_bytes(gzip.compress(
        json.dumps(bindings,ensure_ascii=False,sort_keys=True).encode(),mtime=0))
    models={}
    for kind,path in MODELS.items():
        p=Path(path)
        files=[f for f in p.iterdir() if f.is_file() and (f.suffix in ('.json','.safetensors') or f.name=='tokenizer.model')]
        models[kind]=dict(path=path,files={f.name:sha(f) for f in sorted(files)})
    atomic(OUT/'model_manifests.json',models)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    free=shutil.disk_usage('/data').free
    assert free>=20*2**30
    atomic(OUT/'resource_preflight.json',dict(at=now(),authorization='User explicitly authorized at most two idle GPUs for this round',
        gpu_snapshot=gpu,data_free_bytes=free,min_free_bytes=20*2**30,
        qwen_status='BLOCKED_RESOURCE_BEFORE_GPU',qwen_reason='Frozen default max_pixels=16777216 yields over 16000 image tokens; a single native eager FP32 softmax exceeds 32 GiB, before model weights; RTX3090 has 24 GiB. No resize/cap/kernel change authorized.',
        qwen_counterexamples={'MME':'MME__01340','SEED':'SEED__06472'},
        qwen_no_gpu_inference_started=True,llava_image_tokens=576))
    longest={};smoke=[]
    for rank in range(2):
        for bench in ('MME','SEED'):
            pool=[s for s in rows if s['shard']==rank and s['benchmark']==bench]
            lo=min(pool,key=lambda s:(len(bindings[s['key']]['input_ids']),s['key']))
            hi=max(pool,key=lambda s:(len(bindings[s['key']]['input_ids']),s['width']*s['height'],s['key']))
            # The largest source image also exercises real preprocessing.
            large=max(pool,key=lambda s:(s['width']*s['height'],s['key']))
            smoke += [dict(key=s['key'],rank=rank) for s in (lo,hi,large)]
    smoke=[dict(key=k,rank=r) for k,r in sorted({(s['key'],s['rank']) for s in smoke})]
    atomic(OUT/'protocol.json',dict(status='FROZEN_BEFORE_GPU',at=now(),quality_only=True,
        base_commit='fecac1ee7af632ffbbdd48ca79555f9f5ee6ff4f',
        llava_original_source='5627bc822a7827ae11edfaec84c31f6b73398e25: historical ResearchWriter/AnswerWriter physical C3-V2',
        qwen_original_source='30d3e7a0c9318e71a5951aca9a05b6b78108db72: Physical-Original Execution',
        conditions=['Native','Physical-C3'],models=MODELS,
        model_manifests_sha256=sha(OUT/'model_manifests.json'),
        sample_manifest_sha256=sha(OUT/'sample_manifest.json'),
        llava_bindings_sha256=sha(OUT/'llava_input_bindings.json.gz'),
        gpu_uuids=UUIDS,seed=20261106,smoke=smoke,
        llava=dict(dtype='FP16',transformers='4.49.0',eager=True,cutoff=20,skip_layers=[2,3,4,5]),
        qwen=dict(dtype='BF16',transformers='4.57.6',eager=True,cutoff=22,skip_layers=list(range(1,7)),
            numerical_status='Known dense/physical difference disclosed; no repairs; resource gate currently BLOCKED'),
        generation=dict(do_sample=False,max_new_tokens=32,logits_to_keep=1,eos='original generation_config'),
        mme=dict(n=2374,images=1187,parser='Pinned official first-four-character yes/no parser',
            scoring='task 100*Accuracy +100*Accuracy+; sum tasks; Perception/2000,Cognition/800,total/2800'),
        seed_bench=dict(n=14233,images=5624,unique_key='original annotation record index; duplicate upstream IDs retained',
            primary_metric='Accuracy percent; option with minimum mean candidate NLL is predicted',
            aggregation='FP32 mean negative conditional log probability of candidate content',
            llava_candidate_suffix='one native assistant separator space + complete choice text + </s>',
            qwen_candidate_suffix='complete choice content only, model-specific official Qwen interface convention',
            answer_steps='first candidate token predicted at P-1; every later candidate token uses actual incremental Decode',
            cache_fork='independent deepcopy of immutable prompt cache; strict equality to cold complete option requests required in Smoke',
            protocol_adaptation='Native HF chat template and frozen historical Processor retained. Official mean-NLL ranking is ported; published scores may use another conversation template/preprocessing and are not directly interchangeable.',
            source_revision='991e849be802b0de1cd4d45d929bf18e249f5b54'),
        bootstrap=dict(repetitions=20000,seed=20261106,cluster='Image SHA256',confidence=.95,exploratory=True,
            mme='Recompute task Accuracy and paired-image Accuracy+ within each bootstrap; weight all questions in sampled clusters'),
        no_timing=True,no_physical_changes=True,no_training=True,no_extra_conditions=True))
    atomic(OUT/'source_manifest.json',dict(at=now(),files=source_hashes()))
    print('PROTOCOL FROZEN',len(rows),'SMOKE',smoke,flush=True)

if __name__=='__main__':
    import shutil
    main()
