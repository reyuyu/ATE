"""CPU-only reuse audit and pre-prediction protocol freeze."""
import collections
import shutil
import subprocess
from .common import *


def main():
    assert not (OUT/'protocol.json').exists(),'Never overwrite a frozen protocol'
    assert shutil.disk_usage('/data').free>=20*2**30
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    names=['sample_manifest.json','image_manifest.json','qwen_input_bindings.json.gz',
        'qwen_processor_2048.json','qwen_processor_validation.json','qwen_image_processor_audit.json',
        'qwen_visual_token_statistics.csv','model_manifests.json','analysis_plan.json']
    reuse=[]
    for name in names:
        src=PREVIOUS/name
        blob=subprocess.check_output(['git','show',f'{PARENT}:{src.relative_to(ROOT)}'],cwd=ROOT)
        import hashlib
        assert hashlib.sha256(blob).hexdigest()==sha(src),name
        shutil.copyfile(src,OUT/name);assert sha(OUT/name)==sha(src)
        reuse.append(dict(file=name,sha256=sha(src),source_commit=PARENT,byte_identical=True))
    assert read(OUT/'qwen_processor_validation.json')['actual_cap_exceedances']==0
    rows=read(OUT/'sample_manifest.json');bindings=read(OUT/'qwen_input_bindings.json.gz')
    assert len(rows)==16607 and len(bindings)==16607
    assert len({s['key'] for s in rows})==len(rows)
    assert collections.Counter(s['benchmark'] for s in rows)=={'MME':2374,'SEED':14233}
    shards=collections.defaultdict(set)
    for s in rows:shards[s['image_sha256']].add(s['shard'])
    assert all(len(x)==1 for x in shards.values())
    images=read(OUT/'image_manifest.json');assert len(images)==6811
    for path,v in images.items():assert sha(path)==v['sha256'],path
    model=read(OUT/'model_manifests.json')['qwen']
    for name,h in model['files'].items():assert sha(Path(MODELS['qwen'])/name)==h,name
    # Choice lengths, tensor shapes and image geometry only; no prediction access.
    lookup={s['key']:s for s in rows};specs=[];seen=set()
    def add(s,reason):
        k=s['key']
        if k not in seen:specs.append(dict(key=k,rank=s['shard'],reasons=[reason]));seen.add(k)
        else:next(x for x in specs if x['key']==k)['reasons'].append(reason)
    add(lookup['MME__01341'],'previous OOM; 8160 ViT patches')
    add(lookup['MME__01189'],'maximum 8192 ViT patches')
    compare=[]
    for rank in (0,1):
        pool=[s for s in rows if s['shard']==rank]
        for bench in ('MME','SEED'):
            bp=[s for s in pool if s['benchmark']==bench]
            add(max(bp,key=lambda s:(len(bindings[s['key']]['input_ids']),s['key'])),bench+' longest actual input')
        sp=[s for s in pool if s['benchmark']=='SEED']
        add(max(sp,key=lambda s:(len(bindings[s['key']]['input_ids'])+max(map(len,bindings[s['key']]['candidate_token_ids'])),s['key'])),
            'longest complete SEED candidate path')
        ordered=sorted(pool,key=lambda s:(len(bindings[s['key']]['input_ids']),s['key']))
        add(ordered[len(ordered)//2],'median actual sequence length')
        short=min(pool,key=lambda s:(bindings[s['key']]['image_tokens'],len(bindings[s['key']]['input_ids']),s['key']))
        add(short,'short input; backend numerical comparison');compare.append(dict(key=short['key'],rank=rank))
    specs.sort(key=lambda x:(x['rank'],-len(bindings[x['key']]['input_ids']),x['key']))
    protocol=dict(status='FROZEN_BEFORE_GPU',at=now(),parent_commit=PARENT,model_revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
        dtype='bfloat16',torch='2.6.0+cu124',transformers='4.57.6',batch_size=1,
        backend=BACKENDS,vision_blocks=27,text_layers=36,
        efficient_sdpa_allowlist=['FLASH_ATTENTION','EFFICIENT_ATTENTION'],math_and_cudnn_disabled=True,
        official_processor_max_language_image_tokens=2048,processor_longest_edge=2048*32*32,
        physical='unchanged qwen3vl_c3v2_physical.engine.Execution condition C / Physical-Original',
        write_layers=list(range(22)),image_hidden_exit_before_layer=22,answer_skip_layers=list(range(1,7)),
        answer_image_keys_forbidden=True,persistent_cache='original compact text KV',
        conditions=['Native','Physical-C3'],greedy=True,max_new_tokens=32,natural_eos=True,logits_to_keep=1,
        scoring={'MME':'pinned official yes/no Acc + Acc+; Score Points',
            'SEED':'frozen local HF full candidate mean conditional negative LogP; content only, no EOS; final ACC'},
        bootstrap={'iterations':20000,'seed':20261106,'clusters':'Image SHA256',
            'MME':'stratified by task-membership signature; whole clusters','SEED':'whole image clusters',
            'interval':'exploratory unadjusted two-sided percentile 95%; no noninferiority claims'},
        backend_numerical_gates={'features_mean_abs':0.02,'features_max_abs':0.5,
            'native_same_prefix_logits_mean_abs':0.05,'native_same_prefix_logits_max_abs':1.0,
            'free_generation_token_equality':'observed and reported; not required across backends',
            'finite_and_shape_equal':True,'same_backend_hf_noop':'all effective logits torch.equal'},
        preserved_dense_physical_strict_gate={'mean_abs':0.02,'max_abs':0.5,'tokens_equal':True,
            'status':'known historical failure; report without relaxing or using as new semantic gate'},
        smoke=specs,backend_comparison=compare,memory_safety_margin_bytes=512*2**20,
        gpu_uuids=UUIDS,gpu_authorization_required_this_round=True,disk_min_free_bytes=20*2**30,
        stop_on=['OOM','nonfinite','new mask/position/cache semantics error','SHA mismatch','backend fallback'],
        no_timing=True,no_profiler=True,no_flops=True,no_search=True,llava_not_rerun=True)
    atomic(OUT/'protocol.json',protocol)
    atomic(OUT/'reuse_audit.json',dict(status='PASSED',at=now(),files=reuse,raw_image_sha_rechecked=len(images),
        model_files_sha_rechecked=len(model['files']),shards_disjoint=True,all_records_preserved=True))
    atomic(OUT/'source_manifest.json',dict(at=now(),files=source_hashes()))
    atomic(OUT/'completion.json',dict(status='CPU_PREPARED_AWAITING_GPU_AUTHORIZATION',expected_conditions=33214,
        completed_conditions=0,previous_results_unchanged=True,at=now()))
    print('CPU PREPARED',len(rows),'questions; smoke',specs,flush=True)


if __name__=='__main__':main()
