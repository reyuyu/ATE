import argparse,time,traceback,shutil,fcntl
from benchmarks.smoke import hook_footprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from paper_rq1.engine import cache_fingerprint
from vlm_common import load_model,gpu_memory
from relay_source_causality.common import hardware
from .engine import observed,plain,Capture
from .common import *
def verify():
    p=read(OUT/'protocol.json');assert p['sources']==sources() and p['case_manifest_sha256']==binding() and p['rules_sha256']==sha256(OUT/'ranking_rules.json');return p
def independent(a,b):
    return all(x.data_ptr()!=y.data_ptr() for aa,bb in zip(a.to_legacy_cache(),b.to_legacy_cache()) for x,y in zip(aa,bb))
@torch.inference_mode()
def work(rank):
    torch.set_num_threads(4);started=time.perf_counter();p=verify();gpu=p['gpus'][rank];state=hardware();u=gpu['uuid'];fields=[x.strip() for x in next(l for l in state['table'].splitlines() if l.split(',')[1].strip()==u).split(',')];busy=[l for l in state['compute_processes'].splitlines() if l.split(',')[0].strip()==u]
    ok=not busy and int(fields[3])<200 and int(fields[4])>22000 and int(fields[5])==0;atomic_json(OUT/f'worker{rank}/gpu_preflight.json',dict(**state,selected=gpu,passed=ok));assert ok and os.environ.get('CUDA_VISIBLE_DEVICES')==u
    lock=(OUT/f'worker{rank}/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model,proc,env=load_model(MODEL);assert torch.cuda.device_count()==1;atomic_json(OUT/f'worker{rank}/environment.json',{**env,'pid':os.getpid(),'selected_gpu':gpu})
    serial=p.get('single_gpu',False)
    subset=cases() if serial else [v for v in cases() if v['rank']==rank];assert len(subset)==(16 if serial else 8)
    for i,v in enumerate(subset):
        if list(OUT.glob('worker*/STOP.json')):raise RuntimeError('Another worker stopped')
        q=v['question_id'];s=v['sample'];inputs=prepared(model,proc,s);assert tensor_hashes(inputs)==v['input_hashes'];footprint=hook_footprint(model)
        if v['smoke']:
            ref,features,parent=plain(model,inputs);fp=cache_fingerprint(parent)
        logits,mass,legal,probs,audit,cache=observed(model,inputs,keep_image=v['smoke'])
        if v['smoke']:
            assert torch.equal(ref,logits) and features==audit['visual_features_sha256'] and cache_fingerprint(parent)==fp and independent(parent,cache)
            nl,nm,nk,npb,na,nc=observed(model,inputs,keep_image=True)
            assert torch.equal(logits,nl) and np.array_equal(mass,nm) and np.array_equal(probs,npb) and np.array_equal(legal,nk) and na['visual_features_sha256']==features
            assert independent(cache,nc) and cache_fingerprint(parent)==fp
            al,am,ak,apb,aa,ac=observed(model,inputs,allwriter=True,keep_image=True)
            assert torch.equal(torch.from_numpy(probs),torch.from_numpy(apb)) and np.array_equal(mass[:,:,:-1],am[:,:,:-1]),'Native and All-Writer R attention mismatch'
            assert aa['visual_features_sha256']==features and independent(cache,ac) and cache_fingerprint(parent)==fp
            assert all(x['mask_modified_edges']==576 for x in aa['layer_audits']) and hook_footprint(model)==footprint
            try:
                with Capture(model,inputs):raise RuntimeError('intentional hook cleanup test')
            except RuntimeError as e:assert str(e)=='intentional hook cleanup test'
            assert hook_footprint(model)==footprint
            smoke_raw=atomic_npz(RAW/f'worker{rank}'/(q+'_smoke.npz'),native_R_image_probabilities=probs,allwriter_R_image_probabilities=apb,native_valid_logits=logits.numpy(),noop_valid_logits=nl.numpy(),allwriter_valid_logits=al.numpy())
            atomic_json(OUT/f'worker{rank}'/('smoke_'+q+'.json'),dict(passed=True,question_id=q,native_noop_logits_torch_equal=True,native_noop_R_image_probabilities_torch_equal=True,native_allwriter_R_image_probabilities_torch_equal=True,all32_layers_all32_heads_all576_image_keys=True,independent_KV_caches=True,input_and_visual_hash_exact=True,hook_cleanup_on_exception=True,native_audit=audit,allwriter_audit=aa,raw=smoke_raw))
            del ref,parent,nl,nm,nk,npb,na,nc,al,am,ak,apb,aa,ac
            if not serial:
                atomic_json(OUT/f'worker{rank}/ready.json',dict(passed=True,pid=os.getpid(),rank=rank))
                deadline=time.monotonic()+90
                while not (OUT/'GO.json').exists():
                    assert not list(OUT.glob('worker*/STOP.json')) and time.monotonic()<deadline,'Smoke barrier failed or timed out'
                    time.sleep(.2)
            print('SMOKE PASS',q,flush=True)
        del cache,probs
        raw=atomic_npz(OUT/'measurements'/(q+'.npz'),image_mass=mass,legal_key_counts=legal,query_positions=np.array(v['spec']['relay_positions']+[s['input_mapping']['p_minus_one']]),native_valid_logits=logits.numpy())
        atomic_json(OUT/'checkpoints'/(q+'.json'),dict(passed=True,question_id=q,rank=rank,kind=v['kind'],case_manifest_sha256=binding(),image_sha256=s['image_sha256'],input_hashes=v['input_hashes'],native_audit=audit,raw=raw,condition='Native Full; no hard mask'))
        assert footprint==hook_footprint(model);print('NATIVE PREFILL DONE',q,rank,'seconds',round(time.perf_counter()-started,2),flush=True)
    verify();atomic_json(OUT/f'worker{rank}/completion.json',dict(passed=True,pid=os.getpid(),rank=rank,cases=len(subset),native_main_prefills=len(subset),validation_extra_prefills=6 if serial else 3,decode_steps=0,new_generated_answers=0,seconds=time.perf_counter()-started,memory=gpu_memory()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--rank',type=int,required=True);args=a.parse_args()
    try:work(args.rank)
    except BaseException:
        atomic_json(OUT/f'worker{args.rank}/STOP.json',dict(passed=False,pid=os.getpid(),traceback=traceback.format_exc(),hardware=hardware()));raise
