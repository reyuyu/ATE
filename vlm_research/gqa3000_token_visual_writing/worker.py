"""One immutable image shard and one model per UUID; resumable atomic records."""
import argparse,fcntl,json,os,time,traceback
import numpy as np
import torch,transformers
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from paper_rq1_single_layer_gqa6.run import prepared,tensor_hashes
from semantic_visual_writers.engine import generate as old_generate,WriterMask as OldMask
from vlm_common import load_model,gpu_memory
import query_role_smoke_fix.run as repaired
from .common import *
from .engine import generate,teacher_force,native_teacher,ReceiverMask

def spec_for(s,index):
    info=index[key(s)];assert sha256(OUT/info['path'])==info['sha256'];spec=read(OUT/info['path'])
    assert spec['selection_binding']==info['selection_binding']==digest(spec['writers']);return spec

@torch.inference_mode()
def smoke(model,processor,p,rank,samples,index,gold,hashes,attempt):
    results=[];cross={};footprint=hook_footprint(model)
    names=['All-Writer','No-Writer','Semantic-k','Question-Last-k','R-Last-k',f'Question-Random-k-{SEEDS[0]}',
        f'R-Random-k-{SEEDS[0]}','Drop-Semantic','Semantic-first-Q20','Question-Last-Q100']
    for qid in read(OUT/'input_design_audit.json')['smoke_ids']:
        s=next(s for s in samples if s['question_id']==qid);spec=spec_for(s,index);inputs=prepared(model,processor,s);ids=gold[key(s)]['token_ids']
        assert tensor_hashes(inputs)==hashes[key(s)];versions={n:v._version for n,v in model.named_parameters()}
        native_ids,rec=repaired.native(model,inputs)
        reference=[v[-1].float() for v in rec.logits]
        native_gold=native_teacher(model,inputs,ids)
        cross[qid+'_native_gold']=native_gold;cross[qid+'_native_greedy']=np.stack([v.numpy() for v in reference])
        parent_result=generate(model,processor.tokenizer,inputs,[],noop=True,capture=True,retain_cache=True)
        assert parent_result[0]['token_ids']==native_ids and len(reference)==len(parent_result[2])
        assert all(torch.equal(torch.from_numpy(a).float(),b) for a,b in zip(parent_result[2],reference))
        parent=parent_result[4];parent_hash=cache_fingerprint(parent)
        for name in ['No-Writer','Semantic-k','Semantic-first-Q20']:
            g,first,trace,_,_=generate(model,processor.tokenizer,inputs,spec['writers'][name],noop=True,capture=True)
            tf,tlogits,_,_=teacher_force(model,inputs,spec['writers'][name],ids,noop=True)
            assert g['token_ids']==native_ids and len(trace)==len(reference)
            assert all(torch.equal(torch.from_numpy(a).float(),b) for a,b in zip(trace,reference))
            assert torch.equal(torch.from_numpy(tlogits).float(),torch.from_numpy(native_gold).float())
        checks={}
        for name in names:
            writers=spec['writers'][name];counts=[0]*32;handles=[]
            try:
                for k,layer in enumerate(model_layers(model)):
                    def mlp(m,args,out,k=k):assert torch.isfinite(out).all();counts[k]+=1
                    handles.append(layer.mlp.register_forward_hook(mlp))
                gen,first,trace,_,own=generate(model,processor.tokenizer,inputs,writers,capture=True,retain_cache=True)
                tf,tlogits,_,tf_cache=teacher_force(model,inputs,writers,ids,retain_cache=True)
            finally:
                for h in handles:h.remove()
            assert counts==[gen['length']+len(ids)]*32
            assert torch.equal(torch.from_numpy(first),torch.from_numpy(tlogits[0]))
            assert [x['absolute_query'] for x in tf['cache_lengths']]==gold[key(s)]['target_query_positions']
            assert cache_fingerprint(parent)==parent_hash
            for a,b in [(parent,own),(own,tf_cache),(parent,tf_cache)]:
                assert all(x.data_ptr()!=y.data_ptr() for layer_a,layer_b in zip(a.to_legacy_cache(),b.to_legacy_cache()) for x,y in zip(layer_a,layer_b))
            old,_,old_trace,_=old_generate(model,processor.tokenizer,inputs,writers,capture=True)
            assert gen['token_ids']==old['token_ids'] and len(trace)==len(old_trace)
            assert all(torch.equal(torch.from_numpy(a).float(),b) for a,b in zip(trace,old_trace))
            with OldMask(model,inputs,writers):old_tf=native_teacher(model,inputs,ids)
            assert torch.equal(torch.from_numpy(tlogits).float(),torch.from_numpy(old_tf).float())
            assert tensor_hashes(inputs)==hashes[key(s)] and hook_footprint(model)==footprint
            checks[name]=dict(greedy=gen,teacher_forced=tf,old_new_every_logit_exact=True,greedy_tf_first_exact=True,cache_independent=True,mlp_counts=counts)
            del own,tf_cache
        try:
            with ReceiverMask(model,inputs,[],False):raise RuntimeError('intentional hook cleanup')
        except RuntimeError as e:assert str(e)=='intentional hook cleanup'
        assert hook_footprint(model)==footprint and versions=={n:v._version for n,v in model.named_parameters()}
        atomic_json(OUT/f'shard{rank}/attempts/{attempt}/smoke_cases/{qid}.json',dict(passed=True,question_id=qid,checks=checks,all_noops_native_exact=True,all_gold_positions_aligned=True,exception_cleanup=True))
        results.append(dict(question_id=qid,passed=True,gold_tokens=len(ids),conditions=len(names)));print('SMOKE PASS',rank,qid,flush=True)
    rr=atomic_npz(RAW/f'smoke/shard{rank}/{attempt}/cross_gpu.npz',**cross)
    return dict(passed=True,rank=rank,binding=digest(p),cases=results,cross_gpu_raw=rr,strict_torch_equal=True,old_mask_compatibility_exact=True)

def paths(s,uid,rank):
    return OUT/f'shard{rank}/checkpoints'/key(s)/(uid+'.json'),RAW/f'main/shard{rank}'/key(s)/(uid+'.npz')

def validate_saved(row,s,spec,uid,p,rank,gold,hashes):
    assert row['binding']==digest(p) and row['code_commit']==p['code_commit'] and row['rank']==rank
    assert row['sample_sha256']==digest(s) and row['selection_binding']==spec['selection_binding'] and row['canonical_id']==uid
    assert row['writers']==spec['unique'][uid]['positions'] and row['labels']==spec['unique'][uid]['labels'] and row['input_hashes']==hashes[key(s)]
    assert sha256(row['raw']['path'])==row['raw']['sha256'];g=row['generation'];tf=row['teacher_forced']
    assert row['result_sha256']==digest(dict(generation=g,teacher_forced=tf))
    assert g['prediction']==s['answer'] if g['correct'] else g['prediction']!=s['answer']
    assert g['audit']['passed'] and tf['audit']['passed'] and tf['gold_token_ids']==gold[key(s)]['token_ids']
    with np.load(row['raw']['path'],allow_pickle=False) as z:
        metadata=dict(row);metadata.pop('raw');assert metadata==json.loads(str(z['record_metadata']))
        per,nll=gold_losses(z['gold_logits'],tf['gold_token_ids']);assert per==tf['per_token_nll'] and nll==tf['nll']
        assert np.array_equal(z['first_logits'],z['gold_logits'][0]) and int(z['first_logits'].argmax())==g['token_ids'][0]
        for label in ['greedy_audit','gold_audit']:
            a=z[label];assert (a[:,0]==0).all() and (a[:,1]<=.002).all() and (a[:,2]==1).all() and (a[:,3]>=0).all() and (a[:,4:]==1).all()
    return row

def collect_record(model,processor,s,inputs,spec,uid,p,rank,gold,hashes,capture):
    dst,raw_path=paths(s,uid,rank)
    if dst.exists():validate_saved(read(dst),s,spec,uid,p,rank,gold,hashes);return 'resumed'
    if raw_path.exists():
        # Recover a fully fsynced raw record left just before atomic checkpoint publication.
        with np.load(raw_path,allow_pickle=False) as z:row=json.loads(str(z['record_metadata']))
        row['raw']=dict(path=str(raw_path),sha256=sha256(raw_path),bytes=raw_path.stat().st_size)
        validate_saved(row,s,spec,uid,p,rank,gold,hashes);atomic_json(dst,row);return 'raw_recovered'
    before=tensor_hashes(inputs);footprint=hook_footprint(model);versions={n:v._version for n,v in model.named_parameters()}
    writers=spec['unique'][uid]['positions'];g,first,trace,ga,_=generate(model,processor.tokenizer,inputs,writers,capture=capture)
    tf,tlogits,ta,_=teacher_force(model,inputs,writers,gold[key(s)]['token_ids'])
    assert torch.equal(torch.from_numpy(first),torch.from_numpy(tlogits[0]))
    assert [v['absolute_query'] for v in tf['cache_lengths']]==gold[key(s)]['target_query_positions']
    assert before==hashes[key(s)]==tensor_hashes(inputs) and footprint==hook_footprint(model) and versions=={n:v._version for n,v in model.named_parameters()}
    g['correct']=int(g['prediction']==s['answer'])
    row=dict(binding=digest(p),code_commit=p['code_commit'],rank=rank,sample_sha256=digest(s),question_id=s['question_id'],task=s['task'],
        image_sha256=s['image_sha256'],gold=s['answer'],canonical_id=uid,selection_binding=spec['selection_binding'],writers=writers,
        labels=spec['unique'][uid]['labels'],input_hashes=before,generation=g,teacher_forced=tf,result_sha256=digest(dict(generation=g,teacher_forced=tf)))
    arrays=dict(first_logits=first,gold_logits=tlogits,greedy_audit=ga,gold_audit=ta,record_metadata=np.array(json.dumps(row,sort_keys=True,separators=(',',':'))))
    if capture:arrays['greedy_valid_logits']=trace
    row['raw']=atomic_npz(raw_path,**arrays);atomic_json(dst,row);return 'new'

def main(rank,attempt):
    started=time.perf_counter();torch.set_num_threads(4);p=verify();lockpath=OUT/f'shard{rank}/worker.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert_no_stop()
    preflight(p['gpus'][rank],rank,attempt);assert transformers.__version__=='4.49.0'
    samples=read(OUT/'sample_manifest.json')['samples'];index=read(OUT/'selection_index.json');gold=read(OUT/'gold_manifest.json');hashes=read(OUT/'input_hashes.json');shards=read(OUT/'shards.json')['image_assignments']
    own=[s for s in samples if shards[s['image_sha256']]==rank];model,processor,env=load_model(MODEL);assert torch.cuda.device_count()==1
    env.update(pid=os.getpid(),rank=rank,gpu=p['gpus'][rank],python_executable=os.sys.executable)
    atomic_json(OUT/f'shard{rank}/attempts/{attempt}/environment.json',env)
    smoke_path=OUT/f'shard{rank}/attempts/{attempt}/smoke.json';assert not smoke_path.exists(),'Attempt already exists: select a distinct audited resume attempt'
    atomic_json(smoke_path,smoke(model,processor,p,rank,samples,index,gold,hashes,attempt))
    print('WAITING BOTH-GPU GATE',rank,flush=True)
    gate_path=OUT/f'attempts/{attempt}/both_gpu_smoke_gate.json'
    while not gate_path.exists():
        assert_no_stop();assert time.perf_counter()-started<1800,'GPU smoke barrier timeout';time.sleep(.25)
    gate=read(gate_path);assert gate['passed'] and gate['binding']==digest(p)
    swapkeys={k for pair in read(OUT/'image_swap_manifest.json')['pairs'] for k in pair['sample_keys']};counts={'new':0,'resumed':0,'raw_recovered':0}
    for i,s in enumerate(own):
        assert_no_stop();spec=spec_for(s,index);inputs=prepared(model,processor,s);assert tensor_hashes(inputs)==hashes[key(s)]
        for uid in spec['unique']:
            assert_no_stop();capture=key(s) in swapkeys and uid==spec['aliases']['No-Writer']
            mode=collect_record(model,processor,s,inputs,spec,uid,p,rank,gold,hashes,capture);counts[mode]+=1
        if (i+1)%10==0:
            print('MAIN',rank,i+1,'/',len(own),'canonical',sum(counts.values()),'seconds',round(time.perf_counter()-started,1),flush=True)
            # Small append-only progress stream; completed scientific records are immutable.
            with (OUT/f'shard{rank}/progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(questions=i+1,canonical=sum(counts.values()),seconds=time.perf_counter()-started))+'\n')
    verify();expected=sum(index[key(s)]['unique_count'] for s in own)
    assert sum(counts.values())==expected==len(list((OUT/f'shard{rank}/checkpoints').glob('*/*.json')))
    atomic_json(OUT/f'shard{rank}/completion.json',dict(passed=True,binding=digest(p),rank=rank,attempt=attempt,pid=os.getpid(),questions=len(own),canonical_evaluations=expected,
        logical_records=len(own)*len(CONDITIONS),counts=counts,seconds=time.perf_counter()-started,memory=gpu_memory(),one_model=True))
    print('WORKER COMPLETE',rank,expected,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,choices=[0,1],required=True);parser.add_argument('--attempt',default='initial');a=parser.parse_args()
    assert a.attempt.replace('_','').replace('-','').isalnum()
    try:main(a.rank,a.attempt)
    except BaseException:
        failure=dict(status='FAILED',rank=a.rank,pid=os.getpid(),traceback=traceback.format_exc(),hardware=hardware())
        if torch.cuda.is_initialized():
            try:failure['cuda_memory_summary']=torch.cuda.memory_summary()
            except BaseException:pass
        atomic_json(OUT/f'shard{a.rank}/failure_{time.time_ns()}.json',failure)
        if not (OUT/'STOP.json').exists():atomic_json(OUT/'STOP.json',failure)
        raise
