"""New SDPA Native/Original matched OCRBench1000, restart-safe per request.

Executed inside the single resident GPU worker after independently saved gates.
All completed and validated same-backend requests are reused, never regenerated.
"""
import sys,json,time,gc,gzip,shutil,os,datetime,traceback
from pathlib import Path
O=Path('/home/featurize/ate-experiments/qwen3vl-ocrbench-memory-efficient-attention-22875-20261113');L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113');T=Path('/home/featurize/work/ATE/vlm_research/results/qwen3vl_ocrbench_memory_efficient_attention')
sys.path[:0]=[str(O),str(L),str(L/'sources/qwen')]
import torch
from run_common import atomic,sha,score,RAW
from qwen_frozen_benchmark_inputs import prepare,digest
from physical_engine import EfficientExecution
from runtime import generate
model=MODEL;proc=PROCESSOR;P=PROTOCOL;pb=PROTOCOL_SHA256
assert json.loads((O/'gate_decision.json').read_text())['formal_allowed']
assert json.loads((O/'native_memory_gate.json').read_text())['status']=='PASS'
assert json.loads((O/'svh_physical_gate.json').read_text())['mask_position_deepstack_bypass_KV_PASS']
assert len(model.model.language_model.layers)==36
for receipt in ['new_native_source_binding.json','new_physical_source_binding.json']:
    binding=json.loads((O/receipt).read_text());assert binding['protocol_sha256']==pb
    for f,h in binding['source_sha256'].items():assert sha(O/f)==h
rows=json.loads((L/'results/qwen/rows.json').read_text());by={r['question_id']:r for r in rows};conditions=['Native','Physical-Original']
assert len(rows)==1000 and len({r['image_sha256'] for r in rows})==930
records={c:{} for c in conditions};pending=[];uuids=set();start=time.time();new_count=0;layouts_saved=0
for c in conditions:
    for f in sorted((O/'predictions'/c).glob('*.json')):
        r=json.loads(f.read_text());q=r['question_id'];s=by[q];g=r['generation']
        assert f.stem==q and r['protocol_sha256']==pb and r['input_layout_sha256']==P['all_layout_hashes'][q]
        assert r['image_sha256']==s['image_sha256'] and r['request_sha256']==s['request_SHA256']
        assert g['condition']==c and g['backend']=='SDPA_EFFICIENT_ATTENTION_ONLY' and g['finite_logits'] and g['independent_cache'] and g['natural_eos']
        assert len(g['token_ids'])==g['generated_tokens']<=128
        assert g['request_cache_uuid'] not in uuids;uuids.add(g['request_cache_uuid'])
        assert score(q,g['text'])=={k:r[k] for k in ['official_score','independent_score']}
        records[c][q]=r;pending.append(f)
reused={c:len(records[c]) for c in conditions}
assert all(n>=3 for n in reused.values())
source={f:sha(O/f) for f in ['attention_backend.py','runtime.py','physical_engine.py','formal_run.py']}
metadata=dict(status='RUNNING',protocol_sha256=pb,source_sha256=source,model=P['model'],revision=P['revision'],dtype='BF16',attention='SDPA_EFFICIENT_ATTENTION_ONLY',expected_each=1000,reused_same_new_backend=reused,old_Eager_predictions_reused=0,model_loads=1,run_order='Original question order; Native then Physical for each missing question, independent KV per request',start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gate_decision_sha256=sha(O/'gate_decision.json'),numerical_warning=True)
atomic(O/'run_metadata.json',metadata)
light=['protocol.json','environment.json','cpu_inputs.json','cpu_tests.json','physical_cpu_tests.json','compact_mathematical_equivalence.json','static_memory.json','installed_source.json','historical_protected.json','history_search.json','gpu_backend_probe.json','native_memory_gate.json','svh_physical_gate.json','gate_decision.json','gate_report_zh.md','frozen_core_noop.json','numerical_comparison.json','numerical_comparison.csv','memory_profile.csv','oom_root_cause_zh.md','attention_backend_compatibility.md','new_native_source_binding.json','new_physical_source_binding.json','run_metadata.json']
light+=list(source)+['cpu_prepare.py','physical_cpu_test.py','compact_math_test.py','native_gate_run.py','svh_stage.py','review_gate.py']
def copy_atomic(p,d):
    d.parent.mkdir(parents=True,exist_ok=True);tmp=d.with_suffix(d.suffix+'.tmp');shutil.copyfile(p,tmp);os.replace(tmp,d)
def gzip_atomic(p,d):
    d.parent.mkdir(parents=True,exist_ok=True);tmp=d.with_suffix(d.suffix+'.tmp')
    with p.open('rb') as src,gzip.open(tmp,'wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
    os.replace(tmp,d)
def backup():
    for p in pending:
        if p.is_relative_to(O/'predictions'):dest=T/'checkpoint'/p.relative_to(O)
        else:dest=T/'checkpoint'/p.relative_to(O)
        gzip_atomic(p,dest.with_suffix(dest.suffix+'.gz')) if p.suffix=='.json' else copy_atomic(p,dest)
    pending.clear()
    for f in light:
        if (O/f).exists():copy_atomic(O/f,T/f)
    for f in ['progress.json','failure.json']:
        if (O/f).exists():copy_atomic(O/f,T/f)
def progress(status,current=None,**kw):
    atomic(O/'progress.json',dict(status=status,current=current,completed={c:len(records[c]) for c in conditions},expected_each=1000,new_generated=new_count,reused_same_backend=reused,elapsed_seconds=time.time()-start,**kw))
    return
progress('FORMAL_RUNNING');backup()
try:
    for s in rows:
        q=s['question_id']
        if all(q in records[c] for c in conditions):continue
        progress('FORMAL_RUNNING',dict(question_id=q,phase='CPU_INPUT_VALIDATION'))
        cpu_begin=time.perf_counter();x,l=prepare(proc,s);assert digest(l)==P['all_layout_hashes'][q]
        assert l['image_tokens']==len(l['image_positions']) and l['input_ids']==x['input_ids'][0].tolist()
        p=O/'input_layouts'/(q+'.json.gz');p.parent.mkdir(exist_ok=True)
        if not p.exists():
            tmp=p.with_suffix('.tmp')
            with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=3) as f:json.dump(l,f,ensure_ascii=False)
            os.replace(tmp,p)
        pending.append(p);layouts_saved+=1
        cpu_ms=(time.perf_counter()-cpu_begin)*1000
        for c in conditions:
            if q in records[c]:continue
            progress('FORMAL_RUNNING',dict(question_id=q,condition=c,input_length=l['input_length'],image_tokens=l['image_tokens']))
            assert torch.cuda.mem_get_info()[0]>4*2**30,'GPU_RESOURCE_CHANGED_BEFORE_REQUEST'
            execution=EfficientExecution(model,l,'C',max_tokens=128,audit=False) if c=='Physical-Original' else None
            g,_=generate(model,proc,x,l,c,profile=False,capture=False,execution=execution)
            assert g['request_cache_uuid'] not in uuids;uuids.add(g['request_cache_uuid'])
            assert g['max_new_tokens']==128 and g['finite_logits'] and g['natural_eos'] and g['independent_cache']
            if c=='Physical-Original':
                a=g['physical_audit'];assert a['original_visual_KV_live']==a['original_visual_KV_capacity']==0
                assert a['deepstack']==[0,1,2] and not a['physical_fallback']
                assert all(k['live_tokens']==0 for k in g['kv']['layers'] if k['layer'] in range(1,7))
                assert all(k['live_tokens']==l['input_length']-l['image_tokens']+g['generated_tokens']-1 for k in g['kv']['layers'] if k['layer'] not in range(1,7))
            r=dict(question_id=q,image_sha256=s['image_sha256'],request_sha256=s['request_SHA256'],protocol_sha256=pb,input_layout_sha256=digest(l),input_tensor_hashes=l['input_tensor_hashes'],source_sha256=source,input_CPU_ms=cpu_ms,generation=g,**score(q,g['text']))
            p=O/'predictions'/c/(q+'.json');atomic(p,r);records[c][q]=r;pending.append(p);new_count+=1
            print('FORMAL_DONE',c,q,'n',l['input_length'],'tokens',g['generated_tokens'],'EOS',g['eos'],'score',r['official_score'],'coverage',len(records[c]),'peak_GiB',round(g['memory']['peak_allocated']/2**30,3),flush=True)
            del execution,g,r;gc.collect();torch.cuda.empty_cache()
            progress('FORMAL_RUNNING',dict(question_id=q,condition=c))
            if new_count%25==0:backup()
        del x,l;gc.collect()
    assert all(len(records[c])==1000 for c in conditions)
    progress('GENERATION_COMPLETE_AWAIT_CPU_FINAL_AUDIT');metadata['status']='GENERATION_COMPLETE_AWAIT_CPU_FINAL_AUDIT';metadata['completed']={c:len(records[c]) for c in conditions};metadata['new_generated']=new_count;metadata['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(O/'run_metadata.json',metadata);backup()
    print('NEW_BACKEND_MATCHED_GENERATION_COMPLETE_1000_PLUS_1000',flush=True)
except BaseException as e:
    progress('BLOCKED',error_type=type(e).__name__,error=str(e))
    atomic(O/'formal_failure.json',dict(status='BLOCKED',error_type=type(e).__name__,error=str(e),request_partial=getattr(e,'request_partial',None),traceback=traceback.format_exc()))
    copy_atomic(O/'formal_failure.json',T/'formal_failure.json');backup();raise
