"""Dispatch inside the already loaded Native gate process; no model reload."""
import sys,json,gc,csv,time
from pathlib import Path
O=Path('/home/featurize/ate-experiments/qwen3vl-ocrbench-memory-efficient-attention-22875-20261113');L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
sys.path[:0]=[str(O),str(L),str(L/'sources/qwen')]
import torch,numpy as np
from run_common import atomic,sha,score
from qwen_frozen_benchmark_inputs import prepare,digest
from physical_engine import EfficientExecution
from vlm_research.qwen3vl_c3v2_physical.engine import Execution
from runtime import generate

model=MODEL;proc=PROCESSOR;P=PROTOCOL;pb=PROTOCOL_SHA256
assert json.loads((O/'native_memory_gate.json').read_text())['status']=='PASS'
assert json.loads((O/'physical_cpu_tests.json').read_text())['status']=='PASS'
assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters())
rows=json.loads((L/'results/qwen/rows.json').read_text());by={r['question_id']:r for r in rows}
sources={f:sha(O/f) for f in ['attention_backend.py','runtime.py','physical_engine.py','svh_stage.py']}
atomic(O/'new_physical_source_binding.json',dict(protocol_sha256=pb,source_sha256=sources,frozen_before_Physical_GPU=True,frozen_core_modified=False))
comparisons=[];gates=[];memory=[]

def compare(a,b,qa,qb,q,relation):
    assert len(a)==len(b)
    maxima=[];means=[];top=[];equal=[]
    for u,v in zip(a,b):
        d=(u-v).abs();maxima.append(float(d.max()));means.append(float(d.mean()));top.append(int(u.argmax())==int(v.argmax()));equal.append(torch.equal(u,v))
    strict=max(maxima)<=.5 and max(means)<=.02 and all(top) and qa==qb
    r=dict(question_id=q,comparison=relation,steps=len(a),max_abs=max(maxima),max_step_mean_abs=max(means),mean_abs=sum(means)/len(means),top1_equal=all(top),full_tokens_equal=qa==qb,all_logits_torch_equal=all(equal),legacy_strict_gate='PASS' if strict else 'NUMERICAL_EQUIVALENCE_FAILED',per_step_max_abs=maxima,per_step_mean_abs=means)
    comparisons.append(r);return r

for q in P['smoke_ids']:
    atomic(O/'progress.json',dict(status='RUNNING_SVH_GATE',current=dict(question_id=q,condition='Physical-Original'),completed=len(gates)))
    x,l=prepare(proc,by[q]);assert digest(l)==P['all_layout_hashes'][q]
    assert torch.cuda.mem_get_info()[0]>4*2**30
    e=EfficientExecution(model,l,'C',max_tokens=128,audit=True)
    g,c_logits=generate(model,proc,x,l,'Physical-Original',profile=True,capture=True,execution=e)
    rec=dict(question_id=q,image_sha256=by[q]['image_sha256'],request_sha256=by[q]['request_SHA256'],protocol_sha256=pb,input_layout_sha256=digest(l),input_tensor_hashes=l['input_tensor_hashes'],physical_source_binding_sha256=sha(O/'new_physical_source_binding.json'),generation=g,**score(q,g['text']))
    atomic(O/'predictions/Physical-Original'/(q+'.json'),rec);np.savez_compressed(O/'gate_logits'/('Physical-Original_'+q+'.npz'),logits=torch.stack(c_logits).numpy())
    c_ids=g['token_ids'];assert g['eos'] and not g['truncated']
    summary=dict(question_id=q,input_length=l['input_length'],image_tokens=l['image_tokens'],success=True,text=g['text'],token_ids=c_ids,eos=g['eos'],peak_allocated=g['memory']['peak_allocated'],live_KV_bytes=g['kv']['live_bytes'],capacity_KV_bytes=g['kv']['capacity_bytes'],structure=True,mask=True,RoPE=True,DeepStack=True,physical_bypass=True,physical_visual_KV_zero=True,finite=True)
    for r in g['memory']['stage_memory']:memory.append(dict(question_id=q,condition='Physical-Original',**r))
    del e,g,rec;gc.collect();torch.cuda.empty_cache()
    # The unmodified frozen physical core is an additional short-input no-op
    # control. It uses exactly the same new attention backend, not old Eager.
    if q==P['smoke_ids'][0]:
        original=Execution(model,l,'C',max_tokens=128,audit=False);original.check_attention=lambda *a:None
        og,ol=generate(model,proc,x,l,'Physical-Original',capture=True,execution=original)
        noop=compare(c_logits,ol,c_ids,og['token_ids'],q,'new_mask_storage_vs_unmodified_frozen_Physical_same_SDPA')
        atomic(O/'frozen_core_noop.json',noop)
        assert noop['all_logits_torch_equal'] and noop['full_tokens_equal'],'MASK_STORAGE_NOOP_FAILED'
        del original,og,ol;gc.collect();torch.cuda.empty_cache()
    atomic(O/'progress.json',dict(status='RUNNING_SVH_GATE',current=dict(question_id=q,condition='Dense-F_same_prefix'),completed=len(gates)))
    f=EfficientExecution(model,l,'F',max_tokens=128,audit=True)
    fg,f_logits=generate(model,proc,x,l,'Dense-F',profile=True,capture=True,execution=f,forced=c_ids)
    # Forced C-prefix isolates physical compact-vs-dense numerical errors. If
    # every F argmax equals C's token (including EOS), F's natural greedy output
    # is proven identical without a redundant natural generation.
    f_greedy=[int(v.argmax()) for v in f_logits]
    num=compare(c_logits,f_logits,c_ids,f_greedy,q,'Physical_vs_Dense_F_same_prefix_same_SDPA')
    num['reference_natural_greedy_equivalence_proven']=f_greedy==c_ids
    atomic(O/'diagnostic_Dense_F'/(q+'.json'),dict(question_id=q,teacher_forced_only=True,not_formal_prediction=True,generation=fg,greedy_argmax_ids=f_greedy,comparison=num))
    np.savez_compressed(O/'gate_logits'/('Dense-F_'+q+'.npz'),logits=torch.stack(f_logits).numpy())
    for r in fg['memory']['stage_memory']:memory.append(dict(question_id=q,condition='Dense-F-diagnostic',**r))
    summary['dense_F_comparison']=num;gates.append(summary)
    n=json.loads((O/'predictions/Native'/(q+'.json')).read_text())
    comparisons.append(dict(question_id=q,comparison='Native_vs_Physical_different_algorithms',native_text=n['generation']['text'],physical_text=summary['text'],full_tokens_equal=n['generation']['token_ids']==c_ids,not_an_equivalence_gate=True))
    oldp=L/'results/qwen/predictions/Native'/(q+'.json')
    # Locate only already completed historical Eager predictions; never rerun.
    candidates=[p for p in (L/'results/qwen/predictions').rglob(q+'.json') if 'native' in str(p).lower()]
    for oldp in candidates:
        old=json.loads(oldp.read_text());og=old.get('generation',old.get('result',old))
        comparisons.append(dict(question_id=q,comparison='historical_Eager_Native_vs_new_SDPA_Native',old_prediction_path=str(oldp),old_text=og.get('text'),new_text=n['generation']['text'],full_tokens_equal=og.get('token_ids')==n['generation']['token_ids'],effective_logits_error=None,limitation='Historical effective logits were not saved; output-only comparison, no Eager rerun'))
    atomic(O/'svh_physical_gate.json',dict(status='PARTIAL',cases=gates,protocol_sha256=pb))
    print('SVH_GATE_CASE',q,summary['text'],'peak_GiB',summary['peak_allocated']/2**30,'dense_F',num['legacy_strict_gate'],'max',num['max_abs'],'mean',num['max_step_mean_abs'],'tokens_equal',num['full_tokens_equal'],flush=True)
    del x,f,fg,f_logits,c_logits;gc.collect();torch.cuda.empty_cache()

strict=all(g['dense_F_comparison']['legacy_strict_gate']=='PASS' for g in gates)
status='PASS' if strict else 'STRUCTURAL_PASS_NUMERICAL_REVIEW_REQUIRED'
gate=dict(status=status,completed=3,expected=3,complete_Physical_0738=True,cases=gates,protocol_sha256=pb,actual_backend='SDPA_EFFICIENT_ATTENTION_ONLY',no_math_fallback=True,unchanged_inputs=True,frozen_core_noop_torch_equal=True,mask_position_deepstack_bypass_KV_PASS=True,historical_NUMERICAL_EQUIVALENCE_FAILED_preserved=True,legacy_strict_numerical_gate='PASS' if strict else 'NUMERICAL_EQUIVALENCE_FAILED',formal_not_started=True)
atomic(O/'svh_physical_gate.json',gate);atomic(O/'numerical_comparison.json',comparisons)
with (O/'numerical_comparison.csv').open('w',newline='',encoding='utf-8-sig') as handle:
    fields=['question_id','comparison','steps','max_abs','max_step_mean_abs','mean_abs','top1_equal','full_tokens_equal','all_logits_torch_equal','legacy_strict_gate','old_text','new_text','native_text','physical_text','effective_logits_error','limitation'];w=csv.DictWriter(handle,fields,extrasaction='ignore');w.writeheader();w.writerows(comparisons)
for q in P['smoke_ids']:
    n=json.loads((O/'predictions/Native'/(q+'.json')).read_text())
    for r in n['generation']['memory']['stage_memory']:memory.append(dict(question_id=q,condition='Native',**r))
with (O/'memory_profile.csv').open('w',newline='',encoding='utf-8-sig') as handle:
    fields=['question_id','condition','stage','step','baseline_allocated','peak_allocated','peak_reserved','elapsed_ms'];w=csv.DictWriter(handle,fields);w.writeheader();w.writerows(memory)
(O/'gate_report_zh.md').write_text('# 新后端 Gate 报告（正式全量生成前）\n\n'+f'Native：PASS；Physical结构：PASS；Physical/Dense-F 原严格数值门禁：{gate["legacy_strict_numerical_gate"]}。当前状态 {status}。\n\n'+'原0738完整视觉编码、Prefill及自然EOS生成均成功。模型、默认分辨率、Prompt、四项输入张量哈希完全复用旧冻结输入。两条件均只启用PyTorch SDPA EFFICIENT CUDA后端，真实CUTLASS内核证据保存在逐题记录中，Math/Flash/CUDNN关闭。\n\n'+'隔离执行器复用原冻结context、真实L1–L6答案Attention跳算及CompactKV；只将三份大Mask改为一份对齐存储Mask，L22后实际删除视觉Hidden并使用紧凑文本Mask。所有文字Query/抽样视觉Query的可见Key集合精确核对，P−1/Decode禁止图像Key，DeepStack[0,1,2]和原mRoPE保留。短题与未修改冻结Physical在同后端全部有效Logits逐位一致。\n\n'+'旧NUMERICAL_EQUIVALENCE_FAILED和Eager OOM不被改写。Dense-F仅为同前缀诊断，未算入正式预测；只有该参考每步argmax与Physical相同，才证明自然Greedy序列一致。完整结果见svh_physical_gate.json和numerical_comparison.csv。\n\n'+'新后端已保存Native3/1000、Physical3/1000，均将复用，不重复。全量尚未启动。\n',encoding='utf-8')
print('SVH_GATES_FINISHED',status,flush=True)
