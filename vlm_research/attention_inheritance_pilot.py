"""Only case15133706: inherit Full L23/L24 visual shape at L24..31, P-1."""
import argparse
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import traceback
import numpy as np
import torch
import transformers
from transformers.models.llama import modeling_llama
from vlm_common import ROOT, load_model, write_json
from knockout_probe import resource_preflight, model_layers, write_csv
from roi_sink_probe import inputs_and_history, score
from roi_sink_common import forward
from attention_inheritance import VisualAttentionInheritance, TOLERANCES

OUTPUT=ROOT/'results/attention_inheritance_15133706'
HISTORY=ROOT/'results/phase3_2_fixed_k_exit'
CACHE=ROOT/'results/phase3_4_roi_sink_diagnostic/attention_rows/15133706_task.npz'
QID='15133706'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def native_signature():
    return dict(library_file=str(Path(modeling_llama.__file__).resolve()),
                library_file_sha256=sha(modeling_llama.__file__),
                forward_source_sha256=hashlib.sha256(inspect.getsource(modeling_llama.LlamaAttention.forward).encode()).hexdigest(),
                eager_source_sha256=hashlib.sha256(inspect.getsource(modeling_llama.eager_attention_forward).encode()).hexdigest())

def plot(output,rows,full,physical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5.5))
    for source,color in [(23,'#2266aa'),(24,'#d75e17')]:
        rr=[r for r in rows if r['source_layer']==source]
        ax.plot([r['lambda'] for r in rr],[r['margin'] for r in rr],'-o',label=f'Inherit Full L{source}',color=color)
        for r in rr:ax.annotate(f"{r['margin']:+.5f}",(r['lambda'],r['margin']),xytext=(5,8 if source==23 else -16),textcoords='offset points',fontsize=9)
    ax.axhline(full,color='#555555',ls='--',label=f'Native Full {full:+.5f}')
    ax.axhline(physical,color='#38854b',ls=':',label=f'Historical Physical K24 {physical:+.5f}')
    ax.axhline(0,color='black',lw=.8,label='Decision boundary')
    ax.set(xlabel='Inheritance strength lambda',ylabel='logP(right) - logP(left) [nats]',xticks=[0,.5,1])
    ax.set_title('15133706: Girl / Boy; correct answer = right\nL24-L31: only P-1 -> Image conditional attention is mixed',fontsize=12)
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.text(.5,.015,'Current per-head visual mass preserved within FP16 tolerance; no token deletion. Single-case diagnostic.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(output/'margin_comparison.png',dpi=180);plt.close(fig)

def report(output,rows,full,physical,validation):
    table='\n'.join(f"| L{r['source_layer']} | {r['lambda']:g} | {r['margin']:+.8f} | {r['gain_vs_full']:+.8f} | {r['predicted_candidate']} | {r['forward_seconds_two_candidates']:.3f} |" for r in rows)
    v=validation
    text=f'''# Single-case visual attention inheritance: 15133706

Question: **Is the young girl to the left or to the right of the boy?** Correct candidate: **right**; unchanged order: left, right. This case was selected by the user because of its earlier Physical K24 flip; no population inference is possible.

## Fixed intervention

Only L24-L31 and query **P-1=602** to all **576** image keys (positions5..580). Full prompt length603; original matched teacher-forcing input length604. No cache, generation, physical deletion, training or new images. FP16/eager LLaVA-1.5-7B and Transformers4.49.0 unchanged.

For each head h, let a be the **current intervened forward's** pre-edit visual row, m=sum(a), and p_ref the frozen Full source layer's row divided by its own sum. Replace only those cells by `m * ((1-lambda) * a/m + lambda * p_ref)` after FP32 softmax has been cast to native FP16 and before multiplication by current-layer V. lambda0 bypasses assignment. Full L23 and L24 references are each reused across all eight target layers, head h to head h; no claim that different depths' heads have identical semantic roles. Values, weights, nonvisual cells and all other queries receive no direct edits. Later layers' pre-edit attentions may naturally change through the intervened hidden state; local non-target invariance does not imply equality with Full.

Reference rows freshly captured from this same Full pass match the Phase3.4 cache bitwise; each head is normalized independently in FP32. Zero/invalid reference masses cause failure rather than substitution. No global monkeypatch or third-party file modification: only selected model-instance forward functions have private globals, restored in finally.

## Actual results

Native Full margin **{full:+.8f}**, historical Physical K24 **{physical:+.8f}** (historical only; Physical was not rerun). Score is complete single-token candidate log probability with double log-softmax over the full vocabulary. Both teacher-forcing candidates are run separately and their P-1 logits must be bitwise identical. Raw candidate logits and log probabilities are retained.

| Frozen source | lambda | Answer margin (nats) | Gain vs Full | Prediction | Two-candidate diagnostic seconds |
| --- | --- | --- | --- | --- | --- |
{table}

Positive margin means right beats left **among these two candidates**; it does not imply high unconditional right probability. The original Full and Physical margins are close to zero. FP16 backbone / output logits remain in use; double log-softmax does not undo FP16 quantization. No full-FP32 backbone validation or independent statistical replication was performed.

![Margin comparison](margin_comparison.png)

## Correctness / numerical checks

- CPU tests passed before GPU execution; see `cpu_tests.log`.
- Exact historical prompt/candidate/question mappings; P-1 is disjoint from real Question queries. Native Full log probabilities match historical Full exactly. Fresh source rows match cached rows exactly.
- lambda0 for both sources: entire P-1 vocabulary logits and final Question hidden states exactly equal native Full, including both candidates.
- All targeted local audits passed; 8 layers x2 candidates x6 conditions. Other queries and nonvisual cells are bitwise unchanged by replacement, finite and nonnegative. Final Question hidden states remain bitwise Full for every condition.
- Predeclared row sum tolerance0.002; observed max error **{v['max_row_sum_abs_error']:.9g}**. Per-head visual mass tolerance atol2e-5+rtol0.001; observed max absolute error **{v['max_visual_mass_abs_error']:.9g}**, max relative **{v['max_visual_mass_relative_error']:.9g}**. Conditional mixing observed max L1 **{v['max_conditional_l1_error']:.9g}**. Every patched value equals the formula cast to FP16 bitwise, and conditional L1 passes a dtype/mass-derived round-to-nearest bound (details in layer audits). **{v['nominal_conditional_threshold_exceeded_audits']}** layer audits exceed the initial nominal0.002 threshold; this is retained explicitly. No re-normalization of the full row, because it would alter nonvisual weights.
- Instance overrides and hooks restored; post-experiment native Full exact; installed Transformers library SHA256 unchanged. No NaN/Inf or CUDA OOM.
- Peak allocated **{v['peak_allocated_mib']:.3f} MiB**, peak reserved **{v['peak_reserved_mib']:.3f} MiB** across recorded forwards. GPU is shared; times include validation overhead, are diagnostic and not speedup measurements. Model load recorded separately in environment.json.

## Initial validation stop and arithmetic-only correction

`initial_attempt_fp16_validation/` preserves the first attempt. Both lambda0 source-L23 candidate passes were exact. Source-L23 lambda0.5 stopped in L25 before obtaining a candidate score: conditional L1=0.00429884 exceeded the initial fixed0.002 threshold in a head with visual mass about0.00210; visual mass/non-target invariants passed. This was an insufficient validator for FP16 subnormal rounding, not a changed intervention or favorable outcome selection.

Before rerunning, the formula and FP16 computation were left identical. Validation was strengthened to require per-cell bitwise equality with the exact FP32 formula cast to FP16, and a derived rounding bound: `u=eps/2; d=tiny*eps; E=u*sum(abs(x))+576*d/2; conditional_L1 <= 2*E/sum(round(x))+2e-6`. This follows the triangle inequality and includes subnormal rounding; it is calculated from dtype and row mass, not fitted to measured margin or error. The original nominal threshold, failures and hashes remain recorded. Row sum and visual mass tolerances were not changed. No failed partial run is counted as a completed condition.

## Observation versus interpretation

'''
    for source in [23,24]:
        rr=[r for r in rows if r['source_layer']==source and r['lambda']>0]
        text+=f"- L{source}: "+'; '.join(f"lambda={r['lambda']:g}: gain{r['gain_vs_full']:+.8f}, margin{r['margin']:+.8f} ({'right' if r['margin']>0 else 'left' if r['margin']<0 else 'tie'})" for r in rr)+'.\n'
    text+='''
This is an intervention on attention shape, not just an attention correlation. Any observed margin change is the effect of this precisely defined local replacement under this one FP16 input. It does **not** establish that inherited attention is semantically better, that a sink is harmful, or that it explains Physical K24: physical deletion affects all text queries, hidden states, sequence computation and many subsequent connections, while this intervention changes only one row's visual shape and preserves its mass. A sign flip in this weak-margin selected case is insufficient evidence of a general capability improvement. No outcomes were used to choose new strengths, layers, sources or samples.

## Reproduction and provenance

```bash
conda activate vlm_research
cd /home/zsy/ATE/vlm_research
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest test_attention_inheritance -v
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python attention_inheritance_pilot.py
```

The runner refuses to overwrite an existing configured output directory. For intentional reruns specify a new `--output-dir`. Resource checks run before model load; failures are saved, other GPU processes are never terminated. See `experiment_config.json`, `environment.json`, `gpu_resource.json`, `validation.json`, `per_condition_results.csv/json`, `reference_distributions.npz`, per-condition audit JSON and `prediction_logits.npz`. All old experiments remain untouched. Stop here; no expansion.
'''
    (output/'report.md').write_text(text)
    (output/'README.md').write_text('# 15133706 单案例注意力继承\n\n[完整报告](report.md) · [逐条件结果](per_condition_results.csv) · [正确性验证](validation.json)\n\n![Margin](margin_comparison.png)\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=OUTPUT);args=p.parse_args();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'experiment_config.json').exists():raise ValueError('Refuse to overwrite configured run; select a new output directory')
    if transformers.__version__!='4.49.0':raise ValueError('Only pinned Transformers4.49.0 is supported')
    manifest_path=HISTORY/'manifest.json';manifest=json.loads(manifest_path.read_text());sample=next(s for s in manifest['samples'] if s['question_id']==QID)
    if sha(sample['image_path'])!=sample['image_sha256']:raise ValueError('Image hash changed')
    lib=native_signature()
    config=dict(task='single_case_attention_inheritance',started=datetime.now().astimezone().isoformat(),question_id=QID,sample=sample,
                source_layers=[23,24],target_layers=list(range(24,32)),lambdas=[0,.5,1],query='P-1 only',
                tolerance_preregistered=TOLERANCES,reference='Frozen native Full per-head visual-conditional distributions, verified against Phase3.4 cache',
                formula='m*((1-lambda)*a/m + lambda*p_full_reference)',historical_manifest_sha256=sha(manifest_path),
                cache_sha256=sha(CACHE),library=lib,code_sha256={f:sha(ROOT/f) for f in ['attention_inheritance.py','attention_inheritance_pilot.py','test_attention_inheritance.py','vlm_common.py','roi_sink_common.py','roi_sink_probe.py','physical_exit_pilot.py']},
                previous_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(output/'experiment_config.json',config)
    cpu=subprocess.run(['/home/zsy/.conda/envs/vlm_research/bin/python','-m','unittest','test_attention_inheritance','-v'],cwd=ROOT,capture_output=True,text=True,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':''})
    (output/'cpu_tests.log').write_text(cpu.stdout+cpu.stderr)
    if cpu.returncode:raise RuntimeError('CPU tests failed')
    resource=resource_preflight();write_json(output/'gpu_resource.json',resource)
    if not resource['ready']:raise RuntimeError(resource.get('reason','GPU not ready'))
    model,processor,env=load_model();write_json(output/'environment.json',env)
    prompt,candidates,qpos,apos,history,qrecords,counts=inputs_and_history(model,processor,sample)
    visual=torch.where(prompt['input_ids'][0]==model.config.image_token_index)[0]
    if len(visual)!=576 or len(model_layers(model))!=32 or apos!=602 or apos in qpos:raise AssertionError('Unexpected case structure')
    write_json(output/'input_mapping.json',dict(question_id=QID,counts=counts,question_mapping=qrecords,answer_query=apos,visual_indices=visual.cpu().tolist(),candidate_mapping=history['candidate_mapping']))
    full=[forward(model,x,qpos,apos,observe=(i==0)) for i,(x,rec) in enumerate(candidates)]
    logs=[score(o,rec) for o,(x,rec) in zip(full,candidates)]
    if logs!=[c['logp'] for c in history['full_candidates']]:raise AssertionError('Historical Full mismatch')
    if not torch.equal(full[0]['prediction_logits'],full[1]['prediction_logits']):raise AssertionError('Future candidate leakage')
    with np.load(CACHE) as cached:
        if not np.array_equal(full[0]['rows'],cached['attention']):raise AssertionError('Fresh Full rows differ from Phase3.4 cache')
    target=sample['candidates'].index(sample['answer']);wrong=1-target;full_margin=logs[target]-logs[wrong]
    import csv
    old=next(r for r in csv.DictReader((HISTORY/'per_sample_results.csv').open()) if r['question_id']==QID and r['cutoff']=='24')
    physical=float(old['answer_margin'])
    refs={layer:full[0]['rows'][layer,:,1,:].astype(np.float32) for layer in [23,24]}
    np.savez_compressed(output/'reference_distributions.npz',**{f'layer{layer}_raw':ref for layer,ref in refs.items()},**{f'layer{layer}_conditional':ref/ref.sum(-1,keepdims=True) for layer,ref in refs.items()})
    modules=[l.self_attn for l in model_layers(model)];rows=[];audits=[];all_measurements=[o['measurement'] for o in full];all_logits={'native_full':np.stack([o['prediction_logits'].numpy() for o in full])}
    for source in [23,24]:
        for strength in [0,.5,1]:
            outs=[];conditions=[]
            for i,(x,rec) in enumerate(candidates):
                with VisualAttentionInheritance(modules,visual.cpu(),apos,refs[source],strength) as intervention:
                    value=forward(model,x,qpos,apos)
                    intervention.assert_complete()
                if any('forward' in m.__dict__ for m in modules):raise AssertionError('Override leaked')
                if not torch.equal(value['question_hidden'],full[i]['question_hidden']):raise AssertionError('Earlier Question states changed')
                if strength==0 and not torch.equal(value['prediction_logits'],full[i]['prediction_logits']):raise AssertionError('lambda0 differs from Full')
                conditions.append(dict(candidate=sample['candidates'][i],layer_audits=intervention.records,measurement=value['measurement'],question_hidden_bitwise_full=True,lambda_zero_logits_bitwise_full=True if strength==0 else None))
                outs.append(value);audits.extend(intervention.records);all_measurements.append(value['measurement'])
            if not torch.equal(outs[0]['prediction_logits'],outs[1]['prediction_logits']):raise AssertionError('Candidate leaked into prediction')
            logp=[score(o,rec) for o,(x,rec) in zip(outs,candidates)];margin=logp[target]-logp[wrong]
            token_ids=[rec[0]['token_id'] for x,rec in candidates]
            row=dict(question_id=QID,source_layer=source,**{'lambda':strength},target_first_layer=24,target_last_layer=31,answer_query=apos,
                     margin=margin,gain_vs_full=margin-full_margin,full_margin=full_margin,historical_physical_k24_margin=physical,
                     correct_candidate=sample['answer'],predicted_candidate=sample['answer'] if margin>0 else sample['candidates'][wrong] if margin<0 else 'tie',
                     correct_logp=logp[target],wrong_logp=logp[wrong],correct_logit=float(outs[0]['prediction_logits'][token_ids[target]]),wrong_logit=float(outs[0]['prediction_logits'][token_ids[wrong]]),
                     correct_candidate_probability=float(np.exp(logp[target])),right_vs_left_probability=float(1/(1+np.exp(-margin))),
                     forward_seconds_two_candidates=sum(o['measurement']['forward_seconds'] for o in outs),
                     peak_allocated_mib=max(o['measurement']['gpu_memory']['peak_allocated_mib'] for o in outs),
                     peak_reserved_mib=max(o['measurement']['gpu_memory']['peak_reserved_mib'] for o in outs),
                     all_local_checks_passed=True,question_hidden_bitwise_full=True,candidate_logits_bitwise_identical=True)
            rows.append(row);write_json(output/f'audit_L{source}_lambda{strength:g}.json',conditions)
            write_json(output/'per_condition_results.json',rows);write_csv(output/'per_condition_results.csv',rows)
            all_logits[f'L{source}_lambda{strength:g}']=np.stack([o['prediction_logits'].numpy() for o in outs])
            np.savez_compressed(output/'prediction_logits.npz',**all_logits)
            print(json.dumps(row),flush=True)
    after=forward(model,candidates[0][0],qpos,apos);all_measurements.append(after['measurement'])
    if not torch.equal(after['prediction_logits'],full[0]['prediction_logits']) or not torch.equal(after['question_hidden'],full[0]['question_hidden']):raise AssertionError('Post-cleanup native Full changed')
    if native_signature()!=lib:raise AssertionError('Installed library changed')
    validation=dict(status='completed',passed=True,only_question_id=QID,condition_count=len(rows),target_layer_audit_count=len(audits),
                    historical_full_exact=True,source_cache_bitwise_exact=True,lambda_zero_both_sources_exact=True,question_hidden_all_exact=True,
                    post_cleanup_native_exact=True,third_party_source_unchanged=True,other_queries_and_nonvisual_bitwise_exact=all(a['other_queries_bitwise_exact'] and a['nonvisual_bitwise_exact'] for a in audits),
                    max_row_sum_abs_error=max(a['row_sum_max_abs_error'] for a in audits),max_visual_mass_abs_error=max(a['visual_mass_max_abs_error'] for a in audits),
                    max_visual_mass_relative_error=max(a['visual_mass_max_relative_error'] for a in audits),max_conditional_l1_error=max(a['conditional_l1_max_error'] for a in audits),
                    nominal_conditional_threshold_exceeded_audits=sum(a['conditional_nominal_exceeded'] for a in audits),
                    per_cell_formula_rounding_exact=all(a['formula_rounded_target_bitwise_exact'] for a in audits),
                    dtype_derived_conditional_bound_passed=all(a['conditional_rounding_bound_passed'] for a in audits),
                    peak_allocated_mib=max(m['gpu_memory']['peak_allocated_mib'] for m in all_measurements),peak_reserved_mib=max(m['gpu_memory']['peak_reserved_mib'] for m in all_measurements),
                    measured_forwards=len(all_measurements),diagnostic_forward_seconds_sum=sum(m['forward_seconds'] for m in all_measurements),measurements=all_measurements)
    write_json(output/'validation.json',validation);plot(output,rows,full_margin,physical);report(output,rows,full_margin,physical,validation)
    print('COMPLETE',output,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        # Save actual failure; never label a partial run completed.
        parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=OUTPUT);a,_=parser.parse_known_args()
        a.output_dir.mkdir(parents=True,exist_ok=True)
        if not (a.output_dir/'validation.json').exists():write_json(a.output_dir/'failure.json',dict(status='incomplete',error=repr(exc),traceback=traceback.format_exc()))
        raise
