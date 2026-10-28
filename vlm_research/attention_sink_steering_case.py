"""Frozen 15133706-only offline analysis and positive post-softmax steering.

Stages: prepare (CPU/cache only), gpu (resource-gated), summarize (CPU only).
Every attempted GPU run has a separate directory. No automatic resource polling.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch
from PIL import Image

from attention_sink_steering import conditional, spatial_similarity, factors_for, VisualAttentionReweight
from attention_inheritance import TOLERANCES
from attention_inheritance_pilot import native_signature
from knockout_probe import model_layers, resource_preflight, write_csv
from roi_sink_common import geometry, map_box, forward
from roi_sink_probe import inputs_and_history, score
from vlm_common import ROOT, default_model, load_model, write_json

OUTPUT = ROOT/'results/attention_sink_steering_case_15133706'
P34 = ROOT/'results/phase3_4_roi_sink_diagnostic'
P32 = ROOT/'results/phase3_2_fixed_k_exit'
QID = '15133706'
SOURCES = ['attention_sink_steering.py', 'attention_sink_steering_case.py',
           'test_attention_sink_steering.py', 'attention_inheritance.py',
           'vlm_common.py', 'roi_sink_common.py', 'roi_sink_probe.py',
           'attention_inheritance_pilot.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def make_conditions(entry):
    """31 predeclared conditions; controls are frozen Phase3.4 spatial matches."""
    rows = []
    for alpha in [0., .5, 1.]:
        rows.append(dict(id=f'A_debias_a{alpha:g}', family='A', target='all_visual',
                         active_layers=list(range(14,24)), strength=alpha, epsilon=1e-6,
                         patch_indices=list(range(576)), control_seed=None))
    for family, window in [('B', list(range(14,24))), ('C24',list(range(24,32))),
                           ('C27',list(range(27,32))), ('D',list(range(2,5)))]:
        region = 'roi' if family == 'B' else 'sink'
        targets = [(region, entry['roi_indices'] if region=='roi' else entry['sink_nonroi_indices'], None)]
        for seed in [3401,3402,3403]:
            control = entry['random_controls'][f'random_{region}_s{seed}']
            if not control['available']:
                raise ValueError('Required frozen count-matched control unavailable')
            targets.append((f'random_{region}',control['indices'],seed))
        for strength in ([2.] if family=='B' else [.5,.1]):
            for target, indices, seed in targets:
                suffix = '' if seed is None else f'_s{seed}'
                rows.append(dict(id=f'{family}_{target}_f{strength:g}{suffix}',family=family,
                                 target=target,active_layers=window,strength=strength,
                                 patch_indices=indices,control_seed=seed))
    assert len(rows)==31 and len({r['id'] for r in rows})==31
    return rows


def plt_module():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def mark(ax, sample, entry, scale=14):
    from matplotlib.patches import Rectangle
    for obj,color in zip(sample['objects'],['cyan','magenta']):
        x0,y0,x1,y1 = np.array(obj['crop_xyxy'])/scale
        ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor=color,lw=1.3))
    for key,color in [('sink_nonroi_indices','lime'),('roi_sink_overlap_indices','orange')]:
        for i in entry[key]:
            ax.add_patch(Rectangle((i%24-.5,i//24-.5),1,1,fill=False,edgecolor=color,lw=.7))
    ax.set_xticks([]);ax.set_yticks([])


def offline_plots(output, raw, b, sample, entry):
    from matplotlib.colors import LogNorm
    plt=plt_module()
    p=conditional(raw)
    maps=np.stack([b.mean(0),p[14:24].mean((0,1)),p[27:32].mean((0,1))])  # window,group,patch
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    norm=LogNorm(vmin=1e-5,vmax=max(float(maps.max()),1e-4))
    for g,name in enumerate(['Question','Answer P-1']):
        for j,title in enumerate(['B: L2-L4','L14-L23','L27-L31']):
            ax=axes[g,j]
            im=ax.imshow(maps[j,g].reshape(24,24),norm=norm,cmap='inferno',interpolation='nearest')
            layers=[range(2,5),range(14,24),range(27,32)][j]
            mass=raw[list(layers),:,g].sum(-1).mean()
            ax.set_title(f'{name}: {title}\nmean raw visual mass={mass:.4f}')
            mark(ax,sample,entry)
        ax=axes[g,3];ax.imshow(plt.imread(output/'processor_crop.png'),extent=(-.5,23.5,23.5,-.5))
        mark(ax,sample,entry);ax.set_title('True Processor crop / frozen ROIs')
    fig.colorbar(im,ax=list(axes[:,:3].ravel()),fraction=.018,pad=.015,label='Per-head visual-conditional p, then mean heads/layers')
    fig.suptitle('15133706 Girl/Boy: cyan/magenta=object boxes; lime=non-ROI Sink; orange=ROI/Sink overlap')
    fig.subplots_adjust(left=.02,right=.93,top=.86,bottom=.1,wspace=.25,hspace=.3)
    fig.savefig(output/'offline_spatial_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for g,name in enumerate(['Question','Answer P-1']):
        cos,js=spatial_similarity(p[:,:,g],b[:,g])
        for j,(values,label) in enumerate([(cos,'Cosine similarity to B'),(js,'Jensen-Shannon distance to B')]):
            ax=axes[g,j];ax.plot(range(32),values.mean(1))
            ax.fill_between(range(32),np.quantile(values,.1,axis=1),np.quantile(values,.9,axis=1),alpha=.2,label='10-90% of heads (not CI)')
            ax.axvspan(14,23,color='orange',alpha=.1);ax.axvspan(27,31,color='green',alpha=.1)
            ax.set(title=f'{name}: {label}',ylabel=label,xlabel='Layer index');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'offline_similarity_by_layer.png',dpi=160);plt.close(fig)


def prepare(output):
    if (output/'protocol.json').exists():
        raise ValueError('Prepared protocol already exists; do not overwrite')
    output.mkdir(parents=True,exist_ok=True)
    manifest=read(P34/'manifest.json');sink=read(P34/'sink_manifest.json')
    sample=next(s for s in manifest['samples'] if s['question_id']==QID)
    entry=next(s for s in sink['samples'] if s['question_id']==QID)
    cached=P34/'attention_rows'/f'{QID}_task.npz'
    inventory=read(P34/'artifact_sha256.json')['files']
    for file in [P34/'manifest.json',P34/'sink_manifest.json',cached]:
        assert sha(file)==inventory[str(file.relative_to(P34))]['sha256']
    assert sha(sample['image_path'])==sample['image_sha256']
    assert entry['roi_indices']==sample['roi_indices']
    roi,sinks=set(entry['roi_indices']),set(entry['sink_indices'])
    assert set(entry['sink_nonroi_indices'])==sinks-roi
    assert set(entry['roi_sink_overlap_indices'])==sinks&roi
    for name,control in entry['random_controls'].items():
        target=roi if name.startswith('random_roi') else sinks-roi
        assert control['available'] and len(control['indices'])==len(target)
        assert not set(control['indices'])&(roi|sinks)
    # Actual local AutoProcessor, no model or GPU forward.
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(default_model(),local_files_only=True)
    with Image.open(sample['image_path']) as image:
        geom,crop=geometry(image,processor.image_processor)
    assert geom==sample['geometry']
    for obj in sample['objects']:
        assert map_box(obj['original_xywh'],geom)['patch_indices']==obj['patch_indices']
    conditions=make_conditions(entry)
    protocol=dict(question_id=QID,created=datetime.now().astimezone().isoformat(),sample=sample,sink_entry=entry,
                  conditions=conditions,condition_count=31,query='P-1 only; expected index602, verified from frozen token mapping',
                  reference='B[h,group,v]=mean L2..L4 of per-head visual-conditional Full attention; Q/A separate. A uses Answer B.',
                  formula='m_current * normalize(p_current * positive_factor), after FP32 softmax cast to native FP16, before P@V',
                  positive_bias='B: beta=ln(2), equivalent to multiplying ROI weights by2 before visual-only normalization',
                  strengths=dict(alpha=[0,.5,1],epsilon=1e-6,roi_factor=2.,sink_factors=[.5,.1]),
                  controls='Exact frozen Phase3.4 random_roi/random_sink, seed labels3401/3402/3403. Same masks reused across windows/strengths; no outcome-based selection.',
                  early_control='D uses identical13 non-ROI Sink keys and both factors; also all three matched-background controls.',
                  similarity='Cosine similarity; JS distance=sqrt(JSD in bits), per matched head plus head-mean map. Stable top58 ties by row-major index.',
                  low_mass_threshold=1e-3,tolerances=TOLERANCES,
                  planned_gpu_smoke='Uniform factor1 at all four layer windows exact native; no-op and nonzero formula audits before scoring conditions.',
                  source_sha256={str(f.relative_to(ROOT)):sha(f) for f in [P34/'manifest.json',P34/'sink_manifest.json',cached,P32/'manifest.json',P32/'per_sample_results.csv']},
                  code_sha256={f:sha(ROOT/f) for f in SOURCES},library=native_signature(),
                  previous_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  interpretation='One post-selected weak-margin FP16 case. Spatial similarity is descriptive, not sink causality. No weight edits, training, physical deletion, decode, speed claim or parameter search.')
    write_json(output/'protocol.json',protocol)  # frozen before computing new similarities or GPU outcomes
    Image.fromarray(crop).save(output/'processor_crop.png')
    raw=np.load(cached)['attention'].astype(np.float64)
    if raw.shape!=(32,32,2,576):raise AssertionError('Unexpected cached attention axes')
    p=conditional(raw);b=p[2:5].mean(0)
    np.savez_compressed(output/'early_reference.npz',B=b,raw_visual_mass=raw.sum(-1))
    headrows=[];summary=[]
    for g,name in enumerate(['question','answer']):
        for label,layers in [('L2-L4',list(range(2,5))),('L14-L23',list(range(14,24))),('L27-L31',list(range(27,32)))]+[(f'L{l}',[l]) for l in range(32)]:
            target=p[layers,:,g].mean(0)
            c,j=spatial_similarity(target,b[:,g]);mc,mj=spatial_similarity(target.mean(0),b[:,g].mean(0))
            top0=np.argsort(-b[:,g],axis=-1,kind='stable')[:,:58]
            top1=np.argsort(-target,axis=-1,kind='stable')[:,:58]
            overlaps=[len(set(x)&set(y)) for x,y in zip(top0,top1)]
            mean_top0=set(np.argsort(-b[:,g].mean(0),kind='stable')[:58])
            mean_top1=set(np.argsort(-target.mean(0),kind='stable')[:58])
            summary.append(dict(query=name,window=label,head_mean_cosine=float(c.mean()),head_median_cosine=float(np.median(c)),
                                head_mean_js_distance=float(j.mean()),map_cosine=float(mc),map_js_distance=float(mj),
                                mean_top58_overlap=float(np.mean(overlaps)),head_mean_map_top58_overlap=len(mean_top0&mean_top1),
                                early_top58_nonroi_sink_count=len(mean_top0&set(entry['sink_nonroi_indices'])),
                                target_top58_nonroi_sink_count=len(mean_top1&set(entry['sink_nonroi_indices'])),
                                shared_top58_nonroi_sink_count=len(mean_top0&mean_top1&set(entry['sink_nonroi_indices'])),
                                shared_top58_roi_count=len(mean_top0&mean_top1&set(entry['roi_indices'])),
                                low_visual_mass_head_layer_count=int((raw[layers,:,g].sum(-1)<1e-3).sum()),
                                roi_share=float(target[:,entry['roi_indices']].sum(-1).mean()),
                                nonroi_sink_share=float(target[:,entry['sink_nonroi_indices']].sum(-1).mean()),
                                roi_sink_overlap_share=float(target[:,entry['roi_sink_overlap_indices']].sum(-1).mean())))
            for h in range(32):
                headrows.append(dict(query=name,window=label,head=h,cosine=float(c[h]),js_distance=float(j[h]),top58_overlap=overlaps[h]))
    write_csv(output/'offline_similarity_summary.csv',summary);write_json(output/'offline_similarity_summary.json',summary)
    write_csv(output/'offline_similarity_per_head.csv',headrows)
    write_json(output/'roi_sink_positions.json',dict(roi_indices=entry['roi_indices'],sink_indices=entry['sink_indices'],
               sink_nonroi_indices=entry['sink_nonroi_indices'],overlap_indices=entry['roi_sink_overlap_indices'],
               roi_count=len(roi),sink_count=len(sinks),nonroi_sink_count=len(sinks-roi),overlap_count=len(sinks&roi),
               processor_geometry=geom,processor_pixels_exact=True))
    offline_plots(output,raw,b,sample,entry)
    write_json(output/'status.json',dict(status='offline_complete_gpu_pending',gpu_completed=False,completed_conditions=0,planned_conditions=31))
    summarize(output)


def validate_binding(output):
    p=read(output/'protocol.json')
    for name,value in {**p['source_sha256'],**p['code_sha256']}.items():
        if sha(ROOT/name)!=value:raise ValueError(f'Frozen source/config changed: {name}')
    if sha(p['sample']['image_path'])!=p['sample']['image_sha256']:raise ValueError('Image changed')
    if native_signature()!=p['library']:raise ValueError('Installed library changed')
    return p


def regional_rows(pairs,active,entry,condition):
    rows=[]
    for l,pair in zip(active,pairs):
        for stage,raw in zip(['before','after'],pair):
            p=conditional(raw);mass=raw.sum(-1)
            for h in range(len(raw)):
                row=dict(condition=condition,layer=l,head=h,stage=stage,visual_mass=float(mass[h]),
                         normalized_entropy=float(-(p[h]*np.log(np.maximum(p[h],1e-300))).sum()/np.log(576)))
                for name,indices in [('roi',entry['roi_indices']),('sink_nonroi',entry['sink_nonroi_indices']),('overlap',entry['roi_sink_overlap_indices'])]:
                    row[name+'_share']=float(p[h,indices].sum());row[name+'_mass']=float(raw[h,indices].sum())
                rows.append(row)
    return rows


def condition_plot(path,pairs,active,sample,entry,title,full_rows):
    plt=plt_module()
    # Fixed positions in each active window: first, middle, last. No head/condition selection.
    indices=sorted({0,len(active)//2,len(active)-1})
    maps=[]
    for i in indices:
        maps.extend([conditional(full_rows[active[i],:,1]).mean(0),
                     conditional(pairs[i,0]).mean(0),conditional(pairs[i,1]).mean(0)])
    vmax=max(float(m.max()) for m in maps)
    fig,axes=plt.subplots(len(indices),3,figsize=(10,3*len(indices)),squeeze=False)
    for ri,i in enumerate(indices):
        for ci,(raw,label) in enumerate([(full_rows[active[i],:,1],'Native Full'),(pairs[i,0],'Local before edit'),(pairs[i,1],'After edit')]):
            ax=axes[ri,ci];im=ax.imshow(conditional(raw).mean(0).reshape(24,24),vmin=0,vmax=vmax,cmap='inferno')
            mark(ax,sample,entry);ax.set_title(f'L{active[i]} {label}\nmean raw visual mass={raw.sum(-1).mean():.5f}',fontsize=10)
    fig.suptitle(title+'\nMean per-head visual-conditional p; one shared linear scale within this figure',fontsize=10)
    fig.colorbar(im,ax=list(axes.ravel()),fraction=.02,pad=.015)
    fig.subplots_adjust(left=.02,right=.9,bottom=.03,top=.88,hspace=.3)
    fig.savefig(path,dpi=140);plt.close(fig)


def gpu(output):
    protocol=validate_binding(output)
    if read(output/'status.json').get('gpu_completed'):raise ValueError('Completed experiment is immutable')
    run=output/'gpu_attempts'/datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True,exist_ok=False)
    try:
        cpu=subprocess.run([sys.executable,'-m','unittest','test_attention_sink_steering','-v'],cwd=ROOT,capture_output=True,text=True,
                           env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        (run/'cpu_tests.txt').write_text(cpu.stdout+cpu.stderr)
        if cpu.returncode:raise RuntimeError('CPU tests failed; GPU not attempted')
        resource=resource_preflight();write_json(run/'gpu_resource.json',resource)
        if not resource['ready']:
            write_json(output/'status.json',dict(status='gpu_resource_blocked',gpu_completed=False,completed_conditions=0,planned_conditions=31,latest_attempt=str(run.relative_to(output)),reason=resource.get('reason')))
            summarize(output);print('GPU_BLOCKED',resource.get('reason'),flush=True);return
        model,processor,env=load_model();write_json(run/'environment.json',env)
        sample,entry=protocol['sample'],protocol['sink_entry']
        prompt,candidates,qpos,apos,history,qrecords,counts=inputs_and_history(model,processor,sample)
        visual=torch.where(prompt['input_ids'][0]==model.config.image_token_index)[0]
        if len(visual)!=576 or apos!=602 or apos in qpos or len(model_layers(model))!=32:raise AssertionError('Invalid mapping')
        write_json(run/'input_mapping.json',dict(question_mapping=qrecords,answer_query=apos,counts=counts,visual_indices=visual.cpu().tolist(),candidate_mapping=history['candidate_mapping']))
        full=[forward(model,x,qpos,apos,observe=(i==0)) for i,(x,rec) in enumerate(candidates)]
        logs=[score(o,rec) for o,(x,rec) in zip(full,candidates)]
        if logs!=[c['logp'] for c in history['full_candidates']]:raise AssertionError('Historical Full differs')
        if not torch.equal(full[0]['prediction_logits'],full[1]['prediction_logits']):raise AssertionError('Candidate leakage')
        cached=np.load(P34/'attention_rows'/f'{QID}_task.npz')['attention']
        if not np.array_equal(full[0]['rows'],cached):raise AssertionError('Native Full/cache mismatch')
        b=np.load(output/'early_reference.npz')['B'][:,1]
        if not np.allclose(b,conditional(cached.astype(np.float64))[2:5,:,1].mean(0),atol=0,rtol=0):raise AssertionError('Frozen B changed')
        modules=[l.self_attn for l in model_layers(model)]
        measurements=[o['measurement'] for o in full];smoke=[]
        # Validate no-op at all four windows before any nonzero condition scoring.
        for active in [list(range(2,5)),list(range(14,24)),list(range(24,32)),list(range(27,32))]:
            with VisualAttentionReweight(modules,visual.cpu(),apos,np.ones((32,576)),active) as edit:
                out=forward(model,candidates[0][0],qpos,apos);edit.assert_complete()
            if not torch.equal(out['prediction_logits'],full[0]['prediction_logits']) or not torch.equal(out['question_hidden'],full[0]['question_hidden']):raise AssertionError('No-op differs')
            smoke.append(dict(active_layers=active,logits_bitwise_full=True,question_hidden_bitwise_full=True,audits=edit.records))
            measurements.append(out['measurement'])
        write_json(run/'gpu_smoke.json',dict(passed=True,windows=smoke))
        rows=[];regional=[];all_logits={'full':np.stack([o['prediction_logits'].numpy() for o in full])};audits=[]
        write_json(run/'baseline.json',dict(question_id=QID,left_logp=logs[0],right_logp=logs[1],margin=logs[1]-logs[0],historical_physical_k24_margin=.0390625))
        for condition in protocol['conditions']:
            name=condition['id'];outs=[];local_audits=[];pairs=None
            for i,(x,record) in enumerate(candidates):
                with VisualAttentionReweight(modules,visual.cpu(),apos,factors_for(condition,b),condition['active_layers']) as edit:
                    out=forward(model,x,qpos,apos);edit.assert_complete()
                if any('forward' in m.__dict__ for m in modules):raise AssertionError('Wrapper leaked')
                if not torch.equal(out['question_hidden'],full[i]['question_hidden']):raise AssertionError('Earlier Question changed')
                if condition['family']=='A' and condition['strength']==0 and not torch.equal(out['prediction_logits'],full[i]['prediction_logits']):raise AssertionError('alpha0 differs')
                current=np.stack(edit.rows)
                if i==0:pairs=current
                elif not np.array_equal(pairs,current):raise AssertionError('Future candidate changes edited P-1 attention')
                local_audits.append(dict(candidate=sample['candidates'][i],layers=edit.records));audits.extend(edit.records)
                measurements.append(out['measurement']);outs.append(out)
            if not torch.equal(outs[0]['prediction_logits'],outs[1]['prediction_logits']):raise AssertionError('Candidate prediction mismatch')
            cp=[score(o,rec) for o,(x,rec) in zip(outs,candidates)];margin=cp[1]-cp[0]
            reg=regional_rows(pairs,condition['active_layers'],entry,name);regional.extend(reg)
            row=dict(question_id=QID,condition=name,family=condition['family'],target=condition['target'],strength=condition['strength'],
                     control_seed=condition['control_seed'],first_layer=condition['active_layers'][0],last_layer=condition['active_layers'][-1],patch_count=len(condition['patch_indices']),
                     left_logp=cp[0],right_logp=cp[1],right_probability=float(np.exp(cp[1])),
                     margin=margin,full_margin=logs[1]-logs[0],gain_vs_full=margin-(logs[1]-logs[0]),wrong_to_correct=margin>0,
                     predicted_candidate='right' if margin>0 else 'left' if margin<0 else 'tie',historical_physical_k24_margin=.0390625,
                     diagnostic_forward_seconds=sum(o['measurement']['forward_seconds'] for o in outs),
                     peak_allocated_mib=max(o['measurement']['gpu_memory']['peak_allocated_mib'] for o in outs),
                     peak_reserved_mib=max(o['measurement']['gpu_memory']['peak_reserved_mib'] for o in outs),all_checks_passed=True)
            for region in ['roi','sink_nonroi','overlap']:
                for stage in ['before','after']:
                    row[f'{region}_{stage}_share']=float(np.mean([r[f'{region}_share'] for r in reg if r['stage']==stage]))
            rows.append(row)
            write_json(run/f'audit_{name}.json',local_audits)
            np.savez_compressed(run/f'rows_{name}.npz',attention_before_after=pairs,layers=np.array(condition['active_layers']))
            condition_plot(run/f'heatmap_{name}.png',pairs,condition['active_layers'],sample,entry,f'{name}: Margin={margin:+.5f}; Full=-.03125',cached)
            all_logits[name]=np.stack([o['prediction_logits'].numpy() for o in outs])
            np.savez_compressed(run/'prediction_logits.npz',**all_logits)
            write_json(run/'per_condition_results.json',rows);write_csv(run/'per_condition_results.csv',rows)
            write_csv(run/'regional_attention_per_head.csv',regional)
            print(json.dumps(row),flush=True)
        after=forward(model,candidates[0][0],qpos,apos);measurements.append(after['measurement'])
        if not torch.equal(after['prediction_logits'],full[0]['prediction_logits']) or not torch.equal(after['question_hidden'],full[0]['question_hidden']):raise AssertionError('Cleanup changed native')
        if native_signature()!=protocol['library']:raise AssertionError('Installed library changed')
        validation=dict(status='completed',passed=True,only_question_id=QID,condition_count=len(rows),audits=len(audits),
                        lambda_zero_native_exact=True,all_noop_windows_exact=True,question_hidden_all_exact=True,
                        other_queries_and_nonvisual_bitwise_exact=all(r['other_queries_bitwise_exact'] and r['nonvisual_bitwise_exact'] for r in audits),
                        all_rounded_formulas_exact=all(r['formula_rounded_target_bitwise_exact'] for r in audits),
                        conditional_rounding_bound_passed=all(r['conditional_rounding_bound_passed'] for r in audits),
                        max_row_sum_error=max(r['row_sum_max_abs_error'] for r in audits),max_mass_abs_error=max(r['visual_mass_max_abs_error'] for r in audits),
                        max_mass_relative_error=max(r['visual_mass_max_relative_error'] for r in audits),
                        nominal_conditional_l1_exceeded=sum(r['conditional_nominal_exceeded'] for r in audits),
                        post_cleanup_native_exact=True,third_party_source_unchanged=True,measurements=measurements,
                        peak_allocated_mib=max(m['gpu_memory']['peak_allocated_mib'] for m in measurements),
                        peak_reserved_mib=max(m['gpu_memory']['peak_reserved_mib'] for m in measurements),
                        diagnostic_forward_seconds_sum=sum(m['forward_seconds'] for m in measurements))
        assert len(rows)==31
        write_json(run/'validation.json',validation)
        write_json(output/'status.json',dict(status='completed',gpu_completed=True,completed_conditions=31,planned_conditions=31,latest_attempt=str(run.relative_to(output))))
        summarize(output)
    except BaseException as exc:
        write_json(run/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        completed=len(read(run/'per_condition_results.json')) if (run/'per_condition_results.json').exists() else 0
        write_json(output/'status.json',dict(status='gpu_incomplete',gpu_completed=False,completed_conditions=completed,planned_conditions=31,latest_attempt=str(run.relative_to(output)),reason=repr(exc)))
        summarize(output)
        raise


def summarize(output):
    p=read(output/'protocol.json');state=read(output/'status.json');summ=read(output/'offline_similarity_summary.json')
    table='\n'.join(f"| {r['query']} | {r['window']} | {r['head_mean_cosine']:.4f} | {r['head_mean_js_distance']:.4f} | {r['map_cosine']:.4f} | {r['head_mean_map_top58_overlap']}/58 |" for r in summ if r['window'] in ['L14-L23','L27-L31'])
    entry=p['sink_entry'];text=f'''# 15133706 Attention Sink 去偏与 ROI 增强

问题：**{p['sample']['question']}**；正确答案 **right**，候选顺序保持left/right。只研究这一事后选出的近零Margin历史案例，不扩样本。

## 当前真实状态

**{state['status']}**；完整GPU条件 {state['completed_conditions']}/31。Full历史Margin=−0.03125，Physical K24历史Margin=+0.0390625；历史值不冒充本轮新GPU结果。{state.get('reason','')}

## 离线分层空间比较

只读Phase3.4原始FP16缓存[32层,32Head,2组,576Patch]。每层每Head先除以自身视觉总质量，B[h,g]=L2～L4条件分布均值；对L14～L23及L27～L31同样按层平均，Q和P−1分别计算。表中的Head指标按相同编号Head比较后平均，map指标为上述分布再平均Head。JS Distance=sqrt(JSD in bits)，范围[0,1]，0相同；Cosine为相似度，1相同。Top58稳定行优先打破并列，不预设重合。

| Query | 比较窗口 | 逐Head均值Cosine | 逐Head均值JS Distance | 平均图Cosine | 平均图Top58重合 |
| --- | --- | ---: | ---: | ---: | ---: |
{table}

ROI共{len(entry['roi_indices'])}Patch，冻结Sink共{len(entry['sink_indices'])}，非ROI Sink {len(entry['sink_nonroi_indices'])}，交集{entry['roi_sink_overlap_indices']}单独记录。标记：cyan/magenta为两对象框、lime为非ROI Sink、orange为ROI/Sink交集。实际本地Processor重建并逐像素校验336裁剪，原图500×375→448×336→中心裁剪横移56；逐Patch映射与历史精确相同。这里的早层B仅是该任务提示下的早层分布，不重新将B的所有热点认定为无关Sink。冻结Sink仍来自Phase3.4中性提示规则。

![空间比较](offline_spatial_comparison.png)

![逐层相似性](offline_similarity_by_layer.png)

## 预先固定的干预与对照

全部只在P−1=602→视觉位置5..580，逐Head做 `new_visual=m_current*normalize(p_current*factors)`，在FP32 Softmax转成原生FP16后、乘当前V前编辑；非视觉、其他Query和因果Mask不改。不做全行归一化；FP16不可避免的舍入用原有行和.002、Mass atol2e−5+rtol.001与dtype推导界验证，不改变容差追求结果。

- A L14～L23：factor=(B_answer+1e−6)^(-alpha)，alpha=0/.5/1；alpha0直接跳过写入。
- B L14～L23：ROI factor2，等价视觉相对log偏置ln2；三组等133Patch背景增强对照，沿用历史冻结集合。
- C L24～L31、L27～L31：非ROI Sink factor.5/.1，各配三组等13Patch背景抑制。
- D L2～L4：同一13个非ROI Sink、同两强度，同样配三背景集合。不将早层Sink视为可安全删除。
- 31条件在任何新分数之前固定；不追加成功配置。随机种子标签3401/3402/3403及具体索引复用旧Manifest；Sink对照6×6位置bin精确匹配100%，ROI对照约65.4%，残余位置混杂必须保留。

### 空间重合的实际范围

'''
    for r in summ:
        if r['window'] not in ['L14-L23','L27-L31']:continue
        text+=f"- {r['query']} {r['window']}：平均图Top58与早层重合{r['head_mean_map_top58_overlap']}/58，其中{r['shared_top58_nonroi_sink_count']}个是冻结非ROI Sink、{r['shared_top58_roi_count']}个在ROI内。逐Head Top58平均只重合{r['mean_top58_overlap']:.2f}/58；低视觉质量(<.001)的Head×Layer有{r['low_visual_mass_head_layer_count']}个，保留在主结果。不能用平均图相似代替每个Head都在同一位置聚焦。\n"
    text+='\n这些计数直接检验热点是否重合，不能仅凭相似图案命名Sink，也不能从单图的重合数推导跨样本稳定性或功能。\n\n'
    text+='## GPU结果与验证\n\n'
    if state.get('latest_attempt'):
        attempt=output/state['latest_attempt']
        text+=f"本次尝试资料：[{state['latest_attempt']}]({state['latest_attempt']}/)。\n\n"
        if (attempt/'per_condition_results.json').exists():
            rows=read(attempt/'per_condition_results.json')
            text+='| 条件 | right logP | left logP | Margin | Gain vs Full | right胜出 |\n| --- | ---: | ---: | ---: | ---: | --- |\n'
            text+='\n'.join(f"| {r['condition']} | {r['right_logp']:.8f} | {r['left_logp']:.8f} | {r['margin']:+.8f} | {r['gain_vs_full']:+.8f} | {r['wrong_to_correct']} |" for r in rows)+'\n\n'
            plt=plt_module();fig,ax=plt.subplots(figsize=(12,8))
            ax.barh(range(len(rows)),[r['margin'] for r in rows],color=['#34845a' if r['margin']>0 else '#597bb5' for r in rows])
            ax.set_yticks(range(len(rows)),[r['condition'] for r in rows],fontsize=8);ax.invert_yaxis()
            ax.axvline(0,color='black',lw=.8);ax.axvline(-.03125,color='grey',ls='--',label='Full historical/native check -.03125')
            ax.axvline(.0390625,color='orange',ls=':',label='Historical Physical K24 +.0390625')
            ax.set_xlabel('logP(right) - logP(left), nats');ax.legend(fontsize=8);fig.tight_layout()
            fig.savefig(output/'margin_comparison.png',dpi=170);plt.close(fig)
            text+='![Margin](margin_comparison.png)\n\n'
            if state['gpu_completed']:
                val=read(attempt/'validation.json')
                text+=f"31条件全部完成。四窗口no-op全部logits精确Full；alpha0两候选精确；新Full Attention精确缓存；每条件两候选P−1 logits与局部行一致；Question最终hidden全条件精确Full；所有局部非目标连接bitwise不变，收尾原生Full精确、库源码SHA未改、无NaN/Inf/OOM。峰值allocated{val['peak_allocated_mib']:.3f}MiB/reserved{val['peak_reserved_mib']:.3f}MiB；诊断Forward合计{val['diagnostic_forward_seconds_sum']:.4f}秒，含审计，无严格预热重复计时，不作加速主张。\n\n"
                for family in ['A','B','C24','C27','D']:
                    targets=[r for r in rows if r['family']==family and not r['target'].startswith('random')]
                    text+=f"- {family}实际目标："+'；'.join(f"strength={r['strength']:g}, Margin={r['margin']:+.8f}, Gain={r['gain_vs_full']:+.8f}" for r in targets)+'。\n'
                text+='\n每个条件的原始/局部编辑前/后热图、逐Head ROI/Sink质量、原始logits及审计均在尝试目录。局部before来自已受前层干预影响的当前Forward，不等于Native Full；两者明确分栏。各图内部三列共享线性色标，跨条件比较请读原始NPZ和Mass而非假设颜色范围相同。\n'
        else:
            text+='**本轮没有任何新GPU候选分数、干预热图或因果效应结果。** 不从缓存模拟分数，不将旧Full/Physical/继承实验当作本次成功。\n'
    else:text+='GPU阶段尚未执行，无新候选分数。\n'
    text+='''
## 解释边界与恢复

Cosine/JS与热点重合只描述空间相关；ROI是标注框不是分割，高注意力或早晚重合不证明有害Sink。Sink候选来自冻结中性提示，不能用本轮Margin重新定义。ROI增强使用真实对象框，属于oracle诊断，不是无需标注的方法。重分布保留视觉Mass但改变区域份额，也可能使其他视觉Patch得到补偿；这种干预与Physical删除全部视觉位置不同。正确候选绝对概率和二选一相对Margin须分开；单FP16弱Margin案例不能支持泛化、无损抑制或机制证明。没有FP32主干复验，不排除数值敏感性。

```bash
conda activate vlm_research
cd /home/zsy/ATE/vlm_research
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest test_attention_sink_steering -v
# protocol已冻结，不重复prepare；GPU空闲时手动恢复同31条件
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python attention_sink_steering_case.py --stage gpu
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python attention_sink_steering_case.py --stage summarize
```

失败不覆盖，下一次GPU尝试使用新的时间戳子目录；不混合不同尝试凑完成条件。代码/源数据/图片/库指纹变更时拒绝运行，须调查差异。没有自动轮询、不终止其他任务、不重装环境或下载模型。本轮只此案例，完成后停止。
'''
    (output/'report.md').write_text(text)
    (output/'README.md').write_text('# 15133706 Sink steering / ROI enhancement\n\n[当前真实状态与完整报告](report.md) · [冻结协议](protocol.json) · [离线相似性](offline_similarity_summary.csv)\n\n![离线图](offline_spatial_comparison.png)\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','gpu','summarize'],required=True)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args=parser.parse_args();output=args.output_dir.resolve()
    if args.stage=='prepare':prepare(output)
    elif args.stage=='gpu':gpu(output)
    else:validate_binding(output);summarize(output)


if __name__=='__main__':main()
