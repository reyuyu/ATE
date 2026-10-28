"""One descriptive paired pilot summary; no significance tests or condition search."""
import csv
import gzip
import json
from pathlib import Path
import textwrap

import numpy as np
from .protocol import OUTPUT,CONDITIONS,protocol,read,sha,atomic
from .run import checkpoint


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def descriptive(rows):
    x=np.array([r['gain'] for r in rows]);m=np.array([r['margin'] for r in rows])
    return dict(n=len(rows),accuracy=float(np.mean(m>0)),full_accuracy=float(np.mean([r['full_margin']>0 for r in rows])),
        mean_gain=float(x.mean()),median_gain=float(np.median(x)),min_gain=float(x.min()),max_gain=float(x.max()),
        positive=int(np.sum(x>0)),negative=int(np.sum(x<0)),zero=int(np.sum(x==0)),ties=int(np.sum(m==0)),
        harmed=sum(r['full_correct_to_wrong'] for r in rows),recovered=sum(r['full_wrong_to_correct'] for r in rows),
        mean_absolute_gold_logp_gain=float(np.mean([r['absolute_correct_logp_gain'] for r in rows])),
        worst_samples=sorted(rows,key=lambda r:(r['gain'],r['question_id']))[:5])


def verify_body(body,sample,root):
    audit_path=next(name for name in body['artifacts'] if name.endswith('audit.json.gz'))
    with gzip.open(root/audit_path,'rt') as f:audit=json.load(f)
    b=audit['b']
    if not b['noop_effective_text_bitwise_exact'] or not b['post_cleanup_native_bitwise_exact']:
        raise ValueError('No-op/cleanup acceptance failed')
    qpos=[r['sequence_index'] for r in sample['input_mapping']['question_mapping']]
    apos=sample['input_mapping']['answer_query'];visual=sample['input_mapping']['visual_indices']
    native_masks=set();count=0;local_count=0
    for condition in CONDITIONS:
        aa=b['audits'][condition]['audits']
        if len(aa)!=2:raise ValueError('Incomplete candidate audit')
        for candidate in aa:
            if not all(candidate[k] for k in ['input_unchanged','weights_version_unchanged','hooks_removed']):
                raise ValueError('Mutation/cleanup audit failure')
            layers=candidate['layer_attention']
            if [r['layer'] for r in layers]!=list(range(32)):raise ValueError('Incomplete native layer observation')
            for r in layers:
                if r['visual_tokens']!=576 or r['row_sum_error']>.002 or r['future_attention_max'] or r['padding_attention_max']:
                    raise ValueError('Attention normalization/causal/token retention failed')
                count+=1
            if condition!='B0':
                records=candidate['mask_records']
                if [r['applied'] for r in records]!=[i<2 for i in range(32)]:raise ValueError('Wrong active layers')
                for r in records:
                    native_masks.add(r['native_mask_sha256'])
                    if not r['native_mask_unmodified'] or r['visual_tokens_retained']!=576:
                        raise ValueError('Native mask/token retention altered')
                    if r['layer']>=2 and r['effective_mask_sha256']!=r['native_mask_sha256']:
                        raise ValueError('Later-layer mask changed')
                for item in candidate['local_redistribution']:
                    expected_q=(list(range(max(visual)+1,apos+2)) if condition=='B1' else qpos if condition=='B3' else [apos])
                    expected_k=qpos if condition=='B4' else visual
                    if item['query_indices']!=expected_q or item['key_indices']!=expected_k:
                        raise ValueError('Query/key mask orientation or dynamic mapping failed')
                    if item['masked_mass_after'] or not item['non_target_queries_bitwise_exact'] or item['max_target_row_mass_error']>.002:
                        raise ValueError('Local target/non-target audit failed')
                    if not np.isfinite(np.asarray(item['before_group_mass'])).all() or not np.isfinite(np.asarray(item['after_group_mass'])).all():
                        raise ValueError('Nonfinite redistribution')
                    local_count+=1
    if len(native_masks)!=1:raise ValueError('Same-input native masks differ across layers/conditions')
    if sample['pair']:
        if len(body['a']['rows'])!=4:raise ValueError('Incomplete A layer/query coverage')
        if sample['matched_pair'] and not audit['a']['matched_l0_qk_invariance_passed']:
            raise ValueError('Matched L0 sanity gate failed')
    return count,local_count


def heatmaps(root,sample,paths,output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Patch
    from PIL import Image
    from transformers import AutoProcessor
    from roi_sink_common import geometry
    from .pathways import spatial
    config,_=protocol()
    ip=AutoProcessor.from_pretrained(config['model_path'],local_files_only=True).image_processor
    with Image.open(sample['image_path']) as image:_,crop=geometry(image.convert('RGB'),ip)
    arrays=[]
    for p in paths:
        with np.load(root/p,allow_pickle=False) as stored:arrays.append(stored['attention'])
    maps=[]
    for layer in (0,1):
        for group,selector in [('Question',slice(None,-1)),('P-1',slice(-1,None))]:
            raw=[a[layer,:,selector,:].astype(float).mean((0,1)) for a in arrays]
            cond=[spatial(a[layer,:,selector,:])['distribution'] for a in arrays]
            maps.append((layer,group,raw,cond))
    rawmax=max(float(v.max()) for _,_,rr,_ in maps for v in rr)
    condmax=max(float(v.max()) for _,_,_,rr in maps for v in rr)
    fig,axes=plt.subplots(4,4,figsize=(15,16),layout='constrained')
    raw_artist=cond_artist=None
    for row,(layer,group,raw,cond) in enumerate(maps):
        for col,v in enumerate(raw+cond):
            ax=axes[row,col];ax.imshow(crop,extent=(0,336,336,0))
            artist=ax.imshow(v.reshape(24,24),extent=(0,336,336,0),cmap='magma',vmin=0,
                             vmax=rawmax if col<2 else condmax,alpha=.78,interpolation='nearest')
            if col<2:raw_artist=artist
            else:cond_artist=artist
            for question,color in [(sample,'cyan'),(sample['pair'],'lime')]:
                x0,y0,x1,y1=question['objects'][1]['crop_xyxy']
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,lw=1.7))
            number=col%2+1;mass=float(raw[number-1].sum())
            ax.set_title(f'L{layer} {group} Q{number}: '+('raw visual probability' if col<2 else 'visual conditional')+f'\nraw Mass={mass:.5g}',fontsize=9)
            ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(raw_artist,ax=axes[:,:2],shrink=.6,label='Raw visual probability (shared all raw panels)')
    fig.colorbar(cond_artist,ax=axes[:,2:],shrink=.6,label='Visual conditional probability (shared all conditional panels)')
    title=f"Image {sample['img_id']}; Q1 {sample['question_id']}: {sample['question']}\nQ2 {sample['pair']['question_id']}: {sample['pair']['question']}\nCyan: Q1 subject; green: Q2 subject. Mean of per-head/per-query conditional distributions."
    fig.suptitle('\n'.join(textwrap.fill(line,115) for line in title.splitlines()),fontsize=10)
    fig.savefig(output,dpi=125);plt.close(fig)


def plots(rows,output):
    import matplotlib.pyplot as plt
    names=CONDITIONS[1:];ids=sorted({r['question_id'] for r in rows});lookup={(r['question_id'],r['condition']):r for r in rows}
    colors=['#287ca0','#bd4a5c','#32976a','#a87924']
    fig,ax=plt.subplots(figsize=(15,5))
    for j,name in enumerate(names):
        ax.plot(range(len(ids)),[lookup[q,name]['gain'] for q in ids],marker='o',lw=1,label=name,color=colors[j],ms=4)
    ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(len(ids)),ids,rotation=55,ha='right')
    ax.set(ylabel='Paired Margin Gain [nats]',title='Frozen 16-image stratified pilot; every image retained');ax.legend()
    fig.tight_layout();fig.savefig(output/'paired_margin_gain.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for ax,name in zip(axes.flat,names):
        for label,color in [('left','#287ca0'),('right','#bd4a5c')]:
            rr=[lookup[q,name] for q in ids if lookup[q,name]['answer']==label]
            ax.scatter([r['full_margin'] for r in rr],[r['margin'] for r in rr],label=label,color=color)
        lo=min(*ax.get_xlim(),*ax.get_ylim());hi=max(*ax.get_xlim(),*ax.get_ylim())
        ax.plot([lo,hi],[lo,hi],'--',color='gray');ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
        ax.set(title=name,xlabel='Native Full Margin',ylabel='Intervened Margin');ax.legend()
    fig.tight_layout();fig.savefig(output/'margin_vs_full.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,14))
    for row,(layer,group) in enumerate([(0,'question'),(0,'answer'),(1,'question'),(1,'answer')]):
        for col,key in enumerate(['cosine_distance_mean','change_norm_mean']):
            ax=axes[row,col]
            for q in ids:
                ax.plot(range(4),[lookup[q,n][f'l{layer}_{group}_{key}'] for n in names],alpha=.3,lw=.6,color='gray')
            for j,name in enumerate(names):
                ax.scatter([j]*len(ids),[lookup[q,name][f'l{layer}_{group}_{key}'] for q in ids],s=16,color=colors[j],alpha=.8)
            ax.set_xticks(range(4),names);ax.set(title=f'L{layer} {group}',ylabel=key)
    fig.tight_layout();fig.savefig(output/'early_hidden_drift.png',dpi=150);plt.close(fig)
    groups=['image','question','other_text'];palette=['#32976a','#bd4a5c','#8b9095']
    fig,axes=plt.subplots(2,4,figsize=(15,8))
    for layer in (0,1):
        for j,name in enumerate(names):
            ax=axes[layer,j];rr=[lookup[q,name] for q in ids]
            deltas=np.array([[r[f'l{layer}_local_redistributed_{g}_mass'] for g in groups] for r in rr]).mean(0)
            ax.bar(groups,deltas,color=palette);ax.axhline(0,color='gray',lw=.6)
            blocked=np.mean([r[f'l{layer}_masked_local_native_mass_mean'] for r in rr])
            ax.set(title=f'{name} L{layer}; blocked Mass={blocked:.4g}',ylabel='Local masked - native Mass')
            ax.tick_params(axis='x',rotation=25)
    fig.suptitle('Softmax redistribution, same actual local hidden input; not global Full L1')
    fig.tight_layout();fig.savefig(output/'local_mass_redistribution.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for layer in (0,1):
        for col,group in enumerate(['question','answer']):
            ax=axes[layer,col];bottom=np.zeros(5)
            for g,color in zip(groups,palette):
                values=[np.mean([lookup[q,n][f'l{layer}_{group}_{g}_mass'] for q in ids]) for n in CONDITIONS]
                ax.bar(CONDITIONS,values,bottom=bottom,color=color,label=g);bottom+=values
            ax.set(title=f'L{layer} {group}',ylabel='Actual group-mean Attention Mass',ylim=(0,1.02));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'full_vs_mask_group_mass.png',dpi=150);plt.close(fig)


def report(result):
    b=result['b'];a=result['a'];config=result['configuration'];n=config['experiment_a_pairs']
    lines=['# Phase 3.7: L0-L1 Early Fusion and Information Pathways','',
           '**Research status: completed.** Frozen descriptive stratified pilot: 16 B images; '+str(n)+' real same-image A question pairs.',
           f"Baseline `{config['baseline']}`; frozen implementation `{config['code_commit']}`.",
           'Only one GPU; original FP16/eager LLaVA-1.5-7B / Transformers4.49.0. No training, decode, cache, physical exit or extra layer scan.','',
           '## Frozen Sampling and Provenance','',
           config['rules']['selection'],config['rules']['pair'],config['rules']['source'],
           f"The fixed source has {config['source_available_pair_images']} geometry-valid pair images among the historical240. The stratified pilot has {n} pairs, {config['experiment_a_matched_pairs']} with identical prompt length/P-1 position/token. Missing pairs remain missing; no invented replacement.",
           'Historical outcomes are used only for stratification; this is not an unbiased population estimate. Current Native Full is the paired baseline. No statistical-significance claim or post-hoc best condition.','',
           '## Experiment B','',
           '| Condition | Accuracy | Harmed / recovered / ties | Mean / median Gain | Minimum Gain | Mean absolute gold LogP change |',
           '| --- | ---: | --- | --- | ---: | ---: |']
    for name in CONDITIONS:
        r=b[name]['overall'];lines.append(f"| {name} | {r['accuracy']:.4f} | {r['harmed']} / {r['recovered']} / {r['ties']} | {r['mean_gain']:+.6f} / {r['median_gain']:+.6f} | {r['min_gain']:+.6f} | {r['mean_absolute_gold_logp_gain']:+.6f} |")
    lines+=['','B1: all valid post-image Text queries -> Image keys; B2: P-1 -> Image; B3: true Question -> Image; B4: P-1 -> true Question. Only L0/L1. P-1 is before the teacher-forced answer, not the answer token itself.',
            '','### Left / Right','', '| Condition | Left accuracy / mean Gain | Right accuracy / mean Gain |','| --- | --- | --- |']
    for name in CONDITIONS:
        l,r=b[name]['left'],b[name]['right'];lines.append(f"| {name} | {l['accuracy']:.4f} / {l['mean_gain']:+.6f} | {r['accuracy']:.4f} / {r['mean_gain']:+.6f} |")
    lines+=['','## Experiment A','',config['rules']['spatial'],config['rules']['hidden'],
            'Each heatmap labels raw visual Mass separately from conditional spatial distribution. Both subject boxes are shown, with shared scales for all raw panels and all conditional panels within a pair.',
            '', '| Layer / Query | Mean spatial JS distance | Mean cosine | Q1 / Q2 visual Mass | Q1 / Q2 normalized entropy | Mean paired own-vs-other ROI preference |',
            '| --- | ---: | ---: | --- | --- | ---: |']
    for name,r in a.items():
        lines.append(f"| {name} | {r['js_distance_mean']:.6g} | {r['cosine_mean']:.6g} | {r['q1_mass_mean']:.6g} / {r['q2_mass_mean']:.6g} | {r['q1_entropy_mean']:.6g} / {r['q2_entropy_mean']:.6g} | {r['roi_preference_mean']:+.6g} |")
    lines+=['','ROI preference = half the sum of (own target enrichment - other question target enrichment) for Q1 and Q2. Enrichment divides conditional ROI probability by ROI patch fraction; box overlaps and empty background are retained, not invented. Values are descriptive, not a validated test of task selection.',
            f"All {config['experiment_a_matched_pairs']} matched pairs pass the L0 P-1 input / visual-input / visual-QK and FP32 visual-only distribution exact checks. Any nonzero spatial discrepancy from normalized stored FP16 probabilities is not evidence against the structural L0 invariance.",
            '','## Seven Research Questions','',
            '1. Task specificity: L0 P-1 spatial selectivity cannot depend on changed Question content under the matched conditions, as checked directly. Real Question queries differ lexically, so their L0 map differences alone do not isolate task-dependent visual reasoning. L1 can incorporate prior-layer context; use paired ROI preference and heatmaps, not JS difference alone, to assess alignment. Strong universal spatial-selection claims remain unsupported by this small pilot.',
            f"2. Early all-Text visual blocking B1: mean Gain {b['B1']['overall']['mean_gain']:+.6f}, median {b['B1']['overall']['median_gain']:+.6f}, harmed/recovered {b['B1']['overall']['harmed']}/{b['B1']['overall']['recovered']}. This is the measured local causal effect, not evidence about removing visual tokens.",
            f"3. P-1 direct image reading B2 versus Question image reading B3: mean Gain {b['B2']['overall']['mean_gain']:+.6f} versus {b['B3']['overall']['mean_gain']:+.6f}. Compare absolute per-image changes and hidden drifts as well as signs; a larger beneficial mean is not necessarily a larger required information contribution.",
            f"4. Early P-1 -> Question B4: mean Gain {b['B4']['overall']['mean_gain']:+.6f}, harmed/recovered {b['B4']['overall']['harmed']}/{b['B4']['overall']['recovered']}. Other text keys and all later layers remain accessible. This does not test absence of all textual dependence.",
            '5. Support for early global fusion: diffuse/high-entropy maps, matched L0 P-1 spatial invariance and broad redistribution are compatible with early fusion, but entropy alone is not proof of global semantic integration. Contrast each actual result with the figures and numeric tables.',
            '6. Contradictory evidence must be retained: positive task-aligned ROI preference at either layer, sharp maps, adverse per-image mask effects, and differences between Question/P-1 pathways would limit an undifferentiated global-fusion claim. Worst losses and every paired sample remain in CSV/summary; no result-based filtering.',
            '7. L2-and-later compensation is possible: every later layer still has native masks and all visual tokens. The present design does not intervene on compensation or isolate alternative text routes. Lack of accuracy harm does not establish that a pathway has no function.',
            '', 'Attention association, local causal intervention, Hidden change and final prediction are distinct evidence. Hidden drift is not information quality; B2 is not a Question-only visual route, B3 is not a visually empty Question, and B4 is not removal of all text. No automatic extension is authorized.',
            '', '## Audit and Numerical Differences','',
            f"Recorded B attention audits: {result['audit']['b_layer_audits']}; local masked-layer/candidate audits: {result['audit']['local_mask_audits']}. CPU and two-image GPU smoke gates passed before the formal pilot. No-op valid-text logits and post-cleanup Native Full are exact.",
            'All masks reuse the existing clone/-inf implementation. Blocked target attention is zero; causal/padding constraints and nonempty legal rows hold; non-target queries are locally exact; L2-L31 masks and all 576 visual tokens are retained. Softmax redistribution of unblocked keys is expected and recorded.',
            'Full before/after model parameter+buffer byte hashes match; input/parameter versions and installed Transformers files are unchanged. Exception cleanup is CPU-tested. No failure is silently counted as complete. Historical Full Margin differences are recorded per image; they are never substituted for current paired Full or fitted with new tolerances.',
            '', '## Artifacts','',
            '[Per-image B scores and drift/mass metrics](per_sample_b.csv), [per-pair A metrics](per_pair_a.csv), [complete descriptive summary and worst cases](summary.json), [frozen manifest](manifest.json).',
            'Raw observations, complete candidate mappings, compressed audits, source/model/input hashes and environment are under pilot/ and smoke/. Reproduce with the frozen implementation commit, not a later result-only commit.',
            '', '![Paired Gain](paired_margin_gain.png)','![Margins versus Full](margin_vs_full.png)',
            '![Early hidden drift](early_hidden_drift.png)','![Local redistribution](local_mass_redistribution.png)',
            '![Actual group Mass](full_vs_mask_group_mass.png)','']
    for s in config['samples']:
        if s['pair']:lines.append(f"![A image {s['img_id']}](heatmaps/{s['question_id']}.png)")
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def aggregate():
    import matplotlib
    matplotlib.use('Agg')
    c,binding=protocol()
    if (OUTPUT/'summary.json').exists():raise FileExistsError('Do not aggregate the completed pilot twice')
    root=OUTPUT/'pilot';gate=read(root/'validation.json')
    ids=[s['question_id'] for s in c['samples']]
    if not gate['passed'] or gate['binding']!=binding or gate['question_ids']!=ids or gate['count']!=16:
        raise ValueError('Formal complete gate failed; report partial instead')
    if gate['weights_sha256_before']!=gate['weights_sha256_after'] or not gate['third_party_source_unchanged']:
        raise ValueError('Model/library audit failed')
    if {p.stem for p in (root/'checkpoints').glob('*.json')}!=set(ids):raise ValueError('Wrong sample coverage')
    rows=[];arows=[];layer_count=local_count=0;heatmap_paths=[]
    for s in c['samples']:
        p=root/'checkpoints'/f"{s['question_id']}.json"
        if sha(p)!=gate['checkpoints_sha256'][s['question_id']]:raise ValueError('Completed checkpoint changed')
        body=checkpoint(p,binding,s,root);n,k=verify_body(body,s,root);layer_count+=n;local_count+=k
        rows+=body['b']['rows']
        if body['a']:
            arows+=body['a']['rows']
            heatmap_paths.append((s,[next(p for p in body['artifacts'] if p.endswith(f'A_Q{i}.npz')) for i in (1,2)]))
    if len(rows)!=80 or len(arows)!=c['experiment_a_pairs']*4:raise ValueError('Incomplete condition coverage')
    for r in rows:
        if not np.isfinite([r['gain'],r['margin'],r['correct_logp'],r['wrong_logp']]).all() or abs(r['margin']-(r['correct_logp']-r['wrong_logp']))>2e-12:
            raise ValueError('Invalid score arithmetic')
    b={name:{g:descriptive([r for r in rows if r['condition']==name and (g=='overall' or r['answer']==g)])
             for g in ['overall','left','right']} for name in CONDITIONS}
    a={}
    for layer in (0,1):
        for group in ['question','answer']:
            rr=[r for r in arows if r['layer']==layer and r['query_group']==group]
            if not rr:continue
            preference=[(r['q1_own_target_enrichment']-r['q1_other_question_target_enrichment']+
                         r['q2_own_target_enrichment']-r['q2_other_question_target_enrichment'])/2 for r in rr]
            a[f'L{layer}_{group}']=dict(n=len(rr),js_distance_mean=float(np.mean([r['js_distance'] for r in rr])),
                cosine_mean=float(np.mean([r['cosine'] for r in rr])),q1_mass_mean=float(np.mean([r['q1_image_mass'] for r in rr])),
                q2_mass_mean=float(np.mean([r['q2_image_mass'] for r in rr])),
                q1_entropy_mean=float(np.mean([r['q1_normalized_entropy'] for r in rr])),
                q2_entropy_mean=float(np.mean([r['q2_normalized_entropy'] for r in rr])),
                roi_preference_mean=float(np.mean(preference)),roi_preference_positive=sum(x>0 for x in preference))
    write_csv(OUTPUT/'per_sample_b.csv',rows);write_csv(OUTPUT/'per_pair_a.csv',arows)
    result=dict(binding=binding,configuration=c,b=b,a=a,audit=dict(b_layer_audits=layer_count,local_mask_audits=local_count))
    atomic(OUTPUT/'summary.json',result)
    plots(rows,OUTPUT);heatmap_dir=OUTPUT/'heatmaps';heatmap_dir.mkdir(exist_ok=False)
    for s,paths in heatmap_paths:heatmaps(root,s,paths,heatmap_dir/f"{s['question_id']}.png")
    report(result)
    atomic(OUTPUT/'summary_validation.json',dict(passed=True,status='completed',binding=binding,code_commit=c['code_commit'],
        images_b=16,rows_b=80,pairs_a=c['experiment_a_pairs'],rows_a=len(arows),b_layer_audits=layer_count,
        local_mask_audits=local_count,aggregation_count=1,
        artifacts={str(p.relative_to(OUTPUT)):sha(p) for p in OUTPUT.rglob('*') if p.is_file() and p.name not in ['.lock','summary_validation.json']}))
    print('SUMMARY_COMPLETE',16,len(arows),flush=True)


if __name__=='__main__':aggregate()
