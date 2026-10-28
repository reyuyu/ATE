"""Only summarize the five frozen conditions after the complete 16-image gate."""
import csv
from datetime import datetime, timezone
import numpy as np
from phase38_sink_stage import (OUTPUT, CONDITIONS, HIDDEN, protocol, read,
                                sha256, atomic_json, write_csv, digest)

LABELS = {'C0':'Native Full','C1':'L2-L4 Attention bypass','C2':'L2-L4 MLP bypass',
          'C3':'L2-L4 BOS only','C4':'L5-L7 Attention bypass'}


def stats(rows):
    x=np.array([r['gain'] for r in rows]);a=np.array([r['gold_logp_gain'] for r in rows])
    return dict(n=len(rows),correct=sum(r['correct'] for r in rows),accuracy=float(np.mean([r['correct'] for r in rows])),
        mean_gain=float(x.mean()),median_gain=float(np.median(x)),min_gain=float(x.min()),max_gain=float(x.max()),
        p5_gain=float(np.quantile(x,.05)),p95_gain=float(np.quantile(x,.95)),
        positive_gain=int((x>0).sum()),negative_gain=int((x<0).sum()),zero_gain=int((x==0).sum()),
        correct_to_wrong=sum(r['correct_to_wrong'] for r in rows),wrong_to_correct=sum(r['wrong_to_correct'] for r in rows),
        mean_gold_logp_gain=float(a.mean()),gold_logp_increased=int((a>0).sum()))


def plots(scores,mechanics,hidden):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#666666','#287da1','#bc4770','#40986e','#a87322']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    layers=list(range(11))
    fields=['bos_mass','image_mass','question_mass','other_text_mass']
    masses=[[np.mean([r[k] for r in mechanics if r['layer']==i]) for i in layers] for k in fields]
    axes[0].stackplot(layers,*masses,labels=['BOS','Image','Question','Other text'],colors=['#287da1','#bc4770','#40986e','#d6b35c'])
    axes[0].set(xlabel='Layer (zero-based)',ylabel='Native P-1 raw Attention mass',ylim=(0,1.01));axes[0].legend(loc='lower right')
    for field,label in [('sink_inside_image','Mean of per-head Sink / Image'),('sink_ratio_of_head_mean_mass','Ratio of head-mean masses')]:
        axes[1].plot(layers,[np.mean([r[field] for r in mechanics if r['layer']==i]) for i in layers],marker='o',label=label)
    axes[1].set(xlabel='Layer (zero-based)',ylabel='Frozen Sink fraction inside Image',ylim=(0,1.01));axes[1].legend()
    fig.suptitle('16-image native observations; conditional fractions are not stacked with raw mass')
    fig.tight_layout();fig.savefig(OUTPUT/'attention_structure.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for offset,field,label,color in [(-.13,'attention_relative_norm','Attention / layer input','#287da1'),(.13,'mlp_relative_to_layer_input','MLP / layer input','#bc4770')]:
        for i in (2,3,4):
            values=[r[field] for r in mechanics if r['layer']==i]
            axes[0].scatter(np.linspace(i+offset-.07,i+offset+.07,len(values)),values,s=14,alpha=.65,color=color)
        axes[0].plot(np.array([2,3,4])+offset,[np.mean([r[field] for r in mechanics if r['layer']==i]) for i in (2,3,4)],marker='D',label=label,color=color)
    for field,label,color in [('attention_cosine','Attention vs entering residual','#287da1'),('mlp_cosine','MLP vs pre-MLP residual','#bc4770')]:
        axes[1].plot([2,3,4],[np.mean([r[field] for r in mechanics if r['layer']==i and r[field] is not None]) for i in (2,3,4)],marker='o',label=label,color=color)
    axes[0].set(xlabel='Layer',ylabel='Actual output-update relative L2 norm',xticks=[2,3,4]);axes[0].legend()
    axes[1].axhline(0,color='gray',lw=.7);axes[1].set(xlabel='Layer',ylabel='Cosine with actual corresponding residual',xticks=[2,3,4]);axes[1].legend()
    fig.suptitle('Native P-1 updates at o_proj output and MLP output; individual image norms retained')
    fig.tight_layout();fig.savefig(OUTPUT/'actual_updates.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for j,name in enumerate(CONDITIONS):
        rr=[r for r in scores if r['condition']==name];g=[r['gain'] for r in rr]
        axes[0].scatter(np.linspace(j-.14,j+.14,16),g,s=20,color=colors[j],alpha=.7)
        axes[0].plot(j,np.mean(g),marker='_',ms=23,color='black')
        axes[1].bar(j,np.mean([r['correct'] for r in rr]),color=colors[j])
        axes[2].bar(j-.16,sum(r['correct_to_wrong'] for r in rr),width=.3,color='#bc4770',label='Correct to wrong' if j==0 else None)
        axes[2].bar(j+.16,sum(r['wrong_to_correct'] for r in rr),width=.3,color='#40986e',label='Wrong to correct' if j==0 else None)
    for ax in axes:ax.set_xticks(range(5),list(CONDITIONS))
    axes[0].axhline(0,color='gray',lw=.7);axes[0].set_ylabel('Margin gain [nats], every image')
    axes[1].set(ylabel='Pilot accuracy (not population estimate)',ylim=(0,1))
    axes[2].set_ylabel('Paired flip count, ties incorrect');axes[2].legend()
    fig.tight_layout();fig.savefig(OUTPUT/'five_condition_outcomes.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.5));locations=[str(k) for k in HIDDEN]
    for j,name in enumerate(list(CONDITIONS)[1:],1):
        for ax,field in zip(axes,['relative_l2','cosine_distance']):
            for q in sorted({r['question_id'] for r in hidden}):
                rr={r['location']:r for r in hidden if r['question_id']==q and r['condition']==name}
                ax.plot(range(7),[rr[k][field] for k in locations],color=colors[j],alpha=.1,lw=.6)
            ax.plot(range(7),[np.mean([r[field] for r in hidden if r['condition']==name and r['location']==k]) for k in locations],color=colors[j],marker='o',label=name)
    for ax in axes:ax.set_xticks(range(7),['L1','L2','L3','L4','L7','L31','Final norm']);ax.legend();ax.set_xlabel('Post-layer P-1 hidden / final RMSNorm')
    axes[0].set_ylabel('Relative L2 drift from same-machine Full');axes[1].set_ylabel('Cosine distance from same-machine Full')
    fig.tight_layout();fig.savefig(OUTPUT/'hidden_drift.png',dpi=170);plt.close(fig)


def report(summary,mechanics,hidden,scores):
    lines=['# Phase 3.8: L2-L4 Attention Sink and Computational Stability','',
        'Status: **completed, 16/16 frozen images, C0-C4 only**. This is a small mechanism pilot, not an overall accuracy benchmark.',
        f"Frozen implementation: `{summary['code_commit']}`. Only physical GPU1 was used. GPU0 worktree, source, branch and results were not modified.",
        '','## Selection and Protocol','',
        'Eight left and eight right images are selected deterministically from the frozen Phase3.4 main48; discovery15133706 and the other historical cases are excluded. Right strata use historical Full correctness, never current intervention effects. Full selection rules, image/source hashes and frozen Sink/ROI intersections are in manifest.json.',
        'LLaVA-1.5-7B, FP16/eager, Transformers4.49.0, fixed model revision, original prompt and complete candidate conditional scoring. Every sequence retains all576 image tokens and original positions. P-1, BOS and Question locations are verified dynamically against original mappings.',
        'C1/C4 zero only P-1 o_proj output before Attention residual addition; MLP runs normally. C2 zeroes only P-1 MLP output before its residual addition. C3 masks all P-1 keys except causal BOS before softmax; output projection, residual and MLP remain active. BOS-only renormalizes to1 and is neither a bypass nor a mass-matched control.',
        '','## Outcomes','',
        '| Condition | Correct /16 | Mean / median gain | Min / max gain | Correct to wrong / wrong to correct | Mean gold absolute LogP change |',
        '| --- | ---: | --- | --- | --- | ---: |']
    for name in CONDITIONS:
        s=summary['conditions'][name]['overall']
        lines.append(f"| {name}: {LABELS[name]} | {s['correct']}/16 | {s['mean_gain']:+.6f} / {s['median_gain']:+.6f} | {s['min_gain']:+.6f} / {s['max_gain']:+.6f} | {s['correct_to_wrong']} / {s['wrong_to_correct']} | {s['mean_gold_logp_gain']:+.6f} |")
    lines+=['','Ties are incorrect. Gold absolute LogP uses the full vocabulary, not a two-candidate renormalization. Small or positive Margin changes are not capability improvement. All negative samples are preserved.','',
            '| Condition | Left correct /8; mean Gain | Right correct /8; mean Gain |','| --- | --- | --- |']
    for name in CONDITIONS:
        l,r=(summary['conditions'][name][g] for g in ('left','right'))
        lines.append(f"| {name} | {l['correct']}/8; {l['mean_gain']:+.6f} | {r['correct']}/8; {r['mean_gain']:+.6f} |")
    def sm(name):return summary['conditions'][name]['overall']
    lines+=['','## Eight Research Questions','',
        f"1. **Can L2-L4 Attention be skipped with little impact?** C1 mean Gain {sm('C1')['mean_gain']:+.6f}, range [{sm('C1')['min_gain']:+.6f},{sm('C1')['max_gain']:+.6f}], flips {sm('C1')['correct_to_wrong']}/{sm('C1')['wrong_to_correct']}. Report these magnitudes and individual examples, not an equivalence claim: no equivalence margin or population power was established.",
        f"2. **MLP versus Attention?** C2 mean Gain {sm('C2')['mean_gain']:+.6f} versus C1 {sm('C1')['mean_gain']:+.6f}; C2 flips {sm('C2')['correct_to_wrong']}/{sm('C2')['wrong_to_correct']}. Updates have different norms/directions below. This intervention comparison does not measure intrinsic module importance under matched perturbation sizes.",
        f"3. **Does BOS-only preserve behavior?** C3 mean Gain {sm('C3')['mean_gain']:+.6f}, range [{sm('C3')['min_gain']:+.6f},{sm('C3')['max_gain']:+.6f}], flips {sm('C3')['correct_to_wrong']}/{sm('C3')['wrong_to_correct']}. The sole legal key really has probability1, but K/V and BOS hidden are not edited. Functional similarity must be judged from scores and drift, not high native BOS mass alone.",
        f"4. **Is L2-L4 special compared with L5-L7?** C1 mean Gain {sm('C1')['mean_gain']:+.6f} versus C4 {sm('C4')['mean_gain']:+.6f}; paired C1-C4 mean difference {summary['paired_attention_windows_mean_gain_difference']:+.6f}. Same three-layer count is controlled; this is not a scan or a universal depth claim.",
        f"5. **Does high BOS mass mean a large update?** Across16 images, the descriptive Pearson correlation of each image's mean L2-L4 BOS mass with its mean Attention/layer-input norm ratio is {summary['bos_mass_vs_attention_relative_norm_pearson']}. See actual raw mass and output-vector norms below. A probability mass is not a vector contribution: Values, output projection, direction and residual scale all matter. Source vector norms cannot be added.",
        '6. **Do early hidden changes recover later?** The following all-image summaries compare post-L4 drift with final RMSNorm drift. A smaller later cosine drift or retained answer can indicate compensation, but does not prove early interaction was unused. Different residual scales and final normalization preclude comparing raw L2 magnitudes as if they were the same coordinates.',
        '| Condition | Mean L4 relative L2 / cosine | Mean final relative L2 / cosine | Images with lower final cosine drift than L4 |',
        '| --- | --- | --- | ---: |']
    for name in list(CONDITIONS)[1:]:
        a,b=summary['hidden_summary'][name]['4'],summary['hidden_summary'][name]['final_norm']
        lines.append(f"| {name} | {a['relative_l2']:.6f} / {a['cosine_distance']:.6f} | {b['relative_l2']:.6f} / {b['cosine_distance']:.6f} | {summary['hidden_compensation_descriptive'][name]}/16 |")
    h1=(f"C1 causes {sm('C1')['correct_to_wrong']} correct-to-wrong flips: these images directly oppose the claim that their L2-L4 Attention can simply be omitted without answer harm."
        if sm('C1')['correct_to_wrong'] else
        'C1 causes no correct-to-wrong flips in these16 images, which is limited pilot evidence of answer retention, not numerical equivalence or proof that Attention is unnecessary.')
    h2=(f"C3 causes {sm('C3')['correct_to_wrong']} correct-to-wrong flips; BOS-only is not sufficient to retain all native answers."
        if sm('C3')['correct_to_wrong'] else
        'C3 retains all initially correct pilot answers, but its nonzero score/hidden effects still need inspection; this does not identify a BOS stabilization mechanism.')
    lines+=['',
        '7. **Stable computation hypothesis?** '+h1+' '+h2+' C2/C1 differ in both sublayer and actual update size, so their relative damage cannot establish intrinsic importance. H2 is not uniquely identified: BOS-only tests a particular sufficiency intervention, not BOS necessity or a mass-matched stability control. Neither high sink mass nor later compensation demonstrates a passive stage. The strong no-new-interaction interpretation must be constrained by the actual causal losses, not assumed.',
        '8. **Why not delete most image/text tokens?** Interventions affect one query at three layers, not the token representations, other queries, K/V storage or other layers. Removing tokens changes all those computations, positions and normalization. No token deletion, decoding or other task was tested.',
        '','## Actual Native Updates at L2-L4','',
        '| Layer | BOS / Image raw mass | Sink raw / mean headwise Image fraction | Attention norm / layer-input norm | MLP norm / layer-input norm | MLP / pre-MLP norm | Attention / MLP cosines |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for i in (2,3,4):
        rr=[r for r in mechanics if r['layer']==i]
        mean=lambda k:float(np.mean([r[k] for r in rr]))
        lines.append(f"| L{i} | {mean('bos_mass'):.6f} / {mean('image_mass'):.6f} | {mean('sink_raw_mass'):.6f} / {mean('sink_inside_image'):.6f} | {mean('attention_norm'):.6f} / {mean('attention_relative_norm'):.6f} | {mean('mlp_norm'):.6f} / {mean('mlp_relative_to_layer_input'):.6f} | {mean('mlp_relative_norm'):.6f} | {mean('attention_cosine'):.6f} / {mean('mlp_cosine'):.6f} |")
    lines+=['',
        'Raw BOS/Image/Question/other-text masses partition keys. Sink raw mass is a subset of Image, not another additive partition. Sink conditional fraction is mean of headwise ratios; ratio of head-mean masses is separately saved. Native vectors are measured at actual o_proj and MLP outputs, and relative/cosine residuals are explicitly labeled. Layer hidden states are post-layer; final_norm is separately post-final RMSNorm.',
        '','## Integrity and Limits','',
        f"CPU regression: {summary['cpu_tests']} tests, no skips. Two real GPU smoke images pass before16 formal images. All actual model state_dict tensor hashes agree before/after each GPU stage; temporary hooks, input tensors and installed third-party source remain unchanged. L1 and all causally earlier text states remain bitwise Full; candidate P-1 predictions agree. Final Native Full restoration is exact. No successful-run sample is missing or duplicated.",
        'Historical/current Full numerical differences are recorded per image, not used to fit tolerance or select data. The10 package pins and canonical model files are preserved; actual Python patch/hardware are saved. Baseline weights/data/model configuration and Phase2-3.7 files are not modified.',
        'No confidence interval over heads/layers is used. The16 selected images and historical correctness strata are not a representative population accuracy evaluation. Full per-image scores, hidden drifts, headwise raw rows and logs are retained. No additional research condition is introduced.',
        '', '![Native structure](attention_structure.png)','![Actual updates](actual_updates.png)',
        '![Outcomes](five_condition_outcomes.png)','![Hidden drift](hidden_drift.png)']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    if (OUTPUT/'summary_validation.json').exists():raise FileExistsError('Master summary already completed')
    c,b=protocol();gate=read(OUTPUT/'run/validation.json')
    if not gate['passed'] or gate['count']!=16 or gate['binding']!=b or gate['question_ids']!=[s['question_id'] for s in c['samples']]:raise ValueError('All16 samples required before summarizing')
    scores=[];mechanics=[];hidden=[];data=[]
    for s in c['samples']:
        q=s['question_id'];folder=OUTPUT/'run'/f'sample_{q}'
        e=read(folder/'checkpoint.json');d=read(folder/'sample.json')
        if e['binding']!=b or e['sample_sha256']!=digest(s) or e['sample_json_sha256']!=sha256(folder/'sample.json') or gate['sample_sha256'][q]!=sha256(folder/'sample.json') or d['raw_artifact_sha256']!=sha256(folder/'raw_vectors_logits_attention.npz'):raise ValueError('Changed sample/artifact')
        if not d['passed'] or not d['post_cleanup_native_exact'] or [r['condition'] for r in d['scores']]!=list(CONDITIONS) or len(d['hidden_drift'])!=35 or len(d['mechanics'])!=32:raise ValueError('Invalid sample coverage')
        scores+=d['scores'];mechanics+=d['mechanics'];hidden+=d['hidden_drift'];data.append(d)
    if len(scores)!=80 or len({(r['question_id'],r['condition']) for r in scores})!=80:raise ValueError('Duplicate or missing scores')
    summary=dict(code_commit=c['code_commit'],binding=b,n=16,cpu_tests=read(OUTPUT/'tests/validation.json')['tests'],conditions={},hidden_summary={},hidden_compensation_descriptive={})
    for name in CONDITIONS:
        rr=[r for r in scores if r['condition']==name]
        summary['conditions'][name]={g:stats([r for r in rr if g=='overall' or r['answer']==g]) for g in ['overall','left','right']}
        hh=[r for r in hidden if r['condition']==name]
        summary['hidden_summary'][name]={str(k):{f:float(np.mean([r[f] for r in hh if r['location']==str(k)])) for f in ['l2','relative_l2','cosine_distance']} for k in HIDDEN}
        lookup={(r['question_id'],r['location']):r for r in hh}
        summary['hidden_compensation_descriptive'][name]=sum(lookup[s['question_id'],'final_norm']['cosine_distance']<lookup[s['question_id'],'4']['cosine_distance'] for s in c['samples'])
    summary['paired_attention_windows_mean_gain_difference']=float(np.mean([r['gain'] for r in scores if r['condition']=='C1'])-np.mean([r['gain'] for r in scores if r['condition']=='C4']))
    bos=[];update=[]
    for s in c['samples']:
        rr=[r for r in mechanics if r['question_id']==s['question_id'] and r['layer'] in (2,3,4)]
        bos.append(np.mean([r['bos_mass'] for r in rr]));update.append(np.mean([r['attention_relative_norm'] for r in rr]))
    summary['bos_mass_vs_attention_relative_norm_pearson']=float(np.corrcoef(bos,update)[0,1]) if np.std(bos) and np.std(update) else None
    summary['historical_full_differences']={d['question_id']:dict(candidate_logp_difference=d['historical_candidate_logp_difference'],historical_margin=d['historical_full_margin'],current_margin=d['same_machine_full_margin']) for d in data}
    write_csv(OUTPUT/'per_sample_scores.csv',scores);write_csv(OUTPUT/'native_vector_statistics.csv',mechanics);write_csv(OUTPUT/'hidden_drift.csv',hidden)
    atomic_json(OUTPUT/'summary.json',summary)
    plots(scores,mechanics,hidden);report(summary,mechanics,hidden,scores)
    atomic_json(OUTPUT/'summary_validation.json',dict(passed=True,n=16,score_rows=80,native_vector_rows=512,hidden_drift_rows=560,
        code_commit=c['code_commit'],binding=b,timestamp=datetime.now(timezone.utc).isoformat(),
        artifacts={p.name:sha256(p) for p in OUTPUT.iterdir() if p.is_file() and p.name!='summary_validation.json'}))
    print('COMPLETE SUMMARY',16,80,512,560,flush=True)


if __name__=='__main__':main()
