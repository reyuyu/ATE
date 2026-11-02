"""CPU-only analysis and paper figures; never changes frozen GPU workers."""
from collections import defaultdict
import json
import math
from pathlib import Path
import textwrap
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

from benchmarks.dataset_registry import digest, sha256
from paper_rq1.scorers import caption_scores, gqa_prediction
from temporal_visual_gap.common import (OUT, HROOT, SEED, TASKS, CONDITIONS, read, write,
                                       csv_write, verify, gate)
from temporal_visual_gap.run import checkpoint

REPS = 20000
METRICS = ['CIDEr', 'SPICE']
ROLES = ['Function', 'Visual', 'Relation']


def bounds(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def paired(a, b, indices):
    delta = np.asarray(a, float)-np.asarray(b, float)
    lo, hi = bounds(delta[indices].mean(1))
    return float(delta.mean()), lo, hi


def first_difference(a, b):
    return next((i for i, (x,y) in enumerate(zip(a,b)) if x != y), min(len(a),len(b)) if len(a) != len(b) else -1)


def cluster_summary(rows, key, repetitions=REPS):
    groups = defaultdict(list)
    for row in rows: groups[row['image_sha256']].append(float(row[key]))
    values = list(groups.values())
    totals = np.array([sum(v) for v in values]); counts = np.array([len(v) for v in values])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), (repetitions, len(values)))
    samples = totals[indices].sum(1)/counts[indices].sum(1)
    lo, hi = bounds(samples)
    return dict(mean=float(totals.sum()/counts.sum()), ci_low=lo, ci_high=hi,
                tokens=int(counts.sum()), images=len(values))


def collect_A(p, manifest):
    tokens, concentrations, baseline, records = [], [], [], {}
    for s in manifest['COCO']+manifest['GQA']:
        q = s['question_id']; key = q.replace(':','_')
        r = checkpoint(OUT/f'checkpoints/A/{key}.json', p, s, 'A', 'Native-ImageOFF')
        records[q] = r
        assert len(r['token_metrics']) == len(r['generation']['token_ids'])
        labels = r['word_alignment']['tokens']; words = r['word_alignment']['words']
        assert r['word_alignment']['roundtrip_exact']
        coco = s.get('dataset') == 'coco_caption'
        total = len(labels)
        for row, label in zip(r['token_metrics'], labels):
            t = row['t']; assert row['full_token'] == label['token_id']
            matches = [w for w in words if t in w['token_steps']]
            word = matches[0] if len(matches) == 1 else None
            role = label['category'] if label['category'] in ROLES+['EOS', 'Punctuation'] else 'Unknown'
            consumed_role = labels[t-1]['category'] if t else None
            stage = 'P-1' if t == 0 else 'EOS' if row['is_eos'] else 'Content Decode'
            tokens.append(dict(dataset='COCO' if coco else 'GQA', task='COCO' if coco else s['task'],
                question_id=q, image_sha256=s['image_sha256'], stage=stage,
                **row, relative_position=t/max(total-1,1), role=role,
                raw_role=label['category'], consumed_role=consumed_role,
                predicted_piece=label['decoded_piece'], consumed_piece=labels[t-1]['decoded_piece'] if t else None,
                lexical_word=label['lexical_word'], word_first=bool(word and t == word['token_steps'][0]),
                subword_index=word['token_steps'].index(t) if word else None,
                word_token_length=word['token_length'] if word else None,
                image_key_count=len(r['image_keys']), image_key_first=min(r['image_keys']),
                image_key_last=max(r['image_keys']), image_key_indices=json.dumps(r['image_keys'])))
        gen = r['generation']
        baseline.append(dict(dataset='COCO' if coco else 'GQA', task='COCO' if coco else s['task'], question_id=q,
            image_sha256=s['image_sha256'], text=gen['text'], token_ids=json.dumps(gen['token_ids']),
            non_eos_tokens=sum(t != 2 for t in gen['token_ids']), eos=gen['eos'], truncated=gen['truncated'],
            empty=gen['empty'], accuracy=None if coco else int(gqa_prediction(gen['text']) == s['answer'])))
        if coco:
            content = [x for x in r['token_metrics'] if x['t'] >= 1 and not x['is_eos']]
            loss = np.maximum([x['signed_logp_loss'] for x in content], 0.)
            ordered = np.sort(loss)[::-1]; denom = float(loss.sum()); k = math.ceil(.2*len(loss))
            large = next((x for x in content if x['signed_logp_loss'] >= p['large_loss_nats']), None)
            concentrations.append(dict(question_id=q, image_sha256=s['image_sha256'],
                content_decode_positions=len(content), positive_positions=int((loss > 0).sum()),
                top1_flips=sum(x['top1_flip'] for x in content),
                top1_flip_rate=sum(x['top1_flip'] for x in content)/len(content) if content else None,
                positive_logp_loss_total=denom, top20_count=k,
                top1_share=float(ordered[0]/denom) if denom else None,
                top20_share=float(ordered[:k].sum()/denom) if denom else None,
                rest_share=float(ordered[k:].sum()/denom) if denom else None,
                large_loss_positions=sum(x['signed_logp_loss'] >= p['large_loss_nats'] for x in content),
                first_large_t=large['t'] if large else None,
                first_large_relative=large['t']/max(total-1,1) if large else None,
                large_loss_cutoff_nats=p['large_loss_nats']))
    csv_write(OUT/'per_token.csv', tokens)
    csv_write(OUT/'native_generations.csv', baseline)
    csv_write(OUT/'concentration_per_image.csv', concentrations)
    summary = []
    for dataset in ['COCO', 'GQA']:
        for task in (['COCO'] if dataset == 'COCO' else ['ALL']+TASKS):
            for stage in ['P-1', 'Content Decode', 'EOS']:
                group = [r for r in tokens if r['dataset'] == dataset and (task == 'ALL' or r['task'] == task) and r['stage'] == stage]
                if not group: continue
                for metric in ['signed_logp_loss', 'absolute_logp_change', 'top1_flip', 'js_divergence']:
                    value = cluster_summary(group, metric)
                    array = np.asarray([r[metric] for r in group], dtype=float)
                    summary.append(dict(dataset=dataset, task=task, stage=stage, metric=metric,
                        **value, median=float(np.median(array)), p5=float(np.percentile(array,5)),
                        p95=float(np.percentile(array,95))))
    csv_write(OUT/'single_step_summary.csv', summary)
    return tokens, concentrations, baseline, records, summary


def controlled_roles(tokens):
    result, diagnostics = [], []
    for subset in ['First subword', 'All subwords']:
        eligible = [r for r in tokens if r['dataset'] == 'COCO' and r['stage'] == 'Content Decode'
                    and r['role'] in ROLES and (subset == 'All subwords' or r['word_first'])]
        groups = defaultdict(list)
        for r in eligible: groups[r['image_sha256']].append(r)
        names = ['Visual vs Function', 'Relation vs Function', 'position', 'relative_position', 'native_logp', 'entropy']
        all_x = np.array([[float(r['role']=='Visual'), float(r['role']=='Relation'), r['t'], r['relative_position'],
                           r['full_logp'], r['full_entropy']] for r in eligible])
        scale = all_x.std(0); scale[:2] = 1.
        assert np.all(scale > 0)
        matrices, vectors, counts = [], [], []
        for group in groups.values():
            x = np.array([[float(r['role']=='Visual'), float(r['role']=='Relation'), r['t'], r['relative_position'],
                           r['full_logp'], r['full_entropy']] for r in group])/scale
            y = np.array([r['signed_logp_loss'] for r in group])
            x -= x.mean(0); y -= y.mean()
            matrices.append(x.T@x); vectors.append(x.T@y); counts.append(len(group))
        xx, xy = np.stack(matrices), np.stack(vectors)
        total, target = xx.sum(0), xy.sum(0)
        assert np.linalg.matrix_rank(total) == len(names), 'Do not silently report rank-deficient controls'
        beta = np.linalg.solve(total, target)
        rng = np.random.default_rng(SEED)
        indices = rng.integers(0, len(groups), (REPS, len(groups)))
        weights = np.stack([(indices == i).sum(1) for i in range(len(groups))], -1)
        bx = np.einsum('bc,cij->bij', weights, xx); by = weights@xy
        boot = np.einsum('bij,bj->bi', np.linalg.pinv(bx, rcond=1e-10), by)
        for i, name in enumerate(names):
            lo, hi = bounds(boot[:, i])
            result.append(dict(subset=subset, coefficient=name, estimate_nats=float(beta[i]),
                ci95_low=lo, ci95_high=hi, tokens=len(eligible), images=len(groups),
                scale=float(scale[i]), interpretation='Image fixed-effect association; not causal role necessity'))
        diagnostics.append(dict(subset=subset, rank=int(np.linalg.matrix_rank(total)),
                                condition=float(np.linalg.cond(total)), images=len(groups), tokens=len(eligible)))
    csv_write(OUT/'word_role_controlled_associations.csv', result)
    write(OUT/'role_model_diagnostics.json', diagnostics)
    return result


def collect_B(p, manifest):
    samples = manifest['COCO']; refs = {s['question_id']:s['answer'] for s in samples}
    generations, records, scores, table, gates = [], {}, {}, [], []
    indices = np.random.default_rng(SEED).integers(0, 60, (REPS, 60))
    for condition in CONDITIONS:
        predictions, byq = {}, {}
        for s in samples:
            q = s['question_id']; key = q.replace(':','_')
            r = checkpoint(OUT/f'checkpoints/B/{key}/{condition}.json', p, s, 'B', condition)
            g = r['generation']; ids = g['token_ids']
            assert g['decode_steps'] == len(ids)-1 and g['eos'] == (ids[-1] == 2)
            assert len(r['gates']) == g['decode_steps'] and r['first_logits_exact']
            base = records['Full'][q]['generation'] if condition != 'Full' else g
            assert ids[0] == base['token_ids'][0]
            for row in r['gates']:
                assert row['on'] == gate(condition, s['image_sha256'], row['t'])
                gates.append(dict(question_id=q, image_sha256=s['image_sha256'], condition=condition,
                                  t=row['t'], on=row['on'], observed_predicted_token=ids[row['t']],
                                  consumed_token=ids[row['t']-1]))
            byq[q] = r; predictions[q] = g['text']
            triples = [tuple(ids[i:i+3]) for i in range(max(len(ids)-2,0))]
            generations.append(dict(question_id=q, image_sha256=s['image_sha256'], condition=condition,
                text=g['text'], token_ids=json.dumps(ids), non_eos_tokens=sum(t != 2 for t in ids),
                decode_steps=g['decode_steps'], visual_open_steps=g['visual_open_steps'],
                actual_open_fraction=g['visual_open_fraction'], eos=g['eos'], truncated=g['truncated'],
                empty=not bool(g['text']), repeated_trigram_count=len(triples)-len(set(triples)),
                first_divergence=first_difference(ids,base['token_ids']),
                text_changed=g['text'] != base['text'], historical_reused=bool(r['historical_binding'])))
        records[condition] = byq
        binding = digest(dict(refs=refs, predictions=predictions, scorer=p['scorer_source']))
        path = OUT/f'scores/{condition}.json'
        if path.exists():
            cached = read(path); assert cached['binding'] == binding
            overall, per = cached['overall'], cached['per_image']
        else:
            hp = HROOT/f'results/coco_decode_visual_regions_60/scores/{"Decode None" if condition == "None" else condition}.json'
            if condition in ['Full','None'] and hp.exists() and read(hp)['binding'] == binding:
                cached = read(hp); overall, per = cached['overall'], cached['per_image']
            else:
                overall, per = caption_scores(refs,predictions)
            write(path, dict(binding=binding, overall=overall, per_image=per))
        scores[condition] = per
        group = [r for r in generations if r['condition'] == condition]
        row = dict(condition=condition, images=60, **overall,
            delta_CIDEr_vs_Full=overall['CIDEr']-table[0]['CIDEr'] if table else 0.,
            delta_SPICE_vs_Full=overall['SPICE']-table[0]['SPICE'] if table else 0.,
            mean_actual_open_fraction=float(np.mean([r['actual_open_fraction'] for r in group])),
            pooled_actual_open_fraction=sum(r['visual_open_steps'] for r in group)/sum(r['decode_steps'] for r in group),
            mean_non_eos_tokens=float(np.mean([r['non_eos_tokens'] for r in group])),
            eos=sum(r['eos'] for r in group), truncated=sum(r['truncated'] for r in group),
            empty=sum(r['empty'] for r in group), repeated_trigram_outputs=sum(r['repeated_trigram_count'] > 0 for r in group),
            sequence_changed=sum(r['first_divergence'] >= 0 for r in group), text_changed=sum(r['text_changed'] for r in group))
        for m in METRICS:
            a = np.array([per[s['question_id']][m] for s in samples])
            f = np.array([scores['Full'][s['question_id']][m] for s in samples])
            row[m+'_ci_low'], row[m+'_ci_high'] = bounds(a[indices].mean(1))
            _, row['delta_'+m+'_ci_low'], row['delta_'+m+'_ci_high'] = paired(a,f,indices)
        table.append(row)
        print(condition, overall, flush=True)
    comparisons = [(c,'Full') for c in CONDITIONS[1:]]+[(c,'None') for c in CONDITIONS[2:]]
    comparisons += [('Periodic50','Periodic25')]+[(f'Random50-{seed}',f'Random25-{seed}') for seed in [20261102,20261103]]
    pairs = []
    for a,b in comparisons:
        for m in METRICS:
            aa = [scores[a][s['question_id']][m] for s in samples]
            bb = [scores[b][s['question_id']][m] for s in samples]
            delta, lo, hi = paired(aa,bb,indices)
            pairs.append(dict(condition=a, reference=b, metric=m, delta=delta, ci95_low=lo, ci95_high=hi,
                images=60, repetitions=REPS, images_improved=sum(x > y for x,y in zip(aa,bb)),
                images_worsened=sum(x < y for x,y in zip(aa,bb)), images_tied=sum(x == y for x,y in zip(aa,bb))))
    for row in generations:
        row.update(scores[row['condition']][row['question_id']])
    csv_write(OUT/'caption_quality_summary.csv', table)
    csv_write(OUT/'paired_comparisons.csv', pairs)
    csv_write(OUT/'per_image_scores.csv', generations)
    csv_write(OUT/'random_gate_records.csv', gates)
    return table, generations, scores, records, pairs


def save_figure(fig, number):
    for ext in ['pdf','png']: fig.savefig(OUT/f'figure_{number}.{ext}', dpi=260, bbox_inches='tight')
    plt.close(fig)


def figures(tokens, concentrations, summary, table, generations, scores, records, manifest):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'ps.fonttype':42})
    content = [r for r in tokens if r['dataset']=='COCO' and r['stage']=='Content Decode']
    fig, ax = plt.subplots(1,3,figsize=(12,3.3), layout='constrained')
    loss = [r['signed_logp_loss'] for r in content]
    ax[0].hist(loss,bins=45,color='#367d91'); ax[0].axvline(0,color='black',lw=.7)
    ax[0].set_xlabel('Signed logP loss (nat)'); ax[0].set_ylabel('Native content positions')
    binned = []
    for b in range(5):
        rows = [r for r in content if min(4,int(r['relative_position']*5)) == b]
        for metric in ['signed_logp_loss', 'top1_flip']:
            if rows: binned.append(dict(bin=b, metric=metric, **cluster_summary(rows,metric)))
    for axis, metric, label in [(ax[1],'signed_logp_loss','Mean signed logP loss (nat)'),
                                (ax[2],'top1_flip','Top-1 flip rate')]:
        rr = [r for r in binned if r['metric']==metric]
        y = np.array([r['mean'] for r in rr]); lo = np.array([r['ci_low'] for r in rr]); hi = np.array([r['ci_high'] for r in rr])
        axis.errorbar([.1+.2*r['bin'] for r in rr], y, yerr=[y-lo,hi-y], fmt='o-', color='#367d91',capsize=3)
        axis.set_xlabel('Relative native generation position'); axis.set_ylabel(label); axis.grid(alpha=.15)
    fig.suptitle('COCO: single-step ImageOFF on native prefixes (non-EOS Decode)')
    csv_write(OUT/'position_bin_summary.csv', binned); save_figure(fig,1)
    cc = sorted([r for r in concentrations if r['top1_share'] is not None],key=lambda r:r['top20_share'],reverse=True)
    fig, ax = plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    ax[0].bar(range(len(cc)),[r['top1_share'] for r in cc], label='Largest position',color='#366a92')
    ax[0].bar(range(len(cc)),[r['top20_share']-r['top1_share'] for r in cc],bottom=[r['top1_share'] for r in cc],label='Other top 20%',color='#4ca89a')
    ax[0].bar(range(len(cc)),[r['rest_share'] for r in cc],bottom=[r['top20_share'] for r in cc],label='Remaining positions',color='#c9cecf')
    ax[0].set_xlabel('Images ordered by top-20% concentration'); ax[0].set_ylabel('Share of positive single-step logP loss')
    ax[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.13),ncol=3,frameon=False); ax[0].set_ylim(0,1)
    ax[1].scatter([r['content_decode_positions'] for r in cc],[r['top20_share'] for r in cc],color='#4d8b79',label='Top 20%')
    ax[1].scatter([r['content_decode_positions'] for r in cc],[r['top1_share'] for r in cc],color='#366a92',marker='x',label='Top 1')
    ax[1].set_xlabel('Non-EOS Decode positions'); ax[1].set_ylabel('Concentration share'); ax[1].legend(fontsize=8)
    fig.suptitle('COCO: concentration within each native caption'); save_figure(fig,2)
    groups = [('GQA','ALL','P-1'),('GQA','CompareAttr','P-1'),('COCO','COCO','P-1'),
              ('COCO','COCO','Content Decode'),('COCO','COCO','EOS'),('GQA','ALL','Content Decode'),('GQA','ALL','EOS')]
    names = ['GQA\nP-1 (120)','CompareAttr\nP-1 (20)','COCO\nP-1 (60)',
             'COCO\ncontent (891)','COCO\nEOS (60)','GQA\ncontent (23)','GQA\nEOS (120)']
    fig,ax = plt.subplots(1,2,figsize=(12,3.8),layout='constrained')
    for axis,metric,ylabel in [(ax[0],'signed_logp_loss','Mean signed logP loss (nat)'),(ax[1],'top1_flip','Top-1 flip rate')]:
        rr = [next(r for r in summary if (r['dataset'],r['task'],r['stage'],r['metric'])==(*g,metric)) for g in groups]
        means=np.array([r['mean'] for r in rr]); low=np.array([r['ci_low'] for r in rr]); high=np.array([r['ci_high'] for r in rr])
        axis.errorbar(range(len(rr)),means,yerr=[means-low,high-means],fmt='o',capsize=4,color='#367d91')
        axis.set_xticks(range(len(rr)),names,fontsize=8); axis.set_ylabel(ylabel); axis.axhline(0,color='black',lw=.6)
    fig.suptitle('Matched causal metric; different tasks and generation stages'); save_figure(fig,3)
    if table is None:
        return
    fig,ax = plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    colors = {'Full':'#333333','None':'#777777','Periodic25':'#167d9a','Periodic50':'#167d9a',
        'Random25-20261102':'#ce7043','Random50-20261102':'#ce7043',
        'Random25-20261103':'#55935d','Random50-20261103':'#55935d'}
    for axis,m in zip(ax,METRICS):
        for r in table:
            c=r['condition']; x=r['mean_actual_open_fraction']*100; y=r['delta_'+m+'_vs_Full']
            marker='o' if c.startswith('Periodic') else 's' if c.endswith('20261102') else '^' if c.endswith('20261103') else 'D'
            label=c.replace('20261102','seed 1').replace('20261103','seed 2')
            axis.errorbar(x,y,yerr=[[y-r['delta_'+m+'_ci_low']],[r['delta_'+m+'_ci_high']-y]],fmt=marker,color=colors[c],capsize=3,label=label)
        axis.set_xlabel('Actual visual-open Decode fraction (%)'); axis.set_ylabel('Delta '+m+' vs Full'); axis.grid(alpha=.15)
        axis.axhline(0,color='black',lw=.7)
        axis.set_xlim(-4,104)
    ax[1].legend(fontsize=6.8,loc='best',ncol=2)
    fig.suptitle('Temporal re-access budget vs. official caption quality'); save_figure(fig,4)
    case_figures(manifest, records, scores)


def case_figures(manifest, records, scores):
    samples=manifest['COCO']
    ranked=sorted(samples,key=lambda s:scores['Periodic25'][s['question_id']]['CIDEr']-scores['None'][s['question_id']]['CIDEr'])
    selected=ranked[:3]+ranked[-3:]
    cases=[]
    with PdfPages(OUT/'figure_5.pdf') as pdf:
        for i,s in enumerate(selected):
            q=s['question_id']; fig=plt.figure(figsize=(13,9))
            ax=fig.add_axes([.025,.38,.33,.57]); ax.imshow(Image.open(s['image_path'])); ax.axis('off')
            delta=scores['Periodic25'][q]['CIDEr']-scores['None'][q]['CIDEr']
            ax.set_title(f'{q}\nPeriodic25 - None CIDEr: {delta:+.3f}',fontsize=11)
            ref=fig.add_axes([.03,.03,.32,.32]); ref.axis('off')
            ref.text(0,1,'Official references\n\n'+'\n\n'.join(textwrap.fill(t,48) for t in s['answer']),va='top',fontsize=8)
            ax=fig.add_axes([.39,.04,.59,.91]); ax.axis('off')
            base=records['Full'][q]['generation']
            for j,c in enumerate(CONDITIONS):
                g=records[c][q]['generation']; d=first_difference(g['token_ids'],base['token_ids'])
                title=f'{c}  |  CIDEr {scores[c][q]["CIDEr"]:.3f}  SPICE {scores[c][q]["SPICE"]:.3f}  |  first split t={d}'
                ax.text(0,1-j*.12,title,fontsize=9,weight='bold',va='top',color='#176c83' if c.startswith('Periodic') else '#222222')
                ax.text(0,.97-j*.12,textwrap.fill(g['text'],98),fontsize=9,va='top')
                cases.append(dict(question_id=q,condition=c,delta_periodic25_vs_none_CIDEr=delta,
                    first_divergence=d,caption=g['text'],selection='Post-hoc illustration: bottom3/top3 Periodic25-None CIDEr; not additional hypothesis tests'))
            pdf.savefig(fig,bbox_inches='tight')
            fig.savefig(OUT/f'figure_5_case_{i+1}.png',dpi=190,bbox_inches='tight')
            if i==0: fig.savefig(OUT/'figure_5.png',dpi=240,bbox_inches='tight')
            plt.close(fig)
    csv_write(OUT/'caption_difference_cases.csv',cases)


def report(p, summary, concentrations, table, pairs, roles, baseline, tokens):
    def lookup(dataset,task,stage,metric):
        return next(r for r in summary if (r['dataset'],r['task'],r['stage'],r['metric'])==(dataset,task,stage,metric))
    c=lookup('COCO','COCO','Content Decode','top1_flip')
    valid=[r for r in concentrations if r['top1_share'] is not None]
    g=lookup('GQA','ALL','P-1','signed_logp_loss'); ca=lookup('GQA','CompareAttr','P-1','signed_logp_loss')
    cp=lookup('COCO','COCO','P-1','signed_logp_loss')
    cd=lookup('COCO','COCO','Content Decode','signed_logp_loss')
    ce=lookup('COCO','COCO','EOS','signed_logp_loss')
    concentration_summary=[]
    for metric in ['top1_share','top20_share','rest_share']:
        vv=[dict(image_sha256=r['image_sha256'],value=r[metric]) for r in valid]
        concentration_summary.append(dict(metric=metric,**cluster_summary(vv,'value')))
    csv_write(OUT/'concentration_summary.csv',concentration_summary)
    large=[r for r in valid if r['first_large_relative'] is not None]
    multi=sum(r['dataset']=='GQA' and r['non_eos_tokens']>=2 for r in baseline)
    lines=['# GQA-COCO Gap：逐Token视觉必要性与时间稀疏重访','',
        '## 核心结果','',
        f'- COCO非EOS增量Decode：{c["tokens"]}个位置、{c["images"]}张图；Top-1翻转率{c["mean"]*100:.2f}%（图片簇95%CI [{c["ci_low"]*100:.2f}, {c["ci_high"]*100:.2f}]%）。',
        f'- COCO平均每图最大单位置贡献正向单步损害的{np.mean([r["top1_share"] for r in valid])*100:.2f}%；前20%位置贡献{np.mean([r["top20_share"] for r in valid])*100:.2f}%。分母为0的图不定义比例：{60-len(valid)}张，保留其原始记录。',
        f'- {sum(r["top1_flips"]>0 for r in concentrations)}/60张图至少一个内容位置翻转，{sum(r["top1_flips"]>=2 for r in concentrations)}/60张至少两个；每图内容位置翻转数中位数{np.median([r["top1_flips"] for r in concentrations]):.1f}。非EOS内容loss有{sum(r["signed_logp_loss"]<0 for r in tokens if r["dataset"]=="COCO" and r["stage"]=="Content Decode")}个负值，未截为0。',
        f'- 首次达到预设0.5 nat损害的图为{len(large)}/60；相对位置中位数{np.median([r["first_large_relative"] for r in large]) if large else float("nan"):.3f}，未达到阈值不当作0。EOS另列。',
        f'- GQA首Token平均signed loss={g["mean"]:.4f} nat；CompareAttr={ca["mean"]:.4f}；COCO P-1={cp["mean"]:.4f}、内容Decode={cd["mean"]:.4f}、EOS={ce["mean"]:.4f}。',
        f'- GQA120的Full多Token答案{multi}/120；与COCO沿同一Full前缀、同一个固定Full预测Token比较，未把首Token损害称为Gold Loss。','',
        '| 条件 | CIDEr | ΔFull [95%CI] | SPICE | ΔFull [95%CI] | 实际开放步比例 |',
        '|---|---:|---|---:|---|---:|']
    for r in table:
        lines.append(f'|{r["condition"]}|{r["CIDEr"]:.4f}|{r["delta_CIDEr_vs_Full"]:+.4f} [{r["delta_CIDEr_ci_low"]:+.4f}, {r["delta_CIDEr_ci_high"]:+.4f}]|{r["SPICE"]:.4f}|{r["delta_SPICE_vs_Full"]:+.4f} [{r["delta_SPICE_ci_low"]:+.4f}, {r["delta_SPICE_ci_high"]:+.4f}]|{r["mean_actual_open_fraction"]*100:.2f}%|')
    lines+=['','## 时间预算的配对比较','', '|比较|指标|差值|图片簇95%CI|改善/退化/相同图片数|','|---|---|---:|---|---|']
    for r in pairs:
        if r['reference']!='Full':
            lines.append(f'|{r["condition"]} − {r["reference"]}|{r["metric"]}|{r["delta"]:+.4f}|[{r["ci95_low"]:+.4f}, {r["ci95_high"]:+.4f}]|{r["images_improved"]}/{r["images_worsened"]}/{r["images_tied"]}|')
    lines+=['','## 词类、位置与置信度','',
        '沿用冻结离线词表，Visual只是词类候选，不表示已验证视觉事实。原始CSV保留多子词、Unknown、Punctuation与EOS。主词类分析只用首子词，所有子词作为补充；不把子词视为独立图片。',
        '图像固定效应OLS同时控制生成位置、相对位置、Full LogP和Entropy；控制量标准化，词类系数保持nat。20,000次图片整簇Bootstrap，不作因果词类推断。',
        '|子词口径|词类对比|signed loss差值|95%CI|','|---|---|---:|---|']
    for r in roles:
        if ' vs ' in r['coefficient']:
            lines.append(f'|{r["subset"]}|{r["coefficient"]}|{r["estimate_nats"]:+.4f}|[{r["ci95_low"]:+.4f}, {r["ci95_high"]:+.4f}]|')
    lines+=['','## H1与H2的解释边界','',
        'H1（少数关键位置）与H2（多位置持续需求）不是互斥的二元标签。集中度衡量原生前缀下正向损害是否长尾，时间稀疏自由生成则衡量固定、非自适应开放规则的累积输出质量。两者须共同解读，不能把单步损害相加解释为CIDEr下降。',
        '单步结果支持“损害集中于较少位置”的长尾成分，但不支持“每图仅一次决策即可解释全部视觉需求”的强版本：绝大多数图存在多个受影响内容位置，且图像需求并不限于EOS。尚未通过专门的自由延续对照隔离某一次错误的连锁传播，因此不将传播机制当作已证实。',
        '周期/随机门控不识别关键位置。低预算质量恢复不能证明存在可靠在线触发器；低预算失败也不能单独排除未测试的oracle关键位置规则。本轮不添加oracle或额外条件。',
        '详见全部真实负值、极端点、位置分布、每图集中度及每图质量结果；不能只用极端案例解释整体任务差异。',
        '', '## 数据、实现及验证','',
        'COCO严格复用历史60张图与每图五条官方参考。GQA按冻结输入哈希在六任务各选20题，120个互异图片SHA，不参考输出；属于历史探索数据而非独立确认。',
        '实验A每一步从同一未干预父Cache深复制为Full和ImageOFF分支，OFF Cache立即丢弃，只有Full结果推进原生轨迹。Smoke逐步检查父Cache完整SHA不变，并与独立原生完整轨迹的全部有效Logits及生成Token精确一致。t0只改P-1，非P-1状态/Logits/Attention哈希不变。',
        '实验B全部Prefill和首Token原生，所有新条件每图首Token完整词表Logits与本轮Full严格相同。原生Full/None历史结果在Smoke中校验完整序列与首Logits，所有60图Full轨迹也逐图与历史绑定一致。每一个新Decode步骤均检查真实绝对位置、Cache长度及32层实际概率，OFF为精确0，ON保持正视觉概率。',
        'CPU五项测试与三题GPU Smoke通过；无Hook残留、输入和权重版本改变。异常/NaN/Inf立即失败，不放宽容差、不静默跳样。',
        '另有三项CPU统计回归测试。首次CPU汇总触发NumPy不支持bool百分位减法的错误，已仅将分位数输入显式转为float；失败日志保留。未修改运行中的GPU源码、任何模型结果或容差。',
        '官方CIDEr/SPICE原始量纲；按图片整簇20,000次配对Bootstrap。CIDEr逐图分数条件于原60图参考语料的固定IDF，重采样不重估IDF。普通探索性95%区间，不作非劣效或多重检验确认性声明。',
        '无翻转样本的经验Bootstrap可退化为[0,0]，这不是总体安全保证。图3括号为预测位置数，不是独立图片数。实际开放步比例包含用于预测EOS的真实Decode步骤，不能误称内容Token开放比例。',
        '直接Image Key Hard Mask会重新归一化其余合法Key；Image Token/KV、原生Prefill、非视觉状态及间接视觉信息仍存在。不是物理KV删除或实际加速。',
        '', '## 质量异常与耗时','',
        '|条件|EOS|截断|空答案|重复三元Token输出|完整序列改变|','|---|---:|---:|---:|---:|---:|']
    for r in table: lines.append(f'|{r["condition"]}|{r["eos"]}|{r["truncated"]}|{r["empty"]}|{r["repeated_trigram_outputs"]}|{r["sequence_changed"]}|')
    done=read(OUT/'completion.json')
    lines += ['', f'GPU1执行（模型加载、Smoke、A+B正式采集）：{done["seconds"]/60:.2f}分钟。峰值allocated {done["peak"]["peak_allocated_mib"]:.0f} MiB，reserved {done["peak"]["peak_reserved_mib"]:.0f} MiB。',
        '实验A的Full与ImageOFF全词表有效Logits保存于独立资源目录，检查点记录绝对路径和SHA，不提交大数组到GitHub；不保存大型KV Cache。GitHub保留全部逐Token标量、Token ID、审计、图表和绑定，可查验/重新计算。',
        '', '## 图表及产物','',
        '- Figure 1：所有COCO非EOS Decode单步loss分布及五个位置区间的均值/翻转率，未隐藏负值和极端点。',
        '- Figure 2：每图正向单步损害集中程度，最大1个、前20%及其余位置。',
        '- Figure 3：GQA（含CompareAttr）与COCO按P-1、内容Decode、EOS分开比较。',
        '- Figure 4：实际视觉开放预算和相对Full质量；两随机种子分别展示，区间为图片簇配对ΔFull区间，绝对得分及两两配对区间见CSV。',
        '- Figure 5：6页原图/八条件Caption，按Periodic25−None CIDEr最高与最低各三图事后选作正反示例；未作为新增推断样本。首次分叉t为零起始预测步，−1表示完全一致。',
        '', '完成后仅提交独立分支，不修改main，不追加任何实验。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


def main():
    started=time.time(); p=verify(); assert read(OUT/'completion.json')['passed']
    manifest=read(OUT/'sample_manifest.json')
    tokens, concentration, baseline, a_records, summary=collect_A(p,manifest)
    roles=controlled_roles(tokens)
    table,generations,scores,records,pairs=collect_B(p,manifest)
    figures(tokens,concentration,summary,table,generations,scores,records,manifest)
    report(p,summary,concentration,table,pairs,roles,baseline,tokens)
    write(OUT/'validation.json',dict(passed=True, experiment_A_questions=180, experiment_A_token_positions=len(tokens),
        COCO_images=60,GQA_questions=120,GQA_image_clusters=len({s['image_sha256'] for s in manifest['GQA']}),
        B_conditions=8,new_B_generations=360,historical_B_generations=120,
        all_checkpoints_complete=True,source_hashes_unchanged=verify()['sources']==p['sources'],
        cpu_tests=read(OUT/'cpu_validation.json')['tests'],GPU_smoke_passed=read(OUT/'gpu_smoke.json')['passed'],
        first_tokens_exact=True, masked_actual_probabilities_exact_zero=True,
        nan_inf_oom=False,analysis_seconds=time.time()-started,
        analysis_source_sha256=sha256(__file__)))
    print('Analysis complete',flush=True)


if __name__=='__main__': main()
