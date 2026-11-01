"""Publication and official scoring only; no model forwards or new conditions."""
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.dataset_registry import digest, sha256
from paper_rq1.scorers import caption_scores
from paper_rq1_single_layer_gqa6.run import atomic_json
from rq2_sparse_caption import (OUT, ROOT, KEYS, NEW, read, verify, historical,
                               checkpoint, layers, history)


def save(name, rows):
    assert rows
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def csv_read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def divergence(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),
                min(len(a),len(b)) if len(a)!=len(b) else -1)


def main():
    tick=time.time();p=verify();binding=digest(p)
    completion=read(OUT/'completion.json');assert completion['binding']==binding and completion['generations']==180
    smoke=read(OUT/'smoke.json');assert smoke['passed'] and smoke['protocol_sha256']==sha256(OUT/'protocol.json')
    samples=read(OUT/'sample_manifest.json')['samples'];rows=[];scores={};summary=[];audit={};parity={}
    refs={s['question_id']:s['answer'] for s in samples}
    for key in KEYS:
        pred={};changed=0;allrows=[]
        for s in samples:
            q=s['question_id'];base,first=historical(q,'A')
            if key in NEW:
                r=checkpoint(OUT/f"checkpoints/{q.replace(':','_')}/{key}.json",binding,s,key)
                g=r['generation'];allrows.extend(r['probability_rows'])
                with np.load(OUT/r['raw_file']) as raw:
                    exact=np.array_equal(raw['first_logits'],first.numpy())
                if key in ['DS','DN']:assert exact
                parity[f'{q}/{key}']=dict(first_logits_exact=exact,first_token_same=g['token_ids'][0]==base['token_ids'][0])
                assert g['prefill_blocked']==layers(key,32)[0] and g['decode_blocked']==layers(key,32)[1]
                assert len(r['probability_rows'])==32*len(g['token_ids'])
                assert r['hooks_clean'] and r['inputs_unchanged'] and r['parameters_unchanged']
                for z in r['probability_rows']:
                    assert z['blocked']==(z['layer'] in layers(key,32)[int(z['step']>0)])
                    assert z['image_probability_max']==0. if z['blocked'] else z['image_probability_max']>0.
            else:g,_=historical(q,key)
            ids=g['token_ids'];assert g['eos']==(ids[-1]==2) and len(ids)<=96
            assert g['decode_steps']==len(ids)-1
            diff=divergence(ids,base['token_ids']);changed+=diff!=-1
            pred[q]=g['text']
            triples=[tuple(ids[i:i+3]) for i in range(max(0,len(ids)-2))]
            rows.append(dict(question_id=q,image_sha256=s['image_sha256'],condition=key,
                provenance='historical_RQ2.6' if key in ['A','B'] else 'new_RQ2.7',
                text=g['text'],token_ids=json.dumps(ids),non_eos_length=sum(t!=2 for t in ids),
                decode_steps=g['decode_steps'],eos=g['eos'],truncated=g['truncated'],
                empty=not bool(g['text'].strip()),token_sequence_changed=diff!=-1,
                first_divergence_zero_based=diff,
                repeated_token_trigram_count=len(triples)-len(set(triples))))
        if key in ['A','B']:
            historical_scores=csv_read(history.OUT/f'caption_{key}_official_per_image.csv')
            per={r['question_id']:{m:float(r[m]) for m in ['CIDEr','SPICE']} for r in historical_scores}
            assert set(per)==set(refs)
            old=next(r for r in csv_read(history.OUT/'caption_summary.csv') if r['condition']==key)
            score={m:float(old[m]) for m in ['CIDEr','SPICE']}
        else:
            score,per=caption_scores(refs,pred)
            save(f'{key}_official_per_image.csv',[dict(question_id=q,**v) for q,v in per.items()])
        scores[key]=per
        rr=[r for r in rows if r['condition']==key]
        summary.append(dict(condition=key,provenance=rr[0]['provenance'],n=60,**score,
            delta_CIDEr_vs_Full=score['CIDEr']-summary[0]['CIDEr'] if summary else 0.,
            delta_SPICE_vs_Full=score['SPICE']-summary[0]['SPICE'] if summary else 0.,
            mean_non_eos_length=float(np.mean([r['non_eos_length'] for r in rr])),
            mean_decode_steps=float(np.mean([r['decode_steps'] for r in rr])),
            eos=sum(r['eos'] for r in rr),truncated=sum(r['truncated'] for r in rr),empty=sum(r['empty'] for r in rr),
            changed_sequences=changed,change_rate=changed/60,
            repeated_trigram_outputs=sum(r['repeated_token_trigram_count']>0 for r in rr)))
        audit[key]=dict(probability_rows=len(allrows),
            blocked_probability_max=max([r['image_probability_max'] for r in allrows if r['blocked']],default=0),
            all_steps_32_layers=True if key in NEW else 'historical_audited',
            unchanged_native_prefill_non_p1=smoke['checks'][key]['non_p1_prefill_logits_states_attention_exact'] if key in NEW else 'historical_audited')
    save('per_image_generations.csv',rows)
    save('accuracy_and_quality_summary.csv',summary)
    paired=[];per_image=[]
    rng=np.random.default_rng(p['bootstrap']['seed'])
    indices=rng.integers(0,60,(p['bootstrap']['repetitions'],60))
    for a,b in p['comparisons']:
        for metric in ['CIDEr','SPICE']:
            values=np.array([scores[a][s['question_id']][metric]-scores[b][s['question_id']][metric] for s in samples])
            lo,hi=np.quantile(values[indices].mean(1),[.025,.975])
            paired.append(dict(comparison=f'{a}-{b}',metric=metric,delta=float(values.mean()),
                ci95_low=float(lo),ci95_high=float(hi),n=60,clusters=60,
                improved=int((values>0).sum()),worsened=int((values<0).sum()),tied=int((values==0).sum()),
                status='descriptive; fixed full-corpus official per-image scores; no NI/equivalence inference'))
    for s in samples:
        q=s['question_id']
        for key in KEYS:
            per_image.append(dict(question_id=q,image_sha256=s['image_sha256'],condition=key,**scores[key][q],
                delta_CIDEr_vs_A=scores[key][q]['CIDEr']-scores['A'][q]['CIDEr'],
                delta_SPICE_vs_A=scores[key][q]['SPICE']-scores['A'][q]['SPICE']))
    save('paired_caption_scores.csv',per_image);save('paired_comparisons.csv',paired)
    atomic_json(OUT/'decode_visual_audit.json',dict(passed=True,conditions=audit))
    atomic_json(OUT/'first_token_parity_audit.json',dict(passed=True,DS_DN_all60_exact=True,checks=parity))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
    colors=['#777777','#aaaaaa','#be6d39','#267e67','#477cac']
    fig,axs=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
    for ax,m in zip(axs,['CIDEr','SPICE']):
        bars=ax.bar(range(5),[r[m] for r in summary],color=colors,width=.65)
        ax.set_xticks(range(5),['A\nFull [H]','B\nRelay [H]','S\nSparse [N]','DS\nDecode sparse [N]','DN\nNo vision [N]'],fontsize=8)
        ax.set_title(m);ax.set_ylim(0,max(r[m] for r in summary)*1.18)
        ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Caption Quality (60 paired COCO images)',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(OUT/f'figure_a_caption_quality.{ext}',dpi=240)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7,3),layout='constrained')
    bykey={r['condition']:r for r in summary}
    for ax,m in zip(axs,['CIDEr','SPICE']):
        yy=[bykey[k][m] for k in ['DN','DS','A']]
        ax.plot([0,2,32],yy,'o-',color='#267e67',lw=1.6)
        for x,y,label in zip([0,2,32],yy,['DN','DS','A [H]']):
            ax.annotate(label,(x,y),xytext=(6,7),textcoords='offset points',fontsize=9)
        ax.set_xlim(-2,37);ax.set_xticks([0,2,32],['0','2','32']);ax.set_ylim(0,max(yy)*1.25)
        ax.set_xlabel('Decode layers with direct visual access');ax.set_ylabel(m);ax.grid(alpha=.15)
    fig.suptitle('Native Prefill; Decode Visual Access vs. Quality',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(OUT/f'figure_b_decode_visual_access.{ext}',dpi=240)
    plt.close(fig)
    save('figure_a_source.csv',summary)
    save('figure_b_source.csv',[dict(condition=k,decode_visual_layers=n,**{m:bykey[k][m] for m in ['CIDEr','SPICE']}) for k,n in [('DN',0),('DS',2),('A',32)]])
    lines=['# RQ2.7: Sparse Visual Access during Caption Decoding','',
        '|Condition|CIDEr|SPICE|Delta CIDEr vs Full|Delta SPICE vs Full|','|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['condition']}|{r['CIDEr']:.6f}|{r['SPICE']:.6f}|{r['delta_CIDEr_vs_Full']:+.6f}|{r['delta_SPICE_vs_Full']:+.6f}|")
    lines+=['','A/B strictly reused from RQ2.6 commit '+p['base']+'. S/DS/DN: exactly 180 new formal generations.',
        'Only 60 fixed images, descriptive paired image bootstrap. Official scorer raw scales, not multiplied by 100. BLEU was not in the existing audited scorer and is omitted.',
        'Paired intervals condition on the original 60-image corpus CIDEr document frequencies; no claim of population equivalence or noninferiority.',
        '', '|Comparison|Metric|Difference|Paired 95% CI|','|---|---|---:|---|']
    for r in paired:lines.append(f"|{r['comparison']}|{r['metric']}|{r['delta']:+.6f}|[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]|")
    lines+=['','## Validation and scope',
        'CPU regression and real GPU0 smoke passed: historical A/B complete sequences and first logits exact; native/no-op full logits exact; native HF token IDs exact. All non-P-1 Prefill attention/state hashes and logits exact. DS/DN first full-vocabulary logits exact on all 60 images.',
        'All actual masked attention probabilities are exactly zero. L14/L19 retained paths have positive actual mass. Every real Decode step checks cache length, absolute position, image mapping and all 32 layers. Independent native KV cache/prefix per condition; no weight/input mutation or leaked hooks.',
        'Token repetition count is a syntactic screening flag, not a semantic hallucination or quality metric. No new hallucination scorer. No physical KV pruning, skipped layer computation or speedup claim.',
        f"Formal GPU generation: {completion['formal_seconds']/60:.2f} min; model load: {completion['model_load_seconds']:.1f} s; smoke: {completion['smoke_seconds']:.1f} s.",
        f"CPU official scoring/analysis/figures: {time.time()-tick:.1f} s. End-to-end tracked time from preparation start: {(time.time()-p['recorded_start_epoch'])/60:.2f} min (excludes earlier attachment reading/code implementation and later Git push).",'']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    atomic_json(OUT/'validation.json',dict(passed=True,images=60,new_formal_generations=180,
        historical_generations_reused=120,cpu_tests=read(OUT/'cpu_validation.json')['tests'],
        checkpoints=180,first_token_DS_DN_exact=120,scoring_analysis_seconds=time.time()-tick,
        all_conditions_eos={r['condition']:r['eos'] for r in summary},
        empty={r['condition']:r['empty'] for r in summary},truncated={r['condition']:r['truncated'] for r in summary},
        source_unchanged=verify()['source']==p['source']))
    print(json.dumps(dict(summary=summary,paired=paired),indent=2),flush=True)


if __name__=='__main__':main()
