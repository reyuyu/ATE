"""Complete33-condition scoring, image-paired intervals and unsmoothed figures."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import re
import textwrap
import time
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from benchmarks.dataset_registry import digest,freeze_json,sha256
from benchmarks.scoring import textvqa_score
from paper_rq1.scorers import caption_scores
from paper_rq1_token_role_decode.labels import align
from paper_rq1_single_layer_gqa6.analyze import descriptive_correlations
from .protocol import ROOT,OUT,MODEL_PATH,PROTOCOL,CONDITIONS,condition_layer
from .statistics import bootstrap,interval_summary
from .run import verify,checkpoint_valid


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def csv_read(path):
    with path.open() as f:return list(csv.DictReader(f))


def dataset_rows():
    config=verify();binding=digest(config)
    environments=[];all_attempt_environments=[]
    for rank in range(3):
        status=json.loads((OUT/f'generation/shard{rank}/completion.json').read_text())
        if status['binding']!=binding or status['status']!='complete':raise ValueError('Require all three complete frozen workers')
        paths=sorted((OUT/f'generation/shard{rank}').glob('environment_*.json'));path=paths[-1]
        all_attempt_environments.extend(json.loads(p.read_text()) for p in paths)
        environments.append(json.loads(path.read_text()))
    for field in ['python','torch','transformers','cuda_runtime','cudnn','gpu','model_revision','model_path','parameter_dtypes','binding','code_commit']:
        if len({digest(e[field]) for e in all_attempt_environments})!=1:raise ValueError('Worker/resume environment mismatch:'+field)
    freeze_json(OUT/'environment.json',dict(worker_environments=environments,common_environment_exact=True,all_attempts_checked=len(all_attempt_environments)))
    manifest=json.loads((OUT/'sample_manifest.json').read_text());data={};seen=set();audit=[]
    for s in manifest['samples']:
        key=(s['dataset'],s['question_id'])
        if key in seen:raise ValueError('Duplicate sample')
        seen.add(key);values={}
        folder=OUT/f"generation/shard{s['rank']}/checkpoints"/(s['dataset']+'__'+s['question_id'])
        for c in CONDITIONS:
            p=folder/(c+'.json')
            if not checkpoint_valid(p,binding,s,c):raise ValueError('Missing prescribed condition')
            values[c]=json.loads(p.read_text())['generation']
        if len({v['token_ids'][0] for v in values.values()})!=1:raise ValueError('First-token invariant failed')
        if not (folder/'audit.json').is_file():raise ValueError('Missing parent cache/input audit')
        data[key]=values;audit.append(dict(dataset=key[0],question_id=key[1],conditions=len(values),first_token=values['Full']['token_ids'][0]))
    for unit in ['coco_caption','textvqa']:
        if sum(k[0]==unit for k in data)!=200:raise ValueError('Wrong frozen sample coverage')
    expected=400*33;actual=len(list((OUT/'generation').glob('shard*/checkpoints/*/L*-off.json')))+len(list((OUT/'generation').glob('shard*/checkpoints/*/Full.json')))
    if actual!=expected:raise ValueError('Unexpected duplicate/extra conditions')
    freeze_json(OUT/'generation_coverage.json',dict(passed=True,images=400,conditions=33,generations=actual,samples=audit))
    return manifest,data


def score():
    manifest,data=dataset_rows();coco=[s for s in manifest['samples'] if s['dataset']=='coco_caption'];scores={};scorer_dir=OUT/'official_scores';scorer_dir.mkdir(exist_ok=True)
    references={s['question_id']:s['answer'] for s in coco}
    for condition in CONDITIONS:
        predictions={s['question_id']:data[('coco_caption',s['question_id'])][condition]['text'] for s in coco}
        binding=digest(dict(references=references,predictions=predictions,scorer='pycocoevalcap1.2 fixed200-IDF'))
        path=scorer_dir/(condition+'.json')
        if path.is_file():
            saved=json.loads(path.read_text())
            if saved['binding']!=binding:raise ValueError('Changed official score inputs')
        else:
            total,per=caption_scores(references,predictions);saved=dict(binding=binding,corpus=total,per_image=per)
            freeze_json(path,saved)
        scores[condition]=saved['per_image'];print('Official COCO scoring',condition,saved['corpus'],flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,use_fast=True)
    full=[];changed=[]
    for s in manifest['samples']:
        key=(s['dataset'],s['question_id']);base=data[key]['Full'];annotation=align(tokenizer,base['token_ids'])
        for condition in CONDITIONS:
            value=data[key][condition];first_diff=next((i for i,(a,b) in enumerate(zip(base['token_ids'],value['token_ids'])) if a!=b),None)
            if first_diff is None and len(base['token_ids'])!=len(value['token_ids']):first_diff=min(len(base['token_ids']),len(value['token_ids']))
            if first_diff==0:raise AssertionError('Decode-only changed first token')
            original=annotation['tokens'][first_diff] if first_diff is not None and first_diff<len(annotation['tokens']) else {}
            actual=value['token_ids'][first_diff] if first_diff is not None and first_diff<len(value['token_ids']) else None
            row=dict(dataset=s['dataset'],question_id=s['question_id'],image_identity=s['image_identity'],image_sha256=s['image_sha256'],
                condition=condition,layer=condition_layer(condition),text=value['text'],token_ids=json.dumps(value['token_ids']),length=value['length'],
                content_tokens=value['length']-int(value['eos']),decode_steps=value['decode_steps'],eos=value['eos'],truncated=value['truncated'],
                full_text=base['text'],text_changed=value['text']!=base['text'],tokens_changed=value['token_ids']!=base['token_ids'],first_diff=first_diff,
                first_full_token_id=original.get('token_id'),first_full_token=original.get('decoded_piece'),first_full_word=original.get('lexical_word'),
                first_full_category=original.get('category'),first_changed_token_id=actual,
                first_changed_token=tokenizer.decode([actual],clean_up_tokenization_spaces=False) if actual is not None else None,
                full_content_tokens=base['length']-int(base['eos']),full_length_group='1' if base['length']-int(base['eos'])<=1 else '2-3' if base['length']-int(base['eos'])<=3 else '4+')
            if s['dataset']=='coco_caption':
                for metric in ['CIDEr','SPICE']:
                    row[metric]=scores[condition][s['question_id']][metric];row['full_'+metric]=scores['Full'][s['question_id']][metric]
                    row['delta_'+metric]=row[metric]-row['full_'+metric]
            else:
                row['soft_score']=float(textvqa_score(value['text'],s['answer']));row['full_soft_score']=float(textvqa_score(base['text'],s['answer']))
                row['delta_soft_score']=row['soft_score']-row['full_soft_score']
            (full if condition=='Full' else changed).append(row)
    csv_write(OUT/'full_generation_per_sample.csv',full);csv_write(OUT/'layerwise_generation_per_sample.csv',changed)
    print('Scored400 Full and12800 ablations',flush=True)


def aggregate():
    rows=csv_read(OUT/'layerwise_generation_per_sample.csv');summaries=[];all_boot=[];changes=[];lengths=[]
    for unit,metrics in [('coco_caption',['CIDEr','SPICE']),('textvqa',['TextVQA'])]:
        units=[r for r in rows if r['dataset']==unit];qids=sorted({r['question_id'] for r in units});by={(r['question_id'],int(r['layer'])):r for r in units}
        matrix=np.array([[float(by[(qid,k)]['delta_'+('soft_score' if m=='TextVQA' else m)]) for m in metrics for k in range(32)] for qid in qids])
        images=[by[(q,0)]['image_identity'] for q in qids]
        estimate=bootstrap(matrix,images)
        for k in range(32):
            selected=[by[(qid,k)] for qid in qids]
            summary=dict(dataset=unit,layer=k,n=len(selected),text_changed=sum(r['text_changed']=='True' for r in selected),
                token_changed=sum(r['tokens_changed']=='True' for r in selected),mean_length=np.mean([int(r['length']) for r in selected]),
                eos=sum(r['eos']=='True' for r in selected),truncated=sum(r['truncated']=='True' for r in selected))
            for im,metric in enumerate(metrics):
                col=im*32+k;sc='soft_score' if metric=='TextVQA' else metric
                delta=np.array([float(r['delta_'+sc]) for r in selected]);bound=interval_summary(estimate['mean'][col],estimate['ordinary'][:,col],estimate['corrected'][:,col],PROTOCOL['statistics']['margins'][metric])
                bound.update(dataset=unit,layer=k,metric=metric,n=len(selected),images=estimate['clusters'],repetitions=estimate['repetitions'],family=96,
                    expected_extreme_tail_draws=estimate['expected_extreme_tail_draws'],full_score=np.mean([float(r['full_'+sc]) for r in selected]),
                    ablated_score=np.mean([float(r[sc]) for r in selected]),improved=int(sum(delta>1e-12)),harmed=int(sum(delta< -1e-12)),unchanged=int(sum(np.abs(delta)<=1e-12)),
                    mean_absolute_change=float(np.mean(np.abs(delta))),median_delta=float(np.median(delta)),p5=float(np.quantile(delta,.05)),p95=float(np.quantile(delta,.95)),worst_delta=float(delta.min()))
                all_boot.append(bound)
                summary.update({metric+'_'+key:value for key,value in bound.items() if key not in ['dataset','layer','n']})
                if unit=='textvqa':
                    for group in ['1','2-3','4+']:
                        sub=[r for r in selected if r['full_length_group']==group]
                        lengths.append(dict(layer=k,full_content_length_group=group,n=len(sub),full_score=np.mean([float(r['full_soft_score']) for r in sub]) if sub else '',
                            delta=np.mean([float(r['delta_soft_score']) for r in sub]) if sub else '',improved=sum(float(r['delta_soft_score'])>1e-12 for r in sub),
                            harmed=sum(float(r['delta_soft_score'])< -1e-12 for r in sub),text_changed=sum(r['text_changed']=='True' for r in sub)))
            summaries.append(summary)
            counts=Counter(r['first_full_category'] or 'NoTokenChange' for r in selected)
            changes.append(dict(dataset=unit,layer=k,n=len(selected),text_change_rate=summary['text_changed']/len(selected),token_change_rate=summary['token_changed']/len(selected),
                first_diff_mean=np.mean([int(r['first_diff']) for r in selected if r['first_diff']!='']) if any(r['first_diff']!='' for r in selected) else '',
                categories=json.dumps(dict(counts)),eos=summary['eos'],truncated=summary['truncated']))
    csv_write(OUT/'bootstrap_statistics.csv',all_boot)
    csv_write(OUT/'layerwise_coco_quality_summary.csv',[r for r in summaries if r['dataset']=='coco_caption'])
    csv_write(OUT/'layerwise_textvqa_score_summary.csv',[r for r in summaries if r['dataset']=='textvqa'])
    csv_write(OUT/'caption_change_analysis.csv',changes);csv_write(OUT/'textvqa_length_analysis.csv',lengths)
    csv_write(OUT/'quality_win_loss_cases.csv',sorted(rows,key=lambda r:(r['dataset'],float(r.get('delta_CIDEr') or r.get('delta_soft_score') or 0))))
    print('Paired bootstrap complete;200000 draws;96-comparison family',flush=True)


def savefig(fig,name):
    import matplotlib.pyplot as plt
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def layer_ticks(ax):
    positions=list(range(0,32,2))
    ax.set_xticks(positions,labels=[str(k) for k in positions])


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    stats=csv_read(OUT/'bootstrap_statistics.csv');changes=csv_read(OUT/'caption_change_analysis.csv');rows=csv_read(OUT/'layerwise_generation_per_sample.csv')
    colors={'CIDEr':'#176b91','SPICE':'#3a8157','TextVQA':'#a64a57'}
    for metric,name in [('CIDEr','figure1_coco_cider'),('SPICE','figure2_coco_spice'),('TextVQA','figure3_textvqa')]:
        values=sorted([r for r in stats if r['metric']==metric],key=lambda r:int(r['layer']));x=np.arange(32);scale=100 if metric=='TextVQA' else 1
        base=float(values[0]['full_score'])*scale;y=np.array([float(r['ablated_score']) for r in values])*scale
        lo=np.array([float(r['ci95_low']) for r in values])*scale+base;hi=np.array([float(r['ci95_high']) for r in values])*scale+base
        flo=np.array([float(r['family95_low']) for r in values])*scale+base;fhi=np.array([float(r['family95_high']) for r in values])*scale+base
        fig,ax=plt.subplots(figsize=(12,4.2));ax.vlines(x,flo,fhi,color=colors[metric],alpha=.18,lw=4,label='Bonferroni family96 intervals')
        ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt='o-',markersize=4,capsize=2,color=colors[metric],label='Single-layer Decode-off; paired95%CI')
        ax.axhline(base,color='black',ls='--',label='Full');ax.axhline(base-float(values[0]['ni_margin'])*scale,color='grey',ls=':',label='Frozen noninferiority margin')
        ax.set(xlabel='Ablated layer (normal Prefill; Decode-only)',ylabel=metric+(' soft score (%)' if metric=='TextVQA' else ' (raw)'),xlim=(-.5,31.5),title=('TextVQA200 questions /200 images' if metric=='TextVQA' else 'COCO200 images; fixed reference IDF'));layer_ticks(ax)
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.23));savefig(fig,name)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for metric,ax in zip(['CIDEr','SPICE','TextVQA'],axes):
        v=sorted([r for r in stats if r['metric']==metric],key=lambda r:int(r['layer']));scale=100 if metric=='TextVQA' else 1
        ax.plot(range(32),[float(r['delta'])*scale for r in v],'o-',color=colors[metric],markersize=3);ax.axhline(0,color='black',ls='--');ax.set(ylabel='Delta '+metric+(' (pp)' if metric=='TextVQA' else ' (raw)'));ax.grid(alpha=.2)
    axes[-1].set(xlabel='Ablated Decode Attention layer');layer_ticks(axes[-1]);fig.suptitle('Cross-task quality changes (different raw scales; not summed)');savefig(fig,'figure4_cross_task_sensitivity')
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True);v=sorted([r for r in changes if r['dataset']=='coco_caption'],key=lambda r:int(r['layer']))
    axes[0].plot(range(32),[float(r['text_change_rate'])*100 for r in v],'o-',color='#176b91');axes[0].set(ylabel='Caption text changed (%)')
    groups=[np.array([float(r['delta_CIDEr']) for r in rows if r['dataset']=='coco_caption' and int(r['layer'])==k]) for k in range(32)]
    axes[1].boxplot(groups,positions=range(32),widths=.6,showfliers=True,flierprops={'markersize':2,'alpha':.45});axes[1].axhline(0,color='black',ls='--');axes[1].set(ylabel='Per-image Delta CIDEr')
    positive=[int(sum(g>1e-12)) for g in groups];negative=[int(sum(g< -1e-12)) for g in groups]
    axes[2].bar(np.arange(32)-.2,positive,width=.4,label='Score up',color='#3a8157');axes[2].bar(np.arange(32)+.2,negative,width=.4,label='Score down',color='#a64a57');axes[2].set(ylabel='Images',xlabel='Ablated Decode Attention layer');axes[2].legend()
    for ax in axes:ax.grid(alpha=.15)
    layer_ticks(axes[-1]);fig.suptitle('Text changes are not equivalent to quality harm; all negative tails retained');savefig(fig,'figure5_caption_change_and_harm')
    case_figure(rows)


def choose_cases(rows):
    coco=[r for r in rows if r['dataset']=='coco_caption' and r['text_changed']=='True']
    if not coco:raise ValueError('No changed captions for required cases')
    same_words=[r for r in coco if re.findall(r'[a-z0-9]+',r.get('full_text','').lower())==re.findall(r'[a-z0-9]+',r.get('text','').lower()) and r.get('full_text') and r.get('text')]
    chosen=[('Largest automatic CIDEr decrease',min(coco,key=lambda r:float(r['delta_CIDEr']))),
        ('Largest automatic CIDEr increase',max(coco,key=lambda r:float(r['delta_CIDEr']))),
        ('Punctuation/case-only change (inspect)' if same_words else 'Changed text / near-equal automatic scores',min(same_words or coco,key=lambda r:abs(float(r['delta_CIDEr']))+abs(float(r['delta_SPICE']))))]
    used={(r['question_id'],r['layer']) for _,r in chosen}
    pool=[r for r in coco if (r['question_id'],r['layer']) not in used]
    truncated=[r for r in pool if r['truncated']=='True']
    chosen.append(('Length-cap / repetition candidate (inspect image)' if truncated else 'Semantic-risk candidate (requires image inspection)',
        min(truncated or pool,key=lambda r:float(r['delta_SPICE']))))
    return chosen


def case_figure(rows):
    import matplotlib.pyplot as plt
    manifest=json.loads((OUT/'sample_manifest.json').read_text());samples={s['question_id']:s for s in manifest['samples'] if s['dataset']=='coco_caption'}
    chosen=choose_cases(rows);csv_write(OUT/'selected_case_studies.csv',[dict(case=label,**r) for label,r in chosen])
    fig,axes=plt.subplots(4,2,figsize=(15,15),gridspec_kw={'width_ratios':[1,1.65]})
    for (label,r),(a,b) in zip(chosen,axes):
        with Image.open(samples[r['question_id']]['image_path']) as image:a.imshow(image.convert('RGB'))
        a.axis('off');a.set_title(r['question_id']+' | L'+r['layer']+'-off',fontsize=11)
        text=label+'\n\nFull: '+r['full_text']+'\n\nAblated: '+r['text']+'\n\nDelta CIDEr '+format(float(r['delta_CIDEr']),'+.4f')+'; Delta SPICE '+format(float(r['delta_SPICE']),'+.4f')+'\nFirst change t='+r['first_diff']+': '+repr(r['first_full_token'])+' -> '+repr(r['first_changed_token'])+' ('+r['first_full_category']+')'
        wrapped='\n'.join(textwrap.fill(line,92) if line else '' for line in text.split('\n'));b.text(0,1,wrapped,va='top',fontsize=10,linespacing=1.35);b.axis('off')
    fig.suptitle('Actual paired captions. Automatic score gain is not human-verified factual improvement.',fontsize=12);fig.tight_layout(rect=[0,0,1,.98]);savefig(fig,'figure6_actual_cases')


def history():
    records=[]
    paths=[ROOT/'results/paper_rq1_token_role_decode/per_token_layer_ablation.csv',ROOT/'results/paper_rq1_late_attention_sources/source_ablation_per_token.csv']
    for path in paths:
        if not path.exists():continue
        prior=csv_read(path);group=defaultdict(list)
        for r in prior:
            mode=r.get('mode','')
            if mode not in ['Whole','Whole-off']:continue
            layer=int(r.get('layer',-1));role=r.get('word_type','')
            loss=r.get('loss') or r.get('logp_loss') or r.get('full_token_logp_loss')
            if loss is None and r.get('delta_logp'):loss=-float(r['delta_logp'])
            if layer>=0 and loss is not None:group[(layer,role)].append(float(loss))
        for (layer,role),values in group.items():records.append(dict(source=str(path.relative_to(ROOT)),layer=layer,role=role,mean_loss=np.mean(values),n=len(values),source_sha256=sha256(path)))
    csv_write(OUT/'historical_single_step_comparison.csv',records)
    quality=csv_read(OUT/'bootstrap_statistics.csv');comparisons=[]
    for source in sorted({r['source'] for r in records}):
        for role in sorted({r['role'] for r in records if r['source']==source}):
            selected=[r for r in records if r['source']==source and r['role']==role]
            for metric in ['CIDEr','SPICE']:
                q={int(r['layer']):-float(r['delta']) for r in quality if r['metric']==metric}
                x=np.array([r['mean_loss'] for r in selected]);y=np.array([q[r['layer']] for r in selected])
                if len(x)>=3 and np.std(x)>0 and np.std(y)>0:
                    pearson,spearman=descriptive_correlations(x,y)
                    comparisons.append(dict(source=source,role=role,metric=metric,layers=len(x),pearson=pearson,spearman=spearman,
                        interpretation='Descriptive cross-layer correlation across DIFFERENT image cohorts/protocols; layers not independent images; no causal p-value.'))
    csv_write(OUT/'historical_quality_correlations.csv',comparisons)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['score','aggregate','figures','history','all']);args=parser.parse_args()
    started=time.perf_counter()
    if args.stage=='all':score();aggregate();figures();history()
    else:globals()[args.stage]()
    print('Analysis stage',args.stage,'seconds',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
