"""CPU-only official scoring, paired image statistics, and scientific figures."""
import csv
import difflib
import json
from pathlib import Path
import subprocess
import textwrap
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from benchmarks.dataset_registry import digest, sha256
from knockout_probe import write_csv
from paper_rq1.scorers import caption_scores, gqa_prediction
from vlm_common import write_json
from .engine import CONDITIONS, REGIONS
from .run import OUT, ROOT, read, verify, uid, validate_checkpoint


def ci(values, images, seed=20261102, repetitions=20000):
    """Image-cluster paired bootstrap; retain question-weighted point estimate."""
    values = np.asarray(values, dtype=float)
    unique = sorted(set(images)); mapping = {h:i for i,h in enumerate(unique)}
    sums = np.zeros(len(unique)); counts = np.zeros(len(unique))
    for value, image in zip(values, images):
        sums[mapping[image]] += value; counts[mapping[image]] += 1
    rng = np.random.default_rng(seed); draws = []
    for start in range(0, repetitions, 1000):
        ix = rng.integers(0,len(unique),(min(1000,repetitions-start),len(unique)))
        draws.extend((sums[ix].sum(1)/counts[ix].sum(1)).tolist())
    low, high = np.quantile(draws,[.025,.975])
    return float(low), float(high)


def divergence(a, b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),
                min(len(a),len(b)) if len(a)!=len(b) else -1)


def delta_words(before, after):
    removed, added = [], []
    a, b = before.split(), after.split()
    for code, x1, x2, y1, y2 in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
        if code in ['replace','delete']: removed.extend(a[x1:x2])
        if code in ['replace','insert']: added.extend(b[y1:y2])
    return ' '.join(removed), ' '.join(added)


def repetition_count(tokens):
    triples=[tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
    return len(triples)-len(set(triples))


def load(p):
    m = read(OUT/'sample_manifest.json'); samples = m['samples']; results = {}; audit=[]
    completions = [read(OUT/f'shard{i}/completion.json') for i in range(3)]
    assert sum(c['generations'] for c in completions) == 4620
    assert not list(OUT.glob('shard*/failure*.json'))
    for i,c in enumerate(completions):
        assert c['protocol_sha256'] == digest(p) and read(OUT/f'shard{i}/smoke.json')['passed']
    for s in samples:
        name = uid(s); rank = m['assignments'][name]; results[name] = {}
        for key in CONDITIONS:
            path=OUT/f"shard{rank}/checkpoints/{s.get('task','caption')}__{s['question_id'].replace(':','_')}/{key}.json"
            r=validate_checkpoint(path,s,key,p); g=r['generation']; a=r['audit']
            with np.load(path.with_suffix('.npz')) as raw: first=raw['first_logits'].copy()
            assert r['hooks_inputs_weights_unchanged'] and a['rows']==32*len(g['token_ids'])
            assert a['text_probability_max_blocked']==0 and a['attention_off_max_norm']==0
            assert a['min_image_mass']>0 and a['max_row_error']<=.002
            region=None if key=='Full' else key.split()[0]
            active_steps=len(g['token_ids'])-int(s['dataset']=='coco_caption')
            expected=0 if region is None else len(REGIONS[region])*active_steps
            assert a['masked_text_rows']==(expected if 'Text' in key else 0)
            assert a['attention_off_rows']==(expected if 'Attention' in key else 0)
            assert g['prefill_active']==(s['dataset']!='coco_caption')
            assert g['eos']==(g['token_ids'][-1]==2) and g['decode_steps']==len(g['token_ids'])-1
            assert len(g['token_ids'])<=(96 if s['dataset']=='coco_caption' else 32)
            assert np.isfinite(first).all()
            results[name][key]=dict(generation=g,first=first,rank=rank)
            audit.append(dict(uid=name,condition=key,**a))
        if s['dataset']=='coco_caption':
            assert all(np.array_equal(results[name]['Full']['first'],r['first']) for r in results[name].values())
    write_csv(OUT/'mask_and_projection_audit.csv',audit)
    return samples,results,completions


def gqa_statistics(samples,results,p):
    summary=[];tasks=[];paired=[];rows=[]
    images=[s['image_sha256'] for s in samples]
    scores={k:np.array([results[uid(s)][k]['generation']['score'] for s in samples]) for k in CONDITIONS}
    for k in CONDITIONS:
        y=scores[k]; full=scores['Full']; diff=100*(y-full)
        lo,hi=ci(diff,images)
        entry=dict(condition=k,n=len(samples),accuracy_percent=100*float(y.mean()),
            delta_pp=float(diff.mean()),ci95_low_pp=lo,ci95_high_pp=hi,
            correct_to_wrong=int(((full==1)&(y==0)).sum()),wrong_to_correct=int(((full==0)&(y==1)).sum()))
        rr=[]
        for s in samples:
            result=results[uid(s)][k];g=result['generation'];base=results[uid(s)]['Full'];b=base['generation']
            target=int(base['first'].argmax()); first_flip=int(result['first'].argmax())!=target
            # Exact FP32 full-vocabulary log_softmax, matching the generation studies.
            import torch
            loss=float(torch.log_softmax(torch.from_numpy(base['first']).float(),-1)[target]-
                       torch.log_softmax(torch.from_numpy(result['first']).float(),-1)[target])
            row=dict(question_id=s['question_id'],task=s['task'],image_id=s['image_id'],
                image_sha256=s['image_sha256'],condition=k,physical_gpu=p['gpus'][result['rank']]['physical_gpu'],
                input_sha256=digest(s['input_mapping']),sample_sha256=digest(s),
                gold=s['answer'],answer=g['text'],prediction=g['prediction'],score=g['score'],
                full_score=b['score'],first_token_flip=first_flip,full_first_token=target,
                first_token=int(result['first'].argmax()),full_top1_logp_loss=loss,
                generated_text_changed=g['text']!=b['text'],token_sequence_changed=g['token_ids']!=b['token_ids'],
                first_divergence_zero_based=divergence(g['token_ids'],b['token_ids']),
                token_ids=json.dumps(g['token_ids']),non_eos_length=sum(t!=2 for t in g['token_ids']),
                repeated_token_trigrams=repetition_count(g['token_ids']),
                eos=g['eos'],truncated=g['truncated'],empty=not bool(g['text']))
            rr.append(row);rows.append(row)
        entry.update(first_token_flip_count=sum(r['first_token_flip'] for r in rr),
                     generated_text_changed=sum(r['generated_text_changed'] for r in rr),
                     mean_non_eos_length=float(np.mean([r['non_eos_length'] for r in rr])),
                     mean_full_top1_logp_loss=float(np.mean([r['full_top1_logp_loss'] for r in rr])),
                     repeated_trigram_outputs=sum(r['repeated_token_trigrams']>0 for r in rr),
                     eos=sum(r['eos'] for r in rr),truncated=sum(r['truncated'] for r in rr),empty=sum(r['empty'] for r in rr))
        summary.append(entry)
        for task in sorted({s['task'] for s in samples}):
            ix=[i for i,s in enumerate(samples) if s['task']==task]
            d=diff[ix];low,high=ci(d,[images[i] for i in ix])
            tasks.append(dict(task=task,condition=k,n=len(ix),accuracy_percent=100*float(y[ix].mean()),
                delta_pp=float(d.mean()),ci95_low_pp=low,ci95_high_pp=high,
                correct_to_wrong=int(((full[ix]==1)&(y[ix]==0)).sum()),wrong_to_correct=int(((full[ix]==0)&(y[ix]==1)).sum())))
    for region in REGIONS:
        a=f'{region} Text OFF';b=f'{region} Attention OFF';d=100*(scores[a]-scores[b]);low,high=ci(d,images)
        paired.append(dict(region=region,comparison='Text OFF minus Attention OFF',metric='GQA Accuracy pp',
            delta=float(d.mean()),ci95_low=low,ci95_high=high,
            text_better=int((d>0).sum()),attention_better=int((d<0).sum()),tied=int((d==0).sum())))
    write_csv(OUT/'gqa_summary.csv',summary);write_csv(OUT/'gqa_task_accuracy.csv',tasks)
    write_csv(OUT/'gqa_per_question.csv',rows)
    return summary,tasks,paired,rows


def coco_statistics(samples,results,p):
    refs={s['question_id']:s['answer'] for s in samples};images=[s['image_sha256'] for s in samples]
    scores={};summary=[];rows=[];paired=[]
    for k in CONDITIONS:
        predictions={s['question_id']:results[uid(s)][k]['generation']['text'] for s in samples}
        cache=OUT/f'official_scores/{k}.json';binding=digest(dict(references=refs,predictions=predictions,source=p['scorer_source']))
        if cache.exists():
            r=read(cache);assert r['binding']==binding;overall,per=r['overall'],r['per_image']
        else:
            overall,per=caption_scores(refs,predictions)
            write_json(cache,dict(binding=binding,overall=overall,per_image=per))
        scores[k]={metric:np.array([per[s['question_id']][metric] for s in samples]) for metric in ['CIDEr','SPICE']}
        e=dict(condition=k,n=60,**overall)
        for metric in ['CIDEr','SPICE']:
            v=scores[k][metric];assert abs(v.mean()-overall[metric])<1e-10
            delta=v-scores['Full'][metric];lo,hi=ci(delta,images)
            e[f'delta_{metric}']=float(delta.mean());e[f'{metric}_delta_ci95_low']=lo;e[f'{metric}_delta_ci95_high']=hi
            e[f'{metric}_relative_drop_percent']=100*(summary[0][metric]-overall[metric])/summary[0][metric] if summary else 0.
        rr=[]
        for s in samples:
            g=results[uid(s)][k]['generation'];b=results[uid(s)]['Full']['generation']
            removed,added=delta_words(b['text'],g['text']);diff=divergence(g['token_ids'],b['token_ids'])
            assert diff==-1 or diff>=1
            row=dict(question_id=s['question_id'],image_id=s['raw_row']['id'],image_sha256=s['image_sha256'],
                condition=k,physical_gpu=p['gpus'][results[uid(s)][k]['rank']]['physical_gpu'],
                input_sha256=digest(s['input_mapping']),sample_sha256=digest(s),caption=g['text'],
                token_ids=json.dumps(g['token_ids']),first_divergence_zero_based=diff,
                generated_text_changed=g['text']!=b['text'],token_sequence_changed=diff!=-1,
                removed_words=removed,added_words=added,non_eos_length=sum(t!=2 for t in g['token_ids']),
                repeated_token_trigrams=repetition_count(g['token_ids']),
                eos=g['eos'],truncated=g['truncated'],empty=not bool(g['text']),**per[s['question_id']])
            rows.append(row);rr.append(row)
        e.update(generated_text_changed=sum(r['generated_text_changed'] for r in rr),
                 repeated_trigram_outputs=sum(r['repeated_token_trigrams']>0 for r in rr),
                 mean_non_eos_length=float(np.mean([r['non_eos_length'] for r in rr])),
                 eos=sum(r['eos'] for r in rr),truncated=sum(r['truncated'] for r in rr),empty=sum(r['empty'] for r in rr))
        summary.append(e);print('Official COCO:',k,overall,flush=True)
    for region in REGIONS:
        for metric in ['CIDEr','SPICE']:
            d=scores[f'{region} Text OFF'][metric]-scores[f'{region} Attention OFF'][metric];low,high=ci(d,images)
            paired.append(dict(region=region,comparison='Text OFF minus Attention OFF',metric=f'COCO {metric}',
                delta=float(d.mean()),ci95_low=low,ci95_high=high,
                text_better=int((d>0).sum()),attention_better=int((d<0).sum()),tied=int((d==0).sum())))
    write_csv(OUT/'coco_summary.csv',summary);write_csv(OUT/'coco_per_image.csv',rows)
    return summary,paired,rows,scores


def caption_terms(rows):
    """Descriptive noun/verb changes using the existing official scorer's POS models."""
    backend=Path('/data/lk/ATE-resources/paper_rq1/metric_deps/pycocoevalcap/spice/lib')
    smoke=read(OUT/'lexical_smoke.txt.json')['sentences']
    assert len(smoke)==2 and ('ride','VBG') in [(t['lemma'],t['pos']) for t in smoke[0]['tokens']]
    nonempty=[r for r in rows if r['caption'].strip()]
    folder=OUT/'lexical';folder.mkdir(exist_ok=True)
    text='\n'.join(r['caption'].replace('\n',' ').replace('\r',' ') for r in nonempty)+'\n'
    source=folder/'captions.txt';source.write_text(text)
    with (folder/'stanford.log').open('w') as log:
        subprocess.run(['java','-Xmx2G','-cp',str(backend/'*'),
            'edu.stanford.nlp.pipeline.StanfordCoreNLP','-annotators','tokenize,ssplit,pos,lemma',
            '-ssplit.eolonly','true','-file',str(source),'-outputFormat','json','-outputDirectory',str(folder)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    sentences=read(folder/'captions.txt.json')['sentences'];assert len(sentences)==len(nonempty)
    terms={}
    for r,sentence in zip(nonempty,sentences):
        terms[(r['question_id'],r['condition'])]={
            'nouns':{t['lemma'].lower() for t in sentence['tokens'] if t['pos'].startswith('NN')},
            'verbs':{t['lemma'].lower() for t in sentence['tokens'] if t['pos'].startswith('VB')}}
    per=[];summary=[]
    for r in rows:
        key=(r['question_id'],r['condition']);full=(r['question_id'],'Full')
        t=terms.get(key,dict(nouns=set(),verbs=set()));b=terms.get(full,dict(nouns=set(),verbs=set()))
        d=dict(question_id=r['question_id'],image_sha256=r['image_sha256'],condition=r['condition'])
        for tag in ['nouns','verbs']:
            d[tag]=json.dumps(sorted(t[tag]));d[f'added_{tag}']=json.dumps(sorted(t[tag]-b[tag]))
            d[f'removed_{tag}']=json.dumps(sorted(b[tag]-t[tag]));d[f'{tag}_changed']=t[tag]!=b[tag]
        per.append(d)
    for k in CONDITIONS:
        rr=[r for r in per if r['condition']==k]
        summary.append(dict(condition=k,n=60,noun_set_changed=sum(r['nouns_changed'] for r in rr),
            verb_set_changed=sum(r['verbs_changed'] for r in rr),
            interpretation='Lexical noun/verb lemma differences, including auxiliary verbs; NOT grounded object/action error or hallucination metrics'))
    write_csv(OUT/'coco_object_action_terms.csv',per);write_csv(OUT/'coco_object_action_change_counts.csv',summary)
    write_json(OUT/'lexical/validation.json',dict(passed=True,smoke_sentences=2,caption_sentences=len(sentences),
        tagger='Existing SPICE Stanford CoreNLP 3.6.0 POS+lemma; NN* and VB*',
        backend_sha256={str(p):sha256(p) for p in sorted(backend.glob('stanford-corenlp-3.6.0*.jar'))},
        no_grounded_error_claim=True))


def paired_figure(gqa,coco):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
    colors={'Attention':'#b34c66','Text':'#217b83'}
    for label,summary,metric,delta,low,high in [
            ('gqa_accuracy',gqa,'GQA Accuracy change (pp)','delta_pp','ci95_low_pp','ci95_high_pp'),
            ('coco_cider',coco,'COCO CIDEr change','delta_CIDEr','CIDEr_delta_ci95_low','CIDEr_delta_ci95_high'),
            ('coco_spice',coco,'COCO SPICE change','delta_SPICE','SPICE_delta_ci95_low','SPICE_delta_ci95_high')]:
        by={r['condition']:r for r in summary};fig,ax=plt.subplots(figsize=(6.6,3.9),layout='constrained')
        for kind,offset in [('Attention',-.13),('Text',.13)]:
            r=[by[f'{region} {kind} OFF'] for region in REGIONS];y=np.array([v[delta] for v in r])
            ax.errorbar(np.arange(3)+offset,y,yerr=[y-[v[low] for v in r],[v[high] for v in r]-y],
                        fmt='o',capsize=4,ms=7,color=colors[kind],label=f'{kind} OFF')
        ax.axhline(0,color='#555555',lw=1,ls='--');ax.set_xticks(range(3),['Early\nL1-L10','Middle\nL11-L20','Late\nL21-L31'])
        ax.set_ylabel(metric);ax.grid(axis='y',alpha=.16);ax.legend(frameon=False)
        ax.set_title('GQA: P-1 + Decode' if label.startswith('gqa') else 'COCO: native Prefill; Decode only')
        for ext in ['png','pdf']:fig.savefig(OUT/f'figure_{label}.{ext}',dpi=250)
        plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,summary,metric,scale,title in [(axs[0],gqa,'delta_pp',1,'GQA Accuracy (pp)'),
        (axs[1],coco,'delta_CIDEr',100/coco[0]['CIDEr'],'COCO CIDEr (% Full)'),
        (axs[2],coco,'delta_SPICE',100/coco[0]['SPICE'],'COCO SPICE (% Full)')]:
        by={r['condition']:r for r in summary}
        for kind in ['Attention','Text']:
            ax.plot(range(3),[by[f'{region} {kind} OFF'][metric]*scale for region in REGIONS],
                    'o-',color=colors[kind],label=f'{kind} OFF')
        ax.axhline(0,color='#555555',lw=.8,ls='--');ax.set_xticks(range(3),list(REGIONS));ax.set_title(title);ax.grid(alpha=.15)
    axs[0].legend(frameon=False);fig.suptitle('Different timing: GQA P-1+Decode vs COCO Decode-only',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(OUT/f'figure_cross_task_regions.{ext}',dpi=250)
    plt.close(fig)


def qualitative(samples,rows,scores):
    qids=[s['question_id'] for s in samples];selected=[];criteria=[]
    for region in REGIONS:
        d=scores[f'{region} Attention OFF']['SPICE']-scores['Full']['SPICE']
        for ascending in [True,False]:
            order=sorted(range(60),key=lambda i:(d[i] if ascending else -d[i],samples[i]['image_sha256']))
            index=next(i for i in order if qids[i] not in selected)
            selected.append(qids[index]);criteria.append(dict(question_id=qids[index],
                criterion=f'{region} Attention OFF: '+('largest SPICE harm' if ascending else 'largest gain / least harm'),
                delta_SPICE=float(d[index])))
    write_json(OUT/'qualitative_selection.json',dict(posthoc=True,criteria=criteria,aggregate_filtering=False))
    lookup={(r['question_id'],r['condition']):r for r in rows};fig,axes=plt.subplots(6,2,figsize=(13,21),
        gridspec_kw={'width_ratios':[1,2.1]},layout='constrained');csvrows=[]
    for j,criterion in enumerate(criteria):
        q=criterion['question_id'];s=samples[qids.index(q)]
        with Image.open(s['image_path']) as im:axes[j,0].imshow(im.convert('RGB'))
        axes[j,0].axis('off');axes[j,0].set_title(q,fontsize=10)
        ax=axes[j,1];ax.axis('off');texts=[]
        for k in CONDITIONS:
            r=lookup[(q,k)]
            text=textwrap.fill(f"{k} [C {r['CIDEr']:.2f}; S {r['SPICE']:.3f}]: {r['caption']}",width=110)
            texts.append((k,text));csvrows.append(dict(question_id=q,selection=criterion['criterion'],condition=k,
                caption=r['caption'],CIDEr=r['CIDEr'],SPICE=r['SPICE'],removed_words=r['removed_words'],added_words=r['added_words']))
        step=.89/sum(t.count('\n')+1.55 for _,t in texts);y=.98
        for k,t in texts:
            ax.text(0,y,t,va='top',fontsize=7.7,linespacing=1.1,transform=ax.transAxes,
                    color='#555555' if k=='Full' else '#b34c66' if 'Attention' in k else '#217b83')
            y-=step*(t.count('\n')+1.55)
        ax.text(0,.01,criterion['criterion'],fontsize=7.5,transform=ax.transAxes)
    fig.suptitle('Six posthoc COCO examples: Full and all six interventions',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(OUT/f'qualitative_six_coco.{ext}',dpi=200)
    plt.close(fig);write_csv(OUT/'qualitative_captions.csv',csvrows)


def report(gqa,tasks,coco,paired,completed):
    lines=['# LLaVA层段Attention更新与Text来源消融','',
        '## 协议与边界',
        '固定LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager/Transformers4.49.0；L0始终原生。Early=L1–L10，Middle=L11–L20，Late=L21–L31（后段11层，前/中段10层，不能混同干预层数）。',
        'GQA：原3000题中固定输入哈希选每任务100，共600题，600个图片SHA，未按输出筛选；对P−1及所有真实Decode干预，最多32 Token。',
        'COCO：原60图、原Prompt和官方参考Caption；整个Prefill含P−1原生，只对真实Decode干预，最多96 Token。两数据集干预时机不同，差异不能仅归因任务类型。',
        'Attention OFF仅置零目标Query的o_proj更新，QKV/cache、MLP和Residual不变；Text OFF是Softmax前−inf文本Key Mask，保留原BOS、全部Image与当前Self，并动态屏蔽已生成历史。两者都不是Image Key Mask，不是关闭Block或物理剪枝，也不代表真实加速。',
        '', '## GQA Accuracy',
        '|条件|Accuracy %|ΔFull pp|95%CI pp|对→错|错→对|首Token翻转|EOS/截断|',
        '|---|---:|---:|---|---:|---:|---:|---|']
    for r in gqa:lines.append(f"|{r['condition']}|{r['accuracy_percent']:.3f}|{r['delta_pp']:+.3f}|[{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}]|{r['correct_to_wrong']}|{r['wrong_to_correct']}|{r['first_token_flip_count']}|{r['eos']}/{r['truncated']}|")
    lines+=['','六任务结果详见gqa_task_accuracy.csv；完整答案、Token ID、首Token翻转与LogP Loss、输出变化、长度及EOS见gqa_per_question.csv。',
        '', '## COCO官方指标（原始尺度）','|条件|CIDEr|ΔFull|相对降幅%|SPICE|ΔFull|相对降幅%|EOS/截断|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in coco:lines.append(f"|{r['condition']}|{r['CIDEr']:.6f}|{r['delta_CIDEr']:+.6f}|{r['CIDEr_relative_drop_percent']:+.2f}|{r['SPICE']:.6f}|{r['delta_SPICE']:+.6f}|{r['SPICE_relative_drop_percent']:+.2f}|{r['eos']}/{r['truncated']}|")
    lines+=['','COCO各条件相对Full的同图配对95%CI见coco_summary.csv。负的相对降幅表示点估计提升，不代表已证实提升。',
        '', '## 同一区段：Text OFF − Attention OFF',
        '|层段|指标|差值|95%CI|Text更高/Attention更高/相同|','|---|---|---:|---|---|']
    for r in paired:lines.append(f"|{r['region']}|{r['metric']}|{r['delta']:+.6f}|[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]|{r['text_better']}/{r['attention_better']}/{r['tied']}|")
    lines+=['','## 本轮数据读数（探索性）']
    for kind in ['Attention','Text']:
        group=[r for r in gqa if f' {kind} OFF' in r['condition']]
        worst=min(group,key=lambda r:r['delta_pp']);best=max(group,key=lambda r:r['delta_pp'])
        lines.append(f"GQA {kind} OFF：点估计损失最大为{worst['condition']}（{worst['delta_pp']:+.3f} pp），最小为{best['condition']}（{best['delta_pp']:+.3f} pp）；不能仅按排序作显著性或无损判断。")
        group=[r for r in coco if f' {kind} OFF' in r['condition']]
        worst=min(group,key=lambda r:r['delta_SPICE']);best=max(group,key=lambda r:r['delta_SPICE'])
        lines.append(f"COCO {kind} OFF：SPICE点估计损失最大为{worst['condition']}（{worst['delta_SPICE']:+.6f}），最小为{best['condition']}（{best['delta_SPICE']:+.6f}）；CIDEr、SPICE和人工图文核对可能不一致，应分别阅读。")
    for r in paired:
        lines.append(f"{r['region']} {r['metric']}：Text OFF比Attention OFF点估计{r['delta']:+.6f}，普通配对95%CI [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]。")
    lines+=['','## 统计与解释限制',
        '20,000次图片SHA整簇配对Bootstrap，保留题目加权估计；本次GQA和COCO各自均为一图一题。所有层段和分任务比较均探索性，普通95%CI未经多重比较校正；区间跨0不证明无功能损害或等价。',
        '沿用历史官方COCO评分器和统计口径：CIDEr的document frequencies固定于原60图参考语料，重采样官方逐图贡献；SPICE重采样官方逐图F值。区间条件于原参考语料，不是新语料重估IDF的总体区间。',
        'Text OFF的保留概率会重新分配；o_proj OFF则移除整个Attention更新。这两种干预损失不可相加分解成独立来源贡献，不足以发现语言中枢或视觉中枢。',
        'repeated_token_trigrams是Token三元组复读的语法性筛查，不是新的质量或幻觉指标；与EOS、截断、空答案分别记录。',
        '每图词序差异见coco_per_image.csv；coco_object_action_terms.csv及change_counts.csv使用现有SPICE依赖Stanford POS/lemma抽取NN*/VB*名词/动词变化（含助动词）。这些是描述词差异，不是视觉事实标注、对象召回率或幻觉评分；不将名词/动词变化自动判为错误。六案例按预声明规则事后选取，人工对象/动作核对另见qualitative_notes_zh.md，不作为新增质量指标。',
        '', '## 验证与运行',
        '6项CPU测试及三卡真实Smoke通过：Full/no-op有效生成Logits精确一致，官方原生生成Token一致；非目标Prefill的Attention、Hidden及Logits精确不变。COCO整个Prefill及首Token保持精确一致。',
        '目标o_proj输出严格0；Text目标概率严格0；非目标Key Mask未变、Image实际Mass保持正值，合法Key存在；原始P−1在Decode时正确成为文本历史。检查每层真实Cache位置/长度、独立Cache、Hook清理、输入哈希和参数版本不变；GPU前向无NaN/Inf/OOM。',
        '官方SPICE首次CPU评分在长重复Caption的Stanford解析中耗尽Java堆并失败，原始日志保存在analysis_attempt1.log。仅通过_JAVA_OPTIONS提高堆上限至32 GiB、限制可见CPU核数为2后重试；没有修改Caption、官方评分器或指标定义，没有重跑GPU生成。个别长Caption的缓存键过长警告不等于评分失败，所有最终逐图分数均检查有限且完整。',
        '三卡正式运行分钟：'+', '.join(f"{c['formal_seconds']/60:.2f}" for c in completed)+ '。GPU1未使用。',
        f"执行代码Commit：{read(OUT/'execution.json')['commit']}。全部4620条为本轮真实配对生成，没有复用历史分数。",'']
    (OUT/'report_zh.md').write_text('\n'.join(lines))


def main():
    tick=time.time();p=verify();samples,results,completed=load(p)
    gqa=[s for s in samples if s['dataset']!='coco_caption'];coco=[s for s in samples if s['dataset']=='coco_caption']
    gs,ts,gp,gr=gqa_statistics(gqa,results,p);cs,cp,cr,raw=coco_statistics(coco,results,p)
    write_csv(OUT/'same_region_comparisons.csv',gp+cp)
    caption_terms(cr);paired_figure(gs,cs);qualitative(coco,cr,raw);report(gs,ts,cs,gp+cp,completed)
    write_json(OUT/'validation.json',dict(passed=True,samples=660,gqa_questions=600,coco_images=60,conditions=7,
        complete_generations=4620,cpu_tests=6,three_gpu_smokes=True,source_unchanged=True,
        caption_all_prefill_logits_exact=True,mask_and_projection_audits=4620,
        gqa_eos={r['condition']:r['eos'] for r in gs},gqa_truncated={r['condition']:r['truncated'] for r in gs},
        coco_eos={r['condition']:r['eos'] for r in cs},coco_truncated={r['condition']:r['truncated'] for r in cs},
        analysis_source_sha256=sha256(Path(__file__)),analysis_seconds=time.time()-tick))
    print('Official scoring and paired analysis complete',flush=True)


if __name__=='__main__':main()
