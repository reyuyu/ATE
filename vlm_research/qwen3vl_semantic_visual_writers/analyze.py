"""CPU-only full checkpoint replay, paired image inference, and cross-model plots."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .common import *
from ..qwen3vl_query_role.analyze import paired_bootstrap,exact_mcnemar
from ..qwen3vl_early_windows.experiment import gqa_prediction


def first_metrics(reference,current):
    token=int(reference.argmax())
    a=reference.astype(np.float64)
    b=current.astype(np.float64)
    return dict(flip=int(token!=int(b.argmax())),
        signed_loss=float(a[token]-np.logaddexp.reduce(a)-b[token]+np.logaddexp.reduce(b)))


def validate(record,s,spec,c,p,tokenizer,eos_ids):
    assert record['binding']==digest(p) and record['sample_sha256']==digest(s)
    assert record['input_binding']==spec['binding'] and record['condition']==c
    assert record['question_id']==s['question_id'] and record['task']==s['task']
    assert record['image_sha256']==s['image_sha256'] and record['gold']==s['answer']
    g=record['generation']
    assert digest(g)==record['result_sha256']
    assert g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer'].lower())
    assert tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text']
    ids=g['token_ids']
    assert 1<=len(ids)<=32 and g['eos']==(ids[-1] in eos_ids)
    assert g['empty']==(not bool(g['text'])) and g['truncated']==(len(ids)==32 and ids[-1] not in eos_ids)
    repeated=any(ids[i:i+3]==ids[j:j+3] for i in range(max(0,len(ids)-2)) for j in range(i+3,max(0,len(ids)-2)))
    assert g['repeated_trigram']==repeated
    assert g['generated_tokens']==len(ids) and g['decode_queries']==len(ids)-1
    assert g['all_logits_finite'] and g['cache_initial_empty'] and g['independent_native_cache']
    assert g['logits_shapes']==[[1,1,p['vocab_size']]]*len(ids)
    assert g['logits_to_keep']==1 and g['all_36_answer_image_blocked'] and not g['noop']
    assert g['hook_events']==[] and g['target_prefill_queries']==g['target_decode_queries']==0
    writers=spec['selection']['writers'][c]
    assert g['writer_positions']==writers and g['writer_count']==len(writers)
    assert g['relay_count']==len(spec['roles']['R'])
    assert g['prefill_role_counts']=={k:len(v) for k,v in spec['roles'].items()}
    n=spec['layout']['input_length']
    assert g['cache_lengths']==[n+len(ids)-1]*36 and g['cache_forward_count']==len(ids)
    assert len(g['source_events'])==36*len(ids) and len(g['mrope_events'])==len(ids)
    for t in range(len(ids)):
        e=g['mrope_events'][t]
        assert e['step']==t and e['native_mrope']
        if not t:
            assert e['position_ids_sha256']==spec['mrope']['position_ids_sha256']
            assert e['shape']==[3,1,n] and e['cache_position']==[0,n-1]
        else:
            assert e['shape']==[3,1,1] and e['cache_position']==[n+t-1]
            position=np.full((3,1,1),n+t-1+spec['mrope']['delta'],dtype=np.int64)
            assert e['position_ids_sha256']==hashlib.sha256(position.tobytes()).hexdigest()
        for k,e in enumerate(g['source_events'][36*t:36*t+36]):
            assert e['layer']==k and e['step']==t and not e['noop']
            assert e['phase']==('decode' if t else 'prefill')
            assert e['absolute_start']==(n+t-1 if t else 0)
            assert e['query_length']==(1 if t else n) and e['key_length']==n+t
            expected=len(spec['roles']['I'])*(1+(len(spec['roles']['R'])-len(writers) if not t else 0))
            assert e['blocked_connections_per_head']==expected and e['blocked_probability_max']==0
            assert e['row_sum_max_error']<=2**-7 and e['non_target_mask_exact'] and e['prefix_self_legal']
    v=raw(record)
    assert v.ndim==1 and int(v.argmax())==ids[0] and digest(v.tolist())==g['first_logits_sha256']
    return g,v


def distribution(a,b):
    exact=np.array_equal(a,b)
    x,y=a.astype(np.float64),b.astype(np.float64)
    lp,lq=x-np.logaddexp.reduce(x),y-np.logaddexp.reduce(y)
    lm=np.logaddexp(lp,lq)-math.log(2)
    js=.5*(np.sum(np.exp(lp)*(lp-lm))+np.sum(np.exp(lq)*(lq-lm)))
    return dict(first_logits_exact=exact,first_logits_max_abs=float(np.max(np.abs(x-y))),
        first_distribution_js=0. if exact else max(0.,float(js)),
        first_distribution_tv=0. if exact else float(.5*np.abs(np.exp(lp)-np.exp(lq)).sum()))


def replay():
    from transformers import AutoTokenizer
    p=verify()
    assert read(OUT/'generation_completion.json')['passed']
    assert read(OUT/'generation_completion.json')['binding']==digest(p)
    assert read(OUT/'gpu_smoke_validation.json')['passed'] and read(OUT/'image_independence_validation.json')['passed']
    assert read(OUT/'gpu_release.json')['passed']
    assert len(list((OUT/'checkpoints').glob('*/*.json')))==2400
    assert len(list((OUT/'swap_checkpoints').glob('*/*.json')))==400
    samples=read(OUT/'sample_manifest.json')['samples']
    specs=read(OUT/'token_group_manifest.json')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    eos=read(MODEL/'generation_config.json')['eos_token_id']
    eos=[eos] if isinstance(eos,int) else eos
    generations={c:[] for c in CONDITIONS}
    per,hashes,raw_hashes=[],{},{}
    for s in samples:
        assert sha256(s['image_path'])==s['image_sha256']
        spec=specs['main'][key(s)]
        records,vectors={},{}
        for c in CONDITIONS:
            path=checkpoint(key(s),c)
            r=read(path)
            records[c],vectors[c]=validate(r,s,spec,c,p,tokenizer,eos)
            hashes[str(path.relative_to(OUT))]=sha256(path)
            raw_hashes[r['raw']['path']]=r['raw']['sha256']
            generations[c].append(records[c])
        for c,g in records.items():
            a,b=records['All-Writer'],records['No-Writer']
            fa,fb=first_metrics(vectors['All-Writer'],vectors[c]),first_metrics(vectors['No-Writer'],vectors[c])
            selected=spec['selection']
            per.append(dict(question_id=s['question_id'],task=s['task'],image_id=s['image_id'],image_sha256=s['image_sha256'],
                question=s['question'],gold=s['answer'],condition=c,text=g['text'],prediction=g['prediction'],token_ids=g['token_ids'],
                correct=g['correct'],All_correct=a['correct'],No_correct=b['correct'],
                all_correct_to_wrong=int(a['correct'] and not g['correct']),all_wrong_to_correct=int(not a['correct'] and g['correct']),
                no_correct_to_wrong=int(b['correct'] and not g['correct']),no_wrong_to_correct=int(not b['correct'] and g['correct']),
                first_top1_flip_all=fa['flip'],first_top1_flip_no=fb['flip'],
                signed_first_logp_loss_all=fa['signed_loss'],signed_first_logp_loss_no=fb['signed_loss'],
                sequence_changed_all=g['token_ids']!=a['token_ids'],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],
                repeated_trigram=g['repeated_trigram'],generated_tokens=len(g['token_ids']),
                writer_count=g['writer_count'],relay_count=g['relay_count'],writer_ratio=g['writer_count']/g['relay_count'],
                k=selected['k'],writer_positions=g['writer_positions'],writer_tokens_decoded=spec['writers_decoded'][c],
                actual_image_tokens=len(spec['roles']['I']),image_grid_thw=spec['layout']['image_grid_thw'],
                question_token_count=len(selected['question_positions']),core_candidate_tokens=selected['candidate_core_tokens'],
                whole_word_budget_reduction=selected['budget_reduction_for_whole_word_suffix'],
                scaffold_deficit=selected['scaffold_budget_deficit'],input_binding=spec['binding']))
    swaps=[]
    layouts={r['pair_id']:r for r in read(OUT/'swap_layout_manifest.json')['pairs']}
    for pair in read(OUT/'image_swap_manifest.json')['pairs']:
        for c in SWAP_CONDITIONS:
            sides=[]
            for side,s in enumerate(pair['samples']):
                stem=f"{pair['pair_id']}__{side}"
                path=checkpoint(stem,c,True)
                r=read(path)
                g,v=validate(r,s,specs['swaps'][stem],c,p,tokenizer,eos)
                sides.append((g,v,r))
                hashes[str(path.relative_to(OUT))]=sha256(path)
                raw_hashes[r['raw']['path']]=r['raw']['sha256']
            a,b=sides
            same=layouts[pair['pair_id']]['same_layout']
            trace_exact=None
            if c=='No-Writer':
                trace_exact=np.array_equal(raw(a[2],True),raw(b[2],True))
                if same:
                    assert trace_exact and a[0]['token_ids']==b[0]['token_ids']
            swaps.append(dict(pair_id=pair['pair_id'],condition=c,question=pair['samples'][0]['question'],
                qid_left=pair['samples'][0]['question_id'],qid_right=pair['samples'][1]['question_id'],
                image_sha_left=pair['samples'][0]['image_sha256'],image_sha_right=pair['samples'][1]['image_sha256'],
                gold_left=pair['samples'][0]['answer'],gold_right=pair['samples'][1]['answer'],
                text_left=a[0]['text'],text_right=b[0]['text'],token_ids_left=a[0]['token_ids'],token_ids_right=b[0]['token_ids'],
                correct_left=a[0]['correct'],correct_right=b[0]['correct'],both_correct=int(a[0]['correct'] and b[0]['correct']),
                answer_changed=a[0]['prediction']!=b[0]['prediction'],sequence_changed=a[0]['token_ids']!=b[0]['token_ids'],
                first_top1_changed=a[0]['token_ids'][0]!=b[0]['token_ids'][0],same_layout=same,
                positional_confound=not same,differing_layout_fields=layouts[pair['pair_id']]['differing_fields'],
                no_writer_all_valid_logits_exact=trace_exact,**distribution(a[1],b[1])))
    return p,samples,generations,per,swaps,hashes,raw_hashes


def statistics(samples,generations,per):
    matrix=np.array([[g['correct'] for g in generations[c]] for c in CONDITIONS]).T.astype(float)
    draws=paired_bootstrap(matrix,[s['image_sha256'] for s in samples])
    idx={c:i for i,c in enumerate(CONDITIONS)}
    random_idx=[idx[f'Random-k-{s}'] for s in RANDOM_SEEDS]
    random_mean,random_draws=matrix[:,random_idx].mean(1),draws[:,random_idx].mean(1)
    summary=[]
    for c in CONDITIONS:
        j=idx[c]
        rows=[r for r in per if r['condition']==c]
        acc_ci=np.quantile(draws[:,j],[.025,.975])
        all_ci=np.quantile(draws[:,j]-draws[:,0],[.025,.975])
        no_ci=np.quantile(draws[:,j]-draws[:,1],[.025,.975])
        summary.append(dict(condition=c,questions=240,correct=int(matrix[:,j].sum()),accuracy_pct=float(matrix[:,j].mean()*100),
            accuracy_ci_low=float(acc_ci[0]),accuracy_ci_high=float(acc_ci[1]),
            delta_all_pp=float((matrix[:,j]-matrix[:,0]).mean()*100),all_ci_low=float(all_ci[0]),all_ci_high=float(all_ci[1]),
            delta_no_pp=float((matrix[:,j]-matrix[:,1]).mean()*100),no_ci_low=float(no_ci[0]),no_ci_high=float(no_ci[1]),
            all_harm=sum(r['all_correct_to_wrong'] for r in rows),all_rescue=sum(r['all_wrong_to_correct'] for r in rows),
            no_harm=sum(r['no_correct_to_wrong'] for r in rows),no_rescue=sum(r['no_wrong_to_correct'] for r in rows),
            first_flips_all=sum(r['first_top1_flip_all'] for r in rows),first_flips_no=sum(r['first_top1_flip_no'] for r in rows),
            signed_first_logp_loss_all_mean=float(np.mean([r['signed_first_logp_loss_all'] for r in rows])),
            signed_first_logp_loss_no_mean=float(np.mean([r['signed_first_logp_loss_no'] for r in rows])),
            writer_count_mean=float(np.mean([r['writer_count'] for r in rows])),
            writer_ratio_mean=float(np.mean([r['writer_ratio'] for r in rows])),
            writer_count_min=min(r['writer_count'] for r in rows),writer_count_max=max(r['writer_count'] for r in rows),
            eos=sum(r['eos'] for r in rows),empty=sum(r['empty'] for r in rows),truncated=sum(r['truncated'] for r in rows),
            repeated_trigram=sum(r['repeated_trigram'] for r in rows)))
    paired=[]
    for k,(a,b) in enumerate(PRIMARY+ [('Semantic-k',f'Random-k-{s}') for s in RANDOM_SEEDS]):
        x,xd=matrix[:,idx[a]],draws[:,idx[a]]
        y,yd=(random_mean,random_draws) if b=='Random-mean(3)' else (matrix[:,idx[b]],draws[:,idx[b]])
        ci=np.quantile(xd-yd,[.025,.975])
        corrected=np.quantile(xd-yd,[.025/7,1-.025/7]) if k<7 else [None,None]
        harm,rescue=float(((1-x)*y).sum()),float((x*(1-y)).sum())
        paired.append(dict(comparison=a+' minus '+b,delta_pp=float((x-y).mean()*100),ci_low=float(ci[0]),ci_high=float(ci[1]),
            bonferroni7_low=None if corrected[0] is None else float(corrected[0]),
            bonferroni7_high=None if corrected[1] is None else float(corrected[1]),primary_family=k<7,
            harm_or_seed_average_harm=harm,rescue_or_seed_average_rescue=rescue,fractional_reference=b=='Random-mean(3)',
            mcnemar_exact=None if b=='Random-mean(3)' else exact_mcnemar(int(harm),int(rescue)),clusters=240,repetitions=20000,seed=SEED))
    tasks=[]
    for t in sorted({s['task'] for s in samples}):
        for c in CONDITIONS:
            rows=[r for r in per if r['task']==t and r['condition']==c]
            tasks.append(dict(task=t,condition=c,questions=len(rows),correct=sum(r['correct'] for r in rows),
                accuracy_pct=float(np.mean([r['correct'] for r in rows])*100),
                delta_all_pp=float(np.mean([r['correct']-r['All_correct'] for r in rows])*100),
                all_harm=sum(r['all_correct_to_wrong'] for r in rows),all_rescue=sum(r['all_wrong_to_correct'] for r in rows)))
    return summary,paired,tasks


def swap_summary(swaps):
    rows=[]
    for c in SWAP_CONDITIONS:
        for group in ['all','same_layout','different_layout']:
            selected=[r for r in swaps if r['condition']==c and (group=='all' or r['same_layout']==(group=='same_layout'))]
            rows.append(dict(condition=c,layout_group=group,pairs=len(selected),
                answer_changed=sum(r['answer_changed'] for r in selected),both_correct=sum(r['both_correct'] for r in selected),
                endpoint_correct=sum(r['correct_left']+r['correct_right'] for r in selected),
                first_top1_changed=sum(r['first_top1_changed'] for r in selected),
                mean_first_js=float(np.mean([r['first_distribution_js'] for r in selected])) if selected else None,
                mean_first_tv=float(np.mean([r['first_distribution_tv'] for r in selected])) if selected else None,
                no_writer_all_valid_exact=sum(r['no_writer_all_valid_logits_exact'] is True for r in selected)))
    return rows


def cross_model(samples,summary):
    llava={r['condition']:r for r in read_csv(OUT/'llava_accuracy_summary.csv')}
    qwen={r['condition']:r for r in summary}
    comparison=[]
    for c in CONDITIONS:
        l,q=llava[c],qwen[c]
        comparison.append(dict(condition=c,llava_correct=int(l['correct']),qwen_correct=q['correct'],
            llava_accuracy_pct=float(l['accuracy_pct']),qwen_accuracy_pct=q['accuracy_pct'],
            llava_delta_all_pp=float(l['delta_all_pp']),qwen_delta_all_pp=q['delta_all_pp'],
            llava_writer_count_mean=float(l['writer_count_mean']),qwen_writer_count_mean=q['writer_count_mean'],
            llava_writer_ratio_mean=float(l['writer_ratio_mean']),qwen_writer_ratio_mean=q['writer_ratio_mean'],
            llava_all_harm=int(l['all_harm']),qwen_all_harm=q['all_harm'],
            llava_all_rescue=int(l['all_rescue']),qwen_all_rescue=q['all_rescue']))
    lspecs=read(OUT/'llava_token_group_manifest.json')['main']
    qspecs=read(OUT/'token_group_manifest.json')['main']
    selected=[]
    for s in samples:
        l,q=lspecs[key(s)],qspecs[key(s)]
        choices=[]
        words=[]
        for spec in [l,q]:
            question_words=[w for w in spec['words'] if w['in_question']]
            chosen=set(spec['selection']['writers']['Semantic-k'])
            choices.append({i for i,w in enumerate(question_words) if set(w['positions'])<=chosen})
            words.append([w['text'] for w in question_words])
        assert words[0]==words[1]
        a,b=choices
        selected.append(dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],question=s['question'],
            llava_k=l['selection']['k'],qwen_k=q['selection']['k'],
            llava_selected_words=[words[0][i] for i in sorted(a)],qwen_selected_words=[words[1][i] for i in sorted(b)],
            selected_word_occurrences_equal=a==b,selected_word_jaccard=len(a&b)/len(a|b) if a|b else 1.,
            llava_budget_reduction=l['selection']['budget_reduction_for_whole_word_suffix'],
            qwen_budget_reduction=q['selection']['budget_reduction_for_whole_word_suffix']))
    return comparison,selected


def figures(summary,paired,cross,swaps,tasks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        for suffix in ['png','pdf']:
            fig.savefig(OUT/(name+'.'+suffix),dpi=180)
        plt.close(fig)
    colors=['#303030','#888888','#247f68','#477bb6','#c96d30','#8d62a0','#4c8b98','#4c8b98','#4c8b98','#bb4545']
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),constrained_layout=True)
    for i,r in enumerate(summary):
        axes[0].scatter(100*r['writer_ratio_mean'],r['accuracy_pct'],s=65,color=colors[i],
            marker=['o','o','o','o','o','o','v','s','D','o'][i],label=r['condition'].replace('Random-k-','Random '))
    axes[0].set(xlabel='Mean fraction of R allowed to read image keys (%)',ylabel='Complete-answer accuracy (%)',
        xlim=(-3,103),title='Qwen3-VL / fixed 240 images')
    axes[0].legend(frameon=False,fontsize=7,ncol=2,loc='best')
    for i,r in enumerate(paired[:4]):
        axes[1].errorbar(r['delta_pp'],i,xerr=[[r['delta_pp']-r['ci_low']],[r['ci_high']-r['delta_pp']]],fmt='o',capsize=4,color='#247f68')
    axes[1].axvline(0,color='gray',linewidth=.8)
    axes[1].set(yticks=range(4),yticklabels=['vs random mean (3)','vs question last','vs R last','vs scaffold'],
        xlabel='Semantic accuracy minus control (pp), exploratory 95% CI',title='20,000 paired image-cluster bootstrap')
    axes[1].invert_yaxis()
    save(fig,'figure_quality_budget')
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),constrained_layout=True)
    selected=[r for r in cross if not r['condition'].startswith('Random-')]
    labels=[r['condition'].replace('-Writer','').replace('-k','') for r in selected]
    x=np.arange(len(selected))
    axes[0].bar(x-.18,[r['llava_accuracy_pct'] for r in selected],width=.36,label='LLaVA',color='#477bb6')
    axes[0].bar(x+.18,[r['qwen_accuracy_pct'] for r in selected],width=.36,label='Qwen3-VL',color='#247f68')
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Complete-answer accuracy (%)',title='Same 240 questions; native model protocols')
    plt.setp(axes[0].get_xticklabels(),rotation=30,ha='right')
    axes[0].legend(frameon=False)
    lp={r['comparison']:r for r in read_csv(OUT/'llava_paired_comparisons.csv')}
    for i,r in enumerate(paired[:3]):
        for offset,color,source in [(-.12,'#477bb6',lp[r['comparison']]),(.12,'#247f68',r)]:
            delta,lo,hi=[float(source[k]) for k in ['delta_pp','ci_low','ci_high']]
            axes[1].errorbar(delta,i+offset,xerr=[[delta-lo],[hi-delta]],fmt='o',color=color,capsize=3)
    axes[1].axvline(0,color='gray',linewidth=.8)
    axes[1].set(yticks=range(3),yticklabels=['Semantic - random mean','Semantic - question last','Semantic - R last'],
        xlabel='Paired accuracy difference (pp), ordinary 95% CI',title='Model-conditional effects, not architecture isolation')
    axes[1].invert_yaxis()
    save(fig,'figure_cross_model')
    fig,ax=plt.subplots(figsize=(9,4.8),constrained_layout=True)
    rows=[r for r in swaps if r['layout_group']=='all']
    x=np.arange(5)
    ax.bar(x-.18,[r['answer_changed'] for r in rows],width=.36,color='#477bb6',label='Answer changes')
    ax.bar(x+.18,[r['both_correct'] for r in rows],width=.36,color='#247f68',label='Both correct')
    audit=read(OUT/'swap_layout_manifest.json')
    ax.set(xticks=x,xticklabels=['All','None','Semantic','R last','Random seed1'],ylabel='Pairs (of 40)',ylim=(0,42),
        title=f"Native image swaps: {audit['same_layout']} matched / {audit['different_layout']} layout-confounded")
    ax.legend(frameon=False)
    save(fig,'figure_image_swap')
    fig,ax=plt.subplots(figsize=(9,5.4),constrained_layout=True)
    names=sorted({r['task'] for r in tasks})
    values=np.array([[next(r['accuracy_pct'] for r in tasks if r['task']==t and r['condition']==c) for t in names] for c in CONDITIONS])
    im=ax.imshow(values,vmin=0,vmax=100,aspect='auto',cmap='viridis')
    ax.set(xticks=range(6),xticklabels=names,yticks=range(10),yticklabels=CONDITIONS,title='Qwen3-VL / six tasks, 40 questions each')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
    for i in range(10):
        for j in range(6):
            ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]<65 else 'black')
    fig.colorbar(im,ax=ax,label='Accuracy (%)')
    save(fig,'figure_task_accuracy')


def report(p,summary,paired,cross,selections,swaps):
    by={r['condition']:r for r in summary}
    sem,all_,no=by['Semantic-k'],by['All-Writer'],by['No-Writer']
    budget=read(OUT/'token_budget_audit.json')
    layout=read(OUT/'swap_layout_manifest.json')
    def interval(r,prefix=''):
        return f"[{r[prefix+'ci_low']:+.3f}, {r[prefix+'ci_high']:+.3f}]"
    def corrected(r):
        return f"[{r['bonferroni7_low']:+.3f}, {r['bonferroni7_high']:+.3f}]"
    def conclusion(r):
        if r['bonferroni7_low']>0:
            return '七项校正后仍支持正向差异'
        if r['bonferroni7_high']<0:
            return '七项校正后仍支持负向差异'
        return '七项校正区间包含0，尚不能确立差异，也不是等价证明'
    lsem=next(r for r in cross if r['condition']=='Semantic-k')
    if paired[0]['bonferroni7_low']>0:
        selectivity='Qwen的语义启发式优势在预声明校正后仍成立，与LLaVA的选择性方向一致；这支持原生协议下的跨模型复现，而不是少量任意Token都等效。'
    elif paired[0]['bonferroni7_high']<0:
        selectivity='Qwen中语义策略在校正后反而低于随机平均，没有复现LLaVA的语义优势；不能以跨模型一致性概括本结果。'
    else:
        selectivity='Qwen中尚未建立校正后的语义优势，因此不能宣称LLaVA的语义选择性已经跨模型复现；区间含0也不证明所有选点等效。'
    if paired[5]['bonferroni7_low']>0:
        sparsity='少量选定文本的直接读图确实比完全关闭写入保留更多正确答案，支持部分间接读取的功能充分性线索；相对All的真实损害仍必须保留。'
    else:
        sparsity='本批尚未建立少量选定文本相对No-Writer的校正后收益，不能据此宣称稀疏视觉写入已经足够。'
    lines=['# Qwen3-VL Semantic Visual Writers：跨模型复现','',
        '**COMPLETE：240题×10条件共2400条全部新生成；另有40组×2端×5条件共400条换图生成。历史原生Full不充当All-Writer。**','',
        '## 直接回答','',
        f"Qwen Semantic-k为{sem['correct']}/240（{sem['accuracy_pct']:.3f}%），平均开放{sem['writer_count_mean']:.2f}个R Token（平均逐题R比例{100*sem['writer_ratio_mean']:.2f}%）。All-Writer为{all_['correct']}/240，No-Writer为{no['correct']}/240。",'',
        f"**是否只需要少量特定文本Token读图？** Semantic−All为{paired[4]['delta_pp']:+.3f} pp，普通95%CI {interval(paired[4])}，校正区间{corrected(paired[4])}；相对All原对变错{sem['all_harm']}题、原错变对{sem['all_rescue']}题。Semantic−No为{paired[5]['delta_pp']:+.3f} pp，{conclusion(paired[5])}。是否保留足够预测能力只能依这些数值及换图结果判断；没有预声明非劣效界限，不能由接近或CI跨0宣称无损。",'',
        f"**语义优势是否跨模型成立？** Qwen Semantic−三个随机种子逐题平均为{paired[0]['delta_pp']:+.3f} pp，普通95%CI {interval(paired[0])}，校正区间{corrected(paired[0])}，{conclusion(paired[0])}。LLaVA对应效应为+11.250 pp，校正区间[+5.000,+17.500]。应区分方向一致、两模型各自证据强度与真正的模型交互；不能用显著/不显著之差替代交互检验。",'',
        f"Semantic−Question-Last为{paired[1]['delta_pp']:+.3f} pp，{conclusion(paired[1])}；Semantic−R-Last为{paired[2]['delta_pp']:+.3f} pp，{conclusion(paired[2])}。这两个末端对照不同，不能合并后宣称语义普遍胜出。",'',
        f"Drop-Semantic−All为{paired[6]['delta_pp']:+.3f} pp，普通95%CI {interval(paired[6])}，校正区间{corrected(paired[6])}，{conclusion(paired[6])}。这是移除最终选定集合读图权限的条件必要性实验，不等于视觉信息唯一储存在它们。",'',
        '**综合判定：** '+selectivity+sparsity+'没有预先声明非劣效或等价界限，任何接近的净Accuracy都不是严格无损证明。','',
        '## 冻结规则与跨模型边界','',
        f"模型revision `{REVISION}`，历史原生BF16/Eager、Transformers4.57.6、Processor默认图像分辨率、DeepStack和mRoPE不变。显式logits_to_keep=1，Greedy最多32 Token，真实独立KV及前缀。240题及图片SHA完全复用，未重采样。源码冻结Commit `{p['code_commit']}`。",
        '只改变36层Softmax前Query→原始Image Key的Hard Mask。全部条件P−1及每个真实Decode Query（包括EOS预测）均不能直接读图；仅选定Prefill R Query开放。其他来源、原因果/Pad Mask、MLP、Residual、Attention输出与KV更新均原生。',
        '**R包括最后一个实际Image Token之后、P−1之前全部原始位置，包括`vision_end`及模板特殊Token。** 历史Qwen的o_proj角色实验排除了vision_end，但本轮若不管控它，会留下未受控的文本视觉中继。该边界在GPU前冻结，且没有改动原Token或位置。',
        f"Image来自实际Processor的image-pad IDs，不是576固定Token；本批范围{budget['image_tokens_min']}–{budget['image_tokens_max']}。Question由原生offset映射识别；输入映射逐题与历史Qwen240题完全一致。",'',
        f"词法选择器原文件SHA256 `{SELECTOR_SHA}`，与LLaVA逐字节相同，包括STOP、完整Subword词、同预算DP和三个随机种子。Qwen重新分词后k为{budget['k_min']}–{budget['k_max']}，{budget['reduced_for_whole_word_suffix']}/240题存在整词末端预算约束缩减，Scaffold不足{budget['scaffold_deficits']}题。缩减和不足均记录，不删除题。",
        'Semantic是输入词法启发式，不是验证过的语义标注器。Drop-Semantic取最终Semantic-k集合的补集。Random为整词组随机排列后的精确预算DP，不是独立均匀子词抽样。选词不读取Gold、历史正确性、图片内容或模型输出。',
        f"两模型选中相同Question词出现位置集合的题数为{sum(r['selected_word_occurrences_equal'] for r in selections)}/240，平均词集合Jaccard为{np.mean([r['selected_word_jaccard'] for r in selections]):.4f}。LLaVA Semantic平均写入{lsem['llava_writer_count_mean']:.2f}Token，Qwen为{lsem['qwen_writer_count_mean']:.2f}；同规则不等于同绝对Token预算。",'',
        '## 主实验十条件','',
        '|条件|正确/240|Accuracy|平均写入Token / R比例|ΔAll pp (95%CI)|All损害/纠正|首Token翻转|',
        '|---|---:|---:|---:|---|---:|---:|']
    for r in summary:
        lines.append(f"|{r['condition']}|{r['correct']}|{r['accuracy_pct']:.3f}%|{r['writer_count_mean']:.2f} / {100*r['writer_ratio_mean']:.2f}%|{r['delta_all_pp']:+.3f} {interval(r,'all_')}|{r['all_harm']}/{r['all_rescue']}|{r['first_flips_all']}|")
    lines+=['','各条件相对No的损益和CI、signed首Token LogP Loss、EOS/空/截断/重复见accuracy_summary.csv；六任务各40题见task_accuracy.csv；全部答案和预算见per_question.csv。','',
        '## 预声明比较','',
        '20,000次Image SHA256整簇配对Bootstrap，固定seed20261103。七项主要对比给出普通95%和Bonferroni7区间；单随机种子补充比较只有普通区间。随机平均在每题内完成，不当成720张独立图片。随机平均的损害/纠正为种子平均数，不作整数二项翻转或McNemar。','',
        '|比较|Δpp|普通95%CI|七项校正区间|','|---|---:|---|---|']
    for r in paired:
        lines.append(f"|{r['comparison']}|{r['delta_pp']:+.3f}|{interval(r)}|{corrected(r) if r['primary_family'] else ''}|")
    lines+=['','## 图像交换与位置混杂','',
        f"冻结40组中，{layout['same_layout']}组原始Token布局、视觉网格与mRoPE完全一致，{layout['different_layout']}组布局不一致。未改变图片分辨率来制造匹配，也未丢弃不匹配配对。No-Writer只在匹配布局组要求所有有效步骤全词表Logits严格一致；不匹配组的差异单独标注为位置编码/布局混杂，不能单凭非零差异判为视觉泄漏。",'',
        '|条件|布局组|对数|答案随图改变|两端均正确|端点正确|平均首分布JS|平均TV|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in swaps:
        js='' if r['mean_first_js'] is None else f"{r['mean_first_js']:.6f}"
        tv='' if r['mean_first_tv'] is None else f"{r['mean_first_tv']:.6f}"
        lines.append(f"|{r['condition']}|{r['layout_group']}|{r['pairs']}|{r['answer_changed']}|{r['both_correct']}|{r['endpoint_correct']}|{js}|{tv}|")
    lines+=['','分布指标为固定首答案预测位置全词表JS（自然对数）及TV，非仅Top1或候选排序。答案改变不等于正确依赖图片；要同时看两端都正确。匹配布局No-Writer逐步精确不变性由image_independence_validation.json与原始数组校验。布局不匹配时No-Writer即使两端都正确，也可能来自布局先验，不能称为图像内容理解。','',
        '## 跨模型表述与负面案例','',
        'cross_model_accuracy.csv及跨模型图采用同240题、同词法算法，但保留各自Tokenizer、原Prompt、原图像处理及FP16/BF16。结果是模型原生协议下的复现，不隔离纯架构效应。不能声称所有VLM或其他数据集都成立。',
        'negative_cases.csv保留所有错误和相对All/No的真实损害；swap_negative_cases.csv保留未能两端都正确的案例。净Accuracy不抵消逐题风险。没有根据输出挑Token、改k、挑随机种子或改层。',
        '低预算保留表现只支持所定义路径的条件功能充分性线索。它不证明唯一存储、不可替代、真正信息瓶颈或严格无损，更没有实现/测得物理加速。','',
        '## 验证与资源','',
        '先CPU测试，再三道按原生输入长度冻结的真实GPU Smoke。原生Full/No-op所有有效Logits同形状torch.equal；十条件Mask概率严格为0、非目标Mask不变、前缀/Self合法、独立真实Cache及父Cache不变、DeepStack特征与原生BF16残差/MLP、mRoPE和正常/异常Hook清理均审计。原生Full首Logits和完整Token另与历史Qwen Full核对，但不复用为All。',
        'Logits不使用近似容差。概率行和采用预先固定的BF16 dtype eps舍入口径；这是概率量化审计，非放宽Logits一致性。模型文件在冻结、GPU加载前和采集后校验全量SHA256。',
        f"单物理GPU{p['gpu']['index']} UUID `{p['gpu']['uuid']}`，只加载一份模型，其他GPU任务不干预；结束正常退出并保存gpu_release.json。原始词表数组存服务器外部资源目录，Git保存路径和哈希，不上传模型/图像或大缓存。",'',
        '所有2400主记录、400交换记录及原始数组重放后才标记COMPLETE。最终只读--check再次复算表格、Bootstrap和文件哈希，不覆盖完成记录。','',
        '本轮到此停止：不扩样、不训练、不搜索写入层、不做Activation Patching、Router或加速实现。','',
        '![质量与写入预算](figure_quality_budget.png)','',
        '![跨模型对照](figure_cross_model.png)','',
        '![换图](figure_image_swap.png)','',
        '![六任务](figure_task_accuracy.png)','']
    path=OUT/'report_zh.md'
    assert not path.exists()
    path.write_text('\n'.join(lines),encoding='utf-8')


def check_tables(tables):
    for name,rows in tables.items():
        actual=read_csv(OUT/name)
        assert len(actual)==len(rows),name
        for left,right in zip(actual,rows):
            assert set(left)==set(right),name
            for k,v in right.items():
                if isinstance(v,float):
                    assert abs(float(left[k])-v)<1e-10,(name,k)
                elif isinstance(v,(list,dict)):
                    assert json.loads(left[k])==v,(name,k)
                else:
                    assert left[k]==('' if v is None else str(v)),(name,k,left[k],v)


def main(check=False):
    p,samples,generations,per,swaps,hashes,raws=replay()
    summary,paired,tasks=statistics(samples,generations,per)
    swap_sums=swap_summary(swaps)
    cross,selections=cross_model(samples,summary)
    negatives=[r for r in per if not r['correct'] or r['all_correct_to_wrong'] or r['no_correct_to_wrong'] or r['empty'] or r['truncated'] or r['repeated_trigram']]
    tables={'per_question.csv':per,'accuracy_summary.csv':summary,'paired_comparisons.csv':paired,'task_accuracy.csv':tasks,
        'image_swap_per_pair.csv':swaps,'image_swap_summary.csv':swap_sums,'cross_model_accuracy.csv':cross,
        'cross_model_token_selection.csv':selections,'negative_cases.csv':negatives,
        'swap_negative_cases.csv':[r for r in swaps if not r['both_correct']]}
    if check:
        check_tables(tables)
        completion=read(OUT/'completion.json')
        assert completion['status']=='COMPLETE' and completion['main_records']==2400 and completion['swap_generations']==400
        for name,h in completion['artifacts'].items():
            assert sha256(OUT/name)==h,name
        v=read(OUT/'validation.json')
        assert v['checkpoint_hashes']==hashes and v['raw_hashes']==raws
        print('READ-ONLY PASS: 2800 records, masks/cache/mRoPE/raw arrays, tables, statistics and artifact hashes',flush=True)
        return
    assert not (OUT/'completion.json').exists(), 'Already aggregated: use --check'
    for name,rows in tables.items():
        table(OUT/name,rows)
    figures(summary,paired,cross,swap_sums,tasks)
    report(p,summary,paired,cross,selections,swap_sums)
    write(OUT/'validation.json',dict(passed=True,binding=digest(p),main_records=2400,swap_generations=400,
        checkpoint_hashes=hashes,raw_hashes=raws,mask_cache_mrope_replayed=True,
        bootstrap_repetitions=20000,statistics_replayed=True,no_tolerance_relaxation=True))
    artifacts={str(path.relative_to(OUT)):sha256(path) for path in sorted(OUT.rglob('*')) if path.is_file()
        and path.name not in ['completion.json','worker.lock','analysis.log','verification.log']}
    write(OUT/'completion.json',dict(status='COMPLETE',binding=digest(p),main_records=2400,
        new_main_generations=2400,swap_generations=400,code_commit=p['code_commit'],artifacts=artifacts))
    print('COMPLETE: all2400 new main /400 swaps and cross-model analysis',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    try:
        main(args.check)
    except BaseException:
        if not args.check:
            import time,traceback
            write(OUT/'failures'/f'analysis_{time.time_ns()}.json',dict(status='PARTIAL',traceback=traceback.format_exc()))
        raise
