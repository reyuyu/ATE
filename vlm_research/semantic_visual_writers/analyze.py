"""Replay immutable checkpoints, paired image bootstrap, and Chinese reporting."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from query_role_attention.analyze import paired_bootstrap
from query_role_formal.common import repeated_trigram, divergence
from first_answer_attention.engine import first_metrics
from paper_rq1.scorers import gqa_prediction
from .common import *


def validate(r, s, spec, inputs, protocol, tokenizer):
    g = r['generation']
    condition = r['condition']
    assert r['binding'] == digest(protocol) and r['sample_sha256'] == digest(s)
    assert r['result_sha256'] == digest(g) and r['selection_binding'] == spec['selection_binding']
    assert r['input_hashes'] == inputs and r['image_sha256'] == s['image_sha256'] and r['gold'] == s['answer']
    writers = spec['selection']['writers'][condition]
    assert r['writer_positions'] == writers and r['writer_count'] == len(writers)
    assert r['relay_count'] == len(spec['relay_positions']) and g['own_cache'] and g['own_prefix']
    assert tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip() == g['text']
    assert g['prediction'] == gqa_prediction(g['text'])
    assert g['correct'] == int(g['prediction'] == s['answer'])
    assert g['length'] == len(g['token_ids']) and 1 <= g['length'] <= 32
    assert g['eos'] == (g['token_ids'][-1] == 2)
    assert g['empty'] == (not bool(g['text']))
    assert g['truncated'] == (g['length'] == 32 and not g['eos'])
    assert g['repeated_trigram'] == repeated_trigram(g['token_ids'])
    logits = first_logits(r)
    assert int(logits.argmax()) == g['token_ids'][0] and digest(logits.tolist()) == g['first_logits_sha256']
    if r['provenance']['reused']:
        assert condition == 'All-Writer'
        assert sha256(ROOT/r['provenance']['checkpoint']) == r['provenance']['sha256']
        assert g['writer_layers'] == list(range(32)) and g['answer_image_blocked']
    else:
        assert g['writer_positions'] == writers and not g['noop'] and g['all_logits_finite']
        assert g['logits_to_keep'] == 1 and g['answer_image_blocked']
    layout = g['query_layout']
    n = layout['prompt_length']
    assert layout['R'] == spec['relay_positions'] and len(layout['I']) == 576
    assert len(g['source_events']) == 32*g['length'] and len(g['cache_lengths']) == g['length']
    for t in range(g['length']):
        before = n+t-1 if t else 0
        assert g['cache_lengths'][t] == dict(t=t,before=before,after=n+t,
            consumed_token=g['token_ids'][t-1] if t else None,absolute_query=n+t-1)
        for k,e in enumerate(g['source_events'][32*t:32*t+32]):
            assert e['layer'] == k and e['phase'] == ('decode' if t else 'prefill')
            assert e['absolute_start'] == before and e['query_length'] == (1 if t else n) and e['key_length'] == n+t
            expected = 576*(1+(len(layout['R'])-len(writers) if not t else 0))
            assert e['blocked_connections_per_head'] == expected
            assert e['blocked_probability_max'] == 0 and e['row_sum_max_error'] <= .002
            assert e['non_target_mask_unchanged']
            if r['provenance']['reused']:
                assert e['answer_image_masked'] and not e['R_image_masked'] and e['BOS_Self_legal']
            else:
                assert e['answer_image_blocked'] and e['BOS_self_legal'] and e['writer_queries'] == len(writers)
    return g,logits


def distribution_metrics(left,right):
    exact = torch.equal(left,right)
    lp,lq = left.double().log_softmax(-1),right.double().log_softmax(-1)
    lm = torch.logaddexp(lp,lq)-math.log(2)
    js = .5*((lp.exp()*(lp-lm)).sum()+(lq.exp()*(lq-lm)).sum())
    return dict(first_logits_exact=exact, first_logits_max_abs=float((left-right).abs().max()),
        first_distribution_js=0. if exact else max(0.,float(js)),
        first_distribution_tv=0. if exact else float(.5*(lp.exp()-lq.exp()).abs().sum()))


def replay():
    protocol = verify()
    assert read(OUT/'gpu_completion.json')['passed']
    assert read(OUT/'gpu_completion.json')['binding'] == digest(protocol)
    assert read(OUT/'gpu_smoke_validation.json')['passed'] and read(OUT/'image_independence_validation.json')['passed']
    assert len(list((OUT/'checkpoints').glob('*/*.json'))) == 2400
    assert len(list((OUT/'swap_checkpoints').glob('*/*.json'))) == 400
    samples = read(OUT/'sample_manifest.json')['samples']
    mappings = read(OUT/'input_mappings.json')
    specs = read(OUT/'token_group_manifest.json')
    tokenizer = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    generations = {c:[] for c in CONDITIONS}
    per, hashes, raw_hashes = [], {}, {}
    reused = 0
    for s,mapping in zip(samples,mappings):
        spec = specs['main'][key(s)]
        assert sha256(s['image_path']) == s['image_sha256']
        records, vectors, source = {}, {}, {}
        for c in CONDITIONS:
            path = OUT/'checkpoints'/key(s)/(c+'.json')
            r = read(path)
            g,v = validate(r,s,spec,mapping['input_hashes'],protocol,tokenizer)
            assert g['query_layout'] == mapping['role_layout']
            records[c],vectors[c],source[c] = g,v,r
            generations[c].append(g)
            hashes[str(path.relative_to(OUT))] = sha256(path)
            raw_hashes[r['raw']['path']] = r['raw']['sha256']
            reused += int(r['provenance']['reused'])
        for c,g in records.items():
            a,b = records['All-Writer'],records['No-Writer']
            fa,fb = first_metrics(vectors['All-Writer'],vectors[c]),first_metrics(vectors['No-Writer'],vectors[c])
            d = divergence(a['token_ids'],g['token_ids'])
            writers = spec['selection']['writers'][c]
            per.append(dict(question_id=s['question_id'],task=s['task'],image_id=s['image_id'],image_sha256=s['image_sha256'],
                question=s['question'],gold=s['answer'],condition=c,text=g['text'],prediction=g['prediction'],
                token_ids=json.dumps(g['token_ids']),correct=g['correct'],All_correct=a['correct'],No_correct=b['correct'],
                all_correct_to_wrong=int(a['correct'] and not g['correct']),all_wrong_to_correct=int(not a['correct'] and g['correct']),
                no_correct_to_wrong=int(b['correct'] and not g['correct']),no_wrong_to_correct=int(not b['correct'] and g['correct']),
                first_top1_flip_all=fa['first_top1_flip'],first_top1_flip_no=fb['first_top1_flip'],
                first_logp_loss_all=fa['first_logp_loss'],first_logp_loss_no=fb['first_logp_loss'],
                first_divergence_all='' if d is None else d,sequence_changed_all=g['token_ids'] != a['token_ids'],
                length=g['length'],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],
                writer_count=len(writers),relay_count=len(spec['relay_positions']),writer_ratio=len(writers)/len(spec['relay_positions']),
                k=spec['selection']['k'],question_token_count=len(spec['selection']['question_positions']),
                writer_positions=json.dumps(writers),writer_tokens_decoded=spec['writers_decoded'][c],
                core_candidate_tokens=spec['selection']['candidate_core_tokens'],
                whole_word_budget_reduction=spec['selection']['budget_reduction_for_whole_word_suffix'],
                scaffold_deficit=spec['selection']['scaffold_budget_deficit'],
                reused=source[c]['provenance']['reused'],input_hashes_sha256=digest(mapping['input_hashes']),
                selection_binding=spec['selection_binding'],raw_logits_path=source[c]['raw']['path'],raw_logits_sha256=source[c]['raw']['sha256']))
    assert reused == 240
    swaps = []
    swap_inputs = read(OUT/'image_swap_inputs.json')
    for pair in read(OUT/'image_swap_manifest.json')['pairs']:
        for c in SWAP_CONDITIONS:
            sides = []
            for side,s in enumerate(pair['samples']):
                stem = f"{pair['pair_id']}__{side}"
                path = OUT/'swap_checkpoints'/stem/(c+'.json')
                r = read(path)
                assert not r['provenance']['reused']
                g,v = validate(r,s,specs['swaps'][stem],swap_inputs[stem]['input_hashes'],protocol,tokenizer)
                sides.append((g,v,r))
                hashes[str(path.relative_to(OUT))] = sha256(path)
                raw_hashes[r['raw']['path']] = r['raw']['sha256']
            a,b = sides
            metrics = distribution_metrics(a[1],b[1])
            trace_exact = None
            if c == 'No-Writer':
                traces = []
                for _,_,r in sides:
                    artifact = r['valid_trace_raw']
                    assert sha256(artifact['path']) == artifact['sha256']
                    raw_hashes[artifact['path']] = artifact['sha256']
                    with np.load(artifact['path'],allow_pickle=False) as data:
                        traces.append(torch.from_numpy(data['valid_logits'].copy()))
                trace_exact = torch.equal(*traces)
                assert trace_exact and metrics['first_logits_exact']
                assert a[0]['token_ids'] == b[0]['token_ids']
            swaps.append(dict(pair_id=pair['pair_id'],condition=c,question=pair['samples'][0]['question'],
                task_left=pair['samples'][0]['task'],task_right=pair['samples'][1]['task'],
                qid_left=pair['samples'][0]['question_id'],qid_right=pair['samples'][1]['question_id'],
                image_sha_left=pair['samples'][0]['image_sha256'],image_sha_right=pair['samples'][1]['image_sha256'],
                gold_left=pair['samples'][0]['answer'],gold_right=pair['samples'][1]['answer'],
                text_left=a[0]['text'],text_right=b[0]['text'],prediction_left=a[0]['prediction'],prediction_right=b[0]['prediction'],
                correct_left=a[0]['correct'],correct_right=b[0]['correct'],both_correct=int(a[0]['correct'] and b[0]['correct']),
                answer_changed=a[0]['prediction'] != b[0]['prediction'],sequence_changed=a[0]['token_ids'] != b[0]['token_ids'],
                first_top1_changed=a[0]['token_ids'][0] != b[0]['token_ids'][0],
                writer_count_left=a[2]['writer_count'],writer_count_right=b[2]['writer_count'],
                no_writer_all_step_logits_exact=trace_exact,**metrics))
    return protocol,samples,generations,per,swaps,hashes,raw_hashes


def statistics(samples,generations,per):
    matrix = np.array([[g['correct'] for g in generations[c]] for c in CONDITIONS]).T.astype(float)
    draws = paired_bootstrap(samples,matrix,repetitions=20000,seed=SEED)*100
    random_idx = [CONDITIONS.index(f'Random-k-{seed}') for seed in RANDOM_SEEDS]
    random_mean = matrix[:,random_idx].mean(1)
    random_draws = draws[:,random_idx].mean(1)
    idx = {c:i for i,c in enumerate(CONDITIONS)}
    summary = []
    for c in CONDITIONS:
        j = idx[c]
        rows = [r for r in per if r['condition'] == c]
        all_ci = np.quantile(draws[:,j]-draws[:,0],[.025,.975])
        no_ci = np.quantile(draws[:,j]-draws[:,1],[.025,.975])
        summary.append(dict(condition=c,questions=240,correct=int(matrix[:,j].sum()),accuracy_pct=float(matrix[:,j].mean()*100),
            delta_all_pp=float((matrix[:,j]-matrix[:,0]).mean()*100),all_ci_low=float(all_ci[0]),all_ci_high=float(all_ci[1]),
            delta_no_pp=float((matrix[:,j]-matrix[:,1]).mean()*100),no_ci_low=float(no_ci[0]),no_ci_high=float(no_ci[1]),
            all_harm=sum(r['all_correct_to_wrong'] for r in rows),all_rescue=sum(r['all_wrong_to_correct'] for r in rows),
            no_harm=sum(r['no_correct_to_wrong'] for r in rows),no_rescue=sum(r['no_wrong_to_correct'] for r in rows),
            first_flips_all=sum(r['first_top1_flip_all'] for r in rows),first_flips_no=sum(r['first_top1_flip_no'] for r in rows),
            first_logp_loss_all_mean=float(np.mean([r['first_logp_loss_all'] for r in rows])),
            first_logp_loss_no_mean=float(np.mean([r['first_logp_loss_no'] for r in rows])),
            writer_count_mean=float(np.mean([r['writer_count'] for r in rows])),
            writer_ratio_mean=float(np.mean([r['writer_ratio'] for r in rows])),
            writer_count_min=min(r['writer_count'] for r in rows),writer_count_max=max(r['writer_count'] for r in rows),
            eos=sum(r['eos'] for r in rows),empty=sum(r['empty'] for r in rows),truncated=sum(r['truncated'] for r in rows),
            repeated_trigram=sum(r['repeated_trigram'] for r in rows)))
    contrasts = [('Semantic-k','Random-mean(3)'),('Semantic-k','Question-Last-k'),('Semantic-k','R-Last-k'),
        ('Semantic-k','Scaffold-k'),('Semantic-k','All-Writer'),('Semantic-k','No-Writer'),('Drop-Semantic','All-Writer')]
    contrasts += [('Semantic-k',f'Random-k-{seed}') for seed in RANDOM_SEEDS]
    paired = []
    for k,(a,b) in enumerate(contrasts):
        x,xd = matrix[:,idx[a]],draws[:,idx[a]]
        y,yd = (random_mean,random_draws) if b == 'Random-mean(3)' else (matrix[:,idx[b]],draws[:,idx[b]])
        delta,delta_draws = x-y,xd-yd
        ci = np.quantile(delta_draws,[.025,.975])
        adjusted = np.quantile(delta_draws,[.025/7,1-.025/7]) if k < 7 else [None,None]
        paired.append(dict(comparison=a+' minus '+b,delta_pp=float(delta.mean()*100),ci_low=float(ci[0]),ci_high=float(ci[1]),
            bonferroni7_low=None if adjusted[0] is None else float(adjusted[0]),
            bonferroni7_high=None if adjusted[1] is None else float(adjusted[1]),primary_family=k<7,
            harm_or_seed_average_harm=float(((1-x)*y).sum()),rescue_or_seed_average_rescue=float((x*(1-y)).sum()),
            fractional_reference=b=='Random-mean(3)',clusters=240,repetitions=20000,seed=SEED))
    tasks = []
    for task in TASKS:
        for c in CONDITIONS:
            rows = [r for r in per if r['task']==task and r['condition']==c]
            tasks.append(dict(task=task,condition=c,questions=len(rows),correct=sum(r['correct'] for r in rows),
                accuracy_pct=float(np.mean([r['correct'] for r in rows])*100),
                delta_all_pp=float(np.mean([r['correct']-r['All_correct'] for r in rows])*100),
                delta_no_pp=float(np.mean([r['correct']-r['No_correct'] for r in rows])*100),
                all_harm=sum(r['all_correct_to_wrong'] for r in rows),all_rescue=sum(r['all_wrong_to_correct'] for r in rows)))
    strata = []
    for low,high,label in [(1,6,'k1-6'),(7,10,'k7-10'),(11,10**9,'k11+')]:
        for c in CONDITIONS:
            rows = [r for r in per if r['condition']==c and low<=r['k']<=high]
            if rows:
                strata.append(dict(stratum=label,condition=c,n=len(rows),k_mean=float(np.mean([r['k'] for r in rows])),
                    writer_count_mean=float(np.mean([r['writer_count'] for r in rows])),
                    writer_ratio_mean=float(np.mean([r['writer_ratio'] for r in rows])),
                    accuracy_pct=float(np.mean([r['correct'] for r in rows])*100)))
    return summary,paired,tasks,strata


def swap_summary(swaps):
    result = []
    for c in SWAP_CONDITIONS:
        rows = [r for r in swaps if r['condition']==c]
        assert len(rows)==40
        result.append(dict(condition=c,pairs=40,answer_changed=sum(r['answer_changed'] for r in rows),
            sequence_changed=sum(r['sequence_changed'] for r in rows),both_correct=sum(r['both_correct'] for r in rows),
            endpoint_correct=sum(r['correct_left']+r['correct_right'] for r in rows),
            first_top1_changed=sum(r['first_top1_changed'] for r in rows),
            mean_first_js=float(np.mean([r['first_distribution_js'] for r in rows])),
            mean_first_tv=float(np.mean([r['first_distribution_tv'] for r in rows])),
            first_logits_equal=sum(r['first_logits_exact'] for r in rows),
            no_writer_all_step_exact=sum(r['no_writer_all_step_logits_exact'] is True for r in rows)))
    return result


def figures(summary,paired,tasks,strata,swaps):
    colors = {'Semantic-k':'#247f68','Question-Last-k':'#477bb6','R-Last-k':'#c96d30',
              'Scaffold-k':'#8d62a0','All-Writer':'#303030','No-Writer':'#8c8c8c','Drop-Semantic':'#bb4545'}
    labels = {'Semantic-k':'Semantic','Question-Last-k':'Question last','R-Last-k':'R last',
              'Scaffold-k':'Scaffold','All-Writer':'All','No-Writer':'None','Drop-Semantic':'Drop semantic'}
    fig,axes = plt.subplots(1,2,figsize=(13,5.2),constrained_layout=True)
    for i,r in enumerate(summary):
        c=r['condition']
        if c.startswith('Random-'):
            color,label,marker='#4c8b98','Random '+c.split('-')[-1],['v','s','D'][RANDOM_SEEDS.index(int(c.split('-')[-1]))]
        else:
            color,label,marker=colors[c],labels[c],'o'
        axes[0].scatter(100*r['writer_ratio_mean'],r['accuracy_pct'],color=color,marker=marker,s=65,label=label)
    axes[0].set(xlabel='Mean fraction of R directly reading image keys (%)',ylabel='Complete-answer accuracy (%)',
                title='Fixed policies, same 240 images',xlim=(-3,103))
    axes[0].legend(frameon=False,fontsize=7,loc='lower right',ncol=2)
    selected=paired[:4]
    for i,r in enumerate(selected):
        axes[1].errorbar(r['delta_pp'],i,xerr=[[r['delta_pp']-r['ci_low']],[r['ci_high']-r['delta_pp']]],fmt='o',capsize=4,color='#247f68')
    axes[1].axvline(0,color='gray',linewidth=.8)
    axes[1].set(yticks=range(4),yticklabels=['vs random mean (3)','vs question last','vs R last','vs scaffold'],
        xlabel='Semantic accuracy minus matched control (pp), 95% CI',title='20,000 paired image-cluster bootstrap')
    axes[1].invert_yaxis()
    fig.suptitle('Which Text Tokens Need to See? / LLaVA-1.5-7B')
    save_figure(fig,'figure_quality_budget')
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    for c in ['Semantic-k','Question-Last-k','R-Last-k','Scaffold-k']:
        values=[r for r in strata if r['condition']==c]
        ax.plot([r['writer_count_mean'] for r in values],[r['accuracy_pct'] for r in values],
                'o-',color=colors[c],label=labels[c])
    values=[]
    for label in ['k1-6','k7-10','k11+']:
        rows=[r for r in strata if r['stratum']==label and r['condition'].startswith('Random-')]
        if rows:values.append((np.mean([r['writer_count_mean'] for r in rows]),np.mean([r['accuracy_pct'] for r in rows])))
    ax.plot([r[0] for r in values],[r[1] for r in values],'s--',color='#4c8b98',label='Random mean (3)')
    ax.set(xlabel='Mean allowed writer tokens within input-only k stratum',ylabel='Complete-answer accuracy (%)',
           title='Input-only k strata (not a within-question budget sweep)')
    ax.legend(frameon=False,fontsize=9)
    save_figure(fig,'figure_budget_strata')
    fig,ax=plt.subplots(figsize=(9,5.5),constrained_layout=True)
    values=np.array([[next(r['accuracy_pct'] for r in tasks if r['task']==t and r['condition']==c) for t in TASKS] for c in CONDITIONS])
    im=ax.imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto')
    ax.set(xticks=range(6),xticklabels=TASKS,yticks=range(10),yticklabels=CONDITIONS,title='Six tasks / 40 images each')
    plt.setp(ax.get_xticklabels(),rotation=20,ha='right')
    for i in range(10):
        for j in range(6):ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]<65 else 'black')
    fig.colorbar(im,ax=ax,label='Accuracy (%)')
    save_figure(fig,'figure_task_accuracy')
    fig,ax=plt.subplots(figsize=(9,4.6),constrained_layout=True)
    x=np.arange(5)
    ax.bar(x-.18,[r['answer_changed'] for r in swaps],width=.36,label='Answer changes with image',color='#477bb6')
    ax.bar(x+.18,[r['both_correct'] for r in swaps],width=.36,label='Both endpoints correct',color='#247f68')
    ax.set(xticks=x,xticklabels=['All','None','Semantic','R last','Random seed1'],ylabel='Pairs (of 40)',ylim=(0,42),title='Frozen image-swap controls (separate from main 240)')
    ax.legend(frameon=False,fontsize=8)
    save_figure(fig,'figure_image_swap')


def save_figure(fig,name):
    for suffix in ['png','pdf']:
        fig.savefig(OUT/(name+'.'+suffix),dpi=180)
    plt.close(fig)


def interval(row,prefix=''):
    return f"[{row[prefix+'ci_low']:+.3f}, {row[prefix+'ci_high']:+.3f}]"


def report(protocol,summary,paired,tasks,strata,swaps,per):
    by={r['condition']:r for r in summary}
    sem,all_,no=by['Semantic-k'],by['All-Writer'],by['No-Writer']
    budget=read(OUT/'token_budget_audit.json')
    random_mean=np.mean([by[f'Random-k-{seed}']['accuracy_pct'] for seed in RANDOM_SEEDS])
    def decision(r):
        if r['ci_low']>0:return '普通95%CI支持正向差异'
        if r['ci_high']<0:return '普通95%CI支持负向差异'
        return '区间含0，未建立清晰差异；也不是等价证明'
    lines=['# Which Text Tokens Need to See?：稀疏视觉写入位置探索','',
        '**COMPLETE：固定240题×10条件共2400条主记录，其中240条All-Writer经严格审计复用，新生成2160条；另有40组×2端×5条件共400条换图生成，分开统计。**','',
        '## 核心回答','',
        f"Semantic-k为{sem['correct']}/240（{sem['accuracy_pct']:.3f}%），平均开放{sem['writer_count_mean']:.2f}个R Token（平均R比例{sem['writer_ratio_mean']*100:.2f}%）。All-Writer为{all_['correct']}/240，No-Writer为{no['correct']}/240。",'',
        f"**1. 语义是否优于任意同预算Token？** 三个预定随机种子平均Accuracy为{random_mean:.3f}%；Semantic−Random平均为{paired[0]['delta_pp']:+.3f} pp，95%CI {interval(paired[0])}，{decision(paired[0])}。不挑最佳随机种子，不将三个种子视为720张独立图片。",'',
        f"**2. 语义还是末端汇总？** Semantic−Question-Last为{paired[1]['delta_pp']:+.3f} pp，CI {interval(paired[1])}；Semantic−R-Last为{paired[2]['delta_pp']:+.3f} pp，CI {interval(paired[2])}。前者{decision(paired[1])}；后者{decision(paired[2])}。位置效应与词汇效应只能按这些直接配对结果判断，不能预设语义优势。",'',
        f"**3. 稀疏写入是否足够？** Semantic−All为{sem['delta_all_pp']:+.3f} pp，CI {interval(sem,'all_')}，原对变错{sem['all_harm']}题、原错变对{sem['all_rescue']}题；Semantic−No为{sem['delta_no_pp']:+.3f} pp，CI {interval(sem,'no_')}。前一差值衡量是否接近全开放，后一差值检验是否确有写入功能贡献。未预声明非劣效界限，不能由Accuracy接近或CI跨0宣称无损。",'',
        f"**4. 核心Token是否不可替代？** Drop-Semantic−All为{paired[6]['delta_pp']:+.3f} pp，CI {interval(paired[6])}，{decision(paired[6])}。这是移除最终Semantic-k集合直接读图权限的条件必要性检验，不排除其他文本Token或层间传播替代它们。",'',
        '总体证据只支持本批固定候选策略在所定义路径上的相对功能贡献。即使少数Token支持较多正确答案，也不能说视觉信息唯一储存在这些Token，或任意Token普遍足够。所有稀疏策略都保持其余文本传播和32层MLP正常。本实验没有实现或测得实际加速。','',
        '## 条件、范围与词法定义','',
        '全部条件均在0-based L0–L31禁止P−1及每一步真实Decode（包括EOS预测）直接读原始576个Image Key。Prefill仅选中的R Query在全部32层开放Image Key；其他R Query封闭。只修改Softmax前Query→Key Hard Mask；BOS/Self、原因果性、其余来源、Attention输出、MLP、Residual及真实KV更新不变。Mask会重归一化剩余来源。',
        'R为原图像结束后、P−1之前的全部文本位置，包括真实Question、指令、模板和空白，不仅是Question。原Prompt、图像、Tokenizer和输入映射完全保留。','',
        'Semantic是预先冻结的**词法启发式**，不是经过验证的语义标注器：排除封闭功能词表后保留Question中的核心候选整词，包括对象/属性词、数字、关系词、疑问类型、逻辑运算和指代锚点。选择器函数只接收文本及Tokenizer映射，不接收Gold、图像、历史正确性或模型输出。不要将未检出优势外推为真实语义不重要。',
        f"为同时满足完整Subword词和精确末端k-Token对照，取候选核心整词中最大可行预算；Semantic按固定从左到右子集规则破同分。{budget['reduced_for_whole_word_suffix']}/240题的最终集合少于全部词法候选，k范围{budget['k_min']}–{budget['k_max']}，均非空。候选集、最终集及减少数量都保存在Manifest/逐题CSV。Drop-Semantic严格取**最终Semantic-k集合的补集**，并不是额外剥夺未入选候选词的读图权限。", 
        'Question-Last和R-Last确为对应序列最后k个Tokenizer Token，且不切开任何整词组。Random按三个固定种子打乱R整词组，用可达预算DP选精确k，属于整词约束随机对照，不声称对子词独立均匀采样。随机键仅使用问题文本和seed，换图两端集合相同。',
        f"Scaffold仅从非核心候选组按位置取最大不超过k的可行预算，本批不足预算为{budget['scaffold_deficits']}题。规则仍会记录不足而不删除题。",'',
        '## 十条件完整答案结果','',
        '|条件|正确/240|Accuracy|平均写入Token / R比例|ΔAll pp (95%CI)|ΔNo pp (95%CI)|All损害/纠正|首Token翻转vsAll|',
        '|---|---:|---:|---:|---|---|---:|---:|']
    for r in summary:
        lines.append(f"|{r['condition']}|{r['correct']}|{r['accuracy_pct']:.3f}%|{r['writer_count_mean']:.2f} / {100*r['writer_ratio_mean']:.2f}%|{r['delta_all_pp']:+.3f} {interval(r,'all_')}|{r['delta_no_pp']:+.3f} {interval(r,'no_')}|{r['all_harm']}/{r['all_rescue']}|{r['first_flips_all']}|")
    lines+=['','## 同预算主要配对','',
        '20,000次Image SHA256整簇配对Bootstrap，seed=20261103，240张不同图片各一题，所有条件复用同一抽样。随机平均先在每题内合并三个seed，再按图像抽样。全部为探索性普通95%CI，另列七项预先声明主要对照的Bonferroni区间；单seed对照仅给普通区间。任务与预算分层描述性分析不用于事后选择策略。','',
        '|比较|ΔAccuracy pp|普通95%CI|七项校正区间|', '|---|---:|---|---|']
    for r in paired:
        adjusted='' if not r['primary_family'] else f"[{r['bonferroni7_low']:+.3f}, {r['bonferroni7_high']:+.3f}]"
        lines.append(f"|{r['comparison']}|{r['delta_pp']:+.3f}|{interval(r)}|{adjusted}|")
    lines+=['','## 六类GQA','', '|条件|'+'|'.join(TASKS)+'|','|---|'+'---:|'*6]
    for c in CONDITIONS:
        lines.append('|'+c+'|'+'|'.join(f"{next(r['accuracy_pct'] for r in tasks if r['condition']==c and r['task']==t):.1f}%" for t in TASKS)+'|')
    lines+=['','每类40题，不按任务结果过滤样本或调规则。逐题全部答案、首Token翻转、signed LogP Loss、Token预算与损益见per_question.csv；No参考的损害/纠正及各任务细节见accuracy_summary.csv、task_accuracy.csv。','',
        '## 图像交换控制','',
        '复用历史冻结40组同Question、异图像、异Gold配对，主240题之外单独计数，不重选对模型有利的案例。五条件共400条完整生成。No-Writer在主矩阵前完成所有40组逐步Logits严格torch.equal门控；失败则停止并排查图像路径。其他条件的分布差异是固定首Token全词表分布的JS散度（自然对数）和TV距离，不是仅二候选排序。','',
        '|条件|答案随图改变/40|两端均正确/40|端点正确/80|首Token改变/40|平均JS|平均TV|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in swaps:
        lines.append(f"|{r['condition']}|{r['answer_changed']}|{r['both_correct']}|{r['endpoint_correct']}|{r['first_top1_changed']}|{r['mean_first_js']:.6f}|{r['mean_first_tv']:.6f}|")
    lines+=['','No-Writer全部有效Logits严格相同，首Token JS/TV和最大绝对差异均为0；不同Gold的两端不可能同时正确。它的主实验正确题应解释为语言先验等非图像来源，而不能冒充视觉理解。答案变化不等于正确使用图像，因此必须同时看两端都正确及端点正确率。','',
        '## 验证、复用与资源','',
        f"固定LLaVA-1.5-7B revision `{REVISION}`，原FP16/eager，Transformers4.49.0，Greedy、最多32 Token、显式logits_to_keep=1。源码冻结Commit `{protocol['code_commit']}`。",
        '15项新增CPU测试及历史回归覆盖：Mask方向、BOS/Self、合法行非空、禁止Softmax概率严格为0、原因果性、整词分组、精确预算、固定随机种子及清理。三道固定真实GPU Smoke覆盖十条件，HF原生Full与No-op全部返回有效Logits同形状torch.equal；独立Cache真实存储、父Cache不变、32层MLP、绝对位置、正常/异常Hook清理、输入及权重版本均验证。未放宽容差；概率行和沿用历史0.002 FP16口径。',
        '历史All-Writer来自同协议visual_write_window_240，CPU逐题验证240检查点、输入/图像哈希、来源Mask事件、词表原始Logits、绑定与模型版本，GPU三题再次首向量及完整Token严格一致后复用。原生Full允许答案读图，绝不当作All-Writer。新主实验2160条、交换400条，没有重复跑240条All。',
        f"Manifest逐字节复用SHA256 `{MANIFEST_SHA}`；物理GPU{protocol['gpu']['index']} UUID `{protocol['gpu']['uuid']}`，只加载一份模型、单卡串行，未抢占或终止其他GPU进程。资源释放快照单独保存。",
        '所有2800主/交换记录逐条审计，检查点保留每层每步实际禁止概率、Mask作用范围和Cache；模型权重文件在冻结前与采集后全量SHA256复核。原始词表数组只保存在服务器资源目录，Git保存路径和哈希，避免上传大缓存。','',
        '## 负面案例及解释边界','',
        'negative_cases.csv包含每个条件下全部错误/空/截断/重复以及相对All或No的损害；swap_negative_cases.csv保留未能两端都正确的配对。没有删除反例。精确相同生成序列的first_divergence_all写空字段，不将None字符串误判为失败。',
        '质量—预算总图比较实际冻结策略，不是逐题预算扫描。预算分层图使用输入预先确定的k区间1–6、7–10、≥11；不同分层题目不同，不能从曲线斜率推出增加预算的因果收益。',
        '“允许直接读图”是路径权限的操作定义；开放某Token后预测较好不证明视觉信息只存于该Token。其他R和答案可通过文本来源间接读取信息；图像KV也已历经此前层计算。稀疏集合能恢复部分任务，最多支持特定条件下的功能充分性线索，不等于严格无损、唯一存储或普遍不可替代。',
        '本轮到此停止：不扩样，不做层×Token联合搜索、Activation Patching、Router训练、Kernel或真实加速实现。','',
        '![质量与写入预算](figure_quality_budget.png)','', '![预算分层](figure_budget_strata.png)','',
        '![六任务结果](figure_task_accuracy.png)','', '![图像交换](figure_image_swap.png)','']
    assert not (OUT/'report_zh.md').exists()
    (OUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def check_tables(tables):
    for name,rows in tables.items():
        actual=read_csv(OUT/name)
        assert len(actual)==len(rows),name
        for left,right in zip(actual,rows):
            assert set(left)==set(right),name
            for k,v in right.items():
                if isinstance(v,float):assert abs(float(left[k])-v)<1e-10,(name,k)
                elif v is None:assert left[k]=='',(name,k)
                else:assert left[k]==str(v),(name,k,left[k],v)


def main(check=False):
    protocol,samples,generations,per,swaps,hashes,raws=replay()
    summary,paired,tasks,strata=statistics(samples,generations,per)
    swap_sums=swap_summary(swaps)
    negatives=[r for r in per if not r['correct'] or r['all_correct_to_wrong'] or r['no_correct_to_wrong']
               or r['empty'] or r['truncated'] or r['repeated_trigram']]
    swap_negatives=[r for r in swaps if not r['both_correct']]
    tables={'per_question.csv':per,'accuracy_summary.csv':summary,'paired_comparisons.csv':paired,
        'task_accuracy.csv':tasks,'budget_strata.csv':strata,'image_swap_per_pair.csv':swaps,
        'image_swap_summary.csv':swap_sums,'negative_cases.csv':negatives,'swap_negative_cases.csv':swap_negatives}
    if check:
        check_tables(tables)
        c=read(OUT/'completion.json')
        assert c['status']=='COMPLETE' and c['main_records']==2400 and c['swap_generations']==400
        for name,h in c['artifacts'].items():assert sha256(OUT/name)==h,name
        v=read(OUT/'validation.json')
        assert v['checkpoint_hashes']==hashes and v['raw_hashes']==raws
        print('READ-ONLY PASS: 2800 records, masks/cache/raw arrays, tables, bootstrap and delivery hashes',flush=True)
        return
    assert not (OUT/'completion.json').exists(),'Already aggregated; use --check'
    assert read(OUT/'gpu_release.json')['passed']
    for name,rows in tables.items():csv_write(OUT/name,rows)
    figures(summary,paired,tasks,strata,swap_sums)
    report(protocol,summary,paired,tasks,strata,swap_sums,per)
    write(OUT/'validation.json',dict(passed=True,binding=digest(protocol),main_records=2400,
        new_main_generations=2160,reused_main_generations=240,swap_generations=400,
        checkpoint_hashes=hashes,raw_hashes=raws,source_masks_replayed=True,cache_traces_replayed=True,
        statistics_replayed=True,bootstrap_repetitions=20000,no_tolerance_relaxation=True))
    artifacts={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name not in ['completion.json','worker.lock','analysis.log','verification.log']}
    write(OUT/'completion.json',dict(status='COMPLETE',complete=True,binding=digest(protocol),
        main_records=2400,new_main_generations=2160,reused_main_generations=240,swap_generations=400,
        code_commit=protocol['code_commit'],artifacts=artifacts))
    print('COMPLETE: main2400 (new2160/reused240), swap400; all reporting finished',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    try:
        main(args.check)
    except BaseException:
        if not args.check:
            import traceback,time
            write(OUT/'failures'/f'analysis_{time.time_ns()}.json',dict(status='PARTIAL',traceback=traceback.format_exc()))
        raise
