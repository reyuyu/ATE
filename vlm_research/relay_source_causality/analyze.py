"""Replay main, frozen swaps and unscored Question-only diagnostic separately."""
import argparse,csv,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from query_role_attention.analyze import paired_bootstrap,summary
from query_role_formal.common import divergence,repeated_trigram
from .common import *
from .engine import source_edges

PAIRS=[('R-Bypass','Full'),('R-Image-OFF','Full'),('R-Text-OFF','Full'),('R-Image-OFF','R-Text-OFF')]

def raw(row):
    assert sha256(row['raw']['path'])==row['raw']['sha256']
    with np.load(row['raw']['path']) as data:return torch.from_numpy(data['first_logits'].copy())

def js(a,b):
    # FP64 complete-vocabulary distributions; clamp only roundoff below zero.
    if torch.equal(a,b):return 0.
    pa=torch.softmax(a.double(),-1);pb=torch.softmax(b.double(),-1);m=(pa+pb)/2
    out=.5*(torch.xlogy(pa,pa/m).sum()+torch.xlogy(pb,pb/m).sum())
    assert torch.isfinite(out) and float(out)>=-1e-12
    return max(0.,float(out))

def validate_record(row,s,tokenizer,binding,input_hashes,full_logits=None):
    g=row['generation'];assert row['binding']==binding and row['sample_sha256']==digest(s) and row['result_sha256']==digest(g)
    assert row['input_hashes']==input_hashes and row['image_sha256']==s['image_sha256'] and row['gold']==s['answer']
    assert g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer'])
    assert tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text']
    assert g['length']==len(g['token_ids']) and 1<=g['length']<=32
    assert g['eos']==(g['token_ids'][-1]==2) and g['empty']==(not bool(g['text']))
    assert g['truncated']==(g['length']==32 and not g['eos']) and g['repeated_trigram']==repeated_trigram(g['token_ids'])
    logits=raw(row);assert torch.isfinite(logits).all() and int(logits.argmax())==g['token_ids'][0] and digest(logits.tolist())==g['first_logits_sha256']
    if full_logits is not None:
        assert all(g[k]==v for k,v in first_metrics(full_logits,logits).items())
        assert g['full_first_logits_sha256']==digest(full_logits.tolist())
    layout=g['query_layout'];n=layout['prompt_length'];assert len(layout['I'])==576 and layout['A']==[n-1]
    assert layout['R']==list(range(layout['I'][-1]+1,n-1)) and g['own_cache'] and g['own_prefix']
    assert len(g['source_events'])==4*g['length'] and len(g['cache_lengths'])==g['length']
    blocked=source_edges(layout,0,row['condition'],'cpu');blocked_hash=digest(blocked.nonzero().tolist())
    for t in range(g['length']):
        assert g['cache_lengths'][t]==dict(t=t,before=n+t-1 if t else 0,after=n+t,absolute_query=n+t-1,consumed_token=g['token_ids'][t-1] if t else None)
        events=g['source_events'][4*t:4*t+4];assert [e['layer'] for e in events]==LAYERS
        for e in events:
            active=not t and row['condition'] in NEW
            assert e['applied']==active and e['blocked_connections_per_head']==(int(blocked.sum()) if active else 0)
            assert e['phase']==('decode' if t else 'prefill') and e['absolute_start']==(n+t-1 if t else 0)
            assert e['key_length']==n+t and e['query_length']==(1 if t else n)
            assert e['proposed_blocked_sha256']==(None if t else blocked_hash)
            assert e['row_sum_max_error']<=.002 and e['non_target_mask_unchanged'] and e['BOS_Self_legal'] and e['P_minus_one_mask_native'] and e['decode_mask_native']
            if active:assert e['blocked_probability_max']==0
            if not t:
                assert [x['query'] for x in e['source_masses']]==layout['R']
                for x in e['source_masses']:
                    assert abs(sum(x[k] for k in ['image','history','BOS','Self'])-1)<=.002
                    assert x['blocked_keys']==(int(blocked[x['query']].sum()) if active else 0)
                    if active:assert x['image' if row['condition']==NEW[0] else 'history']==0
    return g,logits

def replay():
    p=verify();binding=digest(p);assert read(OUT/'gpu_completion.json')['passed'] and read(OUT/'gpu_smoke_validation.json')['passed']
    assert read(OUT/'gpu_completion.json')['binding']==binding
    assert audit_history()==read(OUT/'historical_baseline_validation.json')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    samples=read(OUT/'sample_manifest.json')['samples'];mapping=read(OUT/'input_mappings.json');gens={c:[] for c in CONDITIONS};rows=[];hashes={};raw_hashes={}
    assert len(list((OUT/'checkpoints').glob('*/*.json')))==480 and len(list((OUT/'swap_checkpoints').glob('*/*.json')))==240
    for i,s in enumerate(samples):
        fullrow=read(checkpoint(s,'Full'));fl=raw(fullrow);fg=fullrow['generation']
        for c in CONDITIONS:
            if c in BASELINES:g=read(checkpoint(s,c))['generation']
            else:
                path=OUT/f"checkpoints/{s['task']}__{s['question_id']}/{c}.json";r=read(path)
                g,logits=validate_record(r,s,tokenizer,binding,mapping[i]['input_hashes'],fl)
                assert g['query_layout']==mapping[i]['role_layout']
                hashes[str(path.relative_to(OUT))]=sha256(path);raw_hashes[r['raw']['path']]=r['raw']['sha256']
            gens[c].append(g)
            rows.append(dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],condition=c,
                text=g['text'],prediction=g['prediction'],token_ids=json.dumps(g['token_ids']),correct=g['correct'],length=g['length'],
                correct_to_wrong=int(fg['correct'] and not g['correct']),wrong_to_correct=int(not fg['correct'] and g['correct']),
                first_top1_flip=g['first_top1_flip'],first_logp_loss=g['first_logp_loss'],sequence_changed=g['token_ids']!=fg['token_ids'],first_divergence=divergence(fg['token_ids'],g['token_ids']),
                eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],R_queries=len(g['query_layout']['R']),target_layers=json.dumps(LAYERS),
                input_ids_sha256=mapping[i]['input_hashes']['input_ids'],pixel_values_sha256=mapping[i]['input_hashes']['pixel_values'],
                source_checkpoint=str(checkpoint(s,c).relative_to(ROOT)) if c in BASELINES else str(path.relative_to(ROOT)),
                code_commit=read(OLD/'protocol.json')['code_commit'] if c in BASELINES else p['code_commit'],source='historical reused' if c in BASELINES else 'new main'))
    swapmap=read(OUT/'image_swap_inputs.json');swaprows=[];swapgens=[]
    for pair in read(OUT/'image_swap_manifest.json')['pairs']:
        records={c:[] for c in ['Full']+NEW};vectors={c:[] for c in records}
        for side,s in enumerate(pair['samples']):
            fl=None
            for c in records:
                path=OUT/f"swap_checkpoints/{pair['pair_id']}__{side}/{c}.json";r=read(path)
                g,lg=validate_record(r,s,tokenizer,binding,swapmap[f"{pair['pair_id']}__{side}"]['input_hashes'],fl)
                if c=='Full':fl=lg
                records[c].append(g);vectors[c].append(lg)
                hashes[str(path.relative_to(OUT))]=sha256(path);raw_hashes[r['raw']['path']]=r['raw']['sha256']
                swapgens.append(dict(pair_id=pair['pair_id'],side=side,question_id=s['question_id'],image_sha256=s['image_sha256'],gold=s['answer'],condition=c,text=g['text'],token_ids=json.dumps(g['token_ids']),correct=g['correct']))
        for c,gg in records.items():
            fg=records['Full']
            swaprows.append(dict(pair_id=pair['pair_id'],condition=c,question=pair['question'],left_question_id=pair['samples'][0]['question_id'],right_question_id=pair['samples'][1]['question_id'],
                left_gold=pair['samples'][0]['answer'],right_gold=pair['samples'][1]['answer'],left_text=gg[0]['text'],right_text=gg[1]['text'],left_correct=gg[0]['correct'],right_correct=gg[1]['correct'],
                both_correct=bool(gg[0]['correct'] and gg[1]['correct']),answers_changed=gg[0]['prediction']!=gg[1]['prediction'],tokens_changed=gg[0]['token_ids']!=gg[1]['token_ids'],
                first_distribution_js_nats=js(*vectors[c]),both_correct_change_from_Full=int(gg[0]['correct'] and gg[1]['correct'])-int(fg[0]['correct'] and fg[1]['correct']),
                answers_changed_change_from_Full=int(gg[0]['prediction']!=gg[1]['prediction'])-int(fg[0]['prediction']!=fg[1]['prediction']),
                js_change_from_Full=js(*vectors[c])-js(*vectors['Full'])))
    swap_summary=[]
    for c in ['Full']+NEW:
        rr=[r for r in swaprows if r['condition']==c]
        swap_summary.append(dict(condition=c,pairs=40,endpoints=80,correct_endpoints=sum(r['left_correct']+r['right_correct'] for r in rr),both_correct_pairs=sum(r['both_correct'] for r in rr),
            answers_changed_pairs=sum(r['answers_changed'] for r in rr),tokens_changed_pairs=sum(r['tokens_changed'] for r in rr),mean_first_JS_nats=float(np.mean([r['first_distribution_js_nats'] for r in rr])),
            both_correct_delta_Full=sum(r['both_correct_change_from_Full'] for r in rr),answers_changed_delta_Full=sum(r['answers_changed_change_from_Full'] for r in rr),mean_JS_delta_Full=float(np.mean([r['js_change_from_Full'] for r in rr]))))
    textrows=[]
    for s in read(OUT/'text_diagnostic_manifest.json')['samples']:
        rr=[];logits=[]
        for label in ['Native','Bypass']:
            path=OUT/f"text_checkpoints/{s['question_id']}/{label}.json";r=read(path);g=r['generation'];lg=raw(r)
            assert r['binding']==binding and r['sample_sha256']==digest(s) and r['result_sha256']==digest(g) and r['input_hashes']==s['input_hashes']
            assert tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text'] and int(lg.argmax())==g['token_ids'][0]
            assert not g['VQA_accuracy_valid'];rr.append(g);logits.append(lg)
            hashes[str(path.relative_to(OUT))]=sha256(path);raw_hashes[r['raw']['path']]=r['raw']['sha256']
        textrows.append(dict(question_id=s['question_id'],task=s['task'],native_text=rr[0]['text'],bypass_text=rr[1]['text'],native_token_ids=json.dumps(rr[0]['token_ids']),bypass_token_ids=json.dumps(rr[1]['token_ids']),sequence_changed=rr[0]['token_ids']!=rr[1]['token_ids'],first_JS_nats=js(*logits),VQA_accuracy_valid=False))
    assert len(textrows)==12 and len(list((OUT/'text_checkpoints').glob('*/*.json')))==24
    return p,samples,gens,rows,swaprows,swap_summary,swapgens,textrows,hashes,raw_hashes

def statistics(samples,gens):
    scores=np.array([[gens[c][i]['correct'] for c in CONDITIONS] for i in range(240)],float)
    draws=paired_bootstrap(samples,scores,repetitions=20000,seed=SEED);comparisons=[]
    for a,b in PAIRS:
        ia,ib=CONDITIONS.index(a),CONDITIONS.index(b);lo,hi=np.percentile(100*(draws[:,ia]-draws[:,ib]),[2.5,97.5])
        comparisons.append(dict(comparison=a+' - '+b,a=a,b=b,delta_pp=float(100*(scores[:,ia]-scores[:,ib]).mean()),ci95_low_pp=float(lo),ci95_high_pp=float(hi),
            b_correct_a_wrong=int(((scores[:,ib]==1)&(scores[:,ia]==0)).sum()),b_wrong_a_correct=int(((scores[:,ib]==0)&(scores[:,ia]==1)).sum()),seed=SEED,bootstrap_repetitions=20000,questions=240,image_clusters=240))
    summaries=[]
    for k,c in enumerate(CONDITIONS):
        lo,hi=np.percentile(100*draws[:,k],[2.5,97.5])
        summaries.append(dict(condition=c,**summary(gens[c],gens['Full']),ci95_accuracy_low_pct=float(lo),ci95_accuracy_high_pct=float(hi),repeated_trigram=sum(g['repeated_trigram'] for g in gens[c])))
    tasks=[]
    for t in TASKS:
        ii=[i for i,s in enumerate(samples) if s['task']==t]
        for c in CONDITIONS:tasks.append(dict(task=t,condition=c,**summary([gens[c][i] for i in ii],[gens['Full'][i] for i in ii])))
    return summaries,comparisons,tasks

def plots(summaries,pairs,tasks,swaps):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42});colors=['#486b89','#9a9caa','#ce7258','#75a5ad'];labels=['Full','R-All','R->Image OFF','R->Text OFF']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));ys=[r['accuracy_pct'] for r in summaries];upper=[r['ci95_accuracy_high_pct'] for r in summaries]
    axes[0].bar(labels,ys,color=colors)
    axes[0].errorbar(range(4),ys,yerr=[[r['accuracy_pct']-r['ci95_accuracy_low_pct'] for r in summaries],[r['ci95_accuracy_high_pct']-r['accuracy_pct'] for r in summaries]],fmt='none',color='black',capsize=4)
    for k,r in enumerate(summaries):axes[0].text(k,upper[k]+1.5,f"{r['correct']}/240",ha='center',fontsize=9)
    axes[0].set(ylim=(0,103),ylabel='Accuracy (%)');axes[0].tick_params(axis='x',rotation=15)
    vals=[r['delta_pp'] for r in pairs]
    axes[1].errorbar(vals,range(4),xerr=[[r['delta_pp']-r['ci95_low_pp'] for r in pairs],[r['ci95_high_pp']-r['delta_pp'] for r in pairs]],fmt='o',capsize=4)
    axes[1].axvline(0,color='gray',ls='--');axes[1].set(yticks=range(4),yticklabels=['R-All - Full','Image OFF - Full','Text OFF - Full','Image OFF - Text OFF'],xlabel='Percentage points (exploratory 95% CI)');axes[1].invert_yaxis()
    fig.tight_layout();savefig(fig,'figure_sources')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(6)
    for k,c in enumerate(CONDITIONS):axes[0].bar(x+(k-1.5)*.2,[r['accuracy_pct'] for r in tasks if r['condition']==c],.2,label=labels[k],color=colors[k])
    axes[0].set(xticks=x,xticklabels=TASKS,ylim=(0,105),ylabel='Accuracy (%)');axes[0].tick_params(axis='x',rotation=30);axes[0].legend(ncol=2,loc='lower left',bbox_to_anchor=(0,1.01),fontsize=8)
    for k,c in enumerate(NEW):
        rr=[r for r in tasks if r['condition']==c]
        axes[1].bar(x+(k-.5)*.35,[r['delta_full_pp'] for r in rr],.35,label=labels[k+2],color=colors[k+2])
    axes[1].axhline(0,color='gray',ls='--');axes[1].set(xticks=x,xticklabels=TASKS,ylabel='Accuracy change vs Full (pp)');axes[1].tick_params(axis='x',rotation=30);axes[1].legend(loc='lower left',bbox_to_anchor=(0,1.01),fontsize=8)
    fig.tight_layout();savefig(fig,'figure_tasks')
    fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(3);names=['Full','Image OFF','Text OFF']
    axes[0].bar(x-.18,[r['both_correct_pairs'] for r in swaps],.36,label='Both answers correct',color=colors[0]);axes[0].bar(x+.18,[r['answers_changed_pairs'] for r in swaps],.36,label='Answers differ across images',color=colors[2]);axes[0].set(xticks=x,xticklabels=names,ylim=(0,43),ylabel='Pairs / 40');axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.01),fontsize=8)
    axes[1].bar(names,[r['mean_first_JS_nats'] for r in swaps],color=[colors[0],colors[2],colors[3]]);axes[1].set(ylabel='Mean full-vocabulary first-token JS (nats)',title='Same question, different image and gold')
    fig.tight_layout();savefig(fig,'figure_image_swap')

def savefig(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)

def report(p,summaries,pairs,tasks,swaps,textrows,rows):
    f,r,im,txt=summaries;image_pair,text_pair=pairs[1:3]
    lines=['# 实验五：关键文本 Query 的 Attention 是否在直接读取视觉信息？','',
        '**COMPLETE：主实验新增 480 条来源 Mask 生成，复用 Full/R-All 各 240 条；换图控制新增 240 条；Question-only 诊断新增 24 条。三者分开统计。**','',
        '## 核心观察','',
        f"R-All 为 {r['correct']}/240（{r['accuracy_pct']:.3f}%），R→Image-OFF 为 {im['correct']}/240（{im['accuracy_pct']:.3f}%），R→Text-OFF 为 {txt['correct']}/240（{txt['accuracy_pct']:.3f}%），Full 为 {f['correct']}/240。目标严格限定为 0-based L2–L5 的原始 Prefill R Query；P−1 和全部 Decode 没有来源 Mask。",'',
        '|条件|正确/240|Accuracy|ΔFull pp|Full 正确→错误|Full 错误→正确|EOS|空/截断/重复|','|---|---:|---:|---:|---:|---:|---:|---|']
    for v in summaries:lines.append(f"|{v['condition']}|{v['correct']}|{v['accuracy_pct']:.3f}%|{v['delta_full_pp']:+.3f}|{v['correct_to_wrong']}|{v['wrong_to_correct']}|{v['eos']}|{v['empty']}/{v['truncated']}/{v['repeated_trigram']}|")
    lines+=['','## 配对统计','', '20,000 次 Image SHA256 整簇配对 Bootstrap；seed=20261102；所有比较使用同一抽样；普通探索性双侧 95% 百分位区间。没有等价或非劣效声明。','',
        '|比较|差值 pp|95% CI pp|后者正确→前者错误|后者错误→前者正确|','|---|---:|---|---:|---:|']
    for v in pairs:lines.append(f"|{v['comparison']}|{v['delta_pp']:+.3f}|[{v['ci95_low_pp']:+.3f}, {v['ci95_high_pp']:+.3f}]|{v['b_correct_a_wrong']}|{v['b_wrong_a_correct']}|")
    lines+=['','## 六个研究问题','',
        f"1. **为什么 R-All 损害巨大，来源 Mask 能解释多少？** R-All 相对 Full {pairs[0]['delta_pp']:+.3f} pp，Image-OFF {image_pair['delta_pp']:+.3f} pp，Text-OFF {text_pair['delta_pp']:+.3f} pp。两种来源 Mask 保留其他来源并重新归一化，整个 o_proj 输出置零则删除全部更新，二者强度和计算效果不等价。可以对比功能敏感性，不能把损害相减或相除当作视觉/文本信息贡献比例；还未分离 BOS、Self、跨来源补偿和一般残差更新作用。",
        f"2. **R 是否必须直接读取 Image Key？** Image-OFF 的净差值为 {image_pair['delta_pp']:+.3f} pp，CI [{image_pair['ci95_low_pp']:+.3f}, {image_pair['ci95_high_pp']:+.3f}]，真实损害 {im['correct_to_wrong']} 题、纠正 {im['wrong_to_correct']} 题。"+('区间完全为负，支持该直接访问在此固定设置下具有功能作用。' if image_pair['ci95_high_pp']<0 else '区间没有完全低于零，不能据此确认总体退化或宣称无损；按实际损益判断其敏感性。')+'其他层、此前 Hidden 和非图像 Key 仍可携带视觉信息，低损害不证明视觉信息不存在，也不证明该路径普遍不必要。',
        f"3. **对非图像历史 Key 的依赖多强？** Text-OFF 净差值为 {text_pair['delta_pp']:+.3f} pp，CI [{text_pair['ci95_low_pp']:+.3f}, {text_pair['ci95_high_pp']:+.3f}]，损害 {txt['correct_to_wrong']} 题、纠正 {txt['wrong_to_correct']} 题。保留 BOS、Self 和全部可见 Image Key，且只遮蔽 k<q 的非图像非 BOS 历史。其损害与上下文整合受影响相容，但非图像 Key 自身可能带有视觉信息，不能等同纯语言贡献。",
        '4. **换图控制是否改变图像条件性回答？** 见下表。两端都正确衡量语义正确性；答案随图改变和完整首 Token 分布 JS 衡量图像条件性，两者分别报告。答案改变不保证正确，不变也可能来自相同预测或模型错误；这些 40 对来自历史冻结、有不同官方 Gold 的子集，不是独立泛化证据。',
        '5. **能否称 R 是关键视觉信息整合位置？** 现有证据至多描述来源特异性功能作用和图像条件性变化。Question-only 只是诊断而非有效的纯文本 QA；不能据此证明视觉专属性。其他层的视觉读取、文本 KV 已携带的视觉信息、一般上下文/模板整合、BOS/Self 以及 Hard Mask 重新归一化仍未被排除。不能称 R 是唯一视觉写入节点。',
        '6. **对 Look Hard, Answer Cheap 与优化设计的意义？** 角色敏感性和来源敏感性是两条不同轴。R-All 的高损害不能直接当作早层 R 读图必要性的证据，也不能仅凭来源 Mask 的低损害删除整个 R Attention。优化候选应先保留有效的上下文更新，再在独立数据上验证源访问限制与生成质量。本轮只是功能控制，未跳过计算、删除视觉 KV 或测得真实加速。','',
        '## 图像交换控制','', '|条件|两端正确/40|答案随图改变/40|Token 序列改变/40|首Token平均JS nat|两端正确相对Full变化|答案变化相对Full变化|','|---|---:|---:|---:|---:|---:|---:|']
    for v in swaps:lines.append(f"|{v['condition']}|{v['both_correct_pairs']}|{v['answers_changed_pairs']}|{v['tokens_changed_pairs']}|{v['mean_first_JS_nats']:.6f}|{v['both_correct_delta_Full']:+d}|{v['answers_changed_delta_Full']:+d}|")
    lines+=['','逐对完整答案、两端评分、完整首 Token 分布的 FP64 JS 及相对 Full 差异保存于 image_swap_per_pair.csv。Full、Image-OFF、Text-OFF 在当前同形状路径各生成 80 条；历史 Handoff Full 未显式 logits_to_keep=1，因此没有直接混用其数值结果。','',
        '## Question-only 文本诊断','',f"仓库未定位可审计的现成纯文本 QA；固定取原 Manifest 每任务最前两题，共 12 题，仅移除图像并保持原文本内容。非终止文本 Query 范围重新定义为原始所有非 BOS、严格早于 P−1 的文本位置；只置零其 L2–L5 o_proj 输出，P−1 与 Decode 原生。Native 与 Bypass 有 {sum(x['sequence_changed'] for x in textrows)}/12 条完整输出不同。没有给这些无图问题计算 VQA Accuracy，也没有编造 Gold。该诊断不能与有图 GQA Accuracy 差作因果等价比较，更不能证明一般文本能力或视觉专属性。",'',
        '## 六任务与负面案例','', '|任务|Full|R-All|Image OFF|Text OFF|','|---|---:|---:|---:|---:|']
    for t in TASKS:lines.append('|'+t+'|'+'|'.join(f"{next(v['accuracy_pct'] for v in tasks if v['task']==t and v['condition']==c):.1f}%" for c in CONDITIONS)+'|')
    lines+=['','所有 Full 正确而任一干预错误的记录均保存于 negative_cases.csv；未按有利结果筛选。逐题主表保存答案、Token、Gold、损益和输入绑定；检查点保存每层每个 R Query 的真实来源概率质量、禁止概率最大值、Mask 边数与 Cache 位置。','',
        '## 正确性与执行','',f"基准 `{BASE}`；冻结执行源码 `{p['code_commit']}`。历史 Full/R-All 的 480 条记录及服务器原始首 Token 数组、输入和评分完整审计；没有复制未经核验的分数。Manifest 原样复制，实际完整 SHA256 为 `{MANIFEST_SHA}`，任务书给出的 59 位字符串沿用历史协议中的抄写差异说明。",'',
        f"CPU 测试 {read(OUT/'cpu_validation.json')['tests']} 项通过，三道固定 GPU Smoke 全通过。HF/Full/No-op 有效 Logits 用 torch.equal 严格一致；显式 logits_to_keep=1。历史 Full/R-All 的首向量与完整 Token 精确匹配。目标层实际 Softmax 后禁止概率严格为零，概率行和沿用既有 0.002 上界；BOS/Self 合法，非目标 Mask 原值精确保留，P−1/Decode 未 Mask。32 层 MLP、独立真实 Cache、绝对位置、输入与权重版本、正常/异常 Hook 清理通过。",'',
        f"单卡物理 GPU{p['gpu']['physical_index']}，UUID `{p['gpu']['uuid']}`，只暴露该设备、一份模型串行采集。主实验、40 对换图、12 题诊断分别冻结及计数。模型完整 SHA256 在冻结前和采集后核验；进程退出后记录 GPU 释放状态，未终止其他进程。原始首 Token 数组保留服务器资源目录，GitHub 只含路径和哈希，不包含大型数组。",'',
        '## 结果图','', '![来源干预与配对区间](figure_sources.png)','', '![六任务结果](figure_tasks.png)','', '![换图语义正确性与图像条件性](figure_image_swap.png)','',
        '## 解释边界与未来假说','',
        '已完成观察是四条件的完整生成行为、来源 Mask 的真实概率、配对准确率差和换图/诊断结果。机制解释限于被干预路径的功能敏感性；文本状态承载视觉信息、跨来源补偿或一般上下文整合的具体中介机制仍是待检验假说。没有临时调整层集合、样本、Prompt 或评分。未来如需分离更具体的机制，应预声明更强的文本任务与路径控制；本轮不设计或执行追加实验。','',
        '**到此停止：不跨模型、不扩样、不训练 Router、不实现 Kernel 或 KV 剪枝。**','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))

def main(check=False):
    p,samples,gens,rows,swaprows,swaps,swapgens,textrows,hashes,raw_hashes=replay();summaries,pairs,tasks=statistics(samples,gens)
    outputs=[('accuracy_summary.csv',summaries),('paired_comparisons.csv',pairs),('task_accuracy.csv',tasks),('per_question.csv',rows),('image_swap_summary.csv',swaps),('image_swap_per_pair.csv',swaprows),('image_swap_generations.csv',swapgens),('text_diagnostic.csv',textrows),('negative_cases.csv',[r for r in rows if r['correct_to_wrong']])]
    if check:
        for name,expected in outputs:
            with (OUT/name).open(newline='') as f:saved=list(csv.DictReader(f))
            assert len(saved)==len(expected)
            for a,b in zip(expected,saved):
                for k,v in a.items():
                    if isinstance(v,(int,float,np.number)) and not isinstance(v,(bool,np.bool_)):assert abs(float(v)-float(b[k]))<1e-12,(name,k)
                    else:assert b[k]==('' if v is None else str(v)),(name,k)
        c=read(OUT/'completion.json');assert c['complete']
        for name,h in c['artifacts'].items():assert sha256(OUT/name)==h,name
        assert not torch.cuda.is_initialized();print('READ-ONLY CHECK PASS: main, swaps, text, all CSV statistics and artifact hashes');return
    for name,data in outputs:
        if data:csv_write(OUT/name,data)
        else:
            with (OUT/name).open('w',newline='') as f:csv.DictWriter(f,fieldnames=list(rows[0])).writeheader()
    plots(summaries,pairs,tasks,swaps);report(p,summaries,pairs,tasks,swaps,textrows,rows)
    env=read(OUT/'environment.json');assert not Path(f"/proc/{env['pid']}").exists()
    state=hardware();apps=[x for x in state['compute_processes'].splitlines() if x.split(',')[0].strip()==p['gpu']['uuid']]
    write(OUT/'gpu_release.json',dict(our_process_exited=True,pid=env['pid'],gpu=p['gpu'],hardware=state,selected_gpu_currently_idle=not apps,no_external_process_terminated=True))
    for path,info in p['model_files'].items():assert sha256(path)==info['sha256']
    verify();assert not torch.cuda.is_initialized()
    write(OUT/'validation.json',dict(passed=True,new_main_generations=480,reused_main_generations=480,swap_generations=240,text_diagnostic_generations=24,
        first_metric_replay_difference=0.,checkpoint_hashes=hashes,raw_hashes=raw_hashes,model_full_sha256_unchanged=True,historical_artifacts_unchanged=True,
        actual_probability_checks=True,source_scope_checks=True,token_text_score_exact=True,cuda_initialized=False,raw_arrays_on_github=False,gpu_process_exited=True))
    artifacts={str(f.relative_to(OUT)):sha256(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['worker.lock','completion.json']}
    write(OUT/'completion.json',dict(status='COMPLETE',complete=True,new_main_generations=480,reused_main_generations=480,swap_generations=240,text_diagnostic_generations=24,frozen_code_commit=p['code_commit'],binding=digest(p),artifacts=artifacts))
    print(json.dumps(dict(summaries=summaries,comparisons=pairs,swap=swaps,text_changed=sum(r['sequence_changed'] for r in textrows)),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    try:main(args.check)
    except BaseException:
        if not args.check:
            write(OUT/'failures'/f'analysis_{time.time_ns()}.json',dict(status='PARTIAL',traceback=traceback.format_exc()));write(OUT/'completion.json',dict(status='PARTIAL',complete=False))
        raise
