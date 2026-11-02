"""Audit records and quantify the point frontier without non-inferiority claims."""
import argparse,csv,io,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from query_role_attention.analyze import paired_bootstrap
from query_role_formal.common import repeated_trigram,divergence
from first_answer_attention.engine import first_metrics
from paper_rq1.scorers import gqa_prediction
from .common import *
from visual_write_window.analyze import raw,validate as historical_validate

def quality_flags(correct,reference,n=240):
    delta=100*(correct-reference)/n
    return dict(point_minus1_candidate=delta>=-1.,point_minus2_candidate=delta>=-2.)

def frontier_flags(rows):
    return [not any(b['retained_layers']<=a['retained_layers'] and b['correct']>=a['correct'] and
                   (b['retained_layers']<a['retained_layers'] or b['correct']>a['correct']) for b in rows) for a in rows]

def validate_new(row,s,tok,binding,input_hashes):
    g=row['generation'];c=row['condition'];assert c in NEW
    assert row['binding']==binding and row['sample_sha256']==digest(s) and row['result_sha256']==digest(g)
    assert row['input_hashes']==input_hashes and row['gold']==s['answer'] and row['image_sha256']==s['image_sha256']
    assert g['writer_layers']==KEEP[c] and g['own_cache'] and g['own_prefix'] and g['answer_image_blocked']
    assert tok.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text']
    assert g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer'])
    assert g['length']==len(g['token_ids']) and 1<=g['length']<=32 and g['eos']==(g['token_ids'][-1]==2)
    assert g['truncated']==(g['length']==32 and not g['eos']) and g['empty']==(not bool(g['text'])) and g['repeated_trigram']==repeated_trigram(g['token_ids'])
    v=raw(row);assert torch.isfinite(v).all() and int(v.argmax())==g['token_ids'][0] and digest(v.tolist())==g['first_logits_sha256']
    layout=g['query_layout'];n=layout['prompt_length'];assert len(layout['I'])==576 and layout['R']==list(range(layout['I'][-1]+1,n-1)) and layout['A']==[n-1]
    assert len(g['source_events'])==32*g['length'] and len(g['cache_lengths'])==g['length']
    for t in range(g['length']):
        before=n+t-1 if t else 0
        assert g['cache_lengths'][t]==dict(t=t,before=before,after=n+t,absolute_query=n+t-1,consumed_token=g['token_ids'][t-1] if t else None)
        for k,e in enumerate(g['source_events'][32*t:32*t+32]):
            rblocked=not t and k not in KEEP[c]
            assert e['layer']==k and e['phase']==('decode' if t else 'prefill') and e['absolute_start']==before
            assert e['query_length']==(1 if t else n) and e['key_length']==n+t
            assert e['R_image_open']==(k in KEEP[c]) and e['answer_image_masked'] and e['R_image_masked']==rblocked
            assert e['blocked_connections_per_head']==576*(1+(len(layout['R']) if rblocked else 0))
            assert e['blocked_probability_max']==0 and e['answer_image_mass_mean']==0 and e['row_sum_max_error']<=.002
            assert e['non_target_mask_unchanged'] and e['BOS_Self_legal']
            if rblocked:assert e['relay_image_mass_mean']==0
    return g,v

def replay():
    p=verify();binding=digest(p);assert read(OUT/'gpu_completion.json')['passed'] and read(OUT/'gpu_completion.json')['binding']==binding
    assert read(OUT/'gpu_smoke_validation.json')['passed']
    assert sha256(OLD/'protocol.json')==p['historical_protocol_sha256'] and sha256(OLD/'completion.json')==p['historical_completion_sha256']
    history=read(OUT/'historical_reuse_audit.json');oldp=read(OLD/'protocol.json');samples=read(OUT/'sample_manifest.json')['samples'];mapping=read(OUT/'input_mappings.json')
    assert len(samples)==240 and len({s['image_sha256'] for s in samples})==240 and all(sum(s['task']==t for s in samples)==40 for t in TASKS)
    assert len(list((OUT/'checkpoints').glob('*/*.json')))==1440
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);gens={c:[] for c in CONDITIONS};rows=[];newhash={};reusedhash={};rawhash={};nativehash={};native_rows=[]
    for i,s in enumerate(samples):
        assert sha256(s['image_path'])==s['image_sha256'];records={};vectors={}
        for c in CONDITIONS:
            if c in BASELINES:
                f=checkpoint(s,c);r=read(f);key=str(f.relative_to(OLD));assert sha256(f)==history['checkpoint_hashes'][key]
                g,v=historical_validate(r,s,tok,digest(oldp),mapping[i]['input_hashes']);reusedhash[key]=sha256(f)
                assert sha256(r['raw']['path'])==history['raw_hashes'][r['raw']['path']]
            else:
                f=OUT/f"checkpoints/{s['task']}__{s['question_id']}/{c}.json";r=read(f);g,v=validate_new(r,s,tok,binding,mapping[i]['input_hashes']);newhash[str(f.relative_to(OUT))]=sha256(f)
            assert g['query_layout']==mapping[i]['role_layout'];records[c]=g;vectors[c]=v;gens[c].append(g);rawhash[r['raw']['path']]=r['raw']['sha256']
        ref=records['All-Writer']
        for c,g in records.items():
            fm=first_metrics(vectors['All-Writer'],vectors[c])
            if c!='All-Writer':assert all(g[k]==value for k,value in fm.items())
            path=checkpoint(s,c) if c in BASELINES else OUT/f"checkpoints/{s['task']}__{s['question_id']}/{c}.json"
            rows.append(dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],condition=c,
                retained_layers=len(KEEP[c]),closed_R_image_layers=32-len(KEEP[c]),writer_layers=json.dumps(KEEP[c]),text=g['text'],prediction=g['prediction'],token_ids=json.dumps(g['token_ids']),correct=g['correct'],
                length=g['length'],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],
                correct_to_wrong=int(ref['correct'] and not g['correct']),wrong_to_correct=int(not ref['correct'] and g['correct']),
                first_top1_flip=fm['first_top1_flip'],first_logp_loss=fm['first_logp_loss'],sequence_changed=g['token_ids']!=ref['token_ids'],first_divergence=divergence(ref['token_ids'],g['token_ids']),
                All_Writer_text=ref['text'],R_queries=len(g['query_layout']['R']),input_ids_sha256=mapping[i]['input_hashes']['input_ids'],pixel_values_sha256=mapping[i]['input_hashes']['pixel_values'],
                source='historical reused' if c in BASELINES else 'new main',code_commit=oldp['code_commit'] if c in BASELINES else p['code_commit'],source_checkpoint=str(path.relative_to(ROOT))))
        f=NATIVE/f"checkpoints/{s['task']}__{s['question_id']}/Full.json";r=read(f);g=r['generation'];key=str(f.relative_to(NATIVE));nh=history['native_full']['audit']
        assert sha256(f)==nh['checkpoint_hashes'][key] and r['input_hashes']==mapping[i]['input_hashes'] and r['sample_sha256']==digest(s) and r['result_sha256']==digest(g)
        assert sha256(r['raw']['path'])==nh['raw_hashes'][r['raw']['path']];v=raw(r)
        assert digest(v.tolist())==g['first_logits_sha256'] and tok.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text'] and g['correct']==int(gqa_prediction(g['text'])==s['answer'])
        nativehash[key]=sha256(f)
        native_rows.append(dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],gold=s['answer'],text=g['text'],prediction=g['prediction'],token_ids=json.dumps(g['token_ids']),correct=g['correct'],source='Separate historical native Full; answer direct image access ON',source_checkpoint=str(f.relative_to(ROOT))))
    assert sum(r['correct'] for r in native_rows)==206
    return p,samples,gens,rows,native_rows,newhash,reusedhash,rawhash,nativehash

def statistics(samples,gens,rows):
    scores=np.array([[gens[c][i]['correct'] for c in CONDITIONS] for i in range(240)],float)
    draws=paired_bootstrap(samples,scores,repetitions=20000,seed=20261102);pairs=[]
    definitions=read(OUT/'protocol.json')['comparisons']
    for definition in definitions:
        a,b,role=definition['a'],definition['b'],definition['role'];ia,ib=CONDITIONS.index(a),CONDITIONS.index(b)
        low,high=np.percentile(100*(draws[:,ia]-draws[:,ib]),[2.5,97.5])
        pairs.append(dict(comparison=a+' - '+b,a=a,b=b,role=role,retained_a=len(KEEP[a]),retained_b=len(KEEP[b]),delta_pp=float(100*(scores[:,ia]-scores[:,ib]).mean()),
            ci95_low_pp=float(low),ci95_high_pp=float(high),b_correct_a_wrong=int(((scores[:,ib]==1)&(scores[:,ia]==0)).sum()),b_wrong_a_correct=int(((scores[:,ib]==0)&(scores[:,ia]==1)).sum()),
            questions=240,image_clusters=240,bootstrap_repetitions=20000,seed=20261102))
    summaries=[];tasks=[]
    for j,c in enumerate(CONDITIONS):
        rr=[r for r in rows if r['condition']==c];low,high=np.percentile(100*draws[:,j],[2.5,97.5]);comparison=next((r for r in pairs if r['a']==c and r['b']=='All-Writer'),None)
        row=dict(condition=c,source='historical reused' if c in BASELINES else 'new main',retained_layers=len(KEEP[c]),closed_R_image_layers=32-len(KEEP[c]),questions=240,
            correct=int(scores[:,j].sum()),accuracy_pct=float(scores[:,j].mean()*100),ci95_accuracy_low_pct=float(low),ci95_accuracy_high_pct=float(high),
            delta_all_pp=float((scores[:,j]-scores[:,0]).mean()*100),delta_ci95_low_pp=comparison['ci95_low_pp'] if comparison else 0.,delta_ci95_high_pp=comparison['ci95_high_pp'] if comparison else 0.)
        row.update(quality_flags(row['correct'],int(scores[:,0].sum())))
        for key in ['correct_to_wrong','wrong_to_correct','first_top1_flip','sequence_changed','eos','empty','truncated','repeated_trigram']:row[key]=sum(r[key] for r in rr)
        row['mean_first_logp_loss']=float(np.mean([r['first_logp_loss'] for r in rr]));row['mean_generated_tokens']=float(np.mean([r['length'] for r in rr]));summaries.append(row)
        for t in TASKS:
            tr=[r for r in rr if r['task']==t];ii=[i for i,s in enumerate(samples) if s['task']==t]
            tasks.append(dict(task=t,condition=c,retained_layers=len(KEEP[c]),questions=40,correct=sum(r['correct'] for r in tr),accuracy_pct=100*sum(r['correct'] for r in tr)/40,
                delta_all_pp=float((scores[ii,j]-scores[ii,0]).mean()*100),correct_to_wrong=sum(r['correct_to_wrong'] for r in tr),wrong_to_correct=sum(r['wrong_to_correct'] for r in tr)))
    for row,flag in zip(summaries,frontier_flags(summaries)):row['point_pareto_frontier']=flag
    return summaries,pairs,tasks

def figures(summaries,pairs,tasks):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.2)
        plt.close(fig)
    colors=dict(zip(NEW,['#2878a8','#df764e','#7b629d','#459c83','#bb607d','#b49639']))
    fig,axes=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1.15,1]},layout='constrained')
    allacc=summaries[0]['accuracy_pct'];front=sorted([r for r in summaries if r['point_pareto_frontier']],key=lambda r:r['retained_layers'])
    axes[0].plot([r['retained_layers'] for r in front],[r['accuracy_pct'] for r in front],color='#526f55',alpha=.5,ls='--',label='Observed point Pareto frontier')
    markers=['o','s','^','D','v','P']
    for k,r in enumerate(summaries):
        c=r['condition'];color=colors.get(c,'#555555');marker=markers[NEW.index(c)] if c in NEW else 'x'
        axes[0].errorbar(r['retained_layers'],r['accuracy_pct'],yerr=[[r['accuracy_pct']-r['ci95_accuracy_low_pct']],[r['ci95_accuracy_high_pct']-r['accuracy_pct']]],fmt=marker,color=color,capsize=3,label=c,markersize=7)
    for bound,style in [(-1.,'--'),(-2.,':')]:axes[0].axhline(allacc+bound,color='#999999',ls=style,lw=1)
    axes[0].set(xticks=[0,4,8,12,16,28,32],xlim=(-1,33),ylim=(50,101),xlabel='Retained R -> Image writer layers',ylabel='Full-answer accuracy (%)',title='Fixed 240-question cohort; exploratory 95% CI')
    axes[0].legend(loc='lower left',bbox_to_anchor=(-.04,-.45),ncol=2,fontsize=8,frameon=False)
    rr=[r for r in pairs if r['role']=='new_vs_all'];values=[r['delta_pp'] for r in rr]
    axes[1].axvspan(-2,-1,color='#eeeeee');axes[1].axvspan(-1,0,color='#e4eee4');axes[1].axvline(0,color='gray',lw=1)
    axes[1].axvline(-1,color='#678267',ls='--',label='-1 pp point screen');axes[1].axvline(-2,color='#888888',ls=':',label='-2 pp point screen')
    axes[1].errorbar(values,range(6),xerr=[[r['delta_pp']-r['ci95_low_pp'] for r in rr],[r['ci95_high_pp']-r['delta_pp'] for r in rr]],fmt='none',ecolor='#555555',capsize=4)
    for i,r in enumerate(rr):axes[1].scatter(r['delta_pp'],i,color=colors[r['a']],s=45)
    axes[1].set(yticks=range(6),yticklabels=[r['a'] for r in rr],xlabel='Accuracy difference vs All-Writer (pp)',title='Point screens do not establish non-inferiority')
    axes[1].invert_yaxis();axes[1].legend(loc='lower left',bbox_to_anchor=(0,-.36),fontsize=8,frameon=False)
    save(fig,'figure_frontier')
    rr=[r for r in pairs if r['role']=='same_layer_count'];fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    ax.errorbar([r['delta_pp'] for r in rr],range(4),xerr=[[r['delta_pp']-r['ci95_low_pp'] for r in rr],[r['ci95_high_pp']-r['delta_pp'] for r in rr]],fmt='o',capsize=4,color='#486b89')
    ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(4),yticklabels=[r['comparison']+f" ({r['retained_a']} layers)" for r in rr],xlabel='Accuracy difference (pp)',title='Same writer-layer budget; exploratory paired 95% CI');ax.invert_yaxis();save(fig,'figure_same_count')
    values=np.array([[next(r['delta_all_pp'] for r in tasks if r['condition']==c and r['task']==t) for t in TASKS] for c in NEW]);lim=max(1.,abs(values).max())
    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained');im=ax.imshow(values,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{values[i,j]:+.1f}',ha='center',va='center',color='white' if abs(values[i,j])>.7*lim else 'black')
    ax.set(xticks=range(6),xticklabels=TASKS,yticks=range(6),yticklabels=NEW,title='Change vs All-Writer by task (40 questions/task; descriptive)');ax.tick_params(axis='x',rotation=25)
    fig.colorbar(im,ax=ax,label='Accuracy difference (pp)',shrink=.85,pad=.025);save(fig,'figure_tasks')

def report(p,summaries,pairs,tasks,rows,native):
    allr=summaries[0];novel=[r for r in summaries if r['condition'] in NEW];best=max(novel,key=lambda r:r['correct'])
    lines=['# Sparse Visual Writing Frontier：稀疏视觉写入前沿','',
        '**COMPLETE：新增6条件×240题＝1440条完整答案；严格复用四组历史基线960条；原生Full另列240条，不混入前沿。**','',
        '## 核心回答','',
        f"All-Writer（32层）为{allr['correct']}/240（{allr['accuracy_pct']:.3f}%）。新增条件最高点估计为{best['condition']}（{best['retained_layers']}层）：{best['correct']}/240（{best['accuracy_pct']:.3f}%），相对All {best['delta_all_pp']:+.3f} pp，探索性95%CI [{best['delta_ci95_low_pp']:+.3f}, {best['delta_ci95_high_pp']:+.3f}]。",'']
    for bound,flag in [(-1,'point_minus1_candidate'),(-2,'point_minus2_candidate')]:
        eligible=[r for r in novel if r[flag]]
        lines.append(f"预设{bound}pp点估计界限的新增候选："+('；'.join(f"{r['condition']}（{r['retained_layers']}层，{r['accuracy_pct']:.3f}%）" for r in eligible) if eligible else '无')+'。')
        if eligible:
            minimum=min(r['retained_layers'] for r in eligible);lines.append(f"在已测试新增组合中，达到该点估计要求的最少保留层数为{minimum}，即最多关闭{32-minimum}层的R→Image直接访问；这不是已证明的安全关闭上限。")
    lines+=['','质量候选只依赖预先声明的点估计规则。CI跨0不证明非劣效；本轮未进行确认性非劣效、等价或多重校正检验。所有条件来自已复用历史探索样本，候选必须留待独立数据验证。','',
        '## 协议与复用','',
        '所有前沿条件均在32层禁止P−1及全部真实Decode（含EOS预测）直接访问原始576个Image Key。仅Prefill R可读图的层集合不同；R为图像后、P−1前的全部非视觉Query，包括问题、指令、模板。Image Query、其他连接、MLP、Residual、RoPE和KV保持原生。每个条件从空Cache独立生成，无o_proj消融。',
        f"模型revision `{REVISION}`；FP16/Eager，Transformers4.49.0，Greedy，max_new_tokens=32，显式logits_to_keep=1。冻结源码 `{p['code_commit']}`。原样复用Manifest SHA256 `{MANIFEST_SHA}` 及原Prompt IDs、Pixel Tensor哈希、Gold和Image SHA。",
        '六个新条件逐层调用历史visual_write_window.engine.apply_mask：保留层选择All-Writer规则，其他层选择No-Writer规则。没有修改历史函数、检查点或结果。审计960条基线的Token解码、评分、真实概率、Cache、原始向量和输入绑定；GPU Smoke进一步检验旧/新入口全部有效Logits及历史保存首向量和完整Token严格一致。',
        '复用All-Writer 202、Drop-W3 205、Only-W2 188、No-Writer 153。原生Full另经审计为206/240（85.833%），其答案阶段可以直接读图，与All-Writer协议不同，单独保存native_full_reference.csv；不把它当作32层All-Writer，也不混合评分。','',
        '## 总体结果与质量筛选','',
        '|条件|保留/关闭层|正确/240|Accuracy|ΔAll pp [95%CI]|原对→错/错→对|−1pp候选|−2pp候选|点Pareto前沿|',
        '|---|---|---:|---:|---|---|---|---|---|']
    for r in summaries:lines.append(f"|{r['condition']}|{r['retained_layers']}/{r['closed_R_image_layers']}|{r['correct']}|{r['accuracy_pct']:.3f}%|{r['delta_all_pp']:+.3f} [{r['delta_ci95_low_pp']:+.3f}, {r['delta_ci95_high_pp']:+.3f}]|{r['correct_to_wrong']}/{r['wrong_to_correct']}|{'是' if r['point_minus1_candidate'] else '否'}|{'是' if r['point_minus2_candidate'] else '否'}|{'是' if r['point_pareto_frontier'] else '否'}|")
    front=[r for r in summaries if r['point_pareto_frontier']]
    lines+=['','点Pareto前沿（只在这十个已测试条件内）：'+ '；'.join(f"{r['condition']}（{r['retained_layers']}层，{r['accuracy_pct']:.3f}%）" for r in sorted(front,key=lambda r:r['retained_layers']))+'。相同预算与准确率的并列组合全部保留。前沿基于点估计，受抽样不确定性和历史复用影响；少数题的净优势不证明总体支配。','',
        '## 8、12、16层与邻近窗口协作','',
        '|比较（前者−后者）|保留层数|ΔAccuracy pp [95%CI]|后者对→前者错/后者错→前者对|',
        '|---|---:|---|---|']
    for r in pairs:
        if r['role']=='same_layer_count':lines.append(f"|{r['comparison']}|{r['retained_a']}|{r['delta_pp']:+.3f} [{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}]|{r['b_correct_a_wrong']}/{r['b_wrong_a_correct']}|")
    lines+=['','12层的直接比较以共同L4–L11为基础，将L0–L3与L12–L15两种额外窗口相比较；8层的三组都保留L8–L11，比较搭配W0、W1或W3的效果。这些背景固定的比较检验组合的功能差异，并不等于对单个窗口信息贡献的可加分解，也未额外估计未经预声明的交互量。','',
        '全部总体与组合比较使用同一组20,000次Image SHA256整簇配对Bootstrap，seed=20261102，双侧百分位探索性95%CI。240题对应240个图像簇。完整比较见paired_comparisons.csv；首Token变化和signed LogP loss不等于完整答案Accuracy。','',
        '## 六任务差异','', '|条件|'+'|'.join(TASKS)+'|','|---|'+'---:|'*6]
    for c in CONDITIONS:lines.append('|'+c+'|'+'|'.join(f"{next(r['accuracy_pct'] for r in tasks if r['task']==t and r['condition']==c):.1f}%" for t in TASKS)+'|')
    lines+=['','每任务40题，任务分析为描述性。完整逐题损益、全部错误答案和相对All损害分别保存在per_question.csv、incorrect_answers.csv、negative_cases.csv。未按输出筛选样本或删除失败案例。','',
        '## 全部负面案例与生成质量','',
        '|新增条件|原对→错|原错→对|首Token翻转|EOS/截断/空/重复|','|---|---:|---:|---:|---|']
    for r in novel:lines.append(f"|{r['condition']}|{r['correct_to_wrong']}|{r['wrong_to_correct']}|{r['first_top1_flip']}|{r['eos']}/{r['truncated']}/{r['empty']}/{r['repeated_trigram']}|")
    lines+=['','按每个新增条件、题号排序各展示最前两例损害；此处只作说明，不改变样本、评分或窗口选择。','',
        '|条件|题号|问题|Gold|All-Writer|条件答案|','|---|---|---|---|---|---|']
    for c in NEW:
        rr=sorted([r for r in rows if r['condition']==c and r['correct_to_wrong']],key=lambda r:r['question_id'])[:2]
        for r in rr:lines.append('|'+ '|'.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ['condition','question_id','question','gold','All_Writer_text','text'])+'|')
    lines+=['','## 验证、资源与限制','',
        'CPU测试、三道固定真实7B GPU Smoke、六个新条件和四个历史基线兼容性均通过。HF/Native/六种No-op使用匹配LM Head形状，全部有效Logits保持torch.equal；禁止连接实际Softmax概率严格为0，概率行和沿用0.002界限，非目标Mask原值不变，BOS/Self合法。32层MLP、独立Cache存储、父Cache指纹、每步绝对位置、输入/权重版本以及正常/异常Hook清理均经过检查。',
        f"仅物理GPU{p['gpu']['physical_index']}，UUID `{p['gpu']['uuid']}`；启动前重新确认空闲，只暴露该设备，一份模型串行。没有抢占或终止其他进程。GPU进程退出后的资源记录见gpu_release.json。原始大数组保留服务器资源目录，检查点记录路径和SHA256；GitHub不包含这些数组。",
        '采集后完整核对模型文件SHA256、检查点和原始向量；CSV按标准序列化逐单元格精确复核，None按标准CSV空字段处理。冻结执行源码与所有历史文件保持不变。',
        'Hard Mask会重归一化其余连接，层数不是单位Query贡献或实际FLOPs。图像KV、全部层和Attention计算仍执行，本轮只探索功能性层稀疏，未减少实际计算，也没有速度或显存收益证据。稀疏点估计前沿不证明普遍安全性、唯一视觉写入通路或跨模型泛化。','',
        '## 图表','', '![Accuracy与保留视觉写入层数](figure_frontier.png)','', '![相同层数的组合比较](figure_same_count.png)','', '![六任务差异](figure_tasks.png)','',
        '## 停止边界','',
        '高质量候选只留待独立数据验证。本轮不追加逐层贪心搜索、Token联合消融、Router训练、Kernel实现或其他条件。']
    path=OUT/'report_zh.md';assert not path.exists();path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def exact_csv(name,values):
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values);buf.seek(0)
    expected=list(csv.DictReader(buf))
    with (OUT/name).open(newline='',encoding='utf-8') as f:actual=list(csv.DictReader(f))
    assert actual==expected,name

def main(check=False):
    p,samples,gens,rows,native,newhash,reusedhash,rawhash,nativehash=replay();sums,pairs,tasks=statistics(samples,gens,rows)
    tables={'accuracy_summary.csv':sums,'paired_comparisons.csv':pairs,'same_layer_comparisons.csv':[r for r in pairs if r['role']=='same_layer_count'],
        'task_accuracy.csv':tasks,'per_question.csv':rows,'negative_cases.csv':[r for r in rows if r['correct_to_wrong']],
        'incorrect_answers.csv':[r for r in rows if not r['correct']],'native_full_reference.csv':native}
    if check:
        for name,values in tables.items():exact_csv(name,values)
        v=read(OUT/'validation.json');assert v['checkpoint_hashes']==newhash and v['reused_checkpoint_hashes']==reusedhash and v['raw_hashes']==rawhash and v['native_checkpoint_hashes']==nativehash
        c=read(OUT/'completion.json');assert c['status']=='COMPLETE'
        for name,h in c['artifacts'].items():assert sha256(OUT/name)==h,name
        print('EXACT READ-ONLY PASS: every CSV cell, new and reused records, actual masks/cache/raw vectors and artifact hashes',flush=True);return
    assert not (OUT/'completion.json').exists(),'No duplicate reporting or overwriting'
    for name,values in tables.items():csv_write(OUT/name,values)
    for name,info in p['model_files'].items():assert sha256(name)==info['sha256'],name
    state=hardware();assert not any(s.split(',')[0].strip()==p['gpu']['uuid'] for s in state['compute_processes'].splitlines())
    write(OUT/'gpu_release.json',dict(passed=True,selected=p['gpu'],snapshot=state,gpu_worker_exited=True))
    figures(sums,pairs,tasks);report(p,sums,pairs,tasks,rows,native)
    for name,values in tables.items():exact_csv(name,values)
    write(OUT/'validation.json',dict(passed=True,binding=digest(p),new_main_generations=1440,reused_main_records=960,separate_native_full_records=240,
        checkpoint_hashes=newhash,reused_checkpoint_hashes=reusedhash,raw_hashes=rawhash,native_checkpoint_hashes=nativehash,
        model_full_sha256_rechecked=True,exact_csv_cells_verified=True,statistics_replayed=True,actual_masks_cache_replayed=True,no_tolerance_relaxation=True))
    artifacts={str(f.relative_to(OUT)):sha256(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['worker.lock','completion.json']}
    write(OUT/'completion.json',dict(status='COMPLETE',complete=True,new_main_generations=1440,reused_main_records=960,separate_native_full_records=240,binding=digest(p),code_commit=p['code_commit'],artifacts=artifacts))
    print('COMPLETE report: exact CSV round trip, protocol and all result records audited',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');a=parser.parse_args()
    try:main(a.check)
    except BaseException:
        if not a.check and not (OUT/'completion.json').exists():
            write(OUT/'failures'/f'analysis_{time.time_ns()}.json',dict(status='PARTIAL',traceback=traceback.format_exc()))
            write(OUT/'completion.json',dict(status='PARTIAL',complete=False,failed_stage='analysis'))
        raise
