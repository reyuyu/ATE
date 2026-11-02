"""Replay every record; paired image-cluster contrasts and predeclared interaction."""
import argparse,csv,io,json,time,traceback
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

def quality_flags(correct,reference,n=240):
    delta=100*(correct-reference)/n
    return dict(point_minus1_candidate=delta>=-1.,point_minus2_candidate=delta>=-2.)
def interaction(j,l,s,a):return j-l-s+a
def unique_harm(j,l,s,a):return bool(a and l and s and not j)

def validate_new(row,s,tok,binding,mapping,spec):
    g=row['generation'];c=row['condition'];layout=mapping['role_layout'];positions=spec['selection']['writers']['Semantic-k'];n=layout['prompt_length']
    assert c in NEW and row['binding']==binding and row['sample_sha256']==digest(s) and row['result_sha256']==digest(g)
    assert row['input_hashes']==mapping['input_hashes'] and row['gold']==s['answer'] and row['image_sha256']==s['image_sha256']
    assert row['selection_binding']==spec['selection_binding']==g['selection_binding']
    assert g['writer_positions']==positions and g['writer_layers']==KEEP[c] and g['query_layout']==layout
    assert g['own_cache'] and g['own_prefix'] and g['answer_image_blocked'] and g['logits_to_keep']==1 and not g['noop'] and g['all_logits_finite']
    assert tok.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text'] and g['prediction']==gqa_prediction(g['text'])
    assert g['correct']==int(g['prediction']==s['answer']) and g['length']==len(g['token_ids']) and 1<=g['length']<=32
    assert g['eos']==(g['token_ids'][-1]==2) and g['truncated']==(g['length']==32 and not g['eos'])
    assert g['empty']==(not bool(g['text'])) and g['repeated_trigram']==repeated_trigram(g['token_ids'])
    v=raw(row);assert torch.isfinite(v).all() and int(v.argmax())==g['token_ids'][0] and digest(v.tolist())==g['first_logits_sha256']
    assert len(layout['I'])==576 and layout['R']==list(range(layout['I'][-1]+1,n-1)) and layout['A']==[n-1]
    assert len(g['source_events'])==32*g['length'] and len(g['cache_lengths'])==g['length']
    allowed=0
    for t in range(g['length']):
        before=n+t-1 if t else 0
        assert g['cache_lengths'][t]==dict(t=t,before=before,after=n+t,absolute_query=n+t-1,consumed_token=g['token_ids'][t-1] if t else None)
        for k,e in enumerate(g['source_events'][32*t:32*t+32]):
            permits=len(positions) if k in KEEP[c] else 0
            assert e['layer']==k and e['phase']==('decode' if t else 'prefill') and e['absolute_start']==before
            assert e['query_length']==(1 if t else n) and e['key_length']==n+t and e['permitted_writer_queries']==permits
            assert e['blocked_connections_per_head']==576*(1+(len(layout['R'])-permits if not t else 0))
            assert e['answer_image_masked'] and e['blocked_probability_max']==0 and e['answer_image_mass_mean']==0 and e['row_sum_max_error']<=.002
            assert e['non_target_mask_unchanged'] and e['BOS_Self_legal']
            if not t:allowed+=576*permits
    assert allowed==576*len(positions)*len(KEEP[c])
    return g,v

def replay():
    p=verify();binding=digest(p);audit=read(OUT/'historical_reuse_audit.json')
    assert read(OUT/'gpu_completion.json')['passed'] and read(OUT/'gpu_completion.json')['binding']==binding and read(OUT/'gpu_smoke_validation.json')['passed']
    samples=read(OUT/'sample_manifest.json')['samples'];mapping=read(OUT/'input_mappings.json');specs=read(OUT/'token_group_manifest.json')['main']
    assert len(samples)==240 and len({s['image_sha256'] for s in samples})==240 and all(sum(s['task']==t for s in samples)==40 for t in TASKS)
    assert len(list((OUT/'checkpoints').glob('*/*.json')))==720 and len(list((OUT/'reused_checkpoints').glob('*/*.json')))==1200
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);gens={c:[] for c in CONDITIONS};rows=[];newhash={};reusedhash={};rawhash={};jointrows=[]
    for i,s in enumerate(samples):
        assert sha256(s['image_path'])==s['image_sha256'];spec=specs[key(s)];records={};vectors={}
        for c in CONDITIONS:
            path=reused_path(s,c) if c in BASELINES else OUT/'checkpoints'/key(s)/(c+'.json');r=read(path)
            if c in BASELINES:
                assert sha256(path)==audit['records'][key(s)+'/'+c+'.json']['sha256']
                g,v=validate_reused(r,s,c,tok,mapping[i],spec,audit['protocols']);reusedhash[str(path.relative_to(OUT))]=sha256(path)
            else:
                g,v=validate_new(r,s,tok,binding,mapping[i],spec);newhash[str(path.relative_to(OUT))]=sha256(path)
            records[c]=g;vectors[c]=v;gens[c].append(g);rawhash[r['raw']['path']]=r['raw']['sha256']
        a=records['All-Writer'];R=len(mapping[i]['role_layout']['R'])
        for c,g in records.items():
            fm=first_metrics(vectors['All-Writer'],vectors[c]);positions=selected(c,mapping[i]['role_layout'],spec);allowed=len(KEEP[c])*len(positions)*576;den=32*R*576
            if c in NEW:assert all(g[k]==v for k,v in fm.items())
            rows.append(dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],condition=c,
                text=g['text'],prediction=g['prediction'],token_ids=json.dumps(g['token_ids']),correct=g['correct'],All_Writer_text=a['text'],All_correct=a['correct'],
                correct_to_wrong=int(a['correct'] and not g['correct']),wrong_to_correct=int(not a['correct'] and g['correct']),
                first_top1_flip=fm['first_top1_flip'],first_logp_loss=fm['first_logp_loss'],sequence_changed=g['token_ids']!=a['token_ids'],first_divergence=divergence(a['token_ids'],g['token_ids']),
                length=g['length'],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],
                kept_layers=len(KEEP[c]),writer_layers=json.dumps(KEEP[c]),selected_R_count=len(positions),total_R_count=R,writer_positions=json.dumps(positions),token_ratio=len(positions)/R,
                allowed_R_image_connections_per_head=allowed,all_R_image_connections_per_head=den,connection_budget_ratio=allowed/den,
                selection_binding=spec['selection_binding'],semantic_k=spec['selection']['k'],semantic_tokens=spec['writers_decoded']['Semantic-k'],
                input_hashes_sha256=digest(mapping[i]['input_hashes']),source='historical reused' if c in BASELINES else 'new main',
                source_checkpoint=str((reused_path(s,c) if c in BASELINES else OUT/'checkpoints'/key(s)/(c+'.json')).relative_to(OUT))))
        for c in NEW:
            j,l,se=records[c],records[MATCH[c]],records['Semantic-k']
            jointrows.append(dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],joint=c,layer=MATCH[c],
                all_correct=a['correct'],layer_correct=l['correct'],semantic_correct=se['correct'],joint_correct=j['correct'],
                all_text=a['text'],layer_text=l['text'],semantic_text=se['text'],joint_text=j['text'],
                interaction=interaction(j['correct'],l['correct'],se['correct'],a['correct']),joint_unique_harm=unique_harm(j['correct'],l['correct'],se['correct'],a['correct']),
                both_singles_correct_joint_wrong=bool(l['correct'] and se['correct'] and not j['correct']),
                joint_unique_correction=bool(not a['correct'] and not l['correct'] and not se['correct'] and j['correct'])))
    return p,samples,gens,rows,jointrows,newhash,reusedhash,rawhash

def statistics(p,samples,gens,rows,jointrows):
    scores=np.array([[gens[c][i]['correct'] for c in CONDITIONS] for i in range(240)],float)
    draws=paired_bootstrap(samples,scores,repetitions=20000,seed=SEED);pairs=[];ints=[]
    for d in p['comparisons']:
        a,b=d['a'],d['b'];ia,ib=CONDITIONS.index(a),CONDITIONS.index(b);lo,hi=np.percentile(100*(draws[:,ia]-draws[:,ib]),[2.5,97.5])
        pairs.append(dict(comparison=a+' - '+b,a=a,b=b,role=d['role'],delta_pp=float(100*(scores[:,ia]-scores[:,ib]).mean()),ci95_low_pp=float(lo),ci95_high_pp=float(hi),
            b_correct_a_wrong=int(((scores[:,ib]==1)&(scores[:,ia]==0)).sum()),b_wrong_a_correct=int(((scores[:,ib]==0)&(scores[:,ia]==1)).sum()),
            first_top1_flip=sum(gens[a][i]['token_ids'][0]!=gens[b][i]['token_ids'][0] for i in range(240)),questions=240,image_clusters=240,bootstrap_repetitions=20000,seed=SEED))
    for c in NEW:
        j,l,se,a=[CONDITIONS.index(x) for x in [c,MATCH[c],'Semantic-k','All-Writer']]
        distribution=100*interaction(draws[:,j],draws[:,l],draws[:,se],draws[:,a]);lo,hi=np.percentile(distribution,[2.5,97.5]);rr=[r for r in jointrows if r['joint']==c]
        ints.append(dict(joint=c,layer=MATCH[c],formula='Joint - Layer - Semantic + All',interaction_pp=float(100*interaction(scores[:,j],scores[:,l],scores[:,se],scores[:,a]).mean()),
            ci95_low_pp=float(lo),ci95_high_pp=float(hi),joint_unique_harm=sum(r['joint_unique_harm'] for r in rr),both_singles_correct_joint_wrong=sum(r['both_singles_correct_joint_wrong'] for r in rr),
            joint_unique_correction=sum(r['joint_unique_correction'] for r in rr),negative_interaction_questions=sum(r['interaction']<0 for r in rr),positive_interaction_questions=sum(r['interaction']>0 for r in rr),
            questions=240,image_clusters=240,bootstrap_repetitions=20000,seed=SEED))
    sums=[];tasks=[]
    for j,c in enumerate(CONDITIONS):
        rr=[r for r in rows if r['condition']==c];co=next((x for x in pairs if x['a']==c and x['b']=='All-Writer'),None);lo,hi=np.percentile(100*draws[:,j],[2.5,97.5])
        row=dict(condition=c,source='historical reused' if c in BASELINES else 'new main',kept_layers=len(KEEP[c]),questions=240,correct=int(scores[:,j].sum()),accuracy_pct=float(scores[:,j].mean()*100),
            accuracy_ci95_low_pct=float(lo),accuracy_ci95_high_pct=float(hi),delta_all_pp=float((scores[:,j]-scores[:,0]).mean()*100),
            delta_ci95_low_pp=co['ci95_low_pp'] if co else '',delta_ci95_high_pp=co['ci95_high_pp'] if co else '',
            connection_budget_pooled_pct=100*sum(r['allowed_R_image_connections_per_head'] for r in rr)/sum(r['all_R_image_connections_per_head'] for r in rr),
            connection_budget_macro_pct=100*float(np.mean([r['connection_budget_ratio'] for r in rr])),
            budget_min_pct=100*min(r['connection_budget_ratio'] for r in rr),budget_max_pct=100*max(r['connection_budget_ratio'] for r in rr),
            token_ratio_pooled_pct=100*sum(r['selected_R_count'] for r in rr)/sum(r['total_R_count'] for r in rr),token_ratio_macro_pct=100*float(np.mean([r['token_ratio'] for r in rr])))
        row.update(quality_flags(row['correct'],int(scores[:,0].sum())))
        for k in ['correct_to_wrong','wrong_to_correct','first_top1_flip','sequence_changed','eos','empty','truncated','repeated_trigram']:row[k]=sum(r[k] for r in rr)
        row['mean_first_logp_loss']=float(np.mean([r['first_logp_loss'] for r in rr]));sums.append(row)
        for t in TASKS:
            tr=[r for r in rr if r['task']==t];jr=[r for r in jointrows if r['joint']==c and r['task']==t]
            tasks.append(dict(condition=c,task=t,questions=40,correct=sum(r['correct'] for r in tr),accuracy_pct=100*sum(r['correct'] for r in tr)/40,
                delta_all_pp=100*(sum(r['correct'] for r in tr)-sum(r['All_correct'] for r in tr))/40,correct_to_wrong=sum(r['correct_to_wrong'] for r in tr),wrong_to_correct=sum(r['wrong_to_correct'] for r in tr),
                interaction_pp=100*sum(r['interaction'] for r in jr)/40 if jr else '',joint_unique_harm=sum(r['joint_unique_harm'] for r in jr) if jr else ''))
    return sums,pairs,ints,tasks,draws

def figures(sums,pairs,ints,tasks):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.15)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained');colors=['#555555','#98659b','#4e88ae','#62a6bd','#8bc0bd','#cd6747','#e39943','#9a563e']
    for r,col in zip(sums,colors):
        y=r['accuracy_pct'];ax.errorbar(r['connection_budget_pooled_pct'],y,yerr=[[y-r['accuracy_ci95_low_pct']],[r['accuracy_ci95_high_pct']-y]],fmt='o' if r['condition'] in NEW else 's',color=col,capsize=3,label=f"{r['condition']} ({r['correct']}/240)")
    for c in NEW:
        line=[next(r for r in sums if r['condition']==x) for x in [c,MATCH[c],'All-Writer']]
        ax.plot([r['connection_budget_pooled_pct'] for r in line],[r['accuracy_pct'] for r in line],color='#aaaaaa',ls=':',lw=1,zorder=0)
    for bound,style in [(-1,'--'),(-2,':')]:ax.axhline(sums[0]['accuracy_pct']+bound,color='#999999',ls=style,lw=1)
    ax.set(xlabel='Allowed R -> Image connections / All-Writer (%)',ylabel='Full-answer accuracy (%)',title='Layer x Semantic writing: fixed 240 questions; exploratory 95% CI',xlim=(-2,104))
    ax.legend(loc='best',fontsize=8,ncol=2);save(fig,'figure_quality_budget')
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    rr=[r for r in pairs if r['b']=='All-Writer'];axes[0].errorbar([r['delta_pp'] for r in rr],range(3),xerr=[[r['delta_pp']-r['ci95_low_pp'] for r in rr],[r['ci95_high_pp']-r['delta_pp'] for r in rr]],fmt='o',capsize=4)
    axes[0].set(yticks=range(3),yticklabels=NEW,xlabel='Joint minus All-Writer (pp)',title='Point screens do not establish non-inferiority');axes[0].axvline(-1,ls='--',color='gray');axes[0].axvline(-2,ls=':',color='gray')
    axes[1].errorbar([r['interaction_pp'] for r in ints],range(3),xerr=[[r['interaction_pp']-r['ci95_low_pp'] for r in ints],[r['ci95_high_pp']-r['interaction_pp'] for r in ints]],fmt='o',capsize=4,color='#b06a48')
    axes[1].set(yticks=range(3),yticklabels=NEW,xlabel='Joint - Layer - Semantic + All (pp)',title='Negative interaction = extra accuracy damage')
    for ax in axes:ax.axvline(0,color='gray');ax.invert_yaxis()
    save(fig,'figure_paired_interaction')
    matrix=np.array([[next(r['delta_all_pp'] for r in tasks if r['condition']==c and r['task']==t) for t in TASKS] for c in NEW]);lim=max(1.,abs(matrix).max())
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');im=ax.imshow(matrix,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    for i in range(3):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.7*lim else 'black')
    ax.set(xticks=range(6),xticklabels=TASKS,yticks=range(3),yticklabels=NEW,title='Change vs All-Writer by task (40/task; descriptive)');ax.tick_params(axis='x',rotation=25);fig.colorbar(im,ax=ax,label='Accuracy difference (pp)',shrink=.8);save(fig,'figure_tasks')

def report(p,sums,pairs,ints,tasks,jointrows):
    joint=[r for r in sums if r['condition'] in NEW];j12=next(r for r in joint if r['condition']=='Joint-12')
    lines=['# Layer × Semantic Joint Visual Writing','',
        '**COMPLETE：新增3×240＝720条完整答案；严格审计并逐字节复用五组历史对照1200条。**','',
        '## 核心回答','',
        f"Joint-12仅允许冻结Semantic-k Token在L0–L11读图，得到{j12['correct']}/240（{j12['accuracy_pct']:.3f}%），相对All-Writer 202/240为{j12['delta_all_pp']:+.3f} pp，探索性95%CI [{j12['delta_ci95_low_pp']:+.3f}, {j12['delta_ci95_high_pp']:+.3f}]。",'',
        f"Semantic-k占R Token的逐题平均为{j12['token_ratio_macro_pct']:.4f}%，全样本Token数比为{j12['token_ratio_pooled_pct']:.4f}%。因此Joint-12的可访问R→Image连接预算，逐题平均为{j12['connection_budget_macro_pct']:.4f}%，按实际连接数汇总为{j12['connection_budget_pooled_pct']:.4f}%。两种口径明确分列，不把预算当作FLOPs或延迟。",'',
        '下面分别判断净准确率、逐题新增损害和配对交互；两种单独条件有效不能自动推出联合安全。所有结论仅限这批历史复用探索样本。','',
        '## 冻结协议与历史复用','',
        f"层稀疏参考 `{BASE}`；语义选择参考 `{SEM_COMMIT}`；冻结执行源码 `{p['code_commit']}`。Manifest SHA256 `{MANIFEST_SHA}`。模型revision `{REVISION}`；FP16/Eager、Transformers4.49.0、Greedy、max_new_tokens=32、logits_to_keep=1。",'',
        'R为最后Image Token之后、P−1之前的全部非视觉文本Query。所有条件在全部32层屏蔽P−1及每个真实Decode Query（包含EOS预测）对576个原始Image Key的直接读取。Joint仅允许既属于冻结Semantic-k集合、又处于指定层的R Query读图，其余R→Image边屏蔽。只用Softmax前Hard Mask，不含o_proj消融、KV剪枝或Image Query改动；所有Attention、MLP、Residual和KV计算保留。',
        '历史token_group_manifest.json逐字节复制，未重新选词、未调用选择器。逐题核对Prompt IDs、Pixel Tensor、图像SHA、Gold、Tokenizer与评分器；复用检查点和原始首Token向量均经过哈希及内容审计。GPU Smoke进一步检验旧/新入口全部有效Logits严格一致以及保存的Token/首向量一致。复用原始记录保存在reused_checkpoints，来源与原提交见historical_reuse_audit.json。','',
        '## Accuracy与连接预算','',
        '|条件|层数|正确/240|Accuracy|连接预算：汇总 / 逐题平均|ΔAll pp|对→错 / 错→对|−1pp / −2pp点估计候选|',
        '|---|---:|---:|---:|---|---:|---|---|']
    for r in sums:lines.append(f"|{r['condition']}|{r['kept_layers']}|{r['correct']}|{r['accuracy_pct']:.3f}%|{r['connection_budget_pooled_pct']:.4f}% / {r['connection_budget_macro_pct']:.4f}%|{r['delta_all_pp']:+.3f}|{r['correct_to_wrong']} / {r['wrong_to_correct']}|{'是' if r['point_minus1_candidate'] else '否'} / {'是' if r['point_minus2_candidate'] else '否'}|")
    lines+=['','连接数按每个Head的原始Prefill R→Image潜在边计数：`保留层数 × 选中R数 × 576`，分母为`32 × 全部R数 × 576`；Head数相同而约去。不是非零概率数，也不包括其他Query/Key连接。每题预算和实际Hook允许数量经核对；分母随问题长度变化，宏平均不必等于汇总比。','',
        '## 三组联合的配对比较','', '|前者−后者|ΔAccuracy pp [探索性95%CI]|后者对→前者错 / 后者错→前者对|首Token翻转|','|---|---|---|---:|']
    for r in pairs:lines.append(f"|{r['comparison']}|{r['delta_pp']:+.3f} [{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}]|{r['b_correct_a_wrong']} / {r['b_wrong_a_correct']}|{r['first_top1_flip']}|")
    lines+=['','按Image SHA256整簇进行20,000次配对Bootstrap，seed=20261103，所有条件和交互共享相同抽样；240题对应240个不同图像簇。区间为未校正的探索性双侧百分位95%CI。首Token翻转不是完整答案正确率。','',
        '## 联合交互与联合独有损害','',
        '预先冻结交互为 `Joint − Layer-only − Semantic-only + All-Writer`，在每题0/1正确性上计算后对同一图片簇重采样；负值表示超出可加单独效应的额外损害。没有把两个边际CI相减。联合独有损害定义为All、Layer、Semantic均正确而Joint错误；另列不要求All正确、仅两单独条件均正确而联合错误的计数。','',
        '|Joint|交互 pp [95%CI]|联合独有损害|两单独均对、联合错|联合独有纠正|','|---|---|---:|---:|---:|']
    for r in ints:lines.append(f"|{r['joint']}|{r['interaction_pp']:+.3f} [{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}]|{r['joint_unique_harm']}|{r['both_singles_correct_joint_wrong']}|{r['joint_unique_correction']}|")
    lines+=['','交互CI跨0不能证明可加或无交互。净准确率可能掩盖损害与纠正的抵消；per_question_joint.csv保留720个逐题四条件组合。unique_joint_harm_cases.csv和negative_cases.csv保留全部相关损害，incorrect_answers.csv保留所有条件的所有错误答案。','',
        '## 质量界限','']
    for bound,flag in [(-1,'point_minus1_candidate'),(-2,'point_minus2_candidate')]:
        names=[r['condition'] for r in joint if r[flag]];lines.append(f"预设{bound}pp点估计界限的Joint候选："+('、'.join(names) if names else '无')+'。')
    lines+=['','点估计候选不等于非劣效证明；CI跨0不能证明无损或可安全组合。层集合由已完成探索实验选择，Semantic Token集合也来自同一历史样本，均不是独立确认集；高质量候选应留待独立数据验证。','',
        '## 六任务表现','', '|条件|'+'|'.join(TASKS)+'|','|---|'+'---:|'*6]
    for c in CONDITIONS:lines.append('|'+c+'|'+'|'.join(f"{next(r['accuracy_pct'] for r in tasks if r['condition']==c and r['task']==t):.1f}%" for t in TASKS)+'|')
    lines+=['','每任务40题，任务差异与交互仅作描述。task_accuracy.csv保留准确率、损害/纠正和Joint任务交互。','',
        '## 联合独有损害示例','', '按题号排序，各条件展示最前两例；全部案例均保留，不按结果筛题。','', '|Joint|题号|问题|Gold|All|Layer|Semantic|Joint答案|','|---|---|---|---|---|---|---|---|']
    for c in NEW:
        for r in sorted([r for r in jointrows if r['joint']==c and r['joint_unique_harm']],key=lambda r:r['question_id'])[:2]:
            lines.append('|'+ '|'.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ['joint','question_id','question','gold','all_text','layer_text','semantic_text','joint_text'])+'|')
    lines+=['','## 验证与资源','',
        'CPU测试、三道冻结题号00150746/11697551/11127723的真实7B Smoke均通过。HF/原生/三种No-op全部有效Logits保持torch.equal，不放宽容差；五种复用对照旧/新入口与保存向量严格匹配。实际Eager Softmax的每个禁止连接概率最大值严格为0；其余Mask不变、BOS/Self合法，行和误差沿用0.002界限。所有32层MLP、独立Cache存储和父Cache指纹、绝对位置、输入/权重版本及正常/异常Hook清理均检查。',
        f"仅使用物理GPU{p['gpu']['physical_index']}，UUID `{p['gpu']['uuid']}`；启动前重新确认空闲，一份模型串行。结束后进程退出，资源释放快照见gpu_release.json。没有抢占或终止其他进程。",
        '采集结束后复核全部模型文件SHA256、720新检查点、1200复用检查点、原始首Token向量和CSV每个单元格。原始大数组位于服务器资源目录，检查点记录路径/SHA256；GitHub不包含这些数组。协议、CPU/GPU验证及完整逐题答案均随本报告保存。',
        'Hard Mask会对剩余连接重新归一化；低敏感不等于没有视觉信息流。文本可通过其他文本状态中继视觉信息，图像KV及全部层计算仍存在。本轮只研究功能性联合稀疏，不证明唯一视觉写入位置、不减少实际FLOPs、不证明延迟或显存收益。','',
        '## 图表','', '![质量与连接预算](figure_quality_budget.png)','', '![配对差异与交互](figure_paired_interaction.png)','', '![六任务差异](figure_tasks.png)','',
        '## 停止边界','', '完成后停止；不追加物理Image Token退出、KV剪枝、Kernel优化、Router训练、跨模型或其他实验。']
    path=OUT/'report_zh.md';assert not path.exists();path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def exact_csv(name,values):
    b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=list(values[0]));w.writeheader();w.writerows(values);b.seek(0)
    with (OUT/name).open(newline='',encoding='utf-8') as f:assert list(csv.DictReader(f))==list(csv.DictReader(b)),name

def main(check=False):
    p,samples,gens,rows,jr,nh,rh,rawh=replay();sums,pairs,ints,tasks,draws=statistics(p,samples,gens,rows,jr)
    # Keep headers even if the experiment has no harms in a category.
    negatives=[]
    for d in p['comparisons']:
        for i,s in enumerate(samples):
            a,b=gens[d['a']][i],gens[d['b']][i]
            if b['correct'] and not a['correct']:negatives.append(dict(joint=d['a'],reference=d['b'],question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],reference_text=b['text'],joint_text=a['text']))
    unique=[r for r in jr if r['joint_unique_harm']];wrong=[r for r in rows if not r['correct']]
    tables={'accuracy_summary.csv':sums,'paired_comparisons.csv':pairs,'interaction_summary.csv':ints,'task_accuracy.csv':tasks,'per_question.csv':rows,'per_question_joint.csv':jr,'incorrect_answers.csv':wrong}
    optional={'negative_cases.csv':(negatives,['joint','reference','question_id','task','image_sha256','question','gold','reference_text','joint_text']),
        'unique_joint_harm_cases.csv':(unique,list(jr[0]))}
    if check:
        for n,v in tables.items():exact_csv(n,v)
        for n,(v,fields) in optional.items():
            if v:exact_csv(n,v)
            else:
                with (OUT/n).open(newline='',encoding='utf-8') as f:reader=csv.DictReader(f);assert reader.fieldnames==fields and not list(reader)
        va=read(OUT/'validation.json');assert va['checkpoint_hashes']==nh and va['reused_checkpoint_hashes']==rh and va['raw_hashes']==rawh
        with np.load(OUT/'bootstrap_replications.npz') as z:assert np.array_equal(z['accuracy'],draws)
        co=read(OUT/'completion.json');assert co['status']=='COMPLETE'
        for n,h in co['artifacts'].items():assert sha256(OUT/n)==h,n
        print('EXACT READ-ONLY PASS: every CSV cell, paired interactions, connection budgets, raw vectors, actual mask/cache records and artifact hashes',flush=True);return
    assert not (OUT/'completion.json').exists(),'No overwrite or duplicate reporting'
    for n,v in tables.items():csv_write(OUT/n,v)
    for n,(v,fields) in optional.items():
        with (OUT/n).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(v)
    for n,info in p['model_files'].items():assert sha256(n)==info['sha256'],n
    state=hardware();assert not any(s.split(',')[0].strip()==p['gpu']['uuid'] for s in state['compute_processes'].splitlines())
    write(OUT/'gpu_release.json',dict(passed=True,selected=p['gpu'],snapshot=state,gpu_worker_exited=True))
    np.savez_compressed(OUT/'bootstrap_replications.npz',accuracy=draws,conditions=np.array(CONDITIONS),seed=SEED)
    figures(sums,pairs,ints,tasks);report(p,sums,pairs,ints,tasks,jr)
    for n,v in tables.items():exact_csv(n,v)
    for n,(v,_) in optional.items():
        if v:exact_csv(n,v)
    write(OUT/'validation.json',dict(passed=True,binding=digest(p),new_main_generations=720,reused_main_records=1200,checkpoint_hashes=nh,reused_checkpoint_hashes=rh,raw_hashes=rawh,
        model_full_sha256_rechecked=True,exact_csv_cells_verified=True,statistics_replayed=True,actual_masks_cache_replayed=True,selection_recomputed=False,no_tolerance_relaxation=True))
    artifacts={str(f.relative_to(OUT)):sha256(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['worker.lock','completion.json']}
    write(OUT/'completion.json',dict(status='COMPLETE',complete=True,new_main_generations=720,reused_main_records=1200,binding=digest(p),code_commit=p['code_commit'],artifacts=artifacts))
    print('COMPLETE: 720 new, 1200 reused; all statistics and exact CSV cells audited',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');a=parser.parse_args()
    try:main(a.check)
    except BaseException:
        if not a.check and not (OUT/'completion.json').exists():
            write(OUT/'failures'/f'analysis_{time.time_ns()}.json',dict(status='PARTIAL',traceback=traceback.format_exc()))
            write(OUT/'completion.json',dict(status='PARTIAL',complete=False,failed_stage='analysis'))
        raise
