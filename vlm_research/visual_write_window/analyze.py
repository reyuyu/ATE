"""Read-only replay, whole-image paired bootstrap and Chinese report."""
import argparse,csv,json,time,traceback
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

def raw(row):
    assert sha256(row['raw']['path'])==row['raw']['sha256']
    with np.load(row['raw']['path']) as data:return torch.from_numpy(data['first_logits'].copy())

def validate(row,s,tokenizer,binding,input_hashes):
    g=row['generation'];c=row['condition'];assert c in CONDITIONS
    assert row['binding']==binding and row['sample_sha256']==digest(s) and row['result_sha256']==digest(g)
    assert row['input_hashes']==input_hashes and row['gold']==s['answer'] and row['image_sha256']==s['image_sha256']
    assert g['writer_layers']==writer_layers(c) and g['own_cache'] and g['own_prefix'] and g['answer_image_blocked']
    assert tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()==g['text']
    assert g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer'])
    assert g['length']==len(g['token_ids']) and 1<=g['length']<=32 and g['eos']==(g['token_ids'][-1]==2)
    assert g['truncated']==(g['length']==32 and not g['eos']) and g['empty']==(not bool(g['text'])) and g['repeated_trigram']==repeated_trigram(g['token_ids'])
    v=raw(row);assert torch.isfinite(v).all() and int(v.argmax())==g['token_ids'][0] and digest(v.tolist())==g['first_logits_sha256']
    l=g['query_layout'];n=l['prompt_length'];assert len(l['I'])==576 and l['R']==list(range(l['I'][-1]+1,n-1)) and l['A']==[n-1]
    assert len(g['source_events'])==32*g['length'] and len(g['cache_lengths'])==g['length']
    for t in range(g['length']):
        before=n+t-1 if t else 0
        assert g['cache_lengths'][t]==dict(t=t,before=before,after=n+t,absolute_query=n+t-1,consumed_token=g['token_ids'][t-1] if t else None)
        for k,e in enumerate(g['source_events'][32*t:32*t+32]):
            rblocked=not t and k not in writer_layers(c)
            assert e['layer']==k and e['phase']==('decode' if t else 'prefill') and e['absolute_start']==before
            assert e['query_length']==(1 if t else n) and e['key_length']==n+t
            assert e['R_image_open']==(k in writer_layers(c)) and e['answer_image_masked'] and e['R_image_masked']==rblocked
            assert e['blocked_connections_per_head']==576*(1+(len(l['R']) if rblocked else 0))
            assert e['blocked_probability_max']==0 and e['answer_image_mass_mean']==0 and e['row_sum_max_error']<=.002
            assert e['non_target_mask_unchanged'] and e['BOS_Self_legal']
            if rblocked:assert e['relay_image_mass_mean']==0
    return g,v

def replay():
    p=verify();binding=digest(p)
    assert read(OUT/'gpu_completion.json')['passed'] and read(OUT/'gpu_completion.json')['binding']==binding
    assert read(OUT/'gpu_smoke_validation.json')['passed'] and read(OUT/'image_independence_validation.json')['passed']
    samples=read(OUT/'sample_manifest.json')['samples'];mapping=read(OUT/'input_mappings.json')
    assert len(samples)==240 and len({s['image_sha256'] for s in samples})==240
    assert all(sum(s['task']==t for s in samples)==40 for t in TASKS)
    assert len(list((OUT/'checkpoints').glob('*/*.json')))==4320 and len(list((OUT/'swap_checkpoints').glob('*/*.json')))==80
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);gens={c:[] for c in CONDITIONS};rows=[];hashes={};raws={}
    for i,s in enumerate(samples):
        assert sha256(s['image_path'])==s['image_sha256'];records={};vectors={}
        for c in CONDITIONS:
            path=OUT/f"checkpoints/{s['task']}__{s['question_id']}/{c}.json";r=read(path)
            g,v=validate(r,s,tokenizer,binding,mapping[i]['input_hashes']);assert g['query_layout']==mapping[i]['role_layout']
            records[c]=g;vectors[c]=v;gens[c].append(g);hashes[str(path.relative_to(OUT))]=sha256(path);raws[r['raw']['path']]=r['raw']['sha256']
        for c,g in records.items():
            a=records['All-Writer'];b=records['No-Writer'];fm=first_metrics(vectors['All-Writer'],vectors[c]);nm=first_metrics(vectors['No-Writer'],vectors[c])
            if c!='All-Writer':assert all(g[k]==value for k,value in fm.items())
            rows.append(dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],condition=c,
                text=g['text'],prediction=g['prediction'],token_ids=json.dumps(g['token_ids']),correct=g['correct'],length=g['length'],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],
                all_correct_to_wrong=int(a['correct'] and not g['correct']),all_wrong_to_correct=int(not a['correct'] and g['correct']),
                no_correct_to_wrong=int(b['correct'] and not g['correct']),no_wrong_to_correct=int(not b['correct'] and g['correct']),
                first_top1_flip_all=fm['first_top1_flip'],first_top1_flip_no=nm['first_top1_flip'],first_logp_loss_all=fm['first_logp_loss'],first_logp_loss_no=nm['first_logp_loss'],
                sequence_changed_all=g['token_ids']!=a['token_ids'],sequence_changed_no=g['token_ids']!=b['token_ids'],first_divergence_all=divergence(a['token_ids'],g['token_ids']),
                all_writer_text=a['text'],no_writer_text=b['text'],writer_layers=json.dumps(writer_layers(c)),R_queries=len(g['query_layout']['R']),
                input_ids_sha256=mapping[i]['input_hashes']['input_ids'],pixel_values_sha256=mapping[i]['input_hashes']['pixel_values'],code_commit=p['code_commit'],source='new main',
                source_checkpoint=f"checkpoints/{s['task']}__{s['question_id']}/{c}.json"))
    swaprows=[];sm=read(OUT/'image_swap_inputs.json')
    for pair in read(OUT/'image_swap_manifest.json')['pairs']:
        gg=[];vv=[]
        for side,s in enumerate(pair['samples']):
            key=f"{pair['pair_id']}__{side}";path=OUT/f'swap_checkpoints/{key}/No-Writer.json';r=read(path)
            g,v=validate(r,s,tokenizer,binding,sm[key]['input_hashes']);gg.append(g);vv.append(v)
            hashes[str(path.relative_to(OUT))]=sha256(path);raws[r['raw']['path']]=r['raw']['sha256']
        assert torch.equal(*vv) and gg[0]['token_ids']==gg[1]['token_ids'] and gg[0]['valid_logits_sha256']==gg[1]['valid_logits_sha256']
        swaprows.append(dict(pair_id=pair['pair_id'],question=pair['question'],condition='No-Writer',left_id=pair['samples'][0]['question_id'],right_id=pair['samples'][1]['question_id'],
            left_gold=pair['samples'][0]['answer'],right_gold=pair['samples'][1]['answer'],left_text=gg[0]['text'],right_text=gg[1]['text'],left_correct=gg[0]['correct'],right_correct=gg[1]['correct'],
            both_correct=bool(gg[0]['correct'] and gg[1]['correct']),answers_changed=gg[0]['prediction']!=gg[1]['prediction'],tokens_changed=False,
            all_effective_logits_exact=True,first_JS_nats=0.,left_token_ids=json.dumps(gg[0]['token_ids']),right_token_ids=json.dumps(gg[1]['token_ids'])))
    return p,samples,gens,rows,swaprows,hashes,raws

def statistics(samples,gens,rows):
    score=np.array([[gens[c][i]['correct'] for c in CONDITIONS] for i in range(240)],float)
    draws=paired_bootstrap(samples,score,repetitions=20000,seed=20261102)
    wanted=[(c,b,'reference') for c in CONDITIONS for b in ['All-Writer','No-Writer'] if c!=b]
    wanted +=[(f'Only-W{k}','No-Writer','sufficiency') for k in range(8)]+[('All-Writer',f'Drop-W{k}','necessity') for k in range(8)]
    pairs=[]
    for a,b,role in wanted:
        ia,ib=CONDITIONS.index(a),CONDITIONS.index(b);lo,hi=np.percentile(100*(draws[:,ia]-draws[:,ib]),[2.5,97.5]);d=score[:,ia]-score[:,ib]
        pairs.append(dict(comparison=a+' - '+b,a=a,b=b,role=role,delta_pp=float(d.mean()*100),ci95_low_pp=float(lo),ci95_high_pp=float(hi),
            b_correct_a_wrong=int(((score[:,ib]==1)&(score[:,ia]==0)).sum()),b_wrong_a_correct=int(((score[:,ib]==0)&(score[:,ia]==1)).sum()),
            seed=20261102,bootstrap_repetitions=20000,image_clusters=240,questions=240))
    sums=[];tasks=[]
    for j,c in enumerate(CONDITIONS):
        rr=[r for r in rows if r['condition']==c];lo,hi=np.percentile(100*draws[:,j],[2.5,97.5])
        row=dict(condition=c,correct=int(score[:,j].sum()),questions=240,accuracy_pct=float(score[:,j].mean()*100),ci95_accuracy_low_pct=float(lo),ci95_accuracy_high_pct=float(hi),
            delta_all_pp=float((score[:,j]-score[:,0]).mean()*100),delta_no_pp=float((score[:,j]-score[:,1]).mean()*100))
        for key in ['all_correct_to_wrong','all_wrong_to_correct','no_correct_to_wrong','no_wrong_to_correct','first_top1_flip_all','first_top1_flip_no','sequence_changed_all','sequence_changed_no','eos','empty','truncated','repeated_trigram']:
            row[key]=sum(r[key] for r in rr)
        row['mean_first_logp_loss_all']=float(np.mean([r['first_logp_loss_all'] for r in rr]));row['mean_generated_tokens']=float(np.mean([r['length'] for r in rr]));sums.append(row)
        for task in TASKS:
            tr=[r for r in rr if r['task']==task];ii=[i for i,s in enumerate(samples) if s['task']==task]
            tasks.append(dict(task=task,condition=c,questions=40,correct=sum(r['correct'] for r in tr),accuracy_pct=100*sum(r['correct'] for r in tr)/40,
                delta_all_pp=float((score[ii,j]-score[ii,0]).mean()*100),delta_no_pp=float((score[ii,j]-score[ii,1]).mean()*100),
                all_correct_to_wrong=sum(r['all_correct_to_wrong'] for r in tr),all_wrong_to_correct=sum(r['all_wrong_to_correct'] for r in tr)))
    return sums,pairs,tasks

def figures(sums,pairs,tasks):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5));x=np.arange(18);ys=[r['accuracy_pct'] for r in sums]
    ax.bar(x,ys,color=['#486b89','#999999']+['#ce7258']*8+['#75a5ad']*8)
    ax.errorbar(x,ys,yerr=[[v-r['ci95_accuracy_low_pct'] for v,r in zip(ys,sums)],[r['ci95_accuracy_high_pct']-v for v,r in zip(ys,sums)]],fmt='none',color='black',capsize=2)
    for k,r in enumerate(sums):ax.text(k,ys[k]+1,str(r['correct']),ha='center',fontsize=8)
    ax.axhline(ys[0],ls='--',color='#486b89',label='All-Writer');ax.axhline(ys[1],ls=':',color='gray',label='No-Writer')
    ax.set(xticks=x,xticklabels=CONDITIONS,ylabel='Accuracy (%)',ylim=(0,103));ax.tick_params(axis='x',rotation=50);ax.legend();fig.tight_layout();save(fig,'figure_accuracy')
    primary=[r for r in pairs if r['role'] in ['sufficiency','necessity']]
    matrix=np.array([[r['delta_pp'] for r in primary if r['role']==role] for role in ['sufficiency','necessity']])
    limit=max(1.,abs(matrix).max());fig,ax=plt.subplots(figsize=(13,3.5));im=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    for y,role in enumerate(['sufficiency','necessity']):
        for k,r in enumerate([r for r in primary if r['role']==role]):
            ax.text(k,y,f"{r['delta_pp']:+.2f}\n[{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}]",ha='center',va='center',fontsize=9,color='black')
    ax.set(xticks=range(8),xticklabels=[f'W{k}\nL{4*k}-{4*k+3}' for k in range(8)],yticks=[0,1],yticklabels=['Only - No\nSufficiency','All - Drop\nNecessity'])
    fig.colorbar(im,ax=ax,label='Accuracy difference (pp); exploratory 95% CI');fig.tight_layout();save(fig,'figure_window_effects')
    fig,axes=plt.subplots(2,1,figsize=(12,8))
    mats=[]
    for role in ['sufficiency','necessity']:
        mats.append(np.array([[next(r['delta_no_pp'] if role=='sufficiency' else -r['delta_all_pp'] for r in tasks if r['task']==t and r['condition']==('Only-' if role=='sufficiency' else 'Drop-')+f'W{k}') for k in range(8)] for t in TASKS]))
    limit=max(1.,max(abs(m).max() for m in mats))
    for ax,m,title in zip(axes,mats,['Only - No: sufficiency by task','All - Drop: necessity by task']):
        im=ax.imshow(m,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        for y in range(6):
            for k in range(8):ax.text(k,y,f'{m[y,k]:+.1f}',ha='center',va='center',fontsize=9)
        ax.set(xticks=range(8),xticklabels=[f'W{k}' for k in range(8)],yticks=range(6),yticklabels=TASKS,title=title)
    fig.colorbar(im,ax=axes.tolist(),label='Accuracy difference (pp)',fraction=.025,pad=.02);fig.subplots_adjust(left=.14,right=.87,hspace=.38);save(fig,'figure_task_effects')

def report(p,sums,pairs,tasks,swaps,rows):
    primary=[r for r in pairs if r['role']!='reference'];allr,nor=sums[:2]
    suff=[r for r in primary if r['role']=='sufficiency'];nec=[r for r in primary if r['role']=='necessity']
    best=max(suff,key=lambda r:r['delta_pp']);strong=[r for r in suff if r['ci95_low_pp']>0];needed=[r for r in nec if r['ci95_low_pp']>0]
    lines=['# Visual Write-Window Discovery：视觉写入窗口探索','',
        '**COMPLETE：240题×18条件共4320条主实验答案全部重新生成，另有40组×2端No-Writer换图控制80条；两者分开统计。**','',
        '## 核心回答','',
        f"All-Writer 为 {allr['correct']}/240（{allr['accuracy_pct']:.3f}%），No-Writer 为 {nor['correct']}/240（{nor['accuracy_pct']:.3f}%）。单窗口中最高的增益来自 {best['a']}：相对 No-Writer {best['delta_pp']:+.3f} pp，探索性95%CI [{best['ci95_low_pp']:+.3f}, {best['ci95_high_pp']:+.3f}]。",'',
        f"充分性增益区间完全高于0的窗口为：{', '.join(r['a'] for r in strong) or '无'}。移除后损害区间完全高于0的窗口为：{', '.join(r['b'] for r in needed) or '无'}。这些是本批Hard-Mask控制下的功能敏感性证据，不是窗口普遍充分、无损、唯一或不可替代的证明；全部16项窗口比较均为探索性，未进行确认性多重比较校正。",'',
        '## 协议与因果范围','',
        '所有18条件都在32层禁止P−1及每一个真实Decode Query直接读取原始576个Image Key，包括EOS预测。唯一变化是R可直接读Image Key的层集合。R包括图像后、P−1前的问题、模板、指令及空白。Image Query、其他文本边、BOS/Self、MLP、Residual、RoPE、KV保留；没有o_proj消融、物理跳算或KV删除。Hard Mask会使剩余连接重归一化。',
        '“写入”在本文只是允许R直接访问原始Image Key的操作定义；其他层可传播已经写入的非图像状态，开放层的Image KV本身也已历经此前计算。结果不能定位唯一语义写入Token或把Accuracy差解释为信息量。',
        f"固定模型revision `{REVISION}`，FP16/Eager、Transformers4.49.0、Greedy、max_new_tokens=32、显式logits_to_keep=1；各条件从空真实KV独立生成。一份模型、单GPU串行。源码Commit `{p['code_commit']}`。",'',
        '## 各条件完整答案结果','',
        '|条件|正确/240|Accuracy|ΔAll pp|ΔNo pp|All对→错/错→对|首Token翻转vsAll|EOS/截断/空/重复|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in sums:lines.append(f"|{r['condition']}|{r['correct']}|{r['accuracy_pct']:.3f}%|{r['delta_all_pp']:+.3f}|{r['delta_no_pp']:+.3f}|{r['all_correct_to_wrong']}/{r['all_wrong_to_correct']}|{r['first_top1_flip_all']}|{r['eos']}/{r['truncated']}/{r['empty']}/{r['repeated_trigram']}|")
    lines+=['','## 窗口充分性与必要性','',
        'Only−No衡量仅开放该4层窗口在No-Writer背景下的功能增益；All−Drop衡量在其他28层开放背景下移除该窗口的净损害。二者背景不同，不能直接相加。CI跨0不证明无损或没有视觉信息流。','',
        '|窗口|层|Only−No pp [95%CI]|Only相对No 对→错/错→对|All−Drop pp [95%CI]|Drop相对All 对→错/错→对|',
        '|---|---|---|---|---|---|']
    for k,(a,b) in enumerate(zip(suff,nec)):
        lines.append(f"|W{k}|L{4*k}–L{4*k+3}|{a['delta_pp']:+.3f} [{a['ci95_low_pp']:+.3f}, {a['ci95_high_pp']:+.3f}]|{a['b_correct_a_wrong']}/{a['b_wrong_a_correct']}|{b['delta_pp']:+.3f} [{b['ci95_low_pp']:+.3f}, {b['ci95_high_pp']:+.3f}]|{b['b_wrong_a_correct']}/{b['b_correct_a_wrong']}|")
    lines+=['','完整的每条件对All和No的差值与区间、首Token变化、signed LogP loss及损益见accuracy_summary.csv、paired_comparisons.csv及per_question.csv。全部比较使用同一组20,000次Image SHA256整簇配对Bootstrap抽样，seed=20261102，双侧百分位探索性95%CI。240题对应240个不同图像簇。','',
        '## 六任务结果','', '|条件|'+'|'.join(TASKS)+'|','|---|'+'---:|'*6]
    for c in CONDITIONS:lines.append('|'+c+'|'+'|'.join(f"{next(r['accuracy_pct'] for r in tasks if r['task']==t and r['condition']==c):.1f}%" for t in TASKS)+'|')
    lines+=['','每类40题，任务差异为描述性，不作为独立确认检验；完整损益见task_accuracy.csv。','',
        '## No-Writer 图像独立性','',
        f"历史冻结40组异图同文本且Gold不同的配对中，No-Writer答案随图改变 {sum(r['answers_changed'] for r in swaps)}/40，生成Token变化0/40，两端均正确 {sum(r['both_correct'] for r in swaps)}/40。全部40组的每一步有效Logits使用torch.equal严格一致，最大差异0；首Token完整分布JS为0。不同Gold下两端不能同时正确。",'',
        '该控制在正式主矩阵前通过，未按模型表现重选配对。它排除了该控制集中No-Writer通过未屏蔽图像路径形成回答的现象；不把No-Writer依靠语言先验的主实验正确题归为视觉理解。配对来自历史3000题冻结子集，任务不均衡，与主240题单独统计。','',
        '## 负面案例','',
        '全部相对All-Writer由对变错的记录保存于negative_cases.csv，不删除反例。以下按条件和题号固定排序展示前12条。','',
        '|条件|题号|问题|Gold|All-Writer|干预答案|','|---|---|---|---|---|---|']
    neg=sorted([r for r in rows if r['all_correct_to_wrong']],key=lambda r:(r['condition'],r['question_id']))
    for r in neg[:12]:lines.append('|'+ '|'.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ['condition','question_id','question','gold','all_writer_text','text'])+'|')
    lines+=['','## 正确性、复用与资源','',
        'CPU全部测试及三道固定真实7B GPU Smoke通过，Smoke覆盖18条件。HF原生、自定义Native和No-op全部有效Logits同形状torch.equal。32层MLP正常，独立Cache存储与父Cache指纹、绝对位置、正常及异常Hook清理、输入和权重版本验证通过。实际Softmax后禁止概率严格为0；概率行和沿用历史0.002界限。每条正式生成记录每层每次真实Forward的作用范围和实际概率审计，全部检查点与原始首向量重新核验。',
        '历史Handoff B语义相近，但主Manifest仅20/240题重合且旧LM Head路径未显式logits_to_keep=1，未复用。历史原生Full允许答案读Image Key，也不等同All-Writer。全部4320条主实验新生成，没有混入旧分数。',
        f"Manifest原样复用，实际SHA256 `{MANIFEST_SHA}`；保持原Question ID、Gold、图像SHA、Prompt IDs和Pixel Tensor哈希。该240题为历史复用探索样本，不是独立确认集。物理GPU{p['gpu']['physical_index']}，UUID `{p['gpu']['uuid']}`；运行前实查空闲，未终止其他进程，退出后的资源状态见gpu_release.json。",'',
        '大型原始首Token数组保存在服务器资源目录，检查点记录路径及SHA256；GitHub不包含这些数组。模型文件完整SHA256在冻结前与采集后核验。','',
        '## 图表','', '![各条件准确率](figure_accuracy.png)','', '![窗口充分性与必要性](figure_window_effects.png)','', '![六任务窗口差异](figure_task_effects.png)','',
        '## 解释边界与停止','',
        '少数Only窗口若表现接近All，只能表明在这批样本和路径定义下可支持较多任务；没有预声明非劣效界限，不能宣称等价。多个Only有效且Drop损失小时，与可替代写入通路相容；若Only都显著落后All，则与跨层协作相容，但不能仅靠这些消融证明具体补偿中介机制。必要性是相对于其他窗口开放背景的条件必要性，不能声称不可替代的唯一层。',
        '本轮仅提供视觉写入阶段的功能机制证据；未测得真实加速或显存收益。后续“稀疏写入、轻量读取”仍需独立样本验证。本轮不自动实施Activation Patching、跨模型、Router、Kernel或KV剪枝。']
    text='\n'.join(lines)+'\n';path=OUT/'report_zh.md'
    assert not path.exists();path.write_text(text,encoding='utf-8')

def csvrows(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def check_csv(name,rows):
    actual=csvrows(OUT/name);assert len(actual)==len(rows)
    for a,b in zip(actual,rows):
        assert set(a)==set(b)
        for k,v in b.items():
            if isinstance(v,float):assert abs(float(a[k])-v)<1e-12,(name,k)
            else:assert a[k]==str(v),(name,k,a[k],v)

def main(check=False):
    p,samples,gens,rows,swaps,hashes,raws=replay();sums,pairs,tasks=statistics(samples,gens,rows)
    tables={'accuracy_summary.csv':sums,'paired_comparisons.csv':pairs,'task_accuracy.csv':tasks,'per_question.csv':rows,
        'image_swap_per_pair.csv':swaps,'negative_cases.csv':[r for r in rows if r['all_correct_to_wrong']]}
    if check:
        for name,values in tables.items():check_csv(name,values)
        c=read(OUT/'completion.json');assert c['status']=='COMPLETE'
        for name,h in c['artifacts'].items():assert sha256(OUT/name)==h,name
        assert read(OUT/'validation.json')['checkpoint_hashes']==hashes and read(OUT/'validation.json')['raw_hashes']==raws
        print('READ-ONLY PASS: all records, source masks, cache traces, raw arrays, statistics and final artifact hashes',flush=True);return
    assert not (OUT/'completion.json').exists(),'No duplicate reporting or overwrite'
    for name,values in tables.items():csv_write(OUT/name,values)
    for path,info in p['model_files'].items():assert sha256(path)==info['sha256'],path
    gpu=hardware();assert not any(x.split(',')[0].strip()==p['gpu']['uuid'] for x in gpu['compute_processes'].splitlines())
    write(OUT/'gpu_release.json',dict(passed=True,selected=p['gpu'],snapshot=gpu,gpu_worker_exited=True))
    figures(sums,pairs,tasks);report(p,sums,pairs,tasks,swaps,rows)
    write(OUT/'validation.json',dict(passed=True,binding=digest(p),questions=240,conditions=18,new_main_generations=4320,reused_main_generations=0,
        swap_generations=80,checkpoint_hashes=hashes,raw_hashes=raws,model_full_sha256_rechecked=True,statistics_replayed=True,
        actual_probability_replayed=True,independent_cache_positions_replayed=True,no_tolerance_relaxation=True))
    artifacts={str(f.relative_to(OUT)):sha256(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['worker.lock','completion.json']}
    write(OUT/'completion.json',dict(status='COMPLETE',complete=True,binding=digest(p),new_main_generations=4320,reused_main_generations=0,swap_generations=80,code_commit=p['code_commit'],artifacts=artifacts))
    print('COMPLETE reporting; frozen data and all comparisons audited',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    try:main(args.check)
    except BaseException:
        if not args.check and not (OUT/'completion.json').exists():
            write(OUT/'failures'/f'analysis_{time.time_ns()}.json',dict(status='PARTIAL',traceback=traceback.format_exc()))
            write(OUT/'completion.json',dict(status='PARTIAL',complete=False,failed_stage='analysis'))
        raise
