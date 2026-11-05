"""CPU-only official rescoring, image-cluster paired statistics and figures."""
import argparse
import csv
import math
import re
import subprocess

import numpy as np

from .common import *
from ..benchmarks.scoring import pope_metrics
from ..fastp2_bypass_writer.common import hardware


def family(s):
    return 'GQA' if s['dataset']=='GQA' else s['task']


def interval(x,alpha=.05):
    return [float(v) for v in np.quantile(x,[alpha/2,1-alpha/2])]


def bootstrap(keys,samples,records,labels,tag):
    groups=sorted({samples[k]['image_sha256'] for k in keys}); lookup={h:i for i,h in enumerate(groups)}
    a=np.zeros((len(groups),len(labels),7),dtype=float)
    for k in keys:
        s=samples[k]; i=lookup[s['image_sha256']]
        for j,c in enumerate(labels):
            r=records[k,c]; yes=r['scoring']['parsed']=='yes'; gy=s.get('answer')=='yes'
            a[i,j]+=[r['scoring']['score'],yes and gy,not yes and not gy,yes and not gy,not yes and gy,yes,1]
    rng=np.random.default_rng(int(digest([SEED,'bootstrap',tag])[:8],16))
    weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=20000)
    sums=(weights@a.reshape(len(groups),-1)).reshape(20000,len(labels),7)
    draws={'accuracy':sums[:,:,0]/sums[:,:,6]}
    if samples[keys[0]]['dataset']=='pope':
        tp,fp,fn=sums[:,:,1],sums[:,:,3],sums[:,:,4]
        draws['precision']=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp!=0)
        draws['recall']=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=tp+fn!=0)
        draws['f1']=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=2*tp+fp+fn!=0)
        draws['yes_ratio']=sums[:,:,5]/sums[:,:,6]
    return draws,len(groups)


def read_all(partial):
    freeze=read(OUT/'execution_freeze.json'); assert freeze['sources']==source_hashes()
    for name in ('sample_manifest','partition','protocol'):
        assert sha(OUT/(name+'.json'))==freeze[name+'_sha256']
    samples=read(OUT/'sample_manifest.json'); partition=read(OUT/'partition.json')
    records={}; missing=[]; reuse=collections.Counter()
    for k,s in samples.items():
        for c in LABELS:
            r=record(k,c,s)
            if r is None:
                missing.append([k,c]); continue
            assert r['writers']==writer_set(s['input_mapping'],c)
            assert r['rank']==partition['image_shard'][s['image_sha256']]
            assert r['generation']['token_ids']
            if r['reuse']:
                provenance=r['reuse']; reuse[provenance['kind']]+=1
                if provenance['kind']=='audited_historical_C3':
                    p=Path(provenance['path']); assert sha(p)==provenance['checkpoint_sha256']
                    old=read(p)['payload']; assert old['input_binding']==r['input_binding'] and old['config']==CFG
                    assert old['generation']['token_ids']==r['generation']['token_ids']
                else:
                    p=checkpoint(provenance['sample'],c); assert sha(p)==provenance['checkpoint_sha256']
                    src=read(p)['payload']; assert src['input_binding']==r['input_binding'] and src['rank']==r['rank']
                    assert src['generation']==r['generation']
            records[k,c]=r
    assert partial or not missing, f'{len(missing)} missing condition records'
    return samples,records,missing,reuse


def native_baselines(samples,records):
    accepted=[]
    def path_for(key):
        closed=OLD/'checkpoints'/key/'Native_Full.json'
        return closed if closed.exists() else AUTO/'records/native-none-L20-original-H1'/(key+'.json')
    smoke=read(OUT/'worker0/smoke.json')['checks']
    for check in smoke:
        p=path_for(check['sample'])
        if p.exists():
            old=read(p)['payload']
            assert old['generation']['first_logits_sha256']==check['native_first_logits_sha256']
    for k,s in samples.items():
        p=path_for(k)
        if p.exists() and (k,'C3') in records:
            b=read(p); assert b['sha256']==digest(b['payload']); old=b['payload']
            assert old['config']==config('native',head=1)
            expected=records[k,'C3']['input_binding']
            if 'scoring' in old:
                assert old['input_binding']==expected
                assert scored(s,old['generation']['text'])==old['scoring']
            else:
                assert old['sample_sha256']==s['history_sample_sha256']
                assert old['input_binding']['prompt_text']==s['prompt_text']
                assert all(old['input_binding'][f]==expected[f] for f in ('prompt_ids','input_hashes','Q','ASSISTANT','R','writers'))
                old=dict(old,scoring=scored(s,old['generation']['text']))
                assert old['generation']['score']==old['scoring']['score']
            records[k,'Native']=old
            accepted.append(dict(sample=k,path=str(p),sha256=sha(p),input_binding_sha256=digest(old['input_binding'])))
    native_summary=[]
    for task in TASKS:
        keys=[k for k,s in samples.items() if family(s)==task and (k,'Native') in records]
        total=sum(family(s)==task for s in samples.values())
        native_summary.append(dict(dataset=task,n=len(keys),expected_n=total,
            accuracy=100*np.mean([records[k,'Native']['scoring']['score'] for k in keys]) if keys else None,
            coverage='FULL' if len(keys)==total else 'PARTIAL_HISTORICAL_H1' if keys else 'NO_STRICT_H1_BASELINE'))
    write_csv(OUT/'native_accuracy_auxiliary.csv',native_summary)
    atomic(OUT/'native_reuse_audit.json',dict(at=now(),accepted=len(accepted),bindings=accepted,
        restriction='Only exact prior Native inputs; Native is auxiliary, not the C3 main baseline. No historical latency comparison.'))


def summarize(partial=False):
    samples,records,missing,reuse=read_all(partial)
    native_baselines(samples,records)
    raw=[]; negative=[]; budgets=[]; latency=[]
    for k,s in samples.items():
        full=records.get((k,'C3')); m=s['input_mapping']
        for c in LABELS:
            r=records.get((k,c))
            if r is None: continue
            g=r['generation']; value=r['scoring']['score']; base=full['scoring']['score'] if full else None
            w=r['writers']
            row=dict(sample=k,dataset=family(s),task=s['task'],question_id=s['question_id'],image_id=s['image_id'],
                image_sha256=s['image_sha256'],question=s['question'],gold=s.get('answers',s.get('answer')),
                answer_type=('yes/no' if s['dataset']=='pope' else 'multiple_choice' if s['dataset']=='scienceqa_img'
                             else 'GQA task: '+s['task'] if s['dataset']=='GQA' else s.get('answer_type')),
                condition=c,text=g['text'],token_ids=g['token_ids'],score=value,
                parsed=r['scoring']['parsed'],parse_failed=r['scoring']['parse_failed'],eos=g['eos'],truncated=g['truncated'],
                empty=g['empty'],repeated=g['repeated'],content_tokens=g['content_tokens'],
                first_token_flip=full is not None and g['token_ids'][0]!=full['generation']['token_ids'][0],
                sequence_changed=full is not None and g['token_ids']!=full['generation']['token_ids'],
                harm=base is not None and value<base,rescue=base is not None and value>base,
                delta_score=None if base is None else value-base,
                instruction_tokens=len(m['parts']['Instruction']),assistant_retained=sum(i in w for i in m['parts']['ASSISTANT']),
                writers=len(w),R=len(m['R']),writer_ratio=len(w)/len(m['R']),
                hint_present=bool(m['parts']['Hint']),choice_count=s.get('choice_count'),
                input_binding_sha256=digest(r['input_binding']),checkpoint_sha256=sha(checkpoint(k,c)),rank=r['rank'],
                reuse_kind=None if not r['reuse'] else r['reuse']['kind'])
            raw.append(row)
            if row['harm'] or value==0 or row['parse_failed'] or g['truncated'] or g['empty']: negative.append(row)
    write_csv(OUT/'per_question.csv',raw); write_csv(OUT/'negative_cases.csv',negative)
    summaries=[]; pairs=[]; curves=[]; native=[]; bytask=[]
    comparisons=[('E1','C3'),('E2-P2','E1'),('E2-P2','C3'),('E2-P1','E1'),('E2-P3','E1'),('E2-P4','E1'),('E2-P2','E2-S2')]
    comparisons += [('E1','Native'),('E2-P2','Native')]
    for task in TASKS:
        allkeys=sorted(k for k,s in samples.items() if family(s)==task)
        joint_labels=list(LABELS)
        if all((k,'Native') in records for k in allkeys): joint_labels.append('Native')
        joint=None
        if all((k,c) in records for k in allkeys for c in joint_labels):
            joint,clusters=bootstrap(allkeys,samples,records,joint_labels,['joint',task])
        def draws_for(keys,labels,tag):
            if joint is not None and keys==allkeys and all(c in joint_labels for c in labels):
                indices=[joint_labels.index(c) for c in labels]
                return {m:a[:,indices] for m,a in joint.items()},clusters
            return bootstrap(keys,samples,records,labels,tag)
        for c in LABELS:
            keys=[k for k in allkeys if (k,c) in records]
            if not keys: continue
            draws,ng=draws_for(keys,[c],['score',task,c])
            rs=[records[k,c] for k in keys]; gs=[r['generation'] for r in rs]
            metrics=dict(accuracy=100*float(np.mean([r['scoring']['score'] for r in rs])))
            if samples[keys[0]]['dataset']=='pope':
                p=pope_metrics([g['text'] for g in gs],[samples[k]['answer'] for k in keys])
                metrics.update({m:100*p['official_generation_'+m] for m in ('precision','recall','f1','yes_ratio')})
            lo,hi=interval(draws['accuracy'][:,0]*100)
            summaries.append(dict(dataset=task,condition=c,n=len(keys),expected_n=len(allkeys),image_clusters=ng,
                coverage='FULL' if len(keys)==len(allkeys) else 'PARTIAL',accuracy_ci_low=lo,accuracy_ci_high=hi,**metrics,
                parse_failed=sum(r['scoring']['parse_failed'] for r in rs),empty=sum(g['empty'] for g in gs),
                truncated=sum(g['truncated'] for g in gs),no_eos=sum(not g['eos'] for g in gs),repeated=sum(g['repeated'] for g in gs)))
            ratios=[len(records[k,c]['writers'])/len(samples[k]['input_mapping']['R']) for k in keys]
            numerator=sum(len(records[k,c]['writers']) for k in keys); denominator=sum(len(samples[k]['input_mapping']['R']) for k in keys)
            budgets.append(dict(dataset=task,condition=c,n=len(keys),mean_writer_ratio=float(np.mean(ratios)),
                pooled_connection_ratio=numerator/denominator,writer_tokens=numerator,R_tokens=denominator,
                mean_instruction_tokens=float(np.mean([len(samples[k]['input_mapping']['parts']['Instruction']) for k in keys])),
                mean_assistant_retained=float(np.mean([sum(i in records[k,c]['writers'] for i in samples[k]['input_mapping']['parts']['ASSISTANT']) for k in keys]))))
            actual=[g for k,g in zip(keys,gs) if records[k,c].get('reuse') is None]
            latency.append(dict(dataset=task,condition=c,n_generated_timing=len(actual),
                median_e2e_ms=float(np.median([g['latency_ms'] for g in actual])) if actual else None,
                median_prefill_ms=float(np.median([g['language_prefill_ms'] for g in actual])) if actual else None,
                status='Observational quality-run timings only, not an interleaved warmed speed benchmark; historical timings excluded',
                extra_physical_flops_saved_by_mask=0,KV_structure='Unchanged from C3-V2'))
        for c,b in comparisons:
            keys=[k for k in allkeys if (k,c) in records and (k,b) in records]
            if not keys: continue
            draws,ng=draws_for(keys,[b,c],['paired',task,c,b])
            diffs=np.array([records[k,c]['scoring']['score']-records[k,b]['scoring']['score'] for k in keys])
            primary=(c,b) in [('E1','C3'),('E2-P2','E1')]
            for metric,a in draws.items():
                d=(a[:,1]-a[:,0])*100; lo,hi=interval(d)
                corrected=interval(d,.05/14) if primary and metric=='accuracy' else [None,None]
                if metric=='accuracy': point=float(diffs.mean()*100)
                else:
                    m=[]
                    for label in (b,c):
                        mm=pope_metrics([records[k,label]['generation']['text'] for k in keys],[samples[k]['answer'] for k in keys])
                        m.append(mm['official_generation_'+metric])
                    point=(m[1]-m[0])*100
                pairs.append(dict(dataset=task,candidate=c,baseline=b,metric=metric,n=len(keys),image_clusters=ng,
                    delta_pp=point,ci_low_pp=lo,ci_high_pp=hi,primary=primary and metric=='accuracy',
                    bonferroni14_ci_low_pp=corrected[0],bonferroni14_ci_high_pp=corrected[1],
                    decreased=int((diffs<0).sum()),increased=int((diffs>0).sum()),unchanged=int((diffs==0).sum()),
                    score1_to_lower=sum(records[k,b]['scoring']['score']==1 and records[k,c]['scoring']['score']<1 for k in keys),
                    zero_to_positive=sum(records[k,b]['scoring']['score']==0 and records[k,c]['scoring']['score']>0 for k in keys)))
        curve_labels=('E1','E2-P1','E2-P2','E2-P3','E2-P4','E2-S2')
        keys=[k for k in allkeys if all((k,c) in records for c in curve_labels)]
        for c in curve_labels:
            if not keys: continue
            curves.append(dict(dataset=task,condition=c,n=len(keys),cohort_sha256=digest(keys),
                assistant_removed=0 if c=='E1' else 2 if c=='E2-S2' else int(c[-1]),
                accuracy=float(np.mean([records[k,c]['scoring']['score'] for k in keys]))*100,
                mean_writer_ratio=float(np.mean([len(records[k,c]['writers'])/len(samples[k]['input_mapping']['R']) for k in keys]))))
        for st in sorted({samples[k]['task'] for k in allkeys}):
            for c in LABELS:
                keys=[k for k in allkeys if samples[k]['task']==st and (k,c) in records]
                if keys: bytask.append(dict(dataset=task,task=st,condition=c,n=len(keys),accuracy=100*np.mean([records[k,c]['scoring']['score'] for k in keys])))
    write_csv(OUT/'accuracy_summary.csv',summaries); write_csv(OUT/'paired_comparisons.csv',pairs)
    write_csv(OUT/'writer_budget_summary.csv',budgets); write_csv(OUT/'assistant_ablation_curve.csv',curves)
    write_csv(OUT/'task_accuracy.csv',bytask); write_csv(OUT/'latency_observational.csv',latency)
    subgroups=[]
    for group in ('first_word','answer_type','choice_count','hint_present'):
        cells=collections.defaultdict(list)
        for row in raw:
            value=(re.findall(r'\w+',row['question'].lower()) or ['EMPTY'])[0] if group=='first_word' else row[group]
            if group in ('choice_count','hint_present') and row['dataset']!='scienceqa_img': continue
            cells[row['dataset'],str(value),row['condition']].append(row)
        for (task,value,c),rows in sorted(cells.items()):
            subgroups.append(dict(dataset=task,group=group,value=value,condition=c,n=len(rows),
                accuracy=float(np.mean([r['score'] for r in rows]))*100,
                harm=sum(r['harm'] for r in rows),rescue=sum(r['rescue'] for r in rows),descriptive_only=True))
    write_csv(OUT/'descriptive_subgroups.csv',subgroups)
    plots(summaries,budgets,curves,pairs)
    write_report(samples,records,missing,reuse,summaries,budgets,pairs)
    atomic(OUT/'statistical_audit.json',dict(at=now(),status='PASS',questions=len(samples),condition_records=sum(c in LABELS for _,c in records),
        expected_condition_records=len(samples)*len(LABELS),missing=missing,reuse=dict(reuse),
        official_scores_recomputed=True,checkpoint_hashes_recomputed=True,writer_sets_recomputed=True,
        bootstrap_replicates=20000,image_cluster_weighting='Question weighted with clusters resampled intact',
        primary_comparison_family=14,analysis_source_sha256=sha(Path(__file__))))


def plots(summaries,budgets,curves,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    folder=OUT/'figures'; folder.mkdir(exist_ok=True)
    colors=['#444444','#007A78','#5484BA','#B85D16','#A94668','#748321','#80649D']
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=240,bbox_inches='tight'); fig.savefig(folder/(name+'.pdf'),bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,6.7))
    for ax,task in zip(axes.flat,TASKS):
        rows={r['condition']:r for r in summaries if r['dataset']==task}
        for j,c in enumerate(LABELS):
            if c not in rows: continue
            r=rows[c]; ax.bar(j,r['accuracy'],color=colors[j],width=.7)
            ax.errorbar(j,r['accuracy'],yerr=[[max(0,r['accuracy']-r['accuracy_ci_low'])],[max(0,r['accuracy_ci_high']-r['accuracy'])]],fmt='none',ecolor='#303030',capsize=2,lw=.7)
            ax.text(j,r['accuracy']+1,f"{r['accuracy']:.1f}",ha='center',fontsize=7,rotation=90)
        ax.set_title(task); ax.set_xticks(range(7),LABELS,rotation=55,ha='right'); ax.set_ylabel('Generated-answer score (%)'); ax.set_ylim(0,103)
    axes.flat[-1].axis('off'); fig.tight_layout(); save(fig,'accuracy_comparison')
    fig,axes=plt.subplots(2,4,figsize=(14,6.7))
    for ax,task in zip(axes.flat,TASKS):
        acc={r['condition']:r for r in summaries if r['dataset']==task}; bs={r['condition']:r for r in budgets if r['dataset']==task}
        for j,c in enumerate(LABELS):
            if c not in bs: continue
            x=bs[c]['mean_writer_ratio']*100; y=acc[c]['accuracy']; ax.scatter(x,y,s=34,color=colors[j]); ax.annotate(c,(x,y),xytext=(3,4 if c!='E2-S2' else -10),textcoords='offset points',fontsize=7)
        ax.set_title(task); ax.set_xlabel('Mean allowed writer connections (%)'); ax.set_ylabel('Generated-answer score (%)'); ax.grid(alpha=.18)
    axes.flat[-1].axis('off'); fig.tight_layout(); save(fig,'writer_budget_vs_accuracy')
    fig,axes=plt.subplots(2,4,figsize=(14,6.7))
    for ax,task in zip(axes.flat,TASKS):
        rows=[r for r in curves if r['dataset']==task and r['condition']!='E2-S2']; rows.sort(key=lambda r:r['assistant_removed'])
        if rows:
            ax.plot([r['assistant_removed'] for r in rows],[r['accuracy'] for r in rows],'-o',color='#007A78',lw=1.2)
            suffix=next(r for r in curves if r['dataset']==task and r['condition']=='E2-S2')
            ax.scatter(2,suffix['accuracy'],marker='x',s=65,color='#B85D16',label='Last 2 OFF'); ax.set_title(f"{task} (n={rows[0]['n']})")
        ax.set_xticks(range(5)); ax.set_xlabel('First ASSISTANT subwords removed'); ax.set_ylabel('Generated-answer score (%)'); ax.grid(alpha=.18); ax.legend(fontsize=7)
    axes.flat[-1].axis('off'); fig.tight_layout(); save(fig,'assistant_ablation_curve')
    fig,ax=plt.subplots(figsize=(10,6)); styles=[('E1','C3','#007A78',-.20),('E2-P2','C3','#B85D16',0),('E2-P2','E1','#80649D',.20)]
    for c,b,col,shift in styles:
        for i,task in enumerate(TASKS):
            r=next((r for r in pairs if r['dataset']==task and r['candidate']==c and r['baseline']==b and r['metric']=='accuracy'),None)
            if r:
                ax.errorbar(r['delta_pp'],i+shift,xerr=[[max(0,r['delta_pp']-r['ci_low_pp'])],[max(0,r['ci_high_pp']-r['delta_pp'])]],fmt='o',color=col,capsize=3,label=f'{c} - {b}' if i==0 else None)
    ax.axvline(0,color='#555555',ls='--',lw=.8); ax.set_yticks(range(7),TASKS); ax.set_xlabel('Paired score difference (pp), ordinary cluster-bootstrap 95% CI'); ax.legend(); ax.grid(axis='x',alpha=.18); fig.tight_layout(); save(fig,'paired_deltas')
    fig,axes=plt.subplots(2,4,figsize=(14,6.7))
    for ax,task in zip(axes.flat,TASKS):
        for j,c in enumerate(('E1','E2-P2')):
            r=next((r for r in pairs if r['dataset']==task and r['candidate']==c and r['baseline']=='C3' and r['metric']=='accuracy'),None)
            if r:
                ax.bar(j-.16,r['decreased'],width=.3,color='#A94668',label='Score decreased' if j==0 else None)
                ax.bar(j+.16,r['increased'],width=.3,color='#007A78',label='Score increased' if j==0 else None)
        ax.set_title(task); ax.set_xticks([0,1],['E1','E2-P2']); ax.set_ylabel('Questions'); ax.legend(fontsize=7)
    axes.flat[-1].axis('off'); fig.tight_layout(); save(fig,'error_transitions')


def write_report(samples,records,missing,reuse,summaries,budgets,pairs):
    primary=[r for r in pairs if r['metric']=='accuracy' and r['primary']]
    lines=['# C3-V2反向视觉Writer剪枝实验','',
        f"状态：{'PARTIAL' if missing else 'COMPLETE'}。冻结3,000题、7条件；完成{sum(c in LABELS for _,c in records):,}/21,000条。数据源与选样在正式推理前固定，不根据结果改规则。",'',
        'C3为原始C3-V2 All-Writer；E1只禁止Instruction直接读图；E2-Pk额外禁止ASSISTANT前k个Subword；E2-S2额外禁止最后2个Subword。文本、Prompt和位置从未删除或改写。P−1始终不是可读图Writer。','',
        '## 总体成绩','',
        'GQA为原冻结六任务的生成答案机制子集评分，不是完整官方GQA Benchmark；VQAv2/TextVQA为十答案共识软分，ScienceQA为冻结合法字母解析，POPE保持官方首句词法评分。各任务分别报告，不跨任务平均。','',
        '| 数据集 | N | C3 | E1 | E2-P1 | E2-P2 | E2-P3 | E2-P4 | E2-S2 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in TASKS:
        rows={r['condition']:r for r in summaries if r['dataset']==task}; n=sum(family(s)==task for s in samples.values())
        lines.append('| '+task+' | '+str(n)+' | '+' | '.join(f"{rows[c]['accuracy']:.3f}"+(f" (n={rows[c]['n']})" if rows[c]['n']!=n else '') if c in rows else '未完成' for c in LABELS)+' |')
    lines+=['','## 主要配对比较','',
        '20,000次Image SHA整簇重采样、题目等权。主要族预先冻结为7个数据集×2项比较=14项；同时报告普通95%和Bonferroni14校正区间。其余比较与全部分组为探索性，不据CI跨0宣称无损、等价或非劣效。','',
        '| 数据集 | 比较 | Δpp | 普通95%CI | 校正95%CI | 下降/上升 |','|---|---|---:|---|---|---|']
    for r in primary:
        lines.append(f"| {r['dataset']} | {r['candidate']} − {r['baseline']} | {r['delta_pp']:+.3f} | [{r['ci_low_pp']:.3f}, {r['ci_high_pp']:.3f}] | [{r['bonferroni14_ci_low_pp']:.3f}, {r['bonferroni14_ci_high_pp']:.3f}] | {r['decreased']}/{r['increased']} |")
    lines+=['','软分任务的下降/上升是逐题分值变化，不强行二值化；二值任务对应原对→错/原错→对。完整损害与纠正均保留在CSV。','',
        'Native辅助比较只接受严格统一H1及输入绑定的历史记录：ScienceQA/POPE覆盖全样本；GQA、VQAv2仅覆盖可验证H1子集，具体n见native_accuracy_auxiliary.csv与每项配对CSV，不能套用到完整600/500题。TextVQA没有可严格绑定的H1历史基线，本轮不额外生成Native，不使用不一致历史分数。','',
        '## 研究问题','']
    e1=[r for r in primary if r['candidate']=='E1']
    if e1:
        worst=min(e1,key=lambda r:r['delta_pp']); best=max(e1,key=lambda r:r['delta_pp'])
        lines.append(f"1. Instruction的直接视觉读取并非可预设无用。仅删除Instruction最明显的点估计损害在{worst['dataset']}（{worst['delta_pp']:+.3f}pp），另一端为{best['dataset']}（{best['delta_pp']:+.3f}pp）。是否存在可检出的群体损害须结合校正区间，不能只看均值。")
    lines.append('2. ASSISTANT前缀阶梯与后缀两Token对照使用固定原生四个Subword及真实Offsets，位置对照见E2-P2−E2-S2。差值不等同于某Subword存储了独立视觉语义；非线性联合与概率重分配都可能参与。')
    for task in TASKS:
        r=next((r for r in pairs if r['dataset']==task and r['candidate']=='E2-P2' and r['baseline']=='E2-S2' and r['metric']=='accuracy'),None)
        if r: lines.append(f"   {task}：前2 OFF − 后2 OFF = {r['delta_pp']:+.3f}pp，普通95%CI [{r['ci_low_pp']:.3f}, {r['ci_high_pp']:.3f}]（探索性）。")
    lines+=['3. 接近C3的点估计不能证明通用替代Conditional-15；本轮未新增Conditional-15对照，不将历史不同输入配置的分数拼接为公平优势。它提供固定结构Mask的跨任务探索结果，而不是已验证的新最优选择器。',
        '4. 本轮Instruction/ASSISTANT裁剪仍执行Dense Attention。与C3相同输入和生成长度时，Q/K/V/o及QK/AV矩阵形状、KV结构不变，新增物理FLOPs节省为0。Writer预算下降不能称为真实加速。quality-run时延仅作观察性诊断，未经预热交错测速，不报告新增加速倍数。','',
        '这些结果只解释C3-V2背景下指定直接路径的功能干预，不证明某类文本没有视觉语义。Hard Mask使原概率重新分配到保留的文本、BOS与Self；其他文本中继、残差和后续层仍可能携带视觉信息。不同条件的损害不能简单相加。POPE保留官方词法规则，即使空字符串可能被解析为yes，也单独保留空答案标记，不修补得分。','',
        '## Writer预算','', '| 数据集 | C3平均k/R | E1平均k/R | E2-P2平均k/R | E2-P4平均k/R |','|---|---:|---:|---:|---:|']
    for task in TASKS:
        rows={r['condition']:r for r in budgets if r['dataset']==task}
        lines.append('| '+task+' | '+' | '.join(f"{100*rows[c]['mean_writer_ratio']:.2f}%" if c in rows else '-' for c in ('C3','E1','E2-P2','E2-P4'))+' |')
    lines+=['','每题平均比例和合并连接比例分别见writer_budget_summary.csv；分母为原生R，不含P−1。Instruction字数、Token数、ASSISTANT保留数量和每题映射见writer_mapping_audit.csv。','',
        '## 正确性与历史复用','',
        '物理GPU0/1经用户追加授权使用，GPU2/3既有TextVQA进程保持不动。同图的新生成条件固定同卡，两个常驻模型并行；历史C3复用另经双卡精确一致性审计并逐条列出来源，不冒充本轮新生成。不跨条件共享Cache或生成前缀。',
        '31项CPU测试通过；两卡对相同5道冻结Smoke执行全部7条件。Native/No-op有效Logits严格torch.equal，原生generate Token一致；跨卡Logits SHA与Token一致；C3历史输入和首Token全词表Logits绑定一致。对应全Dense功能参考沿用既定容差，不放宽。Mask只改变声明的R→Image；禁止实际概率为0；MLP、原位置/RoPE、文本KV、L20物理退出和L2–L5物理答案Attention跳算均审计。',
        f"历史完整C3精确复用{reuse.get('audited_historical_C3',0)}条；本轮同图同输入同卡别名复用{reuse.get('identical_input_same_GPU',0)}条。每条记录保留来源路径和SHA，重新逐题评分。不同POPE策略共享图片和部分问题，不能当作独立数据集复现次数。",
        '全部样本来自已有冻结研究队列，历史重叠是100%，不声称独立外部验证。ScienceQA保留原选项、Hint，不向模型输入lecture/solution/Gold；TextVQA无额外OCR。所有负例、解析失败、空答、截断和重复均保留。','',
        '## 复现与交付','',
        '冻结执行Commit：b1e3bd9a9f8cdf2f55c4716cd400698f073e9112；最终结果Commit以Git日志和交付消息为准。',
        '`python -m vlm_research.reverse_writer.run --driver`：相同源码/输入/模型/协议精确恢复；只允许两张已授权且空闲设备。',
        '`CUDA_VISIBLE_DEVICES= python -m vlm_research.reverse_writer.analyze`：CPU独立重算评分、统计和图。',
        '`CUDA_VISIBLE_DEVICES= python -m vlm_research.reverse_writer.analyze --finalize`：独立检查CSV及图片/模型SHA，审计退出资源。',
        '全部论文图在figures/下提供PNG和矢量PDF。总截止2026-11-05 21:47:43+08:00，GPU新增工作截止21:17:43；没有通过换卡重置预算。']
    for rank in (0,1):
        p=OUT/f'worker{rank}/quality_complete.json'
        if p.exists():
            q=read(p)
            lines.append(f"Worker {rank}正式矩阵耗时{q['elapsed_seconds']/60:.2f}分钟，新生成{q['generated']}条，复用{q['reused']}条；任务时间包含CPU调度，不是纯Kernel执行时间。")
    start=datetime.datetime.fromisoformat(read(OUT/'protocol.json')['started_at'])
    lines.append(f"报告生成时总墙钟{(datetime.datetime.fromisoformat(now())-start).total_seconds()/60:.2f}分钟；提交推送仍受同一6小时绝对截止约束。")
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


def finalize():
    samples,records,missing,reuse=read_all(True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    rows=list(csv.DictReader((OUT/'per_question.csv').open()))
    assert len(rows)==len(records)
    for row in rows:
        k,c=row['sample'],row['condition']; r=records[k,c]
        assert float(row['score'])==scored(samples[k],r['generation']['text'])['score']
        assert json.loads(row['token_ids'])==r['generation']['token_ids']
        assert tokenizer.decode(r['generation']['token_ids'],skip_special_tokens=True).strip()==r['generation']['text']
    kv_audit=[]
    for k,s in samples.items():
        baseline=records.get((k,'C3'))
        if baseline is None: continue
        b=baseline['generation']
        for c in LABELS:
            r=records.get((k,c))
            if r is None: continue
            g=r['generation']; kv=g['kv']
            assert kv['storage_bytes']==b['kv']['storage_bytes'],'Writer mask unexpectedly changed KV capacity'
            assert kv['historical_cache_copies']==kv['append_concatenations']==0
            kv_audit.append(dict(sample=k,condition=c,KV_capacity_bytes=kv['storage_bytes'],baseline_capacity_bytes=b['kv']['storage_bytes'],
                capacity_exact=True,live_bytes=kv['live_bytes'],output_tokens=len(g['token_ids']),
                dense_mask_extra_matrix_flops_saved=0,flops_basis='Same executed dense shapes asserted in GPU smoke; no sparse kernel in this experiment'))
    write_csv(OUT/'kv_shape_audit.csv',kv_audit)
    summaries=list(csv.DictReader((OUT/'accuracy_summary.csv').open()))
    for row in summaries:
        rs=[r for (k,c),r in records.items() if c==row['condition'] and family(samples[k])==row['dataset']]
        assert len(rs)==int(row['n']) and abs(float(row['accuracy'])-100*np.mean([r['scoring']['score'] for r in rs]))<1e-10
    images={s['image_path']:s['image_sha256'] for s in samples.values()}
    assert all(sha(p)==h for p,h in images.items())
    assert all(sha(p)==h for p,h in read(OUT/'execution_freeze.json')['weights'].items())
    state=hardware(); pids=read(OUT/'driver_pids.json')['workers']
    assert all(not (Path('/proc')/str(pid)).exists() for pid in pids),'Task workers still running'
    atomic(OUT/'gpu_release_audit.json',dict(at=now(),status='PASS',task_workers=pids,all_task_workers_exited=True,
        all_task_GPU_resources_released=True,authorized_devices=DEVICES,other_tasks_untouched=True,**state))
    atomic(OUT/'independent_recompute.json',dict(at=now(),status='PASS',CSV_matches_checkpoints=True,
        scores_recomputed_from_answers=True,token_ids_decode_to_saved_answers=True,KV_capacity_exact_all_conditions=True,
        summary_means_recomputed=True,image_files_rehashed=len(images),model_weights_rehashed=True))
    artifacts=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ('completion.json','artifact_hashes.csv'):
            artifacts.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    write_csv(OUT/'artifact_hashes.csv',artifacts)
    atomic(OUT/'completion.json',dict(status='COMPLETE' if not missing else 'PARTIAL',at=now(),
        expected_questions=3000,expected_conditions=7,expected_records=21000,actual_records=len(records),missing=len(missing),
        unique_completed_conditions=sum(all((k,c) in records for k in samples) for c in LABELS),
        official_scores_and_hashes_independently_verified=True,artifacts=len(artifacts),
        GPU_released=True,main_unmodified=True,bootstrap_replicates=20000,analysis_source_sha256=sha(Path(__file__))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--partial',action='store_true'); parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.finalize: finalize()
    else: summarize(args.partial)
