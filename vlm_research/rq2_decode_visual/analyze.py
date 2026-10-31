"""Exploratory image-paired results, with content and EOS changes kept separate."""
import csv
import fcntl
from itertools import zip_longest
import json
import subprocess
import time

import numpy as np
import torch

from benchmarks.dataset_registry import digest, freeze_json, sha256
from knockout_probe import tensor_hash, ROW_ATOL
from paper_rq1.scorers import gqa_prediction
from paper_rq1_early_joint_attention.statistics import bootstrap
from .run import (OUT, ROOT, LABELS, TASKS, SEED, GPU, BLOCKED, BASE, read, verify,
                  samples, histories, length_group, checkpoint, b_logits)


def write_csv(path,rows,fields=None):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def differences(b,c):
    bt,ct=b['token_ids'],c['token_ids']
    first=next((i for i,(a,z) in enumerate(zip_longest(bt,ct)) if a!=z),None)
    return dict(first_token_equal=bt[0]==ct[0],token_sequence_changed=bt!=ct,
        text_changed=b['text']!=c['text'],scored_answer_changed=b['prediction']!=c['prediction'],
        content_tokens_changed=[t for t in bt if t!=2]!=[t for t in ct if t!=2],
        continuation_tokens_changed=[t for t in bt[1:] if t!=2]!=[t for t in ct[1:] if t!=2],
        first_difference_token_ordinal='' if first is None else first+1,
        first_difference_B='' if first is None or first>=len(bt) else bt[first],
        first_difference_C='' if first is None or first>=len(ct) else ct[first],
        first_difference_involves_eos=first is not None and (bt[first:first+1]==[2] or ct[first:first+1]==[2]),
        eos_flag_changed=b['eos']!=c['eos'],length_changed=b['length']!=c['length'],
        token_change_without_text_change=bt!=ct and b['text']==c['text'],
        b_correct_c_wrong=int(b['score']==1 and c['score']==0),
        b_wrong_c_correct=int(b['score']==0 and c['score']==1))


def collect():
    c=verify();binding=digest(c);completion=read(OUT/'completion.json')
    assert completion['status']=='complete' and completion['binding']==binding and completion['new_generations']==240
    assert completion['hooks_clean'] and completion['parameters_unchanged']
    assert len(list((OUT/'checkpoints').glob('*.json')))==240
    rows=[];allg=[];attention_rows=0;decode_queries=0
    for s in samples():
        h=histories()[s['question_id']]
        assert sha256(ROOT/h['full_path'])==h['full_sha256'] and sha256(ROOT/h['b_path'])==h['b_sha256']
        assert h['sample_sha256']==digest(s)
        rec=checkpoint(s,binding);assert rec and rec['hooks_clean'] and rec['inputs_unchanged'] and rec['parameters_unchanged']
        g=dict(Full=h['Full'],B=h['B'],C=rec['generation']);new=g['C']
        with np.load(OUT/rec['logits_file']) as z:first=torch.from_numpy(z['first_logits'].copy())
        assert torch.equal(first,b_logits(s)) and tensor_hash(first)==rec['first_logits_sha256']
        assert g['B']['token_ids'][0]==new['token_ids'][0]==int(first.argmax())
        assert new['masked_layers']==BLOCKED and new['own_prefix'] and new['own_cache']
        assert new['original_p_minus_one']==s['input_mapping']['p_minus_one']
        visual=[i for i,t in enumerate(s['input_mapping']['prompt_ids']) if t==32000]
        assert len(visual)==576 and new['image_key_indices']==visual
        keysha=tensor_hash(torch.tensor(visual,dtype=torch.long))
        assert len(new['decode_mask_audit'])==new['decode_steps']==len(new['token_ids'])-1
        for step,a in enumerate(new['decode_mask_audit'],1):
            assert a['step']==step and a['consumed_token']==new['token_ids'][step-1] and a['predicted_token']==new['token_ids'][step]
            assert [r['layer'] for r in a['masks']]==list(range(32))
            assert [r['layer'] for r in a['probabilities']]==list(range(32))
            query=new['original_p_minus_one']+step
            for mask,p in zip(a['masks'],a['probabilities']):
                assert mask['absolute_query']==mask['cache_before']==query and mask['cache_after']==mask['key_length']==query+1
                assert mask['local_query']==0 and mask['image_keys_sha256']==keysha and mask['image_key_count']==576
                assert mask['non_image_mask_unchanged'] and mask['applied']==p['blocked']==(p['layer'] in BLOCKED)
                assert p['image_probability_max']==0. if p['blocked'] else p['image_mass']>0
                assert p['row_sum_max_error']<=ROW_ATOL
                attention_rows+=1
            decode_queries+=1
        assert decode_queries>=0
        for gen in g.values():
            assert gen['score']==int(gqa_prediction(gen['text'])==s['answer']) and gen['prediction']==gqa_prediction(gen['text'])
            assert len(gen['token_ids'])==gen['length']<=32 and 2 not in gen['token_ids'][:-1]
            assert gen['eos']==(gen['token_ids'][-1]==2)
            assert gen['truncated']==(gen['length']==32 and not gen['eos'])
        row=dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],gold=s['answer'],
            full_non_eos_tokens=sum(t!=2 for t in g['Full']['token_ids']),group=length_group(g['Full']['token_ids']))
        for key,gen in g.items():
            for name in ('text','prediction','score','eos','truncated','length'):
                row[key+'_'+name]=gen[name]
            row[key+'_token_ids']=json.dumps(gen['token_ids'])
            row[key+'_empty']=not bool(gen['text'].strip())
        row.update(differences(g['B'],g['C']),sample_sha256=digest(s),source_commit=BASE,
            full_source=h['full_path'],b_source=h['b_path'],model_revision=c['model_revision'],
            protocol_sha256=sha256(OUT/'protocol.json'),checkpoint_sha256=sha256(OUT/f"checkpoints/{s['task']}__{s['question_id']}.json"),
            abnormal=bool(new['truncated'] or not new['text'].strip()),first_logits_exact_B=True)
        rows.append(row);allg.append(g)
    assert len(rows)==240 and len({r['image_sha256'] for r in rows})==240 and all(r['first_token_equal'] for r in rows)
    assert decode_queries>0
    write_csv(OUT/'per_question.csv',rows)
    write_csv(OUT/'decode_difference_cases.csv',[r for r in rows if r['token_sequence_changed']],fields=list(rows[0]))
    return rows,allg,dict(actual_decode_queries=decode_queries,actual_attention_rows=attention_rows,
        every_blocked_probability_exact_zero=True,retained_layers_positive_image_mass=True,
        all_key_query_cache_mappings_exact=True)


def statistics(rows,generations):
    summaries=[];paired=[];changes=[];tasks=[]
    for group in ['All','One-token','Multi-token','Zero-token']:
        ix=[i for i,r in enumerate(rows) if group=='All' or r['group']==group]
        n=len(ix)
        if not n:
            changes.append(dict(group=group,n=0,text_changed=0,token_changed=0,text_change_pct='',token_change_pct=''))
            continue
        ids=[rows[i]['image_sha256'] for i in ix]
        scores=np.array([[generations[i][k]['score'] for k in ['Full','B','C']] for i in ix],dtype=float)
        stats=bootstrap(100*np.column_stack([scores,scores[:,2]-scores[:,1],scores[:,1]-scores[:,0],scores[:,2]-scores[:,0]]),ids,[1]*6,repetitions=200000,seed=SEED)
        for j,k in enumerate(['Full','B','C']):
            gs=[generations[i][k] for i in ix]
            summaries.append(dict(group=group,condition=k,label=LABELS[k],n=n,correct=int(scores[:,j].sum()),accuracy_pct=stats[j]['mean'],
                ci95_low=stats[j]['ci95_low'],ci95_high=stats[j]['ci95_high'],
                delta_full_pp=float(100*(scores[:,j]-scores[:,0]).mean()),eos=sum(g['eos'] for g in gs),
                truncated=sum(g['truncated'] for g in gs),empty=sum(not g['text'].strip() for g in gs),mean_length=float(np.mean([g['length'] for g in gs]))))
        for j,(a,b) in enumerate([('C','B'),('B','Full'),('C','Full')],3):
            st=stats[j]
            paired.append(dict(group=group,condition=a,reference=b,n=n,images=st['clusters'],delta_pp=st['mean'],
                ci95_low_pp=st['ci95_low'],ci95_high_pp=st['ci95_high'],role='exploratory; no NI declaration',
                reference_correct_to_wrong=sum(generations[i][b]['score']==1 and generations[i][a]['score']==0 for i in ix),
                reference_wrong_to_correct=sum(generations[i][b]['score']==0 and generations[i][a]['score']==1 for i in ix),
                repetitions=st['repetitions'],degenerate_bootstrap=st['degenerate_bootstrap']))
        text=sum(rows[i]['text_changed'] for i in ix);token=sum(rows[i]['token_sequence_changed'] for i in ix)
        changes.append(dict(group=group,n=n,text_changed=text,token_changed=token,text_change_pct=100*text/n,token_change_pct=100*token/n))
    for task in TASKS:
        subset=[g for r,g in zip(rows,generations) if r['task']==task]
        for key in LABELS:tasks.append(dict(task=task,condition=key,n=len(subset),accuracy_pct=100*np.mean([g[key]['score'] for g in subset])))
    for name,table in [('accuracy_summary.csv',summaries),('paired_comparisons.csv',paired),('answer_change_rates.csv',changes),('task_accuracy.csv',tasks)]:
        write_csv(OUT/name,table)
    return summaries,paired,changes


def present(rows,summary,paired,changes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.2))
    overall=[r for r in summary if r['group']=='All']
    axes[0].bar(range(3),[r['accuracy_pct'] for r in overall],color=['#626262','#4f7ca5','#26886b'],width=.6)
    for x,r in enumerate(overall):axes[0].text(x,r['accuracy_pct']+1,f"{r['accuracy_pct']:.2f}",ha='center',fontsize=8)
    axes[0].set_xticks(range(3),['Full','P-1-only','P-1+Decode'],fontsize=8)
    axes[0].set_ylim(0,105);axes[0].set_ylabel('Complete-answer Accuracy (%)')
    for x,group in enumerate(['One-token','Multi-token']):
        r=next(v for v in changes if v['group']==group)
        value=r['text_change_pct'] if r['n'] else 0
        axes[1].bar(x,value,color=['#4f7ca5','#b46639'][x],width=.55)
        axes[1].text(x,value+2,f"{value:.2f}%" if r['n'] else 'N/A',ha='center',fontsize=8)
    axes[1].set_xticks([0,1],[f"{g}\n(n={next(v['n'] for v in changes if v['group']==g)})" for g in ['One-token','Multi-token']],fontsize=8)
    axes[1].set_ylim(0,105);axes[1].set_ylabel('Answer-text change: C vs B (%)')
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.suptitle('Decode Visual Necessity',fontsize=12,fontweight='bold')
    fig.subplots_adjust(left=.09,right=.98,bottom=.21,top=.82,wspace=.4)
    for ext in ('png','pdf'):fig.savefig(OUT/f'decode_visual_necessity.{ext}',dpi=240)
    plt.close(fig)
    primary=next(r for r in paired if r['group']=='All' and r['condition']=='C' and r['reference']=='B')
    lines=['# RQ2.4：Decode视觉Attention轻量机制检查','',
        '**状态：完成。固定240题、六类各40题、240个Image SHA256簇；只新增240次C生成。**','',
        '## 主要结果','',
        f"C−B完整答案Accuracy变化{primary['delta_pp']:+.3f}pp，按Image SHA256整簇配对Bootstrap200,000次的探索性95%CI [{primary['ci95_low_pp']:+.3f}, {primary['ci95_high_pp']:+.3f}]pp；B正确→C错误{primary['reference_correct_to_wrong']}题，B错误→C正确{primary['reference_wrong_to_correct']}题。本轮不作确认性非劣效或等价宣称。",'',
        '| 分组（历史Full非EOS长度） | n | Full | P−1-only B | P−1+Decode C |','|---|---:|---:|---:|---:|']
    for group in ['All','One-token','Multi-token','Zero-token']:
        rr=[r for r in summary if r['group']==group]
        if rr:lines.append('| '+group+f" | {rr[0]['n']} | "+' | '.join(f"{r['accuracy_pct']:.3f}%" for r in rr)+' |')
    lines+=['','## 内容、后续Token与EOS','',
        f"B/C首Token在240题上严格一致，首Token全词表Logits逐题精确一致。完整Token序列改变{sum(r['token_sequence_changed'] for r in rows)}题，去首尾空白后的解码文本改变{sum(r['text_changed'] for r in rows)}题，官方归一化答案改变{sum(r['scored_answer_changed'] for r in rows)}题。", 
        f"第二个及之后非EOS Token序列改变{sum(r['continuation_tokens_changed'] for r in rows)}题；首次分歧涉及EOS {sum(r['first_difference_involves_eos'] for r in rows)}题；生成长度改变{sum(r['length_changed'] for r in rows)}题；EOS到达标记改变{sum(r['eos_flag_changed'] for r in rows)}题。Token序列变化但解码文本不变{sum(r['token_change_without_text_change'] for r in rows)}题。首次分歧涉及EOS可能同时伴随新增/减少内容，不能统称为仅EOS变化。",'',
        '| Full长度分组 | n | 解码文本改变 | 完整Token序列改变 | B对→C错 | B错→C对 |','|---|---:|---:|---:|---:|---:|']
    for change in changes:
        sub=[r for r in rows if change['group']=='All' or r['group']==change['group']]
        lines.append(f"| {change['group']} | {len(sub)} | {change['text_changed']} | {change['token_changed']} | {sum(r['b_correct_c_wrong'] for r in sub)} | {sum(r['b_wrong_c_correct'] for r in sub)} |")
    cases=[r for r in rows if r['token_sequence_changed']]
    lines+=['','### 全部真实差异案例','']
    if cases:
        lines+=['| Task / QID | Full长度组 | B文本 | C文本 | 首次不同Token序号（从1开始） | 首次涉及EOS |','|---|---|---|---|---:|---|']
        for r in cases:
            texts=[r[k+'_text'].replace('|','\\|').replace('\n',' ') for k in ['B','C']]
            lines.append(f"| {r['task']} / {r['question_id']} | {r['group']} | {texts[0]} | {texts[1]} | {r['first_difference_token_ordinal']} | {r['first_difference_involves_eos']} |")
    else:lines.append('没有B/C完整Token序列差异；不存在可列出的真实差异案例。不能据此推广到长答案。')
    lines+=['','## 验证、协议及解释边界','',
        '样本来自上一轮2,656题图像隔离集合，seed20261101，按任务固定顺序及输入ID/Image SHA256哈希选40个未使用图像簇。选择完全不读取模型得分或答案长度；长度分组仅在选样后按历史Full输出冻结。generation_length_summary.csv保存每任务分布。两组均保留全部样本，不为扩大Decode效果重选多Token题。',
        'Full与B精确复用acc137dc历史结果。模型revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager/Transformers4.49.0，Greedy/max32/真实KV Cache。只有当前真实Query的Image Key在30层Softmax前设−inf，L14/L19保留；MLP、BOS、Prompt、自身、位置和其他Prefill Query正常；没有物理Token/KV删除或实际加速。',
        'CPU回归与GPU1真实Smoke通过。No-op所有Prefill及Decode Logits、原生生成和历史Full精确一致；B/C全部Prefill Logits精确相同。其他Prefill Query Attention/Hidden与Full精确相同。每题每一步均直接读取实际Softmax概率，验证30层Image概率严格0、保留两层Image Mass为正、动态绝对Query/576个Image Key/逐层KV长度及非Image Mask不变；不是仅统计Hook调用。原ROW_ATOL=0.002不变。Hook/输入/权重版本、有限输出与EOS均校验，失败不会静默排除。',
        '若变化很少，只能说明当前短答案与实际多Token覆盖下额外Decode限制影响较小；若文本或EOS改变，则后续视觉读取参与了答案延续或结束决定。总体准确率相同不代表逐Token无损。所有结论局限本模型、本固定样本及P−1已受限的前提；不等于所有Prefill Query或长答案都不需要视觉Attention。',
        '图：左为实测Accuracy；右为相对B的去首尾空白解码文本变化率，分组由历史Full非EOS长度确定。原始Token/EOS分歧另列CSV，不混为文本内容改变。图中不放长说明。','']
    complete=read(OUT/'completion.json');smoke=read(OUT/'smoke.json')
    lines += ['## 耗时与输出','',f"正式单GPU任务{complete['seconds']/60:.2f}分钟（含加载），Smoke {smoke['seconds']/60:.2f}分钟；本轮只有240次正式新生成。逐题CSV、差异案例、组别/任务统计、协议、源码/模型/历史绑定、真实Decode概率审计与PNG/PDF均保存。完成后停止，不追加实验。",'']
    (OUT/'report_zh.md').write_text('\n'.join(lines))


def main():
    lock=(OUT/'aggregation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows,generations,audit=collect();summary,paired,changes=statistics(rows,generations)
    present(rows,summary,paired,changes)
    table=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    completion=read(OUT/'completion.json');smoke=read(OUT/'smoke.json')
    own_pids={completion['process_id'],smoke['process_id']}
    assert not any(int(line.split(',')[1].strip()) in own_pids for line in apps.splitlines() if line.strip())
    release=dict(checked_epoch=time.time(),gpu_index=1,gpu_uuid=GPU,our_model_processes_released=True,
        our_model_pids=sorted(own_pids),gpu_snapshot=table,all_compute_processes=apps,other_processes_never_signalled=True)
    freeze_json(OUT/'gpu_release.json',release)
    freeze_json(OUT/'validation.json',dict(status='complete',questions=240,new_generations=240,reused_Full=240,reused_B=240,
        cpu=read(OUT/'cpu_validation.json')['tests'],gpu_smoke_passed=True,all_first_logits_exact_B=True,
        all_first_tokens_exact_B=True,no_missing_or_duplicate_rows=True,all_hooks_clean=True,
        all_inputs_and_parameter_versions_unchanged=True,no_nan_inf_oom=True,**audit,
        eos={k:sum(g[k]['eos'] for g in generations) for k in LABELS},
        truncated={k:sum(g[k]['truncated'] for g in generations) for k in LABELS},
        empty={k:sum(not g[k]['text'].strip() for g in generations) for k in LABELS},
        resource_release=release,protocol_sha256=sha256(OUT/'protocol.json')))
    report=OUT/'report_zh.md';report.write_text(report.read_text()+'\nGPU1本轮模型进程均已退出；资源释放快照见gpu_release.json。\n')
    print('Complete:240 new generations; every Decode mask audited; GPU1 released; experiment stopped',flush=True)
    hashes={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json' and p.suffix not in ('.lock','.tmp')}
    freeze_json(OUT/'artifact_hashes.json',hashes)


if __name__=='__main__':main()
