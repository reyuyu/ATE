"""Single final merge, question-weighted image-cluster intervals and paper figures."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1_early_joint_attention.statistics import bootstrap
from paper_rq1_single_layer_gqa6.engine import metric_rows
from .engine import CONDITIONS, LABELS
from .run import (OUT, ROOT, SEED, CONTRASTS, read, samples, full_path,
                  source_path, historical_logits, verify, old)

KEYS = ['Full',*CONDITIONS]


def write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def plot(summary, tasks, comparisons):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    colors=['#555555','#be4848','#c68b23','#238b69','#5579ab']
    x=np.arange(5)
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    deltas=np.array([s['delta_pp'] for s in summary])
    intervals=[(0.,0.)]+[(r['family95_low_pp'],r['family95_high_pp']) for r in comparisons[:4]]
    for i,(delta,(low,high)) in enumerate(zip(deltas,intervals)):
        ax.errorbar(i,delta,yerr=np.array([[delta-low],[high-delta]]),fmt='o',
            color=colors[i],markersize=7 if i==3 else 5,capsize=4,linewidth=1.6)
    ax.axhline(0,color='#777777',ls='--',lw=.8)
    ax.set_xticks(x,[LABELS[k].replace(' Image ',' Image\n') if k=='A' else LABELS[k] for k in KEYS])
    ax.set_ylabel('Accuracy change vs Full (pp)')
    ax.grid(axis='y',alpha=.18)
    ax.set_xlim(-.5,4.5)
    ax.margins(y=.17)
    fig.suptitle('Visual Attention Sufficiency',y=.98,fontweight='bold',fontsize=13)
    fig.text(.5,.90,'Original Prefill P-1 only; Decode remains native',ha='center',fontsize=9)
    fig.text(.07,.035,'240 questions / 235 image clusters; Bonferroni-6 paired bootstrap 95% intervals.\nExploratory reuse of the layer-discovery cohort; hard mask, not physical acceleration.',fontsize=7)
    fig.subplots_adjust(left=.12,right=.97,top=.82,bottom=.22)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'visual_attention_sufficiency.{ext}',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.4))
    names=list(old.TASKS)
    for j,key in enumerate(KEYS):
        values=[next(r['accuracy_pct'] for r in tasks if r['task']==t and r['condition']==key) for t in names]
        ax.bar(np.arange(6)+(j-2)*.145,values,width=.14,color=colors[j],label=LABELS[key])
    ax.set_xticks(np.arange(6),names)
    ax.set_ylabel('Complete-answer Accuracy (%)')
    ax.set_ylim(0,105);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Visual Attention Sufficiency by GQA Task',y=.98,fontsize=12,fontweight='bold')
    fig.legend(loc='upper center',bbox_to_anchor=(.53,.92),ncol=3,frameon=False,fontsize=8)
    fig.text(.08,.035,'40 questions per task; descriptive breakdown, not independent confirmatory tests.',fontsize=7)
    fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.74)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'task_visual_attention_sufficiency.{ext}',dpi=220)
    plt.close(fig)


def main():
    c=verify();binding=digest(c);ss=samples();assignments=read(OUT/'assignments.json')
    completion=[read(OUT/f'shard{k}/completion.json') for k in range(3)]
    assert all(r['status']=='complete' and r['binding']==binding and r['hooks_clean'] and r['parameter_versions_unchanged'] for r in completion)
    assert sum(r['questions'] for r in completion)==240 and sum(r['generations'] for r in completion)==960
    existing=list(OUT.glob('shard*/checkpoints/*/*.json'))
    assert len(existing)==960
    histories={r['question_id']:r for r in read(OUT/'history_bindings.json')['records']}
    rows=[];scores=[];transition=[];record_hashes={}
    nonadd={k:[] for k in CONDITIONS}
    for s in ss:
        history=histories[s['question_id']]
        for pathkey,hashkey in [('full_file','full_sha256'),('source_file','source_sha256')]:
            assert sha256(ROOT/history[pathkey])==history[hashkey]
        full=read(full_path(s)); fullgen=full['generation'];native=historical_logits(s)
        assert tensor_equivalent(native,history['first_logits_sha256'])
        oldrows=read(source_path(s))['result']['metrics']
        sample_scores=[];harms=[];fixes=[]
        for key in KEYS:
            if key=='Full':
                gen=fullgen
                metric=metric_rows(native,native[None])[0];metric['logp_loss']=0.
                seconds=0.;path=full_path(s)
            else:
                path=OUT/f"shard{assignments[s['question_id']]}/checkpoints/{s['task']}__{s['question_id']}/{key}.json"
                value=read(path)
                assert value['binding']==binding and value['sample_sha256']==digest(s) and value['condition']==key
                gen=value['generation'];metric=value['first_token'];seconds=value['seconds']
                assert digest(gen)==value['result_sha256'] and digest(metric)==value['metric_sha256']
                assert value['hooks_clean'] and value['inputs_unchanged'] and value['parameter_versions_unchanged']
                assert gen['masked_layers']==CONDITIONS[key] and gen['decode_native']
                assert sha256(OUT/value['logits_file'])==value['logits_sha256']
                with np.load(OUT/value['logits_file']) as saved:
                    first=torch.from_numpy(saved['first_logits'].copy())
                recomputed=metric_rows(native,first[None])[0]
                recomputed['logp_loss']=-recomputed['delta_logp']
                assert metric==recomputed
                assert int(first.argmax())==gen['token_ids'][0]
                record_hashes[str(path.relative_to(OUT))]=sha256(path)
                record_hashes[value['logits_file']]=value['logits_sha256']
            assert len(gen['token_ids'])==gen['length'] and gen['length']<=32
            assert gen['eos']==(gen['token_ids'][-1]==2)
            assert gen['truncated']==(gen['length']==32 and not gen['eos'])
            assert (2 not in gen['token_ids'][:-1])
            assert gen['score']==int(gen['prediction']==s['answer'])
            harm=int(fullgen['score']==1 and gen['score']==0)
            fix=int(fullgen['score']==0 and gen['score']==1)
            if key!='Full' and harm:harms.append(key)
            if key!='Full' and fix:fixes.append(key)
            additive=sum(r['logp_loss'] for r in oldrows if r['source']=='Image' and r['layer'] in CONDITIONS.get(key,[]))
            residual=metric['logp_loss']-additive
            if key!='Full':nonadd[key].append([metric['logp_loss'],additive,residual])
            rows.append(dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],
                condition=key,label=LABELS[key],gold=s['answer'],full_prediction=fullgen['prediction'],prediction=gen['prediction'],
                text=gen['text'],token_ids=json.dumps(gen['token_ids']),score=gen['score'],full_score=fullgen['score'],
                delta_score=gen['score']-fullgen['score'],correct_to_wrong=harm,wrong_to_correct=fix,
                eos=gen['eos'],truncated=gen['truncated'],empty=not bool(gen['text'].strip()),length=gen['length'],
                full_token_id=metric['full_token_id'],new_token_id=metric['ablated_token_id'],
                top1_flip=metric['top1_flip'],logp_loss=metric['logp_loss'],js=metric['js'],
                single_layer_additive_logp_loss=additive,joint_minus_additive_logp_loss=residual,seconds=seconds))
            sample_scores.append(gen['score'])
        scores.append(sample_scores)
        transition.append(dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],
                               harmed_conditions=harms,fixed_conditions=fixes))
    scores=np.array(scores,dtype=float)
    index={key:i for i,key in enumerate(KEYS)}
    contrasts=np.stack([100*(scores[:,index[a]]-scores[:,index[b]]) for a,b in CONTRASTS],axis=-1)
    stats=bootstrap(np.concatenate([contrasts,100*scores],axis=1),[s['image_sha256'] for s in ss],
                    [6]*6+[1]*5,repetitions=c['statistics']['repetitions'],seed=SEED)
    paired=[]
    for (a,b),st in zip(CONTRASTS,stats[:6]):
        paired.append(dict(condition=a,reference=b,label=f'{LABELS[a]} minus {LABELS[b]}',n=240,images=235,
            delta_pp=st['mean'],ci95_low_pp=st['ci95_low'],ci95_high_pp=st['ci95_high'],
            family95_low_pp=st['family95_low'],family95_high_pp=st['family95_high'],family=6,
            correct_to_wrong=int(((scores[:,index[b]]==1)&(scores[:,index[a]]==0)).sum()),
            wrong_to_correct=int(((scores[:,index[b]]==0)&(scores[:,index[a]]==1)).sum()),
            repetitions=st['repetitions'],degenerate_bootstrap=st['degenerate_bootstrap']))
    summary=[]
    for i,key in enumerate(KEYS):
        rr=[r for r in rows if r['condition']==key]
        summary.append(dict(condition=key,label=LABELS[key],n=240,correct=int(scores[:,i].sum()),accuracy_pct=100*float(scores[:,i].mean()),
            accuracy_ci95_low=stats[6+i]['ci95_low'],accuracy_ci95_high=stats[6+i]['ci95_high'],
            delta_pp=100*float((scores[:,i]-scores[:,0]).mean()),
            correct_to_wrong=sum(r['correct_to_wrong'] for r in rr),wrong_to_correct=sum(r['wrong_to_correct'] for r in rr),
            mean_logp_loss=float(np.mean([r['logp_loss'] for r in rr])),top1_flips=sum(r['top1_flip'] for r in rr),
            top1_flip_pct=100*float(np.mean([r['top1_flip'] for r in rr])),mean_js=float(np.mean([r['js'] for r in rr])),
            eos=sum(r['eos'] for r in rr),truncated=sum(r['truncated'] for r in rr),empty=sum(r['empty'] for r in rr),
            mean_length=float(np.mean([r['length'] for r in rr]))))
    tasks=[]
    for task in old.TASKS:
        for key in KEYS:
            rr=[r for r in rows if r['task']==task and r['condition']==key]
            tasks.append(dict(task=task,condition=key,n=len(rr),accuracy_pct=100*float(np.mean([r['score'] for r in rr])),
                delta_pp=100*float(np.mean([r['delta_score'] for r in rr])),
                correct_to_wrong=sum(r['correct_to_wrong'] for r in rr),wrong_to_correct=sum(r['wrong_to_correct'] for r in rr),
                mean_logp_loss=float(np.mean([r['logp_loss'] for r in rr]))))
    write_csv(OUT/'per_question.csv',rows)
    write_csv(OUT/'accuracy_summary.csv',summary)
    write_csv(OUT/'task_accuracy.csv',tasks)
    write_csv(OUT/'paired_comparisons.csv',paired)
    write_csv(OUT/'nonadditivity.csv',[dict(condition=k,mean_joint_loss=float(np.mean(v,axis=0)[0]),
        mean_sum_single_layer_loss=float(np.mean(v,axis=0)[1]),mean_joint_minus_sum=float(np.mean(v,axis=0)[2]),
        median_joint_minus_sum=float(np.median(np.array(v)[:,2]))) for k,v in nonadd.items()])
    freeze_json(OUT/'error_transitions.json',transition)
    plot(summary,tasks,paired)
    wall=max(r['finished_epoch'] for r in completion)-min(r['started_epoch'] for r in completion)
    validation=dict(status='complete',questions=240,image_clusters=235,new_generations=960,full_reused=240,
        unique_checkpoint_count=len(existing),all_checkpoint_and_logits_hashes_verified=True,
        first_token_metrics_recomputed_exact=True,history_bindings_verified=True,
        cpu_tests=read(OUT/'cpu_validation.json')['tests'],all_gpu_smokes_passed=True,
        noop_history_and_singleL14_exact=True,other_queries_exact=True,mask_removed_before_decode=True,
        no_nan_inf_oom=True,all_hooks_clean=True,parameter_versions_unchanged=True,
        formal_wall_seconds=wall,formal_gpu_seconds=sum(r['seconds'] for r in completion),
        smoke_gpu_seconds=sum(read(OUT/f'shard{k}/smoke.json')['seconds'] for k in range(3)),
        worker_seconds={str(k+1):r['seconds'] for k,r in enumerate(completion)},
        abnormal_output_counts={r['condition']:dict(eos=r['eos'],truncated=r['truncated'],empty=r['empty']) for r in summary},
        source=c['source'],protocol_sha256=sha256(OUT/'protocol.json'))
    freeze_json(OUT/'validation.json',validation)
    lines=['# RQ2 Experiment 2.2：两层直接视觉 Attention 充分性（探索性）','',
        '240题、235个图像SHA256簇；六类各40题。原始Prompt、模型revision、FP16/eager/Transformers4.49.0及官方GQA完整答案评分不变。',
        '只在原Prefill P−1到576个Image Key的边上施加softmax前−inf；其余Query、MLP、位置和Decode全部原生。四个新条件共960次完整生成，Full完整答案和首Token分布经绑定校验复用。','',
        '## 完整答案结果','',
        '| 条件 | 正确/240 | Accuracy | ΔFull pp | 原对→错 | 原错→对 | 首Token翻转 | 平均LogP Loss nat |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['label']} | {r['correct']} | {r['accuracy_pct']:.3f}% | {r['delta_pp']:+.3f} | {r['correct_to_wrong']} | {r['wrong_to_correct']} | {r['top1_flips']} | {r['mean_logp_loss']:.6f} |")
    lines+=['','## 配对比较','',
        '按整图像簇有放回配对Bootstrap 200000次，保持题目加权Accuracy；下表单位pp。预声明六项比较作为同一族，给出普通95%及Bonferroni六项同时95%区间。','',
        '| 比较 | ΔAccuracy pp | 普通95%CI | 同时95%CI | 参考正确→条件错误 | 参考错误→条件正确 |',
        '|---|---:|---|---|---:|---:|']
    for r in paired:
        lines.append(f"| {r['label']} | {r['delta_pp']:+.3f} | [{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}] | [{r['family95_low_pp']:+.3f}, {r['family95_high_pp']:+.3f}] | {r['correct_to_wrong']} | {r['wrong_to_correct']} |")
    lines+=['','CI包含0不表示等价；本轮未预声明近似Full的等价界限，不能凭点估计宣称无损。所有逐题损益及条件交集保存在per_question.csv和error_transitions.json。','',
        '## 六类任务（描述性）','','| 任务 | Full | All OFF | Keep14 | Keep14+19 | Drop14+19 |','|---|---:|---:|---:|---:|---:|']
    for task in old.TASKS:
        cells=[next(r['accuracy_pct'] for r in tasks if r['task']==task and r['condition']==key) for key in KEYS]
        lines.append('| '+task+' | '+' | '.join(f'{v:.1f}%' for v in cells)+' |')
    core=summary[3];single=summary[2];drop=summary[4];alloff=summary[1]
    lines+=['','## 研究问题','',
        f"1. Keep L14+L19相对Full变化{core['delta_pp']:+.3f}pp，损害{core['correct_to_wrong']}题、纠正{core['wrong_to_correct']}题。请结合上述区间判断，不据此宣布两层充分或等价。",
        f"2. 在Keep L14上再保留L19，Accuracy变化{paired[5]['delta_pp']:+.3f}pp（逐题损益见C−B）。",
        f"3. 从Full仅删除L14+L19，Accuracy变化{drop['delta_pp']:+.3f}pp，损害{drop['correct_to_wrong']}题、纠正{drop['wrong_to_correct']}题。",
        f"4. 全部直接视觉边关闭变化{alloff['delta_pp']:+.3f}pp；恢复两层变化{paired[4]['delta_pp']:+.3f}pp。单层近零损害不能推出30层联合安全；nonadditivity.csv将本轮首Token Loss与匹配样本历史单层Loss之和比较。该和不是Accuracy预测，也不是可加性定理。",
        '5. 本轮与L14/L19来源筛选共用240题，不是独立确认；是否进入剩余2760题应依据损害与不确定性另行决策，本次不运行扩展。',
        '', '## 非加性检查','',
        '| 条件 | 实测首Token Loss | 历史单层Loss求和 | 联合减求和 |','|---|---:|---:|---:|']
    for key,v in nonadd.items():
        avg=np.mean(v,axis=0)
        lines.append(f'| {LABELS[key]} | {avg[0]:.6f} | {avg[1]:.6f} | {avg[2]:+.6f} |')
    lines+=['','LogP Loss固定Full Top-1，允许负值；不是gold答案Loss，不能直接等同Accuracy。恢复比例未报告，主要结论使用直接配对百分点差，避免小分母误导。','',
        '## 验证与耗时','',
        f"CPU {validation['cpu_tests']}项测试通过；三张RTX3090真实Smoke通过。No-op全有效Logits/生成与Full精确一致，历史Full首Token及单层L14精确一致；目标视觉概率为0，其他Query的Attention/Hidden/Logits精确不变；保留层正常读图；异常/正常Hook清理通过。输入、参数版本及模型文件哈希不变。",
        f"正式阶段墙钟{wall:.1f}秒；三Worker累计{validation['formal_gpu_seconds']:.1f}秒（含模型加载），Smoke累计{validation['smoke_gpu_seconds']:.1f}秒。逐卡耗时见validation.json。",
        'EOS、截断与空答案计数见accuracy_summary.csv；异常不被静默删除。原始首Token分布按条件NPZ保存，逐题检查点可恢复。','',
        '## 解释边界','',
        '这是直接P−1→Image K/V读取的层级功能实验，不是整个VLM只需两层视觉处理的证据。其他Question/模板Token仍可读图，残差和文本表示提供间接视觉路径；Decode仍能直接读图。没有物理K/V计算省略或GPU加速结论。',
        '统计为同一探索队列上的机制比较；未训练、未追加条件、未扩大样本。','',
        f"源码冻结Commit：`{c['code_commit']}`；协议SHA256：`{sha256(OUT/'protocol.json')}`。",'']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    files={str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name not in
           ('worker.lock','artifact_hashes.json') and p.suffix not in ('.log','.tmp')}
    freeze_json(OUT/'artifact_hashes.json',files)
    print(json.dumps(dict(summary=summary,comparisons=paired,validation=validation),ensure_ascii=False,indent=2),flush=True)


def tensor_equivalent(tensor, expected):
    from knockout_probe import tensor_hash
    return tensor_hash(tensor)==expected


if __name__=='__main__':main()
