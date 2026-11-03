"""Audit the frozen two-candidate experiment and make paired quality comparisons."""
import csv
import json
from pathlib import Path
import tarfile
import time
import io
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paper_rq1.scorers import gqa_prediction
from visual_writing_necessity.analyze import paired_bootstrap
from .common import (OUT,ROOT,HISTORY,CONDITIONS,TASKS,SEED,read,write,sha256,
                     digest,verify,writers,noninferiority,hardware)

ALL=['All-Writer']+CONDITIONS
COMPARISONS=[(1,0),(2,0),(2,1)]


def csv_file(name,rows,fields=None):
    assert rows or fields
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def audit():
    protocol=verify();binding=digest(protocol)
    samples=read(OUT/'sample_manifest.json')['samples'];smap={s['question_id']:s for s in samples}
    baseline=read(OUT/'historical_all_writer.json');inputs=read(OUT/'input_bindings.json')
    records={(qid,'All-Writer'):r for qid,r in baseline.items()};hashes=[];max_error=0.
    for index in range(2):
        folder=OUT/f'shard{index}';done=read(folder/'complete.json');smoke=read(folder/'smoke.json')
        assert done['passed'] and done['binding']==binding and smoke['passed'] and smoke['binding']==binding
        assert len(smoke['rows'])==3 and all(r['passed'] for r in smoke['rows'])
        assert done['hooks_clean'] and done['weights_inputs_unchanged']
        ids=set(read(OUT/'shards.json')['question_ids'][index])
        paths=sorted((folder/'checkpoints').glob('*/*.json'))
        assert len(paths)==done['generations']==len(ids)*2
        for path in paths:
            r=read(path);qid=r['question_id'];s=smap[qid];c=r['condition'];g=r['generation']
            assert path.name==c+'.json' and path.parent.name==s['task']+'__'+qid
            assert qid in ids and c in CONDITIONS and (qid,c) not in records
            assert r['binding']==binding and r['code_commit']==protocol['code_commit']
            assert r['sample_sha256']==digest(s) and r['image_sha256']==s['image_sha256']
            assert r['task']==s['task'] and r['gold']==s['answer']
            assert r['gpu']==protocol['gpus'][index] and r['gpu']==baseline[qid]['gpu']
            assert r['input_hashes']==inputs[qid]['input_hashes'] and r['generation_sha256']==digest(g)
            assert g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer'])
            assert g['writer_layers']==writers(c) and g['P1_Decode_image_off']
            assert g['independent_cache'] and g['independent_prefix']
            tokens=g['token_ids'];assert 1<=len(tokens)<=32 and tokens[0]==g['first_top1']
            assert g['eos']==(tokens[-1]==2) and g['truncated']==(len(tokens)==32 and tokens[-1]!=2)
            assert g['empty']==(not g['text'].strip()) and len(g['first_logits_sha256'])==64
            assert g['baseline_top1_token']==baseline[qid]['generation']['first_top1']
            assert g['first_top1_flip']==(g['first_top1']!=g['baseline_top1_token'])
            layout=g['layout'];n=layout['prompt_length'];assert layout==baseline[qid]['generation']['layout']
            assert len(layout['I'])==576 and layout['A']==[n-1]
            a=g['audit'];assert a['steps']==len(tokens) and a['layer_events']==32*len(tokens)
            assert a['blocked_probability_max']==0 and a['row_sum_max_error']<=0.002
            assert a['non_target_mask_exact'] and a['Image_Query_mask_native'] and a['BOS_Self_legal']
            assert len(a['actual_positions'])==len(tokens)
            for t,event in enumerate(a['actual_positions']):
                expected=[576*(1+(len(layout['R']) if not t and layer not in writers(c) else 0)) for layer in range(32)]
                assert event['blocked_connections_per_layer']==expected
                assert event['past']==(n+t-1 if t else 0)
                assert event['query_length']==(1 if t else n) and event['key_length']==n+t
            max_error=max(max_error,a['row_sum_max_error'])
            records[qid,c]=r;hashes.append(dict(path=str(path.relative_to(OUT)),sha256=sha256(path)))
        print('AUDIT',index,len(paths),'new generations',flush=True)
    assert len(records)==9000 and len(hashes)==6000
    # Full-content model hashes, not just file modification times, are rechecked after GPU work.
    for path,info in protocol['model_files'].items():assert sha256(path)==info['sha256']
    images={s['image_sha256']:s['image_path'] for s in samples}
    for expected,path in images.items():assert sha256(path)==expected
    write(OUT/'checkpoint_hashes.json',hashes)
    write(OUT/'validation.json',dict(passed=True,new_generations=6000,reused_All_Writer=3000,
        questions=3000,image_clusters=2409,conditions=3,binding=binding,
        strict_HF_noop_logits=True,strict_historical_baseline_first_logits_tokens=True,
        smoke_questions=6,CPU_tests=8,same_historical_pairing_GPU=True,
        original_inputs_and_full_model_file_hashes_unchanged=True,
        image_files_SHA256_checked=len(images),masked_probability_max=0,
        maximum_attention_row_error=max_error,cache_isolation=True,hooks_clean=True,
        no_NaN_Inf_OOM_or_silent_skips=True,all_checkpoints_bound_and_unique=True,
        no_original_Full_substitution=True))
    return samples,records


def analyze_set(name,samples,records):
    n=len(samples);images=[s['image_sha256'] for s in samples]
    scores=np.array([[records[s['question_id'],c]['generation']['correct'] for c in ALL] for s in samples])
    differences=np.stack([(scores[:,a]-scores[:,b])*100 for a,b in COMPARISONS],axis=1)
    draws=paired_bootstrap(differences,images,repetitions=20000,seed=SEED)
    ordinary=np.percentile(draws,[2.5,97.5],axis=0)
    corrected=np.percentile(draws,[1.25,98.75],axis=0)
    np.savez_compressed(OUT/f'bootstrap_{name}.npz',paired_delta_accuracy_pp=draws)
    summary=[];pairs=[]
    for k,c in enumerate(ALL):
        gs=[records[s['question_id'],c]['generation'] for s in samples]
        bs=[records[s['question_id'],'All-Writer']['generation'] for s in samples]
        summary.append(dict(analysis_set=name,questions=n,image_clusters=len(set(images)),condition=c,
            allowed_writing_layers=32 if k==0 else len(writers(c)),correct=int(scores[:,k].sum()),
            accuracy_pct=float(scores[:,k].mean()*100),delta_vs_All_Writer_pp=float((scores[:,k]-scores[:,0]).mean()*100),
            correct_to_wrong=int(((scores[:,0]==1)&(scores[:,k]==0)).sum()),
            wrong_to_correct=int(((scores[:,0]==0)&(scores[:,k]==1)).sum()),
            first_top1_flips=sum(g['first_top1']!=b['first_top1'] for g,b in zip(gs,bs)),
            text_changes=sum(g['text']!=b['text'] for g,b in zip(gs,bs)),
            token_sequence_changes=sum(g['token_ids']!=b['token_ids'] for g,b in zip(gs,bs)),
            mean_generated_tokens=float(np.mean([len(g['token_ids']) for g in gs])),
            eos=sum(g['eos'] for g in gs),truncated=sum(g['truncated'] for g in gs),
            empty=sum(g['empty'] for g in gs),repeated_trigram=sum(g['repeated_trigram'] for g in gs)))
    for j,(a,b) in enumerate(COMPARISONS):
        ca,cb=ALL[a],ALL[b]
        pairs.append(dict(analysis_set=name,questions=n,image_clusters=len(set(images)),candidate=ca,reference=cb,
            delta_accuracy_pp=float(differences[:,j].mean()),ci95_low=float(ordinary[0,j]),ci95_high=float(ordinary[1,j]),
            family95_low=float(corrected[0,j]) if j<2 else '',family95_high=float(corrected[1,j]) if j<2 else '',
            family_comparisons=2 if j<2 else '',
            NI_minus1pp_pass=noninferiority(corrected[0,j],1) if j<2 else '',
            NI_minus2pp_pass=noninferiority(corrected[0,j],2) if j<2 else '',
            NI_status='Primary decision' if name=='primary_2640' and j<2 else 'Supportive/exploratory only',
            reference_correct_candidate_wrong=int(((scores[:,b]==1)&(scores[:,a]==0)).sum()),
            reference_wrong_candidate_correct=int(((scores[:,b]==0)&(scores[:,a]==1)).sum()),
            detectable_positive_benefit_ordinary95=bool(ordinary[0,j]>0)))
    print('BOOTSTRAP',name,n,'questions',len(set(images)),'images',flush=True)
    return summary,pairs


def task_statistics(name,samples,records):
    result=[]
    for task in TASKS:
        ss=[s for s in samples if s['task']==task]
        for c in ALL:
            gs=[records[s['question_id'],c]['generation'] for s in ss]
            bs=[records[s['question_id'],'All-Writer']['generation'] for s in ss]
            result.append(dict(analysis_set=name,task=task,questions=len(ss),condition=c,
                correct=sum(g['correct'] for g in gs),accuracy_pct=float(np.mean([g['correct'] for g in gs])*100),
                delta_accuracy_pp=float(np.mean([g['correct']-b['correct'] for g,b in zip(gs,bs)])*100),
                correct_to_wrong=sum(b['correct']==1 and g['correct']==0 for g,b in zip(gs,bs)),
                wrong_to_correct=sum(b['correct']==0 and g['correct']==1 for g,b in zip(gs,bs))))
    return result


def per_question(samples,records,primary):
    rows=[];negative=[]
    for s in samples:
        qid=s['question_id'];base=records[qid,'All-Writer']['generation']
        for c in ALL:
            r=records[qid,c];g=r['generation']
            row=dict(question_id=qid,task=s['task'],image_id=s['image_id'],image_sha256=s['image_sha256'],
                primary_image_disjoint=qid in primary,gold=s['answer'],condition=c,
                answer=g['text'],prediction=g['prediction'],correct=g['correct'],baseline_correct=base['correct'],
                delta_correct=g['correct']-base['correct'],
                correct_to_wrong=base['correct']==1 and g['correct']==0,wrong_to_correct=base['correct']==0 and g['correct']==1,
                first_top1=g['first_top1'],baseline_top1=base['first_top1'],first_token_flip=g['first_top1']!=base['first_top1'],
                text_changed=g['text']!=base['text'],token_ids=json.dumps(g['token_ids']),
                generated_tokens=len(g['token_ids']),eos=g['eos'],truncated=g['truncated'],empty=g['empty'],
                repeated_trigram=g['repeated_trigram'],gpu_uuid=r['gpu']['uuid'],
                source='audited historical All-Writer' if c=='All-Writer' else 'new paired generation')
            rows.append(row)
            if row['correct_to_wrong']:negative.append(row)
    csv_file('per_question.csv',rows);csv_file('negative_cases.csv',negative,fields=list(rows[0]))
    transitions={}
    for a,b in COMPARISONS:
        key=ALL[a]+' minus '+ALL[b]
        transitions[key]={label:[s['question_id'] for s in samples
            if records[s['question_id'],ALL[b]]['generation']['correct']==before
            and records[s['question_id'],ALL[a]]['generation']['correct']==after]
            for label,before,after in [('correct_to_wrong',1,0),('wrong_to_correct',0,1)]}
    write(OUT/'error_transitions.json',transitions)


def figures(summary,pairs,tasks):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
        'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240})
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.8),layout='constrained')
    for ax,name,title in zip(axes,['primary_2640','overall_3000'],['Primary: image-disjoint (n = 2,640)','Supportive: all questions (n = 3,000)']):
        rr=[r for r in pairs if r['analysis_set']==name and r['reference']=='All-Writer']
        for x,r,color in zip([0,1],rr,['#187d92','#a33863']):
            ax.vlines(x,r['family95_low'],r['family95_high'],color=color,lw=4,alpha=.25)
            ax.vlines(x,r['ci95_low'],r['ci95_high'],color=color,lw=1.4)
            ax.plot(x,r['delta_accuracy_pp'],'o',color=color,ms=7)
        ax.axhline(0,color='#444444',lw=.8)
        ax.axhline(-1,color='#bc3d3d',ls='--',lw=1,label='Primary margin: -1 pp')
        ax.axhline(-2,color='#8c7956',ls=':',lw=1,label='Supplementary: -2 pp')
        ax.set(title=title,xticks=[0,1],xticklabels=['Keep L0-L15','Keep L0-L19'],
               ylabel='Accuracy difference vs All-Writer (pp)',xlim=(-.6,1.6))
        ax.grid(axis='y',alpha=.15)
    limits=[ax.get_ylim() for ax in axes];low=min(x[0] for x in limits);high=max(x[1] for x in limits)
    for ax in axes:ax.set_ylim(low,high)
    axes[0].legend(frameon=False,fontsize=8,loc='lower left')
    fig.suptitle('Visual Writing Budget: 16 vs 20 Layers\nThin: paired 95% CI; wide: two-candidate familywise 95% CI',fontsize=11)
    for ext in ['pdf','png']:fig.savefig(OUT/f'quality_comparison.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,name,title in zip(axes,['primary_2640','overall_3000'],['Primary image-disjoint subset','All 3,000 questions']):
        x=np.arange(6)
        for shift,c,color in [(-.18,CONDITIONS[0],'#187d92'),(.18,CONDITIONS[1],'#a33863')]:
            rr=[next(r for r in tasks if r['analysis_set']==name and r['task']==t and r['condition']==c) for t in TASKS]
            ax.bar(x+shift,[r['delta_accuracy_pp'] for r in rr],width=.34,label=c,color=color)
        ax.axhline(0,color='#444444',lw=.8);ax.set(xticks=x,xticklabels=TASKS,title=title,ylabel='Accuracy difference (pp)')
        ax.tick_params(axis='x',rotation=35,labelsize=8);ax.grid(axis='y',alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    for ext in ['pdf','png']:fig.savefig(OUT/f'task_comparison.{ext}')
    plt.close(fig)


def report(summary,pairs,tasks):
    primary=[r for r in pairs if r['analysis_set']=='primary_2640'];support=[r for r in pairs if r['analysis_set']=='overall_3000']
    passing=[r['candidate'] for r in primary[:2] if r['NI_minus1pp_pass']]
    if passing:recommend=passing[0]+'：通过优先验证集预设 −1 pp 标准的候选中，选择视觉写入层数较少者。'
    else:recommend='两个候选均未通过优先验证集的 −1 pp 非劣效标准；本轮不推荐将任一候选称为满足主要质量要求。'
    gain=primary[2]
    if gain['ci95_low']>0:benefit='额外保留 L16–L19 有可检测的正向 Accuracy 收益。'
    elif gain['ci95_high']<0:benefit='额外保留 L16–L19 反而有可检测的负向 Accuracy 变化。'
    else:benefit='额外保留 L16–L19 的配对区间跨 0，未检测到明确收益；这不证明两候选等价。'
    lines=['# GQA3000 视觉写入：16 层 vs 20 层','',
        '## 核心结论','',recommend,benefit,'',
        '## 协议与基线','',
        '沿用历史固定 LLaVA-1.5-7B revision、FP16/eager、Transformers 4.49.0、原始 GQA3000、Prompt 与官方评分。'
        '所有条件 P−1 及全部真实 Decode 在32层均禁止直接读取 Image。只改变 Prefill R→Image：Keep-0-15 保留16层，Keep-0-19保留20层。'
        '其他 Prefill Query、Image Query、MLP、Residual、QKV、位置和 KV 更新正常；Greedy 最多32 Token，逐条件独立缓存。',
        '历史 All-Writer 3000题逐记录校验归档 SHA、图片/输入/模型/评分/生成绑定；2611/3000正确（87.033333%）。'
        '不是原生 Full。本轮只新增两条件共6000条完整答案，不计算新的 Gold NLL。',
        '优先验证集重新按历史240张探索图片 SHA 排除得到2640题；与历史探索图片重叠的360题保留在总体3000题中。'
        '此子集仅与该探索集图像隔离，并非独立于所有既往队列研究的外部测试集。','',
        '## Accuracy','',
        '| 集合 | 条件 | 正确/题数 | Accuracy | ΔAll-Writer (pp) | 原对→错 | 原错→对 | 首Token翻转 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['analysis_set']} | {r['condition']} | {r['correct']}/{r['questions']} | {r['accuracy_pct']:.4f}% | {r['delta_vs_All_Writer_pp']:+.4f} | {r['correct_to_wrong']} | {r['wrong_to_correct']} | {r['first_top1_flips']} |")
    lines+=['','## 三项预先固定配对比较','',
        '按 Image SHA256 整簇配对 Bootstrap 20000次（seed=20261103），保留题目加权估计；不是将同图多题当成独立图片。'
        '两个对 All-Writer 的候选以 Bonferroni 双侧家族95%区间判断：单比较97.5%，分位数1.25/98.75。'
        '校正下界必须严格高于 −1 pp；−2 pp 仅为补充标准。主要决策仅使用2640题。'
        '20层−16层为预声明次要比较，普通双侧95%区间；六任务只作描述，不宣称确认性分任务显著。','',
        '| 集合 | 比较 | ΔAcc (pp) | 普通95%CI | 两候选家族95%CI | −1pp通过 | −2pp通过 | 参考对→错 / 错→对 |',
        '|---|---|---:|---|---|---|---|---:|']
    for r in pairs:
        corrected=f"[{r['family95_low']:+.4f}, {r['family95_high']:+.4f}]" if r['family95_low']!='' else '不适用'
        lines.append(f"| {r['analysis_set']} | {r['candidate']} − {r['reference']} | {r['delta_accuracy_pp']:+.4f} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {corrected} | {r['NI_minus1pp_pass']} | {r['NI_minus2pp_pass']} | {r['reference_correct_candidate_wrong']} / {r['reference_wrong_candidate_correct']} |")
    lines+=['','总体3000题中的判定只作支持性结果；不能以总体或 −2 pp 通过替代优先集合的 −1 pp 要求。',
        'CI包含0不等于无损；未通过非劣效不等于证明损害必然超出边界。','',
        '## 六任务表现','', '| 集合 | 任务 | 题数 | All-Writer | Keep-0-15 | Keep-0-19 |','|---|---|---:|---:|---:|---:|']
    for name in ['primary_2640','overall_3000']:
        for t in TASKS:
            rr=[next(r for r in tasks if r['analysis_set']==name and r['task']==t and r['condition']==c) for c in ALL]
            lines.append(f"| {name} | {t} | {rr[0]['questions']} | {rr[0]['accuracy_pct']:.3f}% | {rr[1]['accuracy_pct']:.3f}% | {rr[2]['accuracy_pct']:.3f}% |")
    lines+=['','## 输出与有效性','',
        '8项CPU测试、两卡各3题Smoke通过。相同 logits_to_keep=1 形状下 Native/No-op 所有有效返回Logits严格 torch.equal；'
        '同卡历史 All-Writer 首Logits哈希及生成Token严格一致。被禁连接实际概率为0，非目标Mask和前图像前缀输出不变，'
        'Cache/前缀隔离与正常/异常Hook清理通过；完整模型、输入和全部2409张图片哈希复核。',
        'validation.json、GPU日志、6000条新检查点及哈希均保留。per_question.csv含3000条已审计基线和6000条新生成，'
        'negative_cases.csv保留所有相对基线正确→错误案例；error_transitions.json额外列出20层对16层的全部损害/纠正题号。',
        'EOS、截断、空答案、重复和首Token翻转见 accuracy_summary.csv；实际结束状态与显存释放另见 completion.json / gpu_release_audit.json。','',
        '## 边界','',
        '只检验两个已冻结候选，没有追加层组合、按结果删题或训练。16/20表示允许R直接读图的层数，不是整层跳过。'
        'Softmax前Hard Mask不是实际Kernel优化、KV剪枝或加速；质量通过仅支持在本队列及所测协议下的下一步工程候选，不证明信息只存在于保留层。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    write(OUT/'recommendation.json',dict(primary_set='primary_2640',minus1pp_pass=passing,
        recommendation=recommend,additional_L16_L19=benefit,paired_difference=gain,
        all_comparisons_predeclared=True,no_extra_experiments=True))


def archive_raw():
    archives=[]
    for index in range(2):
        folder=OUT/f'shard{index}';path=OUT/f'raw_checkpoints_shard{index}.tar.gz'
        files=sorted((folder/'checkpoints').glob('*/*.json'))
        with tarfile.open(path,'w:gz',compresslevel=6) as archive:
            for file in files:archive.add(file,arcname=str(file.relative_to(OUT)),recursive=False)
        archives.append(dict(path=path.name,records=len(files),bytes=path.stat().st_size,sha256=sha256(path)))
    write(OUT/'raw_archive_manifest.json',dict(archives=archives,total_records=6000))


def main():
    start=time.time();stream=io.StringIO()
    tests=unittest.defaultTestLoader.loadTestsFromName('vlm_research.visual_writing_16_vs_20.analysis_tests')
    checked=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    write(OUT/'analysis_tests_validation.json',dict(passed=checked.wasSuccessful(),tests=checked.testsRun,
        output=stream.getvalue(),source_sha256=sha256(Path(__file__).with_name('analysis_tests.py'))))
    assert checked.wasSuccessful()
    samples,records=audit();sets=read(OUT/'analysis_sets.json');primary=set(sets['primary_image_disjoint'])
    groups=[('primary_2640',[s for s in samples if s['question_id'] in primary]),('overall_3000',samples)]
    summary=[];pairs=[];tasks=[]
    for name,ss in groups:
        a,b=analyze_set(name,ss,records);summary.extend(a);pairs.extend(b);tasks.extend(task_statistics(name,ss,records))
    csv_file('accuracy_summary.csv',summary);csv_file('paired_comparisons.csv',pairs);csv_file('task_accuracy.csv',tasks)
    per_question(samples,records,primary);figures(summary,pairs,tasks);report(summary,pairs,tasks);archive_raw()
    write(OUT/'analysis_complete.json',dict(passed=True,seconds=time.time()-start,
        new_generations=6000,reused_baselines=3000,primary_questions=2640,all_questions=3000,
        protocol_sha256=sha256(OUT/'protocol.json'),analysis_source_sha256=sha256(Path(__file__)),
        artifacts={str(path.relative_to(OUT)):sha256(path) for path in sorted(OUT.iterdir())
            if path.is_file() and path.suffix in ['.json','.csv','.png','.pdf','.npz','.gz']
            and path.name!='analysis_complete.json'}))
    print(json.dumps(dict(accuracy=summary,paired_comparisons=pairs),indent=2),flush=True)


if __name__=='__main__':main()
