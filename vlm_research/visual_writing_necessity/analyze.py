"""CPU-only completeness audit, image-cluster statistics, and paper figures."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import OUT,CONDITIONS,TASKS,SEED,read,write,sha256,digest,verify,writer_layers


def dump_csv(name,rows,fields=None):
    assert rows or fields
    path=OUT/name
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields or list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def paired_bootstrap(values,images,repetitions=20000,seed=SEED):
    """Whole-image pairs, retaining the original question-weighted estimand."""
    values=np.asarray(values,dtype=np.float64)
    if values.ndim==1:values=values[:,None]
    _,inverse=np.unique(images,return_inverse=True);k=int(inverse.max())+1
    sums=np.zeros((k,values.shape[1]),dtype=np.float64)
    np.add.at(sums,inverse,values)
    counts=np.bincount(inverse,minlength=k).astype(np.float64)
    rng=np.random.default_rng(seed);result=np.empty((repetitions,values.shape[1]))
    for begin in range(0,repetitions,128):
        end=min(begin+128,repetitions)
        weights=rng.multinomial(k,np.full(k,1/k),size=end-begin)
        result[begin:end]=(weights@sums)/(weights@counts)[:,None]
    assert np.isfinite(result).all()
    return result


def audit():
    p=verify();binding=digest(p);manifest=read(OUT/'sample_manifest.json')['samples']
    samples={s['question_id']:s for s in manifest};gold=read(OUT/'gold_sequences.json')
    mappings=read(OUT/'input_bindings.json');shards=read(OUT/'shards.json')
    result={};row_errors=[];checkpoints=[]
    for index in range(2):
        folder=OUT/f'shard{index}';done=read(folder/'complete.json');smoke=read(folder/'smoke.json')
        assert done['passed'] and done['binding']==smoke['binding']==binding and smoke['passed']
        ids=set(shards['question_ids'][index]);paths=sorted((folder/'checkpoints').glob('*/*.json'))
        assert len(paths)==len(ids)*34==done['checkpoints']
        for path in paths:
            r=read(path);qid=r['question_id'];c=r['condition'];g=r['generation'];s=samples[qid]
            assert qid in ids and c in CONDITIONS and (qid,c) not in result
            assert r['binding']==binding and r['code_commit']==p['code_commit']
            assert r['sample_sha256']==digest(s) and r['image_sha256']==s['image_sha256']
            assert r['gpu']==p['gpus'][index] and r['input_hashes']==mappings[qid]['input_hashes']
            assert digest(g)==r['generation_sha256'] and g['gold_token_ids']==gold[qid]['token_ids']
            assert len(g['gold_token_logp'])==len(g['gold_token_ids'])
            assert g['gold_nll']==-sum(g['gold_token_logp'])/len(g['gold_token_ids'])
            assert g['gold_EOS_excluded'] and not ({1,2}&set(g['gold_token_ids']))
            assert g['writer_layers']==writer_layers(c) and g['P1_Decode_image_off'] and g['independent_free_gold_caches']
            assert 1<=len(g['token_ids'])<=32 and g['eos']==(g['token_ids'][-1]==2)
            assert g['correct']==int(g['prediction']==s['answer'])
            layout=g['layout'];n=layout['prompt_length']
            assert len(layout['I'])==576 and layout['A']==[n-1]
            for key,steps,begin in [('free_audit',len(g['token_ids']),0),('gold_audit',len(g['gold_token_ids'])-1,1)]:
                a=g[key];assert a['steps']==steps and a['layer_events']==steps*32
                assert a['blocked_probability_max']==0 and a['row_sum_max_error']<=0.002
                assert a['non_target_mask_exact'] and a['Image_Query_mask_native'] and a['BOS_Self_legal']
                for t,event in enumerate(a['actual_positions'],start=begin):
                    assert event['past']==(n+t-1 if t else 0)
                    assert event['query_length']==(1 if t else n) and event['key_length']==n+t
                    expected=[576*(1+(len(layout['R']) if not t and layer not in writer_layers(c) else 0)) for layer in range(32)]
                    assert event['blocked_connections_per_layer']==expected
                row_errors.append(a['row_sum_max_error'])
            assert np.isfinite([g['gold_nll'],g['first_logp_loss'],*g['gold_token_logp']]).all()
            result[qid,c]=r;checkpoints.append(dict(path=str(path.relative_to(OUT)),sha256=sha256(path)))
        print('AUDIT shard',index,len(paths),'checkpoints',flush=True)
    assert len(result)==102000
    write(OUT/'checkpoint_hashes.json',checkpoints)
    audit_record=dict(passed=True,questions=3000,image_clusters=2409,generations=102000,
        Gold_NLL_evaluations=102000,conditions=34,binding=binding,source_commit=p['code_commit'],
        maximum_attention_row_error=max(row_errors),masked_probability_max=0.,all_gold_tokens_EOS_excluded=True,
        same_image_all_conditions_same_GPU=True,strict_GPU_smokes=True,input_cache_hook_checks=True,
        checkpoint_hashes_sha256=sha256(OUT/'checkpoint_hashes.json'))
    write(OUT/'validation.json',audit_record)
    return manifest,result,p


def archive_checkpoints():
    destination=OUT/'raw_checkpoints';destination.mkdir(exist_ok=True)
    archives=[]
    for index in range(2):
        paths=sorted((OUT/f'shard{index}/checkpoints').glob('*/*.json'))
        for begin in range(0,len(paths),5000):
            chunk=paths[begin:begin+5000];target=destination/f'shard{index}_{begin//5000:02d}.tar.gz'
            with tarfile.open(target,'w:gz',compresslevel=6) as archive:
                for path in chunk:archive.add(path,arcname=str(path.relative_to(OUT)),recursive=False)
            assert target.stat().st_size<95*1024**2,'Split archive before GitHub upload'
            with tarfile.open(target,'r:gz') as archive:
                members=archive.getmembers();assert len(members)==len(chunk)
                for member,path in zip(members,chunk):
                    import hashlib
                    assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==sha256(path)
            archives.append(dict(path=str(target.relative_to(OUT)),sha256=sha256(target),records=len(chunk)))
    assert sum(r['records'] for r in archives)==102000
    write(OUT/'raw_archive_manifest.json',dict(passed=True,records=102000,archives=archives,
        recovery='Each tar.gz preserves exact original per-condition checkpoint bytes and relative paths; verify against checkpoint_hashes.json.'))


def table_rows(samples,records):
    rows=[]
    for s in samples:
        qid=s['question_id'];base=records[qid,'All-Writer']['generation']
        for c in CONDITIONS:
            r=records[qid,c];g=r['generation']
            assert g['baseline_top1_token']==base['first_top1']
            assert g['first_top1_flip']==(g['first_top1']!=base['first_top1'])
            assert g['first_logp_loss']==base['first_top1_logp']-g['baseline_token_logp']
            rows.append(dict(question_id=qid,image_id=s['image_id'],image_sha256=s['image_sha256'],task=s['task'],
                condition=c,gold=s['answer'],gold_token_ids=json.dumps(g['gold_token_ids']),
                gold_token_logp=json.dumps(g['gold_token_logp']),gold_nll=g['gold_nll'],
                delta_gold_nll=g['gold_nll']-base['gold_nll'],correct=g['correct'],delta_correct=g['correct']-base['correct'],
                correct_to_wrong=int(base['correct']==1 and g['correct']==0),wrong_to_correct=int(base['correct']==0 and g['correct']==1),
                prediction=g['prediction'],text=g['text'],token_ids=json.dumps(g['token_ids']),
                first_top1_flip=int(g['first_top1_flip']),first_full_token_logp_loss=g['first_logp_loss'],
                generated_text_changed=int(g['text']!=base['text']),generated_tokens_changed=int(g['token_ids']!=base['token_ids']),
                generation_length=len(g['token_ids']),eos=int(g['eos']),truncated=int(g['truncated']),empty=int(g['empty']),
                repeated_trigram=int(g['repeated_trigram']),gpu=r['gpu']['physical_index'],seconds=r['seconds']))
    return rows


def summarize(samples,records,name):
    nll=np.array([[records[s['question_id'],c]['generation']['gold_nll'] for c in CONDITIONS] for s in samples])
    acc=np.array([[records[s['question_id'],c]['generation']['correct'] for c in CONDITIONS] for s in samples])
    dn=nll-nll[:,[0]];da=(acc-acc[:,[0]])*100
    images=[s['image_sha256'] for s in samples]
    draws=paired_bootstrap(np.concatenate([dn,da],axis=1),images)
    point=np.quantile(draws,[.025,.975],axis=0);corrected=np.quantile(draws,[.05/(2*32),1-.05/(2*32)],axis=0)
    rows=[]
    for j,c in enumerate(CONDITIONS):
        generations=[records[s['question_id'],c]['generation'] for s in samples]
        rows.append(dict(subset=name,condition=c,questions=len(samples),image_clusters=len(set(images)),
            accuracy_pct=float(acc[:,j].mean()*100),gold_nll=float(nll[:,j].mean()),delta_accuracy_pp=float(da[:,j].mean()),
            delta_accuracy_ci_low=float(point[0,34+j]),delta_accuracy_ci_high=float(point[1,34+j]),
            delta_accuracy_corrected_low=float(corrected[0,34+j]) if j>=2 else '',
            delta_accuracy_corrected_high=float(corrected[1,34+j]) if j>=2 else '',
            delta_gold_nll=float(dn[:,j].mean()),delta_gold_nll_ci_low=float(point[0,j]),delta_gold_nll_ci_high=float(point[1,j]),
            delta_gold_nll_corrected_low=float(corrected[0,j]) if j>=2 else '',delta_gold_nll_corrected_high=float(corrected[1,j]) if j>=2 else '',
            correct_to_wrong=int(((acc[:,0]==1)&(acc[:,j]==0)).sum()),wrong_to_correct=int(((acc[:,0]==0)&(acc[:,j]==1)).sum()),
            positive_delta_nll=int((dn[:,j]>0).sum()),negative_delta_nll=int((dn[:,j]<0).sum()),
            median_delta_nll=float(np.median(dn[:,j])),p5_delta_nll=float(np.quantile(dn[:,j],.05)),p95_delta_nll=float(np.quantile(dn[:,j],.95)),
            first_top1_flips=sum(g['first_top1_flip'] for g in generations),text_changes=sum(g['text']!=records[s['question_id'],'All-Writer']['generation']['text'] for s,g in zip(samples,generations)),
            eos=sum(g['eos'] for g in generations),truncated=sum(g['truncated'] for g in generations),empty=sum(g['empty'] for g in generations),
            repeated=sum(g['repeated_trigram'] for g in generations)))
    np.savez_compressed(OUT/f'bootstrap_{name}.npz',delta_nll=draws[:,:34],delta_accuracy_pp=draws[:,34:])
    print('BOOTSTRAP',name,len(samples),'questions',len(set(images)),'images',flush=True)
    return rows


def figures(overall,tasks):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':200})
    drops=overall[2:];x=np.arange(32)
    for kind,title,label,color in [('gold_nll','Visual Writing Necessity Profile',r'Mean $\Delta$ Gold NLL (nat/token)','#087e8b'),
        ('accuracy','Accuracy Sensitivity Profile',r'$\Delta$ Accuracy (pp)','#a23b72')]:
        field='delta_gold_nll' if kind=='gold_nll' else 'delta_accuracy_pp'
        prefix='delta_gold_nll' if kind=='gold_nll' else 'delta_accuracy'
        y=np.array([r[field] for r in drops]);lo=np.array([r[prefix+'_ci_low'] for r in drops]);hi=np.array([r[prefix+'_ci_high'] for r in drops])
        clo=np.array([r[prefix+'_corrected_low'] for r in drops]);chi=np.array([r[prefix+'_corrected_high'] for r in drops])
        fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
        ax.vlines(x,clo,chi,color=color,alpha=.25,lw=3,label='32-comparison Bonferroni 95% CI')
        ax.vlines(x,lo,hi,color=color,lw=1,label='Pointwise 95% CI')
        ax.plot(x,y,'o-',color=color,lw=1.2,ms=3)
        ax.axhline(0,color='#404040',ls='--',lw=.8);ax.axvspan(7.5,11.5,color='#e9c46a',alpha=.15,label='Historical L8–L11 window')
        ax.set(xlabel='Dropped visual writing layer',ylabel=label,title=title,xticks=np.arange(0,32,2),xlim=(-.7,31.7))
        ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,fontsize=8)
        stem='figure_A_visual_writing_necessity' if kind=='gold_nll' else 'figure_B_accuracy_sensitivity'
        for ext in ['pdf','png']:fig.savefig(OUT/f'{stem}.{ext}')
        plt.close(fig)
    signed=np.array([r['delta_gold_nll'] for r in drops]);order=np.argsort(-np.maximum(signed,0),kind='stable')
    positive=np.maximum(signed[order],0);total=positive.sum()
    cum=np.cumsum(positive)/total if total>0 else np.zeros(32)
    k80=int(np.searchsorted(cum,.8)+1) if total>0 else None
    concentration=[dict(rank=i+1,layer=int(layer),signed_delta_nll=float(signed[layer]),positive_weight=float(positive[i]/total) if total else 0.,cumulative=float(cum[i])) for i,layer in enumerate(order)]
    dump_csv('writing_concentration.csv',concentration)
    write(OUT/'writing_concentration.json',dict(K80=k80,positive_sum=float(total),negative_layers=np.where(signed<0)[0].tolist(),
        interpretation='Descriptive normalized positive single-drop sensitivity, not an information fraction; single-layer effects are not additive.'))
    fig,ax=plt.subplots(figsize=(6,3.4),layout='constrained');ax.plot(np.arange(1,33),cum,'o-',ms=3,color='#087e8b')
    ax.axhline(.8,color='#777777',ls='--',lw=.8)
    if k80:ax.axvline(k80,color='#a23b72',ls=':',label=f'K80 = {k80}')
    ax.set(xlabel='Number of layers ranked by positive mean ΔNLL',ylabel='Cumulative positive sensitivity',
        title='Writing Importance Concentration',ylim=(-.02,1.03));ax.grid(alpha=.15)
    if k80:ax.legend(frameon=False)
    for ext in ['pdf','png']:fig.savefig(OUT/f'figure_C_concentration.{ext}')
    plt.close(fig)
    for field,label,stem in [('delta_gold_nll','ΔGold NLL (nat/token)','task_gold_nll'),('delta_accuracy_pp','ΔAccuracy (pp)','task_accuracy')]:
        fig,axes=plt.subplots(2,3,figsize=(11,5.4),layout='constrained')
        for ax,task in zip(axes.flat,TASKS):
            rows=[r for r in tasks if r['subset']==task][2:]
            prefix='delta_gold_nll' if field=='delta_gold_nll' else 'delta_accuracy'
            y=np.array([r[field] for r in rows]);lo=np.array([r[prefix+'_ci_low'] for r in rows]);hi=np.array([r[prefix+'_ci_high'] for r in rows])
            ax.plot(x,y,color='#087e8b',lw=1);ax.fill_between(x,lo,hi,color='#087e8b',alpha=.15)
            ax.axhline(0,color='#777777',lw=.8,ls='--');ax.axvspan(7.5,11.5,color='#e9c46a',alpha=.12)
            ax.set(title=task,xlabel='Dropped layer',ylabel=label,xticks=[0,8,16,24,31]);ax.grid(axis='y',alpha=.15)
        for ext in ['pdf','png']:fig.savefig(OUT/f'{stem}.{ext}')
        plt.close(fig)
    return k80


def report(overall,disjoint,tasks,p,k80):
    d=overall[2:];ranked=sorted(d,key=lambda r:r['delta_gold_nll'],reverse=True)
    overlap=read(OUT/'overlap_audit.json');validation=read(OUT/'validation.json')
    lines=['# GQA3000 视觉写入单层必要性分析','',
        '**状态：统计与生成完整；GPU 释放状态另见 `resource_release.json`。**','',
        f"3000 题、2409 个图像 SHA256 簇、34 条件、{validation['generations']:,} 条完整生成及 Gold NLL。",
        f"执行源码：`{p['code_commit']}`；独立分支 `codex/gqa3000-visual-writing-necessity-curve`。",'',
        '## 协议与指标','',
        '所有条件的原始 P−1 与全部真实 Decode/Gold-prefix Query 均在 32 层禁止直接读取 Image Key。All-Writer 仅表示所有层 R Query 可直接读图，绝不是原生 Full。No-Writer 在所有层额外阻断 R→Image。32 个 Drop 条件各只阻断一层 R→Image。',
        'R 严格沿用历史定义：图像之后、P−1 之前的非视觉 Prefill 文本 Query。Image Query、MLP、残差及 KV 更新保持原生。Hard Mask 位于 Softmax 前。',
        '固定官方 Gold 答案通过 `chat + 空格 + answer` 联合 Tokenization，逐题核验原 Prompt 前缀精确一致，不添加 BOS、不含 EOS。每题主 NLL 为该答案全部 Token 的 FP32 全词表负 LogP 均值；全数据以题目等权平均。多 Token 分支使用同条件 Prefill Cache 的独立存储副本，并在 Smoke 中与独立新 Cache Teacher Forcing 精确对齐。',
        'LLaVA-1.5-7B 固定 revision、FP16/eager、Transformers 4.49.0；Greedy 最多 32 Token。每个条件独立生成真实前缀和 Cache。','',
        '## 基线与负对照','',
        '| 条件 | Accuracy | Gold NLL | ΔAccuracy (pp) | ΔGold NLL |',
        '|---|---:|---:|---:|---:|']
    for r in overall[:2]:lines.append(f"| {r['condition']} | {r['accuracy_pct']:.4f}% | {r['gold_nll']:.6f} | {r['delta_accuracy_pp']:+.4f} | {r['delta_gold_nll']:+.6f} |")
    lines+=['','## 单层必要性结果','',
        '| 层 | ΔNLL | 逐点95%CI | 32项校正95%CI | ΔAcc(pp) | 原对→错 / 原错→对 |',
        '|---|---:|---|---|---:|---|']
    for r in d:lines.append(f"| {r['condition'][5:]} | {r['delta_gold_nll']:+.6f} | [{r['delta_gold_nll_ci_low']:.6f}, {r['delta_gold_nll_ci_high']:.6f}] | [{r['delta_gold_nll_corrected_low']:.6f}, {r['delta_gold_nll_corrected_high']:.6f}] | {r['delta_accuracy_pp']:+.4f} | {r['correct_to_wrong']} / {r['wrong_to_correct']} |")
    top=', '.join(f"{r['condition'][5:]} ({r['delta_gold_nll']:+.6f})" for r in ranked[:5])
    window=[r for r in d if 8<=int(r['condition'][6:])<=11]
    outside=[r for r in d if r not in window]
    lines+=['','## 中文研究结论','',
        f'1. 按平均正向 ΔGold NLL 排名前五为 {top} nat/token。描述性 K80={k80}；该值仅描述单层 Drop 正向均值的集中程度，保留所有负值和原始曲线，不代表视觉信息比例。',
        f"2. 历史 L8–L11 四层内的单层 ΔNLL 均值为 {np.mean([r['delta_gold_nll'] for r in window]):+.6f}，其他层为 {np.mean([r['delta_gold_nll'] for r in outside]):+.6f}。最高单层峰位 {'位于' if ranked[0] in window else '不位于'} L8–L11。这里只做层级分布对照；单层必要性不等同四层联合关闭的效应，更不等同 Only-window 的联合充分性。",
        f"3. 校正区间下界严格大于零的 ΔNLL 层为 {[r['condition'][5:] for r in d if r['delta_gold_nll_corrected_low']>0]}。其余区间跨零不能解释为路径无功能；若峰不集中或幅度很小，应保留路径冗余及跨层替代的可能解释。",
        '4. Accuracy 与 NLL 衡量不同内容。NLL 是固定 Gold 序列的连续分布敏感性；Accuracy 是实际自由生成回答的离散评分。所有正负效应、损害和纠正均见逐题 CSV，不以净得分掩盖真实损害。',
        '5. 三十二层/六任务均报告，不因本轮结果重新选择层或样本。本轮不提供非劣效或实际计算加速结论。','',
        '## 样本重叠与图像隔离附加结果','',
        f"历史探索集 {overlap['historical_questions']} 题；题号重叠 {overlap['question_overlap']}；共享图像簇 {overlap['image_overlap_clusters']}；本轮位于这些图像的题共 {overlap['image_overlap_questions']}。额外图像隔离子集 {len(overlap['image_disjoint_question_ids'])} 题，单独报告在 `image_disjoint_summary.csv`，不得将全 3000 题称作独立外部验证。",
        '| 图像隔离子集条件 | Accuracy | ΔAcc(pp) | ΔGold NLL |', '|---|---:|---:|---:|']
    for r in [disjoint[0],disjoint[1],*sorted(disjoint[2:],key=lambda a:a['delta_gold_nll'],reverse=True)[:5]]:
        lines.append(f"| {r['condition']} | {r['accuracy_pct']:.4f}% | {r['delta_accuracy_pp']:+.4f} | {r['delta_gold_nll']:+.6f} |")
    lines+=['','## 统计、验证与限制','',
        '以 Image SHA256 整簇配对重采样 20,000 次，固定种子；每次重采样后仍按题目加权，不能把 3000 题当成 3000 张独立图片。主 NLL 与配套 Accuracy 各自对 32 个 Drop 比较给出 Bonferroni 家族校正百分位区间；逐点区间为 2.5/97.5%，校正为 0.078125/99.921875%。20,000 次下极端尾部约对应 15.625 个顺序统计量，校正区间存在 Monte Carlo 尾部误差。六任务及图像隔离附加结果为描述性分析。',
        'CPU 测试及两卡冻结 Smoke 通过：同 Logits 形状的 HF/No-op 严格 torch.equal；禁用概率为零；原始视觉和因果前序 Query 输出不变；未修改非目标 Mask；Gold Cache 隔离；正常/异常 Hook 清理。全部逐条件审计见 validation.json 与 checkpoint_hashes.json。',
        f"所有层/条件的最大行和舍入误差为 {validation['maximum_attention_row_error']:.8f}，使用历史阈值 0.002，未放宽；禁止连接概率最大值为 0。",'',
        '输出包括主图 A/B/C 的 PNG/PDF、六任务曲线、完整逐题答案/Token/Gold LogP 与损害案例。没有追加 Only-L 扫描、训练、窗口搜索或物理 KV 删除。单层效应不可相加以论证多层关闭安全。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))


def main():
    began=time.time();samples,records,p=audit();rows=table_rows(samples,records)
    dump_csv('per_question.csv',rows)
    dump_csv('negative_cases.csv',[r for r in rows if r['correct_to_wrong'] or r['delta_gold_nll']>0],fields=list(rows[0]))
    overall=summarize(samples,records,'overall');dump_csv('condition_summary.csv',overall)
    dump_csv('layer_necessity_summary.csv',overall[2:])
    task=[]
    for name in TASKS:task.extend(summarize([s for s in samples if s['task']==name],records,name))
    dump_csv('task_summary.csv',task)
    ids=set(read(OUT/'overlap_audit.json')['image_disjoint_question_ids'])
    disjoint=summarize([s for s in samples if s['question_id'] in ids],records,'image_disjoint')
    dump_csv('image_disjoint_summary.csv',disjoint)
    k80=figures(overall,task);report(overall,disjoint,task,p,k80)
    archive_checkpoints()
    hashes={str(path.relative_to(OUT)):sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name not in ['analysis_complete.json','worker_launch.json']}
    write(OUT/'analysis_complete.json',dict(passed=True,questions=3000,image_clusters=2409,generations=102000,
        rows=len(rows),seconds=time.time()-began,bootstrap_repetitions=20000,hashes=hashes,
        analysis_source_sha256=sha256(Path(__file__))))
    print('ANALYSIS COMPLETE',round(time.time()-began,1),'seconds',flush=True)


if __name__=='__main__':main()
