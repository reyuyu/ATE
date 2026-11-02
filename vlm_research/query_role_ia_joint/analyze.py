"""CPU replay of 240 new joint answers and 720 immutable baseline answers."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from query_role_attention.analyze import paired_bootstrap,summary
from query_role_formal.common import divergence,repeated_trigram
from .common import *

def contrasts(scores):
    f,i,a,j=np.moveaxis(scores[...,:4],-1,0)
    return np.stack([j-f,j-i,j-a,j-i-a+f],axis=-1)

def unique_harm(scores):
    return (scores[...,0]==1)&(scores[...,1]==1)&(scores[...,2]==1)&(scores[...,3]==0)

def replay():
    p=verify();binding=digest(p);audit=audit_history()
    assert audit==read(OUT/'historical_baseline_validation.json')
    assert read(OUT/'gpu_completion.json')['passed'] and read(OUT/'gpu_completion.json')['binding']==binding
    smoke=read(OUT/'gpu_smoke_validation.json');assert smoke['passed'] and smoke['binding']==binding and smoke['questions']==3
    samples=read(OUT/'sample_manifest.json')['samples'];mapping=read(OUT/'input_mappings.json')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    assert len(list((OUT/'checkpoints').glob('*/*.json')))==240
    generations={c:[] for c in CONDITIONS};rows=[];hashes={};raw_hashes={}
    for idx,s in enumerate(samples):
        histories={c:read(checkpoint(s,c)) for c in BASELINES};full=histories['Full']['generation']
        path=OUT/f"checkpoints/{s['task']}__{s['question_id']}/{JOINT}.json";r=read(path);g=r['generation'];layout=mapping[idx]['role_layout'];n=layout['prompt_length']
        assert r['binding']==binding and r['code_commit']==p['code_commit'] and r['result_sha256']==digest(g)
        assert r['sample_sha256']==digest(s) and r['condition']==JOINT and r['gold']==s['answer']
        assert r['input_hashes']==mapping[idx]['input_hashes'] and g['query_layout']==layout
        assert r['prompt_sha256']==s['input_mapping']['chat_prompt_sha256'] and r['image_sha256']==s['image_sha256']
        assert g['text']==tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip()
        assert g['prediction']==gqa_prediction(g['text']) and g['correct']==int(g['prediction']==s['answer'])
        assert g['length']==len(g['token_ids']) and 1<=g['length']<=32
        assert g['own_cache'] and g['own_prefix'] and g['decode_steps']==g['length']-1
        assert g['eos']==(g['token_ids'][-1]==2) and g['truncated']==(g['length']==32 and not g['eos'])
        assert g['empty']==(not bool(g['text'])) and g['repeated_trigram']==repeated_trigram(g['token_ids'])
        assert sha256(r['raw']['path'])==r['raw']['sha256']
        with np.load(r['raw']['path']) as data:logits=torch.from_numpy(data['first_logits'].copy())
        with np.load(histories['Full']['raw']['path']) as data:full_logits=torch.from_numpy(data['first_logits'].copy())
        assert torch.isfinite(logits).all() and int(logits.argmax())==g['token_ids'][0]
        assert all(value==g[key] for key,value in first_metrics(full_logits,logits).items())
        assert g['first_logits_sha256']==digest(logits.tolist()) and g['full_first_logits_sha256']==digest(full_logits.tolist())
        assert g['first_divergence']==divergence(full['token_ids'],g['token_ids'])
        assert g['sequence_changed']==(full['token_ids']!=g['token_ids'])
        assert g['full_correct_to_wrong']==int(full['correct'] and not g['correct'])
        assert g['full_wrong_to_correct']==int(not full['correct'] and g['correct'])
        assert len(g['hook_events'])==4*g['length'] and len(g['cache_lengths'])==g['length']
        for t in range(g['length']):
            expected=dict(t=t,before=n+t-1 if t else 0,after=n+t,consumed_token=g['token_ids'][t-1] if t else None,absolute_query=n+t-1)
            assert g['cache_lengths'][t]==expected
            positions=[0] if t else sorted(layout['I']+layout['A'])
            events=g['hook_events'][4*t:4*t+4];assert [e['layer'] for e in events]==LAYERS
            for e in events:
                assert e['target_count']==len(positions) and e['local_positions_sha256']==digest(positions)
                assert e['phase']==('decode' if t else 'prefill') and e['absolute_start']==(n+t-1 if t else 0)
                assert e['query_length']==(1 if t else n)
                assert e['absolute_first']==(n+t-1 if t else layout['I'][0]) and e['absolute_last']==(n+t-1 if t else n-1)
                assert e['applied'] and e['target_exact_zero'] and e['non_target_rows_exact'] and not e['noop']
        assert g['intervened_query_layer_updates']==4*(576+g['length'])
        generations[JOINT].append(g)
        row=dict(question_id=s['question_id'],task=s['task'],image_id=s['image_id'],image_sha256=s['image_sha256'],gold=s['answer'],question=s['question'],
            prompt_sha256=r['prompt_sha256'],input_ids_sha256=r['input_hashes']['input_ids'],attention_mask_sha256=r['input_hashes']['attention_mask'],pixel_values_sha256=r['input_hashes']['pixel_values'],
            joint_text=g['text'],joint_prediction=g['prediction'],joint_token_ids=json.dumps(g['token_ids']),joint_correct=g['correct'],
            eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],first_top1_flip=g['first_top1_flip'],first_logp_loss=g['first_logp_loss'],
            generated_tokens=g['length'],I_prefill_queries=576,A_prefill_queries=1,R_prefill_queries=layout['counts']['R'],joint_prefill_queries=577,
            decode_queries=g['decode_steps'],total_target_queries=576+g['length'],intervened_query_layer_updates=g['intervened_query_layer_updates'],target_layers=json.dumps(LAYERS),
            baseline_commit=BASE,code_commit=p['code_commit'],binding=binding,raw_logits_path=r['raw']['path'],raw_logits_sha256=r['raw']['sha256'])
        for c,tag in zip(BASELINES,['Full','I','A']):
            h=histories[c]['generation'];generations[c].append(h)
            row.update({tag+'_text':h['text'],tag+'_prediction':h['prediction'],tag+'_token_ids':json.dumps(h['token_ids']),tag+'_correct':h['correct'],
                tag+'_to_joint_harm':int(h['correct'] and not g['correct']),tag+'_to_joint_correction':int(not h['correct'] and g['correct']),
                tag+'_sequence_changed':h['token_ids']!=g['token_ids'],tag+'_first_divergence':divergence(h['token_ids'],g['token_ids']),tag+'_checkpoint_sha256':sha256(checkpoint(s,c))})
        row['joint_unique_harm']=bool(all(histories[c]['generation']['correct'] for c in BASELINES) and not g['correct'])
        rows.append(row);hashes[str(path.relative_to(OUT))]=sha256(path);raw_hashes[r['raw']['path']]=r['raw']['sha256']
    return p,samples,generations,rows,hashes,raw_hashes

def statistics(samples,generations):
    scores=np.array([[generations[c][i]['correct'] for c in CONDITIONS] for i in range(240)],float)
    unique=unique_harm(scores);extended=np.column_stack([scores,unique])
    draws=paired_bootstrap(samples,extended,repetitions=20000,seed=SEED)
    ds=contrasts(draws);observed=100*contrasts(scores).mean(0)
    comparisons=[]
    names=['I+A - Full','I+A - I-Bypass','I+A - A-Bypass','I_role = joint-I-A+Full']
    for k,name in enumerate(names):
        lo,hi=np.percentile(100*ds[:,k],[2.5,97.5])
        comparisons.append(dict(comparison=name,delta_pp=float(observed[k]),ci95_low_pp=float(lo),ci95_high_pp=float(hi),
            baseline_correct_joint_wrong=int(((scores[:,k]==1)&(scores[:,3]==0)).sum()) if k<3 else '',
            baseline_wrong_joint_correct=int(((scores[:,k]==0)&(scores[:,3]==1)).sum()) if k<3 else '',
            questions=240,image_clusters=240,bootstrap_repetitions=20000,seed=SEED,interval='Exploratory paired image-cluster percentile two-sided 95%'))
    summaries=[]
    for k,c in enumerate(CONDITIONS):
        lo,hi=np.percentile(100*draws[:,k],[2.5,97.5])
        summaries.append(dict(condition=c,source='new 240 generations' if c==JOINT else 'reused fixed historical baseline',
            **summary(generations[c],generations['Full']),ci95_accuracy_low_pct=float(lo),ci95_accuracy_high_pct=float(hi),repeated_trigram=sum(g['repeated_trigram'] for g in generations[c])))
    tasks=[]
    for task in TASKS:
        indices=[i for i,s in enumerate(samples) if s['task']==task];assert len(indices)==40
        for c in CONDITIONS:tasks.append(dict(task=task,condition=c,**summary([generations[c][i] for i in indices],[generations['Full'][i] for i in indices])))
    lo,hi=np.percentile(100*draws[:,4],[2.5,97.5])
    unique_summary=dict(count=int(unique.sum()),questions=240,percentage=float(100*unique.mean()),ci95_low_pct=float(lo),ci95_high_pct=float(hi),definition='Full/I/A all correct, joint wrong')
    return summaries,comparisons,tasks,unique_summary

def plots(summaries,comparisons,tasks):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    labels=['Full','I','A','I+A']
    fig,ax=plt.subplots(figsize=(7,4));ys=[r['accuracy_pct'] for r in summaries]
    ax.bar(labels,ys,color=['#486b89','#75a5ad','#a5b5ce','#ce7258'])
    ax.errorbar(range(4),ys,yerr=[[y-r['ci95_accuracy_low_pct'] for y,r in zip(ys,summaries)],[r['ci95_accuracy_high_pct']-y for y,r in zip(ys,summaries)]],fmt='none',color='black',capsize=4)
    for x,y,r in zip(range(4),ys,summaries):ax.text(x,y+1,f"{r['correct']}/240",ha='center')
    ax.set(ylim=(0,100),ylabel='Accuracy (%)',title='Image + answer output bypass: fixed 240 GQA questions')
    fig.tight_layout();savefig(fig,'figure_accuracy')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    ax=axes[0];vals=[r['delta_pp'] for r in comparisons]
    ax.errorbar(vals,range(4),xerr=[[r['delta_pp']-r['ci95_low_pp'] for r in comparisons],[r['ci95_high_pp']-r['delta_pp'] for r in comparisons]],fmt='o',capsize=4,color='#ad543e')
    ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(4),yticklabels=['Joint - Full','Joint - I','Joint - A','Interaction'],xlabel='Percentage points (exploratory 95% CI)');ax.invert_yaxis()
    x=np.arange(3);axes[1].bar(x-.18,[r['baseline_correct_joint_wrong'] for r in comparisons[:3]],.36,label='Correct to wrong',color='#ce7258')
    axes[1].bar(x+.18,[r['baseline_wrong_joint_correct'] for r in comparisons[:3]],.36,label='Wrong to correct',color='#75a5ad')
    axes[1].set(xticks=x,xticklabels=['Full','I','A'],ylabel='Questions',title='Paired outcomes: baseline to joint');axes[1].legend()
    fig.tight_layout();savefig(fig,'figure_paired_interaction')
    fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(6)
    for k,c in enumerate(CONDITIONS):axes[0].bar(x+(k-1.5)*.2,[r['accuracy_pct'] for r in tasks if r['condition']==c],.2,label=labels[k])
    axes[0].set(xticks=x,xticklabels=TASKS,ylim=(0,105),ylabel='Accuracy (%)');axes[0].tick_params(axis='x',rotation=30);axes[0].legend(ncol=4)
    jt=[r for r in tasks if r['condition']==JOINT]
    axes[1].bar(x-.18,[r['correct_to_wrong'] for r in jt],.36,label='Full correct to joint wrong',color='#ce7258')
    axes[1].bar(x+.18,[r['wrong_to_correct'] for r in jt],.36,label='Full wrong to joint correct',color='#75a5ad')
    axes[1].set(xticks=x,xticklabels=TASKS,ylabel='Questions');axes[1].tick_params(axis='x',rotation=30);axes[1].legend(fontsize=8)
    fig.tight_layout();savefig(fig,'figure_tasks')

def savefig(fig,name):
    for suffix in ['png','pdf']:fig.savefig(OUT/f'{name}.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)

def report(p,summaries,comparisons,tasks,unique,rows):
    j=summaries[-1];interaction=comparisons[-1]
    lines=['# 实验四：I+A Joint Bypass——保留关键文本 Query 的联合 Attention 输出消融','',
        '**COMPLETE：新增 240 条 I+A 自由生成；复用固定历史 Full/I/A 各 240 条。Smoke 不计入正式结果。**','',
        '## 最终回答','',
        f"在这批历史复用的 240 题探索样本上，I+A 正确 {j['correct']}/240（{j['accuracy_pct']:.3f}%），相对 Full {comparisons[0]['delta_pp']:+.3f} 个百分点；Full 正确→联合错误 {j['correct_to_wrong']} 题、错误→正确 {j['wrong_to_correct']} 题。联合独有损害 {unique['count']} 题（{unique['percentage']:.3f}%，探索性 95% CI [{unique['ci95_low_pct']:.3f}, {unique['ci95_high_pct']:.3f}]）。交互 joint−I−A+Full 为 {interaction['delta_pp']:+.3f} 个百分点，95% CI [{interaction['ci95_low_pp']:.3f}, {interaction['ci95_high_pp']:.3f}]。",
        '本实验检验的是保留 R 的直接 Attention 更新、同时消融 I 与 A 输出更新是否呈现可组合的低敏感性。任何接近 Full 的净准确率都必须与真实损害、纠正和联合独有损害共同解释；区间跨零不能证明等价、非劣效或没有交互。','',
        '## 四条件结果','', '|条件|来源|正确/240|Accuracy|ΔFull pp|Full 对→错|Full 错→对|EOS|空/截断/重复|','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in summaries:lines.append(f"|{r['condition']}|{'本轮新增' if r['condition']==JOINT else '历史复用'}|{r['correct']}|{r['accuracy_pct']:.3f}%|{r['delta_full_pp']:+.3f}|{r['correct_to_wrong']}|{r['wrong_to_correct']}|{r['eos']}|{r['empty']}/{r['truncated']}/{r['repeated_trigram']}|")
    lines+=['','## 配对比较与交互','', 'Image SHA256 整簇配对 Bootstrap，seed=20261102，20,000 次，所有比较和独有损害比例使用同一组抽样。普通探索性双侧 95% 百分位区间，无多重比较确认性声明。','',
        '|比较|差值 pp|95% CI pp|基线对→联合错|基线错→联合对|','|---|---:|---|---:|---:|']
    for r in comparisons:lines.append(f"|{r['comparison']}|{r['delta_pp']:+.3f}|[{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}]|{r['baseline_correct_joint_wrong']}|{r['baseline_wrong_joint_correct']}|")
    lines+=['','## 三种科学情况的判读','']
    if comparisons[0]['ci95_low_pp']<=0<=comparisons[0]['ci95_high_pp'] and interaction['ci95_low_pp']<=0<=interaction['ci95_high_pp']:
        lines.append('联合相对 Full 的区间及交互区间均跨零；该模式与情况 A 的可组合低敏感性相容，但这里没有预设等价界值，不能把该描述当作等价或非劣效验收，也不能排除实际损害和额外联合效应。')
    elif comparisons[0]['ci95_high_pp']<0 and interaction['ci95_high_pp']<0:
        lines.append('结果显示情况 B 所关心的额外联合退化：净准确率及交互区间均为负。可能的路径依赖或补偿仍只是机制假说，不能归因于某个特定 Token 的视觉记忆修复。')
    else:lines.append('本批结果需结合上述净损益和配对区间判断，不能强行划入安全可组合或明确额外退化。路径依赖或补偿的机制解释尚未被直接检验。')
    lines.append(f"情况 C 所关心的联合独有损害为 {unique['count']}/240；完整问题、图像哈希、Gold、四组答案与 Token 保存在 joint_unique_harm_cases.csv。独有损害与总体净变化分别报告，不以纠正题抵消其存在，也不凭单例推断普遍机制。")
    lines+=['','## 六任务结果','', '|任务|Full|I|A|I+A|联合受损/纠正（相对 Full）|','|---|---:|---:|---:|---:|---|']
    for t in TASKS:
        rr=[next(r for r in tasks if r['task']==t and r['condition']==c) for c in CONDITIONS]
        lines.append('|'+t+'|'+'|'.join(f"{r['accuracy_pct']:.1f}%" for r in rr)+f"|{rr[-1]['correct_to_wrong']}/{rr[-1]['wrong_to_correct']}|")
    lines+=['','## 干预定义与因果边界','',
        '仅在 0-based L2–L5 的 self_attn.o_proj 输出行置零。同一次完整 Prefill 联合处理 576 个原始 Image Query 与原始 P−1；每个真实 Decode Query（含预测 EOS 的步骤）继续置零。每条联合答案从自己的完整 Prefill 开始，不拼接历史 Cache。R 和其他 Query 不被 Hook 直接修改，全部 MLP、Residual、其他层、Attention Mask、视觉来源访问与 KV 更新保持原生执行。',
        '“保留 R”仅指未直接消融 R 输出；I 状态改变后，R 在后续层的输入、Hidden 与 KV 可间接变化。不能声称中继完全不受影响、R 是唯一写入位置或已经确定视觉记忆机制。I/R/A 数量不同，不能把总体 Accuracy 差异解释为单位 Query 贡献。',
        '这是功能输出消融，没有跳过 QKV、Attention 或 KV 存储；未实现或测得真实加速、显存收益。本批 240 题是历史复用的探索样本，不是独立确认集。已有 A-Skip + Text-Only（e005bb5c5cb5e14ca6eff46181c5d238b2ae879c）消融的是直接视觉来源访问，不是本轮 I+A 联合，未混入评分。','',
        '## 冻结、审计与资源','',
        f"历史基准 Commit：`{BASE}`；本轮冻结执行源码 Commit：`{p['code_commit']}`。原 Manifest 字节和输入映射原样复制，实际 SHA256 为 `{MANIFEST_SHA}`。任务书抄写的 59 位字符串不是有效完整 SHA256，沿用历史协议的已验证原文件记录，没有重新采样或改变 Prompt。",
        f"CPU 验收 {read(OUT/'cpu_validation.json')['tests']} 项通过。GPU 在三道固定题上验证 HF/Full/No-op 全部返回有效 Logits 的 torch.equal、新旧 Full/I/A 全步 Logits 精确一致、已保存历史首 Token 数组与完整 Token 一致；显式 logits_to_keep=1，不放宽容差。联合目标、R 不直接修改、32 层 MLP、独立 Cache 及真实位置、输入/参数版本、正常和异常 Hook 清理通过。",
        '720 条历史记录在正式前与分析时重验输入、评分、检查点、完整交付文件及实际服务器原始首 Token 数组 SHA256；240 条新增记录从 Token、文本、评分、原始首 Token、Hook 与 Cache 事件回放。所有首 Token 指标严格相等；模型文件完整 SHA256 在冻结前和采集后校验。',
        f"仅使用物理 GPU{p['gpu']['physical_index']}（UUID `{p['gpu']['uuid']}`），只暴露该卡为 cuda:0，一份模型串行采集。GPU0/1 上的其他进程未被终止或干预。进程退出后才生成 GPU 释放审计；实时占用以新检查为准。",
        '原始数组位于服务器资源目录，路径和 SHA256 绑定在检查点和 validation.json；数组未上传 GitHub。Smoke 和必要回放不计入新增 240 条正式答案。','',
        '## 下一步建议与停止边界','',
        '下一项建议是在预声明可接受损害界值和样本量后，用新的独立确认集复核 I+A 的净损害、真实受损率和联合独有损害；这仅为建议，本轮不设计或执行下一轮。未扩样、训练 Router、执行 Qwen、开发 Kernel 或剪枝 KV。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))

def main(check_only=False):
    p,samples,generations,rows,hashes,raw_hashes=replay()
    summaries,comparisons,tasks,unique=statistics(samples,generations)
    if check_only:
        import csv
        for name,expected in [('accuracy_summary.csv',summaries),('paired_comparisons.csv',comparisons),('task_accuracy.csv',tasks),('per_question.csv',rows)]:
            with (OUT/name).open(newline='') as f:saved=list(csv.DictReader(f))
            assert len(saved)==len(expected)
            for a,b in zip(expected,saved):
                for k,v in a.items():
                    if isinstance(v,(int,float,np.number)) and not isinstance(v,(bool,np.bool_)):assert abs(float(b[k])-float(v))<1e-12,(name,k)
                    else:assert b[k]==('' if v is None else str(v)),(name,k)
        complete=read(OUT/'completion.json');assert complete['complete']
        for name,h in complete['artifacts'].items():assert sha256(OUT/name)==h,name
        assert hashes==read(OUT/'validation.json')['checkpoint_hashes'] and raw_hashes==read(OUT/'validation.json')['raw_hashes']
        assert not torch.cuda.is_initialized();print('READ-ONLY REPLAY PASS: all saved metrics, per-question rows, 240 checkpoints, raw arrays and artifact hashes',flush=True);return
    for name,data in [('accuracy_summary.csv',summaries),('paired_comparisons.csv',comparisons),('task_accuracy.csv',tasks),('per_question.csv',rows),
        ('accuracy_harm_cases.csv',[r for r in rows if r['Full_to_joint_harm']]),('joint_unique_harm_cases.csv',[r for r in rows if r['joint_unique_harm']]),
        ('all_paired_harm_cases.csv',[r for r in rows if any(r[t+'_to_joint_harm'] for t in ['Full','I','A'])])]:
        if data:csv_write(OUT/name,data)
        else:
            import csv
            with (OUT/name).open('w',newline='') as f:csv.DictWriter(f,fieldnames=list(rows[0])).writeheader()
    write(OUT/'joint_unique_harm_summary.json',unique);plots(summaries,comparisons,tasks)
    env=read(OUT/'environment.json');assert not Path(f"/proc/{env['pid']}").exists()
    state=hardware();uuid=p['gpu']['uuid'];apps=[line for line in state['compute_processes'].splitlines() if line.split(',')[0].strip()==uuid]
    write(OUT/'gpu_release.json',dict(our_process_exited=True,pid=env['pid'],gpu=p['gpu'],hardware=state,selected_gpu_currently_idle=not bool(apps),no_external_process_terminated=True))
    for path,info in p['model_files'].items():assert sha256(path)==info['sha256'],path
    verify();report(p,summaries,comparisons,tasks,unique,rows)
    write(OUT/'validation.json',dict(passed=True,new_formal_generations=240,reused_baseline_generations=720,questions=240,image_clusters=240,
        token_text_score_replay_exact=True,first_token_metric_max_abs_difference=0.,raw_logits_verified=240,history_raw_logits_verified=720,
        model_full_sha256_unchanged=True,historical_artifacts_unchanged=True,checkpoint_hashes=hashes,raw_hashes=raw_hashes,
        sources=p['sources'],gpu_process_exited=True,cuda_initialized_during_analysis=torch.cuda.is_initialized(),raw_arrays_on_github=False,unique_harm=unique))
    assert not torch.cuda.is_initialized()
    artifacts={str(f.relative_to(OUT)):sha256(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['worker.lock','completion.json']}
    write(OUT/'completion.json',dict(status='COMPLETE',complete=True,new_formal_generations=240,reused_baseline_generations=720,questions=240,
        frozen_code_commit=p['code_commit'],binding=digest(p),artifacts=artifacts))
    print(json.dumps(dict(summaries=summaries,comparisons=comparisons,unique_harm=unique),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    try:main(args.check)
    except BaseException:
        if not args.check:
            write(OUT/'failures'/f'analysis_{time.time_ns()}.json',dict(status='PARTIAL',traceback=traceback.format_exc()))
            write(OUT/'completion.json',dict(status='PARTIAL',complete=False))
        raise
