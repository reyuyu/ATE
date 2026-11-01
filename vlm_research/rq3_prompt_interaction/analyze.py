"""Paired image-cluster interaction, descriptive states, and fixed figures."""
from collections import Counter
import csv
import itertools
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from benchmarks.dataset_registry import digest,sha256
from knockout_probe import tensor_hash,ROW_ATOL
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_single_layer_gqa6.analyze import descriptive_correlations
from paper_rq1_early_joint_attention.statistics import bootstrap
from .engine import ALL,CHECKPOINTS,state_metrics
from .run import OUT,ROOT,SEED,TASKS,read,verify,history,historic_logits,checkpoint,causal_states,old,atomic_json

KEYS=['F','E','M','EM','L']
LABELS=dict(F='Full',E='E-OFF',M='M-OFF',EM='E+M-OFF',L='Late-OFF')
COMPARISONS=['Interaction','E-F','M-F','EM-F','EM-M','L-F','EM-E']


def contrasts(scores):
    f,e,m,em,late=np.asarray(scores,dtype=float).T
    return 100*np.column_stack([em-e-m+f,e-f,m-f,em-f,em-m,late-f,em-e])


def anomaly(sample,g):
    text=g['text'];normal=' '.join(text.lower().split());question=' '.join(sample['question'].lower().split())
    ids=g['token_ids'];words=re.findall(r'[a-z0-9]+',normal)
    repeated_ids=any(a==b==c for a,b,c in zip(ids,ids[1:],ids[2:]))
    repeated_words=any(a==b==c for a,b,c in zip(words,words[1:],words[2:]))
    return dict(empty=not bool(text.strip()),truncated=g['truncated'],question_echo=bool(question and question in normal),
        repeated_text=repeated_ids or repeated_words,
        abnormal_format=bool(text.strip()) and (not bool(words) or 'user:' in normal or 'assistant:' in normal))


def write_csv(name,rows,c):
    binding=dict(protocol_sha256=digest(c),model_revision=c['model_revision'],code_commit=c['code_commit'],
                 sample_manifest_sha256=c['bindings']['sample_manifest.json'])
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[*rows[0],*binding],lineterminator='\n');writer.writeheader()
        writer.writerows(dict(r,**binding) for r in rows)


def collect():
    c=verify();binding=digest(c);manifest=read(OUT/'sample_manifest.json');samples=manifest['samples']
    assert len(samples)==240 and len({s['image_sha256'] for s in samples})==235
    assert samples==read(old.OUT/'sample_manifest.json')['samples']
    completions=[read(OUT/f'shard{k}/completion.json') for k in range(len(manifest['gpus']))]
    assert all(r['status']=='complete' and r['binding']==binding and r['hooks_clean'] and r['parameters_unchanged'] for r in completions)
    assert sum(r['generations'] for r in completions)==720
    rows=[];states_rows=[];attention=[];scores=[];used=set();image_ranks={}
    for s in samples:
        qid=s['question_id'];rank=manifest['assignments'][qid];folder=OUT/f'shard{rank}';stem=f"{s['task']}__{qid}"
        assert image_ranks.setdefault(s['image_sha256'],rank)==rank
        baseline=read(folder/f'baseline/{stem}.json');assert baseline['binding']==binding and baseline['sample_sha256']==digest(s)
        assert baseline['F_logits_exact'] and baseline['M_logits_exact'] and sha256(OUT/baseline['raw_file'])==baseline['raw_sha256']
        with np.load(OUT/baseline['raw_file'],allow_pickle=False) as data:
            fh=torch.from_numpy(data['F'].copy());mh=torch.from_numpy(data['M'].copy())
        assert tensor_hash(fh)==baseline['F_states_sha256'] and tensor_hash(mh)==baseline['M_states_sha256']
        full=historic_logits(s,'F');middle=historic_logits(s,'M');fg=history(s,'F')['generation']
        assert baseline['F_logits_sha256']==tensor_hash(full) and baseline['M_logits_sha256']==tensor_hash(middle)
        per_scores=[]
        for key in KEYS:
            if key in ['F','M']:
                record=history(s,key);g=record['generation'];first=full if key=='F' else middle
                h=fh if key=='F' else mh;arows=baseline[key+'_rows'];metric=metric_rows(full,first[None])[0];metric['logp_loss']=-metric['delta_logp']
                if key=='M':assert metric==record['first_token']
                seconds=0.;gpu=manifest['gpus'][rank]['physical_gpu']
            else:
                path=folder/f'checkpoints/{stem}/{key}.json';assert path not in used;used.add(path)
                r=checkpoint(path,binding,s,key);assert r['question_id']==qid and r['task']==s['task'] and r['image_sha256']==s['image_sha256']
                assert all(r[name] for name in ['hooks_clean','inputs_unchanged','parameters_unchanged'])
                assert r['physical_gpu']==manifest['gpus'][rank]['physical_gpu'] and r['physical_gpu']!=1
                with np.load(OUT/r['raw_file'],allow_pickle=False) as data:
                    first=torch.from_numpy(data['first_logits'].copy());h=torch.from_numpy(data['states'].copy())
                assert first.dtype==torch.float32 and torch.isfinite(first).all() and first.shape==full.shape
                assert tensor_hash(h)==r['states_sha256']
                metric=metric_rows(full,first[None])[0];metric['logp_loss']=-metric['delta_logp'];assert metric==r['first_token']
                g=r['generation'];assert g['masked_layers']==ALL[key] and g['mask_calls']==32
                assert all(g[name] for name in ['own_prefill','own_prefix','decode_native'])
                arows=r['attention_rows'];assert digest(arows)==r['attention_sha256']
                seconds=r['seconds'];gpu=r['physical_gpu']
            ids=g['token_ids'];assert ids and len(ids)<=32 and 2 not in ids[:-1]
            assert ids[0]==int(first.argmax()) and g['length']==len(ids) and g['eos']==(ids[-1]==2)
            assert g['truncated']==(len(ids)==32 and ids[-1]!=2) and g['prediction']==gqa_prediction(g['text'])
            assert g['score']==int(g['prediction']==s['answer']) and h.shape==(6,4096) and torch.isfinite(h).all()
            causal_states(fh,h,key);assert [a['layer'] for a in arows]==list(range(32))
            assert all(arows[layer]['Prompt_max']==0 for layer in ALL[key])
            flags=anomaly(s,g);content_length=len(ids)-int(g['eos'])
            rows.append(dict(task=s['task'],question_id=qid,image_id=s['image_id'],image_sha256=s['image_sha256'],sample_sha256=digest(s),
                condition=key,answer=s['answer'],question=s['question'],text=g['text'],prediction=g['prediction'],score=g['score'],full_score=fg['score'],
                harm=fg['score']==1 and g['score']==0,fix=fg['score']==0 and g['score']==1,token_ids=json.dumps(ids),length=len(ids),
                eos=g['eos'],non_eos_length=content_length,first_top1_flip=bool(metric['top1_flip']),signed_logp_loss=metric['logp_loss'],
                absolute_logp_change=abs(metric['delta_logp']),js=metric['js'],first_token=ids[0],full_first_token=int(full.argmax()),
                new_generation=key not in ['F','M'],seconds=seconds,physical_gpu=gpu,**flags))
            per_scores.append(g['score'])
            for i,layer in enumerate(CHECKPOINTS):
                states_rows.append(dict(task=s['task'],question_id=qid,image_sha256=s['image_sha256'],sample_sha256=digest(s),
                    condition=key,layer_input=layer,signed_logp_loss=metric['logp_loss'],**state_metrics(fh[i],h[i])))
            for layer,a in enumerate(arows):
                assert a['row_error']<=ROW_ATOL and abs(sum(a[k] for k in ['Image','Prompt','BOS','Self'])-1)<=ROW_ATOL
                attention.append(dict(task=s['task'],question_id=qid,image_sha256=s['image_sha256'],condition=key,layer=layer,
                    blocked=layer in ALL[key],row_error=a['row_error'],**{source:a[source] for source in ['Image','Prompt','BOS','Self']}))
        scores.append(per_scores)
    assert used==set(OUT.glob('shard*/checkpoints/*/*.json')) and len(used)==720
    return c,manifest,completions,rows,states_rows,attention,np.asarray(scores,dtype=float)


def plot(summary,tasks,hidden,c):
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
    fig,(ax,hmap)=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'width_ratios':[1,1.25]},layout='constrained')
    colors={'F':'#666666','E':'#287da3','M':'#5f6594','EM':'#bb4b42','L':'#258574'}
    for i,r in enumerate(summary):
        ax.errorbar(i,r['delta_pp'],yerr=[[r['delta_pp']-r['ci95_low']],[r['ci95_high']-r['delta_pp']]],fmt='o',
            color=colors[r['condition']],capsize=3,markersize=6)
    ax.axhline(0,color='#666666',ls='--',lw=.8);ax.set_xticks(range(5),[LABELS[k] for k in KEYS],rotation=20)
    ax.set_ylabel('Accuracy change vs. Full (pp)');ax.set_title('(a) Paired accuracy',loc='left');ax.spines[['top','right']].set_visible(False)
    vals=np.array([[next(r['delta_pp'] for r in tasks if r['task']==t and r['condition']==k) for k in KEYS[1:]] for t in TASKS])
    vmax=max(1.,float(np.abs(vals).max()));im=hmap.imshow(vals,cmap='RdBu',norm=TwoSlopeNorm(0,vmin=-vmax,vmax=vmax),aspect='auto')
    for i,j in itertools.product(range(6),range(4)):
        hmap.text(j,i,f'{vals[i,j]:+.1f}',ha='center',va='center',fontsize=8,color='white' if abs(vals[i,j])>.6*vmax else '#222222')
    hmap.set_xticks(range(4),[LABELS[k] for k in KEYS[1:]],rotation=20);hmap.set_yticks(range(6),TASKS)
    hmap.set_title('(b) Task differences',loc='left');fig.colorbar(im,ax=hmap,label='Accuracy change (pp)',fraction=.045,pad=.03)
    fig.suptitle('Prompt Attention: Accumulation and Cross-Layer Interaction',fontweight='bold',fontsize=11)
    for ext in ['png','pdf']:
        metadata={'Protocol':digest(c)} if ext=='png' else {'Subject':'Protocol SHA256 '+digest(c)}
        fig.savefig(OUT/f'prompt_cross_layer_interaction.{ext}',dpi=300,bbox_inches='tight',metadata=metadata)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for key in KEYS[1:]:
        sub=[r for r in hidden if r['condition']==key]
        for ax,metric,label in zip(axes,['mean_relative_l2_distance','mean_cosine_distance'],['Relative L2 distance','Cosine distance']):
            ax.plot(CHECKPOINTS,[r[metric] for r in sub],marker='o',lw=1.3,label=LABELS[key],color=colors[key])
            ax.set_xticks(CHECKPOINTS);ax.set_xlabel('Decoder layer input');ax.set_ylabel(label);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8);fig.suptitle('P-1 State Propagation',fontsize=11,fontweight='bold')
    for ext in ['png','pdf']:fig.savefig(OUT/f'p1_state_propagation.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    c,manifest,completions,rows,states,attention,scores=collect();samples=manifest['samples'];identities=[s['image_sha256'] for s in samples]
    estimates=bootstrap(contrasts(scores),identities,[1,6,6,6,6,6,6],repetitions=c['statistics']['repetitions'],seed=SEED)
    paired=[dict(comparison=name,role='primary exploratory interaction' if name=='Interaction' else 'secondary exploratory',**r) for name,r in zip(COMPARISONS,estimates)]
    lookup={r['comparison']:r for r in paired};summary=[];tasks=[];first_summary=[]
    for key in KEYS:
        sub=[r for r in rows if r['condition']==key];d=lookup.get(key+'-F')
        summary.append(dict(condition=key,questions=240,correct=sum(r['score'] for r in sub),accuracy_percent=float(np.mean([r['score'] for r in sub])*100),
            delta_pp=d['mean'] if d else 0.,ci95_low=d['ci95_low'] if d else 0.,ci95_high=d['ci95_high'] if d else 0.,
            harm=sum(r['harm'] for r in sub),fix=sum(r['fix'] for r in sub),eos=sum(r['eos'] for r in sub),
            mean_length=float(np.mean([r['length'] for r in sub])),mean_non_eos_length=float(np.mean([r['non_eos_length'] for r in sub])),
            empty=sum(r['empty'] for r in sub),truncated=sum(r['truncated'] for r in sub),
            question_echo=sum(r['question_echo'] for r in sub),repeated_text=sum(r['repeated_text'] for r in sub),abnormal_format=sum(r['abnormal_format'] for r in sub)))
        for task in ['All',*TASKS]:
            tr=[r for r in sub if task=='All' or r['task']==task]
            first_summary.append(dict(task=task,condition=key,questions=len(tr),mean_signed_logp_loss=float(np.mean([r['signed_logp_loss'] for r in tr])),
                mean_absolute_logp_change=float(np.mean([r['absolute_logp_change'] for r in tr])),top1_flips=sum(r['first_top1_flip'] for r in tr),
                top1_flip_rate=float(np.mean([r['first_top1_flip'] for r in tr])),mean_js=float(np.mean([r['js'] for r in tr]))))
            if task!='All':tasks.append(dict(task=task,condition=key,questions=len(tr),correct=sum(r['score'] for r in tr),
                accuracy_percent=float(np.mean([r['score'] for r in tr])*100),delta_pp=float(np.mean([r['score']-r['full_score'] for r in tr])*100),
                harm=sum(r['harm'] for r in tr),fix=sum(r['fix'] for r in tr)))
    overlaps=[]
    for s in samples:
        by={r['condition']:r for r in rows if r['question_id']==s['question_id']};item=dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'])
        for k in KEYS:
            item[k+'_score']=by[k]['score'];item[k+'_harm']=by[k]['harm'];item[k+'_fix']=by[k]['fix']
        item.update(E_and_M_harm=by['E']['harm'] and by['M']['harm'],EM_only_harm=by['EM']['harm'] and not by['E']['harm'] and not by['M']['harm'],
            per_question_interaction_pp=100*(by['EM']['score']-by['E']['score']-by['M']['score']+by['F']['score']))
        overlaps.append(item)
    hidden=[];correlations=[]
    for key,layer in itertools.product(KEYS,CHECKPOINTS):
        sub=[r for r in states if r['condition']==key and r['layer_input']==layer];r=dict(condition=key,layer_input=layer,questions=len(sub))
        for metric in ['cosine_distance','norm_change','relative_norm_change','l2_distance','relative_l2_distance']:
            values=[a[metric] for a in sub if a[metric] is not None]
            r['mean_'+metric]=float(np.mean(values)) if values else None;r['std_'+metric]=float(np.std(values,ddof=1)) if len(values)>1 else None
        hidden.append(r)
        p,sp=descriptive_correlations([a['relative_l2_distance'] for a in sub],[a['signed_logp_loss'] for a in sub])
        correlations.append(dict(condition=key,layer_input=layer,questions=len(sub),pearson=p,spearman=sp,scope='Descriptive paired questions; no p values or causal interpretation'))
    for name,data in [('per_question.csv',rows),('accuracy_summary.csv',summary),('task_accuracy.csv',tasks),('first_token_summary.csv',first_summary),
        ('paired_comparisons.csv',paired),('interaction_summary.csv',[paired[0]]),('error_transition_overlap.csv',overlaps),('generation_anomalies.csv',rows),
        ('hidden_state_propagation.csv',states),('hidden_state_summary.csv',hidden),('hidden_loss_correlations.csv',correlations),('attention_source_mass.csv',attention)]:write_csv(name,data,c)
    table,apps=old.snapshot();pids={str(r['preflight']['pid']) for r in completions}
    assert not any(len(parts:=line.split(', '))>1 and parts[1] in pids for line in apps.splitlines()),'Worker process still present'
    atomic_json(OUT/'gpu_release.json',dict(binding=digest(c),snapshot=table,compute_processes=apps,selected_gpus=manifest['gpus'],
        worker_pids=sorted(pids),worker_compute_processes_absent=True,gpu1_not_used=True))
    elapsed=max(r['finished_epoch'] for r in completions)-min(r['started_epoch'] for r in completions)
    atomic_json(OUT/'hidden_state_audit.json',dict(binding=digest(c),checkpoints=CHECKPOINTS,rows=len(states),F_baselines=240,M_prefills=240,
        new_condition_states=720,all_finite=True,causal_input_invariants_bitwise_exact=True,read_only_smoke_all_logits_exact=True,
        saved_only_original_P1_six_input_vectors=True,full_sequence_states_saved=False))
    atomic_json(OUT/'validation.json',dict(binding=digest(c),status='complete',questions=240,image_clusters=235,new_complete_generations=720,
        reused_F=240,reused_M=240,F_and_M_first_logits_exact_each=240,raw_new_first_metric_replays=720,whole_image_shards=True,
        cpu_tests=read(OUT/'cpu_validation.json')['tests'],all_gpu_smokes_passed=True,other_prefill_queries_exact_in_smoke=True,
        all_target_prompt_probabilities_zero=True,image_bos_self_preserved=True,all_decode_native=True,own_prefix_and_cache=True,
        no_nan_inf_oom=True,no_hook_leaks=True,no_input_weight_changes=True,all240_scored_without_exclusion=True,
        formal_wall_seconds=elapsed,gpus=manifest['gpus'],gpu1_not_used=True,code_and_model_verified=True))
    plot(summary,tasks,hidden,c)
    report(summary,tasks,first_summary,paired,hidden,elapsed,c)
    hashes={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json','worker.lock']}
    atomic_json(OUT/'artifact_hashes.json',dict(binding=digest(c),files=hashes))
    print(json.dumps(dict(accuracy=summary,interaction=paired[0],paired=paired,hidden=hidden,formal_minutes=elapsed/60),ensure_ascii=False,indent=2))


def report(summary,tasks,first,paired,hidden,elapsed,c):
    by={r['condition']:r for r in summary};p={r['comparison']:r for r in paired};interaction=p['Interaction']
    def ci(r):return f"{r['mean']:+.3f}pp，95%CI [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
    lines=['# RQ3.1：Prompt Attention 跨层累积与交互', '',
        '同一批RQ2.5的240题、235个Image SHA256簇，六任务各40题；不是独立确认集。F与M完整答案严格绑定复用，只新增E/EM/L三个条件720次完整生成。',
        'E=L2–L12，M=L13–L16，EM=L2–L16，Late=L17–L31；L1原生。仅原始P−1→Prompt使用Softmax前−inf Hard Mask，其余Prefill Query及全部Decode原生。',
        '固定LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager/Transformers4.49.0，Greedy、32token、EOS停止、原生KV Cache与官方GQA评分。','',
        '|条件|正确/240|Accuracy|ΔFull (pp)，配对95%CI|对→错|错→对|', '|---|---:|---:|---|---:|---:|']
    for r in summary:lines.append(f"|{LABELS[r['condition']]}|{r['correct']}|{r['accuracy_percent']:.3f}%|{r['delta_pp']:+.3f} [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]|{r['harm']}|{r['fix']}|")
    lines+=['', '## 主交互及研究问题', '',
        f"**主要交互 EM−E−M+F：{ci(interaction)}。** 同一次图像簇重采样中使用全部条件的相同权重；不是独立区间拼接。",
        f"Q1：E联合关闭的完整答案变化为{ci(p['E-F'])}。不能将单层低平均LogP Loss自动当成联合删除安全保证。",
        f"Q2：Accuracy非加性交互见上述主要估计；负值为额外联合损害，正值为联合损害较简单加和更小。CI包含0不等于可加性已证明，更不是语义信息流证明。",
        f"Q3：在E已关闭时继续删除M，EM−E为{ci(p['EM-E'])}；单删M−F为{ci(p['M-F'])}。这两个条件效应之差严格等于主要交互。EM−M（M背景上再删E）为{ci(p['EM-M'])}。",
        f"Q4：Late−F为{ci(p['L-F'])}，代表后层直接读取路径干预与最终预测的配对差异，不隔离其语义作用。",
        'Q5：六类任务的完整Accuracy及翻转见task_accuracy.csv和主图；每任务仅40题，不作未经校正的任务差异推断。',
        'Q6：RQ2.1的单层低均值来自另一批发现240题，且是固定Full首Token LogP端点，本轮是完整答案Accuracy。不能跨样本、跨端点把单层数值相加预测联合表现；本轮也没有训练或评估逐题预测器。',
        'Q7：六个固定层输入处P−1状态只读采集成功。E/EM与Full在L2输入精确一致；M到L13输入精确一致；Late到L17输入精确一致。其后距离见hidden_state_summary.csv及补充图，距离不是问题语义量。',
        'Q8：残差状态跨层传播、后层直接重读与Hard Mask重归一化均可能参与结果；当前联合消融不能唯一辨别三者。未做Activation Patching，不能宣布证明稳定语义信息流。', '',
        '## 统计边界与异常', '',
        '全图像SHA256整簇配对Bootstrap100000次，seed20261101，保持题目加权；主交互在GPU前冻结，但这些样本已经使用，仍属机制探索。其余六项比较给出普通95%及事先固定Bonferroni6同时参考区间，不作非劣效、等价或独立确认宣称。',
        'generation_anomalies.csv为全部1200条F/E/M/EM/L结果的预声明启发式标志（空、截断、完整问题回显、三连重复、异常格式）；标志不是新评分器，不剔除任何题。',
        '；'.join(f"{k}：EOS {by[k]['eos']}，空{by[k]['empty']}，截断{by[k]['truncated']}，问题回显{by[k]['question_echo']}，重复{by[k]['repeated_text']}，格式标志{by[k]['abnormal_format']}" for k in KEYS)+'。',
        '首Token Signed LogP Loss固定在Full原生Top1 Token，不是Gold Loss；Mean Absolute LogP Change为逐题绝对值的均值，不是均值的绝对值。内容Token长度排除末尾EOS，EOS与非EOS输出均保存。', '',
        '## 验证与资源', '',
        f"CPU {read(OUT/'cpu_validation.json')['tests']}项测试和每张GPU两题Smoke通过：No-op精确、历史F/M生成和首Token精确、三新条件的目标Mask/其他Query、独立HF Decode、只读状态不改Logits、正常/异常清理。正式每题F及M首Token逐位等于历史；720份新原始FP32向量重算指标一致。",
        f"正式运行{elapsed/60:.2f}分钟，使用物理GPU {[g['physical_gpu'] for g in c['gpus']]}，未使用GPU1。模型进程已退出，峰值显存见各completion.json；释放快照见gpu_release.json。",
        '源码、模型、Tokenizer文件、输入、历史记录、分片和统计绑定见protocol.json、sample_manifest.json、history_bindings.json与各CSV的绑定列。未覆盖旧实验，未追加条件、训练、Decode消融、状态替换或更大样本。本轮完成即停止。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
