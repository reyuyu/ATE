"""One final merge of frozen five-condition checkpoints; no new GPU work."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.dataset_registry import digest,sha256
from knockout_probe import tensor_hash,ROW_ATOL
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_early_joint_attention.statistics import bootstrap
from paper_rq1_single_layer_gqa6.run import atomic_json
from .engine import CONDITIONS,LABELS,S
from .run import OUT,SEED,TASKS,read,verify,checkpoint_valid,historical_full,validation,snapshot

KEYS=['Full',*CONDITIONS]
CONTRASTS=['A-Full','B-Full','C-Full','D-Full','E-Full','C-A','C-B','C-A-B+Full']


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def contrast_values(scores):
    f,a,b,c,d,e=scores.T
    return 100*np.column_stack([a-f,b-f,c-f,d-f,e-f,c-a,c-b,c-a-b+f])


def transitions(full,changed,ids):
    return dict(harm=[qid for qid,f,c in zip(ids,full,changed) if f==1 and c==0],
                fix=[qid for qid,f,c in zip(ids,full,changed) if f==0 and c==1])


def collect():
    protocol=verify();binding=digest(protocol);manifest=read(OUT/'sample_manifest.json')
    samples=manifest['samples'];assert len(samples)==240 and len({s['question_id'] for s in samples})==240
    completions=[read(OUT/f'shard{k}/completion.json') for k in range(len(manifest['gpus']))]
    assert sum(c['generations'] for c in completions)==1200
    assert all(c['status']=='complete' and c['binding']==binding and c['hooks_clean'] and c['parameters_unchanged'] for c in completions)
    rows=[];mass=[];scores=[];used=set();image_rank={}
    for s in samples:
        qid=s['question_id'];rank=manifest['assignments'][qid]
        assert image_rank.setdefault(s['image_sha256'],rank)==rank
        folder=OUT/f'shard{rank}';stem=f"{s['task']}__{qid}"
        baseline=read(folder/f'baseline/{stem}.json')
        full=validation.native_logits(s,True)
        assert baseline['binding']==binding and baseline['sample_sha256']==digest(s) and baseline['logits_exact_history']
        assert baseline['full_logits_sha256']==tensor_hash(full)
        native_rows=baseline['full_native_rows'];assert [r['layer'] for r in native_rows]==list(range(32))
        sample_scores=[]
        for key in KEYS:
            if key=='Full':
                generation=historical_full(s);metrics=metric_rows(full,full[None])[0];metrics['logp_loss']=0.
                attention=native_rows;seconds=0.;physical_gpu=manifest['gpus'][rank]['physical_gpu']
            else:
                path=folder/f'checkpoints/{stem}/{key}.json';assert path not in used;used.add(path)
                r=checkpoint_valid(path,binding,s,key);generation=r['generation'];metrics=r['first_token']
                assert all(r[name] for name in ['hooks_clean','inputs_unchanged','parameters_unchanged'])
                assert r['question_id']==qid and r['task']==s['task'] and r['image_sha256']==s['image_sha256']
                assert r['physical_gpu']==manifest['gpus'][rank]['physical_gpu'] and r['physical_gpu']!=1
                with np.load(OUT/r['logits_file'],allow_pickle=False) as data:
                    first=torch.from_numpy(data['first_logits'].copy())
                assert first.dtype==torch.float32 and torch.isfinite(first).all() and first.shape==full.shape
                replay=metric_rows(full,first[None])[0];replay['logp_loss']=-replay['delta_logp']
                assert replay==metrics and int(first.argmax())==generation['token_ids'][0]
                assert generation['masked_layers']==CONDITIONS[key] and generation['mask_calls']==32
                assert all(generation[name] for name in ['own_prefill','own_prefix','decode_native'])
                attention=r['native_attention_rows'];assert digest(attention)==r['attention_sha256']
                assert [a['layer'] for a in attention]==list(range(32))
                assert all(attention[k]['Prompt_max']==0 for k in CONDITIONS[key])
                seconds=r['seconds'];physical_gpu=r['physical_gpu']
            ids=generation['token_ids'];assert ids and len(ids)<=32 and 2 not in ids[:-1]
            assert generation['length']==len(ids) and generation['eos']==(ids[-1]==2)
            assert generation['truncated']==(len(ids)==32 and ids[-1]!=2)
            assert generation['prediction']==gqa_prediction(generation['text'])
            assert generation['score']==int(generation['prediction']==s['answer'])
            assert int(full.argmax())==historical_full(s)['token_ids'][0]
            sample_scores.append(generation['score'])
            rows.append(dict(task=s['task'],question_id=qid,image_id=s['image_id'],image_sha256=s['image_sha256'],
                condition=key,label=LABELS[key],answer=s['answer'],text=generation['text'],prediction=generation['prediction'],
                score=generation['score'],full_score=historical_full(s)['score'],accuracy_change=generation['score']-historical_full(s)['score'],
                token_ids=json.dumps(ids),length=len(ids),eos=generation['eos'],truncated=generation['truncated'],
                empty=not bool(generation['text'].strip()),full_top1_token=int(full.argmax()),
                first_token=ids[0],first_top1_flip=bool(metrics['top1_flip']),first_logp_loss=metrics['logp_loss'],
                js=metrics['js'],physical_gpu=physical_gpu,new_generation=key!='Full',seconds=seconds))
            for layer,(base,actual) in enumerate(zip(native_rows,attention)):
                assert actual['row_error']<=ROW_ATOL
                assert abs(sum(actual[k] for k in ['Image','Prompt','BOS','Self'])-1)<=ROW_ATOL
                item=dict(task=s['task'],question_id=qid,image_sha256=s['image_sha256'],condition=key,layer=layer,
                          prompt_blocked=key!='Full' and layer in CONDITIONS[key],row_error=actual['row_error'])
                for source in ['Image','Prompt','BOS','Self']:
                    item[source]=actual[source];item['full_'+source]=base[source];item['delta_'+source]=actual[source]-base[source]
                mass.append(item)
        scores.append(sample_scores)
    actual=set(OUT.glob('shard*/checkpoints/*/*.json'));assert used==actual and len(used)==1200
    return protocol,manifest,completions,rows,mass,np.array(scores,dtype=float)


def plot(summary,taskrows,output):
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
    fig,(ax,heat)=plt.subplots(2,1,figsize=(8,6.3),gridspec_kw={'height_ratios':[1.1,1]},layout='constrained')
    x=np.arange(6);colors=['#666666','#4787a6','#4787a6','#be4d4a','#258572','#4787a6']
    for i,r in enumerate(summary):
        ax.errorbar(i,r['delta_pp'],yerr=[[r['delta_pp']-r['ci95_low']],[r['ci95_high']-r['delta_pp']]],
                    fmt='o',color=colors[i],capsize=4,markersize=6)
        ax.annotate(f"{r['accuracy_percent']:.2f}%",(i,r['delta_pp']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
    ax.axhline(0,color='#555555',lw=.8,ls='--');ax.set_xticks(x,[LABELS[k] for k in KEYS],rotation=12)
    ax.set_ylabel('Accuracy change vs. Full (pp)');ax.set_title('Prompt Attention: Necessity vs. Sufficiency',fontweight='bold')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    vals=np.array([[next(r['accuracy_percent'] for r in taskrows if r['task']==t and r['condition']==k) for k in KEYS] for t in TASKS])
    im=heat.imshow(vals,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    for i in range(6):
        for j in range(6):heat.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',color='white' if vals[i,j]>70 else '#222222',fontsize=8)
    heat.set_yticks(range(6),TASKS);heat.set_xticks(x,[LABELS[k] for k in KEYS],rotation=12)
    fig.colorbar(im,ax=heat,label='Accuracy (%)',fraction=.03,pad=.02)
    for ext in ['png','pdf']:fig.savefig(output/f'prompt_necessity_sufficiency.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    protocol,manifest,completions,rows,mass,scores=collect()
    identities=[s['image_sha256'] for s in manifest['samples']];qids=[s['question_id'] for s in manifest['samples']]
    estimates=bootstrap(contrast_values(scores),identities,[7,7,1,7,7,7,7,7],
                        repetitions=protocol['statistics']['repetitions'],seed=SEED)
    paired=[dict(comparison=name,role='primary' if name=='C-Full' else 'exploratory',**r) for name,r in zip(CONTRASTS,estimates)]
    lookup={r['comparison']:r for r in paired};summary=[];taskrows=[];errors={}
    for col,key in enumerate(KEYS):
        sub=[r for r in rows if r['condition']==key];diff=lookup.get(key+'-Full')
        change=transitions(scores[:,0],scores[:,col],qids);errors[key]=change
        summary.append(dict(condition=key,label=LABELS[key],questions=len(sub),correct=int(scores[:,col].sum()),
            accuracy_percent=float(scores[:,col].mean()*100),delta_pp=diff['mean'] if diff else 0.,
            ci95_low=diff['ci95_low'] if diff else 0.,ci95_high=diff['ci95_high'] if diff else 0.,
            harm=len(change['harm']),fix=len(change['fix']),mean_first_logp_loss=float(np.mean([r['first_logp_loss'] for r in sub])),
            first_top1_flips=sum(r['first_top1_flip'] for r in sub),mean_js=float(np.mean([r['js'] for r in sub])),
            eos=sum(r['eos'] for r in sub),truncated=sum(r['truncated'] for r in sub),empty=sum(r['empty'] for r in sub)))
        for task in TASKS:
            tr=[r for r in sub if r['task']==task]
            taskrows.append(dict(task=task,condition=key,questions=len(tr),correct=sum(r['score'] for r in tr),
                accuracy_percent=float(np.mean([r['score'] for r in tr])*100),
                delta_pp=float(np.mean([r['accuracy_change'] for r in tr])*100),
                harm=sum(r['full_score']==1 and r['score']==0 for r in tr),fix=sum(r['full_score']==0 and r['score']==1 for r in tr)))
    patterns={}
    for s,row in zip(manifest['samples'],scores):patterns.setdefault(''.join(str(int(v)) for v in row),[]).append(s['question_id'])
    errors['correctness_patterns_order_FullABCDE']=patterns
    errors['C_harm_also_A']=sorted(set(errors['C']['harm'])&set(errors['A']['harm']))
    errors['C_harm_also_B']=sorted(set(errors['C']['harm'])&set(errors['B']['harm']))
    errors['C_harm_not_A_or_B']=sorted(set(errors['C']['harm'])-set(errors['A']['harm'])-set(errors['B']['harm']))
    write_csv(OUT/'per_question.csv',rows);write_csv(OUT/'attention_redistribution.csv',mass)
    write_csv(OUT/'accuracy_summary.csv',summary);write_csv(OUT/'task_accuracy.csv',taskrows);write_csv(OUT/'paired_comparisons.csv',paired)
    atomic_json(OUT/'error_transitions.json',errors)
    table,apps=snapshot();allowed=[g['uuid'] for g in manifest['gpus']]
    assert not any(any(line.startswith(uuid) for uuid in allowed) for line in apps.splitlines()),'Our selected GPUs still have compute processes; release audit incomplete'
    atomic_json(OUT/'gpu_release.json',dict(snapshot=table,compute_processes=apps,selected_gpus=manifest['gpus'],
        selected_gpu_compute_processes_absent=True,gpu1_not_used=True))
    elapsed=max(c['finished_epoch'] for c in completions)-min(c['started_epoch'] for c in completions)
    atomic_json(OUT/'validation.json',dict(status='complete',questions=240,images=len(set(identities)),new_generations=1200,
        reused_full_generations=240,matched_historical_full_vectors_exact=240,raw_first_logit_metric_replays=1200,
        source_and_protocol_verified=True,discovery_image_overlap=manifest['discovery_image_overlap'],
        per_gpu_smokes=[f'shard{k}/smoke.json' for k in range(len(completions))],cpu_tests=read(OUT/'cpu_validation.json')['tests'],
        all_masked_prompt_probabilities_zero=True,image_bos_self_keys_preserved=True,other_prefill_queries_exact_in_smoke=True,
        decode_native_hf_reference_exact=True,no_hook_leaks=True,no_input_weight_changes=True,no_nan_inf_oom=True,
        all_expected_checkpoints_unique=True,whole_image_shards=True,formal_wall_seconds=elapsed,
        sum_worker_seconds=sum(c['seconds'] for c in completions),gpus=manifest['gpus'],gpu1_not_used=True,gpu_release='gpu_release.json'))
    plot(summary,taskrows,OUT)
    by={r['condition']:r for r in summary};p=lookup['C-Full'];interaction=lookup['C-A-B+Full']
    lines=['# RQ2.5：五个敏感层的直接 Prompt 读取', '',
        f"完成240题（{len(set(identities))}张独立图像）、五个新增条件共1200次完整生成；严格绑定复用240份历史Full完整答案。",
        '模型为固定revision的LLaVA-1.5-7B，FP16/eager/Transformers4.49.0；32-token greedy、原生KV Cache。仅原始P−1的Prompt来源在指定层被阻断，Decode完全原生。', '',
        '## 完整答案结果', '', '|条件|正确/240|Accuracy|ΔFull (pp), 配对95%CI|原对→错|原错→对|', '|---|---:|---:|---|---:|---:|']
    for r in summary:lines.append(f"|{r['condition']} {r['label']}|{r['correct']}|{r['accuracy_percent']:.3f}%|{r['delta_pp']:+.3f} [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]|{r['harm']}|{r['fix']}|")
    lines+=['', '## 预声明问题的回答', '',
        f"1. 删除L1+L13–L16（C）的Accuracy变化为{p['mean']:+.3f}pp，主要比较双侧95%CI [{p['ci95_low']:+.3f}, {p['ci95_high']:+.3f}]。",
        f"2. 单删L1（A）变化{by['A']['delta_pp']:+.3f}pp；联合删L13–L16（B）变化{by['B']['delta_pp']:+.3f}pp。两者仅作预声明的探索性比较，不凭点估计认定总体优劣。",
        f"3. 非加性交互C−A−B+Full为{interaction['mean']:+.3f}pp，普通95%CI [{interaction['ci95_low']:+.3f}, {interaction['ci95_high']:+.3f}]；七项Bonferroni参考区间 [{interaction['family95_low']:+.3f}, {interaction['family95_high']:+.3f}]。负值表示额外联合损害，不等同语义贡献可加分解。",
        f"4. 仅保留五层直接Prompt读取（D）的Accuracy为{by['D']['accuracy_percent']:.3f}%，相对Full {by['D']['delta_pp']:+.3f}pp。此处只比较点估计，不作接近、等价或非劣效认定，也不说明只需五层语言理解。",
        f"5. 全部32层P−1→Prompt关闭（E）的Accuracy为{by['E']['accuracy_percent']:.3f}%，变化{by['E']['delta_pp']:+.3f}pp。",
        '6. 六类任务分开列于task_accuracy.csv与主图热力图，每类仅40题；任务差异属于描述性，不作未经校正的显著性推断。',
        '7. 是否值得扩大验证应结合C的损害、D的点估计和区间不确定性：本轮已消耗这些留出图片，后续只可在新的图像隔离样本中验证固定S，不能将本轮继续作为未使用确认集；本次不自动追加实验。', '',
        '## Prompt 来源重分配', '',
        'attention_redistribution.csv保存每题每层Full及干预后的Image/Prompt/BOS/Self质量和差值。目标层Prompt概率严格为0，保留Image、BOS、Self合法Key。Softmax会重新分配概率；跨层差值还包括此前干预导致的表示变化，不能把损害当作Prompt语义信息的独立因果贡献。',
        'P−1残差已有问题信息；其他Prefill Query与所有Decode仍能读取文本。Keep五层不等于仅保留五个Transformer层。', '',
        '## 统计、验证与范围', '',
        '主要比较仅C−Full；其余七项为探索性，paired_comparisons.csv同时给出普通95%与Bonferroni七项同时参考区间。按Image SHA256整簇配对bootstrap100000次，保留题目加权估计。探索性CI包含0不表示无损；本轮没有非劣效或等价检验。',
        f"CPU {read(OUT/'cpu_validation.json')['tests']}项测试通过；每张实际使用GPU完成两题真实smoke、历史Full与单层PromptOFF复现、所有五条件、独立HF生成参考、No-op有效Logits精确一致与异常清理。正式240份同机Full首Token向量逐位等于历史Full；1200份原始FP32首Token向量重算指标一致。",
        f"正式四舍五入用时{elapsed/60:.2f}分钟；GPU为{[g['physical_gpu'] for g in manifest['gpus']]}，未使用GPU1。GPU进程结束并释放（gpu_release.json）。",
        f"EOS/截断/空答案："+'；'.join(f"{k} {by[k]['eos']}/{by[k]['truncated']}/{by[k]['empty']}" for k in KEYS)+'。截断不删除样本，答案均按原官方GQA协议评分。',
        '', '源码提交、模型哈希、样本与分片、协议绑定见protocol.json。执行日志与每题检查点完整保留。没有训练、新层组合、额外采样或Decode干预。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    artifacts={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json','worker.lock']}
    atomic_json(OUT/'artifact_hashes.json',artifacts)
    print(json.dumps(dict(summary=summary,primary=p,interaction=interaction,formal_minutes=elapsed/60),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
