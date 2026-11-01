"""Aggregate once after all frozen workers finish; preserve all negative results."""
import csv
import fcntl
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1.scorers import gqa_prediction, caption_scores
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_early_joint_attention.statistics import bootstrap
from .engine import CONDITIONS, LABELS
from .run import OUT, ROOT, SEED, read, verify, valid_checkpoint, old, snapshot

KEYS=list(CONDITIONS)
COLORS=['#414141','#147d64','#2876a4','#bc543a']


def csv_write(name, rows):
    path=OUT/name
    if not rows:
        path.write_text('status\nno_rows\n');return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def validate_generation(result,key):
    g=result['generation'];ids=g['token_ids'];n=len(ids)
    assert 1<=n and 2 not in ids[:-1] and g['eos']==(ids[-1]==2)
    assert g['decode_steps']==n-1 and g['own_cache'] and g['own_prefix']
    r_on,d_on=CONDITIONS[key];rows=result['attention_rows'];masks=result['mask_records']
    assert len(rows)==64+32*(n-1) and len(masks)==32*n
    for step in range(n):
        for group in (['Relay','P-1'] if not step else ['Decode']):
            selected=[r for r in rows if r['step']==step and r['query_group']==group]
            assert [r['layer'] for r in selected]==list(range(32))
            expected=not r_on if group=='Relay' else not d_on
            assert all(r['blocked']==expected for r in selected)
            assert all(r['image_probability_max']==0 for r in selected if expected)
        mr=[r for r in masks if r['step']==step];assert [r['layer'] for r in mr]==list(range(32))
        count=576*((0 if r_on else len(g['relay_indices']))+(0 if d_on else 1)) if not step else 576*int(not d_on)
        assert all(r['blocked_connections_per_head']==count for r in mr)
    return len(rows)


def collect():
    protocol=verify();binding=digest(protocol);manifest=read(OUT/'sample_manifest.json')
    completions=[read(OUT/f'shard{k}/completion.json') for k in range(len(manifest['gpus']))]
    assert all(c['status']=='complete' and c['binding']==binding and c['hooks_clean'] and c['parameters_unchanged'] for c in completions)
    assert sum(c['counts']['main'] for c in completions)==720
    assert sum(c['counts']['swap_pairs'] for c in completions)==len(read(OUT/'image_swap_manifest.json')['pairs'])
    assert sum(c['counts']['caption'] for c in completions)==4*len(read(OUT/'caption_manifest.json')['samples'])
    assert len(list(OUT.glob('shard*/main/*/*.json')))==720
    history=read(OUT/'history_bindings.json');records=[];rows=[];all_attention=[];audit_count=0
    for s in manifest['samples']:
        rank=manifest['assignments']['main'][s['question_id']];full=old.prior.native_logits(s,True)
        h=history[s['question_id']];assert sha256(ROOT/h['path'])==h['sha256']
        base=read(OUT/f"shard{rank}/baseline/{s['question_id']}.json")
        assert base['historical_Full_logits_exact'] and base['sample_sha256']==digest(s)
        current={}
        for key in KEYS:
            if key=='A':
                g=h['generation'];metric=metric_rows(full,full[None])[0];metric['logp_loss']=0.;origin=h['path'];kind='Strictly_bound_historical_Full'
            else:
                path=OUT/f"shard{rank}/main/{s['task']}__{s['question_id']}/{key}.json"
                rec=valid_checkpoint(path,binding,s,key);result=rec['result'];audit_count+=validate_generation(result,key)
                assert rec['hooks_clean'] and rec['inputs_unchanged'] and rec['parameters_unchanged']
                assert rec['physical_gpu']==manifest['gpus'][rank]['physical_gpu']
                with np.load(OUT/rec['raw_file'],allow_pickle=False) as z:first=torch.from_numpy(z['first_logits'].copy())
                metric=metric_rows(full,first[None])[0];metric['logp_loss']=-metric['delta_logp'];assert metric==result['first_token']
                g=result['generation'];assert int(first.argmax())==g['token_ids'][0]
                all_attention.extend(dict(question_id=s['question_id'],condition=key,**r) for r in result['attention_rows'])
                origin=str(path.relative_to(ROOT));kind='New_All32_layer_handoff'
            assert g['score']==int(gqa_prediction(g['text'])==s['answer'])
            current[key]=dict(generation=g,metric=metric)
            f=h['generation'];rows.append(dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],condition=key,
                gold=s['answer'],text=g['text'],prediction=g['prediction'],score=g['score'],full_score=f['score'],token_ids=json.dumps(g['token_ids']),
                full_text=f['text'],correct_to_wrong=int(f['score']==1 and g['score']==0),wrong_to_correct=int(f['score']==0 and g['score']==1),
                first_top1_flip=metric['top1_flip'],fixed_Full_token_id=metric['full_token_id'],first_top1_id=metric['ablated_token_id'],
                signed_logp_loss=metric['logp_loss'],absolute_fixed_token_logp_change=abs(metric['logp_loss']),js=metric['js'],
                eos=g['eos'],truncated=g['truncated'],empty=not bool(g['text'].strip()),decode_steps=g['decode_steps'],
                non_eos_length=sum(t!=2 for t in g['token_ids']),origin=origin,origin_kind=kind))
        records.append(current)
    assert len(rows)==960
    csv_write('per_question.csv',rows);csv_write('attention_probability_audit.csv',all_attention)
    csv_write('generation_anomalies.csv',[{k:r[k] for k in ['task','question_id','condition','eos','truncated','empty','decode_steps','non_eos_length','text']} for r in rows])
    return manifest,records,rows,completions,audit_count


def statistics(manifest,records,rows):
    y=np.array([[r[k]['generation']['score'] for k in KEYS] for r in records],dtype=float)*100
    differences=np.stack([y[:,1]-y[:,3],y[:,1]-y[:,0],y[:,2]-y[:,3],y[:,3]-y[:,2]-y[:,1]+y[:,0]],-1)
    stats=bootstrap(np.concatenate([y,differences],-1),[s['image_sha256'] for s in manifest['samples']],
        [1,1,1,1,2,2,1,1],repetitions=100000,seed=SEED)
    paired=[]
    for i,(name,role) in enumerate([('B-D','primary'),('B-A','primary'),('C-D','secondary_exploratory'),('D-C-B+A','secondary_exploratory')]):
        paired.append(dict(comparison=name,role=role,delta_pp=stats[4+i]['mean'],ci95_low=stats[4+i]['ci95_low'],ci95_high=stats[4+i]['ci95_high'],
            family95_low=stats[4+i]['family95_low'],family95_high=stats[4+i]['family95_high'],family=stats[4+i]['family'],
            repetitions=100000,seed=SEED,images=240))
    summary=[];tasks=[];first=[]
    for i,key in enumerate(KEYS):
        subset=[r for r in rows if r['condition']==key]
        summary.append(dict(condition=key,label=LABELS[key],correct=int((y[:,i]/100).sum()),total=240,accuracy_pct=float(y[:,i].mean()),
            accuracy_ci95_low=stats[i]['ci95_low'],accuracy_ci95_high=stats[i]['ci95_high'],delta_full_pp=float((y[:,i]-y[:,0]).mean()),
            correct_to_wrong=sum(r['correct_to_wrong'] for r in subset),wrong_to_correct=sum(r['wrong_to_correct'] for r in subset),
            first_top1_flips=sum(r['first_top1_flip'] for r in subset),mean_signed_logp_loss=float(np.mean([r['signed_logp_loss'] for r in subset])),
            mean_absolute_fixed_token_logp_change=float(np.mean([r['absolute_fixed_token_logp_change'] for r in subset])),mean_js=float(np.mean([r['js'] for r in subset])),
            eos=sum(r['eos'] for r in subset),truncated=sum(r['truncated'] for r in subset),empty=sum(r['empty'] for r in subset),
            mean_non_eos_length=float(np.mean([r['non_eos_length'] for r in subset])),mean_decode_steps=float(np.mean([r['decode_steps'] for r in subset]))))
        for task in old.TASKS:
            sr=[r for r in subset if r['task']==task]
            tasks.append(dict(task=task,condition=key,total=len(sr),correct=sum(r['score'] for r in sr),accuracy_pct=100*np.mean([r['score'] for r in sr]),
                delta_full_pp=100*np.mean([r['score']-r['full_score'] for r in sr]),correct_to_wrong=sum(r['correct_to_wrong'] for r in sr),wrong_to_correct=sum(r['wrong_to_correct'] for r in sr)))
        first.append({k:summary[-1][k] for k in ['condition','first_top1_flips','mean_signed_logp_loss','mean_absolute_fixed_token_logp_change','mean_js']})
    csv_write('accuracy_summary.csv',summary);csv_write('task_accuracy.csv',tasks);csv_write('paired_comparisons.csv',paired)
    csv_write('interaction_summary.csv',[paired[-1]]);csv_write('first_token_summary.csv',first)
    return summary,tasks,paired


def swaps(manifest):
    binding=digest(verify());sw=read(OUT/'image_swap_manifest.json');rows=[];hidden=[];audits=[]
    for pair in sw['pairs']:
        rank=manifest['assignments']['swap'][pair['pair_id']]
        r=valid_checkpoint(OUT/f"shard{rank}/swap/{pair['pair_id']}.json",binding)['result']
        assert r['pair_sha256']==digest(pair) and all(r['audit'][k] for k in ['D_first_logits_exact','D_all_decode_logits_exact','D_nonvisual_prefill_logits_exact','D_full_tokens_exact'])
        audits.append(dict(pair_id=pair['pair_id'],**r['audit']));hidden.extend(r['hidden'])
        for key,record in r['conditions'].items():
            assert record['all_blocked_probabilities_zero']
            generations=record['generations'];assert len(generations)==2
            row=dict(pair_id=pair['pair_id'],condition=key,question=pair['question'],gold_a=pair['samples'][0]['answer'],gold_b=pair['samples'][1]['answer'],
                text_a=generations[0]['text'],text_b=generations[1]['text'],correct_a=generations[0]['score'],correct_b=generations[1]['score'],
                both_correct=int(all(g['score'] for g in generations)),first_token_changed=not record['first_token_same'],tokens_changed=not record['tokens_same'],
                first_js=record['first_metric']['js'],first_logits_max_difference=record['first_logits_max_difference'])
            rows.append(row)
    assert len(rows)==4*len(sw['pairs'])
    csv_write('image_swap_results.csv',rows);csv_write('hidden_state_handoff.csv',hidden)
    freeze_json(OUT/'image_swap_invariance_audit.json',dict(passed=True,pairs=len(sw['pairs']),all_D_exact=True,comparisons=audits,
        hidden_pairs=sw['hidden_pairs'],all_C_D_relay_states_image_invariant=all(r['exact'] for r in hidden if r['condition'] in ['C','D'] and r['group']!='P-1'),
        all_D_states_exact=all(r['exact'] for r in hidden if r['condition']=='D')))
    return rows,hidden


def captions(manifest):
    cm=read(OUT/'caption_manifest.json');rows=[];metrics=[]
    for s in cm['samples']:
        rank=manifest['assignments']['caption'][s['question_id']]
        for key in KEYS:
            path=OUT/f"shard{rank}/caption/{s['question_id'].replace(':','_')}/{key}.json"
            result=valid_checkpoint(path,digest(verify()),s,key)['result'];validate_generation(result,key);g=result['generation']
            rows.append(dict(question_id=s['question_id'],condition=key,image_path=s['image_path'],image_sha256=s['image_sha256'],
                text=g['text'],token_ids=json.dumps(g['token_ids']),eos=g['eos'],truncated=g['truncated'],empty=not bool(g['text'].strip()),
                decode_steps=g['decode_steps'],non_eos_length=sum(t!=2 for t in g['token_ids']),reference_captions=json.dumps(s['answer'],ensure_ascii=False)))
    csv_write('caption_per_question.csv',rows)
    if rows:
        for name,h in cm['scorer_source_sha256'].items():assert sha256(name)==h
        refs={s['question_id']:s['answer'] for s in cm['samples']}
        for key in KEYS:
            pred={r['question_id']:r['text'] for r in rows if r['condition']==key}
            score,per=caption_scores(refs,pred)
            metrics.append(dict(condition=key,n=len(pred),**score,mean_decode_steps=float(np.mean([r['decode_steps'] for r in rows if r['condition']==key])),
                eos=sum(r['eos'] for r in rows if r['condition']==key),truncated=sum(r['truncated'] for r in rows if r['condition']==key)))
            csv_write(f'caption_{key}_official_per_image.csv',[dict(question_id=q,**v) for q,v in per.items()])
    csv_write('caption_summary.csv',metrics)
    return metrics,rows


def savefig(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def figures(summary,tasks,paired,swap,hidden):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    x=np.arange(4);fig,axes=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'width_ratios':[1.35,1]})
    values=np.array([r['accuracy_pct'] for r in summary]);lo=np.array([r['accuracy_ci95_low'] for r in summary]);hi=np.array([r['accuracy_ci95_high'] for r in summary])
    axes[0].bar(x,values,color=COLORS,width=.62);axes[0].errorbar(x,values,yerr=[values-lo,hi-values],fmt='none',color='black',capsize=4)
    axes[0].set_xticks(x,[f'{k}\n{LABELS[k]}' for k in KEYS],fontsize=8);axes[0].set_ylabel('GQA exact-match accuracy (%)');axes[0].set_ylim(0,103)
    for i,v in enumerate(values):axes[0].text(i,hi[i]+2,f'{v:.1f}%',ha='center',fontsize=9)
    axes[1].bar(x-.16,[r['correct_to_wrong'] for r in summary],width=.3,color='#bf4d3f',label='Correct to wrong')
    axes[1].bar(x+.16,[r['wrong_to_correct'] for r in summary],width=.3,color='#267d68',label='Wrong to correct')
    axes[1].set_xticks(x,KEYS);axes[1].set_ylabel('Questions vs Full');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Visual-to-Text Handoff | 240 GQA questions');fig.tight_layout();savefig(fig,'figure_a_visual_text_handoff')
    fig,ax=plt.subplots(figsize=(7,3));v=np.array([r['delta_pp'] for r in paired]);lo=np.array([r['ci95_low'] for r in paired]);hi=np.array([r['ci95_high'] for r in paired]);x=np.arange(4)
    ax.axhline(0,color='#999999',lw=.8);ax.errorbar(x,v,yerr=[v-lo,hi-v],fmt='o',color='#267d68',capsize=5)
    ax.set_xticks(x,['B - D','B - A','C - D','D - C - B + A']);ax.set_ylabel('Paired accuracy difference (pp)');ax.set_title('Path Interaction | image-cluster paired 95% CI');fig.tight_layout();savefig(fig,'figure_b_path_interaction')
    fig,axes=plt.subplots(2,2,figsize=(10,6.7))
    for i,key in enumerate(KEYS):
        sr=[r for r in swap if r['condition']==key];axes[0,0].scatter(np.full(len(sr),i),[r['first_js'] for r in sr],s=12,alpha=.6,color=COLORS[i])
    axes[0,0].set_xticks(np.arange(4),KEYS);axes[0,0].set_ylabel('First-token distribution JS (nat)');axes[0,0].set_title('Same text, different real images')
    for i,key in enumerate(KEYS):
        sr=[r for r in swap if r['condition']==key]
        axes[0,1].bar(i-.15,100*np.mean([r['tokens_changed'] for r in sr]),width=.28,color=COLORS[i])
        axes[0,1].bar(i+.15,100*np.mean([r['both_correct'] for r in sr]),width=.28,color=COLORS[i],alpha=.4,hatch='//')
    axes[0,1].set_xticks(np.arange(4),KEYS);axes[0,1].set_ylabel('Image pairs (%)');axes[0,1].set_ylim(0,105);axes[0,1].set_title('Solid: output changes | hatched: both correct')
    for i,key in enumerate(KEYS):
        for group,style in [('Question','-'),('Instruction','--'),('Template',':'),('P-1','-.')]:
            ls=sorted({r['layer'] for r in hidden if r['condition']==key and r['group']==group})
            if ls:axes[1,0].plot(ls,[np.mean([r['relative_l2'] for r in hidden if r['condition']==key and r['group']==group and r['layer']==l]) for l in ls],style,color=COLORS[i],label=f'{key} {group}',lw=1.2)
    axes[1,0].set_xlabel('Decoder layer output');axes[1,0].set_ylabel('Image-swap relative L2');axes[1,0].set_title('Fixed 6-pair read-only state subset')
    # Explicit zeros remain visible; never log-transform them into missing values.
    axes[1,1].plot(range(len(swap)//4),[r['first_logits_max_difference'] for r in swap if r['condition']=='D'],'o',color=COLORS[3],ms=4)
    axes[1,1].axhline(0,color='#999999',lw=.6);axes[1,1].set_ylim(-.01,.01);axes[1,1].set_xlabel('Frozen image pair');axes[1,1].set_ylabel('D maximum absolute logits difference');axes[1,1].set_title('D: exact Prefill + all-Decode invariance')
    fig.legend(*axes[1,0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=7,frameon=False,bbox_to_anchor=(.5,-.055))
    fig.tight_layout();savefig(fig,'figure_c_image_swap_causality')


def report(summary,tasks,paired,swap,hidden,caption,completions):
    d={r['condition']:r for r in summary};p={r['comparison']:r for r in paired}
    lines=['# Research 2.6：Visual-to-Text Information Handoff','',
        '## 范围与有效性','',
        '固定240题/240个图像SHA256簇，六类GQA各40题；该批已用于RQ2.4，属于机制探索而非独立确认。新增B/C/D共720次完整生成，历史Full逐题严格绑定并在本机重新校验首Token完整Logits，未复用历史30层OFF作为本轮All-layer OFF。',
        '所有32层使用Softmax前−inf，只改变指定Query→576个Image Key；保留Image计算、BOS/Self、其他文本边、MLP、残差、位置及每个条件自己的真实KV和生成前缀。R包含Image之后、P−1之前的所有非视觉位置，含空白/模板/指令，不仅Question。D覆盖P−1及每一步真实Decode。',
        '', '## 主结果','', '|条件|正确/240|Accuracy|相对Full pp|原对变错|原错变对|','|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['condition']} {r['label']}|{r['correct']}/240|{r['accuracy_pct']:.3f}%|{r['delta_full_pp']:+.3f}|{r['correct_to_wrong']}|{r['wrong_to_correct']}|")
    lines+=['','全部指标保留首Token signed LogP Loss、绝对固定Token LogP变化、JS、完整Token IDs、EOS、空答案、截断及六任务结果；见CSV。LogP/翻转不是Accuracy。',
        '', '## 预声明比较','', '|比较|ΔAccuracy pp|普通配对95%CI|角色|','|---|---:|---|---|']
    for r in paired:lines.append(f"|{r['comparison']}|{r['delta_pp']:+.3f}|[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]|{r['role']}|")
    lines+=['','两项Primary另保存Bonferroni-2区间；次要比较为探索性。100,000次整Image SHA256簇配对重采样，Seed=20261101，全部条件/交互使用同一组重采样权重。区间包含0不表示等价；不进行非劣效宣称。',
        '', '## 图像交换与隐藏状态','',
        f"从原3000题按问题文本完全一致、真实图像不同、官方标注答案不同、输入文本IDs完全一致选择固定{len(swap)//4}组；不按模型表现选择。候选组数和取样哈希规则见image_swap_manifest.json。所有D均须首Token完整Logits、全部真实Decode Logits、非视觉Prefill Logits和完整生成Token逐位精确相等，否则停止。",
        '状态仅采集预固定前6组的L1/L6/L13/L19/L25/L31实际Decoder Layer输出（非层输入，非末端Norm输出）。按实际Token位置比较Question、Instruction、Template、P−1；不跨不同文本配对。A/B同图其他Prefill状态相同，C/D同图其他Prefill状态相同，C/D的Relay状态在换图时精确相同。只保存距离/范数/哈希，不保存全层完整激活或执行Patching。']
    for key in KEYS:
        ss=[r for r in swap if r['condition']==key]
        lines.append(f"- {key}：{sum(r['tokens_changed'] for r in ss)}/{len(ss)}组生成Token随图变化；{sum(r['both_correct'] for r in ss)}/{len(ss)}组两端答案均正确；平均首Token JS={np.mean([r['first_js'] for r in ss]):.6f} nat。图像条件性与语义正确性分别报告。")
    lines+=['','## 研究问题回答','',
        f"1. **仅非视觉中继是否有功能？** B={d['B']['accuracy_pct']:.3f}%，D={d['D']['accuracy_pct']:.3f}%。B−D={p['B-D']['delta_pp']:+.3f}pp，95%CI [{p['B-D']['ci95_low']:+.3f}, {p['B-D']['ci95_high']:+.3f}]。据实结合图像交换判断，不将B仍答对的全部题目归因视觉（可能有文本先验）。",
        f"2. **相对Full损害？** B−A={p['B-A']['delta_pp']:+.3f}pp，95%CI [{p['B-A']['ci95_low']:+.3f}, {p['B-A']['ci95_high']:+.3f}]。分任务损害、纠错与损害均见task_accuracy.csv，不宣称等价。",
        f"3. **直接视觉能否弥补？** C={d['C']['accuracy_pct']:.3f}%；C−D={p['C-D']['delta_pp']:+.3f}pp，95%CI [{p['C-D']['ci95_low']:+.3f}, {p['C-D']['ci95_high']:+.3f}]。C的P−1仍可将视觉状态带入后续KV，因此不是每步只靠瞬时图像读取。",
        f"4. **非加性交互？** D−C−B+A={p['D-C-B+A']['delta_pp']:+.3f}pp，95%CI [{p['D-C-B+A']['ci95_low']:+.3f}, {p['D-C-B+A']['ci95_high']:+.3f}]。这是固定干预下Accuracy的交互，不是独立语义信息量相加。",
        '5. **换图与状态证据？** 查看上述完整固定样本池及不变性审计。B保留的唯一目标视觉入口是Prefill Relay；若B随图变化且D严格不变，支持视觉信息经Prefill非视觉状态/后续KV传播。没有证明其内部语义编码、哪个单Token承载信息或普适中继机制。',
        '6. **与Layer-Selective Visual KV Decoding关系？** 可用于形成后续路径假设，但本轮未选择最优层、未删KV、未跳过计算，也没有实际GPU加速证据。不得把Mask置零称为加速。',
        '', '## 限制','',
        'Hard Mask使剩余Key重新归一化；R包括问题、格式指令、模板及空白，不能单独归因Question。GQA严格完整字符串评分可能同时惩罚格式变化，不能将错误全解释为视觉语义丢失。短答案Decode常以EOS为主；交换控制是有标签差异的选定子集，不是总体性能估计。隐藏状态距离不是语义质量或中介效应大小的证明。LLaVA因果结构、固定模型/布局与既有样本限制外推。']
    if caption:
        lines+=['','## 60图长生成辅助检查','', '既有冻结COCO图像、统一Prompt、max_new_tokens=96，全部四条件真实KV生成；复用已审计官方CIDEr/SPICE，不引入新自动幻觉评分。',
            '|条件|CIDEr|SPICE|平均Decode步数|EOS/60|截断|','|---|---:|---:|---:|---:|---:|']
        for r in caption:lines.append(f"|{r['condition']}|{r['CIDEr']:.4f}|{r['SPICE']:.4f}|{r['mean_decode_steps']:.2f}|{r['eos']}|{r['truncated']}|")
        lines+=['','原始输出、参考描述、逐图官方分数完整保存。未进行未经审计的对象遗漏/幻觉自动打分，也不把参考描述未包含某对象视为其不存在；格式/重复只能作为原始输出的有限人工描述。']
    elapsed=max(c['finished_epoch'] for c in completions)-min(c['started_epoch'] for c in completions)
    lines+=['','## 执行与资源','',f'正式GPU任务墙钟 {elapsed/60:.2f} 分钟（含模型加载、240题+交换控制+60图辅助；不含CPU准备/Smoke/官方Caption评分）。GPU0/2/3，GPU1未使用；GPU释放实时检查见gpu_release.json。',
        '全部代码、样本、协议与模型/Transformers源码SHA256在GPU前冻结；逐题原始Logits重算指标一致。No-op和历史Full未放宽任何容差；Row-sum沿用0.002历史阈值。',
        '本轮到此停止，不扩大样本，不追加条件，不训练，不实现KV剪枝/加速。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


def main():
    lock=(OUT/'aggregation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'validation.json').exists():raise RuntimeError('Final aggregation already exists; do not count twice')
    tick=time.perf_counter();manifest,records,rows,completed,count=collect()
    summary,tasks,paired=statistics(manifest,records,rows);swap,hidden=swaps(manifest)
    caption,caption_rows=captions(manifest);figures(summary,tasks,paired,swap,hidden)
    table,apps=snapshot();used={g['uuid'] for g in manifest['gpus']}
    assert not any(line.split(', ')[0] in used for line in apps.splitlines())
    freeze_json(OUT/'gpu_release.json',dict(released=True,gpu_snapshot=table,compute_apps=apps,used_gpus=manifest['gpus'],GPU1_not_used=True))
    report(summary,tasks,paired,swap,hidden,caption,completed)
    freeze_json(OUT/'validation.json',dict(status='complete',questions=240,image_clusters=240,new_main_generations=720,reused_Full=240,
        image_swap_pairs=len(swap)//4,new_swap_generations=2*len(swap),caption_questions=len(caption_rows)//4,new_caption_generations=len(caption_rows),
        all_240_historical_full_logits_exact=True,cpu=read(OUT/'cpu_validation.json')['tests'],all_gpu_smokes_passed=True,
        all_masks_exactly_zero=True,actual_primary_attention_audit_rows=count,all_D_swap_logits_and_tokens_exact=True,
        hooks_clean=True,inputs_and_parameters_unchanged=True,no_nan_inf_oom=True,all_samples_retained=True,
        GPU1_not_used=True,gpus_released=True,analysis_seconds=time.perf_counter()-tick,protocol_sha256=sha256(OUT/'protocol.json')))
    hashes={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json','aggregation.lock','worker.lock']}
    freeze_json(OUT/'artifact_hashes.json',hashes)
    print(json.dumps(dict(summary=summary,paired=paired,caption=caption),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
