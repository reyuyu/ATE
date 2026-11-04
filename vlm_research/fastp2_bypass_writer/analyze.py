"""Read-only checkpoint reduction; image-cluster paired inference, no new GPU work."""
import collections
import csv
import json
from pathlib import Path

import numpy as np

from .common import OUT, ROOT, atomic, read, digest, sha, verify, score, key
from .engine import CONDITIONS

LABELS={'A0':'Fast-P2','A1':'All-query bypass','A2':'I+A partial bypass','B1':'Semantic-k mask',
        'B2-20261103':'Random seed 1103','B2-20261107':'Random seed 1107',
        'B2-20261111':'Random seed 1111','B3':'Physical split writer'}


def csvwrite(path,rows):
    rows=list(rows)
    assert rows
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def cluster_bootstrap(values,images):
    _,inverse=np.unique(images,return_inverse=True)
    groups=inverse.max()+1
    sums=np.zeros((groups,values.shape[1]))
    np.add.at(sums,inverse,values)
    counts=np.bincount(inverse)
    rng=np.random.default_rng(20261104)
    out=np.empty((20000,values.shape[1]))
    for start in range(0,20000,200):
        w=rng.multinomial(groups,np.full(groups,1/groups),size=min(200,20000-start))
        out[start:start+len(w)]=(w@sums)/(w@counts)[:,None]
    return out


def load():
    p=verify()
    samples=read(OUT/'sample_manifest.json')
    specs=read(OUT/'input_specs.json')
    rows={}
    missing=[]
    for k,s in samples.items():
        rank=p['assignment'][s['image_sha256']]
        path=OUT/f'shard{rank}/checkpoints'/(k+'.json')
        if not path.exists():
            missing.append(k)
            continue
        blob=read(path)
        v=blob['payload']
        assert blob['sha256']==digest(v) and v['protocol_sha256']==digest(p)
        assert v['input_spec_sha256']==digest(specs[k]) and v['sample_sha256']==digest(s)
        assert set(v['conditions'])==set(CONDITIONS)
        for name,c in v['conditions'].items():
            q=c['quality']
            assert q['score']==score(s,q['text'])
            assert len(c['measurements'])==(20 if s['dataset']=='pilot60' else 0)
        rows[k]=v
    return p,samples,specs,rows,missing


def qualities(samples,specs,checkpoints):
    per,accuracy,paired,tasks,negative=[],[],[],[],[]
    for dataset in ('pilot60','gqa240','textvqa300'):
        ks=sorted(k for k in checkpoints if samples[k]['dataset']==dataset)
        if not ks: continue
        values=np.array([[checkpoints[k]['conditions'][c]['quality']['score'] for c in CONDITIONS] for k in ks])
        images=[samples[k]['image_sha256'] for k in ks]
        draws=cluster_bootstrap(values*100,images)
        for j,c in enumerate(CONDITIONS):
            gs=[checkpoints[k]['conditions'][c]['quality'] for k in ks]
            bs=[checkpoints[k]['conditions']['A0']['quality'] for k in ks]
            delta=draws[:,j]-draws[:,0]
            counts=dict(score_down=int((values[:,j]<values[:,0]).sum()),score_up=int((values[:,j]>values[:,0]).sum()),
                correct_to_wrong=int(((values[:,0]==1)&(values[:,j]==0)).sum()) if dataset!='textvqa300' else 'N/A soft scoring',
                wrong_to_correct=int(((values[:,0]==0)&(values[:,j]==1)).sum()) if dataset!='textvqa300' else 'N/A soft scoring')
            accuracy.append(dict(dataset=dataset,condition=c,n=len(ks),images=len(set(images)),
                score_pct=100*values[:,j].mean(),delta_pp=100*(values[:,j]-values[:,0]).mean(),
                paired95_low=np.quantile(delta,.025),paired95_high=np.quantile(delta,.975),**counts,
                token_sequence_changes=sum(g['token_ids']!=b['token_ids'] for g,b in zip(gs,bs)),
                first_token_flips=sum(g['token_ids'][0]!=b['token_ids'][0] for g,b in zip(gs,bs)),
                mean_fixed_A0_token_logp_loss=np.mean([b['full_first_token_logp']-g['full_first_token_logp'] for g,b in zip(gs,bs)]),
                eos=sum(g['eos'] for g in gs),truncated=sum(g['truncated'] for g in gs),
                repeated=sum(g['repeated'] for g in gs),empty=sum(g['empty'] for g in gs),
                mean_content_tokens=np.mean([g['content_tokens'] for g in gs])))
            paired.append(dict(dataset=dataset,candidate=c,baseline='A0',delta_pp=100*(values[:,j]-values[:,0]).mean(),
                paired95_low=np.quantile(delta,.025),paired95_high=np.quantile(delta,.975),**counts))
            for k,g,b in zip(ks,gs,bs):
                s=samples[k]
                r=dict(sample=k,dataset=dataset,task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],
                    question=s['question'],gold=s.get('answers',s['answer']),condition=c,baseline_answer=b['text'],
                    answer=g['text'],token_ids=g['token_ids'],score=g['score'],baseline_score=b['score'],
                    score_delta=g['score']-b['score'],sequence_change=g['token_ids']!=b['token_ids'],
                    first_token_change=g['token_ids'][0]!=b['token_ids'][0],
                    first_token_signed_logp_loss=b['full_first_token_logp']-g['full_first_token_logp'],
                    eos=g['eos'],truncated=g['truncated'],repeated=g['repeated'],empty=g['empty'],
                    semantic_k=specs[k]['selection']['k'],R_tokens=len(checkpoints[k]['layout']['relay']))
                per.append(r)
                if g['score']<b['score'] or g['truncated'] or g['empty']: negative.append(r)
            for task in sorted({samples[k]['task'] for k in ks}):
                ti=[i for i,k in enumerate(ks) if samples[k]['task']==task]
                tasks.append(dict(dataset=dataset,task=task,condition=c,n=len(ti),score_pct=values[ti,j].mean()*100,
                    delta_pp=(values[ti,j]-values[ti,0]).mean()*100))
        # Three seeds are averaged within each same question, never treated as 3n observations.
        js=[CONDITIONS.index('B2-'+str(seed)) for seed in (20261103,20261107,20261111)]
        rand=values[:,js].mean(1)
        d=draws[:,js].mean(1)-draws[:,0]
        paired.append(dict(dataset=dataset,candidate='B2 within-question 3-seed mean',baseline='A0',
            delta_pp=(rand-values[:,0]).mean()*100,paired95_low=np.quantile(d,.025),paired95_high=np.quantile(d,.975),
            score_down=int((rand<values[:,0]).sum()),score_up=int((rand>values[:,0]).sum())))
        for c,b in (('A2','A1'),('B3','B1')):
            j,bj=CONDITIONS.index(c),CONDITIONS.index(b)
            d=draws[:,j]-draws[:,bj]
            paired.append(dict(dataset=dataset,candidate=c,baseline=b,delta_pp=(values[:,j]-values[:,bj]).mean()*100,
                paired95_low=np.quantile(d,.025),paired95_high=np.quantile(d,.975),
                score_down=int((values[:,j]<values[:,bj]).sum()),score_up=int((values[:,j]>values[:,bj]).sum()),
                token_sequence_changes=sum(checkpoints[k]['conditions'][c]['quality']['token_ids']!=
                                           checkpoints[k]['conditions'][b]['quality']['token_ids'] for k in ks)))
    for name,data in [('per_question.csv',per),('accuracy_summary.csv',accuracy),('paired_quality.csv',paired),
                      ('task_accuracy.csv',tasks),('negative_cases.csv',negative)]:
        if data: csvwrite(OUT/name,data)
    return accuracy,paired,negative


def performance(samples,checkpoints):
    protocol=verify()
    raw,flops,latency,memory=[],[],[],[]
    natural=[(k,v) for k,v in checkpoints.items() if samples[k]['dataset']=='pilot60']
    modes={'natural':natural,'fixed16':[],'fixed64':[]}
    fixed_missing=[]
    for mode in ('fixed16','fixed64'):
        for path in sorted(OUT.glob(f'shard*/{mode}/*.json')):
            blob=read(path);v=blob['payload']
            assert digest(v)==blob['sha256'] and v['protocol_sha256']==digest(protocol)
            assert all(len(c['measurements'])==20 for c in v['conditions'].values())
            modes[mode].append(('pilot60/'+v['sample'],v))
        fixed_missing.append((mode,60-len(modes[mode])))
    for mode,inputs in modes.items():
        medians={}
        for c in CONDITIONS:
            cs=[]
            for k,v in inputs:
                row=v['conditions'][c]
                measures=row['measurements']
                m={name:float(np.median([r[name] for r in measures])) for name in
                   ('latency_ms','vision_ms','language_prefill_ms','ttft_ms','decode_ms','peak_allocated','peak_reserved')}
                m['decode_ms_per_token']=m['decode_ms']/row['quality']['decode_steps'] if row['quality']['decode_steps'] else 0.
                totals=collections.Counter()
                for f in row['flops']:
                    totals[f['phase']]+=f['estimated_flops']
                    flops.append(dict(sample=k,condition=c,mode=mode,**f))
                m['language_prefill_flops']=totals['prefill']
                m['decode_flops']=totals['decode']
                m['estimated_prefill_tflops']=totals['prefill']/m['language_prefill_ms']/1e9
                m['estimated_decode_tflops']=totals['decode']/m['decode_ms']/1e9 if m['decode_ms'] else 0.
                q=row['quality']
                mem=dict(sample=k,condition=c,mode=mode,live_kv_MiB=q['kv']['live_bytes']/2**20,
                    capacity_kv_MiB=q['kv']['storage_bytes']/2**20,peak_allocated_MiB=m['peak_allocated']/2**20,
                    peak_reserved_MiB=m['peak_reserved']/2**20,kv_allocations=q['kv']['allocations'])
                memory.append(mem)
                rank=protocol['assignment'][samples[k]['image_sha256']]
                raw.append(dict(sample=k,condition=c,mode=mode,gpu_uuid=protocol['devices'][rank]['uuid'],content_tokens=q['content_tokens'],
                    eos=q['eos'],truncated=q['truncated'],**m))
                cs.append((k,m))
            medians[c]=dict(cs)
        for c in CONDITIONS:
            pairs=[(m,medians['A0'][k]) for k,m in medians[c].items()]
            if not pairs: continue
            r=dict(condition=c,mode=mode,inputs=len(pairs),repetitions_per_input=20,warmups_per_input=5)
            for field in pairs[0][0]:
                vals=[a[field] for a,b in pairs]
                r[field]=float(np.median(vals))
                r[field+'_p10']=float(np.quantile(vals,.1))
                r[field+'_p90']=float(np.quantile(vals,.9))
            for field in ('latency_ms','language_prefill_ms','decode_ms_per_token'):
                ratios=[b[field]/a[field] for a,b in pairs if a[field]>0 and b[field]>0]
                reductions=[100*(1-a[field]/b[field]) for a,b in pairs if b[field]>0]
                r[field+'_paired_speedup']=float(np.median(ratios)) if ratios else None
                r[field+'_reduction_pct']=float(np.median(reductions)) if reductions else None
            r['prefill_flops_reduction_pct']=float(np.mean([100*(1-a['language_prefill_flops']/b['language_prefill_flops']) for a,b in pairs]))
            latency.append(r)
    for name,data in [('latency_per_input.csv',raw),('latency_summary.csv',latency),
                      ('estimated_flops.csv',flops),('kv_memory.csv',memory)]:
        if data: csvwrite(OUT/name,data)
    return latency,fixed_missing


def kernels():
    rows=[]
    for path in sorted(OUT.glob('shard*/profiler/*.json')):
        blob=read(path);v=blob['payload'];assert digest(v)==blob['sha256']
        counts=v['cuda_activities']
        kernels_only={name:r for name,r in counts.items() if not any(x in name.lower() for x in ('memcpy','memset'))}
        rows.append(dict(condition=v['condition'],cuda_activity_count=sum(r['count'] for r in counts.values()),
            kernel_count=sum(r['count'] for r in kernels_only.values()),
            summed_kernel_device_ms=sum(r['device_us'] for r in counts.values())/1000,
            self_cpu_ms=sum(r['self_cpu_us'] for r in v['events'])/1000,
            cuda_launch_self_cpu_ms=sum(r['self_cpu_us'] for r in v['events'] if 'cudaLaunchKernel' in r['name'])/1000,
            select_scatter_calls=sum(r['count'] for r in v['events'] if any(w in r['name'] for w in ('index_select','index_copy'))),
            matmul_calls=sum(r['count'] for r in v['events'] if r['name']=='aten::matmul'),
            note='Separate instrumented fixed16 representative; not unprofiled E2E timing; no hardware FLOP or DRAM counter'))
    if rows: csvwrite(OUT/'kernel_profile.csv',rows)
    numerical=[]
    for path in OUT.glob('shard*/smoke.json'):
        for row in read(path)['numerical']:
            numerical.append(dict(shard=path.parent.name,**{k:v for k,v in row.items() if k not in ('operator_rows','MLP_calls','audit')}))
    if numerical: csvwrite(OUT/'numerical_audit.csv',numerical)
    return rows,numerical


def plots(quality,latency,kernel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    x=np.arange(len(CONDITIONS));colors=['#555555','#bc4749','#1c7c54','#246a91','#8b8c89','#8b8c89','#8b8c89','#9d4edd']
    fig,axes=plt.subplots(1,2,figsize=(11.5,3.5),layout='constrained')
    for ax,dataset in zip(axes,('gqa240','textvqa300')):
        rs={r['condition']:r for r in quality if r['dataset']==dataset}
        if not rs: continue
        y=np.array([rs[c]['delta_pp'] for c in CONDITIONS])
        lo=np.array([rs[c]['paired95_low'] for c in CONDITIONS]);hi=np.array([rs[c]['paired95_high'] for c in CONDITIONS])
        ax.bar(x,y,color=colors,width=.65)
        ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='none',color='#333333',capsize=2,lw=.8)
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,CONDITIONS,rotation=35,ha='right')
        ax.set(ylabel='Score change vs Fast-P2 (pp)',title=dataset+' | exploratory paired 95% CI')
    for ext in ('png','pdf'):fig.savefig(OUT/f'figure_quality.{ext}')
    plt.close(fig)
    natural={r['condition']:r for r in latency if r['mode']=='natural'}
    if natural:
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for ax,field,title in zip(axes,('language_prefill_ms','latency_ms','prefill_flops_reduction_pct'),
                                ('Language Prefill (ms)','Natural EOS E2E (ms)','Prefill matrix FLOPs saved (%)')):
            y=[natural[c][field] for c in CONDITIONS]
            ax.bar(x,y,color=colors,width=.7);ax.set_xticks(x,CONDITIONS,rotation=35,ha='right');ax.set_title(title)
            ax.grid(axis='y',alpha=.15)
        for ext in ('png','pdf'):fig.savefig(OUT/f'figure_compute_latency.{ext}')
        plt.close(fig)
    fixed={mode:{r['condition']:r for r in latency if r['mode']==mode} for mode in ('fixed16','fixed64')}
    if all(fixed.values()):
        fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
        for offset,(mode,rs) in zip((-.18,.18),fixed.items()):
            ax.bar(x+offset,[rs[c]['decode_ms_per_token'] for c in CONDITIONS],width=.34,label=mode)
        ax.set_xticks(x,CONDITIONS,rotation=30,ha='right');ax.set_ylabel('Decode ms/token');ax.legend(frameon=False)
        for ext in ('png','pdf'):fig.savefig(OUT/f'figure_decode.{ext}')
        plt.close(fig)


def report(p,samples,specs,quality,paired,latency,kernel,numerical,missing,fixed_missing):
    workers_done=all((OUT/f'shard{r}/completion.json').exists() and
        read(OUT/f'shard{r}/completion.json')['status']=='COMPLETE' for r in range(len(p['devices'])))
    complete=not missing and all(n==0 for mode,n in fixed_missing) and len(kernel)==8 and workers_done
    lines=['# LLaVA Fast-P2：Attention-only 跳算与稀疏写入',
        '', '**状态：'+('COMPLETE' if complete else 'PARTIAL')+'**。历史探索样本，不是独立非劣效验证。',
        '', '## 协议与实际算子',
        'A0/B0为原冻结Fast-P2路径：L20前移除视觉Hidden，所有P−1/Decode不直接读Image，持久KV只保留文本。',
        'A1在L2–L5真正不执行Q/K/V、QK、Softmax、AV、o_proj，不创建这些层KV；保留原生Normalization、Residual和所有MLP。',
        'A2仅计算非Image且非P−1的Prefill Query；K/V仍对全部Prefill Token计算，供R正常读图。被跳过层的后续Decode无Attention更新，不需要持久KV。R未被直接消融，但可因前层图像状态改变而间接受影响。',
        'A3与I+A Zero-Output Reference只用于数值核验，不作为加速条件。A1/A2的参考均实际计算Attention后置零。',
        'B1/三个B2为Functional Only：Softmax前Mask，不省Dense QKV/QK/AV/o_proj矩阵运算。Semantic-k及三个whole-group同预算Random-k严格复用历史选择器。',
        'B3是Eager分组物理原型：可读图Query对完整KV计算，禁止读图Query仅对非视觉KV计算，再scatter恢复全部输出行。保留全部QKV、o_proj、Image Query、MLP；仅省一部分QK/AV乘法，增加gather/scatter、小矩阵与Kernel启动。没有开发CUDA稀疏内核。',
        '', '## 完整生成质量',
        '|数据集|条件|得分%|ΔA0 pp|配对95%CI|损害/纠正或软分下降/上升|',
        '|---|---|---:|---:|---|---|']
    for r in quality:
        lines.append(f"|{r['dataset']}|{r['condition']}|{r['score_pct']:.3f}|{r['delta_pp']:+.3f}|[{r['paired95_low']:+.3f}, {r['paired95_high']:+.3f}]|{r['score_down']}/{r['score_up']}|")
    lines+=['','GQA沿用冻结短答案评分；TextVQA使用原10答案官方MMF软评分，不添加OCR。TextVQA的下降/上升不能称作二值纠错。所有候选均为同题、同卡新生成，未拼接历史消融结果。',
        '图片SHA整簇、问题等权的20,000次配对Bootstrap，seed=20261104。所有比较为探索性，不做非劣效声明。三个随机种子先在同题内平均，种子不是额外图片。',
        '', '## 性能与真实计算',
        '|模式|条件|输入数|Prefill ms|E2E ms|Decode ms/token|Prefill加速|E2E加速|E2E减少%|Prefill FLOPs减少%|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in latency:
        lines.append(f"|{r['mode']}|{r['condition']}|{r['inputs']}|{r['language_prefill_ms']:.2f}|{r['latency_ms']:.2f}|{r['decode_ms_per_token']:.2f}|{r['language_prefill_ms_paired_speedup']:.3f}×|{r['latency_ms_paired_speedup']:.3f}×|{r['latency_ms_reduction_pct']:+.2f}|{r['prefill_flops_reduction_pct']:+.3f}|")
    lines+=['','每题每条件5预热、20正式重复，冷请求KV，条件交错。表中时延为逐题重复中位数的跨题中位数；加速与降幅为同题配对比值中位数，因此不强求等于两列总体中位数相除。p10/p90和每题明细另见CSV。',
        '自然EOS用于任务E2E；fixed16/64可越过EOS，仅作工程Decode测试，不能替代短答案质量或真实回答时延。Vision每次原生计算，无跨请求视觉缓存。',
        'TTFT列沿用历史CUDA Event首预测段口径，即Vision+Language Prefill的GPU时间，不含CPU请求准备；冷请求的控制器/Mask/Cache设置统一计入E2E。没有把GPU-TTFT冒充含全部CPU等待的墙钟首Token时间。',
        'FLOPs由独立实际Linear/Conv/matmul/bmm Shape Census取得，包含CLIP bmm；2 FLOPs/MAC，排除Norm/Softmax/RoPE/激活等非矩阵操作。Dense Mask中的禁用边没有被误算为节省。',
        'Estimated Algorithmic TFLOP/s = 实际Shape估算矩阵FLOPs / 未Profile的CUDA Event阶段时间，不是硬件计数器吞吐。',
        '**HARDWARE_TFLOPS_UNAVAILABLE**：NCU返回ERR_NVGPUCTRPERM。DRAM读写/GBs、硬件TFLOPS、SM/TC吞吐均N/A，不从理论峰值、NVML百分比或Profiler推导。',
        '', '## 数值与资源',
        f"两卡各4输入真实Smoke，{len(numerical)}组物理-功能参考配对。严格Native/Noop、冻结A0 Logits一致；旧FP16 logit_atol=.10、rtol=.003、hidden相对RMSE≤.01、cosine distance≤1e-4均未放宽。原生形状变化的数值风险仍保留，不宣称逐位原生等价。",
        'A1目标QKVO实际调用为0；A2 Prefill仅对保留Query计算Q/o_proj、K/V仍完整，Decode该四层QKVO为0。32层MLP每步均执行。源码、模型和输入哈希冻结；每条检查点内容哈希及协议绑定重验。',
        'KV Live、容量、CUDA峰值Allocated/Reserved独立报告。KV减少比例不能当作总GPU显存减少。Profiler与正式时延分离，Kernel数量和调度时间仅为固定16代表样本，不冒充带宽测量。',
        '', '## 必须回答的研究问题']
    lookup={(r['dataset'],r['condition']):r for r in quality}
    timing={(r['mode'],r['condition']):r for r in latency}
    for c in ('A1','A2','B1','B3'):
        if ('natural',c) not in timing: continue
        t=timing['natural',c]
        g=lookup.get(('gqa240',c));v=lookup.get(('textvqa300',c))
        lines.append(f"- {c}：Prefill配对加速{t['language_prefill_ms_paired_speedup']:.3f}×，自然E2E {t['latency_ms_paired_speedup']:.3f}×，E2E减少{t['latency_ms_reduction_pct']:+.2f}%；实际Prefill矩阵FLOPs减少{t['prefill_flops_reduction_pct']:+.3f}%。"+
            (f" GQA240 Δ={g['delta_pp']:+.3f}pp；TextVQA300 Δ={v['delta_pp']:+.3f}pp。" if g and v else '质量未齐，不能下结论。'))
    if ('gqa240','A1') in lookup and ('gqa240','A2') in lookup:
        a1,a2=lookup['gqa240','A1'],lookup['gqa240','A2']
        lines.append(f"- GQA的A1损害/纠正={a1['score_down']}/{a1['score_up']}，A2={a2['score_down']}/{a2['score_up']}。两者净得分接近与否不能替代真实损害计数。")
    for r in paired:
        if r['candidate']=='A2' and r['baseline']=='A1' and r['dataset']!='pilot60':
            lines.append(f"- 保留R的A2相对A1，在{r['dataset']}提高{r['delta_pp']:+.3f}pp，配对95%CI [{r['paired95_low']:+.3f}, {r['paired95_high']:+.3f}]。这是本队列内比较，不能归结为每个R Query的独立贡献。")
        if r['candidate']=='B3' and r['baseline']=='B1':
            lines.append(f"- {r['dataset']}中B3与B1完整Token序列不同{r['token_sequence_changes']}题；数值近似仍按Smoke误差单独报告。")
    if ('textvqa300','B1') in lookup:
        t=lookup['textvqa300','B1']
        lines.append(f"- Semantic-k迁移到TextVQA：ΔA0={t['delta_pp']:+.3f}pp，95%CI [{t['paired95_low']:+.3f}, {t['paired95_high']:+.3f}]。GQA规则并非通用写入选择器。A0本身是L20交接模型而非原生Full；其TextVQA绝对得分不应被解读为原生模型的质量保持。")
    if (OUT/'independent_audit.json').exists():
        audit=read(OUT/'independent_audit.json')
        lines.append('- 数据集间图片重叠：'+json.dumps(audit['image_cluster_overlap'],ensure_ascii=False)+
                     '。不能把重叠的性能Pilot当作额外独立GQA验证。独立整数公式逐条核对实际Shape Census，所有MLP FLOPs保持。')
    lines+=['- A2是否降低A1损害，应看paired_quality.csv的A2−A1及两数据集，而不是只看跳算行数。R的保留不保证原生视觉状态不受间接影响。',
        '- Eager没有免费的Query-Key边稀疏。B1/B2不减少Dense FLOPs；B3只节约少量QK/AV，而保留权重投影、Image计算和MLP，并付出分组开销。只有实测Prefill/E2E更快且质量可接受，才值得后续集成；本轮不自动集成或扩样。',
        '- A1/A2即使算子确实减少，也不能用更早EOS或错误短输出宣称可用的质量保持加速；需同时读取质量、输出长度、EOS与负面案例。',
        '- 低损害/正向净分及CI跨0不等于无损；全部损害、纠正、非EOS和截断在negative_cases.csv、per_question.csv中保留。',
        '- repeated采用原有非重叠三Token片段重复启发式；URL等合法文字也可能触发，不能将其自动归类为模型崩溃。空答案、截断、无EOS均保留，不为通过而删题。',
        '', '## 完整性与停止',
        f'缺失自然/质量输入：{len(missing)}；fixed长度缺失：{fixed_missing}。严格只在全部600条输入×8条件、60×8×两个fixed模式和8条件Profiler齐全后标COMPLETE。',
        '仅使用/data，最多两张空闲RTX3090，同图全部条件同GPU，未抢占其他任务；资源释放审计在Worker退出后记录。本轮不修改Qwen、不训练、不搜索层或交接点。']
    if kernel:
        lines+=['','## 独立Profiler：固定16代表输入',
                '|条件|Kernel数|CUDA活动数|累计Kernel时间ms|CPU自身时间ms|gather/scatter调用|',
                '|---|---:|---:|---:|---:|---:|']
        for r in kernel:
            lines.append(f"|{r['condition']}|{r['kernel_count']}|{r['cuda_activity_count']}|{r['summed_kernel_device_ms']:.2f}|{r['self_cpu_ms']:.2f}|{r['select_scatter_calls']}|")
        lines.append('累计设备活动可能包含重叠；Profiler使执行变慢，不能把该表当成正式E2E或硬件带宽。')
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    atomic(OUT/'completion.json',dict(status='COMPLETE' if complete else 'PARTIAL',missing=missing,
        fixed_missing=fixed_missing,quality_answers=(len(samples)-len(missing))*len(CONDITIONS),
        expected_quality_answers=4800,expected_timed_requests=28800,expected_warmups=7200,
        protocol_sha256=digest(p)))
def main():
    p,samples,specs,checkpoints,missing=load()
    quality,paired,negative=qualities(samples,specs,checkpoints)
    latency,fixed_missing=performance(samples,checkpoints)
    kernel,numerical=kernels()
    plots(quality,latency,kernel)
    report(p,samples,specs,quality,paired,latency,kernel,numerical,missing,fixed_missing)
    print('ANALYSIS',len(checkpoints),'/600',fixed_missing,'negative_cases',len(negative),flush=True)


if __name__=='__main__': main()
