import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import OUT, CONDITIONS, PAIRS, SEED, atomic, read, write_csv


def analyze(kind):
    folder=OUT/kind
    records=[read(p) for p in sorted((folder/'checkpoints').glob('*/*.json'))]
    assert len(records)==240*len(CONDITIONS)
    by={}
    for r in records: by.setdefault(r['question_id'],{})[r['generation']['condition']]=r
    assert len(by)==240 and all(set(v)==set(CONDITIONS) for v in by.values())
    rows=[]; summaries=[]; task_rows=[]; harms=[];rescues=[];kv=[]
    ordered=list(by.values());rng=np.random.default_rng(SEED)
    # The frozen quality manifest has one image per question: this is whole-image cluster resampling.
    assert len({v['N0']['image_sha256'] for v in ordered})==240
    boot=rng.integers(0,240,size=(20000,240))
    for c in CONDITIONS:
        differences=[]
        for v in ordered:
            r=v[c];g=r['generation'];f=v['N0']['generation']
            row=dict(question_id=r['question_id'],task=r['task'],image_sha256=r['image_sha256'],gold=r['gold'],
                condition=c,text=g['text'],token_ids=g['token_ids'],correct=g['correct'],
                harmed=f['correct'] and not g['correct'],rescued=not f['correct'] and g['correct'],
                first_token_flip=g['token_ids'][0]!=f['token_ids'][0],eos=g['eos'],truncated=g['truncated'],empty=g['empty'],
                kv_bytes=g['cache']['live_kv_bytes'],latency_ms=g['latency_ms'])
            rows.append(row);differences.append(int(g['correct'])-int(f['correct']))
            if row['harmed']:harms.append(row)
            if row['rescued']:rescues.append(row)
            for layer in g['cache']['layers']: kv.append(dict(question_id=r['question_id'],condition=c,**layer))
        selected=[r for r in rows if r['condition']==c]
        dist=np.asarray(differences)[boot].mean(1)*100
        summaries.append(dict(condition=c,questions=240,correct=sum(r['correct'] for r in selected),
            accuracy=100*np.mean([r['correct'] for r in selected]),delta_full_pp=np.mean(differences)*100,
            low95_pp=float(np.quantile(dist,.025)),high95_pp=float(np.quantile(dist,.975)),
            harmed=sum(r['harmed'] for r in selected),rescued=sum(r['rescued'] for r in selected),
            first_token_flips=sum(r['first_token_flip'] for r in selected),eos=sum(r['eos'] for r in selected),
            truncated=sum(r['truncated'] for r in selected),empty=sum(r['empty'] for r in selected)))
        for task in sorted({r['task'] for r in selected}):
            sub=[r for r in selected if r['task']==task]
            task_rows.append(dict(condition=c,task=task,questions=len(sub),accuracy=100*np.mean([r['correct'] for r in sub])))
    for name,data in [('per_question.csv',rows),('accuracy_summary.csv',summaries),('task_accuracy.csv',task_rows),
                      ('harm_cases.csv',harms),('recovered_cases.csv',rescues),('kv_memory_summary.csv',kv)]:
        write_csv(folder/name,data)
    audits=read(folder/'smoke_audit.json');write_csv(folder/'equivalence_audit.csv',audits)
    import csv
    with (folder/'latency_benchmark.csv').open() as f: timing=list(csv.DictReader(f))
    measure={r['condition']:r for r in timing if r['forced_tokens']=='64'}
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7,4))
    for row in summaries:
        c=row['condition'];x=float(measure[c]['end_to_end_ms_median']);ax.scatter(x,row['accuracy']);ax.annotate(c,(x,row['accuracy']),xytext=(3,3),textcoords='offset points')
    ax.set(xlabel='Fixed 64-token end-to-end latency (ms)',ylabel='GQA240 accuracy (%)',title=kind+' — quality vs measured latency')
    save(fig,folder/'figure_A_accuracy_latency')
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(CONDITIONS,[float(measure[c]['live_kv_bytes_median'])/2**20 for c in CONDITIONS]);ax.set(ylabel='Live KV storage (MiB)',title='After 64 generated tokens')
    save(fig,folder/'figure_B_kv_memory')
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(len(CONDITIONS));w=.38
    native=measure['N0']
    ax.bar(x-w/2,[float(native['language_prefill_ms_median'])/float(measure[c]['language_prefill_ms_median']) for c in CONDITIONS],w,label='Prefill')
    ax.bar(x+w/2,[float(native['decode_ms_per_token_median'])/float(measure[c]['decode_ms_per_token_median']) for c in CONDITIONS],w,label='Decode')
    ax.set_xticks(x,CONDITIONS);ax.set(ylabel='Speedup vs Native (matched eager)',title=kind);ax.axhline(1,color='gray',lw=.7);ax.legend()
    save(fig,folder/'figure_C_prefill_decode_speedup')
    report=['# '+kind+' Physical Visual Handoff Pilot','',
        '240张不同图片，六任务各40题；60题初筛后完成同一冻结240题。探索性工程Pilot，不是独立非劣效验证。',
        'N0为Native；D为密集功能参考，P为物理裁剪。主质量指标为自然EOS完整答案；固定16/64/128长度仅用于性能。',
        '同形状No-op严格一致；物理改变形状使用GPU前冻结门槛，Smoke和逐题Greedy一致性分别验证。',
        '| 条件 | Accuracy % | ΔFull pp | 95%CI | 损害/纠正 |','|---|---:|---:|---|---|']
    for r in summaries: report.append(f"| {r['condition']} | {r['accuracy']:.3f} | {r['delta_full_pp']:+.3f} | [{r['low95_pp']:.3f}, {r['high95_pp']:.3f}] | {r['harmed']}/{r['rescued']} |")
    report += ['', '## 限制', 'Softmax Mask不代表加速。本原型通过后层张量行数、实际分配KV存储和Profiler核对计算减少。',
               'Qwen原生mRoPE/DeepStack与LLaVA原生FP16路径分别保留。仅验证单图单轮，不能声称永久丢弃多轮视觉KV。',
               '区间是探索性普通95%区间，多候选未作确认性检验；不将区间跨0解释成无损。',
               '固定长度微基准包含Python同步与Cache拼接开销；没有宣称Kernel最优、通用无损或实际FLOPs按Mask比例减少。']
    (folder/'report_zh.md').write_text('\n'.join(report)+'\n')
    atomic(folder/'validation.json',dict(passed=True,questions=240,labels=len(records),equivalence_smoke=all(r['passed'] and r['tokens_equal'] for r in audits),
           greedy_equivalence_all_questions=all(v[a]['generation']['token_ids']==v[b]['generation']['token_ids'] for v in ordered for a,b in PAIRS),
           bootstrap_repetitions=20000))
    atomic(folder/'completion.json',dict(status='COMPLETE',questions=240,labels=len(records)))


def save(fig,path):
    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=220);fig.savefig(str(path)+'.pdf');plt.close(fig)
