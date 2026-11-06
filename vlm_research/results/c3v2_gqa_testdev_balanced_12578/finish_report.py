import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
"""Complete three-way paired statistics and independently invoke official GQA eval."""
import gzip,json,sqlite3,re,subprocess,shutil
import numpy as np
from vlm_research.gqa_testdev_c3.common import *
from vlm_research.gqa_testdev_c3.analyze import bootstrap,write_rows

def main():
    freeze_check();ss=samples();records={}
    for rank in range(2):
        db=sqlite3.connect(f'file:{OUT}/worker{rank}/results.sqlite?mode=ro',uri=True)
        for row in db.execute('SELECT payload,sha256 FROM results'):
            r=unpack(row);records[r['question_id'],r['condition']]=r
        db.close()
    assert len(ss)==12578 and len(records)==37734
    v=np.array([[records[s['question_id'],c]['score'] for c in LABELS] for s in ss]);means=100*v.mean(0)
    draws=bootstrap(v,np.array([s['image_sha256'] for s in ss]));paired=[]
    for i,j in [(0,2),(1,2),(1,0)]:
        lo,hi=np.quantile(draws[:,i]-draws[:,j],[.025,.975])
        paired.append(dict(comparison=LABELS[i]+' minus '+LABELS[j],delta_pp=float(means[i]-means[j]),ci_low=float(lo),ci_high=float(hi),
            correct_to_wrong=int(((v[:,j]==1)&(v[:,i]==0)).sum()),wrong_to_correct=int(((v[:,j]==0)&(v[:,i]==1)).sum()),
            first_token_flips=sum(records[s['question_id'],LABELS[i]]['generation']['first_token_id']!=records[s['question_id'],LABELS[j]]['generation']['first_token_id'] for s in ss),
            answer_changes=sum(records[s['question_id'],LABELS[i]]['generation']['text']!=records[s['question_id'],LABELS[j]]['generation']['text'] for s in ss),
            token_changes=sum(records[s['question_id'],LABELS[i]]['generation']['token_ids']!=records[s['question_id'],LABELS[j]]['generation']['token_ids'] for s in ss)))
    write_rows(OUT/'paired_comparisons.csv',paired);negative=[];anomalies=[]
    for s,x in zip(ss,v):
        for i,j in [(0,2),(1,2),(1,0)]:
            if x[j] and not x[i]:negative.append(dict(question_id=s['question_id'],comparison=LABELS[i]+' minus '+LABELS[j],gold=s['answer'],
                baseline=records[s['question_id'],LABELS[j]]['generation']['text'],intervention=records[s['question_id'],LABELS[i]]['generation']['text']))
        for c in LABELS:
            g=records[s['question_id'],c]['generation']
            if any(g[k] for k in ['empty','truncated','repeated']):anomalies.append(dict(question_id=s['question_id'],**g))
    if negative:write_rows(OUT/'negative_cases.csv',negative)
    if anomalies:write_rows(OUT/'anomalies.csv',anomalies)
    else:(OUT/'anomalies.csv').write_text('question_id,condition,empty,truncated,repeated\n')
    official=Path('/data/lk/ATE-resources/data/gqa-testdev-balanced-official');checks=[]
    for i,c in enumerate(LABELS):
        path=OUT/f'official_predictions_{i}.json';atomic(path,[dict(questionId=s['question_id'],prediction=prediction(records[s['question_id'],c]['generation']['text'])) for s in ss])
        p=subprocess.run([__import__('sys').executable,str(official/'eval_llava_official.py'),'--tier','testdev_balanced',
            '--questions',str(official/'testdev_balanced_questions.json'),'--predictions',str(path)],capture_output=True,text=True)
        (OUT/f'official_eval_{i}.log').write_text(p.stdout+'\n'+p.stderr)
        assert p.returncode==0,p.stderr
        m=re.search(r'(?im)^accuracy\s*:\s*([\d.]+)',p.stdout);assert m,p.stdout[-2000:]
        acc=float(m.group(1));checks.append(dict(condition=c,main_accuracy=float(means[i]),official_accuracy=acc,delta_pp=float(acc-means[i])))
    write_rows(OUT/'official_score_crosscheck.csv',checks)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['pdf.fonttype']=42
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained');order=[2,0,1]
    ax.bar(range(3),means[order],color=['#496f9b','#c56e50','#4d9385']);ax.set_ylabel('GQA accuracy (%)')
    ax.set_xticks(range(3),['Native Full','Physical C3-V2','C3-V2 + L11–L24']);ax.set_title('Official testdev_balanced: 12,578 questions')
    for x,i in enumerate(order):ax.text(x,means[i]+.3,f'{means[i]:.3f}%',ha='center')
    for ext in ['png','pdf']:fig.savefig(OUT/f'quality_comparison.{ext}',dpi=220)
    plt.close(fig)
    lines=[]
    for i in order:
        ci='—' if i==2 else f"[{paired[i]['ci_low']:+.4f}, {paired[i]['ci_high']:+.4f}]"
        delta='—' if i==2 else f'{means[i]-means[2]:+.4f}'
        lines.append(f'| {LABELS[i]} | {int(v[:,i].sum())} / 12578 | {means[i]:.4f} | {delta} | {ci} |')
    d=paired[-1];table='\n'.join(lines)
    report=f'''# GQA官方testdev_balanced：三条件完整质量对照

状态：COMPLETE。官方12,578题、398张Stanford原图，37,734条完整生成。不同于此前132,062题val_balanced；没有复用其分数或预测。

| Condition | Correct / 12578 | ACC (%) | ΔNative (pp) | 95%CI |
|---|---|---|---|---|
{table}

Reaccess − C3-V2 = {d['delta_pp']:+.4f} pp [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]。回看原对变错{d['correct_to_wrong']}题、原错变对{d['wrong_to_correct']}题。

Published Reference：LLaVA公开GQA 62.0%，仅外部参考，不代替本地Native，不参与配对CI。本轮保持历史HF聊天模板、短答案指令、固定revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9、FP16/Eager、Transformers4.49.0、Greedy、max_new_tokens32、logits_to_keep1、H1末行LM Head。未为公开分数修改Prompt。

C3原版保持L0–L19 R写入、L20视觉Hidden退出、P−1/Decode禁图和L2–L5物理Attention跳算。回看仅在L11–L24为P−1/Decode保留Image KV，L25前退出视觉Hidden，L20以上R不新增写入。Native完整执行。严格复用参考Commit的物理算法，没有训练、层搜索、Kernel优化或测速。

每条件语言KV独立；当前图像视觉特征复用经GPU严格验证。短中长Smoke含Native/HF/No-op有效Logits、Mask、真实跳算、物理退出、RoPE、KV和Hook审计。详见CPU/GPU日志及源码/输入冻结文件。

问题、官方Gold来自Stanford questions1.2.zip的testdev_balanced_questions.json。图片通过lmms-lab-encoder/GQA镜像取得，398张JPEG的字节长度和CRC全部与官方images.zip成员相符；其中80张还与从Stanford直接下载的文件逐字节一致。其余图片记录了本地SHA，但未独立从Stanford重下载做第二次SHA对照。镜像Parquet SHA为ba0a57a72c483d6e4a390ecbbee05d07a55365aeac3e93f4a923ac61c9e465eb。全部图片可读，无重编码或跳题；来源、等价性审计与完整Manifest见source_dataset_audit.json、data_audit.json、protocol.json及sample_manifest.jsonl.gz。

主评分为text.rstrip('.').lower()后与官方短答案严格匹配。官方GQA工具（LLaVA官方文档指定的兼容修订）独立执行，结果及差异见official_score_crosscheck.csv，完整stdout保留；不静默更改主指标。全部EOS、截断、空答案、重复、负面案例均保留。

全部三组比较按图片SHA整簇配对20000次Bootstrap，seed20261106，探索性95%CI，不是非劣效证明。Query、Verify、Choose、Compare、Logical及Semantic分组见task_accuracy.csv。原val_balanced结果保持不变；资源释放见resource_release.json。
'''
    (OUT/'report_zh.md').write_text(report)
    atomic(OUT/'completion.json',dict(status='COMPLETE',questions=12578,generations=37734,accuracy_percent=means.tolist(),paired=paired,official_crosscheck=checks))
    atomic(OUT/'result_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='result_hashes.json' and p.suffix not in ['.sqlite']})
    print(json.dumps(dict(accuracy=means.tolist(),paired=paired),indent=2))

if __name__=='__main__':main()
