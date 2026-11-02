import csv
import json
from pathlib import Path
import sys
import textwrap

import numpy as np

from .common import *


def save(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def interval(values,indices):
    return [float(x) for x in np.quantile(np.asarray(values)[indices].mean(1),[.025,.975])]


def score(kind,samples,protocol):
    # Import the exact pinned scorer, including its corpus-aware CIDEr and sorted SPICE mapping.
    sys.path.insert(0,str(HISTORY))
    from paper_rq1.scorers import caption_scores
    refs={s['question_id']:s['answer'] for s in samples}
    allscores={};gens={};rows=[]
    for key,keep in conditions(kind).items():
        pred={};values={}
        for s in samples:
            path=OUT/kind/'checkpoints'/s['question_id'].replace(':','_')/(key+'.json')
            r=read(path)
            assert r['protocol_sha256']==sha(OUT/'protocol.json') and r['sample_sha256']==digest(s)
            assert r['generation_sha256']==digest(r['generation'])
            assert r['first_logits_file_sha256']==sha(path.with_suffix('.pt'))
            assert r['input_binding']==digest(read(OUT/f'{kind}_input_layouts.json')[s['question_id']])
            value=r['generation'];values[s['question_id']]=value;pred[s['question_id']]=value['text']
            assert value['prefill_blocked']==[] and value['decode_blocked']==sorted(set(range(36 if kind=='qwen' else 28))-set(keep))
            a=value['audit'];assert a['blocked_image_probability_max']==0 and a['retained_image_probability_min']>0
            assert a['decode_masked_rows']==len(value['decode_blocked'])*value['decode_steps']
        binding=digest(dict(references=refs,predictions=pred,scorer=protocol['bindings']))
        path=OUT/kind/'scores'/(key+'.json')
        if path.exists():
            sr=read(path);assert sr['binding']==binding
        else:
            overall,individual=caption_scores(refs,pred)
            sr=dict(binding=binding,overall=overall,per_image=individual);write(path,sr)
        allscores[key]={m:np.array([sr['per_image'][s['question_id']][m] for s in samples]) for m in ['CIDEr','SPICE']}
        for m in ['CIDEr','SPICE']:assert abs(allscores[key][m].mean()-sr['overall'][m])<1e-10
        gens[key]=values
        for s in samples:
            q=s['question_id'];g=values[q];base=gens['Full'][q]
            d=divergent(g['token_ids'],base['token_ids'])
            assert d==-1 or d>=1
            rows.append(dict(model=kind,question_id=q,image_id=s['raw_row']['id'],image_sha256=s['image_sha256'],
                condition=key,retained_layers=json.dumps(keep),caption=g['text'],token_ids=json.dumps(g['token_ids']),
                first_divergence_zero_based=d,first_token_same=g['token_ids'][0]==base['token_ids'][0],
                changed=d!=-1,eos=g['eos'],truncated=g['truncated'],empty=g['empty'],tokens=len(g['token_ids']),
                **sr['per_image'][q]))
        print('OFFICIAL',kind,key,sr['overall'],flush=True)
    return allscores,gens,rows


def figures(allscore,summary,indices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.bbox':'tight'})
    fig,axs=plt.subplots(2,3,figsize=(12,6.2),layout='constrained')
    for col,kind in enumerate(['llava','qwen','internvl']):
        scores=allscore[kind]
        keys=list(scores)
        if kind=='llava':
            keys=['Full','Decode None','Early','Middle','Late','Random10 mean','Random11 mean']
            scores=dict(scores)
            for size in (10,11):
                scores[f'Random{size} mean']={m:np.mean([scores[f'Random{size}-{s}'][m] for s in [20261102,20261103,20261104]],axis=0) for m in ['CIDEr','SPICE']}
        labels=[k.replace('Decode None','None').replace('Random','R').replace('-202611','\n').replace(' mean','\nmean') for k in keys]
        for row,m in enumerate(['CIDEr','SPICE']):
            ax=axs[row,col]
            for x,k in enumerate(keys):
                v=scores[k][m]-scores['Full'][m];lo,hi=interval(v,indices);mean=float(v.mean())
                color='#b04767' if k=='Middle' else '#427b8b' if k in ('Early','Late') else '#707070'
                ax.errorbar(x,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=3,ms=5)
            ax.axhline(0,color='#555555',lw=.8,ls='--');ax.grid(axis='y',alpha=.15)
            ax.set_xticks(range(len(keys)),labels,fontsize=8,rotation=35,ha='right')
            ax.set_ylabel(r'$\Delta$ '+m+' vs own Full')
            if row==0:ax.set_title({'llava':'LLaVA-1.5-7B | 32 layers','qwen':'Qwen3-VL-8B | 36 layers','internvl':'InternVL3-8B | 28 layers'}[kind])
    for ext in ['pdf','png']:fig.savefig(OUT/f'figure_cross_model_decode_regions.{ext}',dpi=300)
    plt.close(fig)


def qualitative(allscore,generations,samples):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    candidates=[]
    for kind in MODELS:
        v=allscore[kind]['Middle']['SPICE']-(allscore[kind]['Early']['SPICE']+allscore[kind]['Late']['SPICE'])/2
        candidates.extend((float(v[i]),kind,i) for i in range(len(samples)))
    selected=[];used=set()
    for positive in [True,False]:
        for value,kind,i in sorted(candidates,key=lambda r:((-r[0] if positive else r[0]),samples[r[2]]['image_sha256'],r[1])):
            if samples[i]['image_sha256'] in used:continue
            used.add(samples[i]['image_sha256']);selected.append(dict(index=i,model=kind,
                question_id=samples[i]['question_id'],image_sha256=samples[i]['image_sha256'],
                rule='largest Middle advantage' if positive else 'strongest counterexample',delta_spice=value))
            if len(selected)==(3 if positive else 6):break
    write(OUT/'qualitative_selection.json',selected)
    with PdfPages(OUT/'qualitative_six_images.pdf') as pdf:
        for selection in selected:
            s=samples[selection['index']];q=s['question_id']
            fig=plt.figure(figsize=(12.5,9))
            imgax=fig.add_axes([.025,.48,.28,.43]);imgax.imshow(Image.open(s['image_path']));imgax.axis('off')
            refax=fig.add_axes([.025,.04,.28,.40]);refax.axis('off')
            refs='Five references\n\n'+'\n\n'.join(textwrap.fill(t,47) for t in s['answer'])
            refax.text(0,1,refs,va='top',fontsize=8)
            for j,kind in enumerate(MODELS):
                ax=fig.add_axes([.33+j*.335,.045,.315,.87]);ax.axis('off')
                lines=[kind.upper(),'']
                for key in ['Full','Early','Middle','Late','Decode None']:
                    value=generations[kind][key][q];i=selection['index']
                    lines.extend([f'{key} | CIDEr {allscore[kind][key]["CIDEr"][i]:.3f} | SPICE {allscore[kind][key]["SPICE"][i]:.3f}',
                                  textwrap.fill(value['text'],61),''])
                body='\n'.join(lines);font=min(8.4,560/max(60,len(body.splitlines())))
                ax.text(0,1,body,va='top',fontsize=font,linespacing=1.18)
            fig.suptitle(f'{q} | {selection["rule"]} ({selection["model"]}, SPICE {selection["delta_spice"]:+.3f})',fontsize=11)
            pdf.savefig(fig);plt.close(fig)


def main():
    protocol=verify();samples=read(OUT/'sample_manifest.json')['samples']
    assert len(samples)==len({s['image_sha256'] for s in samples})==60
    assert all(read(OUT/m/'complete.json')['generations']==480 for m in MODELS)
    rng=np.random.default_rng(20261102);indices=rng.integers(0,60,(20000,60))
    scores={};generations={};rows=[];summary=[];comparisons=[]
    for kind in MODELS:
        scores[kind],generations[kind],r=score(kind,samples,protocol);rows.extend(r)
        for key,keep in conditions(kind).items():
            entry=dict(model=kind,condition=key,images=60,retained_count=len(keep),retained_layers=json.dumps(keep))
            for m in ['CIDEr','SPICE']:
                v=scores[kind][key][m];lo,hi=interval(v,indices);delta=v-scores[kind]['Full'][m];dl,dh=interval(delta,indices)
                entry.update({m:float(v.mean()),m+'_ci95_low':lo,m+'_ci95_high':hi,
                              'delta_'+m:float(delta.mean()),'delta_'+m+'_ci95_low':dl,'delta_'+m+'_ci95_high':dh})
            gs=list(generations[kind][key].values())
            entry.update(eos=sum(g['eos'] for g in gs),truncated=sum(g['truncated'] for g in gs),empty=sum(g['empty'] for g in gs),
                         changed=sum(g['token_ids']!=generations[kind]['Full'][q]['token_ids'] for q,g in generations[kind][key].items()))
            summary.append(entry)
        pairs=[(key,'Full') for key in conditions(kind) if key!='Full']+[('Middle','Early'),('Middle','Late')]
        matched=[k for k,v in conditions(kind).items() if k.startswith('Random') and len(v)==len(conditions(kind)['Middle'])]
        pairs += [('Middle',k) for k in matched]+[('Middle','Matched Random mean')]
        for a,b in pairs:
            for m in ['CIDEr','SPICE']:
                vb=np.mean([scores[kind][k][m] for k in matched],axis=0) if b=='Matched Random mean' else scores[kind][b][m]
                delta=scores[kind][a][m]-vb;lo,hi=interval(delta,indices)
                comparisons.append(dict(model=kind,condition=a,reference=b,metric=m,delta=float(delta.mean()),ci95_low=lo,ci95_high=hi,
                    improved=int((delta>0).sum()),worsened=int((delta<0).sum()),tied=int((delta==0).sum()),
                    count_matched=not(kind=='qwen' and a=='Middle' and b=='Early'),images=60,
                    inference='exploratory paired image percentile; ordinary 95%; fixed-corpus CIDEr IDF'))
    # No new LLaVA forward/scoring: exact historical per-image official results on the same references.
    historical={k:read(LLAVA/'scores'/(k+'.json')) for k in ['Full','Decode None','Early','Middle','Late']+
                [f'Random{size}-{seed}' for size in [10,11] for seed in [20261102,20261103,20261104]]}
    refs={s['question_id']:s['answer'] for s in samples}
    assert read(LLAVA/'sample_manifest.json')['samples']==samples
    scores['llava']={k:{m:np.array([r['per_image'][s['question_id']][m] for s in samples]) for m in ['CIDEr','SPICE']} for k,r in historical.items()}
    save('per_image.csv',rows);save('condition_summary.csv',summary);save('paired_comparisons.csv',comparisons)
    figures(scores,summary,indices);qualitative(scores,generations,samples)
    report(summary,comparisons,protocol)
    write(OUT/'analysis_complete.json',dict(complete=True,models=2,images=60,generations=960,bootstrap_repetitions=20000,
         official_scores=True,all_first_tokens_unchanged=True,paired_image_unit=True))


def report(summary,comparisons,protocol):
    lines=['# COCO Decode视觉重访：Qwen与InternVL跨架构探索','',
           '严格复用历史60张COCO和每图5条参考，960次新增完整生成；整个Prefill和首Token原生，仅真实增量Decode施加Image来源Hard Mask。',
           '两个模型均为BF16/Eager、独立原生KV Cache、Greedy、最多96 Token；只用物理GPU1顺序运行。','',
           '## 官方完整Caption评分','', '| 模型 | 条件 | 保留层数 | CIDEr | ΔFull [95%CI] | SPICE | ΔFull [95%CI] |',
           '|---|---|---:|---:|---|---:|---|']
    for r in summary:
        lines.append(f'| {r["model"]} | {r["condition"]} | {r["retained_count"]} | {r["CIDEr"]:.4f} | {r["delta_CIDEr"]:+.4f} [{r["delta_CIDEr_ci95_low"]:+.4f},{r["delta_CIDEr_ci95_high"]:+.4f}] | {r["SPICE"]:.4f} | {r["delta_SPICE"]:+.4f} [{r["delta_SPICE_ci95_low"]:+.4f},{r["delta_SPICE_ci95_high"]:+.4f}] |')
    lines+=['','## 预声明重点比较','', '| 模型 | 比较 | 指标 | 配对差值 [95%CI] | 改善/退化/平局图片 |','|---|---|---|---|---|']
    for r in comparisons:
        if r['condition']!='Middle' or r['reference']=='Full':continue
        lines.append(f'| {r["model"]} | Middle − {r["reference"]} | {r["metric"]} | {r["delta"]:+.4f} [{r["ci95_low"]:+.4f},{r["ci95_high"]:+.4f}] | {r["improved"]}/{r["worsened"]}/{r["tied"]} |')
    lines+=['','## 解释边界','',
       'Qwen Early保留12层（L0及L1–L11），Middle和Late保留13层。Middle−Early同时包含层数与层位差异，不能单独归因为中层位置。Middle只与两个Random12集合做等层数对照；Random11仅作为冻结参考。',
       'InternVL的Early/Middle/Late和三个Random9均保留10层，包括L0。三个随机种子不是独立图片；随机均值先在同图内求均值，再做图片配对Bootstrap。',
       '每项区间按60个Image SHA256整簇配对抽样20,000次。这里均为普通探索性95%区间，未作为多重校正确认检验；CIDEr沿用历史官方实现，Bootstrap条件于固定60图参考语料的IDF，不逐次重新拟合IDF。',
       'LLaVA历史32层的Early/Middle/Late分别保留11/11/12层，与36层Qwen、28层InternVL并非相同绝对层段。三模型图相对各自Full，只支持各自协议下的探索性结构对照。',
       '原始图片、题号、五条参考及SHA256完全保留，未根据模型输出筛题。六组原图案例按冻结规则作事后描述性正反例展示，未用于改变任何总体分数或实验条件。',
       '来源Mask只阻断当前Decode Query到原始Image Key的直接读取。其他Prefill Query与P−1均已正常处理图像，文本与历史Value可间接携带视觉信息。不能称为模型不需要视觉，也不是物理删除KV、整层跳过或实际加速。',
       '本轮沿用用于形成研究假说的同60图，属于探索性跨架构复现，不是独立确认；分数接近或区间包含0均不能证明无损。','',
       '## 验证与资源','',
       'CPU测试通过后冻结源文件、随机层、输入布局、模型文件和官方评分器哈希。每模型真实Smoke确认官方原生与Full/No-op生成Token及逐步有效Logits精确一致，所有条件整个Prefill逐层Hidden与首Token精确不变。',
       '每层每步审计原始Image Key、被阻断视觉概率严格为0、保留层视觉概率大于0、BF16行和原容差、原生Cache长度和独立性。Qwen DeepStack特征哈希与Full一致；InternVL官方chat输入嵌入与原生生成精确一致，max_num=1且logits_to_keep=1。',
       '所有逐图Caption、Token ID、首次分叉、EOS/截断/空答案、官方逐图CIDEr/SPICE与检查点均保存。误差门槛沿用历史实现，无放宽。']
    for kind in MODELS:
        c=read(OUT/kind/'complete.json')
        lines.append(f'- {kind}: 正式GPU运行 {c["formal_seconds"]/60:.2f} 分钟；含加载/Smoke {c["seconds_including_load_smoke"]/60:.2f} 分钟；峰值allocated/reserved {c["peak_allocated_mib"]:.0f}/{c["peak_reserved_mib"]:.0f} MiB。')
    lines+=['','图：figure_cross_model_decode_regions.pdf/png；六组含正反例原图：qualitative_six_images.pdf。',
            '协议、逐图统计与全部审计随分支保存；模型权重不提交。GPU结束后释放记录见gpu_release.json。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
