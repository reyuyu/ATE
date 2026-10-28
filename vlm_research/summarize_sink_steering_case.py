"""CPU-only artifact audit, paired-control summary and final figures. No forward.

Keeps the preregistered GPU implementation/configuration byte-for-byte intact.
"""
import csv
import json
from pathlib import Path
import shutil
import numpy as np

from attention_sink_steering import conditional, factors_for
from attention_sink_steering_case import (OUTPUT, P34, QID, read, sha, validate_binding,
    summarize, plt_module, mark, write_csv, write_json)


def verify(output):
    protocol=validate_binding(output);status=read(output/'status.json')
    assert status['gpu_completed'] and status['completed_conditions']==31
    run=output/status['latest_attempt'];validation=read(run/'validation.json')
    assert validation['passed'] and validation['condition_count']==31
    rows=read(run/'per_condition_results.json')
    with (run/'per_condition_results.csv').open() as stream:csvrows=list(csv.DictReader(stream))
    conditions=protocol['conditions']
    assert [r['condition'] for r in rows]==[r['id'] for r in conditions]
    assert len(rows)==len(csvrows)==31
    mapping=read(run/'input_mapping.json')
    token_ids={r['label']:r['answer_records'][0]['token_id'] for r in mapping['candidate_mapping']}
    assert mapping['answer_query']==602 and len(mapping['visual_indices'])==576
    logits=np.load(run/'prediction_logits.npz');audits=[]
    assert np.array_equal(logits['full'][0],logits['full'][1])
    assert float(logits['full'][0,token_ids['right']]-logits['full'][0,token_ids['left']])==-.03125
    b=np.load(output/'early_reference.npz')['B'][:,1]
    max_reconstructed_error=0.
    for condition,row,csvrow in zip(conditions,rows,csvrows):
        name=condition['id'];values=logits[name]
        assert np.isfinite(values).all() and np.array_equal(values[0],values[1])
        assert row['question_id']==csvrow['question_id']==QID
        for field in ['margin','left_logp','right_logp','gain_vs_full']:
            assert row[field]==float(csvrow[field])
        assert row['margin']==float(values[0,token_ids['right']]-values[0,token_ids['left']])
        assert row['margin']==row['right_logp']-row['left_logp']
        assert row['gain_vs_full']==row['margin']+.03125
        for label in ['left','right']:
            x=values[0].astype(np.float64);m=x.max()
            logp=x[token_ids[label]]-m-np.log(np.exp(x-m).sum())
            assert abs(logp-row[label+'_logp'])<1e-12
        if condition['family']=='A' and condition['strength']==0:
            assert np.array_equal(values,logits['full'])
        saved=np.load(run/f'rows_{name}.npz');pairs=saved['attention_before_after']
        assert saved['layers'].tolist()==condition['active_layers']
        assert pairs.shape==(len(condition['active_layers']),2,32,576)
        assert np.isfinite(pairs).all() and (pairs>=0).all()
        f=factors_for(condition,b)
        old=pairs[:,0].astype(np.float64);new=pairs[:,1].astype(np.float64)
        expected=old*f;expected=expected/expected.sum(-1,keepdims=True)*old.sum(-1,keepdims=True)
        # Independent float64 reconstruction vs GPU's FP32->FP16 implementation.
        # Half-min-subnormal and relative FP16 rounding, not an output-margin test.
        assert np.allclose(new,expected,atol=6e-8,rtol=.001)
        max_reconstructed_error=max(max_reconstructed_error,float(np.abs(new-expected).max()))
        for candidate in read(run/f'audit_{name}.json'):
            assert [a['layer'] for a in candidate['layers']]==condition['active_layers']
            for a in candidate['layers']:
                assert a['shape']==[1,32,604,604] and a['query']==602 and a['visual_keys']==576
                assert all(a[k] for k in ['passed','other_queries_bitwise_exact','nonvisual_bitwise_exact',
                    'formula_rounded_target_bitwise_exact','conditional_rounding_bound_passed','finite_nonnegative'])
                assert np.allclose(a['visual_mass_after'],a['visual_mass_before'],atol=2e-5,rtol=.001)
                audits.append(a)
        assert (run/f'heatmap_{name}.png').is_file()
    assert len(audits)==validation['audits']==396
    assert sum(a['conditional_nominal_exceeded'] for a in audits)==validation['nominal_conditional_l1_exceeded']
    smoke=read(run/'gpu_smoke.json');assert smoke['passed'] and len(smoke['windows'])==4
    assert all(w['logits_bitwise_full'] and w['question_hidden_bitwise_full'] for w in smoke['windows'])
    write_json(output/'artifact_validation.json',dict(passed=True,cpu_only=True,conditions=31,layer_audits=396,
        no_op_windows=4,code_hashes_verified=len(protocol['code_sha256']),raw_logits_logp_csv_json_consistent=True,
        saved_rows_reproduce_positive_reweighting=True,max_float64_reconstruction_abs_error=max_reconstructed_error,
        implementation_and_protocol_unchanged=True,postprocessor_sha256=sha(Path(__file__))))
    return protocol,status,run,rows


def clean_spatial(output,protocol):
    """Separate colorbar axis prevents occlusion in the initial cached plot."""
    from matplotlib.colors import LogNorm
    plt=plt_module();sample,entry=protocol['sample'],protocol['sink_entry']
    raw=np.load(P34/'attention_rows'/f'{QID}_task.npz')['attention'].astype(np.float64)
    p=conditional(raw)
    maps=np.stack([p[2:5].mean((0,1)),p[14:24].mean((0,1)),p[27:32].mean((0,1))])
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    fig.subplots_adjust(left=.015,right=.925,bottom=.06,top=.86,wspace=.25,hspace=.34)
    norm=LogNorm(vmin=1e-5,vmax=float(maps.max()))
    for g,name in enumerate(['Question','Answer P-1']):
        for j,(title,layers) in enumerate([('B: L2-L4',range(2,5)),('L14-L23',range(14,24)),('L27-L31',range(27,32))]):
            ax=axes[g,j];im=ax.imshow(maps[j,g].reshape(24,24),norm=norm,cmap='inferno',interpolation='nearest')
            mark(ax,sample,entry)
            ax.set_title(f'{name}: {title}\nraw visual mass={raw[list(layers),:,g].sum(-1).mean():.4f}',fontsize=11)
        ax=axes[g,3];ax.imshow(plt.imread(output/'processor_crop.png'),extent=(-.5,23.5,23.5,-.5));mark(ax,sample,entry)
        ax.set_title('True Processor crop / ROI / Sink',fontsize=11)
    fig.colorbar(im,cax=fig.add_axes([.945,.14,.012,.64]),label='Per-head visual-conditional probability, then mean heads / layers')
    fig.suptitle('15133706: Girl/Boy; correct = right\nCyan/magenta: object ROI; lime: frozen non-ROI Sink; orange: ROI-Sink overlap',fontsize=13)
    fig.savefig(output/'offline_spatial_comparison_clean.png',dpi=180);plt.close(fig)


def clean_condition_plots(output,protocol,run,rows):
    plt=plt_module();folder=output/'heatmaps';folder.mkdir(exist_ok=True)
    native=np.load(P34/'attention_rows'/f'{QID}_task.npz')['attention'].astype(np.float64)
    for row in rows:
        saved=np.load(run/f"rows_{row['condition']}.npz")
        pairs=saved['attention_before_after'].astype(np.float64);active=saved['layers'].tolist()
        selected=sorted({0,len(active)//2,len(active)-1})
        maps=[conditional(raw).mean(0) for i in selected for raw in [native[active[i],:,1],pairs[i,0],pairs[i,1]]]
        vmax=max(float(m.max()) for m in maps)
        fig,axes=plt.subplots(len(selected),3,figsize=(11,3.4*len(selected)),squeeze=False)
        fig.subplots_adjust(left=.02,right=.87,top=.88,bottom=.05,hspace=.32,wspace=.18)
        for ri,i in enumerate(selected):
            for ci,(raw,label) in enumerate([(native[active[i],:,1],'Native Full'),(pairs[i,0],'Local before edit'),(pairs[i,1],'After edit')]):
                ax=axes[ri,ci];im=ax.imshow(conditional(raw).mean(0).reshape(24,24),vmin=0,vmax=vmax,cmap='inferno')
                mark(ax,protocol['sample'],protocol['sink_entry'])
                ax.set_title(f'L{active[i]} {label}\nraw visual mass={raw.sum(-1).mean():.5f}',fontsize=10)
        fig.colorbar(im,cax=fig.add_axes([.9,.18,.015,.56]),label='Mean per-head visual-conditional probability')
        fig.suptitle(f"{row['condition']}: Margin={row['margin']:+.8f}; Full=-.03125\nShared linear scale within figure; cyan/magenta ROI, lime non-ROI Sink, orange overlap",fontsize=11)
        fig.savefig(folder/f"{row['condition']}.png",dpi=150);plt.close(fig)


def main():
    protocol,status,run,rows=verify(OUTPUT)
    summarize(OUTPUT)  # Rebuild original report first, then append offline-only final interpretation.
    clean_spatial(OUTPUT,protocol)
    clean_condition_plots(OUTPUT,protocol,run,rows)
    distance_rows=[]
    for row in read(OUTPUT/'offline_similarity_summary.json'):
        distance_rows.append(dict(row,head_mean_cosine_distance=max(0.,1-row['head_mean_cosine']),
                                  map_cosine_distance=max(0.,1-row['map_cosine'])))
    write_csv(OUTPUT/'offline_similarity_with_distances.csv',distance_rows)
    paired=[]
    for row in rows:
        if row['family']=='A' or row['target'].startswith('random'):continue
        controls=[r for r in rows if r['family']==row['family'] and r['strength']==row['strength'] and r['target'].startswith('random')]
        assert len(controls)==3 and all(r['patch_count']==row['patch_count'] for r in controls)
        margins=[r['margin'] for r in controls]
        paired.append(dict(condition=row['condition'],target_margin=row['margin'],random_mean_margin=float(np.mean(margins)),
            random_min_margin=min(margins),random_max_margin=max(margins),
            target_minus_random_mean=row['margin']-float(np.mean(margins)),
            target_beats_all_three=all(row['margin']>m for m in margins),patch_count=row['patch_count'],control_count=3))
    write_csv(OUTPUT/'paired_control_comparison.csv',paired);write_json(OUTPUT/'paired_control_comparison.json',paired)
    for suffix in ['csv','json']:
        shutil.copyfile(run/f'per_condition_results.{suffix}',OUTPUT/f'per_condition_results.{suffix}')
    plt=plt_module();fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(paired));means=np.array([r['random_mean_margin'] for r in paired])
    ax.scatter(x,[r['target_margin'] for r in paired],label='ROI / Sink target',s=50,zorder=3)
    for i,r in enumerate(paired):
        source=next(v for v in rows if v['condition']==r['condition'])
        controls=[v for v in rows if v['family']==source['family'] and v['strength']==source['strength'] and v['target'].startswith('random')]
        ax.scatter([i-.07,i,i+.07],[v['margin'] for v in controls],marker='x',color='grey',label='Matched random seeds (3)' if i==0 else None)
    ax.plot(x,means,ls=':',color='grey',alpha=.5)
    ax.axhline(0,color='black',lw=.8);ax.axhline(-.03125,color='grey',ls='--',label='Full -.03125')
    ax.axhline(.0390625,color='orange',ls=':',label='Historical Physical K24 +.0390625')
    ax.set_xticks(x,[r['condition'] for r in paired],rotation=20,ha='right')
    ax.set_ylabel('logP(right)-logP(left), nats');ax.legend(fontsize=8);ax.grid(alpha=.15)
    ax.set_title('15133706: all seven targeted ROI/Sink settings and all matched controls')
    fig.tight_layout();fig.savefig(OUTPUT/'target_vs_matched_controls.png',dpi=180);plt.close(fig)
    gallery=['# 全部31条件热图与数值\n\n各图均为预定窗口的首/中/末层、全部Head均值。三列为Native Full、局部编辑前、编辑后；图内共享线性色标，跨图范围可能不同。显示视觉内部归一化分布，同时注明原始视觉Mass。\n']
    for row in rows:
        name=row['condition'];prefix=status['latest_attempt']
        gallery.append(f"- [{name}: Margin {row['margin']:+.8f}](heatmaps/{name}.png) · [原始逐Head行]({prefix}/rows_{name}.npz)\n")
    (OUTPUT/'heatmap_gallery.md').write_text('\n'.join(gallery))
    p=OUTPUT/'report.md';s=p.read_text().replace('](offline_spatial_comparison.png)','](offline_spatial_comparison_clean.png)')
    interpretation='''## 实际结论：本例并非“深层Sink有害”的统一模式

- **早晚热点部分重合得到定量支持**：P−1平均图早→中/深Cosine=.9095/.9048、JS Distance=.3902/.3831；Top58重合25/30，其中分别12/13个是冻结非ROI Sink。逐Head的相似性和Top58重合更弱，不能说全分布相同。Q/A晚期模式有公共热点，同时仍有明显分布变化。

这里Cosine列明确是相似度；另外保存[Cosine Distance=1−Cosine及JS Distance表](offline_similarity_with_distances.csv)，P−1平均图早→中/深Cosine Distance分别.09052/.09523，避免把相似度和距离方向混淆。
- **A去偏变差**：alpha.5/1的Margin为−.1796875/−.265625。即使ROI份额提高且非ROI Sink份额大降，答案Margin仍下降；对整张早层B做逆权重并不等于只移除已定义Sink，也不能据此认定早层视觉信息整体有害。
- **B中层ROI增强很小**：Margin−.0234375，比Full高+.0078125，但仍答错；三背景增强均−.046875。目标比随机好+.0234375，属于单例、一个FP16步长的Full改善，未进行联合Sink+ROI条件，不能声称额外叠加收益。
- **C深层Sink抑制没有复现Physical K24转正**：L24～L31 factors.5/.1为−.03125/−.0390625，反而不优于背景−.0234375；L27～L31为−.0234375/−.015625，至多改善+.015625，仍为left。强抑制只比随机多+.0078125，近零数值敏感性明显。历史Physical K24 +.0390625是不同整体干预，本轮未重跑。
- **D早层同一13个非ROI Sink的P−1路径抑制转正**：factors.5/.1为+.0078125/+.1015625，三背景均未转正。强度.1的right logP从−18.59919714升到−18.17673724，绝对概率也提高；但仍只有约1.28e−8，并非自由生成成功。该结果没有显示这条特定早层路径在此例中是保持二选一正确的必要条件；不能推广为早层Sink没有功能、所有Query均可无损抑制。

D强抑制后，未被抑制的ROI/Sink交集Patch364的视觉内份额从.05311增至.20104，占干预后ROI总份额.28294的约71%。这说明“抑制非ROI Sink”也会重新分配到另一个冻结Sink，不能简单描述为消除了全部Sink或只增强了真实物体；本轮没有单独干预该交集，不能证明它驱动了Margin变化。

只有D的两个目标条件转正，31条件里其他29个均未转正。没有按输出再调参、换Head/窗口/样本。两次历史继承L23转正属于先前实验，不混入本轮对照。Sink注意力相关性、正向重分布的局部干预效应、Physical整体退出必须分别讨论。

## 数值与文件复核

108项CPU回归（含7项新测试）通过；四窗口no-op、alpha0精确Full；396个局部层审计通过。Attention行和最大误差0.000331223，视觉Mass最大绝对误差7.2360e−5，最大相对误差.00192952；Mass按原先的**组合**atol2e−5+rtol.001检查，小Mass不能仅用相对误差判定。40个层审计超过原条件L1 .002告警（两候选重复计算），均满足原dtype推导界及逐元素FP16舍入公式检查，未放宽容差。非视觉/其他Query局部bitwise不变，Q最终hidden不变。没有FP32主干实验，微小Margin翻转不等于稳健能力提升。

纯CPU复核逐条件CSV/JSON、完整原始logits/logP、31个NPZ重加权公式、396个审计、4窗口no-op及冻结源码/源数据hash，见[artifact_validation.json](artifact_validation.json)。`summarize_sink_steering_case.py`可完全离线重建配对表与最终报告/图，不运行模型。初始空间/干预图保留；clean版独立色条避免遮挡，heatmaps目录的原生Full Mass注释用FP64求和统一口径，原始FP16数组和模型结果未改。GPU加载5.412696秒；69次诊断Forward合计8.944490秒，图表/主机端写盘及模型加载不计入，不宣称稳定吞吐或加速。

[完整31条件CSV](per_condition_results.csv) · [配对随机对照](paired_control_comparison.csv) · [所有条件热图索引](heatmap_gallery.md)

![目标与随机对照](target_vs_matched_controls.png)

'''
    s=s.replace('## 解释边界与恢复',interpretation+'## 解释边界与恢复')
    p.write_text(s)
    (OUTPUT/'README.md').write_text('''# 15133706：Attention Sink去偏与ROI增强（真实GPU完成）

[报告](report.md) · [31条件完整CSV](per_condition_results.csv) · [冻结协议](protocol.json) · [数值复核](artifact_validation.json) · [全部31热图](heatmap_gallery.md)

中层B逆权重去偏变差；中层ROI增强微小改善但仍错；两个深层窗口Sink抑制均未复现Physical K24转正。只有L2～L4对同一非ROI Sink的P−1抑制转正（factor.5/.1 Margin +.0078125/+.1015625），三背景对照未转正。仅单例FP16探索，不能推断泛化或无损性。

![目标与随机对照](target_vs_matched_controls.png)

![早中深空间分布](offline_spatial_comparison_clean.png)

`gpu_attempts/20261028_230938/`保存实际31条件、完整原始logits、逐Head局部前/后行、396审计、GPU smoke和环境。历史实验不变；本轮完成后停止。
''')
    print(json.dumps(read(OUTPUT/'artifact_validation.json'),indent=2))


if __name__=='__main__':main()
