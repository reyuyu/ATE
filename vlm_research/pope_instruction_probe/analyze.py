"""Exact array/score audit, complete per-token atlases and three case figures."""
import argparse
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

from .common import *
from ..reverse_writer.common import scored

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
CMAP=plt.get_cmap('magma')


def divide(a,b):
    a,b=np.broadcast_arrays(a,b)
    return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)


def metrics(w,roi):
    w=np.asarray(w,dtype=np.float64)
    mass=w.sum(-1)
    edge=w[...,edge_mask()].sum(-1)
    f=np.array(roi['roi_patch_fractions'])
    target=(w*f).sum(-1)
    hit=w[...,f>0].sum(-1)
    ratio=divide(target,mass) if f.sum()>0 else np.full_like(mass,np.nan)
    return dict(image_mass=mass,edge_mass=edge,edge_ratio=divide(edge,mass),
        roi_mass=target,roi_ratio=ratio,roi_enrichment=divide(ratio,f.mean()),
        roi_hit_ratio=divide(hit,mass) if f.sum()>0 else np.full_like(mass,np.nan))


def scalar(v):
    return float(v) if np.isfinite(v) else None


def label(token):
    return token.replace('\u2581','_').replace('<0x0A>','[NL]')


def draw_boxes(ax,boxes,original=False):
    for b in boxes:
        if original:
            x,y,w,h=b['original_xywh']
        else:
            x,y,x1,y1=b['crop_xyxy']; w,h=x1-x,y1-y
        if w>0 and h>0:
            ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#00e5ff',linewidth=1.6))


def overlay(ax,crop,raw,roi,title,vmax):
    mass=raw.sum(); cond=raw/mass if mass>0 else np.zeros(576)
    ax.imshow(crop)
    if mass>0:
        ax.imshow(cond.reshape(24,24),extent=(0,336,336,0),cmap=CMAP,
            vmin=0,vmax=vmax,alpha=.65,interpolation='nearest')
    else:
        ax.text(.5,.5,'IMAGE PATH MASKED\nconditional map undefined',ha='center',va='center',
            transform=ax.transAxes,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'),fontsize=9)
    draw_boxes(ax,roi['boxes'])
    ax.set_title(title,fontsize=10); ax.axis('off')


def head_atlas(weights,crop,token,path):
    """Every L0-L19 x H0-H31 at native 24x24 patch resolution; no hidden selection."""
    cell,pad,left,top=32,3,52,45
    canvas=Image.new('RGB',(left+32*(cell+pad),top+20*(cell+pad)+25),'white')
    d=ImageDraw.Draw(canvas)
    d.text((5,3),token+' | all heads; log2(1+576*p(Image patch | Image)), fixed range 0..6',fill='black')
    for h in range(32): d.text((left+h*(cell+pad)+3,top-16),str(h),fill='black')
    bg=np.asarray(Image.fromarray(crop).resize((cell,cell),Image.Resampling.BILINEAR))/255.
    for layer in range(20):
        d.text((3,top+layer*(cell+pad)+8),'L'+str(layer),fill='black')
        for head in range(32):
            raw=weights[layer,head]; mass=raw.sum()
            if mass>0:
                value=np.clip(np.log2(1+576*raw.reshape(24,24)/mass)/6,0,1)
                heat=(CMAP(value)[...,:3]*255).astype(np.uint8)
                heat=np.asarray(Image.fromarray(heat).resize((cell,cell),Image.Resampling.NEAREST))/255.
                tile=Image.fromarray(np.uint8(np.clip(.35*bg+.65*heat,0,1)*255))
            else:
                tile=Image.new('RGB',(cell,cell),'#dddddd')
            canvas.paste(tile,(left+head*(cell+pad),top+layer*(cell+pad)))
    d.text((5,canvas.height-18),'Gray tile: zero Image mass, NOT evidence of token unimportance.',fill='black')
    canvas.save(path)


def layer_gallery(c3,e1,crop,roi,token,path):
    maps=np.stack([c3.mean(1),e1.mean(1)])  # condition, layer, patch
    conditional=divide(maps,maps.sum(-1)[...,None])
    vmax=float(np.nanmax(conditional)) if np.isfinite(conditional).any() else 1.
    fig,axes=plt.subplots(4,10,figsize=(20,8.5))
    for j,name in enumerate(('C3','E1')):
        for layer in range(20):
            ax=axes[j*2+layer//10,layer%10]
            overlay(ax,crop,maps[j,layer],roi,f'{name} L{layer}\nImage mass={maps[j,layer].sum():.3f}',vmax)
    fig.suptitle(token+' | actual 24x24 patches, head mean, all 20 layers; shared linear color scale',fontsize=13)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0,vmax),cmap=CMAP),ax=axes.ravel().tolist(),fraction=.01,pad=.01)
    cb.set_label('p(patch | Image); zero-mass cells have no conditional distribution')
    fig.subplots_adjust(left=.01,right=.91,top=.92,bottom=.02,hspace=.2,wspace=.03)
    fig.savefig(path,dpi=130); plt.close(fig)


def case_figure(s,mapping,roi,arrays,records,directory):
    qid=str(s['question_id']); crop=np.asarray(Image.open(OUT/'images'/qid/'processor_crop.png').convert('RGB'))
    fig=plt.figure(figsize=(19,17),layout='constrained')
    gs=fig.add_gridspec(4,6,height_ratios=[1.2,1.2,1.6,1.1])
    ax=fig.add_subplot(gs[0,:2]); ax.imshow(Image.open(s['image_path'])); draw_boxes(ax,roi['boxes'],True)
    for b in roi['other_same_class_boxes']:
        x,y,w,h=b['original_xywh']
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#33aa55',lw=1.3,linestyle='--'))
    g=roi['geometry']; x=g['crop_left']/g['scale_x']; y=g['crop_top']/g['scale_y']
    ax.add_patch(Rectangle((x,y),336/g['scale_x'],336/g['scale_y'],fill=False,edgecolor='#e69f00',lw=2))
    ax.set_title('Original: cyan=official bbox; orange=actual crop'); ax.axis('off')
    ax=fig.add_subplot(gs[0,2:4]); ax.imshow(crop); draw_boxes(ax,roi['boxes']); ax.axis('off')
    for b in roi['other_same_class_boxes']:
        x,y,x1,y1=b['crop_xyxy']
        if x1>x and y1>y: ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,edgecolor='#33aa55',lw=1.3,linestyle='--'))
    ax.set_title('Actual model input, 336x336\n'+roi['status'])
    ax=fig.add_subplot(gs[0,4:]); ax.axis('off')
    lines=[f'POPE Random {qid}: {s["question"]}', 'Gold=Yes; C3=Yes; E1=No', '', 'R token position | region | subword']
    for r in mapping:
        lines.append(f'{r["position"]:3d}  {r["region"][:5]:5s}  {label(r["subword"])}')
    ax.text(0,1,'\n'.join(lines),va='top',family='monospace',fontsize=8)
    maps=[]
    for condition in ('C3','E1'):
        for region in ('Q','Instruction','ASSISTANT'):
            ix=[r['r_index'] for r in mapping if r['region']==region]
            maps.append(arrays[condition][:,:,ix].mean((0,1,2)))
    cond=divide(np.stack(maps),np.stack(maps).sum(-1)[:,None])
    vmax=float(np.nanmax(cond))
    for j,raw in enumerate(maps):
        condition=('C3','E1')[j//3]; region=('Question','Instruction','ASSISTANT')[j%3]
        ax=fig.add_subplot(gs[1,j]); values=metrics(raw,roi)
        edge=values['edge_ratio']; edge_text=f'{float(edge)*100:.1f}%' if np.isfinite(edge) else 'NA'
        overlay(ax,crop,raw,roi,f'{condition} {region}\nImage={raw.sum():.3f}, edge={edge_text}',vmax)
    ax=fig.add_subplot(gs[2,:3])
    img=np.stack([arrays[c].sum(-1).mean(1).T for c in ('C3','E1')])
    im=ax.imshow(np.concatenate(img,axis=1),aspect='auto',cmap='viridis',vmin=0,vmax=float(img.max()))
    ax.axvline(19.5,color='white'); ax.set_xticks([0,9,19,20,29,39],['C3 L0','L9','L19','E1 L0','L9','L19'])
    ax.set_yticks(range(len(mapping)),[f'{r["position"]}: {label(r["subword"])}' for r in mapping],fontsize=8)
    ax.set_title('Every R token: head-mean absolute Image mass'); fig.colorbar(im,ax=ax,shrink=.7)
    ax=fig.add_subplot(gs[2,3:])
    ix=[r for r in mapping if r['region']=='Instruction']; singles=[records[f'add1_{j:02d}'] for j in range(10)]
    delta=np.array([r['yes_no_margin']-records['E1']['yes_no_margin'] for r in singles])
    ax.barh(range(10),delta,color=['#009e73' if r['generation']['text'].strip().lower()=='yes' else '#68747d' for r in singles])
    ax.set_yticks(range(10),[label(r['subword'])+' -> '+g['generation']['text'] for r,g in zip(ix,singles)])
    ax.invert_yaxis(); ax.axvline(0,color='black',lw=.8)
    ax.set_xlabel('Change in logP( Yes) - logP( No), nat, relative to E1')
    ax.set_title('Single-subword add-back: green means answer restored to Yes')
    ax=fig.add_subplot(gs[3,:]); bars=[]; names=[]; answers=[]
    for size in (2,4):
        for j in range(10-size+1):
            r=records[f'add{size}_{j:02d}']; bars.append(r['yes_no_margin']-records['E1']['yes_no_margin'])
            names.append(f'{size}:{j}-{j+size-1}'); answers.append(r['generation']['text'])
    ax.bar(range(len(bars)),bars,color=['#009e73' if a.lower()=='yes' else '#68747d' for a in answers])
    ax.axhline(0,color='black',lw=.8); ax.set_xticks(range(len(bars)),[n+'\n'+a for n,a in zip(names,answers)])
    ax.set_ylabel('Delta Yes/No margin (nat)'); ax.set_xlabel('Contiguous instruction offsets, 0-based; all predeclared windows')
    fig.suptitle(f'POPE {qid}: observed visual reading vs causal instruction add-back\n'
        'Image maps: head/layer/query means, visual-conditional, shared linear scale. Zero E1 instruction map is imposed by mask.',fontsize=14)
    fig.savefig(directory/'case_overview.png',dpi=170)
    fig.savefig(directory/'case_overview.pdf'); plt.close(fig)


def analyze(galleries=True):
    freeze=read(OUT/'freeze.json'); assert freeze['sources']==sources()
    samples=read(OUT/'sample_manifest.json'); protocol=read(OUT/'protocol.json')
    table=[]; details=[]; token_summary=[]; group_summary=[]; addbacks=[]; variants=[]; cases=[]
    mapping_all=[]
    import csv
    with (OUT/'token_mapping.csv').open() as f:
        for r in csv.DictReader(f):
            for key in ('r_index','position','token_id'): r[key]=int(r[key])
            mapping_all.append(r)
    for key,s in samples.items():
        qid=str(s['question_id']); directory=OUT/'figures'/qid; directory.mkdir(parents=True,exist_ok=True)
        roi=read(OUT/'images'/qid/'roi.json')
        mapping=[r for r in mapping_all if r['question_id']==qid]
        records={c:load_record(checkpoint(qid,c)) for c,_ in conditions(10)}
        assert records['C3']['generation']['text'].lower()=='yes' and records['E1']['generation']['text'].lower()=='no'
        arrays={}
        for condition in ('C3','E1'):
            meta=read(OUT/'attention'/qid/(condition+'.json'))
            assert sha(meta['path'])==meta['sha256']
            z=np.load(meta['path']); a=z['weights'].astype(np.float32)
            assert a.shape==(20,32,len(mapping),576)
            assert np.isfinite(a).all() and bool((a>=0).all())
            assert z['query_positions'].tolist()==s['input_mapping']['R']
            assert z['image_positions'].tolist()==s['input_mapping']['image']
            if condition=='E1':
                idx=[r['r_index'] for r in mapping if r['region']=='Instruction']
                assert np.count_nonzero(a[:,:,idx])==0
            arrays[condition]=a
            stat=metrics(a,roi)
            for l in range(20):
                for h in range(32):
                    for r in mapping:
                        j=r['r_index']
                        details.append(dict(r,question_id=qid,condition=condition,layer=l,head=h,
                            roi_status=roi['status'],roi_area_fraction=roi['roi_area_fraction'],
                            **{name:scalar(v[l,h,j]) for name,v in stat.items()}))
            for region in ('Q','Instruction','ASSISTANT','Separator'):
                ix=[r['r_index'] for r in mapping if r['region']==region]
                raw=a[:,:,ix].mean((0,1,2))
                group_summary.append(dict(question_id=qid,condition=condition,region=region,queries=len(ix),
                    **{name:scalar(v) for name,v in metrics(raw,roi).items()},roi_status=roi['status']))
        for condition,restored in conditions(10):
            r=records[condition]; g=r['generation']; score=scored(s,g['text'])
            inst=[p for p in mapping if p['region']=='Instruction']
            words=[inst[i]['subword'] for i in restored]
            d=r['yes_no_margin']-records['E1']['yes_no_margin']
            row=dict(question_id=qid,condition=condition,restored_instruction_offsets=restored,
                restored_prompt_positions=[inst[i]['position'] for i in restored],restored_subwords=words,
                restored_count=len(restored),answer=g['text'],score=score['score'],
                token_ids=g['token_ids'],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],
                yes_logp=r['variants'][' Yes']['logp'],no_logp=r['variants'][' No']['logp'],
                yes_no_margin=r['yes_no_margin'],delta_margin_vs_E1=d,
                first_token_flip_vs_E1=g['token_ids'][0]!=records['E1']['generation']['token_ids'][0],
                recovered_yes=g['text'].strip().lower()=='yes',
                output_changed_vs_E1=g['token_ids']!=records['E1']['generation']['token_ids'])
            addbacks.append(row)
            for form,values in r['variants'].items():
                variants.append(dict(question_id=qid,condition=condition,exact_suffix=repr(form),**values))
        causal=[r['r_index'] for r in mapping if r['position']<min(s['input_mapping']['parts']['Instruction'])]
        assert np.array_equal(arrays['C3'][:,:,causal],arrays['E1'][:,:,causal]), 'Earlier queries must be causally invariant'
        for r in mapping:
            j=r['r_index']; raw=arrays['C3'][:,:,j].mean((0,1)); stat=metrics(raw,roi)
            perhead=metrics(arrays['C3'][:,:,j],roi)
            objective=perhead['roi_mass'] if roi['roi_area_fraction']>0 else perhead['edge_mass']
            layer,head=np.unravel_index(objective.argmax(),objective.shape)
            if r['region']=='Instruction':
                offset=[x['position'] for x in mapping if x['region']=='Instruction'].index(r['position'])
                add=records[f'add1_{offset:02d}']
                answer=add['generation']['text']; delta=add['yes_no_margin']-records['E1']['yes_no_margin']
            else:
                answer,delta=None,None
            token_summary.append(dict(r,question_id=qid,roi_status=roi['status'],
                focus_layer=int(layer),focus_head=int(head),focus_rule='max absolute ROI mass' if roi['roi_area_fraction']>0 else 'max absolute edge mass; target cropped out',
                **{name:scalar(v) for name,v in stat.items()},restored_answer=answer,delta_yes_no_margin=delta,
                focus_head_image_mass=float(perhead['image_mass'][layer,head]),
                focus_head_edge_ratio=scalar(perhead['edge_ratio'][layer,head]),
                focus_head_roi_ratio=scalar(perhead['roi_ratio'][layer,head]),
                focus_head_roi_enrichment=scalar(perhead['roi_enrichment'][layer,head])))
            if galleries:
                gallery=directory/'token_galleries'; gallery.mkdir(exist_ok=True)
                crop=np.asarray(Image.open(OUT/'images'/qid/'processor_crop.png').convert('RGB'))
                name=f'{r["position"]}_{r["token_id"]}'
                title=f'POPE {qid} position {r["position"]}: {label(r["subword"])} [{r["region"]}]'
                layer_gallery(arrays['C3'][:,:,j],arrays['E1'][:,:,j],crop,roi,title,gallery/(name+'_layers.png'))
                for condition in ('C3','E1'):
                    head_atlas(arrays[condition][:,:,j],crop,title+' '+condition,gallery/(name+'_'+condition+'_heads.png'))
        case_figure(s,mapping,roi,arrays,records,directory)
        successes={str(n):[c for c,r in records.items() if len(r['restored'])==n and r['generation']['text'].strip().lower()=='yes'] for n in (1,2,4,10)}
        cases.append(dict(question_id=qid,question=s['question'],roi_status=roi['status'],roi_area_fraction=roi['roi_area_fraction'],
            addback_yes_conditions=successes,C3_margin=records['C3']['yes_no_margin'],E1_margin=records['E1']['yes_no_margin']))
        print('ANALYZED',qid,successes,flush=True)
    write_csv(OUT/'per_layer_head_token_attention.csv',details)
    write_csv(OUT/'token_attention_addback_summary.csv',token_summary)
    write_csv(OUT/'region_attention_summary.csv',group_summary)
    write_csv(OUT/'addback_results.csv',addbacks)
    write_csv(OUT/'answer_form_logp_audit.csv',variants)
    atomic(OUT/'case_summary.json',cases)
    report(samples,cases,token_summary,group_summary,addbacks)
    atomic(OUT/'analysis_audit.json',dict(status='PASS',all_attention_arrays_sha256_verified=True,
        all_actual_R_rows_layers_heads_present=True,E1_instruction_probabilities_exact_zero=True,
        pre_instruction_queries_attention_exactly_invariant=True,
        first_answer_not_EOS_query=True,records=len(addbacks),attention_metric_rows=len(details),
        token_summary_rows=len(token_summary),negative_and_no_recovery_cases_retained=True,
        primary_margin='canonical leading-space sentence-case, full vocabulary, complete token sequence, no EOS',
        note='No statistical population claim from three selected failures. No causal inference from attention mass alone.'))


def report(samples,cases,tokens,groups,adds):
    text=['# 三道POPE：Instruction视觉读取与逐Token恢复','',
        '状态：COMPLETE。仅1351/2459/301；原始图像、Prompt、权重和C3-V2路径不变。GPU0来自用户明确授权，GPU1/2/3未使用。',
        '', '## 首要发现：先看模型实际看到了什么','',
        '所有框均来自COCO官方instances_val2014，不按热图或模型输出绘制。模型执行短边缩放至336再中心裁剪；逐像素重建与真实Processor严格一致。24x24顺序为CLIP去CLS后的行优先Patch，未重排。',
        '原图中存在物体，不等于模型输入仍包含物体。目标完全裁出时，ROI比例/富集记NA，不能把原图框强行投到模型可见Patch；也不能把正确Yes解释为从不可见目标提取了直接视觉证据。','',
        '**花瓶301特别说明：** 用户指定的右上方对应官方vase框1154230和1670746，两框均在裁剪外。但左下方还有官方vase框1154146，在模型输入内；以绿色虚线单列，不合并进指定目标ROI。因此不能说该模型输入“没有任何花瓶”，也不能将Yes恢复直接归因于右上方不可见目标。', '',
        '| 题号 | 目标 | 实际输入ROI | 可见框面积比例 | C3 Yes/No Margin | E1 Margin |',
        '|---|---|---|---:|---:|---:|']
    for c in cases:
        text.append(f'|{c["question_id"]}|{TARGETS[c["question_id"]]}|{c["roi_status"]}|{100*c["roi_area_fraction"]:.4f}%|{c["C3_margin"]:.6f}|{c["E1_margin"]:.6f}|')
    text+=['','## 恢复实验：全部预先固定窗口','',
        '每题28条件：E1、完整10-Token Instruction恢复(C3)、10个单Token、9个连续2-Token、7个连续4-Token；共84条完整Greedy答案。所有恢复在L0-L19执行，P−1/Decode禁图、L2-L5答案物理Attention跳算、L20视觉退出均不变。',
        '主Margin为logP(" Yes")−logP(" No")，空格与大小写在GPU前固定，逐候选完整Teacher Forcing且不计EOS、不额外BOS。大小写/无前导空格的8种形式分别保存在answer_form_logp_audit.csv，不挑有利形式。','',
        '| 题号 | 单Token恢复Yes | 连续2个恢复Yes | 连续4个恢复Yes | 完整恢复 |',
        '|---|---|---|---|---|']
    for c in cases:
        q=c['question_id']; by={a['condition']:a for a in adds if a['question_id']==q}
        def describe(n):
            names=c['addback_yes_conditions'][str(n)]
            return '; '.join(' '.join(label(s) for s in by[name]['restored_subwords']) for name in names) or '无'
        text.append(f'|{q}|{describe(1)}|{describe(2)}|{describe(4)}|Yes|')
    text+=['','## Instruction逐Token配对表','',
        '| 题号 | 原生Subword | C3关注层/Head | 平均边缘比例 | 平均ROI比例 | 恢复后答案 | ΔYes/No Margin nat |',
        '|---|---|---|---:|---:|---|---:|']
    def fmt(v): return 'NA' if v is None else f'{v*100:.3f}%'
    for t in tokens:
        if t['region']=='Instruction':
            text.append(f'|{t["question_id"]}|`{t["subword"]}`|L{t["focus_layer"]}/H{t["focus_head"]}|{fmt(t["edge_ratio"])}|{fmt(t["roi_ratio"])}|{t["restored_answer"]}|{t["delta_yes_no_margin"]:+.6f}|')
    text+=['','## 解释边界与核心回答','',
        '- 观察性Attention：edge/ROI条件比例以实际Image mass为分母；E1 Instruction的视觉Mass严格为0，因此空间条件分布未定义，不把零热图解释为无用。',
        '- ROI为官方BBox覆盖，非物体分割；重叠框按几何并集合并，按Patch覆盖面积分数加权；roi_hit_ratio另记录任何重叠Patch的比例。富集=ROI条件比例/可见框面积占比，不等同于准确定位。',
        '- 重点层/Head预定为可见ROI绝对Attention Mass最大处；目标裁出时改列绝对边缘Mass最大处并标注ROI不可用。它是描述性选择，不是确认性显著结果。',
        '- 恢复输出为Yes的Token/窗口是此算法背景下的功能敏感候选；只有Margin变化而不翻转也完整保留。不能凭Attention热点宣称因果有效。',
        '- 对完全裁出的目标，本实验不支持“Instruction直接看到该边缘物体”的解释。恢复若改善Yes，仍可能涉及可见场景线索、文本/位置依赖、概率重新分配和残差路径，不能区分这些机制。',
        '- 三题是事先挑出的历史失分案例，不是无偏样本；不估计总体Accuracy、不作独立外部验证、不追加Token组合。',
        '', '## 图与原始数据','',
        '每题figures/<qid>/case_overview.png/pdf提供原图框与真实裁剪、全部R列表、区域热图、逐Token和连续窗口恢复结果。token_galleries/覆盖每个R Token的C3/E1全部20层Head均值及L0-L19×32 Head完整图册。所有图基于真实576个位置，不插值生成新观测。',
        '层图在同一Token的C3/E1各层间共用线性色标；全Head图册采用预先固定log2(1+576*p(patch|Image))范围0..6，灰格表示视觉Mass为0。图像叠加仅用于显示，CSV和NPZ保留原始概率。',
        '每个Head/Token的绝对Mass、边缘比例、ROI比例/富集见per_layer_head_token_attention.csv；用户要求的合并表见token_attention_addback_summary.csv。精确Attention数组在/data资源目录，attention/*.json记录路径、SHA和维度，不将大数组重复进Git。',
        '', '## 正确性与复现','',
        'CPU回归与3题GPU Smoke后执行；三题C3/E1完整Token和首Token全词表Logits SHA与e936ff3历史严格一致；观测Hook前后所有生成步Logits torch.equal；Native/No-op严格一致；对应Dense参考沿用冻结数值标准；禁止概率0，独立Cache、MLP、位置和Hook清理保留审计。',
        '复现：`python -m vlm_research.pope_instruction_probe.run`（只允许协议GPU0且空闲）；`CUDA_VISIBLE_DEVICES= python -m vlm_research.pope_instruction_probe.analyze`重算CSV与图。完整模型/数据/源码哈希见freeze.json；官方COCO标注只按Range提取验证集成员，来源、ZIP CRC和内容SHA均保留。',
        '本轮没有训练、没有改Prompt、没有额外Decode实验、没有性能加速宣称。GPU完成后释放，到此停止。']
    (OUT/'report_zh.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--no-galleries',action='store_true')
    args=parser.parse_args(); analyze(not args.no_galleries)
