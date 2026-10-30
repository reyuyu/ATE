"""Five real-data paper figures; no interpolation to unmeasured layers/words."""
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .protocol import OUT,LAYERS,DEEP,SPECIAL,TYPES,MODES,GROUPS

COLORS=dict(Whole='#303030',Image='#bf622f',Prompt='#247f91',History='#7953a5',Recent='#388450',Older='#4a73b5')
GROUP_COLORS=dict(BOS='#686868',IMAGE=COLORS['Image'],PROMPT_TEXT=COLORS['Prompt'],GENERATED_HISTORY=COLORS['History'])


def save(fig,name):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def point(ax,x,mean,low,high,color,marker='o',label=None):
    ax.vlines(x,low,high,color=color,lw=1.5)
    ax.plot([x],[mean],marker=marker,color=color,ms=5,linestyle='none',label=label)


def figures(rows,native,summary,source_summary,main,paired,deep,recent):
    if len({r['image_sha256'] for r in rows})!=200:raise ValueError('Never plot incomplete/fake200 data')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    lookup={(r['mode'],r['layer'],r['word_type']):r for r in summary}
    nlookup={(r['source'],r['layer'],r['word_type']):r for r in source_summary}
    fig,axes=plt.subplots(3,3,figsize=(18,12),layout='constrained')
    for i,role in enumerate(TYPES):
        for mode in MODES:
            sub=[lookup[(mode,l,role)] for l in LAYERS]
            # Isolated measured-layer points avoid suggesting values in gaps3..14.
            x=np.arange(len(LAYERS))+(MODES.index(mode)-1.5)*.13
            for j,r in enumerate(sub):point(axes[i,0],x[j],r['mean'],r['ci95_low'],r['ci95_high'],COLORS[mode],label=mode if j==0 else None)
            d=[lookup[(mode,l,role)] for l in DEEP]
            for ax,key,low,high in [(axes[i,1],'mean','ci95_low','ci95_high'),(axes[i,2],'mean_abs_delta_logp','abs_ci95_low','abs_ci95_high')]:
                y=[r[key] for r in d];ax.plot(DEEP,y,'o-',color=COLORS[mode],label=mode,ms=3)
                ax.fill_between(DEEP,[r[low] for r in d],[r[high] for r in d],color=COLORS[mode],alpha=.10)
        axes[i,0].set_xticks(range(len(LAYERS)),LAYERS);axes[i,0].set_ylabel(role+'\nMean LogP loss (nat)')
        axes[i,0].set_xlabel('Ten measured layers (categorical, not complete32)')
        for j in [1,2]:axes[i,j].set_xticks(DEEP);axes[i,j].set_xlabel('Continuous measured late layers')
        axes[i,1].set_ylabel('Mean LogP loss (nat)');axes[i,2].set_ylabel('Mean absolute LogP change (nat)')
        for ax in axes[i]:ax.axhline(0,color='black',ls=':',lw=.8);ax.grid(alpha=.15)
        axes[i,0].legend(fontsize=8,ncol=2)
    axes[0,0].set_title('All measured depths; ordinary image-cluster95% CI')
    axes[0,1].set_title('L25-L31 magnification; signed effect')
    axes[0,2].set_title('Absolute change: signed cancellations retained')
    counts={role:lookup[('Whole',26,role)]['n_images'] for role in TYPES}
    fig.suptitle('Figure1: Native-token source necessity on independent COCO200; n='+str(counts),fontsize=13)
    save(fig,'figure1_late_layer_source_sensitivity')

    focus=[14,16,26,31]
    fig,axes=plt.subplots(3,4,figsize=(19,11),layout='constrained')
    for i,role in enumerate(TYPES):
        for j,key in enumerate(['mass','contribution_norm','relative_residual_ratio']):
            ax=axes[i,j]
            for g,source in enumerate(GROUPS):
                x=np.arange(4)+(g-1.5)*.18
                rs=[nlookup[(source,l,role)] for l in focus]
                ax.bar(x,[r[key+'_mean'] for r in rs],width=.16,color=GROUP_COLORS[source],label=source)
                for k,r in enumerate(rs):ax.vlines(x[k],r[key+'_ci95_low'],r[key+'_ci95_high'],color='black',lw=.7)
            ax.set_xticks(range(4),['L'+str(l) for l in focus]);ax.set_ylabel(role+'\n'+key.replace('_',' '));ax.grid(axis='y',alpha=.15)
        for m,mode in enumerate(MODES):
            rs=[lookup[(mode,l,role)] for l in focus];x=np.arange(4)+(m-1.5)*.18
            axes[i,3].bar(x,[r['mean_abs_delta_logp'] for r in rs],width=.16,color=COLORS[mode],label=mode)
        axes[i,3].set_xticks(range(4),['L'+str(l) for l in focus]);axes[i,3].set_ylabel('Actual mean |Delta LogP|');axes[i,3].grid(axis='y',alpha=.15)
    for ax in axes[0]:ax.legend(fontsize=7,ncol=2)
    for j,title in enumerate(['Native Attention Mass','Bias-free projected source norm','Source norm / actual layer-input residual','Measured ablation effect (not norm)']):axes[0,j].set_title(title)
    fig.suptitle('Figure2: Source mass, vector magnitude and causal prediction change are different quantities',fontsize=13)
    save(fig,'figure2_attention_source_contribution')

    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for j,mode in enumerate(['Image','Prompt','History']):
        for layer_index,layer in enumerate(SPECIAL):
            for analysis,offset,marker in [('raw',-.12,'o'),('adjusted',.12,'D')]:
                r=next((r for r in paired if r['mode']==mode and r['layer']==layer and r['analysis']==analysis),None)
                if r:
                    point(axes[0,j],layer_index+offset,r['mean'],r['familywise95_low'],r['familywise95_high'],
                        '#287f91' if analysis=='raw' else '#bd622f',marker,label=analysis if layer_index==0 else None)
                    axes[0,j].annotate('n='+str(r['n_images']),(layer_index+offset,r['familywise95_high']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
            for role,offset in [('Function',-.12),('Visual',.12)]:
                r=lookup[(mode,layer,role)]
                point(axes[1,j],layer_index+offset,r['mean_abs_delta_logp'],r['abs_ci95_low'],r['abs_ci95_high'],
                    '#287f91' if role=='Function' else '#bd622f',label=role if layer_index==0 else None)
        axes[0,j].set_title(mode+'-off: Visual - Function');axes[0,j].set_ylabel('Same-image difference; family41 corrected95% CI')
        axes[1,j].set_ylabel('Per-role mean absolute LogP change; ordinary95% CI')
        for ax in axes[:,j]:
            ax.set_xticks([0,1],['L26','L31']);ax.set_xlim(-.5,1.5);ax.axhline(0,color='black',ls=':',lw=.8);ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Figure3: Paired late-layer role differences, before/after confidence and position adjustment',fontsize=13)
    save(fig,'figure3_visual_function_source_differences')

    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    labels=[]
    for i,(role,layer) in enumerate((r,l) for r in TYPES for l in SPECIAL):
        labels.append(role+'\nL'+str(layer));pair=next(r for r in recent if r['word_type']==role and r['layer']==layer)
        for mode,offset in [('Recent',-.12),('Older',.12)]:
            r=lookup[(mode,layer,role)]
            point(axes[0],i+offset,r['mean'],r['ci95_low'],r['ci95_high'],COLORS[mode],label=mode if i==0 else None)
            point(axes[2],i+offset,r['mean_abs_delta_logp'],r['abs_ci95_low'],r['abs_ci95_high'],COLORS[mode],label=mode if i==0 else None)
        point(axes[1],i,pair['mean'],pair['familywise95_low'],pair['familywise95_high'],'#7855a6')
        axes[1].annotate('n='+str(pair['n_images']),(i,pair['familywise95_high']),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
    for ax in axes:ax.set_xticks(range(6),labels,fontsize=8);ax.axhline(0,color='black',ls=':',lw=.8);ax.grid(alpha=.15)
    axes[0].set_ylabel('Mean loss; ordinary95% CI');axes[1].set_ylabel('Recent - Older; family41 corrected95% CI');axes[2].set_ylabel('Mean absolute LogP change; ordinary95% CI')
    axes[0].legend();axes[2].legend();fig.suptitle('Figure4: Latest consumed token vs earlier native-generated history; empty Older at t=1 retained',fontsize=12)
    save(fig,'figure4_recent_vs_older_history')

    per={(r['image_id'],r['generation_step'],r['mode'],r['layer']):r for r in rows}
    candidates=[r for r in rows if r['mode']=='Whole' and r['layer']==31 and r['word_type']=='Visual']
    case=max(candidates,key=lambda r:abs(r['loss']-per[(r['image_id'],r['generation_step'],'Image',31)]['loss']))
    chosen=json.loads((OUT/'selected_token_manifest.json').read_text())['images'][case['image_id']]
    sample=next(r for r in json.loads((OUT/'sample_manifest.json').read_text())['samples'] if r['image_identity']==case['image_id'])
    fig=plt.figure(figsize=(17,14),layout='constrained');grid=fig.add_gridspec(4,2,height_ratios=[1.2,1,1,1])
    ax=fig.add_subplot(grid[0,0]);ax.imshow(Image.open(sample['image_path']));ax.axis('off')
    ax=fig.add_subplot(grid[0,1]);ax.axis('off')
    ax.text(0,.95,textwrap.fill(chosen['caption'],60),va='top',fontsize=12)
    ax.text(0,.7,'Post-hoc case: largest absolute L31 Visual\nWhole-Image loss difference. Not representative.\nNo sample removed; no source losses assumed additive.',va='top',fontsize=10)
    targets={r['word_type']:r for r in chosen['selected']}
    ax.text(0,.4,'\n'.join(f"{r}: '{v['word']}', t={v['generation_step']}" for r,v in targets.items()),va='top',fontsize=11)
    case_rows=[]
    for i,role in enumerate(TYPES):
        left=fig.add_subplot(grid[i+1,0]);right=fig.add_subplot(grid[i+1,1])
        if role not in targets:
            left.text(.1,.5,role+' missing in native caption');left.axis('off');right.axis('off');continue
        target=targets[role];t=target['generation_step']
        for mode in ['Whole','Image','History']:
            r=[per[(case['image_id'],t,mode,l)] for l in LAYERS]
            x=np.arange(len(LAYERS));y=[a['loss'] for a in r]
            left.scatter(x,y,color=COLORS[mode],label=mode,s=25)
            flips=[k for k,a in enumerate(r) if a['top1_flip']]
            if flips:left.scatter(flips,[y[k] for k in flips],marker='x',color='red',s=45)
            case_rows.extend(dict(image_id=case['image_id'],word=target['word'],word_type=role,generation_step=t,mode=mode,
                layer=a['layer'],loss=a['loss'],abs_delta_logp=a['abs_delta_logp'],top1_flip=a['top1_flip']) for a in r)
        left.set_xticks(range(len(LAYERS)),LAYERS);left.axhline(0,color='black',ls=':');left.grid(alpha=.15)
        left.set_title(role+": '"+target['word']+"'; red X = native Top1 changes")
        left.set_xlabel('Measured layers only (categorical)');left.set_ylabel('LogP loss (nat)');left.legend(fontsize=8,ncol=3)
        for k,source in enumerate(GROUPS):
            r=[next(a for a in native if a['image_id']==case['image_id'] and a['generation_step']==t and a['source']==source and a['layer']==l) for l in SPECIAL]
            right.bar(np.arange(2)+(k-1.5)*.18,[a['mass'] for a in r],width=.16,color=GROUP_COLORS[source],label=source)
        right.set_xticks([0,1],['L26','L31']);right.set_ylabel('Native source Attention Mass');right.set_title('Same target; source mass is not loss');right.legend(fontsize=7,ncol=2)
    fig.suptitle('Figure5: '+case['image_id']+' real native Caption and frozen targets',fontsize=13)
    save(fig,'figure5_real_caption_source_case')
    from paper_rq1.analyze import write_csv
    write_csv(OUT/'case_study_per_token.csv',case_rows)
