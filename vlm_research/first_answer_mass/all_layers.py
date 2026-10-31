"""All native layers, with explicitly labelled probability and contrast views."""
import csv
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
from benchmarks.dataset_registry import digest, freeze_json, sha256
from .run import OUT, ROOT, read, cohort, verify
from .finalize import original_result_digest
from .analyze import mapped_field, style

DEST = OUT / 'all_layers'
CMAP = 'turbo'


def display_scales(maps):
    positive=maps[maps>0]
    assert len(positive) and np.isfinite(maps).all() and (maps>=0).all()
    minimum=float(positive.min())
    maximum=float(maps.max())
    assert 0<minimum<maximum
    log=LogNorm(vmin=minimum,vmax=maximum,clip=True)
    local=[]
    for grid in maps:
        low,high=np.quantile(grid,[.02,.98])
        if high<=low:
            low,high=float(grid.min()),float(grid.max())
        assert high>low
        local.append(Normalize(float(low),float(high),clip=True))
    return log,local


def overlay(image,grid,g,norm):
    field,mask=mapped_field(grid,g)
    if isinstance(norm,LogNorm):
        field=np.maximum(field,norm.vmin)
    strength=np.asarray(norm(field),float)
    color=plt.get_cmap(CMAP)(strength)[...,:3]
    alpha=(.15+.7*strength)*mask
    base=np.asarray(image,float)/255
    rgb=base*(1-alpha[...,None])+color*alpha[...,None]
    return Image.fromarray(np.uint8(np.clip(rgb,0,1)*255))


def save_individual(folder,image,enhanced,grid,local,sample,layer,mass):
    w,h=960,410
    canvas=Image.new('RGB',(w,h),'white')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    draw.text((10,5),f"{sample['task']} | QID {sample['question_id']} | L{layer} | raw Image mass={mass:.6f}",fill='black',font=font)
    heat=np.uint8(plt.get_cmap(CMAP)(local(grid))[...,:3]*255)
    panels=[image,enhanced,Image.fromarray(heat).resize((288,288),Image.Resampling.NEAREST)]
    titles=['Original image','Shared logarithmic conditional scale','Per-layer 2-98% contrast (not probability)']
    for j,(panel,title) in enumerate(zip(panels,titles)):
        panel=panel.copy()
        panel.thumbnail((306,288))
        canvas.paste(panel,(j*320+(320-panel.width)//2,58+(288-panel.height)//2))
        draw.text((j*320+8,35),title,fill='black',font=small)
    draw.text((10,354),'Log scale is shared over all 32 layers of this case; local contrast is NOT comparable across layers.',fill='black',font=small)
    draw.text((10,376),'Original probabilities are unchanged. Outside the actual CLIP crop is unobserved.',fill='black',font=small)
    canvas.save(folder/f'L{layer:02d}.png')


def contact(sample,result,image,maps,log,local,kind):
    fig,axes=plt.subplots(4,8,figsize=(16,10))
    fig.subplots_adjust(left=.01,right=.94,bottom=.07,top=.88,hspace=.36,wspace=.08)
    for layer,ax in enumerate(axes.flat):
        norm=local[layer] if kind=='local_contrast' else log
        if kind=='pure_heatmap':
            ax.imshow(maps[layer],cmap=CMAP,norm=norm,interpolation='nearest')
        else:
            ax.imshow(overlay(image,maps[layer],result['geometry'],norm))
        ax.set_title(f"L{layer} | M={result['rows'][layer]['image_mass']:.3f}",fontsize=9,pad=3)
        ax.set_axis_off()
    question=f"{sample['task']} | QID {sample['question_id']} | {sample['question']} | Full: {result['full_text'].strip()}"
    fig.text(.012,.965,'\n'.join(textwrap.wrap(question,150)),ha='left',va='top',fontsize=12)
    if kind=='local_contrast':
        caption='Per-layer 2-98% contrast enhancement: colors show relative within-layer display strength, NOT probability. Do not compare color intensity across layers.'
        color_norm=Normalize(0,1)
        label='Within-layer display strength'
    else:
        caption='Visual-conditional probability; one logarithmic color scale shared by all 32 layers in this case. M = raw Image mass.'
        color_norm=log
        label='Conditional visual probability (log scale)'
    fig.text(.012,.905,caption,fontsize=10)
    cax=fig.add_axes([.955,.14,.009,.67])
    fig.colorbar(ScalarMappable(norm=color_norm,cmap=CMAP),cax=cax,label=label)
    fig.text(.012,.025,'Native head-averaged 24 x 24 maps. No probability values changed; no new inference. Outside the CLIP center crop is unobserved.',fontsize=10)
    return fig


def selected_plate(records,samples):
    selected=read(OUT/'showcase_selection.json')['question_ids']
    fig=plt.figure(figsize=(7.2,7.6))
    layout=fig.add_gridspec(8,6,height_ratios=[.23,1]*4,width_ratios=[1]*5+[.045],
        left=.015,right=.965,top=.985,bottom=.035,wspace=.08,hspace=.16)
    for i,qid in enumerate(selected):
        sample=samples[qid]
        result=records[qid]['result']
        image=Image.open(OUT/'case_images'/f'{qid}.png').convert('RGB')
        raw=np.asarray([result['spatial_maps'][str(k)] for k in range(32)],float)
        mass=np.asarray([r['image_mass'] for r in result['rows']])
        maps=raw/mass[:,None,None]
        log,_=display_scales(maps)
        textax=fig.add_subplot(layout[2*i,:5])
        textax.axis('off')
        line=f"{sample['task']} | QID {qid} | Q: {sample['question']} | Full: {result['full_text'].strip()}"
        textax.text(0,.42,'\n'.join(textwrap.wrap(line,91)),ha='left',va='center',fontsize=8)
        for j,k in enumerate([None,0,3,15,31]):
            ax=fig.add_subplot(layout[2*i+1,j])
            ax.imshow(image if k is None else overlay(image,maps[k],result['geometry'],log))
            ax.set_title('Original image' if k is None else f'L{k} | M={mass[k]:.3f}',fontsize=8,pad=3)
            ax.set_axis_off()
        cax=fig.add_subplot(layout[2*i+1,5])
        cb=fig.colorbar(ScalarMappable(norm=log,cmap=CMAP),cax=cax)
        cb.ax.tick_params(labelsize=6)
        cb.set_label('Conditional probability (log)',fontsize=7)
    fig.text(.015,.008,'Shared log scale over all 32 layers within each case; M = raw Image mass. Outside crop is unobserved.',fontsize=7)
    for ext in ['png','pdf']:
        fig.savefig(DEST/f'figure_y_enhanced.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    config=verify()
    assert read(OUT/'completion.json')['status']=='complete'
    DEST.mkdir(exist_ok=True)
    pool=read(OUT/'case_pool.json')['samples']
    samples={s['question_id']:s for s in cohort()}
    metadata,records=[],{}
    style()
    assert float(LogNorm(1e-6,1)(1e-3))==.5
    with (DEST/'patch_attention_all32.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['task','question_id','image_id','layer','patch_row','patch_col','raw_probability','conditional_visual_probability'],lineterminator='\n')
        writer.writeheader()
        for item in pool:
            qid=item['question_id']
            sample=samples[qid]
            record=read(OUT/'checkpoints'/f"{sample['task']}__{qid}.json")
            assert record['binding']==digest(config) and record['sample_sha256']==digest(sample)
            assert record['result_sha256']==original_result_digest(record['result'])
            records[qid]=record
            result=record['result']
            original_hash=original_result_digest(result)
            raw=np.asarray([result['spatial_maps'][str(k)] for k in range(32)],float)
            mass=np.asarray([r['image_mass'] for r in result['rows']])
            assert raw.shape==(32,24,24) and np.isfinite(raw).all() and (mass>0).all()
            assert np.allclose(raw.sum((1,2)),mass,atol=1e-12,rtol=0)
            maps=raw/mass[:,None,None]
            assert np.allclose(maps.sum((1,2)),1,atol=1e-12,rtol=0)
            log,local=display_scales(maps)
            image=Image.open(OUT/'case_images'/f'{qid}.png').convert('RGB')
            folder=DEST/qid
            folder.mkdir(exist_ok=True)
            for layer in range(32):
                enhanced=overlay(image,maps[layer],result['geometry'],log)
                save_individual(folder,image,enhanced,maps[layer],local[layer],sample,layer,mass[layer])
                writer.writerows(dict(task=sample['task'],question_id=qid,image_id=sample['image_id'],layer=layer,
                    patch_row=y,patch_col=x,raw_probability=float(raw[layer,y,x]),conditional_visual_probability=float(maps[layer,y,x]))
                    for y,x in np.ndindex(24,24))
            with PdfPages(folder/'all32_views.pdf') as pdf:
                for kind in ['shared_log_overlay','local_contrast','pure_heatmap']:
                    fig=contact(sample,result,image,maps,log,local,kind)
                    fig.savefig(folder/f'all32_{kind}.png',dpi=160,bbox_inches='tight',facecolor='white')
                    pdf.savefig(fig,bbox_inches='tight',facecolor='white')
                    plt.close(fig)
            assert original_result_digest(result)==original_hash
            metadata.append(dict(task=sample['task'],question_id=qid,image_id=sample['image_id'],question=sample['question'],
                full_text=result['full_text'],conditional_log_vmin=log.vmin,conditional_log_vmax=log.vmax,
                per_layer_contrast=[dict(layer=k,conditional_p2=n.vmin,conditional_p98=n.vmax) for k,n in enumerate(local)],
                original_checkpoint_sha256=sha256(OUT/'checkpoints'/f"{sample['task']}__{qid}.json")))
            print('Rendered',sample['task'],qid,'all32 layers',flush=True)
    selected_plate(records,samples)
    freeze_json(DEST/'display_manifest.json',dict(status='complete',parent_binding=digest(config),cases=12,layers=list(range(32)),
        map_count=384,csv_rows=221184,new_gpu_inference=False,source_sha256=sha256(ROOT/'first_answer_mass/all_layers.py'),
        data_unchanged=True,shared_log='Visual-conditional probability, one actual positive-min/maximum log color scale over all32 layers per case.',
        local_contrast='Per-layer p2-p98 clipped display normalization only. Not probability or cross-layer comparable intensity.',
        colormap=CMAP,overlay_alpha='0.15 + 0.7 * display strength inside observed crop;0 outside.',samples=metadata))
    lines=['# Full L0-L31 Case Gallery','',
        '12 frozen candidates x all32 layers =384 native maps; no additional GPU inference. Original probabilities are unchanged.','',
        '**Shared-log overlay / pure heatmap:** visual-conditional probability on one log scale across all32 layers of a case. Raw Image mass is labelled M.','',
        '**Local contrast:** each layer uses its own 2nd-98th percentile display limits. This is for inspecting spatial patterns ONLY; colors are not probability and their intensity cannot be compared across layers.','',
        '[Enhanced four-case PNG](figure_y_enhanced.png) / [PDF](figure_y_enhanced.pdf). Original linear/raw figures remain unchanged in the parent directory.','',
        'Each individual L00.png-L31.png shows original image, shared-log overlay and local-contrast pure heatmap. Each all32_views.pdf has three pages: shared-log overlay, local-contrast overlay, and pure shared-log heatmaps.','',
        '| Task | QID | All32 shared log | Local contrast | Pure heatmap | Three-page PDF |','|---|---|---|---|---|---|']
    for r in metadata:
        q=r['question_id']
        lines.append(f"| {r['task']} | {q} | [PNG]({q}/all32_shared_log_overlay.png) | [PNG]({q}/all32_local_contrast.png) | [PNG]({q}/all32_pure_heatmap.png) | [PDF]({q}/all32_views.pdf) |")
    lines+=['','[Full32-layer raw and conditional patch data](patch_attention_all32.csv). The display_manifest.json records exact display limits and all original checkpoint hashes.']
    (DEST/'README.md').write_text('\n'.join(lines)+'\n')
    freeze_json(DEST/'artifact_index.json',dict(files={str(p.relative_to(DEST)):sha256(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='artifact_index.json'}))


if __name__=='__main__':
    main()
