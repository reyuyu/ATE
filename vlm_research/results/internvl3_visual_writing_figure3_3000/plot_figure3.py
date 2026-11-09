"""Standalone Figure 3 plotting, from fixed CSV only; no torch/model imports."""
import csv,json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

TASKS=["ChooseAttr","ChooseCat","ChooseRel","CompareAttr","LogicalObj","QueryAttr"]
COLORS=["#087e8b","#c73e48","#4c956c","#735d9b","#d08c18","#386cb0"]
MODELS=["LLaVA-1.5-7B","Qwen3-VL-8B","InternVL3-8B"]
VMAX=10.0  # Common display scale, fixed before new InternVL accuracy was inspected.
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11,
    "axes.labelsize":10,"legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42,
    "savefig.dpi":320,"axes.spines.top":False,"axes.spines.right":False})

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def draw(directory,models,name):
    layer_rows=rows(directory/"plot_layerwise.csv");deep_rows=rows(directory/"plot_deep.csv")
    nrows=len(models)
    fig,axes=plt.subplots(nrows,3,figsize=(17.2,3.15*nrows+0.7),squeeze=False,
        gridspec_kw={"width_ratios":[1.18,1.44,.86],"wspace":.33,"hspace":.65})
    overflow=[]
    for ridx,model in enumerate(models):
        selected=[r for r in layer_rows if r["model"]==model]
        d=next(r for r in deep_rows if r["model"]==model)
        count=int(d["layers"]);start=int(d["deep_start"]);x=np.arange(count)
        ax,heat,bars=axes[ridx]
        for task,color in zip(TASKS,COLORS):
            y=[float(next(r for r in selected if r["task"]==task and int(r["layer"])==i)["delta_accuracy_pp"]) for i in x]
            ax.plot(x,y,lw=1.15,marker="o",ms=2.6,color=color,label=task)
        ax.axhline(0,color=".45",lw=.7)
        ax.set(xlim=(-.45,count-.55),xlabel="Layer",ylabel=r"$\Delta$ACC (pp)")
        ax.set_xticks(sorted(set(list(range(0,count,4))+[count-1])))
        ax.margins(y=.12);ax.grid(axis="y",alpha=.18,linewidth=.6)
        ax.text(0,1.06,f"{model} ({count} layers)",transform=ax.transAxes,fontweight="bold",fontsize=11)
        matrix=np.array([[float(next(r for r in selected if r["task"]==task and int(r["layer"])==i)["damage_percent"])
                          for i in x] for task in TASKS])
        mesh=heat.pcolormesh(np.arange(count+1)-.5,np.arange(7)-.5,matrix,
                            cmap="Blues",vmin=0,vmax=VMAX,shading="flat",rasterized=False)
        heat.set(xlim=(-.5,count-.5),ylim=(5.5,-.5),xlabel="Layer")
        heat.set_yticks(range(6),TASKS)
        heat.set_xticks(sorted(set(list(range(0,count,4))+[count-1])))
        heat.axvline(start-.5,ls="--",color="#dc6b31",lw=1.5)
        for ti,li in zip(*np.nonzero(matrix>VMAX)):
            value=matrix[ti,li]
            heat.text(li,ti,f"{value:.1f}",ha="center",va="center",fontsize=5.4,color="white",fontweight="bold")
            heat.plot(li,ti-.30,marker="^",ms=2.4,color="#ffb166",clip_on=False)
            overflow.append(dict(model=model,task=TASKS[ti],layer=int(li),damage_percent=float(value)))
        heat.tick_params(axis="y",length=0)
        vals=[float(d["all_writer_accuracy_pct"]),float(d["restricted_accuracy_pct"])]
        bars.bar([0,1],vals,color=["#c1c8d4","#4c79a8"],width=.60,edgecolor="white",linewidth=.5)
        bars.set(ylim=(0,100),ylabel="Accuracy (%)")
        bars.set_xticks([0,1],["All-Writer",f"L{start}–L{count-1} OFF"])
        bars.set_yticks([0,25,50,75,100]);bars.grid(axis="y",alpha=.18,lw=.6);bars.set_axisbelow(True)
        for i,value in enumerate(vals):bars.text(i,value+1.5,f"{value:.2f}",ha="center",va="bottom",fontsize=10)
        for j,title in enumerate(["(a) Layer-wise ACC change","(b) Task-wise answer damage","(c) Deep-layer restriction"]):
            if ridx==0:axes[ridx,j].set_title(title,pad=30)
    # Shared heatmap scale, with explicit markers for every value above display maximum.
    fig.subplots_adjust(left=.065,right=.95,bottom=.16 if nrows==1 else .085,top=.82 if nrows==1 else .91)
    cax=fig.add_axes([.965,.27 if nrows==1 else .18,.009,.39 if nrows==1 else .57])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,VMAX),cmap="Blues"),cax=cax,extend="max")
    cb.set_label("Damage (%)");cb.set_ticks([0,2,4,6,8,10])
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.36,.005 if nrows==1 else .012),
               ncol=6,frameon=False,columnspacing=1.25,handlelength=1.6)
    for ext in ["png","pdf"]:fig.savefig(directory/f"{name}.{ext}",dpi=320,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return overflow

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--directory",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();d=args.directory
    overflow=draw(d,["InternVL3-8B"],"figure3_internvl_row")
    all_overflow=draw(d,MODELS,"figure3_three_models")
    (d/"figure_render_validation.json").write_text(json.dumps(dict(rendered=True,png_dpi=320,PDF_vector_heatmap=True,
        common_heatmap_range=[0,VMAX],overflow_marked_with_triangle_and_actual_number=all_overflow,
        no_smoothing=True,negative_delta_preserved=True,bar_y_range=[0,100],
        task_palette=dict(zip(TASKS,COLORS)),palette_source="96a66a30: qwen3vl_visual_writing_necessity/analyze.py",
        exact_uploaded_Figure3_image_available=False,
        style_note="No separate reference Figure 3 image was attached; layout follows the user's explicit 3x3 specification and frozen Qwen task palette."),indent=2))
    caption="""Figure 3. Layer-wise necessity of visual writing on the same six-task GQA cohort (3,000 questions; 2,409 image SHA256 clusters). Rows show LLaVA-1.5-7B (32 layers), Qwen3-VL-8B (36 layers), and InternVL3-8B (28 layers). In every condition, the original last prefill query (P−1) and all autoregressive decoding queries cannot directly attend to original image keys. All-Writer retains visual access for every post-image, pre-P−1 text/template query R; it is not Native Full. (a) Each measured point is ACC(All-Writer) minus ACC(Drop-Li), in percentage points, for one task; negative values indicate improvements after the ablation. (b) Answer damage is 100 times the number of All-Writer-correct answers made incorrect by Drop-Li, divided by the task's All-Writer-correct count. All rows share a blue 0–10% scale; triangles and numeric labels explicitly identify values above 10%. Dashed lines mark the preselected deep-layer boundary. (c) All-Writer versus joint visual-writing restriction at L20–L31, L22–L35, and L17–L27, respectively. Bars report overall accuracy; ordinary paired 95% image-cluster bootstrap intervals for the differences are provided in the accompanying data and report. LLaVA/Qwen values are read from pinned historical CSV files; InternVL is newly measured with 20,000 paired image-cluster bootstrap resamples. Single-layer analyses additionally report Bonferroni-28 intervals for InternVL. Each model retains its own native prompt, architecture, and visual-token budget (LLaVA: 576; Qwen: native variable image tokens and mRoPE/DeepStack; InternVL: one 448-pixel tile, 256 image tokens). These dense permission interventions establish functional effects and do not measure physical pruning, cache compression, or speedup. Cross-model differences are descriptive and do not isolate architecture from preprocessing or prompt differences."""
    (d/"figure_caption_en.md").write_text(caption+"\n")
    print("FIGURE3_RENDER_PASS",flush=True)

if __name__=="__main__":main()
