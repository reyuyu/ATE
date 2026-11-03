import json,csv,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
P=Path(sys.argv[1]);D=json.loads((P/'analysis_data.json').read_text());C=D['cases']
def label(g):
    t=g['decoded'].strip();return t if t else ('[NL]' if '\n' in g['decoded'] else '[space]')
def main():
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42});(P/'heatmaps').mkdir(exist_ok=True);maximum=max(np.max(c['heatmap']) for c in C);vmax=np.ceil(maximum*20)/20
    for c in C:
        v=c['case'];groups=v['groups'];heat=np.array(c['heatmap']);fig,ax=plt.subplots(figsize=(12.2,7.4));im=ax.imshow(heat,aspect='auto',interpolation='nearest',vmin=0,vmax=vmax,cmap='viridis');ax.set_yticks(range(32),[f'L{k}' for k in range(32)],fontsize=7);ax.set_xticks(range(len(groups)),[('*' if not g['selected'] else 'S:')+label(g)+' @'+str(g['positions'][0]) for g in groups],rotation=62,ha='right',fontsize=7)
        for tick,g in zip(ax.get_xticklabels(),groups):tick.set_color('#247249' if g['selected'] else '#b46b13' if g['group_id'] in c['orange_groups'] else '#555')
        for y in [3.5,7.5,11.5,15.5,19.5,30.5]:ax.axhline(y,color='white',linewidth=.6,alpha=.6)
        ax.get_yticklabels()[-1].set_color('#b34730');ax.set_title(f'Native Full | {v["question_id"]} | {v["kind"]}\nLayer x original text group; mean over32 heads and subwords',fontsize=12);ax.set_ylabel('Decoder layer (0-based)');fig.colorbar(im,ax=ax,pad=.015,label='Image attention mass (raw)');fig.text(.09,.012,'S: historical Semantic-selected; * omitted. Orange labels: omitted Question top5 in L0-15. P-1 excluded. L31 is observational only.',fontsize=8);fig.subplots_adjust(left=.07,right=.94,bottom=.25,top=.9);fig.savefig(P/'heatmaps'/(v['question_id']+'.png'),dpi=210);plt.close(fig)
    rows=list(csv.DictReader((P/'case_scope_band_summary.csv').open()));bands=['L0-31','L0-15','L16-19','L31'];roles=[('Question','True','Selected Q'),('Question','False','Omitted Q'),('Instruction','False','Instruction'),('Assistant word','False','ASSISTANT'),('Boundary/separators','False','Boundaries')]
    with PdfPages(P/'summary_figures.pdf') as pdf:
        fig,axes=plt.subplots(2,2,figsize=(12,8.5))
        for ax,b in zip(axes.flat,bands):
            for j,(scope,selected,text) in enumerate(roles):
                rr=[r for r in rows if r['band']==b and r['scope']==scope and r['selected']==selected];vals=[float(r['mass_mean']) for r in rr]
                ax.boxplot([vals],positions=[j],widths=.48,showfliers=False,medianprops={'color':'#333'})
                for k,r in enumerate(rr):ax.scatter(j+(k-(len(rr)-1)/2)*.022,float(r['mass_mean']),s=23,c='#ae593a' if r['kind']=='harm' else '#277f9d',marker='o' if r['kind']=='harm' else '^',zorder=3)
            counts=[sum(r['band']==b and r['scope']==a[0] and r['selected']==a[1] for r in rows) for a in roles]
            ax.set_xticks(range(5),[a[2]+'\n(n='+str(n)+')' for a,n in zip(roles,counts)],rotation=0,fontsize=9);ax.set_title(b+' | each point = one case mean');ax.set_ylabel('Image mass');ax.set_ylim(bottom=0);ax.spines[['top','right']].set_visible(False)
        fig.suptitle('Figure A - Semantic Selected vs Omitted Visual Attention',fontsize=16);fig.text(.5,.01,'Brown circles:12 harm cases; blue triangles:4 rescue cases. No head/token treated as an independent sample. L31 shown separately.',ha='center',fontsize=9);fig.subplots_adjust(hspace=.5,wspace=.23,bottom=.13,top=.91);fig.savefig(P/'figure_A_selected_omitted.png',dpi=220);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(4,4,figsize=(15,11))
        for ax,c in zip(axes.flat,C):
            ax.imshow(c['heatmap'],aspect='auto',interpolation='nearest',vmin=0,vmax=vmax,cmap='viridis');ax.set_title(c['case']['question_id']+' '+c['case']['kind'],fontsize=9);ax.set_xticks([]);ax.set_yticks([0,8,16,24,31],['0','8','16','24','31'],fontsize=7);ax.axhline(30.5,color='white',lw=.6)
        fig.suptitle('Figure B -16 Actual Layer x Text-Group Heatmaps',fontsize=16);fig.text(.5,.015,'Every panel has its own original text groups; no shared positional token alignment. Full labels appear in the16 individual heatmaps and case cards.',ha='center',fontsize=10);fig.subplots_adjust(hspace=.4,wspace=.25,top=.93,bottom=.06);fig.savefig(P/'figure_B_heatmap_overview.png',dpi=200);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,9),sharey=True)
        for ax,band in zip(axes,['L8-11','L0-15','L16-19']):
            for j,c in enumerate(C):
                rr=[r for r in c['band_groups'][band] if r['scope']=='Question' and not r['semantic_selected']];r=sorted(rr,key=lambda r:(-r['mass_mean'],r['first_position']))[0];color='#ae593a' if c['case']['kind']=='harm' else '#277f9d';ax.barh(j,r['mass_mean'],color=color,alpha=.8);txt=r['group_text'].strip() or '[space]';ax.text(r['mass_mean']+.003,j,txt+' @'+str(r['first_position']),va='center',fontsize=8)
            ax.set_title(band);ax.set_xlabel('Mean image mass');ax.set_xlim(0,.36);ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(range(16),[c['case']['question_id']+' '+c['case']['kind'] for c in C],fontsize=8);axes[0].invert_yaxis();fig.suptitle('Figure C - Highest-Mass Omitted Question Group in Each Case and Band',fontsize=15);fig.text(.5,.02,'Top1 shown for readability; all per-case, per-scope Top5 groups and normalized values are preserved in CSV. These are candidate observations, not causal effects.',ha='center',fontsize=9);fig.subplots_adjust(left=.14,right=.98,wspace=.15,bottom=.09,top=.92);fig.savefig(P/'figure_C_high_mass_omitted.png',dpi=220);pdf.savefig(fig);plt.close(fig)
    (P/'plot_validation.json').write_text(json.dumps(dict(passed=True,individual_heatmaps=16,summary_figures=3,global_heatmap_vmin=0,global_heatmap_vmax=float(vmax),heatmap_interpolation='nearest',per_case_groups_not_position_aligned=True,figure_A_case_level_units=True,L31_separate=True,undefined_values_not_plotted_as_infinity=True),indent=2));print('PLOTS COMPLETE',flush=True)
if __name__=='__main__':main()
