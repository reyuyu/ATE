"""Publication-only legend layout from frozen CSV; no model or score changes."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'hidden_state_summary.csv').open()))
binding=rows[0]['protocol_sha256']
assert all(r['protocol_sha256']==binding for r in rows)
layers=[2,6,9,13,17,31]
colors={'E':'#287da3','M':'#5f6594','EM':'#bb4b42','L':'#258574'}
labels={'E':'E-OFF','M':'M-OFF','EM':'E+M-OFF','L':'Late-OFF'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(8,3.6))
fig.subplots_adjust(top=.73,bottom=.20,left=.08,right=.98,wspace=.34)
for key in colors:
    subset={int(r['layer_input']):r for r in rows if r['condition']==key}
    for ax,metric,title in zip(axes,['mean_relative_l2_distance','mean_cosine_distance'],['Relative L2 distance','Cosine distance']):
        ax.plot(layers,[float(subset[k][metric]) for k in layers],marker='o',lw=1.3,color=colors[key],label=labels[key])
        ax.set_xticks(layers);ax.set_xlabel('Decoder layer input');ax.set_ylabel(title)
        ax.spines[['top','right']].set_visible(False)
handles,names=axes[0].get_legend_handles_labels()
fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.92),ncol=4,frameon=False,fontsize=8)
fig.suptitle('P-1 State Propagation',y=.995,fontweight='bold',fontsize=11)
for ext in ['png','pdf']:
    metadata={'Protocol':binding} if ext=='png' else {'Subject':'Protocol SHA256 '+binding}
    fig.savefig(OUT/f'p1_state_propagation.{ext}',dpi=300,bbox_inches='tight',metadata=metadata)
plt.close(fig)
print('Updated legend layout only; unchanged CSV binding',binding)
