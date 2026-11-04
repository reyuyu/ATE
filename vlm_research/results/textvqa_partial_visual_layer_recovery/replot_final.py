"""Layout-only final figures; inferential data and frozen execution source unchanged."""
import pathlib,csv,json,sys
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
out=pathlib.Path(__file__).resolve().parent
summary=list(csv.DictReader((out/'accuracy_summary.csv').open()));budgets=list(csv.DictReader((out/'phase_layer_budget_mass.csv').open()));colors=['#444444','#0072B2','#E69F00','#CC79A7','#009E73']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
vals=[float(r['accuracy_pct']) for r in summary];err=[[v-float(r['CI95_low_pct']) for v,r in zip(vals,summary)],[float(r['CI95_high_pct'])-v for v,r in zip(vals,summary)]]
fig,ax=plt.subplots(figsize=(8.5,4.5));ax.bar(range(5),vals,color=colors,yerr=err,capsize=4);ax.set_xticks(range(5),['A Native\nFull','B All-\nWriter','C P-1\n14+19','D Decode\n14+19','E Both\n14+19']);ax.set_ylabel('Official TextVQA consensus accuracy (%)');ax.set_ylim(0,58)
for i,r in enumerate(summary):ax.text(i,float(r['CI95_high_pct'])+1.2,f"{float(r['accuracy_pct']):.2f}",ha='center',va='bottom')
ax.set_title('Same frozen TextVQA300; no OCR text\nPointwise image-cluster bootstrap 95% CI');fig.tight_layout();save(fig,'figure_accuracy')
fig,ax=plt.subplots(figsize=(8.5,4.8));axins=inset_axes(ax,width='43%',height='53%',loc='center right',borderpad=2)
for i,r in enumerate(summary):
 fraction=(int(r['Pminus1_allowed_layers'])+int(r['Decode_allowed_layers']))/64*100;value=float(r['accuracy_pct']);ax.scatter(fraction,value,color=colors[i],s=70,label=r['condition']+' '+r['name'])
 if r['condition']!='A':
  axins.scatter(fraction,value,color=colors[i],s=55)
  offsets=dict(B=(6,3),C=(-17,6),D=(7,-13),E=(6,3));axins.annotate(r['condition'],(fraction,value),xytext=offsets[r['condition']],textcoords='offset points',fontsize=9)
axins.set_xlim(-.7,7.7);axins.set_ylim(20,29);axins.set_xticks([0,3.125,6.25],['0','3.125','6.25']);axins.set_title('Sparse-permission detail',fontsize=9);axins.tick_params(labelsize=8)
ax.set_xlim(-5,105);ax.set_ylim(19,49);ax.legend(loc='upper left',fontsize=8,frameon=False);ax.set_xlabel('Allowed answer-phase layer permissions (unweighted %)');ax.set_ylabel('Official TextVQA accuracy (%)');ax.set_title('Functional phase/layer permissions and answer quality\nDense QKV, MLP and KV updates retained');fig.subplots_adjust(bottom=.18,top=.82,left=.1,right=.98);save(fig,'figure_functional_budget')
fig,axes=plt.subplots(2,1,figsize=(11,6.4));vmax=max(float(r['actual_image_mass_headmean']) for r in budgets)
for ax,phase in zip(axes,['Pminus1','Decode']):
 values=np.array([[float(next(r['actual_image_mass_headmean'] for r in budgets if r['condition']==c and r['phase']==phase and int(r['layer'])==l)) for l in range(32)] for c in 'ABCDE']);im=ax.imshow(values,aspect='auto',vmin=0,vmax=vmax,cmap='viridis');ax.set_yticks(range(5),list('ABCDE'));ticks=[0,4,8,12,14,16,19,20,24,28,31];ax.set_xticks(ticks);ax.set_title(phase+' image attention mass (head/query mean)');ax.set_xlabel('Zero-based decoder layer');fig.colorbar(im,ax=ax,label='Probability mass')
fig.text(.5,.015,'B: zero inferred from audited historical actual forbidden probabilities; other rows freshly measured.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,1]);save(fig,'figure_actual_image_mass')
print('LAYOUT_REPLOT_PASS')
