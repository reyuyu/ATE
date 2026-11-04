"""Layout-only revisions from frozen plotting CSV; no data or tests changed."""
import pathlib,json,csv,textwrap
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
out=pathlib.Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
rr=list(csv.DictReader((out/'figure_E_plot_data.csv').open()));fig,ax=plt.subplots(figsize=(9,5));colors=dict(A='#444444',B='#777777',C='#E69F00',D='#CC79A7',E='#0072B2',F='#E69F00',G='#CC79A7');offsets=dict(A=(7,0),B=(7,-9),C=(-16,10),D=(9,-14),E=(-10,10),F=(8,7),G=(8,-12))
for j,r in enumerate(rr):
 c=r['condition'];x=100*float(r['mean_actual_edge_ratio']);y=float(r['accuracy_pct']);ax.scatter(x,y,color=colors.get(c,'#0072B2'),s=48);offset=offsets.get(c,(7,7 if j%2 else -12));ax.annotate(c,(x,y),xytext=offset,textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.6,color='gray') if c in ['C','D'] else None)
ax.set_xlabel('Mean actual allowed answer-query to Image edges (%)');ax.set_ylabel('Official TextVQA consensus accuracy (%)');ax.set_title('E. Actual query-weighted edge permissions; TextVQA300\nPoint estimates; condition-dependent lengths; dense execution retained');ax.margins(x=.07);fig.tight_layout();save(fig,'figure_E_actual_edges')
selection=json.loads((out/'figure_F_case_selection.json').read_text())['cases'];categories=['recovery_vs_B','harm_vs_B','higher_budget_harms','Full_credit_but_all_main_sparse_below_full_credit','decode_changes_later_tokens','output_anomaly'];titles=['Recovery vs B','Harm vs B','More layers can harm','Sparse misses native Full','Decode changes continuation','Output anomaly']
fig=plt.figure(figsize=(15,13));gs=fig.add_gridspec(4,3,height_ratios=[1.45,1.35,1.45,1.65],hspace=.23,wspace=.13)
for j,(cat,title) in enumerate(zip(categories,titles)):
 row=(j//3)*2;col=j%3;image_ax=fig.add_subplot(gs[row,col]);text_ax=fig.add_subplot(gs[row+1,col]);image_ax.axis('off');text_ax.axis('off');image_ax.set_title(title,fontsize=11);case=next((x for x in selection if x['category']==cat),None)
 if case is None:text_ax.text(0,1,'No case under frozen selection rule',va='top');continue
 with Image.open(case['image_path']) as im:image_ax.imshow(im.convert('RGB'))
 refs=json.loads(case['answers']);counts={s:refs.count(s) for s in refs};gold=max(counts,key=counts.get);lines=['ID '+case['question_id'],textwrap.fill(case['question'],54),'Reference (one modal annotation):',textwrap.fill(gold,52)]
 sides=['left'] if case['left']==case['right'] else ['right','left']
 for side in sides:lines+=['',case[side]+f" [score {float(case[side+'_score']):.1f}]:",textwrap.fill(case[side+'_answer'],56)]
 if case['category']=='decode_changes_later_tokens':lines+=['','Literal first model token unchanged; later sequence differs.']
 if case['category']=='output_anomaly':lines+=['','Native anomaly retained; no sample excluded.']
 if case['note']:lines+=['',case['note']]
 text_ax.text(0,1,'\n'.join(lines),va='top',transform=text_ax.transAxes,fontsize=8.5,linespacing=1.25)
fig.suptitle('F. First frozen-manifest example per category; original photo aspect ratios preserved',fontsize=13,y=.995);fig.subplots_adjust(top=.95,bottom=.02,left=.025,right=.99);save(fig,'figure_F_cases')
print('LAYOUT_ONLY_REPLOT_PASS')
