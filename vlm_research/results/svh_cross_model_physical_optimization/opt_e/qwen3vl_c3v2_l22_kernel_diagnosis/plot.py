"""Reproducible static scientific plots; CPU only."""
import pathlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=pathlib.Path(__file__).resolve().parents[2]/'vlm_research/results/qwen3vl_c3v2_l22_kernel_diagnosis'
def rows(name):return list(csv.DictReader((O/name).open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
f,ax=plt.subplots(1,3,figsize=(14,4.3),layout='constrained');names=['k_proj','v_proj','gate_proj','up_proj','down_proj'];xx=np.arange(5);rr=rows('same_input_operator_comparison.csv')
for cond,offset,col,label in [('torch_physical26',-.18,'#718494','Native 26 rows'),('Lt_dense_heuristic_0',.18,'#138a72','Fixed Lt, 26 valid rows')]:
 vals=[float(next(r['mean_abs'] for r in rr if r['rank']=='0' and r['operator']==n and r['candidate']==cond)) for n in names];ax[0].bar(xx+offset,vals,.34,label=label,color=col)
ax[0].set_xticks(xx,[n.replace('_proj','').upper() for n in names]);ax[0].set_ylabel('Mean absolute operator error');ax[0].set_title('A  Same-input L22 projections');ax[0].legend(fontsize=8);ax[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax[0].set_xlabel('Fixed-Lt: all errors = 0; Down kept native',fontsize=8)
tim=rows('microbenchmark.csv')
for cond,off,col,label in [('torch_physical26',-.18,'#718494','Native 26'),('Lt_fixed_0__physical26',.18,'#138a72','Fixed Lt')]:
 vals=[float(next(r['gpu_event_ms_median'] for r in tim if r['rank']=='0' and r['label']==n+'__'+cond)) for n in names];ax[1].bar(xx+off,vals,.34,color=col,label=label)
ax[1].set_xticks(xx,[n.replace('_proj','').upper() for n in names]);ax[1].set_ylabel('Local GPU-event median (ms)');ax[1].set_title('B  5 warmups / 20 repeats');ax[1].text(.5,.96,'Down Lt is diagnostic only',transform=ax[1].transAxes,ha='center',va='top',fontsize=9);ax[1].legend(fontsize=8)
lg=rows('logits_comparison.csv');new=[float(next(r['mean_abs'] for r in lg if r['rank']=='0' and r['phase']=='final_fixed_L22_L35' and r['comparison']=='F__C_L35' and r['step']==str(i))) for i in [0,1]]
ax[2].bar(np.array([0,1])-.18,[.05232971906661987,.060191113501787186],.34,label='Parent frozen baseline',color='#718494');ax[2].bar(np.array([0,1])+.18,new,.34,label='Fixed Lt L22-L35',color='#d9853b');ax[2].axhline(.02,color='#c33',ls='--',label='Required mean <= 0.02');ax[2].set_xticks([0,1],['First token','EOS']);ax[2].set_ylabel('Mean absolute valid-logit error');ax[2].set_title('C  Full F/C still fails');ax[2].legend(fontsize=8,loc='upper left');ax[2].set_ylim(0,.10)
(O/'figures').mkdir(exist_ok=True)
for ext in ['png','pdf']:f.savefig(O/'figures'/('kernel_error_and_latency.'+ext),dpi=180)
plt.close(f)
# Honest launch comparison: avoid suggesting that valid M alone equals issued work.
f,ax=plt.subplots(figsize=(10,3),layout='constrained');ax.axis('off');data=[['K / V','Native 26: 64x64 sliced','Dense/fixed: 128x64 + split-K','26 valid; tile M capacity 64'],['Gate / Up','Native 26: CUTLASS 64x64','Dense/fixed: CUTLASS 256x128','26 valid; same Dense launch grid'],['Down','Native 26: CUTLASS 64x64','Control retained native','Already exact; no replacement']];t=ax.table(cellText=data,colLabels=['Operator','Native short kernel','Dense-compatible fixed kernel','Work limitation'],loc='center',cellLoc='left',colWidths=[.12,.25,.31,.32]);t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,2);ax.set_title('Observed CUDA launches: matching numerical behavior is not an E2E speedup',pad=15)
for ext in ['png','pdf']:f.savefig(O/'figures'/('actual_kernel_comparison.'+ext),dpi=180)
