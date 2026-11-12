"""Editable CVPR-size vector plot from measured, validated session means only.

Usage: python s55_plot.py --csv latency_two_sessions.csv --output latency_vs_tokens.pdf
Error bars show min/max of TWO session means, not a confidence interval.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--csv',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 with args.csv.open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==12
 matplotlib.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.labelsize':9,'legend.fontsize':7.5,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7,'savefig.dpi':300})
 fig,ax=plt.subplots(figsize=(3.5,2.65));palette={'Native':('#333333','o','Native Full'),'SVH':('#0072B2','s','SVH'),'SVH-R':('#D55E00','^','SVH-R')};metadata={}
 for method,(color,marker,label) in palette.items():
  data=sorted((r for r in rows if r['method']==method),key=lambda r:int(r['output_tokens']))
  x=[int(r['output_tokens']) for r in data];assert x==[16,32,64,128]
  y=[float(r['E2E_ms']) for r in data];s1=[float(r['E2E_ms_session1']) for r in data];s2=[float(r['E2E_ms_session2']) for r in data]
  assert all(v>0 for v in y+s1+s2)
  assert all(abs(a-(b+c)/2)<1e-6 for a,b,c in zip(y,s1,s2))
  err=[[m-min(a,b) for m,a,b in zip(y,s1,s2)],[max(a,b)-m for m,a,b in zip(y,s1,s2)]]
  ax.errorbar(x,y,yerr=err,color=color,label=label,marker=marker,markersize=4,linewidth=1.45,elinewidth=.8,capsize=2,capthick=.8)
  metadata[method]={'tokens':x,'measured_E2E_ms':y,'session1':s1,'session2':s2}
 ax.set(xlabel='Number of Generated Tokens',ylabel='End-to-End Latency (ms)',xticks=[16,32,64,128],xlim=(10,134),ylim=(0,None))
 ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.grid(axis='y',color='#dddddd',linewidth=.45);ax.set_axisbelow(True);ax.legend(loc='upper left',frameon=False,handlelength=1.6,labelspacing=.45)
 fig.tight_layout(pad=.65);args.output.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(args.output,format='pdf',metadata={'Title':'End-to-End Latency vs. Generated Tokens','Subject':'RTX3090, LLaVA-1.5-7B, FP16/Eager, two independent sessions; error bars: session-mean range','Creator':'Matplotlib; source s55_plot.py'})
 fig.savefig(args.output.with_suffix('.png'),dpi=300);plt.close(fig)
 receipt={'csv_sha256':hashlib.sha256(args.csv.read_bytes()).hexdigest(),'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'only_measured_data':True,'formal_requests':720,'sessions':2,'error_bars':'range of two session means, NOT confidence interval','data':metadata}
 args.output.with_suffix('.plot_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(args.output)
if __name__=='__main__':main()
