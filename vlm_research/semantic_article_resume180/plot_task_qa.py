"""Improve task tick-label legibility; uses saved statistics only."""
import shutil
from .common import *

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((P/'task_accuracy.csv').open()));winner=read(P/'completion.json')['winner'];names=[winner,'Semantic-k','S-assistant','Question-Only','All-Writer']
    matrix=np.array([[float(next(x['ACC'] for x in rows if x['condition']==n and x['task']==t))-float(next(x['ACC'] for x in rows if x['condition']=='All-Writer' and x['task']==t)) for t in TASKS] for n in names])
    archive=P/'figure_qa_pass';archive.mkdir()
    for f in ['task_delta_accuracy.png','task_delta_accuracy.pdf','artifact_index.csv','completion.json']:shutil.copyfile(P/f,archive/f)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42});fig,ax=plt.subplots(figsize=(11,4.8));lim=max(1,float(abs(matrix).max()));im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim);ax.set_yticks(range(len(names)),names)
    labels={'ChooseAttr':'Choose\nattribute','ChooseCat':'Choose\ncategory','ChooseRel':'Choose\nrelation','CompareAttr':'Compare\nattributes','LogicalObj':'Logical\nobject','QueryAttr':'Query\nattribute'};ax.set_xticks(range(len(TASKS)),[labels.get(t,t) for t in TASKS],fontsize=9)
    for i in range(len(names)):
        for j in range(len(TASKS)):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',color='white' if abs(matrix[i,j])>lim*.6 else 'black')
    fig.colorbar(im,ax=ax,label='Accuracy difference from All-Writer (pp)');ax.set_title('Six GQA tasks, 500 questions each');fig.tight_layout();fig.savefig(P/'task_delta_accuracy.png',dpi=220);fig.savefig(P/'task_delta_accuracy.pdf');plt.close(fig)
    atomic_json(P/'figure_QA.json',dict(UTC=utc(),task_ticks_wrapped=True,numerical_data_unchanged=True,source_sha256=sha(__file__),previous_figures_archived=True));print('TASK_PLOT_RELAYOUT_COMPLETE')

if __name__=='__main__':main()
