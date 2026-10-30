"""Post-run descriptive figures/CIs; never fits or selects predictor rules."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmarks.dataset_registry import freeze_json,sha256
from paper_rq1.analyze import write_csv
from bos_cheap.protocol import OUT,EPS
from bos_cheap.predict import load_rows,score_rule,danger,skips
from bos_cheap.analyze import intervals,save,risk_values


def read(name):return list(csv.DictReader((OUT/name).open()))
def number(row,key):return float(row[key]) if row.get(key) else np.nan


def main():
    if not (OUT/'completion.json').exists():raise ValueError('Review requires completed frozen analysis')
    frozen=json.loads((OUT/'frozen_predictors.json').read_text());curves=read('risk_coverage.csv');selected=read('predictor_comparison.csv')
    names=list(dict.fromkeys(r['predictor'] for r in curves));colors={n:plt.get_cmap('tab20')(i/20) for i,n in enumerate(names)}
    fig,axes=plt.subplots(1,3,figsize=(19,6),layout='constrained')
    for ax,epsilon in zip(axes,EPS):
        for name in names:
            for direction in ['high','low']:
                rows=sorted([r for r in curves if r['dataset_part']=='new_exploratory' and float(r['epsilon'])==epsilon and r['predictor']==name and r['direction']==direction and int(r['skipped'])>0 and r['selected_on_calibration']=='False'],key=lambda r:float(r['skip_coverage']))
                if rows:ax.plot([float(r['skip_coverage']) for r in rows],[float(r['dangerous_among_skipped']) for r in rows],'.-' if direction=='high' else '.--',color=colors[name],ms=3,lw=1,label=name+' ('+direction+')')
        ax.axhline(.01,color='#333333',ls=':',lw=1);ax.set_xlabel('Skip coverage');ax.set_ylabel('Dangerous among skipped');ax.set_title(f'epsilon={epsilon}, new exploratory554');ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='outside lower center',fontsize=7)
    fig.suptitle('Frozen threshold sweeps; high/low directions separated, no test-driven envelope or threshold selection')
    save(fig,'figure1_risk_coverage_reviewed')
    fig,axes=plt.subplots(1,3,figsize=(20,6),layout='constrained')
    for part,marker,color in [('calibration','o','#227b96'),('new_exploratory','x','#bd4838')]:
        rows=[r for r in selected if r['dataset_part']==part and float(r['epsilon'])==.03];x=np.arange(len(rows))
        axes[0].scatter(x,[float(r['skip_coverage']) for r in rows],label=part,marker=marker,color=color)
        y=np.array([number(r,'dangerous_among_skipped') for r in rows]);lo=np.array([number(r,'dangerous_among_skipped_ci95_low') for r in rows]);hi=np.array([number(r,'dangerous_among_skipped_ci95_high') for r in rows])
        visible=np.isfinite(y)&(y<=.025)
        axes[1].errorbar(x[visible],y[visible],yerr=[np.maximum(y-lo,0)[visible],np.maximum(hi-y,0)[visible]],fmt=marker,color=color,capsize=2,label=part)
        for i in np.flatnonzero(np.isfinite(y)&~visible):
            axes[1].scatter(i,.024,marker='^',color=color)
            axes[1].annotate(f"{100*y[i]:.0f}% ({rows[i]['dangerous_skipped']}/{rows[i]['skipped']}), off-scale",(i,.024),xytext=(-8,8),textcoords='offset points',ha='right',fontsize=8,color=color)
        if part=='new_exploratory':axes[2].scatter(x,[float(r['simultaneous_hoeffding_risk_upper95']) for r in rows],marker='_',label='Conservative image-cluster bound')
    for ax in axes:
        ax.set_xticks(x,[r['predictor'] for r in rows],rotation=70,ha='right',fontsize=7);ax.legend(fontsize=7);ax.grid(alpha=.15)
    axes[0].set_ylabel('Skip coverage');axes[1].set_ylabel('Danger among skipped, ordinary image-bootstrap95% CI');axes[1].axhline(.01,color='black',ls=':');axes[1].set_ylim(-.0005,.026);axes[2].set_ylabel('Simultaneous upper95, independent-image assumption');axes[2].set_ylim(0,1.05)
    fig.suptitle('epsilon=.03; thresholds froze on calibration. Undefined zero-coverage risks omitted, never shown as safe0.')
    save(fig,'figure7_threshold_comparison_reviewed_v2')
    bins=[];complexity=[]
    rules=frozen['eps']['0.03']['selected']
    for part in ['old','new']:
        rows=load_rows(part)
        if part=='old':rows=[r for r in rows if r['split']=='calibration']
        label='calibration' if part=='old' else 'new_exploratory';y=danger(rows,.03)
        for rule in rules:
            if rule['kind']!='logistic':continue
            p=score_rule(rows,rule);masks=[(p>=lo)&(p<lo+.1 if lo<.9 else p<=1) for lo in np.linspace(0,.9,10)]
            ci=intervals(rows,masks,.03,1000)
            for lo,mask,c in zip(np.linspace(0,.9,10),masks,ci):
                bins.append(dict(dataset_part=label,predictor=rule['name'],epsilon=.03,bin_low=float(lo),conditions=int(mask.sum()),mean_predicted=float(p[mask].mean()) if mask.any() else None,observed_dangerous=float(y[mask].mean()) if mask.any() else None,ci95_low=c['dangerous_among_skipped_ci95_low'],ci95_high=c['dangerous_among_skipped_ci95_high'],image_clusters=c['n_image_clusters']))
    write_csv(OUT/'calibration_bin_image_intervals.csv',bins)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,name in zip(axes,['Layer-only','BOS-QK+Value','BOS-QK+Residual']):
        ax.plot([0,1],[0,1],'k:',lw=.8)
        for part,color in [('calibration','#227b96'),('new_exploratory','#bd4838')]:
            rows=[r for r in bins if r['predictor']==name and r['dataset_part']==part and r['conditions']]
            mean=np.array([r['observed_dangerous'] for r in rows]);lo=np.array([r['ci95_low'] for r in rows]);hi=np.array([r['ci95_high'] for r in rows])
            ax.errorbar([r['mean_predicted'] for r in rows],mean,yerr=[np.maximum(mean-lo,0),np.maximum(hi-mean,0)],fmt='o-',color=color,ms=3,capsize=2,label=part)
        ax.set_title(name);ax.set_xlabel('Mean predicted danger');ax.set_ylabel('Observed danger, image-bootstrap95% CI');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Frozen epsilon=.03 calibration; bin risks evaluated by image, not32 independent layer samples')
    save(fig,'figure6_calibration_reviewed')
    for e,entry in frozen['eps'].items():
        for r in entry['selected']:
            learned=sum(g['parameters']+g['calibration']['parameters'] for g in r['model']['groups'].values()) if r['kind']=='logistic' else 0
            thresholds=32 if r['name']=='Layer-threshold' else 6 if r['name']=='Region-threshold' else 1
            complexity.append(dict(epsilon=float(e),predictor=r['name'],learned_logistic_parameters=learned,threshold_parameters=thresholds,
                feature_columns=json.dumps(r['model']['columns']) if r['kind']=='logistic' else r['column'],
                training_seconds=sum(g['training_seconds']+g['calibration']['training_seconds'] for g in r['model']['groups'].values()) if r['kind']=='logistic' else None,
                no_late_layer_kv_or_attention_outputs_as_online_inputs=r['name']!='BOS-Mass-Oracle'))
    write_csv(OUT/'model_complexity.csv',complexity)
    # Fixed-policy comparison only: these extra/removed sets are not new gates.
    rows=load_rows('new');ids,group=np.unique([r['image_sha256'] for r in rows],return_inverse=True)
    rng=np.random.default_rng(20261030);w=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000).astype(float)
    total=np.bincount(group);comparison=[]
    for epsilon in EPS:
        rules=frozen['eps'][str(epsilon)]['selected'];baseline=next(r for r in rules if r['name']=='Layer-only');bs=skips(rows,baseline)
        bcount=np.bincount(group,weights=bs,minlength=len(ids));boot_baseline=w@bcount;boot_total=w@total
        for rule in rules:
            ss=skips(rows,rule);extra=ss&~bs;removed=bs&~ss;counts=np.bincount(group,weights=ss,minlength=len(ids))
            delta=(w@counts-boot_baseline)/boot_total;lo,hi=np.quantile(delta,[.025,.975]);ci=intervals(rows,[extra,removed],epsilon,2000)
            row=dict(epsilon=epsilon,predictor=rule['name'],baseline='Layer-only',net_coverage_difference=float(ss.mean()-bs.mean()),net_coverage_ci95_low=float(lo),net_coverage_ci95_high=float(hi),interpretation='Post-run descriptive paired audit of frozen policies; added/removed sets are not proposed new gates.')
            for prefix,mask,c in zip(['added','removed'],[extra,removed],ci):
                metrics=risk_values(rows,mask,epsilon)
                row.update({prefix+'_'+k:v for k,v in metrics.items()});row.update({prefix+'_'+k:v for k,v in c.items()})
            comparison.append(row)
    write_csv(OUT/'incremental_over_layer_only.csv',comparison)
    freeze_json(OUT/'postrun_figure_review_v2.json',dict(script_sha256=sha256(Path(__file__)),predictors_sha256=sha256(OUT/'frozen_predictors.json'),fits_or_threshold_changes=0,new_gpu_forwards=0,notes=['Direction-specific threshold sweeps shown separately, no test envelope.','Conditional zero-coverage risks remain undefined.','Calibration bins have whole-image bootstrap intervals.','Threshold risk upper bounds plotted separately from ordinary bootstrap intervals.','Figure7 v2 zooms ordinary risk to2.6%; the100% score-only outlier remains explicitly annotated and all unzoomed figures are retained.']))


if __name__=='__main__':main()
