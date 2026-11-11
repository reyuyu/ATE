"""Question-weighted estimates with paired whole-image cluster bootstrap."""
from collections import defaultdict
import numpy as np


def paired(rows, a='native_score', b='svh_score', seed=20261111, reps=20000):
    if not rows:
        return dict(n=0, images=0, native_acc=None, svh_acc=None, delta_pp=None,
                    ci_low_pp=None, ci_high_pp=None, harms=0, rescues=0)
    groups=defaultdict(list)
    for r in rows:
        groups[r['image_sha256']].append(float(r[b])-float(r[a]))
    values=[groups[k] for k in sorted(groups)]
    sums=np.array([sum(g) for g in values],dtype=np.float64)
    counts=np.array([len(g) for g in values],dtype=np.float64)
    rng=np.random.default_rng(seed)
    draws=[]
    for start in range(0,reps,128):
        ix=rng.integers(0,len(values),size=(min(128,reps-start),len(values)))
        draws.extend((sums[ix].sum(axis=1)/counts[ix].sum(axis=1)*100).tolist())
    low,high=np.quantile(draws,[.025,.975])
    return dict(n=len(rows), images=len(values), native_acc=100*np.mean([r[a] for r in rows]),
        svh_acc=100*np.mean([r[b] for r in rows]), delta_pp=100*sum(sums)/len(rows),
        ci_low_pp=float(low),ci_high_pp=float(high),
        harms=sum(r[a]>r[b] for r in rows),rescues=sum(r[a]<r[b] for r in rows),
        bootstrap_replicates=reps, image_clustered=True, small_n=len(rows)<30)


def summaries(values):
    values=np.asarray(values,dtype=np.float64)
    if not len(values):
        return dict(mean=None,p50=None,p95=None)
    return dict(mean=float(np.mean(values)),p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)))
