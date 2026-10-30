"""Frozen image-paired bootstrap, ordinary and Bonferroni simultaneous endpoints."""
import numpy as np
from .protocol import SEED


def cluster_deltas(deltas,identities):
    values=np.asarray(deltas,dtype=np.float64)
    if values.ndim==1:values=values[:,None]
    if len(values)!=len(identities) or not np.isfinite(values).all():raise ValueError('Invalid paired data')
    unique=sorted(set(identities));index={k:i for i,k in enumerate(unique)}
    sums=np.zeros((len(unique),values.shape[1]));counts=np.zeros(len(unique))
    for key,row in zip(identities,values):sums[index[key]]+=row;counts[index[key]]+=1
    return sums,counts


def bootstrap(deltas,identities,repetitions=200000,seed=SEED,family=96):
    sums,counts=cluster_deltas(deltas,identities)
    if not len(sums) or repetitions<2:raise ValueError('Need nonempty images and repeated bootstrap')
    rng=np.random.default_rng(seed);draws=np.empty((repetitions,sums.shape[1]),dtype=np.float64)
    for first in range(0,repetitions,1000):
        stop=min(repetitions,first+1000);sample=rng.integers(0,len(sums),(stop-first,len(sums)))
        weights=np.stack([np.bincount(row,minlength=len(sums)) for row in sample])
        draws[first:stop]=(weights@sums)/(weights@counts)[:,None]
    return dict(mean=sums.sum(0)/counts.sum(),ordinary=np.quantile(draws,[.025,.975],axis=0),
        corrected=np.quantile(draws,[.025/family,1-.025/family],axis=0),
        clusters=len(sums),repetitions=repetitions,family=family,expected_extreme_tail_draws=repetitions*.025/family)


def interval_summary(mean,ordinary,corrected,margin):
    return dict(delta=float(mean),ci95_low=float(ordinary[0]),ci95_high=float(ordinary[1]),
        family95_low=float(corrected[0]),family95_high=float(corrected[1]),
        ni_margin=margin,ordinary_noninferior=bool(ordinary[0]>-margin),family_noninferior=bool(corrected[0]>-margin))
