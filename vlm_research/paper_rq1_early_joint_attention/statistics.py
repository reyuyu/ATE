"""Question-weighted paired bootstrap with whole-image cluster resampling."""
import math
import numpy as np
from paper_rq1_decode_layerwise_quality.statistics import cluster_deltas
from .protocol import SEED


def bootstrap(values,identities,families,repetitions=200000,seed=SEED):
    sums,counts=cluster_deltas(values,identities)
    if not len(sums) or repetitions<100 or len(families)!=sums.shape[1] or any(f<1 for f in families):
        raise ValueError('Invalid paired bootstrap design')
    rng=np.random.default_rng(seed);draws=np.empty((repetitions,sums.shape[1]),dtype=np.float64)
    for first in range(0,repetitions,500):
        stop=min(repetitions,first+500)
        sampled=rng.integers(0,len(sums),(stop-first,len(sums)))
        weights=np.stack([np.bincount(row,minlength=len(sums)) for row in sampled])
        draws[first:stop]=(weights@sums)/(weights@counts)[:,None]
    records=[]
    for k,family in enumerate(families):
        ordinary=np.quantile(draws[:,k],[.025,.975])
        adjusted=np.quantile(draws[:,k],[.025/family,1-.025/family])
        records.append(dict(mean=float(sums[:,k].sum()/counts.sum()),ci95_low=float(ordinary[0]),ci95_high=float(ordinary[1]),
            family95_low=float(adjusted[0]),family95_high=float(adjusted[1]),family=family,
            clusters=len(sums),repetitions=repetitions,expected_tail_draws=repetitions*.025/family,
            degenerate_bootstrap=bool(np.all(draws[:,k]==draws[0,k]))))
    return records


def mcnemar_exact(harm,gain):
    n=harm+gain;k=min(harm,gain)
    if n==0:return 1.
    return min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n)
