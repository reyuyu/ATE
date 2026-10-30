"""Paired image-cluster intervals and within-image conditional role comparisons."""
import numpy as np
from .protocol import SEED,FAMILY

BOOT=10000
CONTROLS=['full_token_logp','full_token_entropy','generation_step','relative_generation_position','token_length']
_WEIGHTS={}


def weights(ids):
    key=tuple(ids)
    if key not in _WEIGHTS:
        n=len(ids)
        _WEIGHTS[key]=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=BOOT)
    return _WEIGHTS[key]


def interval(samples,family):
    lo,hi=np.quantile(samples,[.025,.975])
    fl,fh=np.quantile(samples,[.05/(2*family),1-.05/(2*family)])
    return dict(ci95_low=float(lo),ci95_high=float(hi),familywise95_low=float(fl),familywise95_high=float(fh))


def ci(rows,values,family=1):
    values=np.asarray(values,dtype=float)
    if len(values)!=len(rows) or not len(rows) or not np.isfinite(values).all():raise ValueError('Need aligned finite nonempty observations')
    ids,inverse=np.unique([r['image_sha256'] for r in rows],return_inverse=True)
    sums=np.bincount(inverse,weights=values);counts=np.bincount(inverse)
    w=weights(ids);samples=(w@sums)/(w@counts)
    return dict(mean=float(values.mean()),n_observations=len(rows),n_images=len(ids),family=family,
        bootstrap_replicates=BOOT,**interval(samples,family))


def fixed_effects(rows):
    roles={}
    for r in rows:roles.setdefault(r['image_sha256'],set()).add(r['word_type'])
    rows=[r for r in rows if {'Function','Visual'}<=roles[r['image_sha256']]]
    if not rows:return dict(status='no_same_image_pairs'),[]
    ids,group=np.unique([r['image_sha256'] for r in rows],return_inverse=True)
    columns=['Visual_vs_Function','Relation_vs_Function']+CONTROLS
    x=np.array([[float(r['word_type']=='Visual'),float(r['word_type']=='Relation')]+[r[c] for c in CONTROLS] for r in rows])
    y=np.array([r['loss'] for r in rows],float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite regression data')
    for i in range(len(ids)):
        mask=group==i;x[mask]-=x[mask].mean(0);y[mask]-=y[mask].mean()
    scale=np.sqrt((x*x).mean(0));keep=scale>1e-12
    dropped=[name for name,k in zip(columns,keep) if not k]
    if not keep[0]:return dict(status='role_not_identifiable',dropped=dropped),[]
    columns=[name for name,k in zip(columns,keep) if k];scale=scale[keep];x=x[:,keep]/scale
    rank=np.linalg.matrix_rank(x,tol=np.linalg.svd(x,compute_uv=False)[0]*1e-10)
    meta=dict(n_images=len(ids),n_positions=len(rows),controls=CONTROLS,dropped_constant_columns=dropped,
        columns=columns,design_rank=int(rank),design_columns=x.shape[1],condition_number=float(np.linalg.cond(x)),
        image_fixed_effects=True,family=FAMILY,bootstrap_replicates=BOOT,
        interpretation='Within-image linear adjustment; not causal role attribution or adjustment for all lexical/context confounding.')
    if rank!=x.shape[1]:return dict(meta,status='rank_deficient_no_claim'),[]
    p=x.shape[1];gram=np.zeros((len(ids),p,p));rhs=np.zeros((len(ids),p))
    for i in range(len(ids)):
        mask=group==i;gram[i]=x[mask].T@x[mask];rhs[i]=x[mask].T@y[mask]
    beta=np.linalg.solve(gram.sum(0),rhs.sum(0))/scale
    w=weights(ids);matrices=np.einsum('bi,ijk->bjk',w,gram);vectors=w@rhs
    eig=np.linalg.eigvalsh(matrices);valid=eig[:,0]>eig[:,-1]*1e-10
    if not valid.any():return dict(meta,status='no_identifiable_bootstrap'),[]
    estimates=(np.linalg.pinv(matrices[valid],rcond=1e-10)@vectors[valid,:,None]).squeeze(-1)/scale
    output=[dict(variable=name,effect=float(beta[i]),**interval(estimates[:,i],FAMILY),
        family=FAMILY,n_images=len(ids),n_positions=len(rows),bootstrap_valid=int(valid.sum())) for i,name in enumerate(columns)]
    return dict(meta,status='estimated',bootstrap_valid=int(valid.sum())),output
