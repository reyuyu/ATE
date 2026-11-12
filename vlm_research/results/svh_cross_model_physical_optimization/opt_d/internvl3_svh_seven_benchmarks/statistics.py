"""Image-SHA cluster paired bootstrap in each benchmark's native score units."""
from .common import *
import numpy as np
from collections import defaultdict
SEED=20261110;REPS=20000
def linear(rows,scale=100):
    groups=defaultdict(lambda:np.zeros(3))
    for r in rows:groups[r["image_sha256"]]+=np.array([1,r["native_score"],r["physical_score"]])
    a=np.stack(list(groups.values()));rng=np.random.default_rng(SEED);dist=[]
    for start in range(0,REPS,128):
        counts=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=min(128,REPS-start))
        dist.extend((counts@(a[:,2]-a[:,1]))/(counts@a[:,0])*scale)
    values=np.array(dist);return float(np.mean(a[:,1])/np.mean(a[:,0])*scale),float(np.mean(a[:,2])/np.mean(a[:,0])*scale),np.quantile(values,[.025,.975]).tolist(),len(a)
def mme(rows,tasks):
    groups=defaultdict(list)
    for r in rows:
        if r["category"] in tasks:groups[r["group"]].append(r)
    images=defaultdict(lambda:np.zeros((len(tasks),3)))
    for rs in groups.values():
        assert len(rs)==2
        i=tasks.index(rs[0]["category"])
        native=.5*sum(r["native_score"] for r in rs)+np.prod([r["native_score"] for r in rs])
        phys=.5*sum(r["physical_score"] for r in rs)+np.prod([r["physical_score"] for r in rs])
        images[rs[0]["image_sha256"]][i]+=np.array([1,native,phys])
    a=np.stack(list(images.values()));den=a[:,:,0];n=a[:,:,1];p=a[:,:,2]
    native=float(np.sum(n.sum(0)/den.sum(0))*100);physical=float(np.sum(p.sum(0)/den.sum(0))*100)
    rng=np.random.default_rng(SEED);dist=[]
    for start in range(0,REPS,128):
        counts=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=min(128,REPS-start));d=counts@den;good=(d>0).all(1)
        dist.extend(np.sum((counts@(p-n))[good]/d[good],axis=1)*100)
    assert len(dist)>=REPS*.999,"Too many bootstrap replicates lack categories"
    return native,physical,np.quantile(dist,[.025,.975]).tolist(),len(a)
def circular(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["group"]].append(r)
    units=[]
    for gid,rs in groups.items():
        units.append({"image_sha256":rs[0]["image_sha256"],"native_score":float(all(r["native_score"]==1 for r in rs)),"physical_score":float(all(r["physical_score"]==1 for r in rs))})
    return linear(units),len(groups)
def pope_f1(rows):
    groups=defaultdict(lambda:np.zeros(6))
    for r in rows:
        gt=r["gold"]=="yes";v=[]
        for prefix in ["native","physical"]:
            yes=r[prefix+"_parsed"]=="yes";v.extend([int(gt and yes),int(not gt and yes),int(gt and not yes)])
        groups[r["image_sha256"]]+=v
    a=np.stack(list(groups.values()))
    def f1(x):return np.divide(2*x[...,0],2*x[...,0]+x[...,1]+x[...,2],out=np.zeros_like(x[...,0],dtype=float),where=(2*x[...,0]+x[...,1]+x[...,2])!=0)*100
    t=a.sum(0);n=float(f1(t[:3]));p=float(f1(t[3:]));rng=np.random.default_rng(SEED);dist=[]
    for start in range(0,REPS,128):
        counts=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=min(128,REPS-start));s=counts@a;dist.extend(f1(s[:,3:])-f1(s[:,:3]))
    return n,p,np.quantile(dist,[.025,.975]).tolist(),len(a)

def pope_macro_f1(rows):
    strategies=["random","popular","adversarial"]
    groups=defaultdict(lambda:np.zeros(18))
    for r in rows:
        slot=strategies.index(r["category"])*6;gt=r["gold"]=="yes";v=[]
        for prefix in ["native","physical"]:
            yes=r[prefix+"_parsed"]=="yes";v.extend([int(gt and yes),int(not gt and yes),int(gt and not yes)])
        groups[r["image_sha256"]][slot:slot+6]+=v
    a=np.stack(list(groups.values()))
    def macro(x):
        values=[]
        for i in range(3):
            sub=x[...,i*6:i*6+6]
            f=lambda z:np.divide(2*z[...,0],2*z[...,0]+z[...,1]+z[...,2],out=np.zeros_like(z[...,0],dtype=float),where=(2*z[...,0]+z[...,1]+z[...,2])!=0)*100
            values.append(np.stack([f(sub[...,:3]),f(sub[...,3:])],axis=-1))
        return np.mean(values,axis=0)
    totals=macro(a.sum(0));rng=np.random.default_rng(SEED);dist=[]
    for start in range(0,REPS,128):
        count=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=min(128,REPS-start));m=macro(count@a);dist.extend(m[:,1]-m[:,0])
    return float(totals[0]),float(totals[1]),np.quantile(dist,[.025,.975]).tolist(),len(a)
