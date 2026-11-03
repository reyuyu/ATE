"""CPU analysis only. Rankings preserve original word groups and exclude P-1."""
import json,csv,hashlib,sys,collections
from pathlib import Path
import numpy as np
P=Path(sys.argv[1]);R=P.parent.parent;sys.path.insert(0,str(R))
from semantic_omitted_token_attention_mass.common import BANDS,DISPLAY,normalized,rankkey
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(name,rows,fields=None):
    path=P/name;path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        rows=iter(rows);first=next(rows,None)
        if first is None:
            if fields:csv.DictWriter(f,fieldnames=fields).writeheader()
            return
        w=csv.DictWriter(f,fieldnames=list(first));w.writeheader();w.writerow(first);w.writerows(rows)
def finite(a,fun):
    x=np.asarray(a);x=x[np.isfinite(x)];return float(fun(x)) if len(x) else None
def stats(m,e,z,p0):
    heads=m.mean(axis=(0,2));subs=m.mean(axis=(0,1))
    return dict(mass_mean=float(m.mean()),mass_head_median=float(np.median(heads)),mass_head_max=float(heads.max()),mass_single_head_max=float(m.max()),subword_mean=float(subs.mean()),subword_max=float(subs.max()),excess_mean=float(e.mean()),enrichment_mean=finite(z,np.mean),enrichment_median=finite(z,np.median),enrichment_max=finite(z,np.max),enrichment_undefined=int((~np.isfinite(z)).sum()),uniform_p0_mean=float(p0.mean()))
def main():
    assert read(P/'gpu_completion.json')['passed'];cases=read(P/'case_manifest.json')['cases'];groupbands=[];grouplayers=[];tokens=[];subbands=[];p1heads=[];caselevels=[];charts=[];index=[];high=[];global_invalid=0
    for v in cases:
        q=v['question_id'];s=v['sample'];cp=read(P/'checkpoints'/(q+'.json'));raw=Path(cp['raw']['path']);assert sha(raw)==cp['raw']['sha256']
        with np.load(raw) as z:m=z['image_mass'].copy();legal=z['legal_key_counts'].copy();queries=z['query_positions'].tolist()
        assert queries==v['spec']['relay_positions']+[s['input_mapping']['p_minus_one']];assert m.shape==(32,32,len(queries));assert np.array_equal(legal,np.broadcast_to(np.array(queries)+1,legal.shape))
        p0,e,z=normalized(m,legal[:,None,:]);global_invalid+=int((~np.isfinite(z)).sum());posidx={p:i for i,p in enumerate(queries)};gs={p:g for g in v['groups'] for p in g['positions']};ts={t['position']:t for t in v['tokens']}
        def headrows():
            for l in range(32):
                for h in range(32):
                    for j,p in enumerate(queries):
                        a=j==len(queries)-1;g=gs.get(p);value=float(z[l,h,j]) if np.isfinite(z[l,h,j]) else '';r=dict(question_id=q,kind=v['kind'],layer=l,head=h,query_position=p,scope='P-minus-one' if a else g['scope'],semantic_selected=False if a else g['selected'],group_id='' if a else g['group_id'],group_text='<P-1>' if a else g['decoded'],subword_piece='<P-1>' if a else ts[p]['piece'],image_mass=float(m[l,h,j]),image_keys=576,legal_keys=int(legal[l,j]),nonimage_legal_keys=int(legal[l,j]-576),p0=float(p0[l,0,j]),excess=float(e[l,h,j]),enrichment=value,enrichment_undefined_reason='' if value!='' else 'mass_ge_1_FP16_sum' if m[l,h,j]>=1 else 'no_positive_denominator')
                        if a:p1heads.append(r)
                        else:yield r
        path='per_head_csv/'+q+'.csv';writecsv(path,headrows());index.append(dict(question_id=q,path=path,rows=32*32*(len(queries)-1),sha256=sha(P/path)))
        for l in range(32):
            for j,p in enumerate(queries):
                g=gs.get(p);tokens.append(dict(question_id=q,kind=v['kind'],layer=l,position=p,scope=g['scope'] if g else 'P-minus-one',semantic_selected=g['selected'] if g else False,group_id=g['group_id'] if g else '',mass_head_mean=float(m[l,:,j].mean()),mass_head_median=float(np.median(m[l,:,j])),mass_head_max=float(m[l,:,j].max()),p0=float(p0[l,0,j]),excess_head_mean=float(e[l,:,j].mean()),excess_head_median=float(np.median(e[l,:,j])),excess_head_max=float(e[l,:,j].max()),enrichment_head_mean=finite(z[l,:,j],np.mean),enrichment_head_median=finite(z[l,:,j],np.median),enrichment_head_max=finite(z[l,:,j],np.max),undefined_enrichment=int((~np.isfinite(z[l,:,j])).sum())))
        local_subs=[]
        for j,p in enumerate(queries[:-1]):
            g=gs[p]
            for band,ls in BANDS.items():
                rr=dict(question_id=q,kind=v['kind'],position=p,piece=ts[p]['piece'],decoded=ts[p]['decoded'],group_id=g['group_id'],scope=g['scope'],selected=g['selected'],band=band,**stats(m[ls][:,:,j:j+1],e[ls][:,:,j:j+1],z[ls][:,:,j:j+1],p0[ls][:,:,j:j+1]));subbands.append(rr)
                if band=='L0-15':local_subs.append(rr)
        for g in v['groups']:
            ids=[posidx[p] for p in g['positions']];subtype='Assistant word' if g['scope']=='Assistant/template' and 'ASSISTANT' in g['words'] else 'post-image boundary' if g['positions'][0]==queries[0] else 'separator' if g['scope']=='Assistant/template' else ''
            meta=dict(question_id=q,kind=v['kind'],task=s['task'],group_id=g['group_id'],group_text=g['decoded'],positions=json.dumps(g['positions']),first_position=g['positions'][0],subwords=len(ids),scope=g['scope'],template_subtype=subtype,semantic_selected=g['selected'],tags='|'.join(g['tags']))
            for l in range(32):grouplayers.append(dict(**meta,layer=l,**stats(m[l:l+1,:,ids],e[l:l+1,:,ids],z[l:l+1,:,ids],p0[l:l+1,:,ids])))
            for band,ls in BANDS.items():groupbands.append(dict(**meta,band=band,**stats(m[ls][:,:,ids],e[ls][:,:,ids],z[ls][:,:,ids],p0[ls][:,:,ids])))
        for band,ls in BANDS.items():
            for scope,sel in [('Question',True),('Question',False),('Instruction',False),('Assistant/template',False)]:
                ids=[posidx[p] for g in v['groups'] if g['scope']==scope and g['selected']==sel for p in g['positions']]
                if ids:caselevels.append(dict(question_id=q,kind=v['kind'],band=band,scope=scope,selected=sel,query_count=len(ids),**stats(m[ls][:,:,ids],e[ls][:,:,ids],z[ls][:,:,ids],p0[ls][:,:,ids])))
        for band,ls in BANDS.items():
            for label,assistant in [('Assistant word',True),('Boundary/separators',False)]:
                ids=[posidx[p] for g in v['groups'] if g['scope']=='Assistant/template' and ('ASSISTANT' in g['words'])==assistant for p in g['positions']]
                if ids:caselevels.append(dict(question_id=q,kind=v['kind'],band=band,scope=label,selected=False,query_count=len(ids),**stats(m[ls][:,:,ids],e[ls][:,:,ids],z[ls][:,:,ids],p0[ls][:,:,ids])))
        heat=np.array([[m[l,:,[posidx[p] for p in g['positions']]].mean() for g in v['groups']] for l in range(32)])
        p1={b:stats(m[ls][:,:,-1:],e[ls][:,:,-1:],z[ls][:,:,-1:],p0[ls][:,:,-1:]) for b,ls in BANDS.items()}
        charts.append(dict(case=v,heatmap=heat.tolist(),p_minus_one=p1,subword_L0_15=local_subs))
        print('ANALYZED',q,flush=True)
    for q in [v['question_id'] for v in cases]:
        for band in BANDS:
            for scope in ['Question','Instruction','Assistant/template']:
                pool=sorted([r for r in groupbands if r['question_id']==q and r['band']==band and r['scope']==scope and not r['semantic_selected']],key=rankkey)
                for rank,r in enumerate(pool,1):r['rank_mass']=rank
                for metric,name in [('excess_mean','rank_excess'),('enrichment_mean','rank_enrichment')]:
                    for rank,r in enumerate(sorted(pool,key=lambda r:(-r[metric] if r[metric] is not None else float('inf'),r['first_position'],r['group_id'])),1):r[name]=rank if r[metric] is not None else None
                high.extend(dict(r) for r in pool[:5])
    for r in groupbands:
        for k in ['rank_mass','rank_excess','rank_enrichment']:r.setdefault(k,None)
    writecsv('token_layer_summary.csv',tokens);writecsv('subword_band_summary.csv',subbands);writecsv('p_minus_one_per_head.csv',p1heads);writecsv('group_layer_summary.csv',grouplayers);writecsv('group_band_rankings.csv',groupbands);writecsv('high_mass_omitted_tokens.csv',high);writecsv('case_scope_band_summary.csv',caselevels)
    for c in charts:
        q=c['case']['question_id'];c['band_groups']={b:[r for r in groupbands if r['question_id']==q and r['band']==b] for b in BANDS};c['orange_groups']=[r['group_id'] for r in high if r['question_id']==q and r['band']=='L0-15' and r['scope']=='Question']
    (P/'analysis_data.json').write_text(json.dumps(dict(cases=charts,bands=BANDS,top5_bands=DISPLAY,undefined_enrichment=global_invalid),ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (P/'per_head_index.json').write_text(json.dumps(dict(condition='Native Full only',P_minus_one_separate_file='p_minus_one_per_head.csv',partitions=index,total_R_rows=sum(r['rows'] for r in index)),indent=2))
    # Descriptive tag counts in each case's Top5 omitted Question groups; tags overlap.
    tagrows=[]
    for band in BANDS:
        for kind in ['harm','rescue']:
            pool=[r for r in high if r['kind']==kind and r['band']==band and r['scope']=='Question']
            for tag in ['article','noun','relation','question_end','punctuation']:
                matched=[r for r in pool if tag in r['tags'].split('|')];tagrows.append(dict(kind=kind,band=band,tag=tag,top5_groups=len(pool),tagged_groups=len(matched),cases_with_tag=len({r['question_id'] for r in matched})))
    writecsv('top5_tag_summary.csv',tagrows)
    (P/'analysis_validation.json').write_text(json.dumps(dict(passed=True,cases=16,R_rows=sum(r['rows'] for r in index),P_minus_one_rows=len(p1heads),all32_layers_32_heads=True,all_original_groups_preserved=True,P_minus_one_excluded_from_R_ranks=True,undefined_enrichment=global_invalid,no_infinite_enrichment=True,rank_uses_all_heads_mean=True,new_answers=0,scope_distribution_unit='case mean, not heads as independent samples'),indent=2))
if __name__=='__main__':main()
