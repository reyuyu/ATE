"""Descriptive budget/overlap tables from frozen input sets, without predictions."""
import csv
from .common import *

def save(path,rows):
    assert not path.exists()
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    index=read(OUT/'selection_index.json');samples=read(OUT/'sample_manifest.json')['samples'];confirm=set(read(OUT/'split_integrity.json')['confirmatory_keys'])
    groups={'all3000_descriptive':[key(s) for s in samples],'image_disjoint_confirmatory':[key(s) for s in samples if key(s) in confirm]}
    groups.update({f'task_{t}':[key(s) for s in samples if s['task']==t] for t in TASKS})
    specs={sk:read(OUT/info['path']) for sk,info in index.items()}
    for sk,info in index.items():assert sha256(OUT/info['path'])==info['sha256']
    rows=[]
    for pop,keys in groups.items():
        for c in CONDITIONS:
            kk=np.array([len(specs[sk]['writers'][c]) for sk in keys]);qq=np.array([len(specs[sk]['question_positions']) for sk in keys]);rr=np.array([len(specs[sk]['relay_positions']) for sk in keys])
            assert (kk<=rr).all()
            rows.append(dict(population=pop,condition=c,questions=len(keys),actual_k_sum=int(kk.sum()),question_token_sum=int(qq.sum()),R_token_sum=int(rr.sum()),mean_k=float(kk.mean()),
                mean_question_tokens=float(qq.mean()),mean_R_tokens=float(rr.mean()),actual_R_fraction_question_mean=float((kk/rr).mean()),actual_R_fraction_pooled=float(kk.sum()/rr.sum()),
                actual_Question_fraction_question_mean=float((kk/qq).mean()),actual_Question_fraction_pooled=float(kk.sum()/qq.sum()),zero_budget_questions=int((kk==0).sum()),
                total_allowed_R_image_edges_per_head=int(kk.sum()*32*576),total_possible_R_image_edges_per_head=int(rr.sum()*32*576),
                figure_C_x_axis='100 times mean_i(k_i/R_i); pooled connection fraction also provided separately',input_only=True))
    save(OUT/'actual_token_budget_summary.csv',rows)
    atomic_json(OUT/'input_budget_audit.json',dict(passed=True,input_only=True,questions=3000,rows=len(rows),source_sha256=sha256(__file__),
        macro_and_pooled_denominators_explicit=True,no_model_outputs_read=True,all_token_manifests_unchanged=True))

if __name__=='__main__':main()
