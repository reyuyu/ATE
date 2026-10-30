"""Read-only result review and native-dtype cost audit, not predictor selection.

This supplement never changes the frozen worker sources or thresholds. Native
cost profiling needs one extra observed Full only after all four workers exit.
"""
import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import transformers.models.llama.modeling_llama as llama
from benchmarks.dataset_registry import freeze_json,sha256,digest
from paper_rq1.analyze import write_csv
from paper_rq1_single_layer_gqa6.analyze import descriptive_correlations
from paper_rq1_single_layer_gqa6 import run as original_run
from paper_rq1.run import preflight
from vlm_common import load_model
from bos_cheap.protocol import OUT,OLD,MODEL_PATH,TASKS
from bos_cheap.run import verify,gate


def baseline():
    with (OLD/'bos_mass_per_question.csv').open() as f:bos={(r['task'],r['question_id'],int(r['layer'])):float(r['bos_mass']) for r in csv.DictReader(f)}
    with (OLD/'first_token_per_question.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=96000 or len(bos)!=96000:raise ValueError('Historical cohort incomplete')
    task=np.array([r['task'] for r in rows]);layer=np.array([int(r['layer']) for r in rows]);x=np.array([bos[(r['task'],r['question_id'],int(r['layer']))] for r in rows])
    delta=np.array([float(r['delta_logp']) for r in rows]);targets=dict(signed_delta_logp=delta,abs_delta_logp=np.abs(delta),logp_loss=np.maximum(-delta,0),top1_flip=np.array([r['top1_flip']=='True' for r in rows],float),js=np.array([float(r['js']) for r in rows]))
    output=[]
    for t in [*TASKS,'ALL']:
        for group,allowed in [('L0-L31',range(32)),('L2-L31',range(2,32))]:
            mask=np.isin(layer,list(allowed))
            if t!='ALL':mask&=task==t
            xx=x[mask];ll=layer[mask];tt=task[mask]
            for name,values in targets.items():
                yy=values[mask]
                for scope in ['across_layer_means','within_task_layer_centered']:
                    if scope=='across_layer_means':a=np.array([xx[ll==l].mean() for l in allowed]);b=np.array([yy[ll==l].mean() for l in allowed])
                    else:
                        a=xx.copy();b=yy.copy()
                        for task_name in TASKS:
                            for l in allowed:
                                indices=(tt==task_name)&(ll==l)
                                if indices.any():a[indices]-=a[indices].mean();b[indices]-=b[indices].mean()
                    pearson,spearman=descriptive_correlations(a,b)
                    output.append(dict(task=t,layer_group=group,scope=scope,x='true_native_bos_mass',y=name,n_values=len(a),pearson=pearson,spearman=spearman,interpretation='Descriptive, overlapping question/image/layer rows are not independent. No causal or significance inference.'))
    write_csv(OUT/'baseline_bos_reassessment.csv',output)
    freeze_json(OUT/'baseline_bos_reassessment_audit.json',dict(script_sha256=sha256(Path(__file__)),input_sha256={str(p):sha256(p) for p in [OLD/'bos_mass_per_question.csv',OLD/'first_token_per_question.csv']},matched_rows=96000,predictors_or_thresholds_modified=False,new_outputs_used=False))


def timed(fn,device):
    for _ in range(20):fn()
    if device=='cuda':
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
        for _ in range(200):fn()
        end.record();torch.cuda.synchronize();return begin.elapsed_time(end)*5
    start=time.perf_counter()
    for _ in range(200):fn()
    return (time.perf_counter()-start)*5000


def native_cost():
    gate('new');verify();pre=preflight(0)
    model,processor,env=load_model(MODEL_PATH)
    sample=json.loads((OUT/'sample_manifest.json').read_text())['old_samples'][0]
    inputs=original_run.prepared(model,processor,sample);before=original_run.tensor_hashes(inputs)
    query=sample['input_mapping']['p_minus_one'];bos=sample['input_mapping']['bos_positions'][0]
    original=llama.eager_attention_forward;captured={}
    def tap(module,q,k,v,mask,scaling,**kwargs):
        if module.layer_idx==0:captured.update(q=q[:,:,query:query+1].detach().clone(),k=k.detach().clone(),v=v.detach().clone(),scaling=scaling)
        return original(module,q,k,v,mask,scaling,**kwargs)
    try:
        llama.eager_attention_forward=tap
        with torch.inference_mode():out=model(**inputs,use_cache=False,return_dict=True)
    finally:llama.eager_attention_forward=original
    reference=json.loads((OUT/f"old/shard{sample['rank']}/checkpoints/{sample['task']}__{sample['question_id']}.json").read_text())
    with np.load(reference['raw_path']) as raw:np.testing.assert_array_equal(out.logits[0,query].float().cpu().numpy(),raw['full_logits'])
    if before!=original_run.tensor_hashes(inputs):raise ValueError('Cost audit changed model input')
    records=[]
    for device in ['cuda','cpu']:
        q,k,v=[captured[n].to(device) for n in ['q','k','v']];scale=captured['scaling']
        native_score=q@k.transpose(-1,-2)*scale
        native_prob=torch.softmax(native_score,dim=-1,dtype=torch.float32).to(q.dtype)
        module=SimpleNamespace(num_key_value_groups=1,training=False)
        functions=dict(native_fp16_qk=lambda:q@k.transpose(-1,-2)*scale,
            native_fp32_softmax_to_fp16=lambda:torch.softmax(native_score,dim=-1,dtype=torch.float32).to(q.dtype),
            native_fp16_av=lambda:native_prob@v,
            native_full_query_attention=lambda:original(module,q,k,v,None,scale,dropout=0),
            bos_qk_fp32_feature=lambda:(q[0,:,0].float()*k[0,:,bos].float()).sum(-1)*scale,
            read_actual_bos_kv=lambda:(k[:,:,bos:bos+1].clone(),v[:,:,bos:bos+1].clone()))
        for name,fn in functions.items():records.append(dict(device=device,component=name,microseconds=timed(fn,device),iterations=200,prefix_length=k.shape[-2],query_rows=1,source='Actual native L0 P-1 Q and full native K/V; pinned FP16/eager path, no synthesized timing tensors.',scope='PyTorch operations+dispatch; not custom-kernel or E2E speedup.'))
    write_csv(OUT/'native_dtype_cost_audit.csv',records)
    freeze_json(OUT/'native_dtype_cost_audit.json',dict(passed=True,script_sha256=sha256(Path(__file__)),sample_sha256=digest(sample),preflight=pre,environment=env,extra_observed_full_forwards=1,new_ablations=0,full_logits_exact=True,source_restored=llama.eager_attention_forward is original,inputs_unchanged=True,explanation='Original Smoke microbench full_query_attention uses FP32 reference arithmetic including casts. This separate audit measures the actual native FP16 QK/AV and FP32-Softmax-to-FP16 path; do not confuse either with E2E acceleration.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['baseline','cost']);a=p.parse_args()
    baseline() if a.stage=='baseline' else native_cost()


if __name__=='__main__':main()
