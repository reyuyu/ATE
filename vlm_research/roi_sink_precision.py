"""Fixed five-case sensitivity check: FP32 readout of the same FP16 backbone states.

Does not cast/modify model parameters; not a full FP32 backbone experiment.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from roi_sink_common import OUTPUT,WINDOWS,forward,RANDOM_SEEDS
from roi_sink_probe import config_and_data,inputs_and_history,gate
from prepare_fixed_k_data import digest
from prepare_physical_exit_data import sha256
from fixed_k_exit import atomic_json
from summarize_fixed_k import write_csv
from vlm_common import load_model
from knockout_probe import resource_preflight


def main():
    config,manifest,samples=config_and_data();binding=digest(config);sink=json.loads((OUTPUT/'sink_manifest.json').read_text());entries={e['question_id']:e for e in sink['samples']}
    cb=digest(dict(config_binding=binding,sink_sha256=sha256(OUTPUT/'sink_manifest.json')));gate('causal',cb,53)
    protocol=json.loads((OUTPUT/'precision_protocol.json').read_text())
    if protocol['question_ids']!=manifest['case_question_ids']:raise ValueError('Precision cases changed')
    resource=resource_preflight();atomic_json(OUTPUT/'logs/precision_resource.json',resource)
    if not resource['ready']:raise RuntimeError('GPU unavailable, no process stopped')
    model,processor,environment=load_model();atomic_json(OUTPUT/'logs/precision_environment.json',environment)
    torch.set_num_threads(4);weight=model.language_model.lm_head.weight.detach().cpu().float();version=model.language_model.lm_head.weight._version
    rows=[];started=time.perf_counter()
    for s in samples:
        q=s['question_id']
        if q not in protocol['question_ids']:continue
        _,candidates,qpos,apos,history,_,_=inputs_and_history(model,processor,s);inputs,records=candidates[0];entry=entries[q]
        plans=[('full','full',None,None,None)]+[(group,name,qpos if group=='question' else [apos],keys,WINDOWS['deep']) for group in ['question','answer'] for name,keys in [('roi',entry['roi_indices']),('sink_nonroi',entry['sink_nonroi_indices'])]+[(f'random_sink_s{seed}',entry['random_controls'][f'random_sink_s{seed}']['indices']) for seed in RANDOM_SEEDS]]
        ids=[recs[0]['token_id'] for x,recs in candidates];gold=s['candidates'].index(s['answer'])
        for group,name,queries,keys,active in plans:
            hidden=[]
            handle=model.language_model.lm_head.register_forward_pre_hook(lambda m,a:hidden.append(a[0][0,apos].detach().cpu().float()))
            try:out=forward(model,inputs,qpos,apos,queries,keys,active)
            finally:handle.remove()
            with torch.inference_mode():fp32=F.linear(hidden[0],weight)
            original=out['prediction_logits'].double();high=fp32.double();lp=original.log_softmax(-1);hp=high.log_softmax(-1)
            old=float(lp[ids[gold]]-lp[ids[1-gold]]);new=float(hp[ids[gold]]-hp[ids[1-gold]])
            rows.append(dict(question_id=q,query_group=group,window='none' if group=='full' else 'deep',condition=name,
                 fp16_margin=old,fp32_readout_margin=new,margin_difference=new-old,classification_changed=(old>0)!=(new>0),
                 max_abs_full_vocab_logits_difference=float((original-high).abs().max()),fp16_correct_logp=float(lp[ids[gold]]),fp32_readout_correct_logp=float(hp[ids[gold]])))
        print('precision',q,flush=True)
    assert model.language_model.lm_head.weight._version==version
    write_csv(OUTPUT/'precision_sensitivity.csv',rows)
    atomic_json(OUTPUT/'precision_sensitivity.json',dict(protocol=protocol,code_sha256=sha256(Path(__file__)),parent_binding=cb,n=len(rows),rows=rows,
         classification_changes=[r for r in rows if r['classification_changed']],max_margin_difference=max(abs(r['margin_difference']) for r in rows),
         weights_unmodified=True,wall_seconds=time.perf_counter()-started,
         limitation='Same FP16 backbone hidden states, FP32 CPU final linear readout using copied unchanged weights. Isolates final-projection rounding; does not establish robustness of the entire FP16 backbone or prove causality of tiny margins.'))

if __name__=='__main__':main()
