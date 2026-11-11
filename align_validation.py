"""Re-execute only old Validation rows needed to keep mixed image clusters on one GPU."""
import argparse
import gc
import subprocess
import time
from collections import defaultdict
from full_common import *


def freeze():
    rows=json.loads((OUT/'sample_manifest.json').read_text())
    old=dict(read_store(OUT/'reused_validation.sqlite'))
    groups=defaultdict(list)
    for r in rows:
        groups[r['image_sha256']].append(r)
    selected=[r for group in groups.values() if any(r['reuse_validation'] for r in group) and not all(r['reuse_validation'] for r in group)
        for r in group if r['reuse_validation'] and old[r['key']+'|Native']['gpu_uuid']!=UUIDS[r['shard']]]
    plan=dict(reason='Canonical same-image hardware pairing; selected solely by SHA and prior GPU UUID, never scores',
        rows=selected, extra_generations=len(selected)*2,source_sha256=sha(__file__),
        precedence='Always use new realigned record, regardless of score; retain old record read-only; no double count',
        protocol_sha256=sha(OUT/'p2_full_protocol.json'))
    write(OUT/'hardware_alignment_plan.json',plan)
    print('FROZEN_HARDWARE_ALIGNMENT',len(selected),flush=True)


def run_alignment(shard):
    p=check()
    plan=json.loads((OUT/'hardware_alignment_plan.json').read_text())
    assert plan['source_sha256']==sha(__file__)
    rows=[r for r in plan['rows'] if r['shard']==shard]
    if not rows:
        return
    assert (OUT/f'complete{shard}.json').exists()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(UUIDS[shard] in line for line in apps.splitlines()),apps
    assert os.environ['CUDA_VISIBLE_DEVICES']==UUIDS[shard]
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from vlm_research.qwen3vl_overnight.common import prepare
    from vlm_research.qwen3vl_overnight.engine import run
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    old=dict(read_store(OUT/'reused_validation.sqlite'))
    store=Store(OUT/f'aligned{shard}.sqlite')
    model=None
    try:
        processor=AutoProcessor.from_pretrained(p['model'],local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(p['model'],local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        for row in rows:
            disk()
            new,text,ids=render(processor,row,p['suffix_exact_from_prior'])
            inputs,layout=prepare(processor,new)
            assert layout['prompt']==text and layout['input_ids']==ids
            assert digest(layout)==old[row['key']+'|Native']['input_binding_sha256']
            for cond in ['Native','SVH']:
                t=time.perf_counter()
                result,_,structure=run(model,processor,inputs,layout,cond,cap=row['max_new_tokens'])
                rec=dict(key=row['key'],benchmark=row['benchmark'],question_id=row['question_id'],gold=row['gold'],
                    image_sha256=row['image_sha256'],circular_group=row.get('circular_group'),prompt='P2',condition=cond,
                    gpu_uuid=UUIDS[shard],input_binding_sha256=digest(layout),protocol_sha256=sha(OUT/'p2_full_protocol.json'),
                    prompt_sha256=new['prompt_sha256'],input_length=layout['input_length'],image_tokens=layout['image_tokens'],
                    state='COMPLETE',source='REALIGNED_VALIDATION_NEW_GENERATION',elapsed_s=time.perf_counter()-t,structural=structure,
                    old_prediction_sha256=digest(old[row['key']+'|'+cond]))
                fields=['text','token_ids','generated_tokens','eos','truncated','empty','vision_ms','prefill_ms','ttft_ms',
                    'decode_ms','decode_ms_per_token','e2e_ms','peak_allocated','peak_reserved']
                rec.update({k:result[k] for k in fields})
                rec.update(scored(row,result['text'],result['truncated']))
                store.put(row['key']+'|'+cond,rec)
                print('REALIGNED',row['key'],cond,flush=True)
        write(OUT/f'alignment_complete{shard}.json',dict(status='COMPLETE',records=2*len(rows)))
    finally:
        store.conn.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write(OUT/f'alignment_release{shard}.json',dict(pid=os.getpid(),gpu_uuid=UUIDS[shard],allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--shard',type=int,choices=[0,1])
    args=parser.parse_args()
    freeze() if args.freeze else run_alignment(args.shard)
