"""Explicitly seal the user-authorized Qwen2048 amendment before each GPU gate."""
import argparse
import shutil
import gzip
from .common import *

def main(phase):
    if phase=='initial':
        assert not list((OUT/'environment').glob('*.json')) if (OUT/'environment').exists() else True
        p=read(OUT/'protocol.json')
        if not (OUT/'protocol_pre_2048_amendment.json').exists():
            atomic(OUT/'protocol_pre_2048_amendment.json',p)
        p['qwen']['max_image_tokens']=2048
        p['qwen']['processor_override']={'size.longest_edge':2048*32*32}
        p['qwen']['pre_gpu_gate']='Actual Processor twice on every image; actual merged tokens <=2048; model-specific input/Smoke freeze in qwen_gpu_protocol.json'
        p['qwen']['user_authorized_protocol_amendment']=True
        p['qwen']['numerical_status']='Known Physical-Original / Dense-F difference; strict equivalence not claimed; evaluate exploratory quality only after semantic and resource gates'
        p['no_timing']=True
        p['smoke_memory_gate']='Peak reserved <= total GPU memory -512MiB; no budget/backend fallback'
        p['at']=now();atomic(OUT/'protocol.json',p)
        atomic(OUT/'source_manifest.json',dict(at=now(),files=source_hashes()))
        print('INITIAL SOURCE AND PROTOCOL SEALED',flush=True)
    else:
        assert not (OUT/'qwen_gpu_protocol.json').exists()
        v=read(OUT/'qwen_processor_validation.json');assert v['status']=='PASSED' and v['actual_cap_exceedances']==0
        rows=read(OUT/'sample_manifest.json');b=read(OUT/'qwen_input_bindings.json.gz');specs=[]
        for rank in (0,1):
            for bench in ('MME','SEED'):
                pool=[s for s in rows if s['shard']==rank and s['benchmark']==bench]
                typical=sorted(pool,key=lambda s:(len(b[s['key']]['input_ids']),s['key']))[len(pool)//2]
                extremes=[max(pool,key=lambda s:(len(b[s['key']]['input_ids']),s['key'])),
                    max(pool,key=lambda s:(b[s['key']]['image_tokens'],s['key'])),
                    max(pool,key=lambda s:(s['width']*s['height'],s['key'])),typical]
                specs += [dict(key=s['key'],rank=rank) for s in extremes]
        specs=[dict(key=k,rank=r) for k,r in sorted({(s['key'],s['rank']) for s in specs})]
        # Worst actual sequence first on each card, before any formal prediction.
        specs.sort(key=lambda x:(x['rank'],-len(b[x['key']]['input_ids']),x['key']))
        atomic(OUT/'qwen_gpu_protocol.json',dict(status='FROZEN_BEFORE_QWEN_GPU',at=now(),smoke=specs,
            source_manifest_sha256=sha(OUT/'source_manifest.json'),
            protocol_sha256=sha(OUT/'protocol.json'),bindings_sha256=sha(OUT/'qwen_input_bindings.json.gz'),
            processor_validation_sha256=sha(OUT/'qwen_processor_validation.json'),max_image_tokens=2048,
            actual_processor_image_count=len(read(OUT/'qwen_image_processor_audit.json')),
            memory_safety_margin_bytes=512*2**20,no_dynamic_budget_changes=True))
        print('QWEN GPU PROTOCOL SEALED',specs,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initial','qwen']);main(p.parse_args().phase)
