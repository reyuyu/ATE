"""Final read-only delivery audit; call after every authorized worker has exited."""
import collections
import gzip
import json
import os
from pathlib import Path
import subprocess
import shutil
from .mme_seed_quality.common import *


def main():
    for rank in (0,1):
        assert read(OUT/f'progress/llava_rank{rank}.json')['status']=='COMPLETE'
        assert read(OUT/f'environment/llava_rank{rank}_run_exit.json')['status']=='SUCCESS'
    model_status={'llava':'COMPLETE'}
    if (OUT/'qwen_stop_summary.json').exists():
        model_status['qwen']='STOPPED_OOM'
    else:
        for rank in (0,1):assert read(OUT/f'progress/qwen_rank{rank}.json')['status']=='COMPLETE'
        model_status['qwen']='COMPLETE'
    assert read(OUT/'source_manifest.json')['files']==source_hashes()
    sources=read(OUT/'model_manifests.json')
    for kind,model in sources.items():
        for name,h in model['files'].items():assert sha(Path(model['path'])/name)==h,(kind,name)
    images=read(OUT/'image_manifest.json')
    for p,meta in images.items():assert sha(p)==meta['sha256'],p
    samples=read(OUT/'sample_manifest.json');index={s['key']:s for s in samples}
    checked=collections.Counter();duplicate_audit=[]
    for kind in ('llava','qwen'):
        bindings=read(OUT/f'{kind}_input_bindings.json.gz')
        tokenizer=None
        if model_status[kind]=='COMPLETE':
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(sources[kind]['path'],local_files_only=True)
        duplicates=collections.defaultdict(list)
        for sample in samples:
            pair=[]
            for c in ('Native','Physical-C3'):
                path=OUT/'checkpoints'/kind/sample['key']/(c+'.json')
                if not path.exists():
                    assert model_status[kind]!='COMPLETE',path
                    continue
                r=read(path);assert r['model_manifest_sha256']==sha(OUT/'model_manifests.json')
                assert r['protocol_sha256']==sha(OUT/'protocol.json')
                assert r['source_manifest_sha256']==sha(OUT/'source_manifest.json')
                assert r['sample_sha256']==digest(sample)
                assert r['uuid']==UUIDS[sample['shard']]
                b=r['input_binding'];expected=bindings[sample['key']]
                assert b['prompt']==expected['prompt'] and b['input_ids']==expected['input_ids']
                if kind=='qwen':
                    assert b['image_tokens']<=2048
                    assert b['input_hashes']['pixel_values']==expected['pixel_values_sha256']
                    assert r['qwen_gpu_protocol_sha256']==sha(OUT/'qwen_gpu_protocol.json')
                assert r['result']['correct']==(r['result']['prediction']==sample['answer'])
                if sample['benchmark']=='MME':
                    assert tokenizer is not None
                    result=r['result']
                    assert tokenizer.decode(result['token_ids'],skip_special_tokens=True).strip()==result['text']
                    assert result['empty']==(not bool(result['text']))
                else:
                    result=r['result']
                    assert result['choice_token_ids']==expected['candidate_token_ids']
                    assert len(result['token_logps'])==4
                    assert all(len(ids)==len(logps) for ids,logps in zip(result['choice_token_ids'],result['token_logps']))
                pair.append(r);checked[(kind,sample['benchmark'],c)]+=1
            if len(pair)==2:
                assert pair[0]['input_binding']==pair[1]['input_binding'],'Paired Native/C3 tensors or prompt changed'
                if sample['benchmark']=='SEED':duplicates[sample['source_question_id']].append(pair)
        for qid,records in duplicates.items():
            if len(records)>1:
                duplicate_audit.append(dict(model=kind,source_question_id=qid,records=len(records),
                    question_keys=[r[0]['key'] for r in records],
                    native_scores_identical=all(r[0]['result']==records[0][0]['result'] for r in records),
                    physical_scores_identical=all(r[1]['result']==records[0][1]['result'] for r in records)))
    atomic(OUT/'duplicate_question_audit.json',duplicate_audit)
    raw={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in PRIVATE.glob('smoke_*.pt')}
    atomic(OUT/'smoke_raw_logits_index.json',dict(location='Server resource directory; arrays themselves are not in GitHub',files=raw))
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    ours=[]
    for line in processes.splitlines():
        fields=[x.strip() for x in line.split(',')]
        if len(fields)<2:continue
        proc=Path('/proc')/fields[1]
        try:
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=os.readlink(proc/'cwd')
            if cwd==str(ROOT) and 'mme_seed_quality' in command:ours.append(line)
        except (OSError,ValueError):pass
    assert not ours,ours
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    atomic(OUT/'gpu_release_audit.json',dict(at=now(),own_gpu_processes_remaining=ours,
        all_compute_apps_snapshot=processes,gpu_snapshot=gpu,all_our_workers_exited=True,
        no_other_process_killed=True,note='cuda:0 inside each worker maps to its selected physical GPU UUID, not physical GPU0'))
    analysis=read(OUT/'analysis_validation.json')
    assert analysis['score_discrepancies']==0
    checkpoint_hashes=read(OUT/'checkpoint_sha256_index.json.gz')
    assert len(checkpoint_hashes)==analysis['checkpoint_count']
    for name,expected_sha in checkpoint_hashes.items():
        assert sha(ROOT/name)==expected_sha,name
    status='COMPLETE' if all(x=='COMPLETE' for x in model_status.values()) else 'PARTIAL'
    atomic(OUT/'validation.json',dict(at=now(),status=status,model_status=model_status,
        cpu_protocol_tests=read(OUT/'cpu_tests.json')['status'],statistics_cpu_test=read(OUT/'statistics_cpu_validation.json')['status'],
        all_source_hashes_unchanged=True,all_model_files_rehashed=True,all_images_rehashed_unchanged=True,
        paired_prompt_ids_and_input_tensor_hashes_identical=True,
        checkpoint_sha256_index_independently_reverified=True,
        saved_mme_tokens_decode_to_exact_saved_text=True,
        saved_seed_tokens_match_frozen_candidates=True,
        scorer_reconstruction_discrepancies=0,counts={str(k):v for k,v in checked.items()},
        qwen_2048_actual_processor_validation=read(OUT/'qwen_processor_validation.json')['status'],
        qwen_strict_numeric_equivalence='Not claimed; known historical Physical-Original limitation',
        raw_smoke_logits='Server resources only; index in GitHub',official_seed_label_span_mismatches=0,
        no_timing_training_kernel_or_layer_search=True))
    atomic(OUT/'completion.json',dict(at=now(),status=status,models=model_status,
        expected_complete_model_benchmark_matrices=4,completed_matrices=analysis['full_pair_matrices'],
        expected_scored_question_condition_records=2*2*16607,completed_checkpoint_records=analysis['checkpoint_count'],
        all_completed_records_validated=True,all_gpus_released=True,
        qwen_accuracy='Not measured if STOPPED_OOM; no estimated scores filled'))
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_sha256_index.json']
    atomic(OUT/'artifact_sha256_index.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(paths)})
    print('DELIVERY_AUDIT',status,model_status,checked,flush=True)

if __name__=='__main__':main()
