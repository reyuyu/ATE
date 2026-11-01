"""Post-run artifact checks only: no forward calls, source edits or score changes."""
import csv
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from benchmarks.dataset_registry import digest, sha256
from knockout_probe import tensor_hash
from rq2_visual_handoff.run import OUT, ROOT, read, verify, valid_checkpoint


def main():
    protocol=verify();binding=digest(protocol);manifest=read(OUT/'sample_manifest.json')
    partition=read(OUT/'source_partition_audit.json');image_gpu={};checks=0
    def bind(s,rank):
        image=s['image_sha256'];assert image_gpu.setdefault(image,rank)==rank
    for s in manifest['samples']:
        rank=manifest['assignments']['main'][s['question_id']];bind(s,rank)
        for key in 'BCD':
            rec=valid_checkpoint(OUT/f"shard{rank}/main/{s['task']}__{s['question_id']}/{key}.json",binding,s,key)
            g=rec['result']['generation'];p=partition[s['question_id']]
            assert g['image_indices']==p['image_keys'] and g['relay_indices']==p['relay_queries'] and g['p_minus_one']==p['p_minus_one']
            assert g['mask_layers']==list(range(32))
            checks+=1
    for pair in read(OUT/'image_swap_manifest.json')['pairs']:
        rank=manifest['assignments']['swap'][pair['pair_id']]
        for s in pair['samples']:bind(s,rank)
    caps=read(OUT/'caption_manifest.json')['samples'];native_d=None;tokens_d=None;text_ids=None;dchecks=0
    for s in caps:
        rank=manifest['assignments']['caption'][s['question_id']];bind(s,rank)
        rec=valid_checkpoint(OUT/f"shard{rank}/caption/{s['question_id'].replace(':','_')}/D.json",binding,s,'D')
        with np.load(OUT/rec['raw_file'],allow_pickle=False) as z:current=torch.from_numpy(z['first_logits'].copy())
        if native_d is None:
            native_d=current;tokens_d=rec['result']['generation']['token_ids'];text_ids=s['input_mapping']['prompt_ids']
        assert s['input_mapping']['prompt_ids']==text_ids
        assert torch.equal(native_d,current) and rec['result']['generation']['token_ids']==tokens_d
        dchecks+=1
    model_audit={}
    for name,record in protocol['model_files'].items():
        found=sha256(name);assert found==record['sha256'];model_audit[name]=found
    source_path=ROOT/'results/paper_rq1_single_layer_gqa6/sample_manifest.json'
    actual_blob=subprocess.check_output(['git','hash-object',str(source_path)],text=True).strip()
    old_blob=subprocess.check_output(['git','rev-parse','221cdde56dde53908aaa953fe26672fd33334f84:vlm_research/results/paper_rq1_single_layer_gqa6/sample_manifest.json'],text=True).strip()
    assert actual_blob==old_blob
    review=read(OUT/'caption_blind_review_notes.json');revealed=[]
    for case in review['cases']:
        i=case['image_ordinal'];s=caps[i];assert s['question_id']==case['question_id']
        keys=list('ABCD');random.Random(20261101+i).shuffle(keys)
        rank=manifest['assignments']['caption'][s['question_id']]
        for note in case['outputs']:
            key=keys[note['blind_ordinal']]
            rec=read(OUT/f"shard{rank}/caption/{s['question_id'].replace(':','_')}/{key}.json")
            revealed.append(dict(question_id=s['question_id'],condition=key,image_path=s['image_path'],
                text=rec['result']['generation']['text'],**note))
    (OUT/'caption_blind_review_revealed.json').write_text(json.dumps(dict(notes=review['scope'],rows=revealed),ensure_ascii=False,indent=2)+'\n')
    budget=dict(formal_main_new=720,formal_image_swap_new=320,formal_caption_new=240,formal_new_total=1280,
        necessary_smoke_generations=42,smoke_breakdown_per_gpu=dict(custom_full=1,noop=1,HF_native=1,interventions=3,image_swap_endpoints_times_conditions=8),
        compatibility_full_prefill_only=240,primary_historical_full_generations_reused=240,
        note='Smoke runs and compatibility Prefills are not counted as independent research observations.')
    result=dict(passed=True,main_input_position_replays=checks,all_32_layers=True,
        cross_component_image_SHA256_sharding_consistent=True,unique_images=len(image_gpu),
        additional_caption_D_same_prompt_first_logits_exact_across_images=dchecks,
        additional_caption_D_token_sequences_exact=True,caption_D_first_logits_sha256=tensor_hash(native_d),
        original_3000_manifest_git_blob_exact=actual_blob,all_frozen_model_file_SHA256_rehashed_after_run=model_audit,
        execution_budget=budget,semantic_blind_review_cases=3,semantic_blind_review_outputs=12,
        JS_roundoff='Existing FP32 metric can yield ~1e-9 nat for identical logits; exact raw-logit comparisons are the invariance criterion. No threshold or numerical metric was changed.')
    (OUT/'additional_readonly_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
