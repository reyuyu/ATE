"""Freeze protocols and smoke examples before generating any benchmark answer."""
from collections import defaultdict,Counter
import importlib.metadata
import platform
import subprocess

from .common import *


def main():
    assert (OUT/'sample_manifest.jsonl.gz').exists()
    rows = manifest()
    selected = []
    for bench in ORDER:
        pool = sorted((s for s in rows if s['benchmark']==bench),
                      key=lambda s:(len(s['prompt']),s['key']))
        selected.extend(dict(benchmark=bench,key=pool[i]['key']) for i in (0,len(pool)//2,len(pool)-1))
    atomic(OUT/'smoke_manifest.json',selected)
    artifacts = {p:sha(OUT/p) for p in ('sample_manifest.jsonl.gz','smoke_manifest.json',
        'dataset_audit.json','author_repo_freeze.json')}
    shared = dict(created_at=now(),model_revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
        native_revision=NATIVE_REV,author_commit=AUTHOR_SHA,greedy=True,temperature=0,num_beams=1,
        batch_size=1,conversation='vicuna_v1',vision='CLIP patch14/336,576tokens,pad-to-square',
        conditions_by_stage=CONDITIONS,shards=4,physical_devices=UUIDS,
        counts=dict(Counter(s['benchmark'] for s in rows)),sources=source_hashes(),artifacts=artifacts,
        image_cluster_bootstrap_repeats=20000,bootstrap_seed=20261106,
        intervals='Exploratory paired image-SHA cluster; MME task-stratified; MMBench circular units',
        performance_benchmark=False,no_search=True,published_reference_is_not_acceptance_gate=True,
        historical_results_not_overwritten=True,
        actual_pope_count=8910,pope_count_difference='Author evaluation archive random=2910,popular=3000,adversarial=3000; do not pad to 9000',
        max_new_tokens={'TextVQA':128,'MME':128,'ScienceQA-IMG':1024,'MMBench':1024,'POPE':128},
        eos='Model generation_config EOS, no added stop strings',
        mmbench='OFFLINE_REPRODUCTION; exact official VLMEvalKit matcher; row/circular both; no server judge',
        author_native_fix=dict(problem='PrunedLLM.forward unbound K when pruned_config=None',
            selection='Use existing author LlamaModel.forward, bound on decoder instance only',
            unmodified_author_checkout=True,config_pruned_config=None,all_pruning_environment_absent=True,
            proof='CPU exact hidden+KV and GPU strict effective-logits equality to inactive K=-1 branch',
            classification='AUTHOR_CODE_REPRODUCTION / native-parent dispatch repair'))
    a = dict(shared,stage='A',implementation='NATIVE_PAPER_FLASH2',
        model='liuhaotian/llava-v1.5-7b',transformers='4.37.2',pytorch='2.5.1+cu124',flash_attention='2.8.3.post1',
        dependency_deviation='Author pyproject torch2.1.2; actual isolated env shares torch2.5.1 and flash2.8.3 read-only; not exact paper environment',
        backend='flash_attention_2',dtype='float16',logits_shape='Author full prefill LM head',
        temporary_weights='Verified original downloaded weights in task-owned /dev/shm staging via /data resource symlink; no history deleted')
    b = dict(shared,stage='B',implementation='HF_PAPER_INPUT_EAGER',model='llava-hf/llava-1.5-7b-hf',
        transformers='4.49.0',pytorch='2.6.0+cu124',backend='eager',dtype='float16',logits_to_keep=1,
        conditions=dict(B0='HF-Native',B1=dict(name='Physical-C3',R_writing=list(range(20)),
            answer_image=[],physical_exit_before=20,answer_attention_skip=[2,3,4,5]),
            B2=dict(name='Physical-L11-L24',R_writing=list(range(20)),answer_image=list(range(11,25)),
            physical_exit_before=25,answer_attention_skip=[2,3,4,5])),
        numerical_thresholds=dict(logit_atol=0.10,logit_rtol=0.003,
            hidden_relative_rmse=0.01,hidden_cosine=1e-4,source='Frozen historical llava physical_visual_handoff TOLERANCES'),
        same_backend_paired_only=True)
    atomic(OUT/'protocol_paper_native.json',a)
    atomic(OUT/'protocol_ate_paper.json',b)
    atomic(OUT/'execution_freeze.json',dict(at=now(),git_base=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=source_hashes(),
        stages={k:sha(OUT/name) for k,name in (('A','protocol_paper_native.json'),('B','protocol_ate_paper.json'))}))
    print('PROTOCOLS_FROZEN',len(rows),selected,flush=True)


if __name__ == '__main__':
    main()
