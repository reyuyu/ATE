"""Freeze prompts and splits without looking at candidate answers."""
import gzip
import json
import subprocess
from collections import Counter, defaultdict

from prompt_common import *


def partition(sha256):
    return 'Discovery' if int(digest([SEED, 'partition', sha256]), 16) % 2 == 0 else 'Validation'


def choose(rows, stage, task):
    pool = [r for r in rows if r['benchmark'] == task and partition(r['image_sha256']) == stage]
    if task != 'MMBench':
        by_image = defaultdict(list)
        for row in pool:
            by_image[row['image_sha256']].append(row)
        keys = sorted(by_image, key=lambda k: digest([SEED, stage, task, k]))[:150]
        assert len(keys) == 150
        return [min(by_image[k], key=lambda r: digest([SEED, r['key']])) for k in keys]
    groups = defaultdict(list)
    for row in pool:
        groups[row['circular_group']].append(row)
    # All rotations share identical pixels; retain every row in each chosen group.
    assert all(len({r['image_sha256'] for r in g}) == 1 for g in groups.values())
    by_image = defaultdict(list)
    for key, group in groups.items():
        by_image[group[0]['image_sha256']].append(key)
    dp = {0: []}
    for image in sorted(by_image, key=lambda k: digest([SEED, stage, task, k])):
        for count in sorted(list(dp), reverse=True):
            new = count + len(by_image[image])
            if new <= 150 and new not in dp:
                dp[new] = dp[count] + [image]
        if 150 in dp:
            selected = [r for image in dp[150] for key in by_image[image] for r in groups[key]]
            assert len({r['circular_group'] for r in selected}) == 150
            return selected
    raise ValueError('Cannot select 150 complete image-disjoint Circular groups')


def main():
    import torch
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized() and not (OUT / 'protocol.json').exists()
    disk()
    original = json.loads((FIRST / 'protocol.json').read_text())
    assert sha(original['inventory_path']) == original['inventory_sha256']
    with gzip.open(original['inventory_path'], 'rt') as stream:
        rows = [r for r in map(json.loads, stream) if r['benchmark'] in TASKS]
    selected = [dict(r, stage=stage) for stage in ['Discovery', 'Validation'] for task in TASKS for r in choose(rows, stage, task)]
    discovery = {r['image_sha256'] for r in selected if r['stage'] == 'Discovery'}
    validation = {r['image_sha256'] for r in selected if r['stage'] == 'Validation'}
    assert not discovery & validation
    grouped = defaultdict(list)
    for r in selected:
        grouped[r['image_sha256']].append(r)
    load = [0, 0]
    for image, group in sorted(grouped.items(), key=lambda x: (-sum(r['image_tokens']**2*r['max_new_tokens'] for r in x[1]), x[0])):
        shard = min(range(2), key=lambda k: load[k])
        for row in group:
            row['shard'] = shard
        load[shard] += sum(r['image_tokens']**2*r['max_new_tokens'] for r in group)
    processor = AutoProcessor.from_pretrained(original['model'], local_files_only=True)
    examples = []
    for task in TASKS:
        pool = sorted([r for r in rows if r['benchmark'] == task], key=lambda r: (r['input_length'], r['key']))
        for label, fraction in [('shortest', 0), ('p25', .25), ('median', .5), ('p95', .95), ('longest', 1)]:
            examples.append(dict(pool[round(fraction*(len(pool)-1))], example_position=label))
    old = historical({r['key'] for r in selected + examples})
    baseline = Store(OUT / 'p0_reused.sqlite')
    for row in selected:
        a, b = old[row['key'], 'Native'], old[row['key'], 'SVH']
        assert a['input_binding'] == b['input_binding']
        assert a['protocol_sha256'] == b['protocol_sha256'] == sha(FIRST / 'protocol.json')
        assert a['image_sha256'] == b['image_sha256'] == row['image_sha256']
        assert a['gold'] == b['gold'] == row['gold']
        for prompt in conditions(row['benchmark']):
            new, text, ids = render(processor, row, prompt)
            if prompt == 'P0':
                assert ids == a['input_binding']['input_ids'] and text == a['input_binding']['prompt']
        for condition in ['Native', 'SVH']:
            rec = old[row['key'], condition]
            assert rec['max_new_tokens'] == row['max_new_tokens']
            baseline.put(row['key']+'|'+condition, rec)
        # Keep only binding hashes in the manifest; original bindings stay compressed.
        row['historical_input_binding_sha256'] = a['input_binding_sha256']
    baseline.conn.close()
    write(OUT / 'sample_manifest.json', selected)
    audit = []
    for row in examples:
        a, b = old[row['key'], 'Native'], old[row['key'], 'SVH']
        new, rendered, ids = render(processor, row, 'P0')
        assert rendered == a['input_binding']['prompt'] and ids == a['input_binding']['input_ids']
        assert a['input_binding'] == b['input_binding']
        audit.append(dict(benchmark=row['benchmark'], position=row['example_position'], key=row['key'],
            question=row['question'], hint=row.get('hint'), choices=row.get('options'), user_text=row['body'],
            complete_chat_prompt=rendered, token_ids=ids, token_ids_sha256=digest(ids), prompt_sha256=sha_bytes(rendered),
            native_svh_byte_identical=True, input_binding_sha256=a['input_binding_sha256'],
            image_tensor_hashes=a['input_binding']['input_tensor_hashes'], max_new_tokens=row['max_new_tokens'],
            generation=original['generation']))
    OUT.joinpath('prompt_examples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=True)+'\n' for r in audit))
    # Freeze deterministic performance probes independent of answers.
    probes = {}
    for shard in [0, 1]:
        for task in TASKS:
            pool = [r for r in selected if r['stage'] == 'Discovery' and r['shard'] == shard and r['benchmark'] == task]
            probes[f'{shard}|{task}'] = min(pool, key=lambda r: digest([SEED, 'performance', r['key']]))['key']
    write(OUT / 'performance_samples.json', probes)
    paths = {str(BASE / p): h for p, h in original['sources'].items()}
    paths.update(original['scorer_sources'])
    paths[str(PARSER)] = sha(PARSER)
    paths.update({str(p): sha(p) for p in ROOT.glob('*.py')})
    p = dict(name='QWEN3VL_SHORT_ANSWER_PROMPT_ABLATION', base_commit='78cbd30d0b322d2dc5e1352d1d9b7d2dcd5ab62a',
        model=original['model'], revision=original['revision'], generation=original['generation'], caps=original['caps'],
        runtime='BF16/Eager, unchanged pixel budget, original SVH algorithm', algorithm=original['semantics'],
        original_protocol_sha256=sha(FIRST / 'protocol.json'), frozen_source_sha256=paths,
        manifest_sha256=sha(OUT / 'sample_manifest.json'), prompts=dict(P0='Original unchanged', P1_GQA=P1_GQA, P1_choices=P1_CHOICE, P2_choices=P2_CHOICE),
        seed=SEED, bootstrap_replicates=20000, split='Global Image SHA hash parity; 150 questions/task or 150 whole Circular groups',
        discovery_selection='Maximize mean of Native/SVH frozen strict scores (Circular for MMBench); ties minimize worst-method damage vs P0, then mean token count, then P0/P1/P2 order. No Validation scores opened before freeze.',
        gpu_uuids=UUIDS, historical_numerical_status='FAIL_RETAINED; not a strict Physical revalidation',
        count=dict(Counter(r['stage']+'|'+r['benchmark'] for r in selected)), performance=dict(modes=['natural_EOS','fixed32'], warmups=1, repeats=3, probes_per_task=2, ignore_eos_only_in_fixed_budget_timing=True),
        disk_policy=dict(output_budget_mib=180, minimum_free_mib=512, main_two_gib_policy_unchanged=True, no_arrays_or_model_copies=True),
        plan='B: ATE unified HF; not author official reproduction', plan_A='BLOCKED', original_results_read_only=True)
    write(OUT / 'protocol.json', p)
    write(OUT / 'cpu_freeze_validation.json', dict(status='PASS', examples=15, image_disjoint=True, candidate_outputs_not_read=True,
        native_svh_identical_inputs=True, p0_reuse_records=len(selected)*2, cuda_initialized=torch.cuda.is_initialized(),
        group_counts={s:len({r['circular_group'] for r in selected if r['stage']==s and r['benchmark']=='MMBench'}) for s in ['Discovery','Validation']}))
    text = ['# 当前真实Prompt审计', '', '本报告读取当前运行树的冻结 inventory、实际渲染函数及两卡原始 Checkpoint，未用历史配方代替。', '',
        'GQA有明确短答指令：`Answer the question using a single word or phrase.`',
        'ScienceQA-IMG、MMBench均以 `Please select the correct answer from the options above. ` 结尾；没有只答字母、禁止解释或“不重复选项”的指令。',
        'ScienceQA/MMBench用户文本：Hint（如有）→Question→按原顺序Options→上述选择指令。原始Chat Template无额外短答System消息。',
        '生成：Greedy，EOS=[151645,151643]；GQA cap32，ScienceQA/MMBench cap128。模型BF16/Eager、原始视觉预算不变。',
        '15个例子的完整Prompt、Token IDs、输入Tensor哈希及Native/SVH逐字节一致性见 prompt_examples.jsonl。', '',
        '历史Physical严格数值门禁仍为FAIL_RETAINED；本轮是独立Prompt协议，不改判历史失败。', '']
    for e in audit:
        text += [f"## {e['benchmark']} / {e['position']} / {e['key']}", '```text', e['complete_chat_prompt'], '```',
                 f"Token数={len(e['token_ids'])}; Token-ID SHA256={e['token_ids_sha256']}; Native/SVH输入完全一致。", '']
    OUT.joinpath('current_prompt_audit_zh.md').write_text('\n'.join(text))
    print(json.dumps(p['count']), flush=True)


def sha_bytes(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == '__main__':
    main()
