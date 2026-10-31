"""Independent result acceptance; keep every executing/frozen worker file intact.

The frozen draft collector compares FP32 subtraction with Python FP64 subtraction
using exact equality. Replay the prescribed FP32 arithmetic instead, without any
tolerance relaxation or change to the experiment, estimates, or plotting code.
"""
from collections import Counter
import fcntl

import numpy as np
import torch

from benchmarks.dataset_registry import digest, freeze_json, sha256
from knockout_probe import tensor_hash, ROW_ATOL
from .run import OUT, ROOT, TASKS, SOURCES, read, verify
from .analyze import write_csv, summaries, figures, report


def collect():
    c=verify();binding=digest(c);manifest=read(OUT/'sample_manifest.json')
    complete=[read(OUT/f'shard{k}/completion.json') for k in range(2)]
    assert all(r['binding']==binding and r['status']=='complete' and r['hooks_clean'] and r['parameters_unchanged'] for r in complete)
    assert len(list(OUT.glob('shard*/checkpoints/*.json')))==240
    shards=[set(),set()];seen=set();collected=[];per=[];eos=0
    for sample in manifest['samples']:
        qid=sample['question_id'];rank=manifest['assignments'][qid]
        r=read(OUT/f"shard{rank}/checkpoints/{sample['task']}__{qid}.json")
        assert qid not in seen
        seen.add(qid);shards[rank].add(sample['image_sha256'])
        assert r['binding']==binding and r['sample_sha256']==digest(sample)
        assert (r['task'],r['question_id'],r['image_id'],r['image_sha256'],r['physical_gpu'])==(sample['task'],qid,sample['image_id'],sample['image_sha256'],rank+2)
        result=r['result']
        assert r['result_sha256']==digest(result)
        assert result['hooks_clean'] and result['inputs_unchanged'] and result['parameters_versions_unchanged']
        assert [(v['layer'],v['source']) for v in result['metrics']]==[(l,s) for l in range(32) for s in SOURCES]
        assert [v['layer'] for v in result['native_rows']]==list(range(32))
        full_path=OUT/result['full_logits_file']
        assert sha256(full_path)==result['full_logits_file_sha256']
        with np.load(full_path) as raw:
            array=raw['full_logits']
            assert array.dtype==np.float32 and array.ndim==1 and np.isfinite(array).all()
            full=torch.from_numpy(array)
        token=int(full.argmax());lp=full.log_softmax(-1)
        eos+=token==2
        for v in result['metrics']:
            layer,source=v['layer'],v['source'];native=result['native_rows'][layer]
            assert np.isfinite([x for x in v.values() if isinstance(x,(int,float))]).all()
            assert v['full_token_id']==token and v['full_token_logp']==float(lp[token])
            assert v['full_logits_sha256']==tensor_hash(full)
            # Exact replay of the original FP32 metric, not a new tolerance.
            expected=float(torch.tensor(v['full_token_logp'],dtype=torch.float32)-
                           torch.tensor(v['ablated_full_token_logp'],dtype=torch.float32))
            assert v['logp_loss']==expected and v['logp_loss']==-v['delta_logp']
            assert v['top1_flip']==(v['ablated_token_id']!=token)
            assert 0<=v['js']<=np.log(2)+2e-6
            expected_keys={'Image':576,'BOS':1,'Prompt':len(sample['input_mapping']['prompt_ids'])-578}[source]
            assert v['masked_connections']==expected_keys
            assert v['full_source_mass']==native[source] and v['native_self_mass']==native['Self']
            assert v['native_prompt_mass_including_self']==native['Prompt']+native['Self']
            assert native['row_error']<=ROW_ATOL and abs(sum(native[s] for s in [*SOURCES,'Self'])-1)<=ROW_ATOL
            per.append(dict(task=sample['task'],question_id=qid,image_id=sample['image_id'],
                image_sha256=sample['image_sha256'],physical_gpu=rank+2,**v))
        collected.append((sample,result))
    assert len(collected)==240 and len(per)==23040
    assert not shards[0]&shards[1]
    assert len(shards[0]|shards[1])==manifest['images']
    assert Counter(s['task'] for s,r in collected)==Counter({t:40 for t in TASKS})
    for history in read(OUT/'history_bindings.json'):
        assert sha256(history['first_token_checkpoint'])==history['checkpoint_sha256']
        assert sha256(history['native_mass_checkpoint'])==history['native_mass_sha256']
    write_csv(OUT/'per_question.csv',per)
    write_csv(OUT/'baseline_comparison.csv',[dict(task=s['task'],question_id=s['question_id'],**r['history']) for s,r in collected])
    freeze_json(OUT/'integrity.json',dict(passed=True,questions=240,interventions=23040,
        per_task=dict(Counter(s['task'] for s,r in collected)),images=manifest['images'],shard_questions=manifest['shard_questions'],
        no_duplicates=True,all_image_clusters_disjoint=True,binding=binding,full_baselines=240,
        native_full_exact_history=sum(r['history']['historical_full_exact'] for s,r in collected),
        historical_full_top1_same=sum(r['history']['historical_full_top1_same'] for s,r in collected),
        historical_full_max_abs_difference=max(r['history']['historical_full_max_abs_difference'] for s,r in collected),
        historical_mass_max_abs_difference=max(r['history']['historical_mass_max_abs_difference'] for s,r in collected),
        max_native_row_error=max(v['row_error'] for s,r in collected for v in r['native_rows']),
        gpu_seconds=[r['seconds'] for r in complete],per_question_csv_sha256=sha256(OUT/'per_question.csv'),
        native_first_token_eos_count=eos,fp32_loss_replay_exact_conditions=23040,
        scope='Original first-answer P-1 only; no EOS-query substitution, no decode, no accuracy.'))
    return collected


def main():
    lock=(OUT/'aggregation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify()
    freeze_json(OUT/'postprocessing_binding.json',dict(
        frozen_execution_protocol_sha256=sha256(OUT/'protocol.json'),
        sources={p:sha256(ROOT/p) for p in ['rq2_source_knockout/finalize.py','rq2_source_knockout/analyze.py']},
        reason='Strict FP32 subtraction replay; frozen analysis draft used Python FP64 exact subtraction. No change to worker code, tolerance, samples, metrics, bootstrap or figure definitions.'))
    collected=collect();print('Accepted all240 questions and23040 FP32 losses',flush=True)
    tables,correlations=summaries(collected)
    print('Image-cluster summaries complete',flush=True)
    figures(tables);report(tables,correlations)
    details=['','## 中层来源与任务一致性的直接对照','',
        '| 范围 | Image OFF平均Loss | Prompt OFF平均Loss | BOS OFF平均Loss | 最大平均Loss来源 |',
        '| --- | --- | --- | --- | --- |']
    leaders=[]
    for task in ['Overall',*TASKS]:
        means={s:float(np.mean([r['mean_logp_loss'] for r in tables
            if r['task']==task and r['source']==s and 14<=r['layer']<=16])) for s in SOURCES}
        lead=max(means,key=means.get)
        if task!='Overall': leaders.append(lead)
        details.append('| '+task+' L14-L16 | '+' | '.join(f'{means[s]:.6f}' for s in SOURCES)+' | '+lead+' |')
    details+=['',f"六类任务的中层最大平均损害来源计数：{dict(Counter(leaders))}。"+
        ('中层主导来源并不完全一致，不能将总体均值当作所有任务的共同机制。' if len(set(leaders))>1 else
         '六类任务在这个描述性均值排序上相同，但损害大小和峰值层并非因此等价，也未得到独立确认。'),'',
        '### L14、L15、L16逐层排序','']
    for layer in [14,15,16]:
        ranked=sorted([r for r in tables if r['task']=='Overall' and r['layer']==layer],key=lambda r:r['mean_logp_loss'],reverse=True)
        details.append(f"- L{layer}："+' > '.join(f"{r['source']} ({r['mean_logp_loss']:.6f})" for r in ranked)+'。')
    bos=[r for r in tables if r['task']=='Overall' and r['source']=='BOS']
    early=[r for r in bos if 2<=r['layer']<=5]
    early_loss=float(np.mean([r['mean_logp_loss'] for r in early]))
    other_loss=float(np.mean([r['mean_logp_loss'] for r in bos if not 2<=r['layer']<=5]))
    details+=['','### BOS主导与BOS路径干预不是同一指标','',
        f"L2-L5的BOS OFF平均Loss为{early_loss:.6f}，其余28层平均为{other_loss:.6f} nats；前者"+
        ('更低' if early_loss<other_loss else '不更低')+'。这是层均值的描述性对照，不是图片级独立确认或等价检验。',
        '若某层Image OFF损害小，只能说这条直接P-1到Image边在该条件下影响较小；不能排除已有残差流、其他文本表示及前后层传递的视觉信息。',
        '图中的灰、绿、浅红背景分别标记L0-L1、L2-L5、L13-L16；曲线没有平滑，热图使用统一对称色标。','']
    report_path=OUT/'report_zh.md'
    report_path.write_text(report_path.read_text()+'\n'.join(details))
    hashes={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['result_hashes.json','aggregation.lock','worker.lock'] and not p.name.endswith('.tmp')}
    freeze_json(OUT/'result_hashes.json',hashes)
    print('Final figures and Chinese report complete',flush=True)


if __name__=='__main__': main()
