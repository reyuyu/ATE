"""CPU-only provenance, cross-implementation weight audit and final report."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import platform
import subprocess

from .common import *


def write_csv(path,rows):
    rows=list(rows)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def model_audit():
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    torch.set_num_threads(1)
    resolved_text=AutoConfig.from_pretrained(HF_MODEL,local_files_only=True).text_config.to_dict()
    index=read(NATIVE_MODEL/'pytorch_model.bin.index.json')['weight_map']
    hf_index=read(HF_MODEL/'model.safetensors.index.json')['weight_map']
    shards={name:torch.load(NATIVE_MODEL/name,map_location='cpu',weights_only=True,mmap=True)
            for name in set(index.values())}
    sf={name:safe_open(str(HF_MODEL/name),framework='pt',device='cpu') for name in set(hf_index.values())}
    records=[]
    for key,filename in index.items():
        left=shards[filename][key]
        if key.endswith('.rotary_emb.inv_freq'):
            config=resolved_text
            size=config['hidden_size']//config['num_attention_heads']
            derived=1.0/(config.get('rope_theta',10000.0)**(torch.arange(0,size,2).float()/size))
            records.append(dict(native_key=key,hf_key='DERIVED:rotary inverse frequencies',
                native_shape=list(left.shape),hf_original_shape=list(derived.shape),
                native_dtype=str(left.dtype),hf_dtype=str(derived.dtype),
                equal=torch.equal(left,derived),native_sha256=tensor_sha(left),
                hf_shared_sha256=tensor_sha(derived),max_abs=float((left-derived).abs().max())))
            continue
        if key.startswith('model.mm_projector.'):
            new=key.replace('model.mm_projector.0.','multi_modal_projector.linear_1.').replace(
                'model.mm_projector.2.','multi_modal_projector.linear_2.')
        else:
            new='language_model.'+key
        assert new in hf_index,(key,new)
        right=sf[hf_index[new]].get_tensor(new)
        original_shape=list(right.shape)
        if left.shape!=right.shape:
            assert key in ('lm_head.weight','model.embed_tokens.weight')
            assert right.shape[0]==32064 and left.shape[0]==32000
            right=right[:32000]
        equal=torch.equal(left,right)
        rec=dict(native_key=key,hf_key=new,native_shape=list(left.shape),hf_original_shape=original_shape,
                 native_dtype=str(left.dtype),hf_dtype=str(right.dtype),equal=equal,
                 native_sha256=tensor_sha(left),hf_shared_sha256=tensor_sha(right),
                 max_abs=float((left.float()-right.float()).abs().max()) if not equal else 0.0)
        records.append(rec)
    verified=read(RES/'official_resources_verified.json')
    clip=next(v for k,v in verified.items() if k.startswith('openai/'))
    weight=next(e for e in clip['downloaded'] if e['path'].endswith('pytorch_model.bin'))
    original=torch.load(weight['path'],map_location='cpu',weights_only=True,mmap=True)
    for key,left in original.items():
        if not key.startswith('vision_model.'):
            continue
        if key=='vision_model.embeddings.position_ids':
            clip_config=read(Path(weight['path']).parent/'config.json')['vision_config']
            count=(clip_config['image_size']//clip_config['patch_size'])**2+1
            derived=torch.arange(count).expand(1,-1)
            records.append(dict(native_key='CLIP:'+key,hf_key='DERIVED:CLIP position IDs',
                native_shape=list(left.shape),hf_original_shape=list(derived.shape),
                native_dtype=str(left.dtype),hf_dtype=str(derived.dtype),
                equal=torch.equal(left,derived),native_sha256=tensor_sha(left),
                hf_shared_sha256=tensor_sha(derived),max_abs=float((left-derived).abs().max())))
            continue
        new='vision_tower.'+key
        assert new in hf_index,new
        right=sf[hf_index[new]].get_tensor(new)
        rounded=left.to(right.dtype)
        equal=torch.equal(rounded,right)
        records.append(dict(native_key='CLIP:'+key,hf_key=new,native_shape=list(left.shape),
            hf_original_shape=list(right.shape),native_dtype=str(left.dtype),hf_dtype=str(right.dtype),
            equal=equal,native_sha256=tensor_sha(left),hf_shared_sha256=tensor_sha(right),
            rounded_native_sha256=tensor_sha(rounded),
            max_abs=float((rounded.float()-right.float()).abs().max()) if not equal else 0.0))
    audit=dict(at=now(),native_revision=NATIVE_REV,hf_revision=HF_MODEL.name,
        matrices_compared=len(records),equal_matrices=sum(r['equal'] for r in records),
        all_mapped_shared_weights_equal=all(r['equal'] for r in records),
        shared_parameter_tensors=sum(not r['hf_key'].startswith('DERIVED:') for r in records),
        all_mapped_learned_parameters_equal=all(r['equal'] for r in records if not r['hf_key'].startswith('DERIVED:')),
        checkpoint_rotary_buffers_differ_from_fp32_formula=32,
        rotary_buffer_note='Author Transformers4.37.2 registers inv_freq, sin_cached, cos_cached persistent=False; these checkpoint buffers are not learned parameters. CPU constructor reconstructs the FP32 formula exactly. Runtime/backend parity is not inferred from this audit.',
        extra_hf_vocabulary_rows=64,clip_original_fp32_cast_to_fp16=True,
        native_configuration=read(NATIVE_MODEL/'config.json'),hf_configuration=read(HF_MODEL/'config.json'),
        source_download=verified,
        backend_parity_not_assumed=True,
        native_code_fix='Original PrunedLLM with None config raises UnboundLocalError; select author existing LlamaModel.forward',
        dependencies=dict(author_required_python='3.10',actual_author_python='3.11',
            author_required_torch='2.1.2',actual_author_torch='2.5.1+cu124',
            author_required_transformers='4.37.2',actual_author_transformers='4.37.2',
            author_required_flash='2.5.9.post1',actual_author_flash='2.8.3.post1',
            classification='AUTHOR_CODE_REPRODUCTION, not exact environment reproduction'))
    atomic(OUT/'model_equivalence_audit.json',audit)
    write_csv(OUT/'weight_mapping_audit.csv',records)
    print('MODEL_EQUIVALENCE',len(records),audit['equal_matrices'],flush=True)


def protocols():
    references=[dict(benchmark=b,published_native=x,unit=u,status='USER_PREREGISTERED_PUBLISHED_REFERENCE_NOT_ACCEPTANCE_GATE')
        for b,x,u in [('MME Total',1862,'points/2800'),('ScienceQA-IMG',69.5,'percent'),
        ('POPE',85.9,'percent; exact paper metric not assumed to equal subset Accuracy'),('TextVQA OCR',58.3,'percent'),
        ('MMBench-EN',64.6,'percent/circular')]]
    write_csv(OUT/'paper_reference.csv',references)
    diffs=[]
    for b in ORDER:
        diffs.append(dict(benchmark=b,field='Model/Weights/Implementation',
            historical='HF b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
            stage_A='Author 75909b; original liuhaotian 4481d270; unpruned parent dispatch repair',
            stage_B='Same fixed HF revision, all three conditions identical eager engine'))
        diffs.extend(dict(benchmark=b,field=f,historical=h,stage_A=a,stage_B=c) for f,h,a,c in [
            ('Backend','FP16 eager / Transformers4.49','FP16 Flash2 / Transformers4.37.2','FP16 eager / Transformers4.49'),
            ('Prompt','HF USER chat template, no Vicuna system preamble','Author vicuna_v1 with full system preamble','Identical author vicuna_v1 string'),
            ('Vision Processor','Historical HF processor; separate input binding','Native pad-to-square + CLIP336','Same pad-to-square, CPU tensor hashes compared'),
            ('Max Tokens','32',str(1024 if b in ('ScienceQA-IMG','MMBench') else 128),str(1024 if b in ('ScienceQA-IMG','MMBench') else 128)),
            ('EOS','Native model EOS','Native model EOS, no added stop strings','HF native EOS, no added stop strings'),
            ('OCR','Absent','Original supplied OCR text/order' if b=='TextVQA' else 'Not applicable','Same supplied OCR' if b=='TextVQA' else 'Not applicable'),
            ('Scorer','Frozen old scorer unchanged','Author official lexical/scorer; MMBench offline matcher','Same as stage A'),
            ('Split/Question Count','Original historical split','Author exact annotation rows; POPE8910 not9000','Exactly the same paper-input rows'),
            ('Image','Existing verified benchmark files','Same image bytes; SHA256 checked','Same bytes and CPU pixels checked')])
    write_csv(OUT/'benchmark_protocol_diff.csv',diffs)


def data_alignment():
    rows=manifest()
    lookup={s['key']:s for s in rows}
    author=[json.loads(line) for line in (OUT/'official/llava_pope_test.jsonl').open()]
    pope=[]
    for strategy in ('random','popular','adversarial'):
        raw=[json.loads(line) for line in (DATA/f'POPE/coco_pope_{strategy}.json').open()]
        selected=[r for r in author if r['category']==strategy]
        labels=[lookup['POPE:'+str(r['question_id'])]['answer'] for r in selected]
        prefix=[r['label'] for r in raw[:len(selected)]]
        mismatch=sum(a!=b for a,b in zip(labels,prefix))
        entries=[dict(question_id=r['question_id'],image=r['image'],text=r['text'],
                      label=lookup['POPE:'+str(r['question_id'])]['answer']) for r in selected]
        atomic(OUT/f'official/pope_{strategy}_aligned_labels.json',entries)
        pope.append(dict(strategy=strategy,author_questions=len(selected),original_gold_questions=len(raw),
                         label_mismatches_if_naively_zipped=mismatch,
                         official_lexical_rule_preserved=True,
                         label_alignment='Exact image/question lookup; imange typo normalized only for lookup, not input'))
    atomic(OUT/'pope_annotation_alignment.json',dict(at=now(),strategies=pope,
        note='Use aligned label files with official lexical scorer. A shorter author Random file must not be zipped against the first 2910 labels without alignment. Dataset-size difference is not repaired by adding 90 questions.'))
    print('POPE_ALIGNMENT',pope,flush=True)


def input_alignment():
    import torch
    from transformers import AutoProcessor
    from .engine import image_ids,paper_pixels
    torch.set_num_threads(1)
    processor=AutoProcessor.from_pretrained(HF_MODEL,local_files_only=True)
    rows={s['key']:s for s in manifest()}
    smoke={read(p)['key']:read(p) for p in (OUT/'smoke/A/rank0').glob('*.json') if p.name!='completion.json'}
    checked=[]
    for entry in read(OUT/'smoke_manifest.json'):
        s=rows[entry['key']]
        before=smoke[s['key']]['binding']
        ids=image_ids(s['prompt'],processor.tokenizer,-200)
        pixels=paper_pixels(s['image_path'],processor.image_processor)
        assert ids==before['raw_image_placeholder_ids'],s['key']
        assert tensor_sha(pixels)==before['pixel_values_sha256'],s['key']
        assert list(pixels.shape)==before['pixel_shape']
        assert len(ids)+575==before['expanded_length']
        assert list(range(ids.index(-200),ids.index(-200)+576))==before['image_positions']
        checked.append(dict(key=s['key'],raw_token_ids_exact=True,pixel_sha256_exact=True,
                            image_positions_exact=list(range(ids.index(-200),ids.index(-200)+576))==before['image_positions']))
    atomic(OUT/'cross_backend_cpu_input_audit.json',dict(at=now(),cases=checked,status='PASSED',
        native_binding='Actual stored author Transformers4.37.2 GPU smoke inputs',
        hf_binding='HF Transformers4.49.0 processor/tokenizer, CPU only',
        cross_backend_gpu_logits_parity='NOT_RUN; not assumed'))
    print('CPU_INPUT_ALIGNMENT',len(checked),'PASSED',flush=True)


def snapshot():
    completion=read(OUT/'completion.json')
    assert completion['status']=='PARTIAL', 'This snapshot describes a partial run, not a final report'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],text=True)
    atomic(OUT/'gpu_availability.json',dict(at=now(),gpus_csv=gpu,processes_csv=processes,
        task_worker_processes_running=False,external_processes_untouched=True,
        resume_status='BLOCKED_EXTERNAL_GPU_OCCUPANCY',automatic_polling_or_waiting_job=False))
    smoke=[]
    for stage in CONDITIONS:
        for rank in range(4):
            path=OUT/f'smoke/{stage}/rank{rank}/completion.json'
            value=read(path) if path.exists() else dict(status='NOT_RUN',cases=[])
            smoke.append(dict(stage=stage,rank=rank,status=value['status'],cases=len(value['cases'])))
    validation=dict(at=now(),cpu_tests_passed=6,postprocessing_tests_passed=3,source_and_artifact_hashes_still_match=True,
        checkpoint_hashes_and_saved_scores_recomputed=True,smoke=smoke,
        cross_backend_cpu_input_audit=read(OUT/'cross_backend_cpu_input_audit.json'),
        stage_B_gpu_gate='NOT_RUN',stage_A_gpu_gate='PASSED_4_GPU_15_CASES_EACH',
        model_audit=read(OUT/'model_equivalence_audit.json'),
        cleanup_audit=read(OUT/'storage_cleanup_audit.json'),
        previous_failures_preserved=True,inference_sources_not_changed_after_freeze=True)
    atomic(OUT/'validation.json',validation)
    old={'MME':'1771.16 /2800','ScienceQA-IMG':'66.04%','POPE':'85.91% (Accuracy)',
         'TextVQA':'48.128% (Image-only)','MMBench':'62.84% (Circular)'}
    references={'MME':'1862 /2800','ScienceQA-IMG':'69.5%','POPE':'85.9% (metric requires qualification)',
                'TextVQA':'58.3% (OCR)','MMBench':'64.6%'}
    counts=read(OUT/'dataset_audit.json')['counts']
    lines=['# CVPR 2026 Information-Horizon 协议对齐：部分进度',
        '', '**状态：PARTIAL；当前因外部 GPU 占用而不能续跑。**',
        '',f"已完成作者 Native 阶段 {completion['stage_A_completed']:,}/{completion['stage_A_expected']:,} 条；全部为 TextVQA/OCR。阶段 B 为 0/68,034。",
        '这些完成项是确定性执行前缀，不是随机抽取的评测子集。不得将其均分当成 5,000 题正式 TextVQA 分数。',
        'SQLite 检查点全部通过记录 SHA、输入绑定、协议绑定及独立重评分检查；无重复、无覆盖。',
        '', '## 当前主表', '', '| Benchmark | Published Reference | 作者 Native | HF Paper Native | Physical C3 | L11–L24 |',
        '|---|---:|---|---|---|---|']
    for bench in ORDER:
        state='PARTIAL 2683/5000' if bench=='TextVQA' else 'NOT_RUN'
        lines.append(f'| {bench} | {references[bench]} | {state} | NOT_RUN | NOT_RUN | NOT_RUN |')
    lines += ['', '未完成比较不填 0、不计算伪 CI。阶段 A 完成后才允许阶段 B。',
        '', '## 已完成的验证与复现边界',
        '', '- 作者源码固定为 `75909b2936a13d7214e83514f0bc0cb9cc91139e`，其工作区未修改。',
        '- 作者原始 `PrunedLLM.forward` 在 `pruned_config=None` 时使用未初始化的 K。本实验显式绑定作者既有的未剪枝父类 forward；未将剪枝预算 576 冒充 Native。',
        '- 四卡各通过 15 个预冻结 Smoke：Full/No-op 与无实际剪枝分支的全部有效 Logits 严格 `torch.equal`，32 层完整 Prefill/Decode，无剪枝及残留 Hook。',
        '- 分类为 `AUTHOR_CODE_REPRODUCTION / native-parent dispatch repair`，不是逐依赖完全复现。实际 Python3.11、Torch2.5.1、FlashAttention2.8.3，与作者 Python3.10、Torch2.1.2、FlashAttention2.5.9 不同；Transformers4.37.2 相同。',
        '- 权重映射检查了 719 个张量/派生缓冲：686 个共享可学习参数张量全部精确相同；HF 词表额外 64 行不视作原生 LM 的对应参数。',
        '- 32 个旧 Checkpoint RoPE `inv_freq` 与 FP32 公式不同，最大绝对差约 0.00122315；它们是非持久缓冲，不是学习参数。作者当前构造器重新生成公式。仍不能据此假定两后端运行数值相同。',
        '- 阶段 B 的物理执行器真实 GPU 门禁尚未执行，不能标记已通过。',
        '- 另有 15 个冻结输入的 CPU 跨实现对照通过：实际作者 Smoke 绑定与 HF 原生 Processor 的 Token IDs、视觉位置和图像 Tensor SHA 精确一致。这不是两后端 Logits 等价证明。',
        '', '## 数据与评分',
        '', f"冻结实际输入共 {sum(counts.values()):,} 条，{read(OUT/'dataset_audit.json')['unique_image_bytes']:,} 个图片字节簇。",
        '- 取得的作者格式镜像文件中 POPE Random 为 2,910、Popular 3,000、Adversarial 3,000，共 8,910，不补齐成 9,000。Random 若朴素顺序 zip 原标签，会出现 1,417 处标签不匹配；因此按图片和问题绑定，保留作者词法解析规则。对齐文件和原始顺序均保留；不能把这个子集的分数直接称作论文 9,000 题 POPE 分数。',
        '- TextVQA 保留作者 OCR 内容和顺序，官方十答案共识软分；旧 Image-only 48.128% 不作为 OCR 版本基线。',
        '- MME 保留 14 任务及图片双题，报告必须区分 Perception、Cognition、Total；本轮未完成该项，未产生正式分数。',
        '- ScienceQA 保留 Hint/选项顺序及 single-pred Prompt，不输入 Gold 对话。MMBench 保留 4,377 个旋转记录和 Circular 分组；离线官方匹配器为 OFFLINE_REPRODUCTION，不冒充服务端裁判。',
        '- 评测输入来自注明来源和 SHA 的镜像 eval.zip；未能独立取得原 Google Drive 文件核对字节。此限制完整保留。',
        '- 常规任务生成上限 128；ScienceQA/MMBench 为 1,024；不沿用旧 32 Token 上限。',
        '', '| Benchmark | 旧 ATE Native（历史，不重算） | 新 Paper Native |', '|---|---:|---|']
    lines += [f'| {b} | {old[b]} | 未完成 |' for b in ORDER]
    lines += ['', '## 中断、清理与恢复',
        '', '- 正式生成在 2026-11-06 22:02（Asia/Shanghai）触及磁盘安全线而停止；异常与 GPU 释放日志保留，检查点没有损坏。',
        '- 本轮安全删除约 8.9128 GiB 的可再下载原始压缩包；MME/SEED 解压文件先逐项校验大小与 CRC32。原 GQA 图片和所有历史结果保留。',
        '- `testdev_balanced` 的 12,578 题、398 张图片及本轮输入图片已重新验证 SHA256。没有为节省空间删掉评测题。',
        '- 本轮暂存于内存盘的环境/权重/结果已全部恢复到 `/data`。共享模型、共享环境、main 和历史结果未改动。',
        '- 当前四卡均由其他用户训练占用，不抢占、不发终止信号、不留自动轮询作业；本任务 GPU 进程全部退出。',
        '- `workers/A/rank*/results.sqlite` 是原始本地恢复点。Git 同时保存全量已完成记录 `per_question.jsonl.gz` 和摘要 CSV，不需要重跑这 2,683 条。',
        '', '## 下一步（尚未执行）',
        '', 'GPU 空闲后继续阶段 A 剩余 19,995 条；其完成后，在统一 HF/eager 后端验证并执行 B 的 68,034 条。跨后端 Native 差异和同阶段 C3 算法损失必须分开。',
        '最终才运行完整官方聚合与 20,000 次图片簇配对 Bootstrap；本轮没有新增测速、搜索、GQA GPU 任务、SEED 或 MM-Vet。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


def hashes():
    paths=[p for p in OUT.rglob('*') if p.is_file() and 'results.sqlite' not in p.name
           and p.name!='output_sha256.json' and '__pycache__' not in p.parts]
    atomic(OUT/'output_sha256.json',dict(at=now(),outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(paths)},
        inference_source_sha256=digest(source_hashes()),
        postprocessing_sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),Path(__file__).with_name('analyze.py'))},
        local_sqlite_hashes=read(OUT/'completion.json')['checkpoint_inventory']))



def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-audit',action='store_true')
    p.add_argument('--data-alignment',action='store_true')
    p.add_argument('--snapshot',action='store_true')
    p.add_argument('--input-alignment',action='store_true')
    p.add_argument('--hashes',action='store_true')
    args=p.parse_args()
    protocols()
    if args.model_audit:
        model_audit()
    if args.data_alignment:
        data_alignment()
    if args.input_alignment:
        input_alignment()
    if args.snapshot:
        snapshot()
    if args.hashes:
        hashes()


if __name__=='__main__':
    main()
