"""Build a static provenance catalog and two NON-EXECUTING protocol designs."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent
WORKTREE = ROOT.parents[3]
BASE = 'f238d375ffda785d3062069ce6e4fd0c729cd9a6'
OLD = '7e18895a6ccd57c4cb3aaedd5d512789e7467b59'
V = '051aa811ec27c494919bda944654de884fd70c41'
L = 'f9b577e76bac2fb5a17f6e2d5f9ad9b69156c292'
Q = '96588727e44c78b25ba03ea03b8e12f7e64fd0da'
REV = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
MODEL = Path('/data/lk/ATE-resources/models/qwen3-vl-8b-instruct') / REV
TASKS = ['GQA', 'ScienceQA-IMG', 'TextVQA', 'POPE', 'MME', 'MMBench']
EVIDENCE = []


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def add(key, path, commit, start, end, category, claim, repo=None):
    path = Path(path).resolve()
    text = path.read_text()
    lines = text.splitlines()
    assert 1 <= start <= end <= len(lines), (key, len(lines), start, end)
    EVIDENCE.append(dict(id=key, evidence_category=category, repository=repo or 'reyuyu/ATE',
                         commit_or_version=commit, source_path=str(path), start_line=start, end_line=end,
                         source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), claim=claim,
                         excerpt_sha256=hashlib.sha256('\n'.join(lines[start-1:end]).encode()).hexdigest()))


def table(name, rows):
    with (ROOT / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build():
    inherited = AUDIT / 'official_sources'
    v = ROOT / 'sources/open-compass__VLMEvalKit'
    l = ROOT / 'sources/EvolvingLMMs-Lab__lmms-eval'
    a = WORKTREE / 'vlm_research/results/qwen3vl_c3_3090_official_eval/official_sources'
    qreadme = inherited / 'QwenLM__Qwen3-VL/README.md'
    add('Q01', qreadme, Q, 1135, 1153, 'MODEL_AUTHOR_DIRECT', 'Official reproduction runtime vLLM; framework names, Instruct global stochastic parameters; no per-task framework SHA', 'QwenLM/Qwen3-VL')
    add('Q02', qreadme, Q, 337, 355, 'MODEL_AUTHOR_README_EXAMPLE', 'Pixel-budget control example (256-1280 visual tokens) is not a six-task evaluation budget', 'QwenLM/Qwen3-VL')
    add('Q03', qreadme, Q, 414, 436, 'MODEL_AUTHOR_README_EXAMPLE', 'Qwen3 image_patch_size=16; avoid duplicate resize after qwen-vl-utils', 'QwenLM/Qwen3-VL')
    add('Q04', qreadme, Q, 858, 879, 'MODEL_AUTHOR_README_DEPLOYMENT', 'vLLM >=0.11.0 deployment recommendation, not an exact evaluation runtime lock', 'QwenLM/Qwen3-VL')
    paper = ROOT / 'sources/arxiv/2511.21631v1.txt'
    add('P01', paper, 'arXiv:2511.21631v1 Appendix A', 4370, 4389, 'MODEL_AUTHOR_DIRECT', 'General VQA benchmark explicitly MMBench-V1.1; does not attest other five requested tasks', 'Qwen3-VL Technical Report')
    add('P02', paper, 'arXiv:2511.21631v1 Appendix B.2', 4610, 4630, 'MODEL_AUTHOR_DIRECT', 'MMBench question/options/select-correct prompt; no exact split/judge/versioned execution command', 'Qwen3-VL Technical Report')
    add('V01', inherited / 'open-compass__VLMEvalKit/vlmeval/config.py', V, 1474, 1485, 'FRAMEWORK_CANDIDATE', 'Registered Qwen3-VL-8B-Instruct uses vLLM, custom prompt false, temperature .7, cap16384', 'open-compass/VLMEvalKit')
    add('V02', v / 'vlmeval/inference.py', V, 175, 193, 'FRAMEWORK_CANDIDATE', 'Effective dispatch selects dataset.build_prompt when model custom prompt disabled', 'open-compass/VLMEvalKit')
    vm = inherited / 'open-compass__VLMEvalKit/vlmeval/vlm/qwen3_vl/model.py'
    add('V03', vm, V, 46, 102, 'FRAMEWORK_CANDIDATE', 'Constructor defaults and HF generation kwargs; presence_penalty absent in HF kwargs', 'open-compass/VLMEvalKit')
    add('V04', vm, V, 105, 170, 'FRAMEWORK_CANDIDATE', 'vLLM seed=0, TP visible devices, unpinned processor/model revision; HF flash_attention_2', 'open-compass/VLMEvalKit')
    add('V05', vm, V, 171, 193, 'FRAMEWORK_CANDIDATE', 'No six-task explicit pixel cap in registration; overrides may arrive via input metadata', 'open-compass/VLMEvalKit')
    add('V06', vm, V, 344, 405, 'FRAMEWORK_CANDIDATE', 'Chat template, patch16, vLLM sampling cap/penalties; stop_token_ids None; request seed not explicit', 'open-compass/VLMEvalKit')
    add('V07', v / 'vlmeval/dataset/image_base.py', V, 362, 379, 'FRAMEWORK_CANDIDATE', 'Base dataset prompt is image then raw question', 'open-compass/VLMEvalKit')
    add('V08', v / 'vlmeval/dataset/image_vqa.py', V, 55, 125, 'FRAMEWORK_CANDIDATE', 'GQA/TextVQA dataset URL+MD5, single-word suffix, GQA accuracy vs TextVQA vqa_score dispatch', 'open-compass/VLMEvalKit')
    add('V09', v / 'vlmeval/dataset/image_mcq.py', V, 58, 80, 'FRAMEWORK_CANDIDATE', 'MMBench v1.0 and v1.1 are distinct IDs; ScienceQA_TEST does not by itself prove IMG-only filter', 'open-compass/VLMEvalKit')
    add('V10', v / 'vlmeval/dataset/image_mcq.py', V, 211, 245, 'FRAMEWORK_CANDIDATE', 'MCQ effective dataset prompt: optional Hint, Question, Options, select-correct suffix', 'open-compass/VLMEvalKit')
    add('V11', v / 'vlmeval/dataset/image_mcq.py', V, 254, 321, 'FRAMEWORK_CANDIDATE', 'Circular inference flag, exact_matching default; nonworking judge silently falls back in framework', 'open-compass/VLMEvalKit')
    add('V12', v / 'vlmeval/dataset/image_yorn.py', V, 15, 102, 'FRAMEWORK_CANDIDATE', 'POPE/MME raw question unchanged by prompt builder; yes/no suffix only added for AMBER; parser/judge dispatch', 'open-compass/VLMEvalKit')
    add('V13', v / 'vlmeval/dataset/utils/multiple_choice.py', V, 359, 395, 'FRAMEWORK_CANDIDATE', 'Rule inference before optional judge; no judge returns failed parse', 'open-compass/VLMEvalKit')
    add('V14', v / 'vlmeval/dataset/utils/multiple_choice.py', V, 511, 562, 'FRAMEWORK_CANDIDATE', 'Circular groups default index modulo 1e6, aggregate all option rotations', 'open-compass/VLMEvalKit')
    add('V15', v / 'run.py', V, 363, 410, 'FRAMEWORK_CANDIDATE', 'CLI judge/--judge-args overrides dataset default; launch command needed to determine effective judge', 'open-compass/VLMEvalKit')
    lm = inherited / 'EvolvingLMMs-Lab__lmms-eval/lmms_eval/models/simple/qwen3_vl.py'
    add('L01', lm, L, 68, 161, 'FRAMEWORK_CANDIDATE', 'qwen3_vl simple adapter is HF BF16; cap128 class default; pixel kwargs 256*1024 to 2048*1024; system message', 'EvolvingLMMs-Lab/lmms-eval')
    add('L02', lm, L, 327, 353, 'FRAMEWORK_CANDIDATE', 'Task generation kwargs override model defaults; greedy if temperature0; explicit tokenizer EOS', 'EvolvingLMMs-Lab/lmms-eval')
    add('L03', lm, L, 403, 417, 'FRAMEWORK_CANDIDATE', 'until fallback decoded tokenizer EOS; configured stop strings also trimmed post-generation', 'EvolvingLMMs-Lab/lmms-eval')
    add('L04', lm, L, 450, 469, 'FRAMEWORK_CANDIDATE', 'Image PIL entries explicitly get min/max pixels; not safe to equate this budget with VLMEvalKit defaults', 'EvolvingLMMs-Lab/lmms-eval')
    add('L05', l / 'lmms_eval/api/task.py', L, 886, 909, 'FRAMEWORK_CANDIDATE', 'Model-specific kwargs use exact model name: qwen_vl alias does not automatically apply to qwen3_vl', 'EvolvingLMMs-Lab/lmms-eval')
    cfgs = [('L10', 'gqa/gqa.yaml'), ('L11', 'scienceqa/scienceqa_img.yaml'),
            ('L12', 'textvqa/textvqa_val.yaml'), ('L13', 'pope/pope_random.yaml'),
            ('L14', 'mme/mme.yaml'), ('L15', 'mmbench/_default_template_mmbench_en_yaml')]
    for key, name in cfgs:
        path = l / 'lmms_eval/tasks' / name
        add(key, path, L, 1, len(path.read_text().splitlines()), 'FRAMEWORK_CANDIDATE', f'Candidate task YAML {name}, not model-author runtime attestation', 'EvolvingLMMs-Lab/lmms-eval')
    add('L16', l / 'lmms_eval/tasks/scienceqa/utils.py', L, 1, 43, 'FRAMEWORK_CANDIDATE', 'Actual default Hint+Question+ordered choices; zero-based Gold index -> letter; response parser', 'EvolvingLMMs-Lab/lmms-eval')
    add('L17', l / 'lmms_eval/tasks/textvqa/utils.py', L, 15, 54, 'FRAMEWORK_CANDIDATE', '10-answer leave-one-out scoring; capitalize question, OCR conditional on missing included template', 'EvolvingLMMs-Lab/lmms-eval')
    add('L18', l / 'lmms_eval/tasks/mmbench/en_utils.py', L, 21, 36, 'FRAMEWORK_CANDIDATE', 'Judge default gpt-4o-2024-11-20 from MODEL_VERSION; external API needed for unresolved outputs', 'EvolvingLMMs-Lab/lmms-eval')
    add('L19', l / 'lmms_eval/tasks/mme/utils.py', L, 30, 45, 'FRAMEWORK_CANDIDATE', 'MME actual prompt removes original yes/no suffix when applying configured post_prompt', 'EvolvingLMMs-Lab/lmms-eval')
    add('S01', a / 'lupantech__ScienceQA/tools/evaluate_acc.py', '2cbf8318e07b9ece895bb2ae605e71e38d623264', 22, 64, 'DATASET_AUTHOR', 'Official Gold is integer; full4241 test, image subset; does not define a universal free-text parser', 'lupantech/ScienceQA')
    add('S02', a / 'facebookresearch__mmf/mmf/utils/m4c_evaluators.py', '72f898ec35af86567423c7a019dd1dc5175d1a5a', 219, 252, 'DATASET_REFERENCE_SCORER', 'TextVQA official-style EvalAI normalized ten-answer consensus', 'facebookresearch/mmf')
    add('S03', a / 'RUCAIBox__POPE/evaluate.py', '08d957b917e5a378a2f99d35b6293c536a66298b', 9, 34, 'DATASET_AUTHOR', 'POPE first sentence; No/not/no words -> no, else yes including empty', 'RUCAIBox/POPE')
    add('S04', a / 'MME_calculation.py', 'dd2950902889cd614d4edf606827240166d29381 tools/eval_tool.zip member eval_tool/calculation.py', 23, 36, 'DATASET_AUTHOR', 'MME first four characters yes/no parser, else other', 'BradyFU/Awesome-Multimodal-Large-Language-Models')
    add('S05', a / 'MME_calculation.py', 'dd2950902889cd614d4edf606827240166d29381 tools/eval_tool.zip member eval_tool/calculation.py', 90, 147, 'DATASET_AUTHOR', 'MME original image pairs, Gold lowercased, per-task Acc+Acc+', 'BradyFU/Awesome-Multimodal-Large-Language-Models')
    add('S06', a / 'open-compass__MMBench/README.md', '8eaac9ae42e22eaf42863b218455acd51268ba1c', 70, 78, 'DATASET_AUTHOR', 'CircularEval all rotations; rule extraction followed by ChatGPT for ambiguous answers', 'open-compass/MMBench')
    add('M01', MODEL / 'generation_config.json', REV, 1, len((MODEL/'generation_config.json').read_text().splitlines()), 'MODEL_REVISION_ASSET', 'EOS151645/151643; stochastic defaults; not a six-task effective cap or seed', 'Qwen/Qwen3-VL-8B-Instruct')
    add('M02', MODEL / 'preprocessor_config.json', REV, 1, len((MODEL/'preprocessor_config.json').read_text().splitlines()), 'MODEL_REVISION_ASSET', 'patch16/merge2; frozen processor pixel range65536..16777216; not author task budget', 'Qwen/Qwen3-VL-8B-Instruct')
    add('M03', MODEL / 'tokenizer_config.json', REV, 229, 235, 'MODEL_REVISION_ASSET', 'Tokenizer EOS im_end, pad endoftext; HF simple adapter explicit single EOS differs from generation_config EOS list', 'Qwen/Qwen3-VL-8B-Instruct')
    add('H01', AUDIT / 'historical_evidence/qwen3vl_c3v2_prefill_av_final/report_zh.md', 'fc3f10eabf48a30b05f4e927a00f50bcd20780ad', 25, 35, 'ATE_HISTORICAL_FAILURE', 'Mean<=.02/max<=.5 frozen gate; median input remains failed; not repaired by protocol design')
    add('H02', AUDIT / 'historical_evidence/qwen3vl_c3_3090_official_eval/protocol_review/report_zh.md', OLD, 40, 48, 'ATE_HISTORICAL_FAILURE', 'TextVQA eager vision15552 tokens and115.33GiB request; old numerical gate not PASS')
    add('H03', WORKTREE / 'vlm_research/qwen3vl_c3v2_physical/engine.py', BASE, 5, 16, 'ATE_REUSABLE_HF_ONLY', 'Physical engine imports HF Qwen3 modeling/Cache/mask; not a vLLM port')

    table('parameter_evidence.csv', EVIDENCE)
    write_json('evidence_catalog.json', EVIDENCE)
    links = ['# Evidence Index', '', 'Line numbers refer to the byte-hashed archived files. Framework candidate != author attestation.', '',
             '| ID | Evidence Type | Commit / Version | Source | Claim |', '| --- | --- | --- | --- | --- |']
    for e in EVIDENCE:
        links.append(f"| {e['id']} | {e['evidence_category']} | `{e['commit_or_version']}` | [L{e['start_line']}-L{e['end_line']}]({e['source_path']}:{e['start_line']}) | {e['claim']} |")
    (ROOT / 'evidence_index.md').write_text('\n'.join(links) + '\n')

    prompts = {
        'GQA': '{question}\nAnswer the question using a single word or phrase.',
        'ScienceQA-IMG': '[Hint: {hint}\n if nonempty]Question: {question}\nOptions:\nA. {choice0}\n...Please select the correct answer from the options above. \n',
        'TextVQA': '{question}\nAnswer the question using a single word or phrase.',
        'POPE': '{question}\nPlease answer yes or no.',
        'MME': '{original_question_including_original_yes_no_suffix}',
        'MMBench': '[Hint: {hint}\n if nonempty]Question: {question}\nOptions:\nA. {choiceA}\n...Please select the correct answer from the options above. \n',
    }
    ids = {
        'GQA': ('GQA_TestDev_Balanced / original testdev_balanced', 12578, 'question_id -> imageId,question,answer', 'V08;L10'),
        'ScienceQA-IMG': ('ScienceQA official test where image nonempty', 2017, 'pid -> zero-based answer -> original choices[answer]', 'S01;L11;L16'),
        'TextVQA': ('TextVQA_0.5.1 validation', 5000, 'question_id -> image_id + ten original answers', 'S02;L12;L17'),
        'POPE': ('RUCAIBox/POPE output/coco 3000 per strategy', 9000, '(strategy,raw_question_id) -> image,text,label', 'S03;L13'),
        'MME': ('MME classic release 14 tasks', 2374, '(task,image_filename,question_index) -> original yes/no; preserve two-question pairs', 'S04;S05;V12;L14'),
        'MMBench': ('MMBench legacy EN dev 20230712, NOT author V1.1', 4377, 'original index -> choices,Gold; Circular1176;1132 image hashes', 'S06;V09;V14;L15;L18'),
    }
    scorers = {
        'GQA': 'ATE exact normalized match: strip/lower only against original answer; no LLM judge',
        'ScienceQA-IMG': 'Pinned lmms sqa_process_results parser (L16) -> letter -> original integer Gold; invalid=wrong',
        'TextVQA': 'Pinned MMF TextVQAAccuracyEvaluator (S02), all ten answers, no judge',
        'POPE': 'Pinned dataset-author evaluate.py extraction (S03); Accuracy/F1/Precision/Recall/yes ratio per strategy; separately flag empty',
        'MME': 'Pinned original calculation.py (S04/S05);14 task Acc+Acc+;report perception,cognition,total',
        'MMBench': 'Pinned VLMEvalKit exact_matching extraction + CircularEval (V11,V13,V14); NO API fallback; diagnostic non-author-judge variant',
    }
    caps = {'GQA':32, 'ScienceQA-IMG':128, 'TextVQA':128, 'POPE':128, 'MME':128, 'MMBench':128}
    candidate_prompts = {
        'GQA': 'VLMEval dataset question + short-word suffix (V07,V08); lmms gqa default same suffix, cap16 (L10,L05)',
        'ScienceQA-IMG': 'VLMEval optional Hint+Question+Options+select (V10), IMG-only filter unresolved; lmms IMG default Context+Q+options+letter request (L11,L16,L05)',
        'TextVQA': 'VLMEval question + short-word suffix no explicit OCR insertion (V08); lmms capitalize+optional OCR (L17); missing included template prevents full cap/OCR resolution (L12)',
        'POPE': 'VLMEval raw dataset question (V12), NOT disabled model PromptMixin yes/no suffix; lmms random default short-word suffix cap128 (L13)',
        'MME': 'VLMEval raw question including any in-data suffix (V12); lmms removes original yes/no suffix then applies short-word default cap16 (L14)',
        'MMBench': 'VLMEval Hint+Question+Options+select (V10); lmms own hint/options + letter request cap1024/until ASSISTANT: (L15)',
    }
    matrix = []
    plan_tasks = []
    for task in TASKS:
        dataset, count, mapping, refs = ids[task]
        matrix.append(dict(task=task, author_status='PARTIAL_EVIDENCE_STILL_UNRESOLVED' if task=='MMBench' else 'UNRESOLVED',
                           A_author_framework_commit='UNRESOLVED; candidates V=' + V + ', L=' + L,
                           A_author_dataset='MMBench-V1.1; exact split/hash UNRESOLVED (P01)' if task=='MMBench' else 'UNRESOLVED author task dataset revision/split',
                           A_author_prompt='Question/Options/select per Appendix B.2; optional Hint and chat/system still UNRESOLVED (P02)' if task=='MMBench' else 'UNRESOLVED',
                           A_author_runtime='vLLM global README (Q01); exact version/backend/command UNRESOLVED',
                           A_author_vision_budget='UNRESOLVED per-task; README examples != evaluation budgets (Q02,Q03,M02)',
                           A_author_OCR='UNRESOLVED per-task',
                           A_author_sampling='global reference sample,T=.7,p=.8,k20,rep1,presence1.5,seed3407 (Q01); task overrides UNRESOLVED',
                           A_author_EOS_cap='UNRESOLVED effective task EOS/cap; global out_seq_length32768 != verified task cap; revision EOS in M01',
                           A_author_scorer_judge='UNRESOLVED model-author chosen scorer version/judge; dataset references '+refs,
                           candidate_effective_prompt=candidate_prompts[task],
                           B_dataset_id=dataset, B_rows=count, B_gold_mapping=mapping,
                           B_framework_commit='ATE '+BASE+' design + explicitly pinned scorer sources '+refs,
                           B_prompt=prompts[task], B_OCR='NO external OCR or reference OCR tokens; image-only input; non-official choice',
                           B_image='RGB, official Qwen dynamic processor patch16/merge2; explicit ATE pixel budget65536..2097152; resize once; native mRoPE/DeepStack',
                           B_sampling='greedy do_sample=false,beams1,rep1,no presence penalty; seed3407; all rows,no few-shot,no subsampling',
                           B_EOS_cap=f'EOS[151645,151643],pad151643,cap{caps[task]},no extra string stopping; report truncated/empty',
                           B_scorer=scorers[task], B_status='DESIGN_ONLY_NOT_AUTHOR_OFFICIAL_NOT_EXECUTED',
                           evidence_ids='Q01;Q02;Q03;M01;M02;'+refs))
        plan_tasks.append(dict(task=task, dataset_id=dataset, count=count, gold_mapping=mapping,
                               body_template=prompts[task], max_new_tokens=caps[task], scorer=scorers[task]))
    table('six_task_protocol_matrix.csv', matrix)
    parameters = []
    for row in matrix:
        for key, value in row.items():
            if key.startswith(('A_', 'B_')):
                parameters.append(dict(task=row['task'], scheme=key[0], parameter=key[2:], value=value,
                                       status=row['author_status'] if key.startswith('A_') else row['B_status'],
                                       evidence_ids=row['evidence_ids'],
                                       note='A missing values stay unresolved; B values are proposed choices, never author facts'))
    table('parameter_matrix.csv', parameters)
    write_json('plan_B_ate_hf_design.json', dict(status='DESIGN_ONLY_NOT_EXECUTED_NON_AUTHOR_OFFICIAL',
               source_audit_commit=BASE, model_revision=REV, framework_candidate_commits=dict(vlmeval=V,lmms=L),
               runtime=dict(backend='HF',torch='2.6.0+cu124',transformers='4.57.6',dtype='BF16',batch_size=1,
                            attention='eager identical in Native and SVH',logits_to_keep=1,
                            generation=dict(do_sample=False,num_beams=1,repetition_penalty=1.0,presence_penalty=None,
                                            seed=3407,eos_token_id=[151645,151643],pad_token_id=151643)),
               input=dict(chat='frozen model apply_chat_template(add_generation_prompt=True)',system_message=None,
                          content_order=['image','task_body'],image_mode='RGB',resize_once=True,
                          pixel_budget=dict(min_pixels=65536,max_pixels=2097152,provenance='ATE_PROPOSED_NOT_AUTHOR'),
                          ocr='no external OCR/no reference OCR list; native vision OCR capability unchanged',
                          fewshot=0,all_rows=True),
               total_questions=sum(t['count'] for t in plan_tasks),tasks=plan_tasks,
               checks=['exact full-image SHA and input tensor binding before any reused predictions',
                       'full model/reference/no-op and same-row LM head strict torch.equal',
                       'physical-vs-dense original numerical gate unchanged; retained FAILED until future successful retest',
                       'full dataset processor/vision memory preflight; OOM -> BLOCKED, no silent cap/split/backend changes',
                       'MMBench exact_matching is diagnostic non-author-judge; any optional judge scored separately, no silent fallback',
                       'no runs authorized by this design artifact'],
               physical_status='NUMERICAL_EQUIVALENCE_FAILED_UNCHANGED',
               confidence_intervals='future image-SHA clustered paired bootstrap, preserve MME pairs/MMBench circular groups'))

    findings = {'scope': 'STATIC_PROVENANCE_ONLY', 'gpu_runs': 0, 'cuda_initialized': False,
                'new_predictions': 0, 'vllm_installed': False, 'physical_algorithm_modified': False,
                'official_author_reproduction_resolved': False,
                'author_task_recipe_status': {t: ('MMBENCH_VERSION_AND_BODY_PROMPT_PARTIALLY_RESOLVED' if t=='MMBench' else 'UNRESOLVED') for t in TASKS},
                'retained_physical_gate': 'NUMERICAL_EQUIVALENCE_FAILED',
                'paper': dict(version='2511.21631v1',html_sha256=hashlib.sha256((paper.with_suffix('.html')).read_bytes()).hexdigest(),
                             extracted_text_sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
                             task_name_matches={t: len(re.findall(r'\b'+re.escape(t)+r'\b',paper.read_text(),re.I)) for t in TASKS},
                             pdf_retrieval='timeout; HTML+deterministic text extraction used; no PDF page anchors asserted'),
                'missing_candidate_dependencies':['VLMEvalKit utils/vqa_eval.py','VLMEvalKit utils/yorn.py',
                                                   'lmms TextVQA _default_template_textvqa_yaml'],
                'network_failures_preserved':['additional_source_manifest.json','supplementary_fetch_manifest.json','targeted_fetch_manifest.json'],
                'route_recommendation':'B is lower engineering cost but still blocked on Physical numerical parity; A needs author provenance and same-runtime port'}
    write_json('static_findings.json', findings)


if __name__ == '__main__':
    build()
