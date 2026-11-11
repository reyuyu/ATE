import sys,json,hashlib,datetime,subprocess,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/'code'))
from common import *
import torch,transformers
from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
p=CODE/'engine.py';s=p.read_text()
start=s.index(' for name,t in x.items():')
end=s.index(" return row,{k:v.to(device)",start)
s=s[:start]+""" for name,t in x.items():
  expected=binding['pixels']['tensor_sha256'] if name=='pixel_values' else binding['tensors'][name]['sha256']
  assert th(t)==expected,('TENSOR_BINDING',key,name)
"""+s[end:]
p.write_text(s)
assert not (OUT/'protocol.json').exists(),'Never overwrite frozen protocol'
assert read(OUT/'cpu_tests.json')['status']=='PASS'
assert not torch.cuda.is_initialized()
protocol=dict(experiment='Qwen3-VL-8B Figure7 Decode-only visual reaccess budget',status='FROZEN_BEFORE_FORMAL_GPU',model='Qwen/Qwen3-VL-8B-Instruct',revision=S['revision'],GPU=GPU,
source_figure7_commit=S['figure7_commit'],source_figure7_result='vlm_research/results/textvqa500_early_middle_late_random_budget',
figure7_original500_manifest_sha256=read(OUT/'input_validation.json')['source_figure7_manifest_sha256'],
sample_manifest_sha256=sha(OUT/'sample_manifest.json'),input_bindings_sha256=sha(OUT/'input_bindings.json'),
budgets=S['budgets'],integer_midpoint_rule='floor the midpoint of each 9-layer quarter',N_layers=N,
middle_rule='start=(36+4-k)//2; continuous nested window; original center17.5 shifted to19.5; odd window lower floor',
layer_sets=configs(),random_seed=20261108,random_rule='one Python3.11 random.Random(seed).shuffle(range(36)); first k nested; never select best seed',
dtype='bfloat16',vision_backend='eager',text_backend='eager',batch=1,greedy=True,max_new_tokens=32,natural_EOS=True,use_cache=True,logits_to_keep=1,
prompt='exact Figure7 question plus newline Answer the question using a single word or phrase.; native Qwen chat template',
processor='native pinned Qwen Processor with original dynamic pixels; no resizing override, no external OCR; Image-only',
prefill='fully native in all36 layers INCLUDING P-1; original Image Query, R and text queries untouched',
decode='only true autoregressive decode; selected layers native, other layers add -inf for original Image Key before Softmax',
native_computation='all QKV, q/k norms, mRoPE, o_proj, MLP, residual, DeepStack and full dense real KV remain original; no SVH skip, hidden exit or physical pruning',
cache_reuse='one native vision+prefill per question; bitwise-cloned full real DynamicCache and rope_deltas for each condition; only if Smoke verifies full logits/tokens against fresh independent prefills',
generation_conditions=30,manifest_N=500,planned_max_generations=15000,
OOM_policy=dict(authorized_user_reply='沿用历史OOM策略，报告共同有效子集（推荐）',keep_frozen500=True,retain_all_BLOCKED=True,score_zero_for_OOM=False,replace_or_resample=False,change_backend_dtype_or_image=False,denominator='intersection of COMPLETE questions across ALL30 conditions; same subset for every plotted point and Native',missingness='OOM resolution-dependent; subset accuracy is not complete500 and not directly equal to LLaVA500 denominator',new_OOM='save original failure, clean only own failed request, retain BLOCKED for remaining conditions, continue other unchanged questions',non_OOM='hard stop on any nonfinite, input/source mismatch, structural/cache/position error or corrupted checkpoint'),
scorer=dict(source='exact frozen Figure7 official M4C TextVQA ten-answer leave-one-out consensus',source_sha256=sha(RUN/'references/textvqa_eval.py'),independent_recompute=True),
statistics=dict(bootstrap_replicates=20000,cluster='Image SHA256 whole clusters, paired questions',seed=20261112,CI='ordinary95%; exploratory multiple comparisons, no multiplicity adjustment, not equivalence'),
environment_differences_from_Figure7=['Qwen model and native template/Processor/DeepStack/mRoPE','36 language layers with frozen9point mapping','BF16, Transformers4.57.6 required by Qwen, vs LLaVA FP16/4.49.0','Qwen native dynamic visual token counts416-4800 vs LLaVA576','explicit common non-OOM subset, frozen500 retained'],
input_Gold_separate=True,scope='functional dense-mask quality only; no efficiency or physical C3 claim',frozen_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
write(OUT/'protocol.json',protocol)
files={f.name:sha(f) for f in CODE.glob('*.py')}
source=dict(files=files,official_transformers_source={str(Path(hf.__file__)):sha(hf.__file__)},scorer={str(RUN/'references/textvqa_eval.py'):sha(RUN/'references/textvqa_eval.py')},parent_commit=S['parent'],figure7_commit=S['figure7_commit'],model_revision=S['revision'])
write(OUT/'source_manifest.json',source)
environment=dict(python=sys.version,python_executable=sys.executable,torch_version=torch.__version__,torch_import_path=torch.__file__,transformers_version=transformers.__version__,transformers_import_path=transformers.__file__,Qwen_HF_import_path=hf.__file__,torch_CUDA_build=torch.version.cuda,CUDA_initialized=torch.cuda.is_initialized(),GPU_readonly=gpu(),model_path=S['model'],environment_path=str(Path(sys.executable).parent.parent),pip_check=subprocess.check_output([sys.executable,'-m','pip','check'],text=True).strip(),nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.limit,power.draw,temperature.gpu,memory.total,memory.used','--format=csv'],text=True),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True))
write(OUT/'environment.json',environment)
audit="""# Qwen Figure7 OOM 策略来源审计

2026-11-12；仅只读核查 GitHub 和已完成的本轮容量 Smoke。

- Commit 066c037f88931aeb73937e34b663b61de3f24ba5，qwen3vl_svhr_l11_l24_budget_search/progress_report_zh.md：历史 Validation 500题中498有效。36759、36747保留 BLOCKED，六条件合计12条。原始Validation_blocked_records.jsonl.gz已独立解压验证，不补零、不换题。
- Commit a9030774cf88205dd5b84b158ff0f7257e4ef3e8，qwen_svh_r_l19_l35_sixbench/report_zh_partial.md：TextVQA Native/SVH各4995/5000；完整分数仍 BLOCKED/PARTIAL，available/common subset另报。qwen_svh_r_full.py169及232–245行确认捕获 OOM、保留失败并清理自身缓存，不更换后台、精度或图片。
- 上述500题不是本轮Figure7的同一500题。独立核对ID：36759在本轮；36747不在本轮。全量另五例的完整ID清单未在上述轻量交付公开，不臆测其他重叠。
- 本轮36759已原图、BF16、Vision/LLM全Eager真实OOM；4800视觉Token，原始输入4833Token。失败在官方Vision eager QK矩阵分配，非Decode权限Mask。
- 用户明确确认沿用历史OOM策略：保持原500 Manifest、原图、全Eager。所有条件共用完整成功交集评分，标明N/500，保留BLOCKED，不补零、不换题、不将子集称为完整500结果。
- 历史SVH算法和Paper-OCR成绩不用于本轮；本轮严格沿用Figure7的Image-only、Native Prefill/P−1、仅真实Decode视觉访问权限。

来源：
https://github.com/reyuyu/ATE/blob/066c037f88931aeb73937e34b663b61de3f24ba5/vlm_research/results/qwen3vl_svhr_l11_l24_budget_search/progress_report_zh.md
https://github.com/reyuyu/ATE/blob/a9030774cf88205dd5b84b158ff0f7257e4ef3e8/vlm_research/results/qwen_svh_r_l19_l35_sixbench/report_zh_partial.md
"""
(OUT/'oom_policy_audit_zh.md').write_text(audit)
print(json.dumps(dict(status='FROZEN',protocol_sha256=sha(OUT/'protocol.json'),source_manifest_sha256=sha(OUT/'source_manifest.json'),CUDA_initialized=torch.cuda.is_initialized())),flush=True)
