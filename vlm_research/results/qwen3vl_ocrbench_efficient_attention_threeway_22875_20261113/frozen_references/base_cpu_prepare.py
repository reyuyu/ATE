"""Phase1: CPU audit and unit tests. No CUDA initialization, no model load."""
import sys,json,gzip,hashlib,shutil,math,inspect
from pathlib import Path
O=Path('/home/featurize/ate-experiments/qwen3vl-ocrbench-memory-efficient-attention-22875-20261113');L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113');OLD=L/'results/qwen';M=Path('/home/featurize/ate-local-models/Qwen3-VL-8B-Instruct')
sys.path[:0]=[str(O),str(L/'sources/qwen'),str(L)]
from run_common import atomic,sha
import torch,transformers
from transformers import AutoProcessor
from PIL import Image
from types import SimpleNamespace
from qwen_frozen_benchmark_inputs import prepare,digest
from attention_backend import sdpa_compute,align_mask
torch.set_num_threads(1)
assert not torch.cuda.is_initialized() and transformers.__version__=='4.57.6' and torch.__version__=='2.6.0+cu124'
assert not (O/'protocol.json').exists()
old=json.loads((OLD/'protocol.json').read_text());rows=json.loads((OLD/'rows.json').read_text());layouts=json.load(gzip.open(OLD/'cpu_layouts.json.gz','rt'));assert len(rows)==len(layouts)==1000
for p,h in old['source_SHA256'].items():assert sha(L/'sources/qwen'/p)==h,p
protected={str(p):sha(p) for p in [OLD/'protocol.json',OLD/'failure.json',OLD/'progress.json',OLD/'cpu_layouts.json.gz',OLD/'rows.json',L/'qwen_frozen_runtime.py',L/'qwen_prepare.py',L/'qwen_run.py',L/'qwen_frozen_benchmark_inputs.py']}
for p in (OLD/'predictions').rglob('*.json'):protected[str(p)]=sha(p)
atomic(O/'historical_protected.json',protected)
tests=[];torch.manual_seed(20261113)
for groups in [1,4]:
 for kind in ['causal','noncausal','custom_mask']:
  q=torch.randn(1,4,9,16,dtype=torch.float64);k=torch.randn(1,4//groups,9,16,dtype=torch.float64);v=torch.randn_like(k);mask=None
  if kind=='custom_mask':mask=torch.zeros(1,1,9,9,dtype=torch.float64);mask[...,0:3]=-torch.inf
  module=SimpleNamespace(num_key_value_groups=groups,is_causal=kind=='causal');actual,_=sdpa_compute(module,q,k,v,mask,16**-.5)
  kk=k.repeat_interleave(groups,dim=1);vv=v.repeat_interleave(groups,dim=1);scores=q@kk.transpose(-2,-1)*16**-.5
  if kind=='causal':scores.masked_fill_(torch.ones(9,9,dtype=torch.bool).triu(1),-torch.inf)
  if mask is not None:scores=scores+mask
  expected=(scores.softmax(-1)@vv).transpose(1,2).contiguous();torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
  if mask is not None:assert align_mask(mask).stride(-2)%8==0 and torch.equal(align_mask(mask),mask)
  tests.append(dict(groups=groups,kind=kind,status='PASS',max_abs=float((actual-expected).abs().max())))
proc=AutoProcessor.from_pretrained(M,local_files_only=True);prepared={};qids=['0308','0601','0738']
for q in qids:
 row=next(r for r in rows if r['question_id']==q);x,l=prepare(proc,row);assert digest(l)==old['all_layout_hashes'][q]
 with Image.open(row['image_path']) as im:size=list(im.size)
 prepared[q]=dict(image_sha256=row['image_sha256'],original_image_wh=size,pixel_values_shape=list(x['pixel_values'].shape),pixel_values_dtype=str(x['pixel_values'].dtype),image_grid_thw=l['image_grid_thw'],input_length=l['input_length'],image_tokens=l['image_tokens'],input_tensor_hashes=l['input_tensor_hashes'],layout_sha256=digest(l));del x
cfg=json.loads((M/'config.json').read_text());w=json.loads((M/'model.safetensors.index.json').read_text())['metadata']['total_size'];l=layouts['0738'];patches=math.prod(l['image_grid_thw']);n=l['input_length'];heads=cfg['vision_config']['num_heads'];text=n-l['image_tokens']
est=dict(vision_scores_shape=[1,heads,patches,patches],vision_scores_bf16_bytes=heads*patches*patches*2,vision_scores_bf16_GiB=heads*patches*patches*2/2**30,language_scores_shape=[1,32,n,n],language_scores_bf16_GiB=32*n*n*2/2**30,language_softmax_fp32_GiB=32*n*n*4/2**30,native_KV_prefill_GiB=36*2*8*n*128*2/2**30,native_KV_cap128_GiB=36*2*8*(n+127)*128*2/2**30,SVH_KV_live_prefill_MiB=30*2*8*text*128*2/2**20,SVH_KV_capacity128_MiB=30*2*8*(text+127)*128*2/2**20,one_dense_SVH_mask_GiB=n*n*2/2**30,old_three_SVH_masks_GiB=3*n*n*2/2**30,weight_GiB=w/2**30,mlp_one_intermediate_MiB=n*12288*2/2**20,vision_sequence_partition=[0,patches],vision_spatial_window=False,temporal_spatial_merge=[cfg['vision_config']['temporal_patch_size'],cfg['vision_config']['spatial_merge_size']],lm_head_dim=128,vision_head_dim=72)
atomic(O/'static_memory.json',est);atomic(O/'cpu_inputs.json',prepared);atomic(O/'cpu_tests.json',dict(status='PASS',cases=tests,CUDA_initialized=torch.cuda.is_initialized(),model_loaded=False))
assert not torch.cuda.is_initialized()
protocol=dict(status='FROZEN_BEFORE_GPU',model=old['model'],revision=old['revision'],old_evidence_commit='a4fb52e3aa4d57385bd1108832c58a28471ecea3',frozen_source_commit=old['source_commit'],source_SHA256=old['source_SHA256'],old_protocol_sha256=sha(OLD/'protocol.json'),rows_sha256=sha(OLD/'rows.json'),cpu_layouts_sha256=sha(OLD/'cpu_layouts.json.gz'),all_layout_hashes=old['all_layout_hashes'],conditions=['Native','Physical-Original'],expected_each=1000,smoke_ids=qids,dtype='bfloat16',torch=torch.__version__,transformers=transformers.__version__,attention='sdpa',allowed_CUDA_backends=['EFFICIENT_ATTENTION'],forbidden_CUDA_backends=['MATH','FLASH_ATTENTION','CUDNN_ATTENTION'],GQA='explicit exact KV head expansion for efficient kernel compatibility',mask_storage='last-dimension stride aligned to8; logical mask elements and keys unchanged',processor=proc.image_processor.to_dict(),max_new_tokens=128,batch_size=1,greedy=True,natural_eos=True,logits_to_keep=1,old_eager_predictions_for_comparison_only=True,no_image_budget_change=True,model_loads_planned=1,phase5_requires_native_and_SVH_gates=True,historical_numerical_failure_preserved=True,legacy_dense_physical_numerical_gate=old['frozen_protocol']['numerical_gate'],cross_backend_differences='diagnostic, not algorithm failure',no_GitHub_push=True)
atomic(O/'protocol.json',protocol)
source=Path(transformers.__file__).parent/'models/qwen3_vl/modeling_qwen3_vl.py';atomic(O/'installed_source.json',dict(path=str(source),sha256=sha(source),eager_scores_line=155,vision_chunks_lines=[224,244],cu_seqlens_lines=[727,735],text_SDPA_dispatch_lines=[440,454]))
(O/'oom_root_cause_zh.md').write_text(f'''# 0738 CPU静态OOM审计\n\n状态：CPU VERIFIED，尚未启动GPU。模型{old['model']}@{old['revision']}，库固定TF4.57.6/PyTorch2.6.0+cu124。\n\n原图{prepared['0738']['original_image_wh']}，pixel_values={prepared['0738']['pixel_values_shape']}({prepared['0738']['pixel_values_dtype']})，grid=[1,304,214]。默认Processor得到65056个未合并Patch；空间merge2×2后16264个LM图像Token，总输入16307。输入四项Tensor哈希完全复现旧冻结布局。\n\nQwen3VLVisionModel.forward源码727–735行用每个图像/时间帧的H×W建立cu_seqlens，本图T=1，cu=[0,65056]。Qwen3VLVisionAttention.forward的224–244行按该序列切块，不是空间小窗口；当前实现未定义Window Attention。16头、head_dim72，eager_attention_forward第155行首先分配[1,16,65056,65056]，BF16={est['vision_scores_bf16_GiB']:.6f}GiB，匹配旧126.13GiB OOM。\n\n语言Prefill的32头Scores约{est['language_scores_bf16_GiB']:.3f}GiB，FP32 softmax单张量约{est['language_softmax_fp32_GiB']:.3f}GiB；加权重{est['weight_GiB']:.3f}GiB不能在24GB卡上使用Dense Eager。Native KV预填约{est['native_KV_prefill_GiB']:.3f}GiB，128上限约{est['native_KV_cap128_GiB']:.3f}GiB。原Physical引擎base/early/late三份4D BF16 Mask约{est['old_three_SVH_masks_GiB']:.3f}GiB；必须同时处理语言Attention，不是只替换Vision。\n\n本次实际选定冻结Original引擎a719b590没有Image K/V补零；不引入其他Zero-Key或Repaired变体。源码与旧预测的保护SHA见historical_protected.json。\n''',encoding='utf-8')
(O/'attention_backend_compatibility.md').write_text('''# Attention兼容性（CPU阶段）\n\nTF4.57.6 Qwen3VL模型声明支持SDPA与FlashAttention2，Vision/Text配置独立；Vision SDPA按cu_seqlens逐帧调用，Text支持非因果/因果和自定义4D additive mask。第三方flash_attn未安装，不新增依赖。\n\nPyTorch2.6现有SDPA包含EFFICIENT/FLASH/MATH/CUDNN。当前计划仅启用EFFICIENT，禁用其他三路，实际GPU内核待验证。由于HF在无Mask时自动enable_gqa，该优化不被efficient kernel支持，新外围适配按官方repeat_kv等价展开分组KV；自定义Mask仅对齐存储stride至8，保持逻辑尺寸、数值和Key集合不变。6项CPU FP64参考比较通过，不能代替CUDA验收。\n\n历史2048视觉Token/32生成Token的Vision-SDPA实验非本轮输入协议，且仅短输入运行、数值门禁失败；不复用为本轮预测、不改写旧失败。本轮保留跨Kernel差异作诊断，新的Native与Physical共用SDPA_EFFICIENT协议。\n''',encoding='utf-8')
atomic(O/'native_memory_gate.json',dict(status='NOT_RUN',phase='CPU_ONLY'));atomic(O/'svh_physical_gate.json',dict(status='NOT_RUN',phase='WAIT_NATIVE_0738'))
print('CPU_PHASE1_PASS',json.dumps(dict(inputs=prepared,memory=est,tests=len(tests),CUDA_initialized=torch.cuda.is_initialized())))
