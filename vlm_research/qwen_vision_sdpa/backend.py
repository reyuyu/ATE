"""Official backend selection with read-only, untimed CUDA operator auditing."""
import collections
from contextlib import contextmanager,ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.nn.attention import sdpa_kernel,SDPBackend
from .common import *


def load():
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    processor=AutoProcessor.from_pretrained(MODELS['qwen'],local_files_only=True)
    processor.image_processor.size={**dict(processor.image_processor.size),'longest_edge':2048*32*32}
    assert processor.image_processor.to_dict()==read(PREVIOUS/'qwen_processor_2048.json')['capped']
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODELS['qwen'],local_files_only=True,
        dtype=torch.bfloat16,attn_implementation=BACKENDS,
        device_map={'':'cuda:0'},use_safetensors=True,low_cpu_mem_usage=True).eval()
    adapter=SimpleNamespace(model=model,processor=processor,layers=model.model.language_model.layers,
        vision=model.model.visual,kind='qwen')
    assert_backends(adapter)
    return adapter


def assert_backends(adapter,vision='sdpa'):
    assert len(adapter.layers)==36 and len(adapter.vision.blocks)==27
    assert all(b.attn.config._attn_implementation==vision for b in adapter.vision.blocks)
    assert all(b.self_attn.config._attn_implementation=='eager' for b in adapter.layers)


@contextmanager
def vision_backend(adapter,value):
    """Use HF's official live setter; no weights or forward implementations change."""
    adapter.model.set_attn_implementation({'vision_config':value,'text_config':'eager'})
    assert_backends(adapter,value)
    try:yield
    finally:
        adapter.model.set_attn_implementation(BACKENDS)
        assert_backends(adapter)


@contextmanager
def efficient_sdpa_only():
    # This official context excludes MATH and CUDNN. No alternative kernel is
    # searched or timed; PyTorch selects from its two supported efficient paths.
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION],set_priority=True):
        assert not torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        yield


class _OperatorTrace(TorchDispatchMode):
    def __init__(self,observer):super().__init__();self.observer=observer
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        name=str(func)
        if 'scaled_dot_product' in name:
            allowed=('aten._scaled_dot_product_flash_attention.default',
                     'aten._scaled_dot_product_efficient_attention.default')
            assert name in allowed,f'Unapproved SDPA dispatch: {name}'
            self.observer.operator_events.append(dict(operator=name,layer=self.observer.current_vision_layer,
                query=list(args[0].shape),dtype=str(args[0].dtype)))
        return func(*args,**(kwargs or {}))


class BackendAudit:
    """Observe operators and layouts; never record durations or retain GPU arrays."""
    def __init__(self,adapter):
        self.adapter=adapter;self.current_vision_layer=None;self.handles=[]
        self.phase=None;self.operator_events=[];self.vision_calls=[];self.text_shapes=collections.Counter()
        self.features=None;self.feature_hashes=False;self.feature_capture=False;self.expected_vision='sdpa'

    def begin(self,label,feature_hashes=False,feature_capture=False,vision='sdpa'):
        assert self.phase is None
        assert_backends(self.adapter,vision)
        self.phase=label;self.operator_events=[];self.vision_calls=[];self.text_shapes=collections.Counter()
        self.features=None;self.feature_hashes=feature_hashes;self.feature_capture=feature_capture
        self.expected_vision=vision

    def finish(self):
        assert self.phase is not None and self.current_vision_layer is None
        assert self.vision_calls,'No Vision forward observed'
        if self.expected_vision=='sdpa':
            assert len(self.operator_events)==27*len(self.vision_calls)
            for i in range(27):assert sum(e['layer']==i for e in self.operator_events)==len(self.vision_calls)
        else:assert not self.operator_events
        result=dict(phase=self.phase,vision_backend=self.expected_vision,text_backend='eager',
            efficient_cuda_operators=dict(collections.Counter(e['operator'] for e in self.operator_events)),
            vision_layer_operators=self.operator_events,vision_forwards=self.vision_calls,
            text_input_shapes=[dict(layer=k[0],shape=list(k[1]),count=v) for k,v in sorted(self.text_shapes.items())],
            text_sdpa_calls=0,math_sdpa_enabled=torch.backends.cuda.math_sdp_enabled(),
            cudnn_sdpa_enabled=torch.backends.cuda.cudnn_sdp_enabled(),timing_or_flops_recorded=False)
        features=self.features;self.phase=None;self.features=None
        return result,features

    def __enter__(self):
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
        self.stack=ExitStack();self.before=hook_counts(self.adapter.model)
        original=torch.nn.functional.scaled_dot_product_attention
        def observed_sdpa(*args,**kwargs):
            assert self.phase is not None and self.current_vision_layer is not None,'Text or unknown module called SDPA'
            assert self.expected_vision=='sdpa'
            before=len(self.operator_events)
            with _OperatorTrace(self):out=original(*args,**kwargs)
            assert len(self.operator_events)==before+1,'Actual efficient SDPA operator not confirmed'
            return out
        self.stack.enter_context(patch.object(torch.nn.functional,'scaled_dot_product_attention',observed_sdpa))
        eager=qm.eager_attention_forward
        def observed_eager(module,q,k,v,*args,**kwargs):
            self.adapter.last_attention_shape=dict(module=type(module).__name__,layer=getattr(module,'layer_idx',None),
                query=list(q.shape),key=list(k.shape),value=list(v.shape),dtype=str(q.dtype))
            if isinstance(module,qm.Qwen3VLVisionAttention):assert self.expected_vision=='eager','Vision fell back to Eager'
            else:assert module.config._attn_implementation=='eager'
            return eager(module,q,k,v,*args,**kwargs)
        self.stack.enter_context(patch.object(qm,'eager_attention_forward',observed_eager))
        for i,block in enumerate(self.adapter.vision.blocks):
            def start(m,a,k,i=i):
                assert self.current_vision_layer is None and self.phase is not None
                self.current_vision_layer=i
                h=a[0] if a else k['hidden_states']
                self.adapter.last_attention_shape=dict(module=type(m).__name__,layer=i,
                    hidden=list(h.shape),backend=m.config._attn_implementation)
            def end(m,a,o,i=i):
                assert self.current_vision_layer==i
                assert bool(torch.isfinite(o).all()),'Non-finite Vision Attention output'
                self.current_vision_layer=None
            self.handles += [block.attn.register_forward_pre_hook(start,with_kwargs=True),
                             block.attn.register_forward_hook(end)]
        for i,layer in enumerate(self.adapter.layers):
            def text(m,a,k,i=i):
                h=a[0] if a else k['hidden_states']
                assert m.config._attn_implementation=='eager'
                self.text_shapes[(i,tuple(h.shape))]+=1
            self.handles.append(layer.self_attn.register_forward_pre_hook(text,with_kwargs=True))
        def visual_start(m,args,kwargs):
            assert self.phase is not None
            h=args[0] if args else kwargs['hidden_states']
            grid=args[1] if len(args)>1 else kwargs['grid_thw']
            self.active_grid=grid.detach().cpu().tolist()
            self.active_patches=int(grid.prod(-1).sum())
            self.active_pixel_shape=list(h.shape)
        def visual_end(m,args,output):
            pooled,deep=output
            assert len(deep)==3
            expected=self.active_patches//4
            assert 0<expected<=2048 and pooled.shape[0]==expected
            tensors=[pooled]+list(deep)
            assert all(t.shape==pooled.shape for t in deep)
            assert all(bool(torch.isfinite(t).all()) for t in tensors),'Non-finite Vision features'
            item=dict(vit_patches=self.active_patches,language_image_tokens=expected,
                grid_thw=self.active_grid,pixel_shape=self.active_pixel_shape,
                output_shapes=[list(t.shape) for t in tensors],deepstack_features=3,
                deepstack_indexes=list(m.deepstack_visual_indexes),all_finite=True)
            if self.feature_hashes:item['feature_sha256']=[tensor_sha(t) for t in tensors]
            if self.feature_capture:self.features=[t.detach().cpu().clone() for t in tensors]
            self.vision_calls.append(item)
        self.handles += [self.adapter.vision.register_forward_pre_hook(visual_start,with_kwargs=True),
                         self.adapter.vision.register_forward_hook(visual_end)]
        return self

    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.stack.close()
        assert hook_counts(self.adapter.model)==self.before
