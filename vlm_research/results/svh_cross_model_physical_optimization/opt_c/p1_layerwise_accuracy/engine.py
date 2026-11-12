"""A single prefill row edit, around otherwise unmodified HF cached generation."""
from contextlib import AbstractContextManager, contextmanager
import sys
from unittest.mock import patch

import torch

from .common import OLD, MODELS, model_path


def tensor_hash(value):
    import hashlib
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def footprint(model):
    return [(id(m),len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]


class PrefillP1SingleLayerBypass(AbstractContextManager):
    def __init__(self, model, layers, length, selected, noop=False):
        if selected is not None and (type(selected) is not int or not 0 <= selected < len(layers)):
            raise ValueError('Invalid single layer')
        self.model,self.layers,self.length,self.selected,self.noop = model,layers,length,selected,noop
        self.handles,self.calls,self.events = [],[],[]

    def begin(self, module, args, kwargs):
        ids = kwargs.get('inputs_embeds')
        if ids is None:
            ids = kwargs['input_ids']
        t = len(self.calls)
        n = ids.shape[1]
        assert ids.shape[0] == 1 and n == (1 if t else self.length)
        cache = kwargs.get('past_key_values')
        past = cache.get_seq_length() if cache is not None and hasattr(cache,'get_seq_length') else 0
        assert past == (self.length+t-1 if t else 0)
        cp = kwargs.get('cache_position')
        if cp is not None:
            assert cp.tolist() == ([self.length+t-1] if t else list(range(self.length)))
        self.calls.append(dict(step=t,query_length=n,past=past))

    def edit(self, module, args, output):
        t = len(self.calls)-1
        assert t >= 0 and output.shape[:2] == (1,self.calls[-1]['query_length'])
        active = t == 0 and not self.noop
        changed = output
        if active:
            changed = output.clone()
            changed[:,self.length-1] = 0
            assert torch.count_nonzero(changed[:,self.length-1]) == 0
            assert torch.equal(changed[:,:self.length-1],output[:,:self.length-1])
        else:
            assert changed is output
        self.events.append(dict(layer=self.selected,step=t,active=active,
            query=self.length-1 if t==0 else 0,non_target_exact=True,decode_native=t>0))
        return changed

    def __enter__(self):
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.begin,with_kwargs=True))
            if self.selected is not None:
                self.handles.append(self.layers[self.selected].self_attn.o_proj.register_forward_hook(self.edit))
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self,*exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class Adapter:
    def __init__(self,kind):
        import transformers
        assert transformers.__version__ == MODELS[kind]['transformers']
        self.kind = kind
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        if kind == 'llava':
            sys.path.insert(0,str(OLD))
            from vlm_common import load_model
            from knockout_probe import model_layers
            self.model,self.processor,self.environment = load_model(str(model_path(kind)))
            self.layers = model_layers(self.model)
        else:
            from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path(kind),local_files_only=True)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path(kind),local_files_only=True,
                dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'},use_safetensors=True).eval()
            self.layers = self.model.model.language_model.layers
            self.environment = dict(torch=torch.__version__,transformers=transformers.__version__)
        assert len(self.layers) == MODELS[kind]['layers'] and not self.model.training
        self.hooks = footprint(self.model)
        self.versions = [(p._version,p.data_ptr()) for p in self.model.parameters()]

    def prepare(self,sample,frozen):
        if self.kind == 'llava':
            from vlm_common import prepare_inputs
            self.inputs,_ = prepare_inputs(self.model,self.processor,sample['image_path'],sample['prompt_text'])
            assert self.inputs['input_ids'][0].tolist() == sample['input_mapping']['prompt_ids']
            assert int((self.inputs['input_ids']==32000).sum()) == 576
        else:
            from ..qwen3vl_early_windows.experiment import prepare
            inputs,layout = prepare(self.processor,sample)
            assert layout == frozen['layout']
            self.inputs = inputs.to('cuda:0')
        self.input_hashes = {k:tensor_hash(v) for k,v in self.inputs.items()}
        assert self.inputs['attention_mask'].bool().all()
        self.visual,self.visual_hashes,self.visual_native_calls = None,None,0

    @contextmanager
    def visual_replay(self,enabled=True):
        target = self.model if self.kind=='llava' else self.model.model
        original = target.get_image_features
        def clone(value):
            if isinstance(value,torch.Tensor):
                return value.detach().clone()
            return type(value)(clone(v) for v in value)
        def hashes(value):
            return tensor_hash(value) if isinstance(value,torch.Tensor) else [hashes(v) for v in value]
        def visual(*args,**kwargs):
            if self.visual is None or not enabled:
                result = original(*args,**kwargs)
                self.visual_native_calls += 1
                value_hashes = hashes(result)
                if self.visual_hashes is not None:
                    assert self.visual_hashes == value_hashes
                self.visual,self.visual_hashes = clone(result),value_hashes
                return result
            assert hashes(self.visual) == self.visual_hashes
            return clone(self.visual)
        with patch.object(target,'get_image_features',visual):
            yield

    def verify(self):
        assert footprint(self.model) == self.hooks
        assert self.versions == [(p._version,p.data_ptr()) for p in self.model.parameters()]
        assert self.input_hashes == {k:tensor_hash(v) for k,v in self.inputs.items()}


@torch.inference_mode()
def generate(adapter,selected=None,noop=False,capture=False,native=False):
    m,inputs,layers = adapter.model,adapter.inputs,adapter.layers
    n = inputs['input_ids'].shape[1]
    before = footprint(m)
    scores,valid_logits,handles,cache_refs,mlp,positions = [],[],[],[],[0]*len(layers),[]
    def observe(module,args,kwargs,out):
        t = len(scores)
        assert out.logits.shape[:2] == (1,1 if t or adapter.kind=='qwen' else n)
        assert torch.isfinite(out.logits).all()
        cache = out.past_key_values
        assert all(cache.get_seq_length(i)==n+t for i in range(len(layers)))
        if t==0:
            cache_refs.append(cache)
        scores.append(out.logits[0,-1].detach().float().cpu().clone())
        if capture:
            valid_logits.append(out.logits.detach().cpu().clone())
    control = PrefillP1SingleLayerBypass(m,layers,n,None if native else selected,noop)
    try:
        handles.append(m.register_forward_hook(observe,with_kwargs=True))
        if adapter.kind=='qwen':
            def mrope(module,args,kwargs):
                pos,cp=kwargs['position_ids'],kwargs['cache_position']
                t=len(positions)
                assert cp.tolist()==([n+t-1] if t else list(range(n)))
                assert pos.shape==(3,1,1 if t else n)
                positions.append(dict(step=t,shape=list(pos.shape),position_sha256=tensor_hash(pos),
                    cache_position_sha256=tensor_hash(cp)))
            handles.append(m.model.language_model.register_forward_pre_hook(mrope,with_kwargs=True))
        if capture:
            for i,block in enumerate(layers):
                def count(module,args,out,i=i):
                    mlp[i] += 1
                handles.append(block.mlp.register_forward_hook(count))
        with control:
            result = m.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,
                logits_to_keep=MODELS[adapter.kind]['logits_to_keep'],return_dict_in_generate=True,
                output_scores=True,output_attentions=False,output_hidden_states=False)
        ids = result.sequences[0,n:].tolist()
        assert len(ids)==len(scores)==len(result.scores) and ids
        assert ids == [int(s.argmax()) for s in scores]
        assert all(result.past_key_values.get_seq_length(i)==n+len(ids)-1 for i in range(len(layers)))
        if selected is not None and not native:
            assert len(control.events)==len(ids)
            assert sum(e['active'] for e in control.events)==(0 if noop else 1)
            assert all(not e['active'] for e in control.events[1:])
        eos = m.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else eos
        text = adapter.processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
        g = dict(token_ids=ids,text=text,prediction=text.rstrip('.').lower(),condition_layer=selected,
            eos=ids[-1] in eos,truncated=len(ids)==32 and ids[-1] not in eos,empty=not bool(text),
            repeated=any(ids[i:i+3]==ids[j:j+3] for i in range(len(ids)-2) for j in range(i+3,len(ids)-2)),
            first_token_id=ids[0],first_logits_sha256=tensor_hash(scores[0]),
            all_step_logits_sha256=[tensor_hash(s) for s in scores],
            intervention_events=control.events,decode_all_native=True,cache_initial_empty=True,
            cache_lengths=[result.past_key_values.get_seq_length(i) for i in range(len(layers))],
            logits_to_keep=MODELS[adapter.kind]['logits_to_keep'],all_logits_finite=True)
        if adapter.kind=='qwen':
            assert len(positions)==len(ids)
            g['native_mrope_positions']=positions
        if capture:
            assert mlp == [len(ids)]*len(layers)
        return g,scores,valid_logits,result.past_key_values if capture else None
    finally:
        for handle in handles:
            handle.remove()
        assert footprint(m)==before and not control.handles
