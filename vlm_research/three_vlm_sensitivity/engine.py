"""Native official forward, sample-local visual replay, and existing P-1 hook."""
from contextlib import ExitStack
from unittest.mock import patch
import torch

from ..internvl3_smoke.engine import P1OutputHook, LayerTrace, hooks_count, tensor_hash, official_preprocessor
from .common import model_path, MODELS, digest


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(clone_tree(v) for v in value)
    raise TypeError(type(value))


def hash_tree(value):
    if isinstance(value, torch.Tensor):
        return tensor_hash(value)
    return [hash_tree(v) for v in value]


class Adapter:
    def __init__(self, name):
        import transformers
        assert transformers.__version__ == MODELS[name]['transformers']
        self.name, self.length = name, MODELS[name]['layers']
        if name == 'qwen':
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path(name), local_files_only=True)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path(name), local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='eager', device_map={'': 'cuda:0'}, use_safetensors=True).eval()
            self.layers = self.model.model.language_model.layers
            assert self.model.config.text_config._attn_implementation == 'eager'
            assert self.model.config.vision_config._attn_implementation == 'eager'
        else:
            from transformers import AutoModel, AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(model_path(name), local_files_only=True, use_fast=False)
            self.model = AutoModel.from_pretrained(model_path(name), local_files_only=True, trust_remote_code=True,
                torch_dtype=torch.bfloat16, use_flash_attn=False, device_map={'': 'cuda:0'}, use_safetensors=True).eval()
            self.layers = self.model.language_model.model.layers
            assert self.model.language_model.config._attn_implementation == 'eager'
            assert not any(layer.attn.use_flash_attn for layer in self.model.vision_model.encoder.layers)
            self.model.img_context_token_id = self.processor.convert_tokens_to_ids('<IMG_CONTEXT>')
            self.loader, _ = official_preprocessor(model_path(name))
        assert len(self.layers) == self.length
        assert {str(p.device) for p in self.model.parameters()} == {'cuda:0'}
        assert {p.dtype for p in self.model.parameters()} == {torch.bfloat16}
        self.hooks = hooks_count(self.model)
        self.versions = [(p._version, p.data_ptr()) for p in self.model.parameters()]

    def prepare(self, sample, frozen):
        self.sample, self.frozen = sample, frozen
        self.visual = None
        if self.name == 'qwen':
            from ..qwen3vl_early_windows.experiment import prepare
            inputs, layout = prepare(self.processor, sample)
            assert layout == frozen['layout'] and digest(layout) == frozen['input_binding']
            self.inputs = inputs.to('cuda:0')
        else:
            pixels = self.loader(sample['image_path'], max_num=1).to('cuda:0', dtype=torch.bfloat16)
            assert pixels.shape[0] == 1
            self.inputs = dict(pixel_values=pixels,
                input_ids=torch.tensor([frozen['input_ids']], device='cuda:0'),
                attention_mask=torch.ones((1, len(frozen['input_ids'])), device='cuda:0', dtype=torch.long))
        self.input_hashes = {k: tensor_hash(v) for k, v in self.inputs.items()}

    @torch.inference_mode()
    def forward(self, layer=None, noop=False, native=False, trace=False):
        from transformers.cache_utils import DynamicCache
        n = self.frozen['layout']['input_length']
        with ExitStack() as stack:
            hook = stack.enter_context(P1OutputHook(self.layers[layer].self_attn.o_proj, n - 1, noop)) if layer is not None else None
            tracer = stack.enter_context(LayerTrace(self.layers, n - 1)) if trace else None
            if self.name == 'qwen':
                original = self.model.model.get_image_features
                def visual(*args, **kwargs):
                    if native or self.visual is None:
                        result = original(*args, **kwargs)
                        self.visual = clone_tree(result)
                        return result
                    return clone_tree(self.visual)
                stack.enter_context(patch.object(self.model.model, 'get_image_features', visual))
                output = self.model(**self.inputs, use_cache=True, past_key_values=None,
                    cache_position=torch.arange(n, device='cuda:0'), logits_to_keep=1,
                    output_attentions=False, output_hidden_states=False)
            else:
                original = self.model.extract_feature
                def visual(*args, **kwargs):
                    if native or self.visual is None:
                        result = original(*args, **kwargs)
                        self.visual = result.detach().clone()
                        return result
                    return self.visual.detach().clone()
                stack.enter_context(patch.object(self.model, 'extract_feature', visual))
                def only_prefill(**kwargs):
                    return self.model.language_model(inputs_embeds=kwargs['inputs_embeds'],
                        attention_mask=kwargs['attention_mask'], use_cache=True, past_key_values=DynamicCache(),
                        output_attentions=False, output_hidden_states=False, return_dict=True)
                # Use the official visual insertion path, stopping at its initial LM prefill.
                stack.enter_context(patch.object(self.model.language_model, 'generate', only_prefill))
                output = self.model.generate(**self.inputs)
            assert output.logits.shape[0] == 1
            logits = output.logits[0, -1].detach().cpu().clone()
            assert torch.isfinite(logits).all()
            assert all(output.past_key_values.get_seq_length(i) == n for i in range(self.length))
            if hook:
                assert hook.calls == 1
            if tracer:
                tracer.validate()
                records = tracer.records
            else:
                records = None
        assert hooks_count(self.model) == self.hooks
        return logits, records

    def verify(self):
        assert hooks_count(self.model) == self.hooks
        assert self.versions == [(p._version, p.data_ptr()) for p in self.model.parameters()]
        assert self.input_hashes == {k: tensor_hash(v) for k, v in self.inputs.items()}

    def clear(self):
        self.inputs, self.visual = None, None
