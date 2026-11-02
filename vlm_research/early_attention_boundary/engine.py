"""Native generation with reused output hooks and read-only query/cache audits."""
from contextlib import ExitStack
from unittest.mock import patch
import time
import torch

from ..three_vlm_sensitivity.engine import Adapter, hash_tree
from ..internvl3_smoke.engine import LayerTrace, tensor_hash, hooks_count
from ..qwen3vl_early_windows.experiment import AttentionBypass, gqa_prediction


class GenerationAdapter(Adapter):
    @torch.inference_mode()
    def generate(self, targets=(), noop=False, audit=False):
        n = self.frozen['layout']['input_length']
        owner = self.model if self.name == 'qwen' else self.model.language_model
        calls, first, ids_capture, visual_hashes, traces = [], [], [], {}, []
        mlp_counts = [0] * self.length
        tracer = None
        handles = []
        baseline = hooks_count(self.model)
        start = time.monotonic()
        torch.cuda.synchronize()

        def before(module, args, kwargs):
            nonlocal tracer
            value = kwargs.get('inputs_embeds')
            value = value if value is not None else kwargs['input_ids']
            q = value.shape[1]
            assert q == (1 if calls else n)
            cache = kwargs.get('past_key_values')
            expected = n + len(calls) - 1 if calls else 0
            assert cache is None or cache.get_seq_length() == expected
            assert not calls or cache is not None
            cp = kwargs.get('cache_position')
            assert cp is not None and cp.tolist() == list(range(expected, expected + q))
            if audit and not calls:
                tracer = LayerTrace(self.layers, n - 1).__enter__()

        def after(module, args, kwargs, output):
            nonlocal tracer
            assert torch.isfinite(output.logits).all()
            cache = output.past_key_values
            expected = n + len(calls)
            assert all(cache.get_seq_length(i) == expected for i in range(self.length))
            logits = output.logits[0, -1].detach().cpu().clone()
            if not calls:
                first.append(logits)
                if tracer:
                    tracer.validate()
                    traces.extend(tracer.records)
                    tracer.__exit__(None, None, None)
                    tracer = None
            calls.append(dict(query_length=n if not calls else 1, cache_length=expected,
                              top1=int(logits.argmax())))

        try:
            with ExitStack() as stack:
                if targets:
                    if self.name == 'qwen':
                        hook = stack.enter_context(AttentionBypass(self.model, targets, n, noop))
                    else:
                        from ..internvl3_stage.engine import WindowHook
                        hook = stack.enter_context(WindowHook(self.layers, targets, n, noop))
                else:
                    hook = None
                handles.extend([owner.register_forward_pre_hook(before, with_kwargs=True),
                                owner.register_forward_hook(after, with_kwargs=True)])
                if audit:
                    for index, layer in enumerate(self.layers):
                        def mlp(module, args, output, index=index):
                            mlp_counts[index] += 1
                            assert torch.isfinite(output).all()
                        handles.append(layer.mlp.register_forward_hook(mlp))
                    modules = ({f'deepstack_{i}': m for i, m in enumerate(self.model.model.visual.deepstack_merger_list)}
                               if self.name == 'qwen' else {'vision': self.model.vision_model, 'mlp1': self.model.mlp1})
                    for name, module in modules.items():
                        def feature(module, args, output, name=name):
                            # Vision encoder returns a ModelOutput; its first item is the actual hidden state.
                            if not isinstance(output, (torch.Tensor, tuple, list)):
                                output = output[0]
                            visual_hashes[name] = hash_tree(output)
                        handles.append(module.register_forward_hook(feature))
                if self.name == 'qwen':
                    generated = self.model.generate(**self.inputs, do_sample=False, num_beams=1,
                        max_new_tokens=32, use_cache=True, output_attentions=False, output_hidden_states=False)
                    ids = generated[0, n:].tolist()
                    text = self.processor.decode(ids, skip_special_tokens=True).strip()
                    eos = self.model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else eos
                else:
                    original = self.model.generate
                    def capture(*args, **kwargs):
                        assert torch.equal(kwargs['input_ids'], self.inputs['input_ids'])
                        assert torch.equal(kwargs['attention_mask'], self.inputs['attention_mask'])
                        assert tensor_hash(kwargs['pixel_values']) == self.input_hashes['pixel_values']
                        result = original(*args, **kwargs)
                        ids_capture.extend(result[0].tolist())
                        return result
                    stack.enter_context(patch.object(self.model, 'generate', capture))
                    text = self.model.chat(self.processor, self.inputs['pixel_values'], self.frozen['request'],
                        dict(do_sample=False, num_beams=1, max_new_tokens=32)).strip()
                    ids = ids_capture
                    eos = [self.processor.convert_tokens_to_ids(self.model.conv_template.sep.strip())]
                torch.cuda.synchronize()
                assert ids and len(ids) == len(calls)
                assert ids == [c['top1'] for c in calls], 'Native greedy argmax/token mismatch'
                if hook and self.name == 'qwen':
                    assert len(hook.events) == len(targets) * len(ids)
                    assert len(hook.calls) == len(ids)
                elif hook:
                    assert all(hook.calls[i] == [n] + [1]*(len(ids)-1) for i in targets)
                if audit:
                    assert mlp_counts == [len(ids)] * self.length
                    assert len(traces) == self.length
                    assert len(visual_hashes) == (3 if self.name == 'qwen' else 2)
                    if not noop:
                        assert all(torch.count_nonzero(traces[i]['attention']) == 0 for i in targets)
                result = dict(text=text, prediction=gqa_prediction(text), token_ids=ids,
                    correct=gqa_prediction(text) == self.sample['answer'], first_token=ids[0],
                    eos=ids[-1] in eos, truncated=len(ids) == 32 and ids[-1] not in eos,
                    empty=not bool(text), non_eos_tokens=sum(i not in eos for i in ids),
                    length=len(ids), decode_steps=len(ids)-1, anomaly=False,
                    seconds=time.monotonic()-start, cache_trace=calls,
                    input_binding=self.frozen['input_binding'],
                    visual_hashes=visual_hashes, mlp_calls=mlp_counts if audit else [],
                    prefill_prefix_hashes=[{k: v for k, v in r.items() if k.endswith('_sha256')} for r in traces])
        finally:
            if tracer:
                tracer.__exit__(None, None, None)
            for handle in handles:
                handle.remove()
        assert hooks_count(self.model) == baseline
        self.verify()
        return result, first[0]
