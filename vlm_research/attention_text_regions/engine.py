from contextlib import AbstractContextManager

import torch

from knockout_probe import layout_from_inputs, model_layers, tensor_hash, ROW_ATOL
from paper_rq1.engine import step_inputs

REGIONS = {'Early': list(range(1, 11)), 'Middle': list(range(11, 21)), 'Late': list(range(21, 32))}
CONDITIONS = ['Full']+[f'{region} {kind} OFF' for region in REGIONS for kind in ['Attention', 'Text']]


class Intervention(AbstractContextManager):
    def __init__(self, model, inputs, condition, step, prefill_active, noop=False,
                 audit=False, expected_visual=576):
        if condition not in CONDITIONS or step < 0:
            raise ValueError('Unknown frozen condition or negative step')
        self.layers = model_layers(model)
        assert len(self.layers) == 32
        layout = layout_from_inputs(inputs, model.config.image_token_index, expected_visual)
        assert bool(inputs['attention_mask'].all()), 'Only an unpadded batch1 prompt is supported'
        self.n = layout['sequence_length']; self.step = step
        self.query = self.n-1 if step == 0 else 0
        self.absolute = self.n+step-1; self.k = self.n+step
        self.images = layout['visual_indices']
        self.bos = torch.where(inputs['input_ids'][0] == 1)[0]
        assert len(self.bos) == 1
        self.text = torch.ones(self.k, dtype=torch.bool, device=inputs['input_ids'].device)
        self.text[self.images] = False; self.text[self.bos] = False; self.text[self.absolute] = False
        assert self.text.any() and not self.text[self.images].any()
        self.kind = 'Full' if condition == 'Full' else condition.split()[1]
        region = None if condition == 'Full' else condition.split()[0]
        self.active = set(REGIONS[region]) if region and (step > 0 or prefill_active) and not noop else set()
        assert 0 not in self.active
        self.handles, self.rows, self.projections, self.states, self.calls = [], [], [], [], []
        self.audit = audit

    def __enter__(self):
        try:
            for index, block in enumerate(self.layers):
                self.handles.append(block.self_attn.register_forward_pre_hook(self.pre(index), with_kwargs=True))
                self.handles.append(block.self_attn.o_proj.register_forward_hook(self.project(index)))
                self.handles.append(block.self_attn.register_forward_hook(self.observe(index)))
                if self.audit:
                    def state(module, args, output, index=index):
                        hidden = output[0]
                        assert torch.isfinite(hidden).all()
                        self.states.append(dict(layer=index, other_queries=tensor_hash(hidden[:, :self.query])))
                    self.handles.append(block.register_forward_hook(state))
            return self
        except BaseException:
            self.__exit__(None, None, None); raise

    def pre(self, index):
        def hook(module, args, kwargs):
            hidden = kwargs['hidden_states']; cache = kwargs.get('past_key_value')
            qlen = self.n if self.step == 0 else 1
            assert hidden.shape[:2] == (1, qlen) and kwargs['use_cache']
            pos = kwargs['cache_position']
            expected = torch.arange(self.n, device=hidden.device) if self.step == 0 else torch.tensor([self.absolute], device=hidden.device)
            assert torch.equal(pos, expected)
            assert cache is not None and cache.get_seq_length(index) == (0 if self.step == 0 else self.absolute)
            old = kwargs['attention_mask']
            assert old is not None and old.shape == (1, 1, qlen, self.k)
            assert bool((old[0, 0, self.query] == 0).all()), 'Current query must retain native causal access'
            new = old
            if index in self.active and self.kind == 'Text':
                new = old.clone(); new[0, 0, self.query, self.text] = -torch.inf
                assert torch.equal(new[0, 0, self.query, ~self.text], old[0, 0, self.query, ~self.text])
                assert bool((new[0, 0, self.query] == 0).any())
                if self.audit:
                    assert torch.equal(new[:, :, :self.query], old[:, :, :self.query])
            self.calls.append(index)
            return args, dict(kwargs, attention_mask=new)
        return hook

    def project(self, index):
        def hook(module, args, output):
            assert torch.isfinite(output).all()
            touched = index in self.active and self.kind == 'Attention'
            changed = output
            if touched:
                changed = output.clone(); changed[0, self.query] = 0
                assert not torch.count_nonzero(changed[0, self.query])
                if self.audit: assert torch.equal(changed[:, :self.query], output[:, :self.query])
            self.projections.append(dict(layer=index, applied=touched,
                output_norm=float(changed[0, self.query].float().norm()),
                native_output_norm=float(output[0, self.query].float().norm())))
            return changed
        return hook

    def observe(self, index):
        def hook(module, args, output):
            weights = output[1]
            assert weights is not None and weights.shape[2:] == (self.n if self.step == 0 else 1, self.k)
            row = weights[0, :, self.query].float()
            assert torch.isfinite(row).all() and bool((row >= 0).all())
            error = float((row.sum(-1)-1).abs().max()); assert error <= ROW_ATOL
            maximum = float(row[:, self.text].max())
            touched = index in self.active and self.kind == 'Text'
            assert maximum == 0. if touched else maximum > 0.
            image_mass = float(row[:, self.images].sum(-1).mean()); assert image_mass > 0.
            r = dict(layer=index, step=self.step, absolute_query=self.absolute,
                text_masked=touched, text_probability_max=maximum,
                text_mass=float(row[:, self.text].sum(-1).mean()), image_mass=image_mass,
                bos_mass=float(row[:, self.bos].sum(-1).mean()), self_mass=float(row[:, self.absolute].mean()),
                generated_history_keys=max(0, self.step-1), row_error=error)
            if self.audit and self.step == 0:
                r['other_queries'] = tensor_hash(weights[:, :, :self.query])
            self.rows.append(r)
            return (output[0], None)+tuple(output[2:])
        return hook

    def __exit__(self, *exc):
        for handle in self.handles: handle.remove()
        self.handles.clear()


@torch.inference_mode()
def generate(model, inputs, condition, prefill_active, limit, noop=False, audit=False,
             collect=False, expected_visual=576):
    assert limit > 0
    tokens, cache, traces, rows, projects, debug = [], None, [], [], [], {}
    n = inputs['input_ids'].shape[1]
    while len(tokens) < limit and (not tokens or tokens[-1] != 2):
        step = len(tokens)
        control = Intervention(model, inputs, condition, step, prefill_active, noop, audit, expected_visual)
        with control:
            output = model(**step_inputs(inputs, tokens), past_key_values=cache, use_cache=True,
                           output_attentions=True, return_dict=True)
        assert not control.handles and control.calls == list(range(32))
        assert [r['layer'] for r in control.rows] == list(range(32))
        assert [r['layer'] for r in control.projections] == list(range(32))
        cache = output.past_key_values
        assert all(cache.get_seq_length(k) == n+step for k in range(32))
        logits = output.logits[0, -1].detach().float().cpu().clone()
        assert torch.isfinite(logits).all()
        rows.extend(control.rows); projects.extend(dict(step=step, **r) for r in control.projections)
        if collect: traces.append(logits)
        if step == 0:
            first = logits
            if audit:
                debug = dict(prefill_logits=output.logits.detach().float().cpu().clone(),
                             states=control.states, other_query_attention=[r['other_queries'] for r in control.rows])
        tokens.append(int(logits.argmax()))
    return dict(token_ids=tokens, eos=tokens[-1] == 2,
                truncated=len(tokens) == limit and tokens[-1] != 2,
                decode_steps=len(tokens)-1, prefill_active=prefill_active,
                own_cache=True, own_prefix=True), first, rows, projects, traces, debug
