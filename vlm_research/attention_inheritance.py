"""Instance-local LLaMA eager post-softmax intervention; no library/global edits."""
from contextlib import AbstractContextManager
import types
import torch
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv

TOLERANCES = dict(row_sum_atol=0.002, visual_mass_atol=2e-5,
                  visual_mass_rtol=0.001, conditional_l1_nominal=0.002, rounding_bound_slack=2e-6)


class VisualAttentionInheritance(AbstractContextManager):
    """Mix frozen Full per-head visual distributions into ONE query, selected layers.

    Non-target invariance is relative to the same forward's local pre-edit tensor,
    not to Full (earlier interventions can change later computed attention).
    """
    def __init__(self, modules, visual_indices, query_position, reference, strength,
                 active_layers=range(24, 32)):
        self.modules = list(modules)
        self.visual = torch.as_tensor(visual_indices, dtype=torch.long)
        self.query = int(query_position)
        self.reference = torch.as_tensor(reference, dtype=torch.float32)
        self.strength = float(strength)
        self.active = list(active_layers)
        if not 0 <= self.strength <= 1 or not self.active or len(set(self.active)) != len(self.active):
            raise ValueError('Invalid strength/layer selection')
        if min(self.active) < 0 or max(self.active) >= len(self.modules):
            raise ValueError('Invalid active layer')
        if self.visual.ndim != 1 or not self.visual.numel() or len(self.visual.unique()) != len(self.visual):
            raise ValueError('Visual indices must be nonempty and unique')
        if self.visual.min() < 0 or self.visual.max() > self.query:
            raise ValueError('Visual keys must be causally accessible')
        if self.reference.ndim != 2 or self.reference.shape[-1] != self.visual.numel():
            raise ValueError('Reference must be [heads, visual_keys]')
        if not torch.isfinite(self.reference).all() or (self.reference < 0).any() or (self.reference.sum(-1) <= 0).any():
            raise ValueError('Invalid frozen reference')
        self.reference = self.reference / self.reference.sum(-1, keepdim=True)
        self.saved = []
        self.records = []
        self.rows = []

    def replace(self, weights, layer):
        if weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[1] != self.reference.shape[0]:
            raise AssertionError('Expected batch1 and aligned reference heads')
        if not 0 <= self.query < weights.shape[-2] or self.visual.max() >= weights.shape[-1]:
            raise AssertionError('Query/key out of range')
        visual = self.visual.to(weights.device)
        before = weights.clone()  # transient audit clone, never saved as a full matrix
        row = before[0, :, self.query, :]
        old = row.index_select(-1, visual).float()
        mass = old.sum(-1, keepdim=True)
        reference = self.reference.to(weights.device)
        mixture = (1-self.strength) * old / mass.clamp_min(1e-30) + self.strength * reference
        exact_target = mass * mixture
        rounded_target = exact_target.to(weights.dtype)
        if self.strength != 0:
            weights[0, :, self.query, visual] = rounded_target
        newrow = weights[0, :, self.query, :]
        new = newrow.index_select(-1, visual).float()
        newmass = new.sum(-1, keepdim=True)
        nonvisual = torch.ones(weights.shape[-1], dtype=torch.bool, device=weights.device)
        nonvisual[visual] = False
        other_queries = (torch.equal(weights[:, :, :self.query], before[:, :, :self.query]) and
                         torch.equal(weights[:, :, self.query+1:], before[:, :, self.query+1:]))
        nonvisual_exact = torch.equal(newrow[:, nonvisual], row[:, nonvisual])
        row_error = float((newrow.float().sum(-1)-1).abs().max())
        mass_error = float((newmass-mass).abs().max())
        mass_relative = float(((newmass-mass).abs()/mass.clamp_min(1e-30)).max())
        conditional_errors = ((new/newmass.clamp_min(1e-30)-mixture).abs().sum(-1))
        conditional_error = float(conditional_errors[mass[:, 0] > 0].max()) if (mass > 0).any() else 0.
        # Round-to-nearest: |round(x)-x| <= u*|x| + half_min_subnormal.
        # E bounds the row's L1 rounding error. Conditioning amplifies it:
        # ||round(x)/sum(round(x)) - x/sum(x)||1 <= 2E/sum(round(x)).
        # This derives from dtype/row mass, never from output margins or observed error.
        info = torch.finfo(weights.dtype)
        rounding_E = info.eps/2 * exact_target.abs().sum(-1) + len(visual)*info.tiny*info.eps/2
        bounds = 2*rounding_E/newmass[:, 0].clamp_min(1e-30) + TOLERANCES['rounding_bound_slack']
        rounding_pass = bool((conditional_errors <= bounds).all())
        formula_exact = torch.equal(new, rounded_target.float()) if self.strength != 0 else torch.equal(weights, before)
        finite = bool(torch.isfinite(weights).all() and (weights >= 0).all())
        passed = (other_queries and nonvisual_exact and finite and row_error <= TOLERANCES['row_sum_atol']
                  and torch.allclose(newmass, mass, atol=TOLERANCES['visual_mass_atol'], rtol=TOLERANCES['visual_mass_rtol'])
                  and formula_exact and (self.strength == 0 or rounding_pass))
        audit = dict(layer=layer, shape=list(weights.shape), query=self.query, visual_keys=len(visual),
                     other_queries_bitwise_exact=other_queries, nonvisual_bitwise_exact=nonvisual_exact,
                     lambda_zero_tensor_exact=torch.equal(weights, before) if self.strength == 0 else None,
                     row_sum_max_abs_error=row_error, visual_mass_max_abs_error=mass_error,
                     visual_mass_max_relative_error=mass_relative, conditional_l1_max_error=conditional_error,
                     conditional_l1_rounding_bound_max=float(bounds.max()), conditional_rounding_bound_passed=rounding_pass,
                     conditional_nominal_exceeded=conditional_error > TOLERANCES['conditional_l1_nominal'],
                     formula_rounded_target_bitwise_exact=formula_exact,
                     finite_nonnegative=finite, visual_mass_before=mass[:, 0].cpu().tolist(),
                     visual_mass_after=newmass[:, 0].cpu().tolist(), passed=bool(passed))
        self.records.append(audit)
        self.rows.append(torch.stack([old, new]).cpu().numpy())
        if not passed:
            raise AssertionError(f'Attention inheritance audit failed: {audit}')
        return weights

    def eager(self, layer):
        controller = self
        def local_eager(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
            if module.training or dropout != 0:
                raise ValueError('Only eval, dropout0, no-cache full prefill supported')
            key_states = repeat_kv(key, module.num_key_value_groups)
            value_states = repeat_kv(value, module.num_key_value_groups)
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = controller.replace(attn_weights, layer)
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights
        return local_eager

    def __enter__(self):
        if self.saved:
            raise RuntimeError('Context already active')
        try:
            for index in self.active:
                module = self.modules[index]
                if not isinstance(module, LlamaAttention) or module.training or module.config._attn_implementation != 'eager':
                    raise ValueError('Expected eval LlamaAttention eager')
                if 'forward' in module.__dict__ or module.forward.__func__ is not LlamaAttention.forward:
                    raise ValueError('Refuse to replace an existing forward override')
                native = module.forward.__func__
                globals_local = dict(native.__globals__, eager_attention_forward=self.eager(index))
                local = types.FunctionType(native.__code__, globals_local, native.__name__, native.__defaults__, native.__closure__)
                local.__kwdefaults__ = native.__kwdefaults__
                self.saved.append(module)
                module.forward = types.MethodType(local, module)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for module in reversed(self.saved):
            del module.__dict__['forward']
        self.saved.clear()
        return False

    def assert_complete(self):
        if [r['layer'] for r in self.records] != sorted(self.active):
            raise AssertionError('Expected exactly one call per selected layer')
