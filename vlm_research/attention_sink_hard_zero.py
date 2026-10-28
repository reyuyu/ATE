"""Exact Sink-key zeroing, visual-only mass redistribution, post-softmax/pre-V.

Uses the unchanged instance-local inheritance wrapper and its FP16 audits.
No library edits, weight changes, full-row normalization or token deletion.
"""
import torch
from attention_inheritance import VisualAttentionInheritance


class VisualSinkHardZero(VisualAttentionInheritance):
    def __init__(self, modules, visual_indices, query_position, blocked_patches, active_layers):
        modules = list(modules)
        visual = torch.as_tensor(visual_indices, dtype=torch.long)
        blocked = list(blocked_patches)
        if (len(set(blocked)) != len(blocked) or
                any(type(i) is not int or not 0 <= i < len(visual) for i in blocked)):
            raise ValueError("Blocked patches must be unique visual-relative integer indices")
        if len(blocked) == len(visual):
            raise ValueError("Cannot preserve visual mass after removing all visual keys")
        heads = modules[0].config.num_attention_heads
        self.blocked = torch.tensor(blocked, dtype=torch.long)
        self.is_noop = not blocked
        super().__init__(modules, visual, query_position, torch.ones(heads, len(visual)),
                         0 if self.is_noop else 1, active_layers=active_layers)

    def replace(self, weights, layer):
        if not self.is_noop:
            visual = self.visual.to(weights.device)
            old = weights[0, :, self.query, :].index_select(-1, visual).float()
            remaining = old.clone()
            remaining[:, self.blocked.to(weights.device)] = 0
            total = old.sum(-1, keepdim=True)
            retained = remaining.sum(-1, keepdim=True)
            if ((total > 0) & (retained <= 0)).any():
                raise ValueError("Positive visual mass has no remaining support; refuse arbitrary fallback")
            self.reference = remaining / retained.clamp_min(1e-30)
        out = super().replace(weights, layer)
        selected = self.visual.index_select(0, self.blocked).to(weights.device)
        zero = bool((out[0, :, self.query, :].index_select(-1, selected) == 0).all())
        self.records[-1].update(intervention="hard_zero_visual_only_renormalization",
                                blocked_patches=self.blocked.tolist(), blocked_weights_exact_zero=zero,
                                no_op=self.is_noop)
        self.records[-1]["passed"] &= zero
        if not zero:
            raise AssertionError("A blocked patch retained positive attention")
        return out
