"""Single-query, per-head positive visual reweighting before P@V.

Reuse the validated instance-local eager wrapper and FP16 audits without
altering the inheritance experiment or any installed library source.
"""
import numpy as np
import torch
from attention_inheritance import VisualAttentionInheritance


def conditional(array):
    x = np.asarray(array, dtype=np.float64)
    mass = x.sum(-1, keepdims=True)
    if not np.isfinite(x).all() or (x < 0).any() or (mass <= 0).any():
        raise ValueError('Undefined visual-conditional distribution')
    return x / mass


def spatial_similarity(p, q):
    """Cosine similarity and sqrt Jensen-Shannon divergence in bits [0,1]."""
    p, q = conditional(p), conditional(q)
    m = (p + q) / 2
    def kl(a):
        return np.sum(a * (np.log(np.maximum(a, 1e-300)) - np.log(np.maximum(m, 1e-300))), axis=-1)
    js = np.sqrt(np.clip((kl(p) + kl(q)) / (2*np.log(2)), 0, 1))
    cosine = (p*q).sum(-1) / np.sqrt((p*p).sum(-1)*(q*q).sum(-1))
    return cosine, js


def factors_for(condition, reference):
    """reference is frozen per-head B, not a head-mean display image."""
    b = conditional(reference)
    if condition['family'] == 'A':
        alpha, eps = condition['strength'], condition['epsilon']
        if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(eps) or eps <= 0:
            raise ValueError('Invalid debiasing parameters')
        return (b + eps) ** (-alpha)
    factors = np.ones_like(b)
    indices = condition['patch_indices']
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < b.shape[-1] for i in indices):
        raise ValueError('Invalid patch subset')
    if not np.isfinite(condition['strength']) or condition['strength'] <= 0:
        raise ValueError('Factors must be positive and finite')
    factors[:, indices] = condition['strength']
    return factors


class VisualAttentionReweight(VisualAttentionInheritance):
    """p_new = normalize(p_current * factors); m_visual stays local/current.

    All-one factors bypass assignment exactly. Zero-mass rows remain zero.
    The superclass validates local mass, row sum, rounded formula, non-target
    invariance and a dtype-derived conditional rounding bound on every call.
    """
    def __init__(self, modules, visual_indices, query_position, factors, active_layers):
        factors = torch.as_tensor(factors, dtype=torch.float32)
        if factors.ndim != 2 or not torch.isfinite(factors).all() or (factors <= 0).any():
            raise ValueError('Expected positive finite [heads, patches] factors')
        self.is_noop = bool((factors == 1).all())
        # Scaling each head's factors does not change the normalized result.
        self.factors = factors / factors.max(-1, keepdim=True).values
        super().__init__(modules, visual_indices, query_position,
                         torch.ones_like(factors), 0 if self.is_noop else 1,
                         active_layers=active_layers)

    def replace(self, weights, layer):
        indices = self.visual.to(weights.device)
        old = weights[0, :, self.query, :].index_select(-1, indices).float()
        weighted = old * self.factors.to(weights.device)
        mass = weighted.sum(-1, keepdim=True)
        self.reference = torch.where(mass > 0, weighted / mass.clamp_min(1e-30),
                                     torch.zeros_like(weighted))
        result = super().replace(weights, layer)
        self.records[-1]['intervention'] = 'positive_visual_reweight'
        self.records[-1]['no_op'] = self.is_noop
        return result
