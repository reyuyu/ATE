"""Extend the existing native SourceTrace, never modifying its eager outputs."""
from contextlib import AbstractContextManager
import numpy as np
import torch
import torch.nn.functional as F
import transformers.models.llama.modeling_llama as llama

from knockout_probe import model_layers, ROW_ATOL
from paper_rq1_late_attention_sources.engine import (
    SourceTrace, partition, contributions, group_head, cosine, gamma,
)
from paper_rq1_token_role_decode.labels import align
from .common import SOURCES, SELECTED


def groups_at(prompt, t, image_id=32000):
    original = partition(prompt, max(t, 1), image_token_id=image_id)
    groups = {name: original[name] for name in ['BOS', 'IMAGE', 'PROMPT_TEXT']}
    groups['GENERATED_OLDER'] = original['OLDER_HISTORY'] if t else []
    groups['SELF'] = original['RECENT'] if t else []
    visible = prompt['input_ids'].shape[1] + t
    flattened = [p for name in SOURCES for p in groups[name]]
    assert sorted(flattened) == list(range(visible)) and len(set(flattened)) == visible
    return groups, original


def history_split(projection, row, values, groups, outputs):
    """Same analytic History=Recent+Older bound as the inherited L26/L31 audit."""
    heads = {name: group_head(row, values, groups[name]) for name in ['GENERATED_HISTORY', 'RECENT', 'OLDER_HISTORY']}
    for name in ['RECENT', 'OLDER_HISTORY']:
        outputs[name] = F.linear(heads[name], projection.weight.float(), None)
    p, v = row.float(), values.float()
    magnitude = (p[:, groups['GENERATED_HISTORY']].abs().unsqueeze(-2) @
                 v[:, groups['GENERATED_HISTORY']].abs()).squeeze(-2).reshape(1, 1, -1)
    bound = (2*gamma(len(groups['GENERATED_HISTORY']))+gamma(2))*magnitude
    bound = F.linear(bound, projection.weight.float().abs(), None)+gamma(projection.weight.shape[-1])*F.linear(
        sum(heads[name].abs() for name in heads), projection.weight.float().abs(), None)
    bound += gamma(2)*sum(outputs[name].abs() for name in heads)
    error = (outputs['GENERATED_HISTORY']-outputs['RECENT']-outputs['OLDER_HISTORY']).abs()
    assert torch.all(error <= bound), 'Frozen analytic history-split bound failed'
    ratio = torch.where(bound > 0, error/bound, torch.zeros_like(bound))
    return dict(history_split_max_abs_error=float(error.max()), history_split_max_error_to_bound=float(ratio.max()))


class WordTrace(SourceTrace):
    """Inherited real Decode tap; only add read-only direction measurements."""
    def __init__(self, model, groups, previous):
        super().__init__(model, groups, observe=True)
        self.active = SELECTED
        self.previous = previous

    def output_hook(self, index):
        def observe(module, args, output):
            assert output.shape[:2] == (1, 1)
            row, values = self.pv.pop(index)
            residual = self.residual.pop(index)
            vectors, rows, audit = contributions(module, row, values, self.groups, args[0], output, residual, index)
            if index == 14:
                audit.update(history_split(module, row, values, self.groups, vectors))
                for name in ['RECENT', 'OLDER_HISTORY']:
                    vector = vectors[name]
                    rows.append(dict(layer=index, source=name, n_keys=len(self.groups[name]),
                        mass=float(row.float()[:, self.groups[name]].sum(-1).mean()) if self.groups[name] else 0.,
                        contribution_norm=float(vector.norm()), relative_residual_ratio=float(vector.norm()/residual.norm()),
                        residual_norm=float(residual.norm()), attention_output_norm=float(output.float().norm()),
                        cosine_with_full_attention=cosine(vector, output.float()), cosine_with_residual=cosine(vector, residual),
                        attention_residual_cosine=cosine(output.float(), residual),
                        image_history_cosine=cosine(vectors['IMAGE'], vectors['GENERATED_HISTORY']),
                        group_cancellation_ratio=rows[0]['group_cancellation_ratio'],
                        **{'cosine_with_'+n: cosine(vector, vectors[n]) for n in ['BOS', 'IMAGE', 'PROMPT_TEXT', 'GENERATED_HISTORY']}))
            for record in rows:
                name = record['source']
                vector = vectors[name].detach().cpu().reshape(-1)
                old = self.previous.get((index, name))
                record['adjacent_source_cosine'] = cosine(vector, old) if old is not None else None
                record['adjacent_relative_norm_change'] = (abs(float(vector.norm())-float(old.norm()))/float(old.norm())
                    if old is not None and float(old.norm()) > 0 else None)
                self.previous[index, name] = vector
                record['source'] = {'RECENT': 'SELF', 'OLDER_HISTORY': 'GENERATED_OLDER'}.get(name, name)
            self.rows.extend(rows)
            self.audits.append(audit)
            return output
        return observe


class Observation(AbstractContextManager):
    def __init__(self, model, prompt, image_id=32000):
        self.model, self.prompt, self.image_id = model, prompt, image_id
        self.n = prompt['input_ids'].shape[1]
        self.mass, self.key_counts, self.contribution_rows, self.audits, self.cache_lengths = [], [], [], [], []
        self.previous, self.trace, self.handles = {}, None, []
        self.original = llama.eager_attention_forward
        self.step = 0

    def before(self, module, args, kwargs):
        assert self.trace is None and llama.eager_attention_forward is self.original
        t = self.step
        value = kwargs.get('input_ids')
        value = value if value is not None else kwargs['inputs_embeds']
        assert value.shape[:2] == (1, self.n if t == 0 else 1)
        cache = kwargs.get('past_key_values')
        expected = self.n+t-1 if t else 0
        assert cache is None or cache.get_seq_length() == expected
        assert t == 0 or cache is not None
        positions = kwargs.get('cache_position')
        assert positions is not None and positions.tolist() == list(range(expected, expected+value.shape[1]))
        self.groups, legacy = groups_at(self.prompt, t, self.image_id)
        self.current = {}
        if t:
            self.trace = WordTrace(self.model, legacy, self.previous)
            self.trace.__enter__()
        native = llama.eager_attention_forward
        def tap(attention, query, key, value, mask, scaling, **kw):
            output, weights = native(attention, query, key, value, mask, scaling, **kw)
            layer = attention.layer_idx
            assert layer not in self.current and query.shape[-2] == (self.n if t == 0 else 1)
            row = weights[0, :, -1].detach().float()
            assert row.shape[-1] == self.n+t and torch.isfinite(row).all() and (row >= 0).all()
            assert float((row.sum(-1)-1).abs().max()) <= ROW_ATOL
            reduced = torch.stack([row[:, self.groups[name]].sum(-1) for name in SOURCES], -1)
            assert float((reduced.sum(-1)-row.sum(-1)).abs().max()) < 1e-6
            self.current[layer] = reduced.cpu().numpy()
            return output, weights
        llama.eager_attention_forward = tap

    def clean_step(self):
        if self.trace is not None:
            self.trace.__exit__(None, None, None)
            self.trace = None
        llama.eager_attention_forward = self.original

    def after(self, module, args, kwargs, output):
        try:
            if output is None:
                return
            assert sorted(self.current) == list(range(32))
            assert torch.isfinite(output.logits).all()
            assert all(output.past_key_values.get_seq_length(i) == self.n+self.step for i in range(32))
            self.cache_lengths.append(self.n+self.step)
            self.mass.append(np.stack([self.current[i] for i in range(32)]))
            self.key_counts.append([len(self.groups[name]) for name in SOURCES])
            if self.trace is not None:
                assert len(self.trace.rows) == 18 and sorted(r['layer'] for r in self.trace.audits) == SELECTED
                self.contribution_rows.extend(dict(t=self.step, **r) for r in self.trace.rows)
                self.audits.extend(dict(t=self.step, **r) for r in self.trace.audits)
            self.step += 1
        finally:
            self.clean_step()

    def __enter__(self):
        assert len(model_layers(self.model)) == 32
        self.handles = [self.model.register_forward_pre_hook(self.before, with_kwargs=True),
                        self.model.register_forward_hook(self.after, with_kwargs=True, always_call=True)]
        return self

    def __exit__(self, *_):
        self.clean_step()
        for handle in self.handles:
            handle.remove()
        self.handles = []


def labels(tokenizer, ids):
    annotation = align(tokenizer, ids)
    assert annotation['roundtrip_exact'], 'Do not invent word alignment for a failed tokenizer roundtrip'
    rows = []
    for t, original in enumerate(annotation['tokens']):
        matches = [w for w in annotation['words'] if t in w['token_steps']]
        word = matches[0] if len(matches) == 1 else None
        raw = original['category']
        role = raw if raw in ['Function', 'Visual', 'Relation', 'Punctuation', 'EOS'] else 'Unknown'
        rows.append(dict(original, raw_role=raw, role=role,
            word_first=bool(word and t == word['token_steps'][0]),
            subword_index=word['token_steps'].index(t) if word else None,
            word_token_length=word['token_length'] if word else None,
            word_first_step=word['token_steps'][0] if word else None,
            rule_source=word['rule_source'] if word else 'nonlexical_or_unaligned',
            ambiguous_mapped_unknown=raw == 'Ambiguous'))
    return annotation, rows
