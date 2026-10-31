from contextlib import AbstractContextManager
import numpy as np
import torch
from attention_probe import SUM_ATOL
from attention_position_probe import locate_question_tokens, partition_keys
from knockout_probe import model_layers

METRICS = ['image_mass', 'bos_mass', 'question_mass', 'prompt_template_mass', 'prompt_mass']


def true_question_positions(processor, sample):
    payload, question = sample['prompt_text'], sample['question']
    assert payload.startswith(question + '\n')
    records, _ = locate_question_tokens(processor, payload, sample['input_mapping']['prompt_ids'], 32000)
    selected = []
    for record in records:
        left, right = record['question_char_span']
        if left < len(question):
            assert 0 <= left < right <= len(question), 'Token crosses question/instruction boundary'
            selected.append(record)
    assert processor.tokenizer.decode([r['token_id'] for r in selected],
        clean_up_tokenization_spaces=False) == question
    assert [r['sequence_index'] for r in records] == sample['input_mapping']['question_and_instruction_positions']
    return [r['sequence_index'] for r in selected]


def groups_for(sample, question_positions):
    ids = sample['input_mapping']['prompt_ids']
    n = len(ids)
    valid, groups = partition_keys(ids, np.ones(n), n - 1, 32000,
        sample['input_mapping']['bos_token_id'], n, question_positions)
    assert groups['visual'].sum() == 576 and groups['bos'].sum() == 1
    assert not groups['previous_answer'].any() and valid.all()
    assert groups['question'].any() and groups['prompt_template'].any()
    return groups


def reduce_row(row, groups):
    row = np.asarray(row, dtype=np.float64)
    assert row.ndim == 2 and row.shape[0] == 32
    assert np.isfinite(row).all() and (row >= 0).all()
    error = float(np.max(np.abs(row.sum(-1) - 1)))
    assert error <= SUM_ATOL, 'Native FP16 row-sum error exceeds historical2e-3 bound'
    values = [float(row[:, groups[g]].sum(-1).mean()) for g in
        ['visual', 'bos', 'question', 'prompt_template']]
    assert abs(sum(values) - 1) <= SUM_ATOL
    values.append(values[2] + values[3])
    return dict(zip(METRICS, values)), error, row[:, groups['visual']].mean(0)


class NativeRows(AbstractContextManager):
    """Read-only observers; return None so native outputs are not replaced."""
    def __init__(self, model, sample, question_positions, spatial=False):
        self.model, self.sample = model, sample
        self.query = sample['input_mapping']['p_minus_one']
        self.groups = groups_for(sample, question_positions)
        self.spatial = spatial
        self.rows, self.maps, self.handles = {}, {}, []

    def __enter__(self):
        try:
            blocks = model_layers(self.model)
            assert len(blocks) == 32
            for layer, block in enumerate(blocks):
                def observe(module, args, output, layer=layer):
                    weights = output[1]
                    assert weights.shape == (1, 32, self.query + 1, self.query + 1)
                    assert layer not in self.rows
                    row = weights[0, :, self.query].detach().float().cpu().numpy()
                    values, error, visual = reduce_row(row, self.groups)
                    self.rows[layer] = dict(layer=layer, **values, max_head_row_sum_error=error,
                        source_sum_error=abs(values['image_mass'] + values['bos_mass'] + values['prompt_mass'] - 1))
                    if self.spatial:
                        self.maps[layer] = visual.reshape(24, 24).tolist()
                self.handles.append(block.self_attn.register_forward_hook(observe))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


@torch.inference_mode()
def observe(model, inputs, sample, positions, spatial=False):
    assert inputs['input_ids'][0].tolist() == sample['input_mapping']['prompt_ids']
    assert inputs['attention_mask'].all()
    with NativeRows(model, sample, positions, spatial) as trace:
        output = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
        logits = output.logits[0, trace.query].detach().float().cpu().clone()
        assert output.past_key_values is None and torch.isfinite(logits).all()
        assert sorted(trace.rows) == list(range(32))
    return [trace.rows[k] for k in range(32)], trace.maps, logits
