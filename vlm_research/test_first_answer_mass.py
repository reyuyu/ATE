import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from transformers import AutoProcessor
from first_answer_mass.engine import NativeRows, reduce_row, groups_for, true_question_positions
from first_answer_mass.analyze import mapped_field
from first_answer_mass.run import OUT, MODEL_PATH, cohort


def fake_sample():
    ids=[1]+[32000]*576+[19,20,21,22]
    return dict(input_mapping=dict(prompt_ids=ids,p_minus_one=len(ids)-1,bos_token_id=1))


class FakeAttention(nn.Module):
    def forward(self, hidden, weights):
        return hidden, weights


class MassTests(unittest.TestCase):
    def test_partition_is_disjoint_and_exhaustive(self):
        groups=groups_for(fake_sample(),[578,579])
        total=np.sum(list(groups.values()),axis=0)
        self.assertTrue((total==1).all())
        self.assertEqual(groups['visual'].sum(),576)
        self.assertFalse(groups['question'][-1])

    def test_mass_reduction_and_zero_visual(self):
        s=fake_sample()
        groups=groups_for(s,[578,579])
        row=np.ones((32,581),dtype=float)/581
        values,error,visual=reduce_row(row,groups)
        self.assertLess(error,1e-12)
        self.assertAlmostEqual(sum(values[k] for k in ['image_mass','bos_mass','prompt_mass']),1.)
        self.assertEqual(values['prompt_mass'],values['question_mass']+values['prompt_template_mass'])
        self.assertAlmostEqual(visual.sum(),values['image_mass'])
        row[:,groups['visual']]=0
        row/=row.sum(-1,keepdims=True)
        self.assertEqual(reduce_row(row,groups)[0]['image_mass'],0)

    def test_bad_values_rejected(self):
        groups=groups_for(fake_sample(),[578,579])
        for bad in [float('nan'),float('inf'),-1.,0.]:
            row=np.ones((32,581))/581
            if bad==0: row[:]=0
            else: row[0,0]=bad
            with self.assertRaises(AssertionError): reduce_row(row,groups)

    def test_read_only_hooks_and_normal_cleanup(self):
        blocks=[SimpleNamespace(self_attn=FakeAttention()) for _ in range(32)]
        weights=torch.full((1,32,1,581),1/581).expand(1,32,581,581)
        hidden=torch.arange(8.)
        with patch('first_answer_mass.engine.model_layers',return_value=blocks):
            with NativeRows(None,fake_sample(),[578,579],True) as trace:
                for block in blocks:
                    output=block.self_attn(hidden,weights)
                    self.assertIs(output[0],hidden)
                    self.assertIs(output[1],weights)
                self.assertEqual(sorted(trace.rows),list(range(32)))
                self.assertEqual(np.asarray(trace.maps[0]).shape,(24,24))
        self.assertTrue(all(not b.self_attn._forward_hooks for b in blocks))

    def test_exception_cleanup(self):
        blocks=[SimpleNamespace(self_attn=FakeAttention()) for _ in range(32)]
        with patch('first_answer_mass.engine.model_layers',return_value=blocks):
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with NativeRows(None,fake_sample(),[578,579]):
                    raise RuntimeError('intentional')
        self.assertTrue(all(not b.self_attn._forward_hooks for b in blocks))

    def test_actual_question_offsets_and_variable_positions(self):
        p=AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
        ids={r['question_id'] for r in json.loads((OUT/'case_pool.json').read_text())['samples']}
        samples=[s for s in cohort() if s['question_id'] in ids]
        lengths=set()
        for s in samples:
            question=true_question_positions(p,s)
            groups_for(s,question)
            self.assertLess(len(question),len(s['input_mapping']['question_and_instruction_positions']))
            lengths.add(s['input_mapping']['p_minus_one'])
        self.assertEqual(len(samples),12)
        self.assertGreater(len(lengths),1)

    def test_inverse_crop_is_not_stretched(self):
        g=dict(original_size=[672,336],resized_size=[672,336],crop_left=168,crop_top=0)
        field,mask=mapped_field(np.ones((24,24)),g)
        self.assertEqual(field.shape,(336,672))
        self.assertTrue((mask[:,0:168]==0).all())
        self.assertTrue((mask[:,168:504]==1).all())
        self.assertTrue((mask[:,504:]==0).all())
        self.assertTrue((field[:,168:504]==1).all())


if __name__=='__main__':
    unittest.main()
