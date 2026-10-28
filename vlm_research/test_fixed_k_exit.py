"""Phase 3.2 CPU correctness, frozen sampling and resume regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from test_knockout import tiny_model, tiny_inputs
from physical_exit import measured_forward, aligned, equivalence
from prepare_fixed_k_data import select_rows, digest
from fixed_k_exit import CUTOFFS, save_checkpoint, load_checkpoint, check_directory


class FixedKTest(unittest.TestCase):
    def test_sampling_balance_exclusion_order_and_unique_images(self):
        rows=[dict(question_id=f'{i:04}',img_id=str(i//2),question='Where?',answer=label,
                   **{'true option':label,'false option':'right' if label=='left' else 'left'})
              for i in range(1200) for label in ('left','right')]
        excluded={str(i) for i in range(12)}
        first=select_rows(rows,excluded)
        self.assertEqual(first,select_rows(rows[::-1],excluded))
        self.assertEqual(len({r['img_id'] for r in first}),240)
        self.assertFalse(excluded & {r['img_id'] for r in first})
        self.assertEqual([sum(r['answer']==x for r in first) for x in ('left','right')],[120,120])
        self.assertNotEqual(first,select_rows(rows,excluded,seed=42))

    def test_invalid_candidate_rows_and_short_pool_rejected(self):
        invalid=[dict(question_id=str(i),img_id=str(i),question='x',answer='left',
                      **{'true option':'left','false option':'behind'}) for i in range(300)]
        with self.assertRaises(ValueError):
            select_rows(invalid,set())

    def test_new_boundaries_real_deletion_mapping_and_equivalence(self):
        torch.set_num_threads(2)
        model=tiny_model(); inputs=tiny_inputs(); keep=torch.where(inputs['input_ids'][0]!=99)[0]
        for k in CUTOFFS:
            with self.subTest(k=k):
                physical=measured_forward(model,inputs,'physical',k,4)
                mask=measured_forward(model,inputs,'mask',k,4)
                self.assertEqual([r['sequence_length'] for r in physical['layers']],
                                 [12 if i<k else 8 for i in range(32)])
                self.assertEqual(physical['original_positions'].tolist(),list(range(12)) if k==32 else keep.tolist())
                self.assertTrue(equivalence(aligned(mask,keep,'hidden'),aligned(physical,keep,'hidden'),
                                            aligned(mask,keep,'logits'),aligned(physical,keep,'logits'),strict=True)['passed'])

    def checkpoint(self):
        sample=dict(question_id='001',img_id='002',answer='left')
        data=dict(k32_exact=True,rows=[dict(question_id='001',img_id='002',answer='left',cutoff=k) for k in CUTOFFS])
        return sample,data

    def test_resume_roundtrip_no_overwrite(self):
        sample,data=self.checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'001.json'
            save_checkpoint(path,'binding',sample,data)
            self.assertEqual(load_checkpoint(path,'binding',sample),data)
            self.assertEqual(len(check_directory(Path(directory),[sample],'binding')),1)
            with self.assertRaises(ValueError):
                save_checkpoint(path,'binding',sample,data)
            for binding, changed in [('new',sample),('binding',dict(sample,answer='right'))]:
                with self.assertRaises(ValueError):
                    load_checkpoint(path,binding,changed)

    def test_tamper_duplicate_incomplete_and_unknown_checkpoint_rejected(self):
        sample,data=self.checkpoint()
        for kind in ('tamper','duplicate','incomplete','unknown'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/('999.json' if kind=='unknown' else '001.json')
                altered=copy.deepcopy(data)
                if kind=='duplicate': altered['rows'][-1]['cutoff']=0
                if kind=='incomplete': altered['rows'].pop()
                save_checkpoint(path,'binding',sample,altered)
                if kind=='tamper':
                    value=json.loads(path.read_text());value['body']['data']['k32_exact']=False
                    path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    check_directory(Path(directory),[sample],'binding')

    def test_flip_statistics_not_net_accuracy(self):
        from summarize_fixed_k import summarize_rows
        rows=[]
        # Both have accuracy 1/2, but one correct->wrong and one wrong->correct.
        for q,full,exit in [('1',1.,-2.),('2',-1.,2.)]:
            for k,margin in [(0,exit),(32,full)]:
                rows.append(dict(question_id=q,answer='left' if q=='1' else 'right',cutoff=k,
                                 full_correct=full>0,candidate_correct=margin>0,full_margin=full,answer_margin=margin,
                                 answer_margin_drop=full-margin,question_mean_cosine_distance=.1 if k==0 else 0.))
        summary=summarize_rows(rows,[0,32])[0]
        self.assertEqual(summary['correct_to_wrong'],1)
        self.assertEqual(summary['wrong_to_correct'],1)
        self.assertEqual(summary['full_correct_flip_rate'],1.)
        self.assertEqual(summary['accuracy_delta'],0.)
        self.assertEqual(summary['margin_drop_mean'],0.)
        self.assertEqual(summary['max_positive_margin_drop'],3.)


if __name__=='__main__':
    unittest.main()
