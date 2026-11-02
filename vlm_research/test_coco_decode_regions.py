import unittest
import torch
from benchmarks.smoke import hook_footprint
from coco_decode_regions import conditions, generate
from rq2_decode_visual.engine import DecodeImageKnockout
from test_rq2_sparse_caption import Tests as OldTests


class Tests(OldTests):
    def call(self, key, **kwargs):
        retained = {'A': list(range(32)), 'S': [14, 19], 'DS': [14, 19], 'DN': []}[key]
        return generate(self.model, self.data, 1, retained, expected_visual=4, limit=3, **kwargs)

    def test_scope_actual_zero_and_unaffected_queries(self):
        _, af, _, _, ad = self.call('A', audit=True)
        for retained in conditions().values():
            g, f, rows, _, d = generate(self.model, self.data, 1, retained,
                                       expected_visual=4, limit=3, audit=True)
            self.assertTrue(torch.equal(ad['prefill_logits'], d['prefill_logits']))
            self.assertEqual(ad['states'], d['states'])
            self.assertEqual(ad['prefill_rows'], d['prefill_rows'])
            self.assertTrue(torch.equal(af, f))
            for r in rows:
                self.assertEqual(r['blocked'], r['step'] > 0 and r['layer'] not in retained)
                self.assertEqual(r['image_probability_max'] == 0., r['blocked'])

    def test_random_sets_frozen(self):
        c = conditions(); self.assertEqual(len(c), 12)
        for key, retained in c.items():
            self.assertEqual(len(retained), len(set(retained)))
            if key not in ['Full', 'Decode None']: self.assertIn(0, retained)
            if key.startswith('Random10'): self.assertEqual(len(retained), 11)
            if key.startswith('Random11'): self.assertEqual(len(retained), 12)
        self.assertEqual(c, conditions())
        self.assertNotEqual(c['Random10-20261102'], c['Random10-20261103'])

    def test_decode_exception_cleanup(self):
        foot = hook_footprint(self.model)
        with self.assertRaises(RuntimeError):
            with DecodeImageKnockout(self.model, self.data, list(range(32)), 1, expected_visual=4):
                raise RuntimeError('intentional')
        self.assertEqual(foot, hook_footprint(self.model))


if __name__ == '__main__': unittest.main()
