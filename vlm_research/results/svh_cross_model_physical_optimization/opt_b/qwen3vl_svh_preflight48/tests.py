"""CPU tests: partition definitions, frozen sampling and unchanged numerical gates."""
import unittest
from collections import Counter

import torch
from .freeze import choose_edges, choose_pretest, grouped_exact, options_body
from .run import strict_equal
from .score import score
from ..qwen3vl_svh_numerical_repair.run import GATE, compare


class Tests(unittest.TestCase):
    def test_options_and_hint(self):
        body = options_body('q?', {'A':'first','B':'second'}, 'visible hint')
        self.assertEqual(body, 'Hint: visible hint\nQuestion: q?\nOptions:\nA. first\nB. second\nPlease select the correct answer from the options above. \n')
        self.assertNotIn('Hint:', options_body('q?', {'A':'x'}, ''))

    def test_groups_never_split(self):
        rows = [dict(key=str(i), group=i//4) for i in range(80)]
        selected = grouped_exact(rows, 'group', 48, 'test')
        self.assertEqual(len(selected), 48)
        self.assertTrue(all(n == 4 for n in Counter(r['group'] for r in selected).values()))
        self.assertEqual(selected, grouped_exact(list(reversed(rows)), 'group', 48, 'test'))

    def test_gate_not_relaxed(self):
        self.assertEqual(GATE, dict(mean_abs=.02, max_abs=.5, top1_equal=True, complete_tokens_equal=True))
        a = dict(token_ids=[1])
        self.assertTrue(compare(a,[torch.tensor([0.,1.])],a,[torch.tensor([0.,1.])])[0]['pass'])
        self.assertFalse(compare(a,[torch.tensor([0.,1.])],a,[torch.tensor([0.,.9])])[0]['pass'])

    def test_strict_noop(self):
        result = dict(token_ids=[1])
        a = result, [torch.tensor([0.,1.])]
        b = result, [torch.tensor([0.,1.00001])]
        self.assertFalse(strict_equal(a,b))
        self.assertTrue(strict_equal(a,a))

    def test_no_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_frozen_scoring(self):
        row = dict(benchmark='ScienceQA-IMG',gold=1,options={'A':'a','B':'b'})
        self.assertEqual(score(row,'B')['score'],1.)
        self.assertEqual(score(row,'The answer is B')['score'],0.)
        self.assertEqual(score(dict(benchmark='TextVQA',gold=['cat']*10),'cat')['score'],1.)
        self.assertEqual(score(dict(benchmark='GQA',gold='yes'),'yes.')['score'],0.)


if __name__ == '__main__':
    unittest.main()
