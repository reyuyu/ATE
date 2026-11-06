import unittest
from .common import *


class ProtocolTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(sum(COUNTS.values()), 367028)
    def test_choices_keep_hint_and_order(self):
        raw = dict(question='Q?', hint='real context', choices=['first', 'second'])
        text = prompt(raw, 'ScienceQA-IMG')
        self.assertEqual(text, 'Context: real context\nQ?\nA. first\nB. second\n' + MC)
        self.assertNotIn('gold', text)
    def test_pope(self):
        s = dict(benchmark='POPE', answer='no')
        self.assertEqual(score(s, 'No, there is not.')['score'], 1)
        self.assertEqual(score(s, 'Yes.')['score'], 0)
    def test_science_parser(self):
        s = dict(benchmark='ScienceQA-IMG', answer=1, raw=dict(choices=['a', 'b']))
        self.assertEqual(score(s, 'B')['score'], 1)
        self.assertEqual(score(s, 'not an option')['score'], 0)
        self.assertTrue(score(s, 'not an option')['parse_failed'])
    def test_vqa_leave_one_out(self):
        s = dict(benchmark='VQAv2', answer=dict(answers=[dict(answer=a) for a in
            ['red'] * 3 + ['blue'] * 7]))
        self.assertAlmostEqual(score(s, 'red')['score'], .9)
    def test_textvqa(self):
        self.assertEqual(score(dict(benchmark='TextVQA', answer=['red'] * 10), 'red')['score'], 1)
    def test_gqa_frozen_conversion(self):
        self.assertEqual(score(dict(benchmark='GQA', answer='red'), 'Red.')['score'], 1)
        self.assertEqual(score(dict(benchmark='GQA', answer='red'), 'Red object.')['score'], 0)
    def test_unavailable_judges_never_fabricate_score(self):
        for b in ('MM-Vet',):
            self.assertIsNone(score(dict(benchmark=b, answer='A'), 'A')['score'])
    def test_mmbench_official_matching(self):
        s = dict(benchmark='MMBench', answer='B', raw=dict(choices=dict(A='red', B='blue')))
        self.assertEqual(score(s, 'B')['score'], 1)
        self.assertEqual(score(s, 'blue')['score'], 1)
        self.assertEqual(score(s, 'A or B')['score'], 0)


if __name__ == '__main__':
    unittest.main()
