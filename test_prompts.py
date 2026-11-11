import unittest

from prompt_common import *
from freeze_prompts import choose, partition
from analyze_prompts import units, bootstrap


class Tests(unittest.TestCase):
    def test_p0_identity(self):
        r=dict(body='Question unchanged\n',benchmark='GQA')
        self.assertEqual(prompt_body(r,'P0'),r['body'])

    def test_gqa_appends_without_removing_short_suffix(self):
        r=dict(body='Q\nAnswer the question using a single word or phrase.',benchmark='GQA')
        self.assertEqual(prompt_body(r,'P1'),r['body']+'\n'+P1_GQA)

    def test_choices_hint_preserved(self):
        r=dict(body='Hint: h\nQuestion: q\nOptions:\nA. x\nB. y',benchmark='ScienceQA-IMG')
        for p,s in [('P1',P1_CHOICE),('P2',P2_CHOICE)]:
            self.assertEqual(prompt_body(r,p),r['body']+'\n'+s)

    def test_no_gold_dependence(self):
        a=dict(body='q',benchmark='MMBench',gold='A')
        b=dict(a,gold='D')
        self.assertEqual(prompt_body(a,'P2'),prompt_body(b,'P2'))

    def test_no_gqa_p2(self):
        with self.assertRaises(AssertionError):
            prompt_body(dict(body='q',benchmark='GQA'),'P2')

    def test_diagnostic_truncated(self):
        p=load_parser().option_choice('The answer is A.',{'A':'x','B':'y'},truncated=True)
        self.assertIsNone(p.letter)

    def test_circular_all_rotations_required(self):
        a=[dict(benchmark='MMBench',circular_group=1,image_sha256='a',score=x) for x in [1,1,0]]
        self.assertEqual(units(a)['1'][1],0)

    def test_identical_bootstrap(self):
        a={'a':('image',1.),'b':('image',0.)}
        stats=bootstrap(a,a)
        self.assertEqual(stats['estimate_pp'],0)
        self.assertEqual(stats['ci_low_pp'],0)
        self.assertEqual(stats['image_clusters'],1)

    def test_partition_stable(self):
        self.assertEqual(partition('abc'),partition('abc'))


if __name__=='__main__':
    unittest.main(verbosity=2)
