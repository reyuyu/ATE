import unittest

from rq2_visual_full.run import split, populations, SECONDARY, PRIMARY, assign, GPUS


class Tests(unittest.TestCase):
    def test_split_excludes_every_shared_image(self):
        discovery=[dict(question_id=str(i),image_sha256=str(i)) for i in range(240)]
        ss=discovery+[dict(question_id='new_shared',image_sha256='0'),
                      dict(question_id='new_unique',image_sha256='unique')]
        flags=split(ss,discovery);groups=populations(ss,flags)
        self.assertEqual(len(groups['validation']),2)
        self.assertEqual([s['question_id'] for s in groups[PRIMARY]],['new_unique'])
        self.assertFalse(flags['new_shared']['exploration'])
        self.assertFalse(flags['new_shared']['image_disjoint'])
        self.assertEqual(split(list(reversed(ss)),discovery),flags)

    def test_primary_and_secondary_are_fixed(self):
        self.assertEqual(PRIMARY,'validation_image_disjoint')
        self.assertEqual(SECONDARY,[('C','A'),('C','B'),('D','Full'),('A','Full')])

    def test_assignment_is_image_clustered(self):
        ss=[dict(question_id=str(i),image_sha256=str(i//3)) for i in range(2760)]
        a=assign(ss);self.assertEqual(a,assign(ss[::-1]))
        for i in range(0,2760,3):self.assertEqual(len({a[str(i+j)] for j in range(3)}),1)
        loads=[list(a.values()).count(i) for i in range(len(GPUS))]
        self.assertEqual(sum(loads),2760)
        self.assertLessEqual(max(loads)-min(loads),3)


if __name__=='__main__':unittest.main()
