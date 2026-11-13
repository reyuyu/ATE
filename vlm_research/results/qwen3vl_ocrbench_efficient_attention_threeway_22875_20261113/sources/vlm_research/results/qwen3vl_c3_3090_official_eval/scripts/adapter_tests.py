"""Small CPU tests for newly introduced I/O and author scorer adapters."""
import unittest
from scorers import score,text_evaluator,option_matcher,mme_parser
class Tests(unittest.TestCase):
 def test_gqa_frozen_conversion(self):
  s=dict(benchmark='GQA',answer='white');self.assertEqual(score(s,'White.')['score'],1);self.assertEqual(score(s,' white')['score'],0)
 def test_textvqa_ten_answer_soft(self):
  s=dict(benchmark='TextVQA',answer=['yes']*3+['no']*7);self.assertAlmostEqual(score(s,'yes')['score'],.9);self.assertEqual(score(s,'no')['score'],1)
 def test_mme_original_no_extra_strip(self):
  s=dict(benchmark='MME',answer='yes');self.assertEqual(score(s,'Yes.')['score'],1);self.assertTrue(score(s,'  yes')['parse_failed'])
 def test_pope_original_rules(self):
  s=dict(benchmark='POPE',answer='no');self.assertEqual(score(s,'No, there is not.')['score'],1);self.assertEqual(score(s,'Yes. No.')['score'],0)
 def test_official_option_matcher(self):
  s=dict(benchmark='MMBench',answer='B',choices=dict(A='apple',B='pear'));self.assertEqual(score(s,'(B)')['score'],1);self.assertTrue(score(s,'Cannot determine the answer')['parse_failed'])
 def test_ocrbench_math_case_and_spaces(self):
  s=dict(benchmark='OCRBench-v1',answer='x + Y',dataset_name='HME100k');self.assertEqual(score(s,'x+Y')['score'],1);self.assertEqual(score(s,'x+y')['score'],0)
if __name__=='__main__':unittest.main(verbosity=2)
