"""Analytical boundary checks, no GPU imports or model inference."""
import unittest
from analysis import bootstrap,mme_metrics,independent,paired_cases,paired,summarize
class Tests(unittest.TestCase):
 def test_cluster_weighting_uses_question_denominator(self):
  # Both clusters have a +1 mean: duplicating a cluster keeps exactly +1.
  self.assertEqual(bootstrap([1,10],[1,10],100),[100.,100.])
 def test_mme_requires_both_questions_for_accuracy_plus(self):
  rows=[dict(task='count',pair_id='a',score=1),dict(task='count',pair_id='a',score=0),dict(task='count',pair_id='b',score=1),dict(task='count',pair_id='b',score=1)]
  value=mme_metrics(rows);self.assertEqual(value['tasks']['count']['Accuracy'],75);self.assertEqual(value['tasks']['count']['AccuracyPlus'],50);self.assertEqual(value['Total'],125)
 def test_pope_empty_defaults_yes_but_mme_does_not(self):
  self.assertEqual(independent(dict(benchmark='POPE',answer='yes'),'',None),1)
  self.assertEqual(independent(dict(benchmark='MME',answer='yes'),'',None),0)
 def test_ocr_math_is_case_sensitive(self):
  self.assertEqual(independent(dict(benchmark='OCRBench-v1',answer=['X + Y'],dataset_name='HME100k'),'x+y',None),0)
  self.assertEqual(independent(dict(benchmark='OCRBench-v1',answer=['X + Y'],dataset_name='HME100k'),'X+Y',None),1)
 def test_scienceqa_integer_gold_no_random_fallback(self):
  s=dict(benchmark='ScienceQA-IMG',answer=0,choices=['r','g'])
  self.assertEqual(independent(s,'A',0),1);self.assertEqual(independent(s,'',-1),0)
  self.assertEqual(independent(s,'A.',-1),0);self.assertEqual(independent(s,'A. red',0),1)
 def test_empty_view_stays_visible_without_formal_score(self):
  rows,_=summarize([],'POPE','OFFICIAL_PROTOCOL',9000,False)
  self.assertEqual(len(rows),2);self.assertTrue(all(r['completed']==0 and r['formal_score'] is None and r['observed_subset_mean_score'] is None for r in rows))
 def test_circular_group_has_no_fake_first_token_counter(self):
  rows=[dict(key='1',condition=c,image_sha256='x',score=1,parsed='True') for c in ['Native','Physical-Original']]
  r=paired(rows,('Physical-Original','Native'),'MMBench','ATE_PAPER_PROTOCOL',metric='Circular Accuracy pp')
  self.assertIsNone(r['first_token_changed']);self.assertIsNone(r['parsed_answer_changed'])
if __name__=='__main__':unittest.main(verbosity=2)
