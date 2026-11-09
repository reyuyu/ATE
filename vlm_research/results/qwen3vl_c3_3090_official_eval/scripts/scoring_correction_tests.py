import unittest
from scorers_v2 import score
class Tests(unittest.TestCase):
 def test_numeric_gold_index_correct_letter(self):
  s=dict(benchmark='ScienceQA-IMG',answer=2,choices=['red','green','blue']);self.assertEqual(score(s,'C')['score'],1);self.assertEqual(score(s,'C')['parsed_index'],2)
 def test_numeric_gold_wrong_letter(self):
  s=dict(benchmark='ScienceQA-IMG',answer=0,choices=['red','green']);self.assertEqual(score(s,'B')['score'],0);self.assertEqual(score(s,'A')['score'],1)
 def test_unparsed_answer_does_not_match_zero(self):
  s=dict(benchmark='ScienceQA-IMG',answer=0,choices=['red','green']);self.assertEqual(score(s,'Cannot determine the answer')['score'],0)
if __name__=='__main__':unittest.main(verbosity=2)
