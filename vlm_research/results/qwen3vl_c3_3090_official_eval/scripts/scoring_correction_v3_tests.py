import unittest
from scorers_v3 import score
from scorers_v2 import score as alternate
class Tests(unittest.TestCase):
 def test_original_letter_and_integer_gold(self):
  s=dict(benchmark='ScienceQA-IMG',answer=2,choices=['r','g','b']);self.assertEqual(score(s,'C')['score'],1);self.assertEqual(score(s,'C')['parsed'],2)
 def test_frozen_parser_rejects_dot_without_space(self):
  s=dict(benchmark='ScienceQA-IMG',answer=0,choices=['r','g']);self.assertEqual(score(s,'A.')['score'],0);self.assertEqual(alternate(s,'A.')['score'],1)
 def test_frozen_dot_space_and_answer_sentence(self):
  s=dict(benchmark='ScienceQA-IMG',answer=1,choices=['r','g']);self.assertEqual(score(s,'B. green')['score'],1);self.assertEqual(score(s,'The answer is B.')['score'],1)
 def test_unparsed_output_no_random_fallback(self):
  s=dict(benchmark='ScienceQA-IMG',answer=0,choices=['r','g']);self.assertEqual(score(s,'')['parsed'],-1);self.assertEqual(score(s,'red')['score'],0)
if __name__=='__main__':unittest.main(verbosity=2)
