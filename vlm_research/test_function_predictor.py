import unittest
import numpy as np
import torch
from paper_rq1_function_attention_sparsify.predictor import prefix_features,hidden_features,OnlinePredictor,category


class PredictorTests(unittest.TestCase):
    def test_prefix_only_consumed(self):
        p=[319,767,338];one=prefix_features(p)
        np.testing.assert_array_equal(one,prefix_features(list(p)))
        self.assertEqual(one[-4],3/64);self.assertEqual(one.shape,(1284,))

    def test_prefix_not_final_length(self):
        self.assertEqual(prefix_features([1,2])[-4],2/64)
        self.assertFalse(np.array_equal(prefix_features([1,2]),prefix_features([1,3])))

    def test_reject_prefill(self):
        with self.assertRaises(ValueError):prefix_features([])

    def test_hidden_pre_attention(self):
        h=torch.ones(4096);x=hidden_features(h,4)
        self.assertAlmostEqual(float(x[:4096].norm()),1.)
        self.assertEqual(x.shape,(4098,))

    def test_invalid_hidden(self):
        with self.assertRaises(ValueError):hidden_features(torch.zeros(4096),1)
        with self.assertRaises(ValueError):hidden_features(torch.full((4096,),float('nan')),1)

    def test_online_cpu_same_formula(self):
        parameters={'A_mean':np.zeros(1284,dtype=np.float32),'A_scale':np.ones(1284,dtype=np.float32),
            'A_weight':np.ones(1284,dtype=np.float32)*.1,'A_bias':np.array(-.5,dtype=np.float32)}
        model=OnlinePredictor(parameters,'A',.5,'cpu')
        expected=torch.sigmoid(torch.from_numpy(prefix_features([12,3]))@torch.from_numpy(parameters['A_weight'])-.5)
        self.assertEqual(float(model.probability([12,3],None)),float(expected))

    def test_eos_kept_negative(self):
        self.assertEqual(category({'tokens':[{'category':'EOS'}],'words':[]},0),'EOS-Special')

    def test_multitoken_kept_negative(self):
        a={'tokens':[{'category':'Function'}],'words':[dict(token_steps=[0,1],exact_complete_single_token=False,token_length=2,word_type='Function')]}
        self.assertEqual(category(a,0),'Multi-token-Unaligned')


if __name__=='__main__':unittest.main()
