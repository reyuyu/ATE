import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from paper_rq1_function_attention_sparsify.engine import sparse_attention,independent_reference,retained,Trace,partition
from paper_rq1_function_attention_sparsify.protocol import PROTOCOL


class SparseTests(unittest.TestCase):
    def groups(self,t=4):
        prompt={'input_ids':torch.tensor([[1,10]+[32000]*576+[11,12]]),'attention_mask':torch.ones(1,580)}
        return partition(prompt,t)

    def test_partition(self):
        g=self.groups();self.assertEqual(sorted(g['BOS']+g['IMAGE']+g['PROMPT_TEXT']+g['GENERATED_HISTORY']),list(range(584)))

    def test_key_sets(self):
        g=self.groups();self.assertEqual(len(retained(g,'No-Image')),8);self.assertEqual(retained(g,'BOS+History'),[0,580,581,582,583]);self.assertEqual(retained(g,'History-only'),[580,581,582,583])

    def test_dynamic_positions(self):
        p={'input_ids':torch.tensor([[10,1,11]+[32000]*576+[12]]),'attention_mask':torch.ones(1,580)}
        self.assertEqual(partition(p,1)['BOS'],[1])

    def test_padding_rejected(self):
        with self.assertRaises(ValueError):partition({'input_ids':torch.ones(1,580,dtype=torch.long),'attention_mask':torch.zeros(1,580)},1)

    def test_sparse_reference(self):
        torch.manual_seed(7);m=SimpleNamespace(num_key_value_groups=2)
        q=torch.randn(1,4,1,8,dtype=torch.float64);k=torch.randn(1,2,17,8,dtype=torch.float64);v=torch.randn_like(k)
        for pos in [[0,16],[16],list(range(17))]:
            a,w=sparse_attention(m,q,k,v,None,8**-.5,pos);b,_=independent_reference(m,q,k,v,None,8**-.5,pos)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
            self.assertEqual(w.shape[-1],len(pos))

    def test_excluded_values_not_read_by_av(self):
        m=SimpleNamespace(num_key_value_groups=1);q=torch.ones(1,2,1,4);k=torch.ones(1,2,6,4);v=torch.ones_like(k);v[:,:,1:5]=float('nan')
        result,_=sparse_attention(m,q,k,v,None,.5,[0,5]);self.assertTrue(torch.isfinite(result).all())

    def test_mask_preserved(self):
        m=SimpleNamespace(num_key_value_groups=1);q=torch.ones(1,1,1,4);k=torch.ones(1,1,5,4);v=torch.arange(20.).view(1,1,5,4)
        mask=torch.zeros(1,1,1,5);mask[:,:,:,-1]=-float('inf')
        _,w=sparse_attention(m,q,k,v,mask,.5,[0,4]);self.assertEqual(float(w[0,0,0,1]),0.)

    def test_unknown_algorithm(self):
        with self.assertRaises(ValueError):Trace(None,algorithm='mask')

    def test_only_fixed_layers(self):
        with self.assertRaises(ValueError):Trace(None,algorithm='zero',layer=5)

    def test_failed_context_cleanup(self):
        import transformers.models.llama.modeling_llama as llama
        original=llama.eager_attention_forward
        with patch('paper_rq1_function_attention_sparsify.engine.model_layers',side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                with Trace(None,algorithm='A2',mode='BOS+History'):pass
        self.assertIs(llama.eager_attention_forward,original)

    def test_labels_not_online_features(self):
        self.assertIn('No final generation length',PROTOCOL['classifier']['features'])
        self.assertEqual(PROTOCOL['max_new_tokens'],64)


if __name__=='__main__':unittest.main()
