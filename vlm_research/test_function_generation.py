import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from paper_rq1_function_attention_sparsify.generation import Policy


class PolicyTests(unittest.TestCase):
    def prompt(self):return {'input_ids':torch.tensor([[1,10]+[32000]*576+[11,12]]),'attention_mask':torch.ones(1,580)}

    def test_no_prefill_pruning(self):
        for strategy in ['S0','S1','S2','S3','S4']:
            p=Policy(None,self.prompt(),strategy);p.next([]);self.assertFalse(p.prune);self.assertIsNone(p.groups)

    def test_static_only_decode(self):
        for s in ['S1','S2']:
            p=Policy(None,self.prompt(),s);p.next([319]);self.assertTrue(p.prune);self.assertEqual(p.groups['GENERATED_HISTORY'],[580])

    def test_dynamic_waits_for_real_hidden(self):
        p=Policy(None,self.prompt(),'S3');p.next([319]);self.assertFalse(p.prune)

    def test_prefix_owned(self):
        prefix=[319,767];p=Policy(None,self.prompt(),'S1');p.next(prefix);prefix.append(99);self.assertEqual(p.prefix,[319,767])

    def test_full_no_new_hooks(self):
        import transformers.models.llama.modeling_llama as llama
        original=llama.eager_attention_forward
        with Policy(None,self.prompt(),'S0') as p:self.assertEqual(p.handles,[])
        self.assertIs(llama.eager_attention_forward,original)

    def test_exception_cleanup(self):
        import transformers.models.llama.modeling_llama as llama
        original=llama.eager_attention_forward
        with self.assertRaises(RuntimeError):
            with Policy(None,self.prompt(),'S1'):raise RuntimeError('intentional')
        self.assertIs(llama.eager_attention_forward,original)

    def test_pipeline_status_mutable_not_checkpoint(self):
        from paper_rq1_function_attention_sparsify.pipeline import status
        with tempfile.TemporaryDirectory() as folder:
            with patch('paper_rq1_function_attention_sparsify.pipeline.OUT',Path(folder)):
                status(stage='analysis',status='running');status(stage='train',status='running')
            self.assertEqual(json.loads((Path(folder)/'pipeline_status.json').read_text())['stage'],'train')

    def test_transformers_scaling_keyword_contract(self):
        import transformers.models.llama.modeling_llama as llama
        module=SimpleNamespace(layer_idx=31,num_key_value_groups=1,training=False)
        q=torch.ones(1,1,1,4);k=torch.ones(1,1,5,4);v=torch.ones_like(k)
        with Policy(None,self.prompt(),'S1') as p:
            p.next([])
            out,weights=llama.eager_attention_forward(module,q,k,v,None,scaling=.5,dropout=0.)
        self.assertEqual(out.shape,(1,1,1,4));self.assertEqual(weights.shape,(1,1,1,5))


if __name__=='__main__':unittest.main()
