"""CPU-only checks before any official model is loaded onto GPU1."""
from pathlib import Path
import unittest

from PIL import Image
import torch
from torch import nn

from .common import OUT, read, write
from .engine import LayerTrace, MassObserver, P1OutputHook, official_preprocessor, partitions, reduce_mass


class HookTests(unittest.TestCase):
    def test_p1_only_zero(self):
        module = nn.Linear(8,8)
        value = torch.randn(1,5,8)
        expected = module(value)
        with P1OutputHook(module,4) as hook:
            result = module(value)
        self.assertEqual(hook.calls,1)
        self.assertTrue(torch.equal(expected[:,:4],result[:,:4]))
        self.assertEqual(int(result[:,-1].count_nonzero()),0)
        self.assertFalse(module._forward_hooks)

    def test_noop_exact(self):
        module = nn.Linear(8,8)
        value = torch.randn(1,3,8)
        expected = module(value)
        with P1OutputHook(module,2,noop=True):
            self.assertTrue(torch.equal(expected,module(value)))

    def test_cleanup_exception(self):
        module = nn.Linear(8,8)
        with self.assertRaisesRegex(RuntimeError,'cleanup'):
            with P1OutputHook(module,1):
                raise RuntimeError('cleanup')
        self.assertFalse(module._forward_hooks)

    def test_mass_partition_and_sum(self):
        groups=dict(anchor=[0],image=[1,2],question=[3],template=[4],self=[5])
        masses,error=reduce_mass(torch.full((4,6),1/6),groups)
        self.assertLess(error,1e-6)
        self.assertTrue(torch.allclose(masses.sum(-1),torch.ones(4)))

    def test_duplicate_or_missing_keys_rejected(self):
        with self.assertRaises(AssertionError):
            reduce_mass(torch.full((4,5),.2),dict(anchor=[0],image=[1],question=[2],template=[2],self=[4]))

    def test_nonfinite_and_bad_sum_rejected(self):
        groups=dict(anchor=[0],image=[1],question=[2],template=[3],self=[4])
        for row in (torch.ones(4,5),torch.full((4,5),float('nan'))):
            with self.assertRaises(AssertionError):
                reduce_mass(row,groups)

    def test_observer_exception_restores_official_function(self):
        import transformers.models.qwen2.modeling_qwen2 as qwen
        original=qwen.eager_attention_forward
        with self.assertRaisesRegex(RuntimeError,'cleanup'):
            with MassObserver({0:nn.Linear(8,8)},{}):
                raise RuntimeError('cleanup')
        self.assertIs(qwen.eager_attention_forward,original)

    def test_real_tiny_decoder_causal_prefix_and_residual(self):
        from transformers import Qwen2Config,Qwen2ForCausalLM
        config=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,attn_implementation='eager')
        model=Qwen2ForCausalLM(config).eval()
        ids=torch.tensor([[1,2,3,4]])
        with torch.inference_mode(),LayerTrace(model.model.layers,3) as native:
            full=model(ids,use_cache=True)
            native.validate()
        with torch.inference_mode(),P1OutputHook(model.model.layers[0].self_attn.o_proj,3),LayerTrace(model.model.layers,3) as changed:
            off=model(ids,use_cache=True)
            changed.validate()
        for a,b in zip(native.records,changed.records):
            self.assertEqual(a['attention_prefix_sha256'],b['attention_prefix_sha256'])
            self.assertEqual(a['output_prefix_sha256'],b['output_prefix_sha256'])
        self.assertEqual(full.past_key_values.get_seq_length(),off.past_key_values.get_seq_length())

    def test_official_preprocessing_limits_and_thumbnail(self):
        directory=Path(read(OUT/'download_manifest.json')['local_path'])
        load_image,dynamic=official_preprocessor(directory)
        sample=read(OUT/'sample_manifest.json')['samples'][-1]
        self.assertEqual(tuple(load_image(sample['image_path'],max_num=1).shape),(1,3,448,448))
        self.assertEqual(tuple(load_image(sample['image_path'],max_num=2).shape),(3,3,448,448))
        with Image.open(sample['image_path']) as image:
            self.assertEqual(len(dynamic(image.convert('RGB'),max_num=2,use_thumbnail=False)),2)

    def test_offsets_bind_exact_slow_tokenizer_ids(self):
        from transformers import AutoTokenizer
        directory=read(OUT/'download_manifest.json')['local_path']
        slow=AutoTokenizer.from_pretrained(directory,use_fast=False,local_files_only=True)
        fast=AutoTokenizer.from_pretrained(directory,use_fast=True,local_files_only=True)
        question='Is the object red?'
        prompt='<|im_start|>user\n<img>'+'<IMG_CONTEXT>'*8+'</img>\n'+question+'\n<|im_end|>\n<|im_start|>assistant\n'
        ids=slow(prompt)['input_ids']
        layout=partitions(slow,fast,prompt,question,ids,slow.convert_tokens_to_ids('<IMG_CONTEXT>'))
        self.assertEqual(layout['image_tokens'],8)
        self.assertEqual(layout['p_minus_1'],len(ids)-1)
        self.assertEqual(layout['actual_bos_positions'],[])


if __name__ == '__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HookTests))
    write(OUT/'cpu_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),cuda_used=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
