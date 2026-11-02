import unittest
import torch
from .run import LegacyLMHeadShape
from .common import divergence,repeated_trigram,MANIFEST_SHA,OLD,sha256


class Tests(unittest.TestCase):
    def test_legacy_argument_and_exception_cleanup(self):
        class Dummy(torch.nn.Module):
            def forward(self,x,logits_to_keep=0):
                self.seen=logits_to_keep
                return x
        model=Dummy();x=torch.randn(1,3,4)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with LegacyLMHeadShape(model):
                self.assertTrue(torch.equal(model(x),x));self.assertEqual(model.seen,1)
                raise RuntimeError('intentional')
        self.assertFalse(model._forward_pre_hooks)
        model(x);self.assertEqual(model.seen,0)
        with self.assertRaises(AssertionError):
            with LegacyLMHeadShape(model):model(x,logits_to_keep=0)
        self.assertFalse(model._forward_pre_hooks)

    def test_divergence_and_repetition(self):
        self.assertIsNone(divergence([1,2],[1,2]))
        self.assertEqual(divergence([1,2],[1,3]),1)
        self.assertEqual(divergence([1,2],[1,2,3]),2)
        self.assertFalse(repeated_trigram([4,2]))
        self.assertTrue(repeated_trigram([3,4,5,3,4,5,2]))

    def test_manifest_is_original_bytes(self):
        self.assertEqual(len(MANIFEST_SHA),64)
        self.assertEqual(sha256(OLD/'sample_manifest.json'),MANIFEST_SHA)
