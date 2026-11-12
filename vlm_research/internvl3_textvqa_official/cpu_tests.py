import unittest

import torch

from .common import KEEP, MODEL, independent_scorer, official_request, render, score, upstream
from .dynamic import engines


def roles(visual):
    image = list(range(5, 5+visual))
    return dict(n=visual+25, image=image, R=list(range(5+visual,visual+24)))


class OfficialInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = upstream()

    def test_exact_official_prompt(self):
        value = official_request(self.scope, 'What word is shown?', '/unused.jpg')
        self.assertEqual(value, '<image>\nWhat word is shown?\nAnswer the question using a single word or phrase.')

    def test_official_soft_score(self):
        for count, expected in enumerate([0, .3, .6, .9, 1]):
            value, _ = score(self.scope, 'cat', ['cat']*count + ['dog']*(10-count))
            self.assertAlmostEqual(value, expected)

    def test_independent_reference_soft_score(self):
        scorer = independent_scorer()
        for count, expected in enumerate([0, .3, .6, .9, 1]):
            value = scorer.eval_pred_list([dict(pred_answer='CAT',
                gt_answers=['cat']*count + ['dog']*(10-count))], disable_tqdm=True)
            self.assertAlmostEqual(value, expected)

    def test_image_cluster_bootstrap(self):
        from .analyze import paired_cluster_ci
        result = paired_cluster_ci([.1, .1, .1], ['x','x','y'], replicates=100)
        self.assertAlmostEqual(result['lower_pp'], 10)
        self.assertAlmostEqual(result['upper_pp'], 10)
        self.assertEqual(result['image_clusters'], 2)

    def test_official_dynamic_tiles(self):
        from PIL import Image
        images = self.scope['dynamic_preprocess'](Image.new('RGB',(1200,800)), max_num=6, use_thumbnail=True)
        self.assertGreater(len(images), 1)
        self.assertLessEqual(len(images), 7)
        self.assertTrue(all(im.size == (448,448) for im in images))

    def test_chat_dynamic_positions(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True, use_fast=False)
        for tiles in [1, 2, 7]:
            _, ids, layout = render(tok, '<image>\nwhat is shown?', tiles)
            self.assertEqual(len(layout['image']), tiles*256)
            self.assertEqual(layout['A_prefill'], [ids.shape[1]-1])
            self.assertTrue(set(layout['R']).isdisjoint(layout['image']))


class DynamicCache(unittest.TestCase):
    def test_no_legacy_global_mutation(self):
        base, late, changes = engines()
        self.assertIs(late.base, base)
        self.assertTrue(all(row['replacements'] for row in changes))

    def test_dynamic_persistent_and_scratch(self):
        _, late, _ = engines()
        for count in [256, 512, 1792]:
            layout = roles(count)
            cache = late.CompactCache(layout, KEEP, 'cpu', cutoff=28, audit=True)
            k = torch.randn(1, 2, layout['n'], 4)
            v = torch.randn_like(k)
            for layer in [0, 19, 27]:
                cache.update(k, v, layer, {'cache_position':torch.arange(layout['n'])})
                expected = layout['n'] if layer in KEEP else 25
                self.assertEqual(cache.key_cache[layer].shape[-2], expected)
            cache.update(k[:,:,:-1], v[:,:,:-1], 2, {'cache_position':torch.arange(layout['n']-1)})
            self.assertIsNone(cache.key_cache[2])
            cache.prefill = False
            for layer in [0,19]:
                a, b = cache.update(k[:,:,:1],v[:,:,:1],layer,{'cache_position':torch.tensor([layout['n']])})
                self.assertEqual(a.shape[-2], layout['n']+1)
                if layer not in KEEP:
                    self.assertEqual(int(a[:,:,layout['image']].count_nonzero()),0)
                    self.assertEqual(int(b[:,:,layout['image']].count_nonzero()),0)
                self.assertEqual(cache.original_positions[layer][-1],layout['n'])

    def test_dynamic_masks_keep_late9_and_R_boundary(self):
        _, late, _ = engines()
        layout = roles(512)
        n = layout['n']
        original = torch.zeros(1,1,n,n)
        for layer in [0,19,20,27]:
            value = late.edit_mask(original,layout,True,layer,set(KEEP))
            target = value[0,0,-1,layout['image']]
            self.assertTrue(bool(torch.isfinite(target).all()) if layer in KEEP else bool(torch.isneginf(target).all()))
            r = value[0,0,layout['R']][:,layout['image']]
            self.assertTrue(bool(torch.isneginf(r).all()) if layer >=20 else bool((r==0).all()))
        self.assertEqual(int(original.count_nonzero()),0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
