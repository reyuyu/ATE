import unittest

import torch

from benchmarks.smoke import hook_footprint
from knockout_probe import knockout_mask
from paper_rq1.engine import step_inputs
from rq2_visual_handoff.engine import PrefillMask, generate, CONDITIONS
from test_rq2_source_knockout import Tiny, inputs
from rq2_visual_handoff.run import assign


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.model, self.data = Tiny(), inputs()

    def call(self, key, **kwargs):
        return generate(self.model, self.data, 1, key, expected_visual=4,
                        checkpoints=[0, 1, 2], **kwargs)

    def test_partition_all_postimage_nonvisual_not_only_question(self):
        expected = {'A': [], 'B': [8], 'C': [6,7], 'D': [6,7,8]}
        for key in CONDITIONS:
            mask = PrefillMask(self.model, self.data, 1, key, expected_visual=4)
            self.assertEqual(torch.where(mask.relay)[0].tolist(), [6,7])
            self.assertEqual(torch.where(mask.layout['text_queries'])[0].tolist(), expected[key])

    def test_hard_masks_preserve_all_other_edges_and_self(self):
        native = torch.full((1,1,9,9), torch.finfo(torch.float32).min)
        native[0,0].masked_fill_(torch.ones(9,9,dtype=torch.bool).tril(), 0)
        for key in 'BCD':
            mask = PrefillMask(self.model, self.data, 1, key, expected_visual=4)
            changed, count = knockout_mask(native, mask.layout)
            target = mask.layout['text_queries'][:,None] & mask.layout['visual'][None,:]
            self.assertTrue(torch.equal(changed[0,0][~target], native[0,0][~target]))
            self.assertTrue(torch.isneginf(changed[0,0][target]).all())
            self.assertEqual(count, int(mask.layout['text_queries'].sum())*4)
            self.assertTrue((changed[0,0].diag()==0).all())

    def test_noop_and_independent_native_generation(self):
        full, ff, ft, fd = self.call('A', limit=4, audit=True, collect=True)
        for key in 'BCD':
            g, f, trace, debug = self.call(key, noop=True, limit=4, audit=True, collect=True)
            self.assertEqual(g['token_ids'], full['token_ids'])
            self.assertTrue(torch.equal(f, ff))
            self.assertTrue(torch.equal(debug['prefill_logits'], fd['prefill_logits']))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(trace, ft)))
        with torch.inference_mode():
            native = self.model(**self.data, use_cache=True).logits[0,-1].float()
        self.assertTrue(torch.equal(native, ff))

    def test_exact_blocked_probabilities_all_layers_and_decode(self):
        for key in CONDITIONS:
            g, first, trace, debug = self.call(key, limit=4, capture=True)
            self.assertEqual(set(debug['states']), {0,1,2})
            self.assertTrue(all(r['image_probability_max']==0 for r in debug['attention_rows'] if r['blocked']))
            groups = {(r['step']>0,r['query_group']) for r in debug['attention_rows'] if r['blocked']}
            if key == 'B': self.assertEqual(groups,{(False,'P-1'),(True,'Decode')})
            if key == 'C': self.assertEqual(groups,{(False,'Relay')})
            if key == 'D': self.assertEqual(groups,{(False,'Relay'),(False,'P-1'),(True,'Decode')})

    def test_other_prefill_queries_causal_invariants(self):
        _,_,_,a = self.call('A', limit=2, audit=True, capture=True)
        _,_,_,b = self.call('B', limit=2, audit=True, capture=True)
        _,_,_,c = self.call('C', limit=2, audit=True, capture=True)
        _,_,_,d = self.call('D', limit=2, audit=True, capture=True)
        self.assertTrue(torch.equal(a['prefill_logits'][0,:8], b['prefill_logits'][0,:8]))
        self.assertTrue(torch.equal(c['prefill_logits'][0,:8], d['prefill_logits'][0,:8]))
        for k in a['states']:
            self.assertTrue(torch.equal(a['states'][k][:-1],b['states'][k][:-1]))
            self.assertTrue(torch.equal(c['states'][k][:-1],d['states'][k][:-1]))

    def test_image_exchange_neither_is_exactly_invariant(self):
        # Alter image embeddings without changing token positions or text IDs.
        ga, fa, ta, da = self.call('D', limit=4, audit=True, collect=True, capture=True)
        with torch.no_grad(): self.model.language_model.model.embed_tokens.weight[60].add_(3)
        gb, fb, tb, db = self.call('D', limit=4, audit=True, collect=True, capture=True)
        self.assertTrue(torch.equal(fa, fb))
        self.assertEqual(ga['token_ids'],gb['token_ids'])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(ta,tb)))
        text = self.data['input_ids'][0] != 60
        self.assertTrue(torch.equal(da['prefill_logits'][0,text],db['prefill_logits'][0,text]))
        self.assertTrue(all(torch.equal(da['states'][k],db['states'][k]) for k in da['states']))

    def test_dynamic_mapping_cleanup_and_no_input_mutation(self):
        before=hook_footprint(self.model);ids=self.data['input_ids'].clone()
        self.data['input_ids']=torch.cat([ids,torch.tensor([[20,21]])],-1)
        self.data['attention_mask']=torch.ones_like(self.data['input_ids'])
        g,_,_,_=self.call('D',limit=3)
        self.assertEqual(g['relay_indices'],[6,7,8,9]);self.assertEqual(g['p_minus_one'],10)
        with self.assertRaises(RuntimeError):
            with PrefillMask(self.model,self.data,1,'C',expected_visual=4): raise RuntimeError('intentional')
        self.assertEqual(before,hook_footprint(self.model))
        self.assertTrue(torch.equal(self.data['input_ids'][0,:9],ids[0]))

    def test_image_cluster_assignment_includes_swap_endpoints(self):
        main=[dict(question_id='q1',image_sha256='image1'),dict(question_id='q2',image_sha256='image2'),dict(question_id='q3',image_sha256='image3')]
        pairs=[dict(pair_id='p1',samples=main[:2])]
        captions=[dict(question_id='cap',image_sha256='image1')]
        ranks,images=assign(main,pairs,captions,3)
        self.assertEqual(ranks['main']['q1'],ranks['main']['q2'])
        self.assertEqual(ranks['main']['q1'],ranks['swap']['p1'])
        self.assertEqual(ranks['main']['q1'],ranks['caption']['cap'])
        self.assertEqual(images['image1'],images['image2'])
        self.assertEqual((ranks,images),assign(list(reversed(main)),pairs,captions,3))

    def test_all_layers_not_historical_thirty_layer_condition(self):
        for key in 'BCD':
            mask=PrefillMask(self.model,self.data,1,key,expected_visual=4)
            self.assertEqual(mask.active,{0,1,2})
        self.assertEqual(PrefillMask(self.model,self.data,1,'A',expected_visual=4).active,set())


if __name__ == '__main__':
    unittest.main()
