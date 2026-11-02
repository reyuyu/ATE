import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from benchmarks.smoke import hook_footprint
from attention_text_regions.engine import CONDITIONS, REGIONS, Intervention, generate
from attention_text_regions.run import select, assign, uid
from test_rq2_source_knockout import Tiny, inputs


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11); self.model = Tiny()
        cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=24,
            num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, attention_dropout=0.)
        cfg._attn_implementation = 'eager'; cfg.image_token_index = 60
        self.model.config = cfg; self.model.language_model = LlamaForCausalLM(cfg).eval()
        self.data = inputs()

    def run_condition(self, condition, prefill_active, **kw):
        return generate(self.model, self.data, condition, prefill_active, 4, expected_visual=4, **kw)

    def test_frozen_regions(self):
        self.assertEqual(len(CONDITIONS), 7)
        self.assertEqual([len(x) for x in REGIONS.values()], [10,10,11])
        self.assertEqual(set().union(*map(set, REGIONS.values())), set(range(1,32)))

    def test_exact_noop_all_logits_and_cache_isolation(self):
        a, f, _, _, trace, debug = self.run_condition('Full', True, collect=True, audit=True)
        for condition in CONDITIONS[1:]:
            g, ff, _, _, tt, dd = self.run_condition(condition, True, noop=True, collect=True, audit=True)
            self.assertEqual(g['token_ids'], a['token_ids'])
            self.assertTrue(torch.equal(debug['prefill_logits'], dd['prefill_logits']))
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(trace,tt)))
        b, *_ = self.run_condition('Full', True)
        self.assertEqual(a['token_ids'], b['token_ids'])

    def test_scope_timing_probabilities_and_projection(self):
        _, f, _, _, _, full = self.run_condition('Full', True, audit=True)
        for caption in [False, True]:
            for condition in CONDITIONS[1:]:
                g, ff, rows, proj, _, debug = self.run_condition(condition, not caption, audit=True)
                self.assertTrue(torch.equal(full['prefill_logits'][:,:-1], debug['prefill_logits'][:,:-1]))
                self.assertEqual(full['states'], debug['states'])
                self.assertEqual(full['other_query_attention'], debug['other_query_attention'])
                if caption: self.assertTrue(torch.equal(full['prefill_logits'], debug['prefill_logits']))
                region, kind, _ = condition.split()
                for row in rows:
                    active = row['layer'] in REGIONS[region] and (row['step'] > 0 or not caption)
                    self.assertEqual(row['text_masked'], active and kind == 'Text')
                    self.assertGreater(row['image_mass'], 0.)
                    if row['text_masked']: self.assertEqual(row['text_probability_max'], 0.)
                for r in proj:
                    active = r['layer'] in REGIONS[region] and (r['step'] > 0 or not caption)
                    self.assertEqual(r['applied'], active and kind == 'Attention')
                    if r['applied']: self.assertEqual(r['output_norm'], 0.)

    def test_actual_history_includes_original_p1_and_past_generated_only(self):
        for step in [0,1,3]:
            c = Intervention(self.model, self.data, 'Early Text OFF', step, True, expected_visual=4)
            self.assertFalse(c.text[0]); self.assertFalse(c.text[c.images].any())
            self.assertFalse(c.text[c.absolute])
            if step: self.assertTrue(c.text[c.n-1])
            self.assertEqual(int(c.text[c.n:].sum()), max(0,step-1))

    def test_exception_hook_cleanup_and_input_unchanged(self):
        foot = hook_footprint(self.model); ids = self.data['input_ids'].clone()
        with self.assertRaises(RuntimeError):
            with Intervention(self.model, self.data, 'Late Text OFF', 0, True, expected_visual=4):
                raise RuntimeError('intentional')
        self.assertEqual(foot, hook_footprint(self.model))
        self.run_condition('Middle Attention OFF', True)
        self.assertTrue(torch.equal(ids, self.data['input_ids']))
        self.assertEqual(foot, hook_footprint(self.model))

    def test_whole_image_shards(self):
        ss = [dict(dataset='gqa', task='t', question_id=str(i), image_sha256=str(i//2)) for i in range(20)]
        a = assign(ss); self.assertEqual(a, assign(list(reversed(ss))))
        for i in range(0,20,2): self.assertEqual(a[uid(ss[i])], a[uid(ss[i+1])])


if __name__ == '__main__': unittest.main()
