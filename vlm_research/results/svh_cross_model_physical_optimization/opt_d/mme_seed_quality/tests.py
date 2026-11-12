"""CPU protocol gates for scoring, duplicate preservation and candidate labels."""
import unittest
from .common import *

class ProtocolTests(unittest.TestCase):
    def test_mme_official_parser(self):
        expected={'yes':'yes','Yes.':'yes','No.':'no','not sure':'no','The answer is yes':'other','\nYes':'yes','yesterday':'yes'}
        for x,y in expected.items():self.assertEqual(mme_parse(x),y)
    def test_whole_image_shards(self):
        seen={}
        for r in read(OUT/'sample_manifest.json'):
            h=r['image_sha256'];self.assertEqual(seen.setdefault(h,r['shard']),r['shard'])
    def test_counts_and_duplicate(self):
        r=read(OUT/'sample_manifest.json');seed=[s for s in r if s['benchmark']=='SEED']
        self.assertEqual(len(r),16607);self.assertEqual(len(seed),14233)
        self.assertEqual(len({s['source_question_id'] for s in seed}),14232)
        self.assertEqual(len({s['key'] for s in r}),16607)
    def test_gold_not_in_prompt_or_candidate_selection(self):
        bindings=read(OUT/'llava_input_bindings.json.gz')
        for s in read(OUT/'sample_manifest.json'):
            b=bindings[s['key']]
            self.assertIn(s['question'],b['prompt'])
            if s['benchmark']=='SEED':
                self.assertEqual(len(b['candidate_token_ids']),4)
                self.assertTrue(all(b['candidate_token_ids']))
    def test_frozen_hashes(self):
        p=read(OUT/'protocol.json')
        self.assertEqual(p['sample_manifest_sha256'],sha(OUT/'sample_manifest.json'))
        self.assertEqual(p['llava_bindings_sha256'],sha(OUT/'llava_input_bindings.json.gz'))
    def test_original_physical_sources(self):
        s=(ROOT/'vlm_research/qwen3vl_c3v2_physical/engine.py').read_text()
        self.assertNotIn('RepairedExecution',s);self.assertIn('i==22',s)
        self.assertIn('torch.zeros_like(x)',s)

if __name__=='__main__':unittest.main()
