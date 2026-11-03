import itertools,unittest
from unittest.mock import patch
from .common import *
from .selection import *
from gqa3000_token_visual_writing.engine import cpu_mask

class SearchTests(unittest.TestCase):
    def test_exact_knapsack_matches_exhaustive_and_never_splits(self):
        g=[dict(positions=[0,1]),dict(positions=[2]),dict(positions=[3,4]),dict(positions=[5])];scores=[.7,-.2,1.1,.5]
        for k in range(7):
            a=exact_knapsack(g,k,scores);actual=sum(scores[i]*len(x['positions']) for i,x in enumerate(g) if set(x['positions'])<=set(a));expected=max(sum(scores[i]*len(g[i]['positions']) for i in range(4) if m[i]) for m in itertools.product([0,1],repeat=4) if sum(len(g[i]['positions'])*m[i] for i in range(4))==k);self.assertAlmostEqual(actual,expected)
    def test_all_frozen_k80_budgets_and_group_closure(self):
        for path in (P/'input_specs').glob('*.json'):
            s=read(path);g=[x for x in s['groups'] if x['question']];answer=exact_knapsack(g,s['k80'],[1]*len(g));self.assertEqual(len(answer),s['k80']);self.assertTrue(intact(answer,s['groups']));self.assertEqual(flatten(g),s['Q'])
    def test_real_masks_and_non_target_probabilities(self):
        layout=dict(prompt_length=9,I=[2,3,4],R=[5,6,7],A=[8],other=[0,1])
        for w in [[],[6],[5,6,7]]:
            for past in [0,9,13]:
                old,new,blocked=cpu_mask(layout,w,past);self.assertTrue(torch.equal(old[0,0][~blocked],new[0,0][~blocked]));a=new.softmax(-1);self.assertEqual(float(a[0,:,blocked].max()),0);self.assertEqual(int(blocked.sum()),3 if past else (4-len(w))*3)
    def test_holdout_is_image_disjoint_and_sealed(self):
        s=read(P/'split_manifest.json');im={k:{s['metadata'][q]['image_sha256'] for q in v} for k,v in s['ids'].items()};self.assertFalse(im['DEV']&im['FINAL'] or im['TUNE']&im['FINAL'] or im['DEV']&im['TUNE'])
        q=s['ids']['FINAL'][0]
        original_exists=Path.exists
        with patch.object(Path,'exists',lambda path:False if path==P/'winner_freeze.json' else original_exists(path)):
            with self.assertRaises(AssertionError):gold(q)
    def test_selector_api_excludes_ID_Gold_image_and_outputs(self):
        import inspect
        self.assertEqual(list(inspect.signature(select).parameters),['sp','config']);s=spec(read(P/'split_manifest.json')['ids']['DEV'][0]);self.assertNotIn('answer',s);self.assertNotIn('question_id',s);self.assertNotIn('image_sha256',s);self.assertNotIn('gold',s)
    def test_role_policy_deterministic_same_exact_budget(self):
        conf=dict(family='role',weights=dict(noun=2,spatial=1,logic=1,position=.1),fraction=.8)
        for q in read(P/'split_manifest.json')['dev_stages']['60']:
            s=spec(q);a=select(s,conf);self.assertEqual(a,select(s,conf));self.assertEqual(len(a),s['k80']);self.assertTrue(intact(a,s['groups']))
    def test_mean_R_is_not_fraction_of_Question(self):
        q=read(P/'split_manifest.json')['ids']['DEV'][0];s=spec(q);self.assertLess(len(s['Q'])/len(s['R']),1);self.assertNotEqual(s['k80']/len(s['R']),s['k80']/len(s['Q']))
    def test_no_Gold_in_public_specs(self):
        for f in (P/'input_specs').glob('*.json'):
            s=read(f);self.assertNotIn('answer',s);self.assertNotIn('gold_token_ids',s);self.assertNotIn('correct',s)
if __name__=='__main__':unittest.main(verbosity=2)
