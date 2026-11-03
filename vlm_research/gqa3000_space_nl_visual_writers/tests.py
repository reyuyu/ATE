import unittest,collections
import torch
from gqa3000_token_visual_writing.engine import cpu_mask
from .common import *
class Audit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=read(OUT/'token_plan.json')['cases']
    def test_all_cases_and_tasks(self):
        self.assertEqual(len(self.plan),3000);self.assertEqual(set(collections.Counter(v['task'] for v in self.plan.values()).values()),{500})
    def test_original_boundary_positions(self):
        for v in self.plan.values():self.assertEqual(v['R'][:2],v['B']);self.assertFalse(set(v['B'])&set(v['Q']));self.assertEqual(v['Q'][0],v['B'][-1]+1)
    def test_all_whole_groups(self):
        for v in self.plan.values():
            for w in v['writers'].values():
                if w is not None:self.assertTrue(intact(w,v['groups']));self.assertTrue(set(w)<=set(v['R']))
    def test_exact_added_budget(self):
        for v in self.plan.values():
            S=set(v['S']);N=set(v['ND']);self.assertEqual(len(set(v['writers']['S+SPACE'])-S),1);self.assertEqual(len(set(v['writers']['S+NL'])-S),1)
            self.assertEqual(len(set(v['writers']['S+SPACE+NL'])-S),2);self.assertEqual(len(set(v['writers']['Noun+Article+SPACE+NL'])-N),2)
            if v['template_control'] is not None:
                self.assertEqual(len(set(v['writers']['S+Template-Control-2'])-S),2);self.assertEqual(len(set(v['writers']['N+D+Template-Control-2'])-N),2)
    def test_no_image_or_answer_receivers(self):
        for v in self.plan.values():
            for w in v['writers'].values():
                if w is not None:self.assertTrue(set(w)<=set(v['R']))
    def test_canonical_aliases(self):
        for v in self.plan.values():
            for n,w in v['writers'].items():
                if w is not None:self.assertEqual(v['aliases'][n],digest(w));self.assertEqual(v['unique'][digest(w)]['positions'],w);self.assertIn(n,v['unique'][digest(w)]['labels'])
    def test_historical_zero_k_case(self):
        v=self.plan['10763612'];self.assertEqual(v['S'],[]);self.assertEqual(v['writers']['S+SPACE+NL'],v['B']);self.assertEqual(len(v['writers']['S+Template-Control-2']),2)
    def test_cohort_disjoint_images(self):
        c=read(OUT/'cohorts.json');old=set(c['old240_image_sha256']);subset=[v for v in self.plan.values() if v['confirmatory2640']];self.assertEqual(len(subset),2640);self.assertFalse({v['image_sha256'] for v in subset}&old)
    def test_pilot_and_harm_untouched(self):self.assertEqual(sum(v['pilot120'] for v in self.plan.values()),120);self.assertEqual(sum(v['historical12'] for v in self.plan.values()),12)
    def test_same_image_same_shard(self):
        s=read(OUT/'prospective_shards.json');self.assertEqual(sum(s['two_worker_questions'].values()),3000)
        for v in self.plan.values():self.assertIn(s['two_workers'][v['image_sha256']],[0,1]);self.assertEqual(s['one_worker'][v['image_sha256']],0)
    def test_actual_masks(self):
        for q in ['001020372','10763612']:
            v=self.plan[q];n=v['R'][-1]+2;layout=dict(prompt_length=n,I=list(range(v['R'][0]-576,v['R'][0])),R=v['R'],A=[n-1],other=list(range(v['R'][0]-576)))
            for label in CONDITIONS:
                w=v['writers'][label]
                if w is None:continue
                native,changed,blocked=cpu_mask(layout,w)
                self.assertTrue(torch.equal(changed[0,0][~blocked],native[0,0][~blocked]));self.assertTrue(torch.isneginf(changed[0,0][blocked]).all())
                self.assertEqual(int(blocked.sum()),(len(v['R'])-len(w)+1)*576)
                self.assertFalse(blocked[layout['I']].any());self.assertTrue(blocked[n-1,layout['I']].all())
                for p in v['B']:self.assertEqual(bool(blocked[p,layout['I']].all()),p not in w)
                a,b,d=cpu_mask(layout,w,past=n);self.assertEqual(int(d.sum()),576);self.assertTrue(torch.equal(b[0,0][~d],a[0,0][~d]))
    def test_correction_family_fixed(self):self.assertEqual(len(COMPARISONS),5);self.assertEqual(.05/len(COMPARISONS),.01)
if __name__=='__main__':unittest.main()
