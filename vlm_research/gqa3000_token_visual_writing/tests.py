import inspect,itertools,unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from .common import CONDITIONS,OUT,read,key,gold_losses,confirmatory_decisions,digest
from .selection import semantic_first,priority_random,stratified,stratum_counts,select,intact,flatten
from .engine import cpu_mask,forward,native_teacher
from .prepare import image_shards,swap_pairs

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.g=[dict(positions=[2,3],core=True),dict(positions=[4],core=False),dict(positions=[5,6],core=True),dict(positions=[7],core=False)]
    def test_semantic_DP_matches_exhaustive_core_optimum(self):
        for k in range(7):
            possible=[list(itertools.compress(self.g,mask)) for mask in itertools.product([0,1],repeat=4) if sum(len(g['positions'])*m for g,m in zip(self.g,mask))==k]
            if not possible:continue
            actual=semantic_first(self.g,k);self.assertEqual(len(actual),k);self.assertTrue(intact(actual,self.g))
            score=lambda ss:sum(len(g['positions']) for g in ss if g['core'])
            self.assertEqual(sum(len(g['positions']) for g in self.g if g['core'] and set(g['positions'])<=set(actual)),max(map(score,possible)))
    def test_random_question_only_seed_reproducible_no_gold_or_image(self):
        self.assertEqual(priority_random(self.g,9,'text'),priority_random(self.g,9,'text'))
        self.assertNotIn('gold',inspect.signature(priority_random).parameters)
        self.assertNotIn('image',inspect.signature(priority_random).parameters)
    def test_stratified_exact_whole_group_quotas(self):
        q=list(range(2,8));semantic=[2,3,5,6]
        for seed in [1,2]:
            actual,target=stratified(self.g,4,semantic,q,seed,'text')
            self.assertTrue(intact(actual,self.g));self.assertEqual(stratum_counts(actual,q),target)
    def test_historical_zero_budget_is_preserved(self):
        groups=[dict(positions=[2],core=True),dict(positions=[3,4],core=False)]
        s=select(groups,[2,3,4],[2,3,4],'which?');self.assertEqual(s['k'],0)
        self.assertEqual(s['writers']['Semantic-k'],[])
    def test_all_frozen_sets_exact_budget_question_scope_and_aliases(self):
        index=read(OUT/'selection_index.json');total=0;zeros=0
        for info in index.values():
            s=read(OUT/info['path']);g=s['groups'];q=s['question_positions'];r=s['relay_positions']
            self.assertEqual(set(s['writers']),set(CONDITIONS));self.assertNotEqual(q,r)
            self.assertEqual(s['writers']['Semantic-first-Q100'],q)
            self.assertEqual(s['writers']['All-Writer'],r)
            zeros+=s['natural']['k']==0;total+=len(s['unique'])
            for name,p in s['writers'].items():
                self.assertTrue(intact(p,g));uid=s['aliases'][name]
                self.assertEqual(s['unique'][uid]['positions'],p);self.assertIn(name,s['unique'][uid]['labels'])
                if '-Q' in name:
                    b=name.rsplit('-Q',1)[1];actual=s['budgets'][b]['actual_k'];target=s['budgets'][b]['target_floor_tokens']
                    self.assertEqual(len(p),actual);self.assertLessEqual(actual,target)
                    feasible=[n for n in range(target+1) if intact(q[-n:] if n else [],g)]
                    self.assertEqual(actual,max(feasible))
                elif name not in ['All-Writer','No-Writer','Drop-Semantic']:self.assertEqual(len(p),s['natural']['k'])
        self.assertEqual(total,109518);self.assertEqual(zeros,4)

class MaskAndGoldTests(unittest.TestCase):
    def test_mask_direction_scope_non_target_exact_and_probability_zero(self):
        l=dict(prompt_length=9,I=[2,3,4],R=[5,6,7],A=[8],other=[0,1])
        for writers in [[],[6],[5,6,7]]:
            for past in [0,9,12]:
                native,changed,blocked=cpu_mask(l,writers,past)
                self.assertTrue(torch.equal(native[0,0][~blocked],changed[0,0][~blocked]))
                w=(torch.randn_like(changed)+changed).float().softmax(-1)
                self.assertEqual(float(w[0,:,blocked].max()),0)
                self.assertTrue((changed[0,0,:,0]==0).all())
                if not past:self.assertFalse(blocked[2:5].any());self.assertTrue((changed[0,0].diagonal()==0).all())
                expected=3 if past else (4-len(writers))*3;self.assertEqual(int(blocked.sum()),expected)
    def test_gold_token_mean_and_signed_difference_not_clipped(self):
        a=np.array([[1.,-1.,0.],[0.,2.,1.]],dtype=np.float16);ids=[0,2]
        values,mean=gold_losses(a,ids)
        expected=np.log(np.exp(a.astype(float)).sum(-1))-a[np.arange(2),ids]
        self.assertTrue(np.allclose(values,expected,atol=1e-14));self.assertAlmostEqual(mean,float(expected.mean()))
        better=a.copy();better[0,0]+=1;self.assertLess(gold_losses(better,ids)[1]-mean,0)
    def test_teacher_positions_prefixes_and_single_logit_shape(self):
        class Cache:
            def __init__(self,n):self.n=n
            def get_seq_length(self,k):return self.n
        calls=[]
        class Model:
            def __call__(self,**kw):
                prefix=kw.pop('test_prefix');t=len(prefix);calls.append((prefix,kw['logits_to_keep']))
                return SimpleNamespace(logits=torch.tensor([[[float(t),1.,2.]]],dtype=torch.float16),past_key_values=Cache(9+t))
        inputs={'input_ids':torch.ones(1,9,dtype=torch.long)}
        with patch(__package__+'.engine.step_inputs',lambda i,p:{'test_prefix':list(p)}):
            values=native_teacher(Model(),inputs,[2,1,0])
        self.assertEqual(calls,[([],1),([2],1),([2,1],1)]);self.assertEqual(values[:,0].tolist(),[0,1,2])

class DesignStatisticsTests(unittest.TestCase):
    def test_whole_image_shards_are_disjoint_exhaustive_balanced(self):
        samples=[dict(image_sha256=f'image{i//2}',task=['a','b'][i%2]) for i in range(20)]
        a=image_shards(samples);self.assertEqual(a,image_shards(samples));self.assertEqual(a['question_counts'],[10,10])
        self.assertEqual(set(a['image_assignments']),{s['image_sha256'] for s in samples})
    def test_frozen_split_and_swap_pairs_not_assumed_from_other_study(self):
        samples=read(OUT/'sample_manifest.json')['samples'];split=read(OUT/'split_integrity.json');old=set(split['exploration_image_sha256']);ids=set(split['exploration_question_ids'])
        actual=[key(s) for s in samples if s['question_id'] not in ids and s['image_sha256'] not in old]
        self.assertEqual(actual,split['confirmatory_keys']);self.assertEqual(len(actual),2640)
        pairs=read(OUT/'image_swap_manifest.json')['pairs'];lookup={key(s):s for s in samples};seen=set()
        self.assertEqual(len(pairs),40)
        for pair in pairs:
            a,b=[lookup[k] for k in pair['sample_keys']];self.assertEqual(a['question'],b['question'])
            self.assertEqual(a['input_mapping']['prompt_ids'],b['input_mapping']['prompt_ids']);self.assertNotEqual(a['answer'],b['answer'])
            for s in [a,b]:self.assertNotIn(s['image_sha256'],seen);seen.add(s['image_sha256'])
    def test_NI_requires_bound_over_margin_not_CI_containing_zero(self):
        d=confirmatory_decisions(np.linspace(-.2,-.1,1000),np.linspace(-3,1,1000))
        self.assertTrue(d['nll_superiority_rejected']);self.assertFalse(d['accuracy_NI_minus2_rejected'])
        d=confirmatory_decisions(np.linspace(-.2,.1,1000),np.linspace(-1.9,.2,1000))
        self.assertFalse(d['nll_superiority_rejected']);self.assertTrue(d['accuracy_NI_minus2_rejected']);self.assertFalse(d['accuracy_minus1_sensitivity'])

if __name__=='__main__':unittest.main(verbosity=2)
