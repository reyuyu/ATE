import copy,unittest
from .prepare import candidates,tags
from .common import intact

class CasePlanTests(unittest.TestCase):
    def spec(self):
        items=[('Is',1,True,False),('the',1,True,False),('zebra',3,True,True),('not',1,True,True),('white',1,True,True),('?',1,True,False),('\n',1,False,False),('Answer',1,False,False),('the',1,False,False),('using',1,False,False),('a',1,False,False),('single',1,False,False),('word',1,False,False),('ASSISTANT',4,False,False)]
        groups=[];p=580
        for text,n,q,core in items:
            groups.append(dict(decoded=text,words=[text] if text.isalpha() else [],positions=list(range(p,p+n)),question=q,core=core));p+=n
        return dict(groups=groups,writers={'Semantic-k':[p for g in groups if g['core'] for p in g['positions']]},question_positions=[p for g in groups if g['question'] for p in g['positions']],relay_positions=[p for g in groups for p in g['positions']])
    def test_selected_negation_never_suggested_as_missing(self):
        ss=self.spec();self.assertFalse(any(c['group']['decoded']=='not' for c in candidates(ss,'q')))
    def test_all_interventions_intact_equal_budget(self):
        ss=self.spec()
        for c in candidates(ss,'q'):
            self.assertTrue(intact(c['writers'],ss['groups']))
            if c['replacement_writers'] is not None:self.assertEqual(len(c['replacement_writers']),len(ss['writers']['Semantic-k']))
            if c['control_added'] is not None:self.assertEqual(len(c['added']),len(c['control_added']));self.assertFalse(set(c['added'])&set(c['control_added']))
    def test_candidate_bound_and_real_boundary_groups(self):
        cc=candidates(self.spec(),'q');self.assertLessEqual(len(cc),6)
        for word in ['?','Answer','ASSISTANT']:self.assertIn(word,[v['group']['decoded'] for v in cc])
    def test_prepare_does_not_mutate_frozen_semantic(self):
        ss=self.spec();old=copy.deepcopy(ss);candidates(ss,'q');self.assertEqual(ss,old)
    def test_same_input_deterministic_candidates_removals(self):self.assertEqual(candidates(self.spec(),'q'),candidates(self.spec(),'q'))
    def test_linguistic_annotations_do_not_mark_all_functions_as_missing(self):
        self.assertIn('negation',tags(dict(words=['not'],decoded='not')))
        self.assertIn('logical_connector',tags(dict(words=['or'],decoded='or')))

if __name__=='__main__':unittest.main()
