import unittest
from .common import *
from .selection import selected
from semantic_visual_writers.groups import intact
from gqa3000_token_visual_writing.engine import cpu_mask

class Tests(unittest.TestCase):
    def test_S_exact(self):
        for q in read(P/'split_manifest.json')['metadata']:
            s=spec(q);self.assertEqual(selected(s,{})[0],s['historical_writers']['Semantic-k'])
    def test_anchors_and_whole_groups(self):
        for q in read(P/'split_manifest.json')['dev_stages']['60']:
            s=spec(q)
            for anchor in ['assistant','qtail1','qtail2','qtail3','postQ','instruction_period','instruction_word','instruction_pair','instruction4']:
                w,_=selected(s,dict(anchors=[anchor]));self.assertTrue(set(s['historical_writers']['Semantic-k'])<=set(w)<=set(s['R']));self.assertTrue(intact(w,s['groups']))
    def test_fill_never_splits_or_removes_S(self):
        for q in read(P/'split_manifest.json')['dev_stages']['60']:
            s=spec(q)
            for c in [dict(anchors=['assistant'],Rcap=.49),dict(extra=4),dict(anchors=['instruction_period'],Rcap=.49)]:
                w,a=selected(s,c);self.assertTrue(intact(w,s['groups']));self.assertTrue(set(s['historical_writers']['Semantic-k'])<=set(w));self.assertGreaterEqual(a['original_k'],0)
    def test_native_mask(self):
        layout=dict(prompt_length=9,I=[2,3,4],R=[5,6,7],A=[8],other=[0,1]);a,b,blocked=cpu_mask(layout,[6],0);self.assertTrue(torch.equal(a[0,0][~blocked],b[0,0][~blocked]));self.assertEqual(float(b.softmax(-1)[0,:,blocked].max()),0.)
    def test_image_split_and_scope(self):
        s=read(P/'split_manifest.json');a={s['metadata'][q]['image_sha256'] for q in s['ids']['DEV']};b={s['metadata'][q]['image_sha256'] for q in s['ids']['CHECK']};self.assertFalse(a&b);self.assertEqual(sum(len(v) for v in s['ids'].values()),3000)

if __name__=='__main__':unittest.main()

