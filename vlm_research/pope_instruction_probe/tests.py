import unittest

from .common import *


class Tests(unittest.TestCase):
    def test_complete_predeclared_windows(self):
        c=conditions(10)
        self.assertEqual(len(c),28)
        self.assertEqual(len({name for name,_ in c}),28)
        self.assertEqual([sum(len(r)==n for _,r in c) for n in (1,2,4)],[10,9,7])

    def test_mask_restoration_only_instruction(self):
        m=dict(R=list(range(1,21)),Pminus1=21,parts=dict(Instruction=list(range(6,16)),ASSISTANT=[16,17,18,19],Q=[1,2,3],Hint=[],Options=[]))
        e1=set(writers(m,[]))
        self.assertEqual(writers(m,list(range(10))),m['R'])
        for _,r in conditions(10):
            actual=set(writers(m,r))
            self.assertEqual(actual-e1,{6+i for i in r})
            self.assertNotIn(21,actual)

    def test_edge_region(self):
        e=edge_mask().reshape(24,24)
        self.assertEqual(int(e.sum()),176)
        self.assertTrue(e[0].all() and e[:,23].all())
        self.assertFalse(e[2:22,2:22].any())

    def test_exact_frozen_token_boundaries(self):
        from transformers import AutoTokenizer
        t=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
        samples=read(HISTORY/'sample_manifest.json')
        for key in KEYS:
            m=samples[key]['input_mapping']
            for form in FORMS:
                ids=candidate_ids(t,m,form)
                self.assertEqual(t.decode(ids).strip().lower(),form.strip().lower())

    def test_geometry_integer_crop_and_outside_roi(self):
        import sys
        sys.path.insert(0,str(ROOT/'vlm_research'))
        from ..roi_sink_common import map_box
        g=dict(scale_x=1.,scale_y=1.,crop_left=100,crop_top=0)
        b=map_box([0,0,30,30],g)
        self.assertEqual(b['patch_indices'],[])
        self.assertEqual(b['crop_xyxy'][0],b['crop_xyxy'][2])

    def test_attention_ratio_and_area_normalization(self):
        from .analyze import metrics
        f=np.zeros(576); f[:2]=1
        roi=dict(roi_patch_fractions=f.tolist())
        x=np.ones((2,3,576),dtype=np.float32)/576
        m=metrics(x,roi)
        np.testing.assert_allclose(m['image_mass'],1,rtol=1e-6)
        np.testing.assert_allclose(m['edge_ratio'],176/576,rtol=1e-6)
        np.testing.assert_allclose(m['roi_ratio'],2/576,rtol=1e-6)
        np.testing.assert_allclose(m['roi_enrichment'],1,rtol=1e-6)

    def test_zero_or_cropped_out_roi_not_false_localization(self):
        from .analyze import metrics
        roi=dict(roi_patch_fractions=[0.]*576)
        m=metrics(np.zeros(576),roi)
        self.assertEqual(m['image_mass'],0)
        self.assertTrue(np.isnan(m['edge_ratio']))
        self.assertTrue(np.isnan(m['roi_ratio']))
        self.assertTrue(np.isnan(m['roi_enrichment']))


if __name__=='__main__':
    unittest.main(verbosity=2)
