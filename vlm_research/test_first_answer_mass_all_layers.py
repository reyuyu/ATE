import unittest
import numpy as np
from PIL import Image
from first_answer_mass.all_layers import display_scales, overlay


class DisplayTests(unittest.TestCase):
    def setUp(self):
        self.maps=np.arange(1,32*24*24+1,dtype=float).reshape(32,24,24)
        self.maps/=self.maps.sum((1,2),keepdims=True)

    def test_shared_log_uses_all32_and_does_not_change_probabilities(self):
        before=self.maps.copy()
        log,local=display_scales(self.maps)
        self.assertEqual(log.vmin,self.maps.min())
        self.assertEqual(log.vmax,self.maps.max())
        self.assertEqual(len(local),32)
        self.assertTrue(np.array_equal(before,self.maps))
        self.assertEqual(float(log(log.vmin)),0)
        self.assertEqual(float(log(log.vmax)),1)

    def test_local_percentiles_are_explicit_clipped_display_limits(self):
        _,local=display_scales(self.maps)
        for k,norm in enumerate(local):
            low,high=np.quantile(self.maps[k],[.02,.98])
            self.assertEqual(norm.vmin,low)
            self.assertEqual(norm.vmax,high)
            display=norm(self.maps[k])
            self.assertGreaterEqual(display.min(),0)
            self.assertLessEqual(display.max(),1)

    def test_overlay_is_finite_and_does_not_invent_outside_crop_attention(self):
        image=Image.new('RGB',(672,336),(80,120,160))
        g=dict(original_size=[672,336],resized_size=[672,336],crop_left=168,crop_top=0)
        log,_=display_scales(self.maps)
        result=np.asarray(overlay(image,self.maps[0],g,log))
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(np.abs(result[:,:168].astype(int)-np.asarray(image)[:,:168].astype(int)).max(),1)
        self.assertLessEqual(np.abs(result[:,504:].astype(int)-np.asarray(image)[:,504:].astype(int)).max(),1)


if __name__=='__main__':
    unittest.main()
