import unittest
from .preflight import bbox, binding, canonical


class CoverageTests(unittest.TestCase):
    def test_leading_zero_ids(self):
        self.assertEqual(canonical('00228612'), canonical('228612.0'))

    def test_invalid_ids(self):
        for value in ('NaN', 'inf', '-1', '1.5'):
            with self.assertRaises(ValueError):
                canonical(value)

    def test_box_is_xywh_not_xyxy(self):
        r = {'object1 id':'12', 'object1 name':'cat', 'object1 x':'2', 'object1 y':'3',
             'object1 w':'4', 'object1 h':'5', 'image_w':'20', 'image_h':'10'}
        self.assertEqual(bbox(r, 'object1')['xywh'], [2,3,4,5])
        for field, value in [('object1 w','0'), ('object1 x','-1'), ('object1 h','nan'), ('object1 y','9')]:
            with self.assertRaises(ValueError):
                bbox({**r,field:value}, 'object1')

    def test_binding_never_uses_object_name_guess(self):
        s = {'image_id':'001', 'question':'Which?', 'answer':'left'}
        r = {'img_id':'1', 'question':'Which?', 'answer':'left'}
        binding(r, s)
        for key in r:
            with self.assertRaises((ValueError, ArithmeticError)):
                binding({**r,key:'2' if key == 'img_id' else 'other'}, s)


if __name__ == '__main__':
    unittest.main()
