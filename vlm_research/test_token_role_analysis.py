import unittest
import numpy as np
from paper_rq1_token_role_decode.statistics import cluster_mean_ci, aggregate_positions, fixed_effects


class Tests(unittest.TestCase):
    def test_images_not_layers_are_clusters(self):
        rows=[dict(image_sha256='same') for _ in range(9)]
        result=cluster_mean_ci(rows,np.arange(9))
        self.assertEqual(result['n_images'],1);self.assertEqual(result['mean'],4)
        self.assertEqual(result['ci95_low'],4);self.assertEqual(result['ci95_high'],4)

    def test_nine_layer_average_preserves_signed_loss(self):
        rows=[dict(image_id='image',image_sha256='sha',generation_step=7,mode='Whole',layer=l,
            loss=float(l-4),abs_delta_logp=abs(l-4),relative_probability_change=.1,top1_flip=l==2,js_divergence=.01) for l in range(9)]
        out=aggregate_positions(rows)
        self.assertEqual(len(out),1);self.assertEqual(out[0]['loss'],0);self.assertEqual(out[0]['measured_layers'],9)
        self.assertAlmostEqual(out[0]['top1_flip'],1/9)

    def test_fixed_effects_known_coefficients(self):
        rng=np.random.default_rng(5);rows=[]
        for image in range(40):
            offset=rng.normal()
            for role in ['Function','Visual','Relation']:
                lp,entropy,time,relative=rng.normal(size=4)
                loss=offset+.7*(role=='Visual')-.2*(role=='Relation')+.2*lp-.1*entropy+.03*time+.04*relative
                rows.append(dict(image_sha256=str(image),word_type=role,loss=loss,full_token_logp=lp,
                    full_token_entropy=entropy,generation_step=time,relative_generation_position=relative,token_length=1))
        meta, effects=fixed_effects(rows);lookup={r['variable']:r for r in effects}
        self.assertEqual(meta['status'],'estimated');self.assertEqual(meta['n_images'],40)
        self.assertEqual(meta['dropped_constant_columns'],['token_length'])
        self.assertAlmostEqual(lookup['Visual_vs_Function']['effect'],.7,places=8)
        self.assertAlmostEqual(lookup['Relation_vs_Function']['effect'],-.2,places=8)

    def test_missing_role_not_invented(self):
        meta, effects=fixed_effects([dict(image_sha256='a',word_type='Function',loss=0)])
        self.assertEqual(meta['status'],'no_function_visual_pairs');self.assertEqual(effects,[])


if __name__=='__main__':unittest.main()
