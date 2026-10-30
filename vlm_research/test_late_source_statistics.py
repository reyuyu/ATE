import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from paper_rq1_late_attention_sources.statistics import ci,fixed_effects


class Tests(unittest.TestCase):
    def test_repeated_layers_not_independent_images(self):
        r=[dict(image_sha256='a'),dict(image_sha256='b')]
        a=ci(r,[0.,2.],41);b=ci([*r,*r],[0.,2.,0.,2.],41)
        self.assertEqual(a['n_images'],2);self.assertEqual(a['ci95_low'],b['ci95_low']);self.assertEqual(a['mean'],b['mean'])
        self.assertLessEqual(a['familywise95_low'],a['ci95_low'])

    def test_fixed_effects_known_signal_and_length_constant(self):
        rng=np.random.default_rng(42);rows=[]
        for i in range(50):
            image_effect=rng.normal()
            for role in ['Function','Visual','Relation']:
                lp,entropy,t,relative=rng.normal(size=4)
                loss=image_effect+(2 if role=='Visual' else -1 if role=='Relation' else 0)+.3*lp+.5*entropy-.2*t+.1*relative
                rows.append(dict(image_sha256=str(i),word_type=role,loss=loss,full_token_logp=lp,
                    full_token_entropy=entropy,generation_step=t,relative_generation_position=relative,token_length=1))
        meta,output=fixed_effects(rows)
        self.assertEqual(meta['status'],'estimated');self.assertIn('token_length',meta['dropped_constant_columns'])
        self.assertAlmostEqual(next(r['effect'] for r in output if r['variable']=='Visual_vs_Function'),2,places=8)
        self.assertEqual(meta['family'],41)

    def test_missing_pairs_and_nonfinite_not_fabricated(self):
        meta,rows=fixed_effects([dict(image_sha256='a',word_type='Visual')])
        self.assertEqual(meta['status'],'no_same_image_pairs');self.assertEqual(rows,[])
        with self.assertRaises(ValueError):ci([dict(image_sha256='a')],[np.nan])

    def test_planned_contrasts_exactly_41_on_isolated_fixture(self):
        from paper_rq1_late_attention_sources import analyze
        from paper_rq1_late_attention_sources.protocol import LAYERS,TYPES,conditions
        rng=np.random.default_rng(19);rows=[]
        for image in range(12):
            steps=rng.choice(np.arange(1,12),size=3,replace=False)
            for role_index,role in enumerate(TYPES):
                lp=-rng.uniform(.1,2);entropy=rng.uniform(.5,3);t=int(steps[role_index])
                for layer in LAYERS:
                    for mode in conditions(layer):
                        loss=.01*layer+.1*role_index+.03*lp+.02*entropy+rng.normal(scale=.01)
                        rows.append(dict(image_sha256=str(image),image_id=str(image),word_type=role,mode=mode,layer=layer,
                            generation_step=t,relative_generation_position=t/(13+image),token_length=1,
                            full_token_logp=lp,full_token_entropy=entropy,loss=loss,abs_delta_logp=abs(loss),top1_flip=loss>.5))
        # Synthetic test outputs never enter the real result directory or figures.
        with tempfile.TemporaryDirectory() as folder,patch.object(analyze,'OUT',Path(folder)):
            primary,paired,deep,recent=analyze.contrasts(rows)
            self.assertEqual(len(primary),41);self.assertEqual(len(paired),12)
            self.assertEqual(len(deep),13);self.assertEqual(len(recent),6)
            self.assertTrue(all(r['family']==41 for r in primary))

    def test_incomplete_gate_cannot_generate_results_or_figures(self):
        from paper_rq1_late_attention_sources import analyze
        with tempfile.TemporaryDirectory() as folder,patch.object(analyze,'OUT',Path(folder)),\
                patch.object(analyze,'verify',return_value={}),patch.object(analyze,'verify_targets',return_value={}),\
                patch.object(analyze,'gate',side_effect=FileNotFoundError('GPU0 not complete')):
            with self.assertRaises(FileNotFoundError):analyze.main()
            self.assertEqual(list(Path(folder).iterdir()),[])


if __name__=='__main__':unittest.main()
