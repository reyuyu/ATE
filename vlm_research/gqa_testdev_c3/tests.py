import tempfile
import unittest
from dataclasses import replace

import torch

from .common import LABELS, KEEP, database, digest, expand, put, score, unpack
from ..c3v2_middle_reaccess.engine import MiddleWriter
from ..c3v2_middle_reaccess.tests import Tests as MiddleTests


class Tests(MiddleTests):
    def test_middle14_dense_reference(self):
        a = self.execute(KEEP[LABELS[1]], 25)
        b = self.execute(KEEP[LABELS[1]], 25, dense=True)
        torch.testing.assert_close(a[0], b[0].index_select(1,a[2].text_index), atol=3e-7, rtol=3e-6)
        torch.testing.assert_close(a[1], b[1], atol=3e-7, rtol=3e-6)
        self.assertEqual(a[2].cache.audit()['retained_image_layers'], list(range(11,25)))

    def test_no_extra_relay_writing_above19(self):
        c = MiddleWriter(self.a, replace(self.layout, cutoff=25), KEEP[LABELS[1]], 4)
        for layer in range(25):
            mask = c.prefill_mask(layer)
            for query in self.layout.relay:
                blocked = torch.isneginf(mask[0,0,query,list(self.layout.image)]).all()
                self.assertEqual(bool(blocked), layer>=20)
            self.assertEqual(bool((mask[0,0,-1,list(self.layout.image)]==0).all()), 11<=layer<=24)

    def test_first_answer_only_change_other_queries_native(self):
        a = self.execute()
        b = self.execute(KEEP[LABELS[1]],25)
        for left,right in zip(a[2].hidden[:20], b[2].hidden[:20]):
            self.assertTrue(torch.equal(left[:,:-1],right[:,:-1]))

    def test_official_gqa_conversion(self):
        self.assertEqual(score({'answer':'yes'},'Yes.'),1)
        self.assertEqual(score({'answer':'light blue'},'Light blue.'),1)
        self.assertEqual(score({'answer':'blue'},'It is blue.'),0)
        self.assertEqual(score({'answer':'2'},'two'),0)

    def test_compact_ids_expand_only_image(self):
        self.assertEqual(expand(dict(compact_prompt_ids=[1,32000,50,99])),[1]+[32000]*576+[50,99])

    def test_transaction_duplicate_and_digest(self):
        from unittest.mock import patch
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp, patch('vlm_research.gqa_testdev_c3.common.OUT',Path(tmp)):
            db = database(0,'frozen')
            r = dict(question_id='0001',condition=LABELS[0],token_ids=[2])
            put(db,r)
            self.assertEqual(unpack(db.execute('SELECT payload,sha256 FROM results').fetchone()),r)
            import sqlite3
            with self.assertRaises(sqlite3.IntegrityError):
                put(db,r)
            db.close()
            with self.assertRaises(AssertionError):
                database(0,'changed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
