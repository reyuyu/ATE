import tempfile
import unittest
from collections import Counter,defaultdict

from .common import *
from .prepare import shard
from .score import score


class IntegrityTests(unittest.TestCase):
    def test_counts(self):
        rows = manifest()
        self.assertEqual(len(rows),22678)
        self.assertEqual(Counter(s['benchmark'] for s in rows)['POPE'],8910)
        self.assertEqual(len({s['key'] for s in rows}),len(rows))

    def test_clusters(self):
        groups = defaultdict(set)
        circular = defaultdict(set)
        for s in manifest():
            groups[s['image_sha256']].add(s['shard'])
            if s['benchmark']=='MMBench':
                circular[s['circular_group']].add(s['shard'])
        self.assertTrue(all(len(x)==1 for x in [*groups.values(),*circular.values()]))

    def test_prompt_caps(self):
        for s in manifest():
            self.assertEqual(s['prompt'],prompt(s['user_text']))
            self.assertEqual(s['prompt'].count('<image>'),1)
            self.assertEqual(s['max_new_tokens'],1024 if s['benchmark'] in ('ScienceQA-IMG','MMBench') else 128)

    def test_official_rules(self):
        self.assertEqual(score(dict(benchmark='POPE',answer='no'),'No, there is none.')['score'],1)
        self.assertEqual(score(dict(benchmark='POPE',answer='yes'),'')['score'],1)
        self.assertEqual(score(dict(benchmark='MME',answer='no'),'No.')['score'],1)
        s = dict(benchmark='ScienceQA-IMG',answer=1,options={'A':'a','B':'b'})
        self.assertEqual(score(s,'B')['score'],1)
        self.assertEqual(score(s,'B)')['score'],0)
        self.assertEqual(score(s,'The answer is B.')['score'],1)

    def test_textvqa(self):
        s = dict(benchmark='TextVQA',answer=['two']*10)
        self.assertEqual(score(s,'2')['score'],1)

    def test_atomic_database(self):
        import sqlite3
        with tempfile.TemporaryDirectory(dir=RES) as tmp:
            db = sqlite3.connect(Path(tmp)/'test.sqlite')
            db.execute('CREATE TABLE results(key TEXT,condition TEXT,benchmark TEXT,payload TEXT,sha256 TEXT,PRIMARY KEY(key,condition))')
            s = dict(key='x',benchmark='x')
            r = dict(sample_sha256=digest(s),answer='yes')
            put(db,s,'Native',r)
            self.assertEqual(prior(db,s,'Native'),r)
            with self.assertRaises(sqlite3.IntegrityError):
                put(db,s,'Native',r)
            db.execute('UPDATE results SET sha256="broken"')
            with self.assertRaises(AssertionError):
                prior(db,s,'Native')


if __name__ == '__main__':
    unittest.main()
