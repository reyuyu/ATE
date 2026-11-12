import tempfile
import unittest
from pathlib import Path

from .common import ROOT, inventory, OUT
from .store import Store


class Tests(unittest.TestCase):
    def setUp(self):
        parent = Path('/data/lk/ATE-resources/experiments/qwen3vl_overnight_cpu')
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.path = Path(self.temp.name) / 'checkpoint.sqlite'

    def tearDown(self):
        self.temp.cleanup()

    def test_atomic_resume(self):
        store = Store(self.path, 'frozen')
        store.plan(['a'], ['Native', 'SVH'])
        store.begin('a', 'Native')
        store.finish('a', 'Native', dict(key='a', condition='Native', status='COMPLETE', text='yes'))
        store.begin('a', 'SVH')
        store.close()
        resumed = Store(self.path, 'frozen')
        self.assertEqual(resumed.state('a', 'Native'), 'COMPLETE')
        self.assertEqual(resumed.state('a', 'SVH'), 'PENDING')
        self.assertEqual(list(resumed.records())[0]['text'], 'yes')
        with self.assertRaises(AssertionError):
            resumed.begin('a', 'Native')
        resumed.close()

    def test_single_owner(self):
        store = Store(self.path, 'frozen')
        with self.assertRaises(BlockingIOError):
            Store(self.path, 'frozen')
        store.close()

    def test_corruption_detection(self):
        store = Store(self.path, 'frozen')
        store.plan(['a'], ['Native'])
        store.begin('a', 'Native')
        store.finish('a', 'Native', dict(key='a', condition='Native', status='SKIPPED_OOM'))
        store.db.execute('UPDATE predictions SET payload_sha256="wrong"')
        store.db.commit()
        with self.assertRaises(AssertionError):
            list(store.records())
        store.close()

    def test_inventory_counts(self):
        from collections import Counter
        from .common import COUNTS
        rows = inventory()
        self.assertEqual(len(rows), 35346)
        self.assertEqual(Counter(r['benchmark'] for r in rows), COUNTS)
        self.assertEqual(len({r['key'] for r in rows}), 35346)


if __name__ == '__main__':
    unittest.main()
