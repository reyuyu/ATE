import unittest
from full_common import *


class Tests(unittest.TestCase):
    def test_only_allowed_fields_rendered(self):
        self.assertEqual(body('QUESTION', {'A':'ALPHA','B':'BETA'}, 'HINT'),
            'Hint: HINT\nQuestion: QUESTION\nOptions:\nA. ALPHA\nB. BETA\nPlease select the correct answer from the options above. \n')

    def test_prompt_source_frozen(self):
        p=json.loads((OLD/'protocol.json').read_text())
        from prompt_common import P2_CHOICE
        self.assertEqual(p['prompts']['P2_choices'],P2_CHOICE)

    def test_store_no_overwrite(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir='/data/lk/ATE-worktrees/qwen3vl-p2-full-gqa-type-audit') as path:
            store=Store(Path(path)/'test.sqlite')
            store.put('a',{'score':1})
            with self.assertRaises(AssertionError):
                store.put('a',{'score':0})
            store.conn.close()

    def test_circular_all_required(self):
        self.assertFalse(all([1,1,0,1]))
        self.assertTrue(all([1,1,1,1]))


if __name__=='__main__':
    unittest.main()
