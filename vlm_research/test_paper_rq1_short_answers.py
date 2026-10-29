"""The partial adapter must never change frozen GPU sources or inference gates."""
import unittest

from paper_rq1 import analyze
from paper_rq1.run import verify
import paper_rq1_short_answers as snapshot


class ShortSnapshotTests(unittest.TestCase):
    def sample_result(self):
        sample = dict(dataset='gqa_balanced', answer='cat')
        result = dict(generations=[dict(window_start=s, mode=m, token_ids=[42, 2],
            length=2, eos=True, generated_text='Cat.', score=1.)
            for s, m in [(None, 'Full')]+[(s, m) for s in (0, 3, 4, 9, 13, 14, 20, 29) for m in analyze.MODES]])
        return sample, result

    def test_scoped_reuse_restores_globals_on_exception(self):
        original = analyze.OUTPUT, analyze.COUNTS, analyze.figure2
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with snapshot.short_aggregation_scope(snapshot.DESTINATION):
                self.assertEqual(analyze.OUTPUT, snapshot.DESTINATION)
                self.assertEqual(len(analyze.COUNTS), 4)
                raise RuntimeError('fixture')
        self.assertEqual((analyze.OUTPUT, analyze.COUNTS, analyze.figure2), original)

    def test_exact_generation_coverage(self):
        sample, result = self.sample_result()
        snapshot.validate_generations(sample, result, (0, 3, 4, 9, 13, 14, 20, 29))
        result['generations'][-1] = result['generations'][0]
        with self.assertRaisesRegex(ValueError, 'conditions'):
            snapshot.validate_generations(sample, result, (0, 3, 4, 9, 13, 14, 20, 29))

    def test_reject_changed_score(self):
        sample, result = self.sample_result()
        result['generations'][0]['score'] = 0.
        with self.assertRaisesRegex(ValueError, 'score'):
            snapshot.validate_generations(sample, result, (0, 3, 4, 9, 13, 14, 20, 29))

    def test_reject_post_eos(self):
        sample, result = self.sample_result()
        result['generations'][0].update(token_ids=[2, 42], eos=False)
        with self.assertRaisesRegex(ValueError, 'EOS'):
            snapshot.validate_generations(sample, result, (0, 3, 4, 9, 13, 14, 20, 29))

    def test_original_sources_and_protocol_still_frozen(self):
        self.assertEqual(verify()['code_commit'], '906103b37b19d56e1752b5b3fc5bc04246ecd6a8')


if __name__ == '__main__':
    unittest.main()
