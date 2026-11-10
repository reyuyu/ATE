"""Small CPU-only regression tests for publication aggregation."""
import unittest

from vlm_research.fusion_results_delivery.audit import aggregate, digest


def record(key, value):
    return {'key': key, 'scoring': {'score': value}}


class AuditTests(unittest.TestCase):
    def test_digest_is_order_independent(self):
        self.assertEqual(digest({'x': 1, 'y': 2}), digest({'y': 2, 'x': 1}))

    def test_digest_rejects_nonfinite_scientific_records(self):
        with self.assertRaises(ValueError):
            digest({'score': float('nan')})

    def test_score_change_invalidates_record(self):
        self.assertNotEqual(digest(record('a', 0)), digest(record('a', 1)))

    def test_textvqa_retains_fractional_scores(self):
        value, units = aggregate('TextVQA', [record('a', .3), record('b', 1.)], {})
        self.assertAlmostEqual(value, 65.)
        self.assertEqual(units, 2)

    def test_circular_accuracy_is_not_row_average(self):
        samples = {'a': {'circular_group': 'g1'}, 'b': {'circular_group': 'g1'},
                   'c': {'circular_group': 'g2'}}
        value, units = aggregate('MMBench', [record('a', 1), record('b', 0), record('c', 1)], samples)
        self.assertEqual(value, 50.)
        self.assertEqual(units, 2)

    def test_mme_requires_complete_pairs(self):
        samples = {'a': {'task': 'task', 'pair_id': 'p'}}
        with self.assertRaises(AssertionError):
            aggregate('MME', [record('a', 1)], samples)

    def test_mme_acc_plus_acc_plus_is_task_sum(self):
        samples, records = {}, []
        for task in range(14):
            for index in range(2):
                key = str((task, index))
                samples[key] = {'task': task, 'pair_id': str(task)}
                records.append(record(key, 1.))
        value, units = aggregate('MME', records, samples)
        self.assertEqual(value, 2800.)
        self.assertEqual(units, 28)


if __name__ == '__main__':
    unittest.main()
